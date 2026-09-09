#!/usr/bin/env python3
"""运行 CookClaw 十组核心多轮对话验收，不访问真实设备或外部服务。

这些验收映射到仓库中已经使用 fake search / fake device / memory store 的
确定性 pytest，用一个报告固定“用户对话 -> 预期 -> 自动化证据”的关系。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AcceptanceScenario:
    scenario_id: str
    name: str
    dialogue: tuple[str, ...]
    expected: tuple[str, ...]
    tests: tuple[str, ...]


SCENARIOS: tuple[AcceptanceScenario, ...] = (
    AcceptanceScenario(
        "AC-01",
        "条件追加",
        ("推荐几个鸡肉菜", "不要辣的", "孩子也能吃", "简单一点"),
        (
            "保持鸡肉主任务",
            "依次合并不辣、儿童、快手条件",
            "条件跟进不重新调用意图分类",
        ),
        (
            "tests/test_conversation.py::test_condition_followups_merge_and_research_without_reclassifying",
        ),
    ),
    AcceptanceScenario(
        "AC-02",
        "序号和指代",
        ("推荐三道菜", "第二个怎么样", "就这个吧"),
        (
            "第二个映射到原候选位置 2",
            "这个映射到最近讨论菜谱",
            "selected_recipe_id 更新为同一真实候选",
        ),
        (
            "tests/test_conversation.py::test_candidate_focus_supports_this_one_selection",
        ),
    ),
    AcceptanceScenario(
        "AC-03",
        "排除与反悔",
        ("推荐三道菜", "第二个不要", "第一个也太麻烦了", "还是刚才第二个吧"),
        (
            "排除不改变原始序号",
            "反悔后恢复第二个真实候选",
            "第二个从排除集合移除，第一项仍保持排除",
        ),
        (
            "tests/test_conversation.py::test_candidate_exclusion_and_reconsideration_keep_original_indexes",
        ),
    ),
    AcceptanceScenario(
        "AC-04",
        "条件修改",
        ("推荐不辣的菜", "算了，稍微辣一点也可以", "不要牛肉"),
        (
            "微辣修改不辣条件而非形成冲突",
            "保留原搜索主语",
            "牛肉进入排除条件",
        ),
        (
            "tests/test_conversation.py::test_condition_revision_replaces_conflict_and_keeps_prior_subject",
        ),
    ),
    AcceptanceScenario(
        "AC-05",
        "搜菜到设备控制",
        ("推荐几个简单的鸡肉菜", "第二个怎么样", "就做这个", "选择设备", "确认"),
        (
            "候选选择与设备选择分层",
            "设备选择后必须再次最终确认",
            "并发或重复确认只领取一次待执行动作",
        ),
        (
            "tests/test_conversation.py::test_pending_device_selection_requires_separate_final_confirmation",
            "tests/test_conversation.py::test_concurrent_affirmations_claim_pending_device_action_once",
        ),
    ),
    AcceptanceScenario(
        "AC-06",
        "取消和停止",
        ("就做第二个", "开始吧", "算了，先别做", "刚才选的是哪个"),
        (
            "取消只撤销待确认设备动作",
            "不发送设备命令",
            "活动烹饪不能被模糊放弃词误停止",
        ),
        (
            "tests/test_conversation.py::test_abandonment_is_resolved_by_state_without_intent_or_device_calls",
        ),
    ),
    AcceptanceScenario(
        "AC-07",
        "设备异常",
        ("开始做吧",),
        (
            "离线、忙碌或未知状态均不声称成功",
            "command_sent 保持 false",
            "待确认任务保留以便安全重试",
        ),
        (
            "tests/test_conversation.py::test_confirmed_device_not_ready_returns_fact_and_keeps_confirmation",
        ),
    ),
    AcceptanceScenario(
        "AC-08",
        "闲聊后回归任务",
        ("推荐几个鸡肉菜", "谢谢你", "第二个怎么样"),
        (
            "简单闲聊不清空候选",
            "回归后仍按原始第二个候选解析",
            "不重复搜索",
        ),
        (
            "tests/test_conversation.py::test_smalltalk_does_not_clear_recipe_candidates",
        ),
    ),
    AcceptanceScenario(
        "AC-09",
        "切换话题后恢复",
        ("推荐几个鸡肉菜", "查一下设备状态", "继续说刚才的菜"),
        (
            "设备查询不覆盖菜谱候选",
            "恢复原候选时不重复搜索",
            "recipe_selection 任务恢复",
        ),
        (
            "tests/test_conversation.py::test_device_status_interrupt_can_resume_same_recipe_list_without_search",
        ),
    ),
    AcceptanceScenario(
        "AC-10",
        "多用户隔离",
        ("用户 A 选择菜谱", "用户 B 同时选择另一菜谱"),
        (
            "QQ thread_id 包含用户身份",
            "持久化任务状态按用户隔离",
            "候选、选择和待执行状态不串线",
        ),
        (
            "tests/test_conversation.py::test_qq_thread_id_isolates_group_users",
            "tests/test_conversation.py::test_task_state_persists_across_service_recreation_and_isolates_users",
        ),
    ),
)


def _selected_scenarios(names: list[str] | None) -> list[AcceptanceScenario]:
    selected = set(names or [])
    return [
        scenario
        for scenario in SCENARIOS
        if not selected or scenario.scenario_id in selected
    ]


def _parse_junit(path: Path) -> dict[str, dict]:
    root = ET.parse(path).getroot()
    results: dict[str, dict] = {}
    for case in root.iter("testcase"):
        name = str(case.attrib.get("name") or "")
        status = "passed"
        detail = ""
        for child_status in ("failure", "error", "skipped"):
            child = case.find(child_status)
            if child is not None:
                status = child_status
                detail = str(child.attrib.get("message") or child.text or "")[:1000]
                break
        results[name] = {
            "status": status,
            "duration_seconds": round(
                float(case.attrib.get("time") or 0),
                3,
            ),
            "detail": detail,
        }
    return results


def _build_report(
    scenarios: list[AcceptanceScenario],
    test_results: dict[str, dict],
    *,
    pytest_exit_code: int,
) -> dict:
    scenario_results = []
    for scenario in scenarios:
        checks = []
        for nodeid in scenario.tests:
            test_name = nodeid.rsplit("::", 1)[-1]
            result = test_results.get(test_name) or {
                "status": "missing",
                "duration_seconds": 0,
                "detail": "pytest JUnit 未返回该测试",
            }
            checks.append({"nodeid": nodeid, **result})
        scenario_results.append({
            **asdict(scenario),
            "passed": bool(checks) and all(
                item["status"] == "passed" for item in checks
            ),
            "checks": checks,
        })
    passed = sum(item["passed"] for item in scenario_results)
    total = len(scenario_results)
    return {
        "schema_version": "core_dialogue_acceptance_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": pytest_exit_code == 0 and passed == total,
        "passed_scenarios": passed,
        "total_scenarios": total,
        "scenario_pass_rate": round(passed / total, 4) if total else None,
        "pytest_exit_code": pytest_exit_code,
        "real_device_called": False,
        "external_service_called": False,
        "scenarios": scenario_results,
    }


def _write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"core-dialogue-acceptance-{stamp}.json"
    markdown_path = output_dir / f"core-dialogue-acceptance-{stamp}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# CookClaw 核心多轮对话验收报告",
        "",
        f"- 时间：{report['created_at']}",
        f"- 结果：{'通过' if report['passed'] else '失败'}",
        (
            f"- 场景：{report['passed_scenarios']}/"
            f"{report['total_scenarios']}"
        ),
        "- 真实设备调用：否",
        "- 外部服务调用：否",
        "",
    ]
    for scenario in report["scenarios"]:
        lines.extend([
            f"## {'✅' if scenario['passed'] else '❌'} "
            f"{scenario['scenario_id']} {scenario['name']}",
            "",
            "对话：" + " → ".join(scenario["dialogue"]),
            "",
        ])
        for expectation in scenario["expected"]:
            lines.append(f"- 预期：{expectation}")
        for check in scenario["checks"]:
            lines.append(
                f"- {check['status'].upper()}：`{check['nodeid']}`"
            )
            if check["detail"]:
                lines.append(f"  - {check['detail']}")
        lines.append("")
    markdown_path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return json_path, markdown_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行十组核心多轮对话的确定性验收",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.scenario_id for scenario in SCENARIOS],
        help="只运行指定场景；可重复传入。默认十组全跑。",
    )
    parser.add_argument("--list", action="store_true", help="列出场景后退出。")
    parser.add_argument("--no-report", action="store_true", help="不写报告文件。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/core_dialogue_acceptance"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    selected = _selected_scenarios(args.scenario)
    if args.list:
        for scenario in selected:
            print(f"{scenario.scenario_id} {scenario.name}")
        return 0

    nodeids = list(dict.fromkeys(
        nodeid
        for scenario in selected
        for nodeid in scenario.tests
    ))
    with tempfile.TemporaryDirectory(prefix="cookclaw-acceptance-") as temp_dir:
        junit_path = Path(temp_dir) / "pytest.xml"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *nodeids,
                f"--junitxml={junit_path}",
            ],
            cwd=_ROOT,
            check=False,
        )
        test_results = (
            _parse_junit(junit_path)
            if junit_path.exists()
            else {}
        )

    report = _build_report(
        selected,
        test_results,
        pytest_exit_code=completed.returncode,
    )
    if not args.no_report:
        json_path, markdown_path = _write_report(report, args.output_dir)
        print(f"JSON_REPORT={json_path.resolve()}")
        print(f"MARKDOWN_REPORT={markdown_path.resolve()}")
    print(
        "CORE_DIALOGUE_ACCEPTANCE="
        f"{'PASS' if report['passed'] else 'FAIL'} "
        f"scenarios={report['passed_scenarios']}/{report['total_scenarios']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
