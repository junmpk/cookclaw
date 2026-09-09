"""校验 Failure Dataset，或通过 Web SSE 入口做低权限多轮回放。

公开 Web API 目前不暴露 route/tool/state 细节，因此本脚本只自动判定回复硬规则；
涉及安全状态的 case 会保留 ``needs_trace_review``，绝不因文案好看而判整例通过。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.evaluation.conversation_failure_dataset import (
    FailureCase,
    evaluate_response_rules,
    load_failure_dataset,
)

_DEFAULT_DATASET = (
    _ROOT
    / "tests"
    / "fixtures"
    / "conversation_failure_dataset"
    / "v1"
    / "cases.yaml"
)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def _requires_fixture_or_internal_observation(case: FailureCase) -> bool:
    unsupported_pre_state = set(case.pre_state) - {"kind"}
    return bool(
        unsupported_pre_state
        or case.thread_policy == "new_thread_same_account"
        or case.channel not in {"web", "all"}
    )


def _web_replay_thread_id(run_id: str, case_id: str) -> str:
    """Return an API-compatible opaque Web session id without leaking case labels."""
    token = hashlib.sha256(f"{run_id}:{case_id}".encode()).hexdigest()[:32]
    return f"web:{token}"


async def _web_turn(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    question: str,
    thread_id: str,
) -> tuple[str, str]:
    chunks: list[str] = []
    async with client.stream(
        "POST",
        f"{endpoint.rstrip('/')}/chat",
        json={"question": question, "thread_id": thread_id},
    ) as response:
        response.raise_for_status()
        resolved_thread = response.headers.get(
            "X-CookClaw-Thread-Id",
            thread_id,
        )
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            parsed = json.loads(payload)
            chunks.append(str(parsed.get("content") or ""))
    return "".join(chunks).strip(), resolved_thread


async def _replay(args: argparse.Namespace) -> int:
    dataset = load_failure_dataset(args.dataset)
    selected = {
        case.case_id: case
        for case in dataset.cases
        if not args.case_id or case.case_id in set(args.case_id)
    }
    unknown = set(args.case_id or []) - set(selected)
    if unknown:
        raise SystemExit(f"unknown case ids: {', '.join(sorted(unknown))}")

    print(
        json.dumps(
            {
                "schema_version": dataset.schema_version,
                "dataset_version": dataset.dataset_version,
                "case_count": len(selected),
                "mode": "validate" if not args.base_url else "web_replay",
            },
            ensure_ascii=False,
        )
    )
    if not args.base_url:
        return 0

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output = args.output or (
        _ROOT / "outputs" / "conversation_failure_eval" / f"{run_id}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()
    rows: list[dict] = []
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for case in selected.values():
            if _requires_fixture_or_internal_observation(case):
                rows.append({
                    "run_id": run_id,
                    "git_sha": git_sha,
                    "dataset_version": dataset.dataset_version,
                    "case_id": case.case_id,
                    "review_status": "skipped_requires_fixture_or_channel_adapter",
                    "hard_pass": False,
                })
                continue
            thread_id = _web_replay_thread_id(run_id, case.case_id)
            for turn_index, turn in enumerate(case.turns, 1):
                started = time.monotonic()
                response_text, thread_id = await _web_turn(
                    client,
                    args.base_url,
                    question=turn.input,
                    thread_id=thread_id,
                )
                response_rules = evaluate_response_rules(
                    response_text,
                    turn.expected.response,
                )
                structural_expected = bool(
                    turn.expected.route
                    or turn.expected.state
                    or turn.expected.allowed_tools
                    or turn.expected.forbidden_tools
                )
                row = {
                    "run_id": run_id,
                    "git_sha": git_sha,
                    "dataset_version": dataset.dataset_version,
                    "case_id": case.case_id,
                    "turn_index": turn_index,
                    "channel": "web",
                    "response_hash": hashlib.sha256(
                        response_text.encode("utf-8")
                    ).hexdigest(),
                    "latency_ms": round((time.monotonic() - started) * 1000, 2),
                    "response_rules_pass": response_rules.passed,
                    "response_rule_failures": response_rules.failures,
                    "structural_assertions_status": (
                        "not_observed" if structural_expected else "not_required"
                    ),
                    "hard_pass": (
                        response_rules.passed and not structural_expected
                    ),
                    "judge_scores": {},
                    "review_status": (
                        "needs_trace_review"
                        if structural_expected
                        else "response_rules_complete"
                    ),
                    "first_failed_stage": (
                        "response_rules"
                        if not response_rules.passed
                        else (
                            "structural_observation"
                            if structural_expected
                            else None
                        )
                    ),
                }
                if args.include_text:
                    row["response"] = response_text
                rows.append(row)

    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "rows": len(rows)}, ensure_ascii=False))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--base-url", help="例如 http://127.0.0.1:8000/api/v1")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="结果默认只存 response hash；仅对已脱敏数据启用正文。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_replay(_parse_args())))
