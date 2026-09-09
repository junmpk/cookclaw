#!/usr/bin/env python3
"""比较两组真实 Conversation Trace，并按显式阈值给出灰度结论。

本脚本没有内置性能或质量数字。没有传入的门禁不会参与放行判断，避免把未经
真实基线确认的目标伪装成项目事实。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.observability.baseline import summarize_traces
from scripts.summarize_conversation_traces import (
    _load_business_metrics,
    _load_rows,
)


def _relative_change(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if before == 0:
        return 0.0 if after == 0 else None
    return round((after - before) / before * 100, 2)


def _summary(path: Path, channel: str) -> tuple[dict, dict | None]:
    rows = _load_rows(path)
    if channel:
        rows = [row for row in rows if row.get("channel") == channel]
    return summarize_traces(rows), _load_business_metrics(path)


def _gate(
    name: str,
    *,
    actual: float | None,
    operator: str,
    threshold: float | None,
) -> dict | None:
    if threshold is None:
        return None
    if actual is None:
        return {
            "name": name,
            "passed": False,
            "actual": None,
            "operator": operator,
            "threshold": threshold,
            "reason": "metric_unavailable",
        }
    passed = actual <= threshold if operator == "<=" else actual >= threshold
    return {
        "name": name,
        "passed": passed,
        "actual": round(actual, 4),
        "operator": operator,
        "threshold": threshold,
        "reason": "threshold_met" if passed else "threshold_exceeded",
    }


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_business: dict | None = None,
    candidate_business: dict | None = None,
    thresholds: dict[str, float | None] | None = None,
    require_deep_agent_used: bool = False,
) -> dict[str, Any]:
    thresholds = thresholds or {}
    baseline_latency = (baseline.get("latency_ms") or {}).get("p95")
    candidate_latency = (candidate.get("latency_ms") or {}).get("p95")
    baseline_tokens = (baseline.get("tokens") or {}).get("average")
    candidate_tokens = (candidate.get("tokens") or {}).get("average")
    deltas = {
        "p95_latency_percent": _relative_change(
            baseline_latency,
            candidate_latency,
        ),
        "average_tokens_percent": _relative_change(
            baseline_tokens,
            candidate_tokens,
        ),
        "model_calls_average": (
            round(
                float((candidate.get("model_calls") or {}).get("average") or 0)
                - float((baseline.get("model_calls") or {}).get("average") or 0),
                2,
            )
        ),
        "tool_calls_average": (
            round(
                float((candidate.get("tool_calls") or {}).get("average") or 0)
                - float((baseline.get("tool_calls") or {}).get("average") or 0),
                2,
            )
        ),
    }

    gates = [
        _gate(
            "max_p95_latency_regression_percent",
            actual=deltas["p95_latency_percent"],
            operator="<=",
            threshold=thresholds.get("max_p95_latency_regression_percent"),
        ),
        _gate(
            "max_average_token_regression_percent",
            actual=deltas["average_tokens_percent"],
            operator="<=",
            threshold=thresholds.get("max_average_token_regression_percent"),
        ),
        _gate(
            "min_effective_turn_rate",
            actual=(
                (candidate_business or {}).get("effective_turn_rate")
            ),
            operator=">=",
            threshold=thresholds.get("min_effective_turn_rate"),
        ),
        _gate(
            "min_intent_accuracy",
            actual=(candidate_business or {}).get("intent_accuracy"),
            operator=">=",
            threshold=thresholds.get("min_intent_accuracy"),
        ),
        _gate(
            "min_route_accuracy",
            actual=(candidate_business or {}).get("route_accuracy"),
            operator=">=",
            threshold=thresholds.get("min_route_accuracy"),
        ),
        _gate(
            "max_tool_failure_rate",
            actual=candidate.get("tool_failure_rate"),
            operator="<=",
            threshold=thresholds.get("max_tool_failure_rate"),
        ),
        _gate(
            "max_timeout_rate",
            actual=candidate.get("timeout_rate"),
            operator="<=",
            threshold=thresholds.get("max_timeout_rate"),
        ),
    ]
    gates = [item for item in gates if item is not None]
    if require_deep_agent_used:
        gates.append({
            "name": "deep_agent_was_exercised",
            "passed": float(candidate.get("deep_agent_used_rate") or 0) > 0,
            "actual": candidate.get("deep_agent_used_rate"),
            "operator": ">",
            "threshold": 0,
            "reason": (
                "deep_agent_trace_present"
                if float(candidate.get("deep_agent_used_rate") or 0) > 0
                else "deep_agent_not_observed"
            ),
        })
    has_data = bool(baseline.get("trace_count")) and bool(
        candidate.get("trace_count")
    )
    passed = (
        False
        if not has_data
        else (all(item["passed"] for item in gates) if gates else None)
    )
    return {
        "schema_version": "conversation_baseline_comparison_v1",
        "has_required_trace_data": has_data,
        "passed": passed,
        "release_decision": (
            "pass"
            if passed is True
            else ("fail" if passed is False else "not_evaluated")
        ),
        "gate_count": len(gates),
        "note": (
            "未配置门禁阈值，仅完成数据对比。"
            if not gates else
            "所有门禁均使用命令行显式提供的阈值。"
        ),
        "baseline": baseline,
        "candidate": candidate,
        "baseline_business_test": baseline_business,
        "candidate_business_test": candidate_business,
        "deltas": deltas,
        "gates": gates,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较 Deep Agent 灰度前后的 Conversation Trace",
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--channel", default="qq")
    parser.add_argument("--max-p95-latency-regression-percent", type=float)
    parser.add_argument("--max-average-token-regression-percent", type=float)
    parser.add_argument("--min-effective-turn-rate", type=float)
    parser.add_argument("--min-intent-accuracy", type=float)
    parser.add_argument("--min-route-accuracy", type=float)
    parser.add_argument("--max-tool-failure-rate", type=float)
    parser.add_argument("--max-timeout-rate", type=float)
    parser.add_argument(
        "--require-deep-agent-used",
        action="store_true",
        help="候选样本必须至少有一轮实际进入 Deep Agent。",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline, baseline_business = _summary(args.baseline, args.channel)
    candidate, candidate_business = _summary(args.candidate, args.channel)
    thresholds = {
        key: getattr(args, key)
        for key in (
            "max_p95_latency_regression_percent",
            "max_average_token_regression_percent",
            "min_effective_turn_rate",
            "min_intent_accuracy",
            "min_route_accuracy",
            "max_tool_failure_rate",
            "max_timeout_rate",
        )
    }
    report = compare(
        baseline,
        candidate,
        baseline_business=baseline_business,
        candidate_business=candidate_business,
        thresholds=thresholds,
        require_deep_agent_used=args.require_deep_agent_used,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if report["passed"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
