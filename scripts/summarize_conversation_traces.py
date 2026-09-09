#!/usr/bin/env python3
"""汇总 CookClaw conversation_turn_trace JSONL。

用法：
  .venv/bin/python scripts/summarize_conversation_traces.py traces.jsonl
  journalctl -u cookclaw --since today -o cat > /tmp/cookclaw.log
  .venv/bin/python scripts/summarize_conversation_traces.py /tmp/cookclaw.log --channel qq
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.observability.baseline import summarize_traces
from app.observability.health import evaluate_health_alerts


def _extract_trace_rows(value) -> list[dict]:
    if isinstance(value, dict):
        if value.get("schema_version") == "conversation_trace_v1":
            return [value]
        rows = []
        for child in value.values():
            rows.extend(_extract_trace_rows(child))
        return rows
    if isinstance(value, list):
        rows = []
        for child in value:
            rows.extend(_extract_trace_rows(child))
        return rows
    return []


def _load_rows(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        document = None
    if document is not None:
        return _extract_trace_rows(document)

    rows = []
    marker = "conversation_turn_trace "
    for line in raw.splitlines():
        text = line.strip()
        if marker in text:
            text = text.split(marker, 1)[1].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        rows.extend(_extract_trace_rows(payload))
    return rows


def _load_business_metrics(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or "effective_turn_rate" not in payload:
        return None
    return {
        key: payload.get(key)
        for key in (
            "passed_turns",
            "total_turns",
            "effective_turn_rate",
            "route_assessed_turns",
            "route_correct_turns",
            "route_accuracy",
            "intent_assessed_turns",
            "intent_correct_turns",
            "intent_accuracy",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 CookClaw 对话 Trace 基线")
    parser.add_argument("path", type=Path, help="JSONL 或包含 conversation_turn_trace 的日志")
    parser.add_argument("--channel", default="", help="只统计指定通道，如 qq")
    parser.add_argument(
        "--deep-agent",
        choices=("true", "false"),
        default=None,
        help="只统计 Deep Agent 开启或关闭的轮次",
    )
    parser.add_argument(
        "--deep-agent-used",
        choices=("true", "false"),
        default=None,
        help="只统计本轮实际进入或未进入 Deep Agent 的轮次",
    )
    parser.add_argument("--min-success-rate", type=float, default=None)
    parser.add_argument("--min-trace-count", type=int, default=None)
    parser.add_argument("--max-p95-latency-ms", type=float, default=None)
    parser.add_argument("--max-timeout-rate", type=float, default=None)
    parser.add_argument("--max-tool-failure-rate", type=float, default=None)
    parser.add_argument("--max-planner-fallback-rate", type=float, default=None)
    parser.add_argument("--max-device-state-unknown-rate", type=float, default=None)
    parser.add_argument(
        "--max-planner-execution-failure-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="存在告警时以状态码 2 退出，供 systemd/CI/监控调用",
    )
    args = parser.parse_args()

    rows = _load_rows(args.path)
    if args.channel:
        rows = [row for row in rows if row.get("channel") == args.channel]
    if args.deep_agent is not None:
        expected = args.deep_agent == "true"
        rows = [
            row for row in rows
            if bool(row.get("deep_agent_enabled")) is expected
        ]
    if args.deep_agent_used is not None:
        expected = args.deep_agent_used == "true"
        rows = [
            row for row in rows
            if bool(row.get("deep_agent_used")) is expected
        ]
    summary = summarize_traces(rows)
    business_metrics = _load_business_metrics(args.path)
    if business_metrics is not None:
        summary["business_test"] = business_metrics
    thresholds = {
        key: value
        for key, value in {
            "min_trace_count": args.min_trace_count,
            "min_success_rate": args.min_success_rate,
            "max_p95_latency_ms": args.max_p95_latency_ms,
            "max_timeout_rate": args.max_timeout_rate,
            "max_tool_failure_rate": args.max_tool_failure_rate,
            "max_planner_fallback_rate": args.max_planner_fallback_rate,
            "max_device_state_unknown_rate": (
                args.max_device_state_unknown_rate
            ),
            "max_planner_execution_failure_rate": (
                args.max_planner_execution_failure_rate
            ),
        }.items()
        if value is not None
    }
    try:
        alerts = evaluate_health_alerts(summary, thresholds)
    except ValueError as exc:
        parser.error(str(exc))
    summary["alert_evaluation"] = {
        "thresholds": thresholds,
        "alert_count": len(alerts),
        "alerts": alerts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if args.fail_on_alert and alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
