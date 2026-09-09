"""从 conversation_trace_v1 记录计算可复现基线。"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    weight = position - lower
    return round(
        ordered[lower] * (1 - weight) + ordered[upper] * weight,
        2,
    )


def summarize_traces(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    traces = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("schema_version") == "conversation_trace_v1"
    ]
    durations = [float(row.get("total_duration_ms") or 0) for row in traces]
    model_calls = [float(row.get("model_call_count") or 0) for row in traces]
    tool_calls = [float(row.get("tool_call_count") or 0) for row in traces]
    tokens = [
        float((row.get("token_usage") or {}).get("total_tokens") or 0)
        for row in traces
    ]
    context_tokens = [
        float((row.get("context") or {}).get("max_model_input_tokens") or 0)
        for row in traces
    ]
    search_durations = [
        float(row.get("search_duration_ms") or 0)
        for row in traces
    ]
    reply_durations = [
        float(row.get("reply_generation_duration_ms") or 0)
        for row in traces
    ]
    assessed = [
        row for row in traces
        if int((row.get("quality") or {}).get("assessed_checks") or 0) > 0
    ]
    effective = [
        row for row in assessed
        if (row.get("quality") or {}).get("effective") is True
    ]
    fallback_turns = [row for row in traces if row.get("fallback_reasons")]
    deep_agent_used_turns = [
        row for row in traces if row.get("deep_agent_used") is True
    ]
    timeout_turns = [
        row for row in traces if int(row.get("timeout_count") or 0) > 0
    ]
    tool_failure_turns = [
        row for row in traces if int(row.get("tool_failure_count") or 0) > 0
    ]
    errors = Counter(
        str(row.get("error_type") or "unknown")
        for row in traces
        if row.get("status") != "success"
    )
    route_actions = Counter(
        str((row.get("route") or {}).get("action") or "unknown")
        for row in traces
    )
    profile_events = [
        event
        for row in traces
        for event in (row.get("events") or [])
        if isinstance(event, dict)
        and event.get("kind") == "profile_memory"
    ]
    profile_actions = Counter(
        str(event.get("action") or "unknown")
        for event in profile_events
    )
    profile_statuses = Counter(
        str(event.get("status") or "unknown")
        for event in profile_events
    )
    profile_failed = sum(
        status in {"write_failed", "version_conflict"}
        for status in profile_statuses.elements()
    )
    events_by_trace = [
        [event for event in row.get("events") or [] if isinstance(event, dict)]
        for row in traces
    ]
    planner_events = [
        event
        for events in events_by_trace
        for event in events
        if event.get("kind") == "planner_active"
    ]
    planner_statuses = Counter(
        str(event.get("status") or "unknown")
        for event in planner_events
    )
    planner_execution_events = [
        event
        for events in events_by_trace
        for event in events
        if event.get("kind") == "planner_execution"
    ]
    planner_execution_statuses = Counter(
        str(event.get("status") or "unknown")
        for event in planner_execution_events
    )
    handler_events = [
        event
        for events in events_by_trace
        for event in events
        if event.get("kind") == "turn_orchestrator"
    ]
    handler_counts = Counter(
        str(event.get("handled_by") or "unknown")
        for event in handler_events
    )
    tool_events = [
        event
        for events in events_by_trace
        for event in events
        if event.get("kind") == "tool_call"
    ]
    tool_failures_by_name = Counter(
        str(event.get("name") or "unknown")
        for event in tool_events
        if event.get("success") is False
    )
    fallback_reasons = Counter(
        str(reason or "unknown")
        for row in traces
        for reason in row.get("fallback_reasons") or []
    )
    message_types = Counter(
        str(row.get("message_type") or "unknown") for row in traces
    )
    duplicate_device_command_turns = 0
    forbidden_planner_action_events = 0
    device_state_unknown_turns = 0
    safety_block_turns = 0
    forbidden_planner_actions = {
        "device.start",
        "device.stop",
        "device.confirm",
        "memory.write",
    }
    for row, events in zip(traces, events_by_trace):
        sent_commands = [
            event
            for event in events
            if event.get("kind") == "tool_call"
            and event.get("name") in {"device_start", "device_stop"}
            and event.get("command_sent") is True
        ]
        command_counts = Counter(
            str(event.get("name")) for event in sent_commands
        )
        if any(count > 1 for count in command_counts.values()):
            duplicate_device_command_turns += 1
        if any(
            event.get("kind") == "planner_execution"
            and event.get("action") in forbidden_planner_actions
            for event in events
        ):
            forbidden_planner_action_events += 1
        if any(
            event.get("kind") == "tool_call"
            and event.get("name") in {"device_start", "device_stop", "device_status"}
            and event.get("state_unknown") is True
            for event in events
        ):
            device_state_unknown_turns += 1
        if (
            str((row.get("route") or {}).get("action") or "")
            == "safety_block"
            or any(
                event.get("kind") == "domain_result"
                and "SAFETY" in str(event.get("result_code") or "").upper()
                for event in events
            )
        ):
            safety_block_turns += 1
    total = len(traces)
    return {
        "trace_count": total,
        "success_rate": (
            round(sum(row.get("status") == "success" for row in traces) / total, 4)
            if total else None
        ),
        "latency_ms": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": round(max(durations), 2) if durations else None,
        },
        "model_calls": {
            "average": round(sum(model_calls) / total, 2) if total else None,
            "p95": _percentile(model_calls, 0.95),
        },
        "tool_calls": {
            "average": round(sum(tool_calls) / total, 2) if total else None,
            "p95": _percentile(tool_calls, 0.95),
        },
        "tokens": {
            "average": round(sum(tokens) / total, 2) if total else None,
            "p95": _percentile(tokens, 0.95),
            "total": int(sum(tokens)),
        },
        "context_tokens": {
            "average_max_per_turn": (
                round(sum(context_tokens) / total, 2) if total else None
            ),
            "p95_max_per_turn": _percentile(context_tokens, 0.95),
        },
        "search_duration_ms": {
            "p50": _percentile(search_durations, 0.50),
            "p95": _percentile(search_durations, 0.95),
        },
        "reply_generation_duration_ms": {
            "p50": _percentile(reply_durations, 0.50),
            "p95": _percentile(reply_durations, 0.95),
        },
        "intent_call_count": sum(int(row.get("intent_call_count") or 0) for row in traces),
        "search_call_count": sum(int(row.get("search_call_count") or 0) for row in traces),
        "timeout_count": sum(int(row.get("timeout_count") or 0) for row in traces),
        "tool_failure_count": sum(
            int(row.get("tool_failure_count") or 0) for row in traces
        ),
        "timeout_rate": (
            round(len(timeout_turns) / total, 4)
            if total else None
        ),
        "tool_failure_rate": (
            round(len(tool_failure_turns) / total, 4)
            if total else None
        ),
        "fallback_rate": (
            round(len(fallback_turns) / total, 4) if total else None
        ),
        "deep_agent_used_rate": (
            round(len(deep_agent_used_turns) / total, 4) if total else None
        ),
        "quality": {
            "assessed_turns": len(assessed),
            "effective_turns": len(effective),
            "effective_rate": (
                round(len(effective) / len(assessed), 4) if assessed else None
            ),
        },
        "route_actions": dict(sorted(route_actions.items())),
        "profile_memory": {
            "event_count": len(profile_events),
            "actions": dict(sorted(profile_actions.items())),
            "statuses": dict(sorted(profile_statuses.items())),
            "failure_rate": (
                round(profile_failed / len(profile_events), 4)
                if profile_events else None
            ),
        },
        "planner": {
            "attempt_count": len(planner_events),
            "statuses": dict(sorted(planner_statuses.items())),
            "fallback_rate": (
                round(planner_statuses.get("fallback", 0) / len(planner_events), 4)
                if planner_events else None
            ),
            "execution_count": len(planner_execution_events),
            "execution_statuses": dict(
                sorted(planner_execution_statuses.items())
            ),
            "execution_failure_rate": (
                round(
                    sum(
                        count
                        for status, count in planner_execution_statuses.items()
                        if status not in {"executed", "deterministic_preflight"}
                    ) / len(planner_execution_events),
                    4,
                )
                if planner_execution_events else None
            ),
        },
        "handlers": dict(sorted(handler_counts.items())),
        "message_types": dict(sorted(message_types.items())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "tool_failures_by_name": dict(sorted(tool_failures_by_name.items())),
        "safety": {
            "safety_block_turns": safety_block_turns,
            "duplicate_device_command_turns": duplicate_device_command_turns,
            "forbidden_planner_action_events": forbidden_planner_action_events,
            "device_state_unknown_turns": device_state_unknown_turns,
            "device_state_unknown_rate": (
                round(device_state_unknown_turns / total, 4)
                if total else None
            ),
        },
        "errors": dict(sorted(errors.items())),
    }
