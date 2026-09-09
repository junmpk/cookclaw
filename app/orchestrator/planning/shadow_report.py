"""Turn Planner Shadow 记录的纯函数汇总。"""

from __future__ import annotations

from collections import Counter
from typing import Any


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(
        ordered[lower] * (1 - fraction) + ordered[upper] * fraction,
        2,
    )


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def summarize_shadow_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("schema_version") == "turn_planner_shadow_v1"
    ]
    statuses = Counter(str(row.get("status") or "unknown") for row in records)
    bypass_reasons = Counter(
        str(row.get("bypass_reason") or "unknown")
        for row in records
        if row.get("status") == "bypassed"
    )
    channels = Counter(
        str((row.get("context_ref") or {}).get("channel") or "unknown")
        for row in records
    )
    legacy_actions = Counter(
        str((row.get("legacy") or {}).get("action") or "unknown")
        for row in records
    )
    planner_actions = Counter()
    mismatch_codes = Counter()
    error_codes = Counter()
    durations: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    validation_values: list[bool] = []
    action_values: list[bool] = []
    risk_values: list[bool] = []
    clarification_values: list[bool] = []
    deterministic_values: list[bool] = []

    for row in records:
        planner = row.get("planner") or {}
        for action in planner.get("step_actions") or []:
            planner_actions[str(action)] += 1
        validation = row.get("validation")
        if isinstance(validation, dict) and isinstance(validation.get("valid"), bool):
            validation_values.append(validation["valid"])
        comparison = row.get("comparison")
        if isinstance(comparison, dict):
            for key, target in (
                ("action_match", action_values),
                ("risk_match", risk_values),
                ("clarification_match", clarification_values),
                ("deterministic_handoff_match", deterministic_values),
            ):
                value = comparison.get(key)
                if isinstance(value, bool):
                    target.append(value)
            mismatch_codes.update(
                str(code) for code in comparison.get("mismatch_codes") or []
            )
        if row.get("status") != "bypassed":
            model = row.get("model") or {}
            try:
                durations.append(max(0.0, float(model.get("duration_ms") or 0)))
            except (TypeError, ValueError):
                pass
            input_tokens.append(max(0, int(model.get("input_tokens") or 0)))
            output_tokens.append(max(0, int(model.get("output_tokens") or 0)))
            if model.get("error_code"):
                error_codes[str(model["error_code"])] += 1

    total = len(records)
    evaluated_total = total - statuses.get("bypassed", 0)
    return {
        "record_count": total,
        "evaluated_record_count": evaluated_total,
        "bypass_count": statuses.get("bypassed", 0),
        "planner_evaluation_rate": (
            round(evaluated_total / total, 4)
            if total else None
        ),
        "bypass_reasons": dict(sorted(bypass_reasons.items())),
        "statuses": dict(sorted(statuses.items())),
        "success_rate": (
            round(statuses.get("success", 0) / evaluated_total, 4)
            if evaluated_total else None
        ),
        "validation_valid_rate": _rate(validation_values),
        "comparison": {
            "action_match_rate": _rate(action_values),
            "risk_match_rate": _rate(risk_values),
            "clarification_match_rate": _rate(clarification_values),
            "deterministic_handoff_match_rate": _rate(deterministic_values),
            "mismatch_codes": dict(sorted(mismatch_codes.items())),
        },
        "latency_ms": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": round(max(durations), 2) if durations else None,
        },
        "tokens": {
            "input_total": sum(input_tokens),
            "output_total": sum(output_tokens),
            "total": sum(input_tokens) + sum(output_tokens),
            "average_total": (
                round(
                    (sum(input_tokens) + sum(output_tokens)) / evaluated_total,
                    2,
                )
                if evaluated_total else None
            ),
        },
        "channels": dict(sorted(channels.items())),
        "legacy_actions": dict(sorted(legacy_actions.items())),
        "planner_actions": dict(sorted(planner_actions.items())),
        "errors": dict(sorted(error_codes.items())),
    }
