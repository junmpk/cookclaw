"""从脱敏 Trace 汇总评估显式生产告警阈值。"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested(summary: Mapping[str, Any], *path: str) -> float | None:
    value: Any = summary
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return _number(value)


def evaluate_health_alerts(
    summary: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """返回机器可读告警，不内置未经业务确认的性能/质量数字。

    设备重复指令和 Planner 越权属于架构红线，阈值固定为 0。其余阈值必须由
    调用方基于同模型、同数据、同设备环境的基线显式提供。
    """
    alerts: list[dict[str, Any]] = []
    limits = dict(thresholds or {})

    rate_thresholds = {
        "min_success_rate",
        "max_timeout_rate",
        "max_tool_failure_rate",
        "max_planner_fallback_rate",
        "max_planner_execution_failure_rate",
        "max_device_state_unknown_rate",
    }
    supported_thresholds = rate_thresholds | {
        "max_p95_latency_ms",
        "min_trace_count",
    }
    unknown_thresholds = sorted(set(limits) - supported_thresholds)
    if unknown_thresholds:
        raise ValueError(
            "unsupported health thresholds: " + ", ".join(unknown_thresholds)
        )
    for name, raw_value in limits.items():
        value = _number(raw_value)
        if value is None or not math.isfinite(value):
            raise ValueError(f"health threshold {name} must be a finite number")
        if name in rate_thresholds and not 0 <= value <= 1:
            raise ValueError(f"health threshold {name} must be between 0 and 1")
        if name == "max_p95_latency_ms" and value < 0:
            raise ValueError(
                "health threshold max_p95_latency_ms must be non-negative"
            )
        if name == "min_trace_count" and (
            value < 0 or not value.is_integer()
        ):
            raise ValueError(
                "health threshold min_trace_count must be a non-negative integer"
            )
        limits[name] = value

    def add(
        code: str,
        *,
        severity: str,
        actual: float,
        threshold: float,
        comparison: str,
    ) -> None:
        alerts.append(
            {
                "code": code,
                "severity": severity,
                "actual": actual,
                "threshold": threshold,
                "comparison": comparison,
            }
        )

    duplicate_commands = _nested(
        summary,
        "safety",
        "duplicate_device_command_turns",
    )
    if duplicate_commands is not None and duplicate_commands > 0:
        add(
            "DUPLICATE_DEVICE_COMMAND",
            severity="critical",
            actual=duplicate_commands,
            threshold=0,
            comparison="max",
        )

    forbidden_actions = _nested(
        summary,
        "safety",
        "forbidden_planner_action_events",
    )
    if forbidden_actions is not None and forbidden_actions > 0:
        add(
            "PLANNER_FORBIDDEN_ACTION",
            severity="critical",
            actual=forbidden_actions,
            threshold=0,
            comparison="max",
        )

    checks = {
        "min_trace_count": (
            ("trace_count",),
            "TRACE_COUNT_LOW",
            "min",
        ),
        "min_success_rate": (
            ("success_rate",),
            "TRACE_SUCCESS_RATE_LOW",
            "min",
        ),
        "max_p95_latency_ms": (
            ("latency_ms", "p95"),
            "TRACE_P95_LATENCY_HIGH",
            "max",
        ),
        "max_timeout_rate": (
            ("timeout_rate",),
            "TRACE_TIMEOUT_RATE_HIGH",
            "max",
        ),
        "max_tool_failure_rate": (
            ("tool_failure_rate",),
            "TOOL_FAILURE_RATE_HIGH",
            "max",
        ),
        "max_planner_fallback_rate": (
            ("planner", "fallback_rate"),
            "PLANNER_FALLBACK_RATE_HIGH",
            "max",
        ),
        "max_planner_execution_failure_rate": (
            ("planner", "execution_failure_rate"),
            "PLANNER_EXECUTION_FAILURE_RATE_HIGH",
            "max",
        ),
        "max_device_state_unknown_rate": (
            ("safety", "device_state_unknown_rate"),
            "DEVICE_STATE_UNKNOWN_RATE_HIGH",
            "max",
        ),
    }
    for threshold_name, (path, code, comparison) in checks.items():
        threshold = _number(limits.get(threshold_name))
        actual = _nested(summary, *path)
        if threshold is None or actual is None:
            continue
        violated = (
            actual < threshold if comparison == "min" else actual > threshold
        )
        if violated:
            add(
                code,
                severity="warning",
                actual=actual,
                threshold=threshold,
                comparison=comparison,
            )

    return alerts
