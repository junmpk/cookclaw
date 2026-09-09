"""把真实工具 Trace 和会话状态差异收敛为阶段 2 运行协议。

本模块只观察已经发生的调用和状态变更，不执行工具、不写状态，也不根据回复文案
反推业务事实。输出只保留工具名、结果码、计数和状态字段名，不包含用户原话、
菜谱内容、设备 ID 或完整工具 payload。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.conversation.models import ConversationTaskState
from app.observability.trace import current_trace
from app.orchestrator.turn.runtime_models import ResponseEnvelope, StatePatch, ToolResult
from app.ports.dialogue_state import snapshot_thread_state


_LOGICAL_STATE_KEYS = (
    "current_task",
    "candidate_recipes",
    "candidate_language",
    "candidate_decision_id",
    "active_search_request",
    "selected_recipe_id",
    "selected_recipe_name",
    "selected_recipe",
    "excluded_recipe_ids",
    "focus",
    "pending_search_clarification",
    "pending_action",
    "pending_device_start",
    "device_execution",
    "menu_task",
    "selected_device_id",
    "active_cooking",
    "language",
    "last_tool_result",
)
_DEVICE_STATE_KEYS = {
    "pending_device_start",
    "device_execution",
    "selected_device_id",
    "active_cooking",
}
_SAFE_TOOL_FACT_FIELDS = (
    "tool_kind",
    "result_count",
    "backend",
    "fallback",
    "timeout",
    "command_sent",
    "state_unknown",
)
_MAX_TOOL_RESULTS = 16
_MAX_STATE_PATCHES = 16


def current_trace_event_cursor() -> int:
    """返回当前轮次 Trace 的事件游标；没有 Trace 时为 0。"""
    trace = current_trace()
    return len(trace.events) if trace is not None else 0


def _clean_label(value: Any, *, limit: int = 80) -> str:
    return str(value or "").strip()[:limit]


def _duration(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_empty_state(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _tool_results_since(event_start: int) -> list[ToolResult]:
    trace = current_trace()
    if trace is None:
        return []
    results: list[ToolResult] = []
    for event in trace.events[max(0, event_start):]:
        if event.get("kind") != "tool_call":
            continue
        name = _clean_label(event.get("name"))
        if not name:
            continue
        success = bool(event.get("success"))
        event_code = _clean_label(
            event.get("result_code") or event.get("error_type")
        )
        facts = {
            key: event[key]
            for key in _SAFE_TOOL_FACT_FIELDS
            if key in event
        }
        results.append(
            ToolResult(
                tool=name,
                success=success,
                code=event_code or ("OK" if success else "TOOL_FAILED"),
                facts=facts,
                error_type=(
                    _clean_label(event.get("error_type")) or None
                    if not success
                    else None
                ),
                duration_ms=_duration(event.get("duration_ms")),
            )
        )
        if len(results) >= _MAX_TOOL_RESULTS:
            break
    return results


def latest_tool_result_since(
    event_start: int,
    tool_name: str,
) -> ToolResult | None:
    """返回游标后最后一次指定工具结果，供领域 Adapter 使用真实调用事实。"""
    expected = _clean_label(tool_name)
    matches = [
        result
        for result in _tool_results_since(event_start)
        if result.tool == expected
    ]
    return matches[-1] if matches else None


def tool_results_since(event_start: int) -> list[ToolResult]:
    """返回游标后的脱敏工具结果副本，供失败回退判断是否已产生外部调用。"""
    return list(_tool_results_since(event_start))


def _state_patches(
    before: ConversationTaskState,
    after: ConversationTaskState,
    *,
    reason_code: str,
) -> list[StatePatch]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for key in _LOGICAL_STATE_KEYS:
        before_value = getattr(before, key)
        after_value = getattr(after, key)
        if before_value == after_value:
            continue
        operation = (
            "clear"
            if not _is_empty_state(before_value) and _is_empty_state(after_value)
            else "set"
        )
        scope = "device" if key in _DEVICE_STATE_KEYS else "session"
        grouped.setdefault((scope, operation), []).append(key)

    normalized_reason = _clean_label(reason_code).upper() or "TURN_STATE_CHANGED"
    return [
        StatePatch(
            scope=scope,
            operation=operation,
            keys=keys,
            reason_code=normalized_reason,
        )
        for (scope, operation), keys in grouped.items()
    ]


@dataclass(frozen=True, slots=True)
class TurnExecutionJournal:
    """一轮 handler 的只读执行账本。"""

    thread_id: str
    state_before: ConversationTaskState
    trace_event_start: int

    @classmethod
    def start(
        cls,
        thread_id: str,
        *,
        trace_event_start: int | None = None,
    ) -> "TurnExecutionJournal":
        return cls(
            thread_id=thread_id,
            state_before=snapshot_thread_state(thread_id),
            trace_event_start=(
                current_trace_event_cursor()
                if trace_event_start is None
                else max(0, int(trace_event_start))
            ),
        )

    def enrich(
        self,
        envelope: ResponseEnvelope,
        *,
        handled_by: str,
    ) -> ResponseEnvelope:
        """将实际调用和最终状态差异附加到内部响应，不改变公开 payload。"""
        after = snapshot_thread_state(self.thread_id)
        reason_code = f"{_clean_label(handled_by).upper()}_COMPLETED"
        observed_tool_results = _tool_results_since(self.trace_event_start)
        observed_state_patches = _state_patches(
            self.state_before,
            after,
            reason_code=reason_code,
        )
        tool_results = [
            *envelope.tool_results,
            *observed_tool_results,
        ][:_MAX_TOOL_RESULTS]
        state_patches = [
            *envelope.state_patches,
            *observed_state_patches,
        ][:_MAX_STATE_PATCHES]
        trace = current_trace()
        if trace is not None:
            for patch in state_patches:
                trace.add_event(
                    "state_patch",
                    scope=patch.scope,
                    operation=patch.operation,
                    keys=",".join(patch.keys),
                    key_count=len(patch.keys),
                    reason_code=patch.reason_code,
                )
        return envelope.model_copy(
            update={
                "tool_results": tool_results,
                "state_patches": state_patches,
            }
        )
