"""把画像/偏好命令结果适配为阶段 2 统一运行协议。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.conversation.command_result import MemoryStateChange
from app.orchestrator.turn.runtime_models import ResponseEnvelope, StatePatch, ToolResult


_FAILED_STATUSES = {
    "disabled",
    "invalid",
    "read_failed",
    "version_conflict",
    "write_failed",
}
_MAX_RESULTS = 16


def _status(value: Any) -> str:
    return str(value or "handled").strip().lower()[:64] or "handled"


def _result_code(prefix: str, status: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", status.upper()).strip("_")
    return f"{prefix}_{normalized or 'HANDLED'}"[:80]


def _memory_envelope(
    message: str,
    *,
    lang: str,
    trace_id: str | None,
) -> ResponseEnvelope:
    payload = {
        "type": "cooking",
        "intent": "chat",
        "lang": lang,
        "data": {},
        "message": str(message or ""),
    }
    return ResponseEnvelope(
        response_type="cooking",
        intent="chat",
        lang=lang,
        message=payload["message"],
        data={},
        channel_payload=payload,
        handled_by="memory_command",
        trace_id=trace_id,
    )


def _state_patches(reply: Any) -> tuple[StatePatch, ...]:
    changes = getattr(reply, "state_changes", ()) or ()
    patches: list[StatePatch] = []
    for change in changes:
        if not isinstance(change, MemoryStateChange) or not change.keys:
            continue
        patches.append(
            StatePatch(
                scope=change.scope,
                operation=change.operation,
                keys=list(change.keys),
                reason_code=str(change.reason_code or "MEMORY_STATE_CHANGED")[:80],
            )
        )
        if len(patches) >= _MAX_RESULTS:
            break
    return tuple(patches)


@dataclass(frozen=True, slots=True)
class MemoryHandlerResult:
    """Memory handler 的显式结果；终止响应与继续路由互斥。"""

    terminal_envelope: ResponseEnvelope | None = None
    continuation_ack: str | None = None
    tool_results: tuple[ToolResult, ...] = ()
    state_patches: tuple[StatePatch, ...] = ()

    def apply_to(self, envelope: ResponseEnvelope) -> ResponseEnvelope:
        """把 Memory 领域结果并入内部协议，不改变公开响应形状。"""
        return envelope.model_copy(
            update={
                "tool_results": [
                    *envelope.tool_results,
                    *self.tool_results,
                ][:_MAX_RESULTS],
                "state_patches": [
                    *envelope.state_patches,
                    *self.state_patches,
                ][:_MAX_RESULTS],
            }
        )


def adapt_profile_reply(
    reply: Any,
    *,
    trace_id: str | None,
    duration_ms: float,
) -> MemoryHandlerResult:
    status = _status(getattr(reply, "status", "handled"))
    success = bool(
        getattr(reply, "success", status not in _FAILED_STATUSES)
    )
    code = _result_code("PROFILE", status)
    patches = _state_patches(reply)
    return MemoryHandlerResult(
        terminal_envelope=_memory_envelope(
            getattr(reply, "message", ""),
            lang=str(getattr(reply, "lang", "zh") or "zh")[:16],
            trace_id=trace_id,
        ),
        tool_results=(
            ToolResult(
                tool="profile_memory",
                success=success,
                code=code,
                facts={"domain": "memory", "status": status},
                error_type=None if success else code,
                duration_ms=max(0.0, float(duration_ms)),
            ),
        ),
        state_patches=patches,
    )


def adapt_preference_reply(
    reply: Any,
    *,
    trace_id: str | None,
    duration_ms: float,
) -> MemoryHandlerResult:
    status = _status(getattr(reply, "status", "handled"))
    success = bool(
        getattr(reply, "success", status not in _FAILED_STATUSES)
    )
    continuation = bool(getattr(reply, "continue_routing", False))
    code = _result_code("PREFERENCE", status)
    patches = _state_patches(reply)
    tool_result = ToolResult(
        tool="preference_memory",
        success=success,
        code=code,
        facts={
            "domain": "memory",
            "status": status,
            "continuation": continuation,
            "changed_field_count": sum(len(patch.keys) for patch in patches),
        },
        error_type=None if success else code,
        duration_ms=max(0.0, float(duration_ms)),
    )
    message = str(getattr(reply, "message", ""))
    lang = str(getattr(reply, "lang", "zh") or "zh")[:16]
    return MemoryHandlerResult(
        terminal_envelope=(
            None
            if continuation
            else _memory_envelope(message, lang=lang, trace_id=trace_id)
        ),
        continuation_ack=message if continuation else None,
        tool_results=(tool_result,),
        state_patches=patches,
    )


def adapt_general_profile_reply(
    reply: Any,
    *,
    trace_id: str | None,
    duration_ms: float,
) -> MemoryHandlerResult:
    """把通用画像写删结果转换为统一 Memory 运行协议。"""
    status = _status(getattr(reply, "status", "handled"))
    success = bool(getattr(reply, "success", False))
    code = _result_code("GENERAL_PROFILE", status)
    patches = _state_patches(reply)
    return MemoryHandlerResult(
        terminal_envelope=_memory_envelope(
            getattr(reply, "message", ""),
            lang=str(getattr(reply, "lang", "zh") or "zh")[:16],
            trace_id=trace_id,
        ),
        tool_results=(
            ToolResult(
                tool="general_profile_memory",
                success=success,
                code=code,
                facts={
                    "domain": "memory",
                    "status": status,
                    "candidate_count": int(
                        getattr(reply, "candidate_count", 0) or 0
                    ),
                    "accepted_count": int(
                        getattr(reply, "accepted_count", 0) or 0
                    ),
                    "rejected_count": int(
                        getattr(reply, "rejected_count", 0) or 0
                    ),
                    "changed_count": int(
                        getattr(reply, "changed_count", 0) or 0
                    ),
                },
                error_type=(
                    None
                    if success
                    else str(getattr(reply, "error_code", None) or code)[:80]
                ),
                duration_ms=max(0.0, float(duration_ms)),
            ),
        ),
        state_patches=patches,
    )
