"""一轮对话内的结构化 Trace、计数器和 Token 汇总。

只记录阶段、计数、耗时、错误类型和哈希标识；不记录用户原文、菜谱正文、
设备 ID、Token、密码或工具完整 payload。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_TRACE_SCHEMA = "conversation_trace_v1"
_MAX_EVENTS = 80
_file_lock = threading.Lock()
_current_trace: ContextVar["TurnTrace | None"] = ContextVar(
    "cookclaw_turn_trace",
    default=None,
)
_trace_collectors: ContextVar[tuple[list[dict[str, Any]], ...]] = ContextVar(
    "cookclaw_trace_collectors",
    default=(),
)


def _fingerprint(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _clean_label(value: object, *, limit: int = 80) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _number(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _usage_from_value(value: Any) -> tuple[int, int]:
    """兼容 AIMessage、ModelResponse 和不同 provider 的 usage 字段。"""
    if value is None:
        return 0, 0
    if isinstance(value, (list, tuple)):
        usages = [_usage_from_value(item) for item in value]
        return sum(item[0] for item in usages), sum(item[1] for item in usages)
    if isinstance(value, dict):
        usage = value.get("usage") or value.get("token_usage") or {}
        if isinstance(usage, dict):
            return (
                _number(usage.get("input_tokens", usage.get("prompt_tokens", 0))),
                _number(usage.get("output_tokens", usage.get("completion_tokens", 0))),
            )

    result = getattr(value, "result", None)
    if isinstance(result, (list, tuple)):
        return _usage_from_value(result)

    usage = getattr(value, "usage_metadata", None)
    if usage:
        getter = usage.get if hasattr(usage, "get") else lambda key, default=0: getattr(
            usage, key, default
        )
        return (
            _number(getter("input_tokens", getter("prompt_tokens", 0))),
            _number(getter("output_tokens", getter("completion_tokens", 0))),
        )

    usage = getattr(value, "usage", None)
    if usage:
        getter = usage.get if hasattr(usage, "get") else lambda key, default=0: getattr(
            usage, key, default
        )
        return (
            _number(getter("input_tokens", getter("prompt_tokens", 0))),
            _number(getter("output_tokens", getter("completion_tokens", 0))),
        )

    metadata = getattr(value, "response_metadata", None)
    if isinstance(metadata, dict):
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
        if isinstance(usage, dict):
            return (
                _number(usage.get("input_tokens", usage.get("prompt_tokens", 0))),
                _number(usage.get("output_tokens", usage.get("completion_tokens", 0))),
            )
    return 0, 0


def extract_token_usage(value: Any) -> dict[str, int]:
    input_tokens, output_tokens = _usage_from_value(value)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


@dataclass
class TurnTrace:
    trace_id: str
    channel: str
    thread_hash: str
    message_type: str
    started_at: float = field(default_factory=time.time)
    started_monotonic: float = field(default_factory=time.monotonic, repr=False)
    deep_agent_enabled: bool = False
    deep_agent_used: bool = False
    deep_agent_cohort: str = "unknown"
    status: str = "running"
    response_type: str | None = None
    error_type: str | None = None
    route: dict[str, Any] = field(default_factory=dict)
    model_call_count: int = 0
    intent_call_count: int = 0
    tool_call_count: int = 0
    search_call_count: int = 0
    timeout_count: int = 0
    tool_failure_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    max_model_input_tokens: int = 0
    search_duration_ms: float = 0.0
    reply_generation_duration_ms: float = 0.0
    durations_ms: dict[str, float] = field(default_factory=dict)
    fallback_reasons: list[str] = field(default_factory=list)
    quality_checks: dict[str, bool] = field(default_factory=dict)
    reply_validation: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def add_duration(self, stage: str, duration_ms: float) -> None:
        key = _clean_label(stage)
        if not key:
            return
        self.durations_ms[key] = round(
            self.durations_ms.get(key, 0.0) + max(0.0, float(duration_ms)),
            2,
        )

    def add_event(self, kind: str, **fields: Any) -> None:
        if len(self.events) >= _MAX_EVENTS:
            return
        event = {
            "kind": _clean_label(kind),
            "at_ms": round(
                max(0.0, (time.monotonic() - self.started_monotonic) * 1000),
                2,
            ),
        }
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, str):
                event[_clean_label(key, limit=40)] = _clean_label(value, limit=120)
            elif isinstance(value, (bool, int, float)):
                event[_clean_label(key, limit=40)] = value
        self.events.append(event)

    def snapshot(self) -> dict[str, Any]:
        assessed = len(self.quality_checks)
        passed = sum(1 for value in self.quality_checks.values() if value)
        return {
            "schema_version": _TRACE_SCHEMA,
            "trace_id": self.trace_id,
            "channel": self.channel,
            "thread_hash": self.thread_hash,
            "message_type": self.message_type,
            "started_at": round(self.started_at, 3),
            "status": self.status,
            "response_type": self.response_type,
            "error_type": self.error_type,
            "deep_agent_enabled": self.deep_agent_enabled,
            "deep_agent_used": self.deep_agent_used,
            "deep_agent_cohort": self.deep_agent_cohort,
            "route": dict(self.route),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "durations_ms": {
                key: round(value, 2)
                for key, value in sorted(self.durations_ms.items())
            },
            "model_call_count": self.model_call_count,
            "intent_call_count": self.intent_call_count,
            "tool_call_count": self.tool_call_count,
            "search_call_count": self.search_call_count,
            "timeout_count": self.timeout_count,
            "tool_failure_count": self.tool_failure_count,
            "token_usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
            "context": {
                "max_model_input_tokens": self.max_model_input_tokens,
            },
            "search_duration_ms": round(self.search_duration_ms, 2),
            "reply_generation_duration_ms": round(
                self.reply_generation_duration_ms,
                2,
            ),
            "fallback_reasons": list(self.fallback_reasons),
            "reply_validation": dict(self.reply_validation),
            "quality": {
                "assessed_checks": assessed,
                "passed_checks": passed,
                "effective": (passed == assessed) if assessed else None,
                "checks": dict(self.quality_checks),
            },
            "events": list(self.events),
        }


def current_trace() -> TurnTrace | None:
    return _current_trace.get()


def current_trace_id() -> str | None:
    trace = current_trace()
    return trace.trace_id if trace else None


@contextmanager
def collect_turn_traces() -> Iterator[list[dict[str, Any]]]:
    """在当前异步上下文中收集最终 Trace，供本地回归器关联到具体轮次。"""
    rows: list[dict[str, Any]] = []
    token = _trace_collectors.set((*_trace_collectors.get(), rows))
    try:
        yield rows
    finally:
        _trace_collectors.reset(token)


def ensure_turn_trace(
    *,
    channel: str,
    thread_id: str,
    message_type: str = "text",
    started_monotonic: float | None = None,
    deep_agent_enabled: bool = False,
    deep_agent_cohort: str = "unknown",
) -> tuple[TurnTrace, Token[TurnTrace | None] | None]:
    existing = current_trace()
    if existing is not None:
        if deep_agent_enabled:
            existing.deep_agent_enabled = True
        if deep_agent_cohort != "unknown":
            existing.deep_agent_cohort = _clean_label(
                deep_agent_cohort,
                limit=40,
            )
        return existing, None
    trace = TurnTrace(
        trace_id=uuid.uuid4().hex,
        channel=_clean_label(channel or "unknown", limit=24).lower(),
        thread_hash=_fingerprint(thread_id),
        message_type=_clean_label(message_type or "text", limit=24).lower(),
        started_monotonic=started_monotonic or time.monotonic(),
        deep_agent_enabled=bool(deep_agent_enabled),
        deep_agent_cohort=_clean_label(deep_agent_cohort, limit=40) or "unknown",
    )
    token = _current_trace.set(trace)
    trace.add_event("turn_started")
    return trace, token


def _write_jsonl(payload: dict[str, Any]) -> None:
    path_value = os.getenv("CONVERSATION_TRACE_JSONL_PATH", "").strip()
    if not path_value:
        return
    path = Path(path_value).expanduser()
    try:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with _file_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:
        logger.warning(
            "conversation trace JSONL write failed: error_type=%s",
            type(exc).__name__,
        )


def finish_turn_trace(
    token: Token[TurnTrace | None] | None,
    *,
    success: bool,
    response_type: str | None = None,
    error_type: str | None = None,
) -> dict[str, Any] | None:
    if token is None:
        return None
    trace = current_trace()
    if trace is None:
        _current_trace.reset(token)
        return None
    trace.status = "success" if success else "error"
    trace.response_type = _clean_label(response_type) or trace.response_type
    trace.error_type = _clean_label(error_type) or None
    trace.total_duration_ms = max(
        0.0,
        (time.monotonic() - trace.started_monotonic) * 1000,
    )
    trace.add_event("turn_finished", success=success)
    payload = trace.snapshot()
    for collector in _trace_collectors.get():
        collector.append(payload)
    if os.getenv("CONVERSATION_TRACE_ENABLED", "true").lower() == "true":
        logger.info(
            "conversation_turn_trace %s",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        _write_jsonl(payload)
    _current_trace.reset(token)
    return payload


@contextmanager
def trace_stage(stage: str) -> Iterator[None]:
    trace = current_trace()
    started = time.monotonic()
    try:
        yield
    except BaseException as exc:
        if trace is not None:
            trace.add_event(
                "stage_failed",
                stage=stage,
                error_type=type(exc).__name__,
            )
        raise
    finally:
        if trace is not None:
            trace.add_duration(stage, (time.monotonic() - started) * 1000)


async def observe_model_call(
    factory: Callable[[], Awaitable[_T]],
    *,
    stage: str,
    model: str = "",
    timeout_seconds: float | None = None,
    intent: bool = False,
    reply_generation: bool = False,
) -> _T:
    """执行并记录一次真实模型请求；一次 Agent 图中的每次模型循环都应调用。"""
    trace = current_trace()
    started = time.monotonic()
    try:
        awaitable = factory()
        result = (
            await asyncio.wait_for(awaitable, timeout=max(0.1, timeout_seconds))
            if timeout_seconds is not None
            else await awaitable
        )
    except asyncio.TimeoutError:
        duration_ms = (time.monotonic() - started) * 1000
        if trace is not None:
            trace.model_call_count += 1
            trace.intent_call_count += int(intent)
            trace.timeout_count += 1
            trace.add_duration(stage, duration_ms)
            if reply_generation:
                trace.reply_generation_duration_ms += duration_ms
            trace.add_event(
                "model_call",
                stage=stage,
                model=model,
                success=False,
                timeout=True,
                duration_ms=round(duration_ms, 2),
                error_type="TimeoutError",
            )
        raise
    except asyncio.CancelledError:
        duration_ms = (time.monotonic() - started) * 1000
        if trace is not None:
            trace.model_call_count += 1
            trace.intent_call_count += int(intent)
            trace.timeout_count += 1
            trace.add_duration(stage, duration_ms)
            if reply_generation:
                trace.reply_generation_duration_ms += duration_ms
            trace.add_event(
                "model_call",
                stage=stage,
                model=model,
                success=False,
                timeout=True,
                duration_ms=round(duration_ms, 2),
                error_type="CancelledError",
            )
        raise
    except Exception as exc:
        duration_ms = (time.monotonic() - started) * 1000
        if trace is not None:
            trace.model_call_count += 1
            trace.intent_call_count += int(intent)
            trace.add_duration(stage, duration_ms)
            if reply_generation:
                trace.reply_generation_duration_ms += duration_ms
            trace.add_event(
                "model_call",
                stage=stage,
                model=model,
                success=False,
                duration_ms=round(duration_ms, 2),
                error_type=type(exc).__name__,
            )
        raise

    duration_ms = (time.monotonic() - started) * 1000
    usage = extract_token_usage(result)
    if trace is not None:
        trace.model_call_count += 1
        trace.intent_call_count += int(intent)
        trace.input_tokens += usage["input_tokens"]
        trace.output_tokens += usage["output_tokens"]
        trace.max_model_input_tokens = max(
            trace.max_model_input_tokens,
            usage["input_tokens"],
        )
        trace.add_duration(stage, duration_ms)
        if reply_generation:
            trace.reply_generation_duration_ms += duration_ms
        trace.add_event(
            "model_call",
            stage=stage,
            model=model,
            success=True,
            duration_ms=round(duration_ms, 2),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )
    return result


def record_tool_call(
    name: str,
    *,
    duration_ms: float,
    success: bool,
    kind: str = "tool",
    timeout: bool = False,
    error_type: str | None = None,
    result_count: int | None = None,
    backend: str | None = None,
    fallback: bool | None = None,
    result_code: str | None = None,
    command_sent: bool | None = None,
    state_unknown: bool | None = None,
) -> None:
    trace = current_trace()
    if trace is None:
        return
    trace.tool_call_count += 1
    if kind == "search":
        trace.search_call_count += 1
        trace.search_duration_ms += max(0.0, float(duration_ms))
    if not success:
        trace.tool_failure_count += 1
    if timeout:
        trace.timeout_count += 1
    normalized_result_code = _clean_label(
        result_code
        or (error_type if not success else None)
        or ("OK" if success else "TOOL_FAILED")
    )
    trace.add_duration(name, duration_ms)
    trace.add_event(
        "tool_call",
        name=name,
        tool_kind=kind,
        success=success,
        timeout=timeout,
        duration_ms=round(duration_ms, 2),
        error_type=error_type,
        result_count=result_count,
        backend=backend,
        fallback=fallback,
        result_code=normalized_result_code,
        command_sent=command_sent,
        state_unknown=state_unknown,
    )


def record_domain_result(
    name: str,
    *,
    duration_ms: float,
    success: bool,
    result_code: str,
    state_patch_count: int = 0,
    continuation: bool = False,
    operation: str | None = None,
    result_count: int | None = None,
    command_sent: bool | None = None,
    state_unknown: bool | None = None,
    risk: str | None = None,
) -> None:
    """记录显式领域 Adapter 结果，不把它伪装成外部工具调用。"""
    trace = current_trace()
    if trace is None:
        return
    trace.add_duration(name, duration_ms)
    trace.add_event(
        "domain_result",
        name=name,
        success=success,
        result_code=result_code,
        duration_ms=round(max(0.0, float(duration_ms)), 2),
        state_patch_count=max(0, int(state_patch_count)),
        continuation=continuation,
        operation=operation,
        result_count=(
            max(0, int(result_count))
            if result_count is not None
            else None
        ),
        command_sent=command_sent,
        state_unknown=state_unknown,
        risk=risk,
    )


def record_timeout(stage: str, *, error_type: str = "TimeoutError") -> None:
    trace = current_trace()
    if trace is not None:
        trace.timeout_count += 1
        trace.add_event(
            "timeout",
            stage=stage,
            error_type=error_type,
        )


def trace_async_tool(name: str, *, kind: str = "tool"):
    """为返回 ``{"ok": bool}`` 的异步业务函数记录一次逻辑 Tool 调用。"""
    def decorator(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            started = time.monotonic()
            try:
                result = await func(*args, **kwargs)
            except asyncio.TimeoutError:
                record_tool_call(
                    name,
                    duration_ms=(time.monotonic() - started) * 1000,
                    success=False,
                    kind=kind,
                    timeout=True,
                    error_type="TimeoutError",
                )
                raise
            except Exception as exc:
                record_tool_call(
                    name,
                    duration_ms=(time.monotonic() - started) * 1000,
                    success=False,
                    kind=kind,
                    error_type=type(exc).__name__,
                )
                raise
            success = not isinstance(result, dict) or bool(result.get("ok", True))
            code = str(result.get("code") or "") if isinstance(result, dict) else ""
            command_sent = (
                result.get("command_sent")
                if isinstance(result, dict)
                and isinstance(result.get("command_sent"), bool)
                else None
            )
            state_unknown = (
                code.upper()
                in {
                    "DEVICE_STATE_UNVERIFIED",
                    "DEVICE_OUTCOME_UNKNOWN",
                    "STOP_OUTCOME_UNKNOWN",
                }
            )
            record_tool_call(
                name,
                duration_ms=(time.monotonic() - started) * 1000,
                success=success,
                kind=kind,
                timeout="TIMEOUT" in code.upper(),
                error_type=(code or None) if not success else None,
                result_code=code or ("OK" if success else "TOOL_FAILED"),
                command_sent=command_sent,
                state_unknown=state_unknown or None,
            )
            return result
        return wrapped
    return decorator


def record_route_decision(decision: Any, *, outcome: str) -> None:
    trace = current_trace()
    if trace is None or decision is None:
        return
    trace.route = {
        "trace_id": _clean_label(getattr(decision, "trace_id", "")),
        "source": _clean_label(getattr(decision, "source", "")),
        "reason_code": _clean_label(getattr(decision, "reason_code", "")),
        "category": _clean_label(getattr(decision, "category", "")),
        "action": _clean_label(getattr(decision, "action", "")),
        "risk": _clean_label(getattr(decision, "risk", "")),
        "outcome": _clean_label(outcome),
        "classifier_called": bool(getattr(decision, "classifier_called", False)),
    }
    trace.add_event(
        "route_decision",
        action=trace.route["action"],
        reason_code=trace.route["reason_code"],
        outcome=outcome,
    )


def mark_fallback(reason: str) -> None:
    trace = current_trace()
    value = _clean_label(reason, limit=120)
    if trace is not None and value and value not in trace.fallback_reasons:
        trace.fallback_reasons.append(value)
        trace.add_event("fallback", reason=value)


def record_reply_validation(*, checked: int, kept: int) -> None:
    trace = current_trace()
    if trace is None:
        return
    trace.reply_validation["checked"] = trace.reply_validation.get("checked", 0) + max(
        0, int(checked)
    )
    trace.reply_validation["kept"] = trace.reply_validation.get("kept", 0) + max(
        0, int(kept)
    )


def record_quality_check(name: str, passed: bool) -> None:
    trace = current_trace()
    key = _clean_label(name, limit=80)
    if trace is not None and key:
        trace.quality_checks[key] = bool(passed)


def record_reply(response_type: str) -> None:
    trace = current_trace()
    if trace is not None:
        trace.response_type = _clean_label(response_type)
        trace.add_event("reply_ready", response_type=trace.response_type)
