"""统一 TurnApplicationService 生命周期回归。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from app.orchestrator.turn.application_service import TurnApplicationService
from app.orchestrator.turn.memory_adapter import MemoryHandlerResult
from app.orchestrator.turn.runtime_models import ResponseEnvelope, TurnRequest


def _request() -> TurnRequest:
    return TurnRequest(
        utterance="继续推荐",
        thread_id="qq:dm:application-service:test",
        channel="qq",
        trace_id="application-service-trace",
    )


class _ScopeRepository:
    def __init__(self, scope_factory):
        self._scope_factory = scope_factory

    def turn_scope(self, thread_id):
        return self._scope_factory(thread_id)

    async def refresh_current_scope(self):
        return None


def test_application_service_hydrates_runtime_once_and_finalizes_once():
    events = []

    @asynccontextmanager
    async def state_scope(thread_id):
        events.append(("hydrate", thread_id))
        yield {"version": 3}
        events.append(("commit", thread_id))

    async def runtime_loader(context):
        events.append(("runtime", context.runtime_short_term_snapshot()))
        return {"loaded": True}

    async def memory(context):
        context.set_memory_result(
            MemoryHandlerResult(continuation_ack="已更新你的偏好。")
        )

    async def exact(context):
        assert await context.runtime() == {"loaded": True}

    async def recipe(context):
        assert await context.runtime() == {"loaded": True}
        return ResponseEnvelope(message="给你推荐三道菜。")

    envelope = asyncio.run(
        TurnApplicationService(
            task_state_repository=_ScopeRepository(state_scope)
        ).handle(
            _request(),
            runtime_loader=runtime_loader,
            handlers={
                "memory_command": memory,
                "exact_command": exact,
                "recipe_handler": recipe,
            },
        )
    )

    assert envelope.handled_by == "recipe_handler"
    assert envelope.message == "已更新你的偏好。\n\n给你推荐三道菜。"
    assert [name for name, _value in events].count("runtime") == 1
    assert events[0][0] == "hydrate"
    assert events[-1][0] == "commit"


def test_terminal_memory_handler_does_not_hydrate_business_runtime():
    runtime_calls = 0

    @asynccontextmanager
    async def state_scope(_thread_id):
        yield None

    async def runtime_loader(_context):
        nonlocal runtime_calls
        runtime_calls += 1
        return object()

    async def memory(context):
        result = MemoryHandlerResult(
            terminal_envelope=ResponseEnvelope(message="已清除。"),
        )
        context.set_memory_result(result)
        return result.terminal_envelope

    envelope = asyncio.run(
        TurnApplicationService(
            task_state_repository=_ScopeRepository(state_scope)
        ).handle(
            _request(),
            runtime_loader=runtime_loader,
            handlers={"memory_command": memory},
        )
    )

    assert envelope.handled_by == "memory_command"
    assert envelope.message == "已清除。"
    assert runtime_calls == 0
