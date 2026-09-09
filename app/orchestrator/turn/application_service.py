"""单轮对话的应用服务边界。

``TurnOrchestrator`` 只决定 handler 顺序；本模块统一拥有一轮请求的状态水合、
延迟运行时加载、执行账本和响应提交。领域 handler 只处理业务，不再各自拼装
这些横切逻辑。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.orchestrator.turn.execution_journal import TurnExecutionJournal
from app.orchestrator.turn.memory_adapter import MemoryHandlerResult
from app.orchestrator.turn.response_renderer import prepend_response_ack
from app.orchestrator.turn.runtime_models import ResponseEnvelope, TurnRequest
from app.orchestrator.turn.turn_orchestrator import TurnOrchestrator
from app.ports.task_state import DialogueTaskStateRepository

RuntimeLoader = Callable[["TurnExecutionContext"], Awaitable[Any]]
ApplicationTurnHandler = Callable[
    ["TurnExecutionContext"],
    Awaitable[ResponseEnvelope | None],
]
_RUNTIME_NOT_LOADED = object()


@dataclass(slots=True)
class TurnExecutionContext:
    """一轮业务共享的运行上下文。

    运行时数据只允许通过 ``runtime()`` 水合一次。Memory handler 若需要继续路由，
    必须先调用 ``set_memory_result``，后续运行时加载会据此丢弃过期快照。
    """

    request: TurnRequest
    short_term_snapshot: Any
    execution_journal: TurnExecutionJournal
    runtime_loader: RuntimeLoader
    task_state_repository: DialogueTaskStateRepository
    started_at: float = field(default_factory=time.monotonic)
    memory_result: MemoryHandlerResult = field(default_factory=MemoryHandlerResult)
    _runtime: Any = field(default=_RUNTIME_NOT_LOADED, init=False, repr=False)

    async def runtime(self) -> Any:
        if self._runtime is _RUNTIME_NOT_LOADED:
            self._runtime = await self.runtime_loader(self)
        return self._runtime

    def set_memory_result(self, result: MemoryHandlerResult) -> None:
        if self._runtime is not _RUNTIME_NOT_LOADED:
            raise RuntimeError("memory result must be resolved before runtime hydration")
        self.memory_result = result

    def runtime_short_term_snapshot(self) -> Any:
        """Memory 已修改本轮事实时，不向后续 handler 暴露旧快照。"""
        if self.memory_result.continuation_ack:
            return None
        return self.short_term_snapshot

    async def refresh_task_state(self) -> None:
        """Memory handler 修改任务态后，显式刷新 Repository baseline。"""
        refreshed = await self.task_state_repository.refresh_current_scope()
        if refreshed is not None:
            self.short_term_snapshot = refreshed

    async def flush_task_state(self) -> None:
        """高风险 handler 完成外部副作用后立即持久化本轮任务变化。"""
        await self.task_state_repository.flush_current_scope()

    def finalize(self, envelope: ResponseEnvelope) -> ResponseEnvelope:
        handled_by = envelope.handled_by or "unknown"
        finalized = self.memory_result.apply_to(envelope)
        finalized = self.execution_journal.enrich(
            finalized,
            handled_by=handled_by,
        )
        if self.memory_result.continuation_ack:
            finalized = prepend_response_ack(
                finalized,
                self.memory_result.continuation_ack,
            )
        return finalized


class TurnApplicationService:
    """统一执行一次业务轮次。

    状态 scope 在 handler 执行前水合，并在整个轮次结束后统一提交。首个 handler
    命中即结束；应用服务只做一次最终账本和 Memory 结果合并。
    """

    def __init__(
        self,
        *,
        task_state_repository: DialogueTaskStateRepository,
    ) -> None:
        self._task_state_repository = task_state_repository

    async def handle(
        self,
        request: TurnRequest,
        *,
        runtime_loader: RuntimeLoader,
        handlers: Mapping[str, ApplicationTurnHandler],
    ) -> ResponseEnvelope:
        async with self._task_state_repository.turn_scope(
            request.thread_id
        ) as short_term_snapshot:
            context = TurnExecutionContext(
                request=request,
                short_term_snapshot=short_term_snapshot,
                execution_journal=TurnExecutionJournal.start(request.thread_id),
                runtime_loader=runtime_loader,
                task_state_repository=self._task_state_repository,
            )

            def adapt(
                handler: ApplicationTurnHandler | None,
            ) -> Callable[[TurnRequest], Awaitable[ResponseEnvelope | None]] | None:
                if handler is None:
                    return None

                async def adapted(_request: TurnRequest) -> ResponseEnvelope | None:
                    return await handler(context)

                return adapted

            orchestrator = TurnOrchestrator(
                image_handler=adapt(handlers.get("image_handler")),
                reset_command_handler=adapt(handlers.get("reset_command")),
                memory_command_handler=adapt(handlers.get("memory_command")),
                exact_command_handler=adapt(handlers.get("exact_command")),
                device_pending_handler=adapt(handlers.get("device_pending")),
                planner_handler=adapt(handlers.get("planner")),
                pending_state_handler=adapt(handlers.get("pending_state")),
                recipe_handler=adapt(handlers.get("recipe_handler")),
                device_handler=adapt(handlers.get("device_handler")),
                conversation_fallback_handler=adapt(
                    handlers.get("conversation_fallback")
                ),
                fallback_handler=adapt(handlers.get("fallback")),
            )
            envelope = await orchestrator.handle(request)
            return context.finalize(envelope)
