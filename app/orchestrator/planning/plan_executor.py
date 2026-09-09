"""Planner action 的封闭单步执行器。

执行器只调用注册时显式传入的 handler；它不动态 import、不反射函数、不开放工具名，
也不允许一轮执行多个 Planner step。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from app.orchestrator.planning.models import PlanStep, TurnContext, TurnPlan
from app.orchestrator.turn.runtime_models import ResponseEnvelope


PlanActionHandler = Callable[
    [PlanStep, TurnPlan, TurnContext],
    Awaitable[ResponseEnvelope],
]


@dataclass(frozen=True, slots=True)
class PlanExecutionResult:
    status: str
    action: str | None = None
    envelope: ResponseEnvelope | None = None
    error_code: str | None = None


class BoundedPlanExecutor:
    """执行一个已通过 Validator 的单步 plan。"""

    def __init__(
        self,
        handlers: Mapping[str, PlanActionHandler],
        *,
        allowed_actions: set[str],
    ) -> None:
        self._handlers = dict(handlers)
        self._allowed_actions = set(allowed_actions)

    async def execute(
        self,
        plan: TurnPlan,
        context: TurnContext,
    ) -> PlanExecutionResult:
        if len(plan.steps) != 1:
            return PlanExecutionResult(
                status="rejected",
                error_code="ACTIVE_PLAN_REQUIRES_ONE_STEP",
            )
        step = plan.steps[0]
        if step.action not in self._allowed_actions:
            return PlanExecutionResult(
                status="rejected",
                action=step.action,
                error_code="ACTIVE_ACTION_NOT_ALLOWED",
            )
        handler = self._handlers.get(step.action)
        if handler is None:
            return PlanExecutionResult(
                status="rejected",
                action=step.action,
                error_code="ACTIVE_ACTION_HANDLER_MISSING",
            )
        try:
            envelope = await handler(step, plan, context)
        except Exception:
            return PlanExecutionResult(
                status="execution_error",
                action=step.action,
                error_code="ACTIVE_ACTION_EXECUTION_ERROR",
            )
        if not isinstance(envelope, ResponseEnvelope):
            return PlanExecutionResult(
                status="execution_error",
                action=step.action,
                error_code="ACTIVE_ACTION_RESPONSE_INVALID",
            )
        return PlanExecutionResult(
            status="executed",
            action=step.action,
            envelope=envelope,
        )
