"""Planner active 模式的调用、校验和脱敏审计。"""

from __future__ import annotations

from dataclasses import dataclass

from app.observability.trace import current_trace, record_domain_result
from app.orchestrator.planning.models import (
    PlanValidationResult,
    PlannerCallResult,
    TurnContext,
    TurnPlan,
)
from app.orchestrator.planning.plan_validator import PlanValidator
from app.orchestrator.planning.planner import TurnPlanner
from app.orchestrator.planning.planner_rollout import planner_active_actions


@dataclass(frozen=True, slots=True)
class ActivePlanDecision:
    status: str
    context: TurnContext
    planner_result: PlannerCallResult
    plan: TurnPlan | None = None
    validation: PlanValidationResult | None = None
    action: str | None = None
    fallback_code: str | None = None

    @property
    def execution_allowed(self) -> bool:
        return self.status == "ready" and self.action is not None


def _observe(decision: ActivePlanDecision) -> None:
    result = decision.planner_result
    plan = decision.plan
    validation = decision.validation
    success = decision.status in {"ready", "deterministic_handoff"}
    record_domain_result(
        "turn_planner",
        duration_ms=result.duration_ms,
        success=success,
        result_code=(
            decision.fallback_code
            or (plan.reason_code if plan is not None else result.error_code)
            or "PLANNER_ACTIVE_UNKNOWN"
        ),
        operation="active_plan",
        risk=(plan.risk if plan is not None else "low"),
    )
    trace = current_trace()
    if trace is not None:
        trace.add_event(
            "planner_active",
            status=decision.status,
            action=decision.action,
            phase=(plan.phase if plan is not None else None),
            reply_act=(plan.reply_act if plan is not None else None),
            plan_reason_code=(plan.reason_code if plan is not None else None),
            valid=(validation.valid if validation is not None else None),
            execution_allowed=(
                validation.execution_allowed
                if validation is not None
                else None
            ),
            fallback_code=decision.fallback_code,
            issue_count=(len(validation.issues) if validation is not None else 0),
        )


async def evaluate_active_plan(
    context: TurnContext,
    *,
    planner: TurnPlanner | None = None,
    validator: PlanValidator | None = None,
) -> ActivePlanDecision:
    """运行一次 Planner；任何异常、越界或多步计划都返回可回退结果。"""
    planner_result = await (planner or TurnPlanner()).plan(context)
    if planner_result.status != "success" or planner_result.plan is None:
        decision = ActivePlanDecision(
            status="fallback",
            context=context,
            planner_result=planner_result,
            fallback_code=(
                planner_result.error_code or "PLANNER_ACTIVE_CALL_FAILED"
            ),
        )
        _observe(decision)
        return decision

    plan = planner_result.plan
    validation = (validator or PlanValidator()).validate(plan, context)
    if not validation.valid:
        decision = ActivePlanDecision(
            status="fallback",
            context=context,
            planner_result=planner_result,
            plan=plan,
            validation=validation,
            fallback_code="PLANNER_ACTIVE_PLAN_INVALID",
        )
        _observe(decision)
        return decision
    if plan.requires_deterministic_handler:
        decision = ActivePlanDecision(
            status="deterministic_handoff",
            context=context,
            planner_result=planner_result,
            plan=plan,
            validation=validation,
            fallback_code="PLANNER_DETERMINISTIC_HANDOFF",
        )
        _observe(decision)
        return decision
    if not validation.execution_allowed:
        decision = ActivePlanDecision(
            status="fallback",
            context=context,
            planner_result=planner_result,
            plan=plan,
            validation=validation,
            fallback_code="PLANNER_ACTIVE_EXECUTION_DENIED",
        )
        _observe(decision)
        return decision
    if len(plan.steps) != 1:
        decision = ActivePlanDecision(
            status="fallback",
            context=context,
            planner_result=planner_result,
            plan=plan,
            validation=validation,
            fallback_code="PLANNER_ACTIVE_STEP_COUNT_UNSUPPORTED",
        )
        _observe(decision)
        return decision

    action = plan.steps[0].action
    if action not in planner_active_actions():
        decision = ActivePlanDecision(
            status="fallback",
            context=context,
            planner_result=planner_result,
            plan=plan,
            validation=validation,
            action=action,
            fallback_code="PLANNER_ACTIVE_ACTION_NOT_ALLOWED",
        )
        _observe(decision)
        return decision

    decision = ActivePlanDecision(
        status="ready",
        context=context,
        planner_result=planner_result,
        plan=plan,
        validation=validation,
        action=action,
    )
    _observe(decision)
    return decision
