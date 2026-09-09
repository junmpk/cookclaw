"""TurnPlan 的确定性校验器。

Planner confidence 不参与授权；任何未知动作、越界参数或高风险步骤均拒绝。
"""

from __future__ import annotations

from typing import Any

from app.orchestrator.planning.actions import ACTION_REGISTRY
from app.orchestrator.planning.models import (
    PlanValidationIssue,
    PlanValidationResult,
    TurnContext,
    TurnPlan,
)


_RISK_RANK = {"low": 0, "medium": 1, "high": 2}
_FORBIDDEN_ARG_KEYS = {
    "action_id",
    "authorization",
    "command",
    "command_id",
    "confirmation_token",
    "device_id",
    "idempotency_key",
    "memory",
    "msg_id",
    "password",
    "receiver_id",
    "receiverid",
    "token",
}
_ALLOWED_EVIDENCE_REFS = {
    "utterance",
    "recent_user_turns",
    "active_constraints",
    "stable_constraints",
    "pending_action",
    "pending_device_start",
    "active_cooking",
    "latest_focus",
    "selected_recipe",
    "previous_candidates",
    "menu_task",
    "derived_facts",
}


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key or "").strip().lower()
            if normalized in _FORBIDDEN_ARG_KEYS:
                return True
            if _contains_forbidden_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _arg_value_valid(key: str, value: Any) -> bool:
    if key == "position":
        return isinstance(value, int) and not isinstance(value, bool)
    if key in {"query", "slot", "recipe_id"}:
        if not isinstance(value, str) or not value.strip():
            return False
        limit = {"query": 500, "slot": 80, "recipe_id": 128}[key]
        return len(value.strip()) <= limit
    return False


def _evidence_ref_valid(value: str, known_recipe_ids: set[str]) -> bool:
    if value in _ALLOWED_EVIDENCE_REFS:
        return True
    if value.startswith("candidate:"):
        recipe_id = value.split(":", 1)[1].strip()
        return bool(recipe_id and recipe_id in known_recipe_ids)
    return False


def _issue(
    issues: list[PlanValidationIssue],
    code: str,
    *,
    step_index: int | None = None,
) -> None:
    candidate = PlanValidationIssue(code=code, step_index=step_index)
    if candidate not in issues:
        issues.append(candidate)


class PlanValidator:
    """校验一个 plan 是否具备未来进入 Executor 的最低条件。"""

    def validate(
        self,
        plan: TurnPlan,
        context: TurnContext,
    ) -> PlanValidationResult:
        issues: list[PlanValidationIssue] = []
        known_recipe_ids = context.known_recipe_ids()
        max_step_risk = "low"
        seen_actions: set[str] = set()

        if plan.requires_deterministic_handler:
            if plan.risk != "high":
                _issue(issues, "DETERMINISTIC_HANDOFF_MUST_BE_HIGH_RISK")
            if plan.steps:
                _issue(issues, "DETERMINISTIC_HANDOFF_MUST_NOT_HAVE_STEPS")
            if plan.reason_code != "DEFER_TO_DETERMINISTIC_HANDLER":
                _issue(issues, "DETERMINISTIC_HANDOFF_REASON_INVALID")
        elif plan.risk == "high":
            _issue(issues, "HIGH_RISK_REQUIRES_DETERMINISTIC_HANDOFF")
        elif not plan.steps:
            _issue(issues, "NON_DETERMINISTIC_PLAN_REQUIRES_STEP")

        for index, step in enumerate(plan.steps):
            spec = ACTION_REGISTRY.get(step.action)
            if spec is None:
                _issue(issues, "UNKNOWN_ACTION", step_index=index)
                continue

            if step.action in seen_actions:
                _issue(issues, "DUPLICATE_ACTION", step_index=index)
            seen_actions.add(step.action)

            if _RISK_RANK[spec.risk] > _RISK_RANK[max_step_risk]:
                max_step_risk = spec.risk

            arg_keys = {str(key) for key in step.args}
            if arg_keys - set(spec.allowed_args):
                _issue(issues, "ACTION_ARGS_NOT_ALLOWED", step_index=index)
            if set(spec.required_args) - arg_keys:
                _issue(issues, "ACTION_ARGS_MISSING", step_index=index)
            for key in arg_keys & set(spec.allowed_args):
                if not _arg_value_valid(key, step.args.get(key)):
                    _issue(issues, "ACTION_ARG_INVALID", step_index=index)
            if _contains_forbidden_key(step.args):
                _issue(issues, "FORBIDDEN_SIDE_EFFECT_ARG", step_index=index)
            if not step.evidence_refs:
                _issue(issues, "MISSING_EVIDENCE_REF", step_index=index)
            elif any(
                not _evidence_ref_valid(ref, known_recipe_ids)
                for ref in step.evidence_refs
            ):
                _issue(issues, "EVIDENCE_REF_OUTSIDE_CONTEXT", step_index=index)

            if step.action == "candidate.select":
                if not context.latest_candidates:
                    _issue(issues, "CANDIDATE_CONTEXT_REQUIRED", step_index=index)
                recipe_id = str(step.args.get("recipe_id") or "").strip()
                position = step.args.get("position")
                if not recipe_id and position is None:
                    _issue(issues, "CANDIDATE_REFERENCE_REQUIRED", step_index=index)
                if recipe_id and position is not None:
                    _issue(issues, "CANDIDATE_REFERENCE_MULTIPLE", step_index=index)
                if recipe_id and recipe_id not in known_recipe_ids:
                    _issue(issues, "RECIPE_ID_OUTSIDE_CONTEXT", step_index=index)
                if position is not None:
                    if (
                        isinstance(position, bool)
                        or not isinstance(position, int)
                        or position < 1
                        or position > len(context.latest_candidates)
                    ):
                        _issue(issues, "CANDIDATE_POSITION_INVALID", step_index=index)
                    elif (
                        recipe_id
                        and context.latest_candidates[position - 1].recipe_id
                        and context.latest_candidates[position - 1].recipe_id
                        != recipe_id
                    ):
                        _issue(
                            issues,
                            "CANDIDATE_REFERENCE_CONFLICT",
                            step_index=index,
                        )

            if (
                step.action == "candidate.compare"
                and len(context.latest_candidates) < 2
            ):
                _issue(
                    issues,
                    "CANDIDATE_COMPARISON_REQUIRES_MULTIPLE",
                    step_index=index,
                )

            if (
                step.action == "candidate.restore_previous"
                and context.previous_candidate_count < 1
            ):
                _issue(
                    issues,
                    "PREVIOUS_CANDIDATE_CONTEXT_REQUIRED",
                    step_index=index,
                )

            if step.action in {"recipe.detail", "device.prepare"}:
                recipe_id = str(step.args.get("recipe_id") or "").strip()
                if recipe_id and recipe_id not in known_recipe_ids:
                    _issue(issues, "RECIPE_ID_OUTSIDE_CONTEXT", step_index=index)

            if step.action == "conversation.clarify":
                slot = str(step.args.get("slot") or "").strip()
                if not plan.missing_slots:
                    _issue(
                        issues,
                        "CLARIFY_REQUIRES_DECLARED_SLOT",
                        step_index=index,
                    )
                elif slot and slot not in plan.missing_slots:
                    _issue(issues, "CLARIFY_SLOT_MISMATCH", step_index=index)
                if plan.reply_act != "ask_one_question":
                    _issue(issues, "CLARIFY_REPLY_ACT_MISMATCH", step_index=index)

        if _RISK_RANK[plan.risk] < _RISK_RANK[max_step_risk]:
            _issue(issues, "RISK_UNDERSTATED")

        if plan.reply_act == "ask_one_question":
            if not plan.missing_slots:
                _issue(issues, "ASK_ONE_QUESTION_REQUIRES_MISSING_SLOT")
            clarify_steps = [
                step for step in plan.steps
                if step.action == "conversation.clarify"
            ]
            if len(clarify_steps) != 1:
                _issue(issues, "ASK_ONE_QUESTION_REQUIRES_CLARIFY_STEP")

        valid = not issues
        return PlanValidationResult(
            valid=valid,
            execution_allowed=(
                valid
                and not plan.requires_deterministic_handler
                and plan.risk != "high"
            ),
            issues=issues,
        )
