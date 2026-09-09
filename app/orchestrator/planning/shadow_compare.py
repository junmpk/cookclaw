"""Planner Shadow 执行、脱敏记录和 Legacy 差异比较。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from contextvars import Context
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.orchestrator.planning.models import (
    LegacyRouteSnapshot,
    PlanValidationResult,
    PlannerCallResult,
    ShadowComparison,
    TurnContext,
    TurnPlan,
)
from app.orchestrator.planning.plan_validator import PlanValidator
from app.orchestrator.planning.planner import TurnPlanner


logger = logging.getLogger(__name__)
_SHADOW_SCHEMA = "turn_planner_shadow_v1"
_shadow_file_lock = threading.Lock()
_shadow_tasks: set[asyncio.Task] = set()
_planner_singleton: TurnPlanner | None = None
_validator_singleton: PlanValidator | None = None
_invalid_mode_warning_emitted = False


_LEGACY_ACTION_MAP = {
    "casual": "conversation.respond",
    "cooking_qa": "conversation.respond",
    "unknown": "conversation.respond",
    "clarify": "conversation.clarify",
    "ambiguous": "conversation.clarify",
    "show_recipe_detail": "recipe.detail",
    "choose_candidate": "candidate.select",
    "compare_candidates": "candidate.compare",
    "device_status": "device.status",
    "progress": "device.status",
    "prepare_device": "device.prepare",
    "web_search": "web.search",
    "web_reference": "conversation.respond",
}
_DETERMINISTIC_LEGACY_ACTIONS = {
    "cancel_pending",
    "confirm_start",
    "memory_correct",
    "safety_block",
    "select_device",
    "stop",
}


def planner_shadow_enabled() -> bool:
    """只有 shadow 模式执行旁路；active 由主链独立处理。"""
    global _invalid_mode_warning_emitted
    raw = str(getattr(settings, "TURN_PLANNER_MODE", "off") or "off")
    normalized = raw.strip().lower()
    if (
        normalized not in {"off", "shadow", "active"}
        and not _invalid_mode_warning_emitted
    ):
        _invalid_mode_warning_emitted = True
        logger.error(
            "invalid TURN_PLANNER_MODE; planner remains fail-closed"
        )
    return normalized == "shadow"


def _get_planner() -> TurnPlanner:
    global _planner_singleton
    if _planner_singleton is None:
        _planner_singleton = TurnPlanner()
    return _planner_singleton


def _get_validator() -> PlanValidator:
    global _validator_singleton
    if _validator_singleton is None:
        _validator_singleton = PlanValidator()
    return _validator_singleton


def legacy_route_snapshot(outcome: Any) -> LegacyRouteSnapshot:
    decision = getattr(outcome, "decision", None)
    intent = getattr(outcome, "intent", None)
    raw_risk = str(getattr(decision, "risk", "low") or "low")
    risk = raw_risk if raw_risk in {"low", "medium", "high"} else "low"
    return LegacyRouteSnapshot(
        outcome=str(getattr(outcome, "kind", "unknown") or "unknown")[:80],
        action=str(
            getattr(decision, "action", None)
            or getattr(intent, "action", None)
            or "unknown"
        )[:80],
        category=str(
            getattr(decision, "category", None)
            or getattr(getattr(intent, "category", None), "value", None)
            or getattr(intent, "category", None)
            or "unknown"
        )[:80],
        reason_code=str(
            getattr(decision, "reason_code", None)
            or getattr(intent, "reason_code", None)
            or "UNKNOWN"
        )[:80],
        risk=risk,
        needs_clarification=bool(
            getattr(decision, "needs_clarification", False)
            or getattr(intent, "needs_clarification", False)
            or getattr(outcome, "kind", "") in {"clarify", "ambiguous"}
        ),
    )


def expected_planner_action(legacy: LegacyRouteSnapshot) -> str | None:
    if legacy.action in _DETERMINISTIC_LEGACY_ACTIONS:
        return "deterministic_handler"
    if legacy.action in {"new_search", "refine_search"}:
        return (
            "recipe.recommend"
            if legacy.category == "recipe_recommend"
            else "recipe.search"
        )
    mapped = _LEGACY_ACTION_MAP.get(legacy.action)
    if mapped:
        return mapped
    return {
        "detail": "recipe.detail",
        "menu_plan": "menu.plan",
        "search": (
            "recipe.recommend"
            if legacy.category == "recipe_recommend"
            else "recipe.search"
        ),
        "web_search": "web.search",
        "greeting": "conversation.respond",
        "device_knowledge": "conversation.respond",
        "safety_block": "deterministic_handler",
        "clarify": "conversation.clarify",
        "ambiguous": "conversation.clarify",
        "agent": "conversation.respond",
    }.get(legacy.outcome)


def compare_shadow_plan(
    legacy: LegacyRouteSnapshot,
    plan: TurnPlan,
    validation: PlanValidationResult,
) -> ShadowComparison:
    expected = expected_planner_action(legacy)
    planner_action = plan.steps[0].action if plan.steps else None
    deterministic_expected = expected == "deterministic_handler"
    deterministic_match = (
        plan.requires_deterministic_handler
        if deterministic_expected
        else not plan.requires_deterministic_handler
    )
    action_match = (
        deterministic_match
        if deterministic_expected
        else (planner_action == expected if expected is not None else None)
    )
    planner_clarifies = (
        plan.reply_act == "ask_one_question"
        or any(step.action == "conversation.clarify" for step in plan.steps)
    )
    clarification_match = planner_clarifies == legacy.needs_clarification
    mismatch_codes: list[str] = []
    if action_match is False:
        mismatch_codes.append("ACTION_MISMATCH")
    if plan.risk != legacy.risk:
        mismatch_codes.append("RISK_MISMATCH")
    if clarification_match is False:
        mismatch_codes.append("CLARIFICATION_MISMATCH")
    if deterministic_match is False:
        mismatch_codes.append("DETERMINISTIC_HANDOFF_MISMATCH")
    if not validation.valid:
        mismatch_codes.append("PLAN_INVALID")
    return ShadowComparison(
        expected_planner_action=expected,
        planner_action=planner_action,
        action_match=action_match,
        risk_match=plan.risk == legacy.risk,
        clarification_match=clarification_match,
        deterministic_handoff_match=deterministic_match,
        mismatch_codes=mismatch_codes,
    )


def _plan_log_summary(plan: TurnPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "schema_version": plan.schema_version,
        "phase": plan.phase,
        "step_actions": [step.action for step in plan.steps],
        "reply_act": plan.reply_act,
        "risk": plan.risk,
        "confidence": round(plan.confidence, 4),
        "requires_deterministic_handler": plan.requires_deterministic_handler,
        "known_fact_count": len(plan.known_facts),
        "missing_slot_count": len(plan.missing_slots),
    }


def build_shadow_record(
    *,
    context: TurnContext,
    legacy: LegacyRouteSnapshot,
    planner_result: PlannerCallResult,
    validation: PlanValidationResult | None,
    comparison: ShadowComparison | None,
) -> dict[str, Any]:
    """构造不含用户原话、候选详情、参数和设备标识的记录。"""
    return {
        "schema_version": _SHADOW_SCHEMA,
        "recorded_at": round(time.time(), 3),
        "trace_id": context.trace_id,
        "planner_mode": "shadow",
        "status": planner_result.status,
        "context_ref": context.context_ref(),
        "legacy": legacy.model_dump(),
        "planner": _plan_log_summary(planner_result.plan),
        "validation": (
            {
                "valid": validation.valid,
                "execution_allowed": validation.execution_allowed,
                "issue_codes": validation.issue_codes,
            }
            if validation is not None
            else None
        ),
        "comparison": comparison.model_dump() if comparison is not None else None,
        "model": {
            "name": planner_result.model,
            "duration_ms": round(planner_result.duration_ms, 2),
            "input_tokens": planner_result.input_tokens,
            "output_tokens": planner_result.output_tokens,
            "error_code": planner_result.error_code,
        },
    }


def build_shadow_bypass_record(
    *,
    context: TurnContext,
    bypass_reason: str,
    legacy_action: str = "early_return",
    category: str = "conversation",
    risk: str = "low",
    needs_clarification: bool = False,
) -> dict[str, Any]:
    """记录未进入 Legacy router 的前置分支，不调用 Planner。"""
    normalized_reason = re.sub(
        r"[^A-Z0-9_]+",
        "_",
        str(bypass_reason or "").strip().upper(),
    ).strip("_")[:80] or "EARLY_RETURN"
    normalized_risk = risk if risk in {"low", "medium", "high"} else "low"
    legacy = LegacyRouteSnapshot(
        outcome="early_return",
        action=str(legacy_action or "early_return")[:80],
        category=str(category or "conversation")[:80],
        reason_code=normalized_reason,
        risk=normalized_risk,
        needs_clarification=bool(needs_clarification),
    )
    return {
        "schema_version": _SHADOW_SCHEMA,
        "recorded_at": round(time.time(), 3),
        "trace_id": context.trace_id,
        "planner_mode": "shadow",
        "status": "bypassed",
        "bypass_reason": normalized_reason,
        "context_ref": context.context_ref(),
        "legacy": legacy.model_dump(),
        "planner": None,
        "validation": None,
        "comparison": None,
        "model": {
            "name": "",
            "duration_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "error_code": None,
        },
    }


def _write_shadow_jsonl(record: dict[str, Any]) -> None:
    path_value = str(
        os.getenv(
            "TURN_PLANNER_SHADOW_JSONL_PATH",
            getattr(settings, "TURN_PLANNER_SHADOW_JSONL_PATH", ""),
        )
        or ""
    ).strip()
    if not path_value:
        return
    path = Path(path_value).expanduser()
    try:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with _shadow_file_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:
        logger.warning(
            "turn planner shadow JSONL write failed: error_type=%s",
            type(exc).__name__,
        )


def record_shadow_result(record: dict[str, Any]) -> None:
    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    logger.info("turn_planner_shadow %s", encoded)
    _write_shadow_jsonl(record)


def record_shadow_bypass(
    context: TurnContext,
    bypass_reason: str,
    *,
    legacy_action: str = "early_return",
    category: str = "conversation",
    risk: str = "low",
    needs_clarification: bool = False,
) -> dict[str, Any]:
    record = build_shadow_bypass_record(
        context=context,
        bypass_reason=bypass_reason,
        legacy_action=legacy_action,
        category=category,
        risk=risk,
        needs_clarification=needs_clarification,
    )
    record_shadow_result(record)
    return record


async def evaluate_shadow(
    context: TurnContext,
    legacy_outcome: Any,
    *,
    planner: TurnPlanner | None = None,
    validator: PlanValidator | None = None,
) -> dict[str, Any]:
    legacy = legacy_route_snapshot(legacy_outcome)
    planner_result = await (planner or _get_planner()).plan(context)
    validation = None
    comparison = None
    if planner_result.plan is not None:
        validation = (validator or _get_validator()).validate(
            planner_result.plan,
            context,
        )
        comparison = compare_shadow_plan(
            legacy,
            planner_result.plan,
            validation,
        )
    record = build_shadow_record(
        context=context,
        legacy=legacy,
        planner_result=planner_result,
        validation=validation,
        comparison=comparison,
    )
    record_shadow_result(record)
    return record


def _consume_shadow_task(task: asyncio.Task) -> None:
    _shadow_tasks.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning(
            "turn planner shadow task failed: error_type=%s",
            type(exc).__name__,
        )


def schedule_shadow_evaluation(
    context: TurnContext,
    legacy_outcome: Any,
) -> asyncio.Task | None:
    """后台运行 Shadow；不会等待、执行 plan 或修改当前 TurnTrace。"""
    if not planner_shadow_enabled():
        return None

    try:
        limit = max(
            1,
            int(getattr(settings, "TURN_PLANNER_SHADOW_MAX_INFLIGHT", 4)),
        )
    except (TypeError, ValueError):
        limit = 4
    active_tasks = [task for task in list(_shadow_tasks) if not task.done()]
    if len(active_tasks) >= limit:
        logger.warning(
            "turn planner shadow skipped: reason=capacity limit=%s trace_id=%s",
            limit,
            context.trace_id,
        )
        return None

    coroutine = evaluate_shadow(context, legacy_outcome)
    try:
        task = asyncio.create_task(
            coroutine,
            name=f"turn-planner-shadow-{context.trace_id[:12]}",
            context=Context(),
        )
    except Exception:
        coroutine.close()
        raise
    _shadow_tasks.add(task)
    task.add_done_callback(_consume_shadow_task)
    return task


async def drain_shadow_tasks(
    *,
    timeout_seconds: float = 2.0,
    cancel_pending: bool = True,
) -> None:
    """测试与进程关闭时收敛后台任务，超时后只取消 Planner 模型调用。"""
    tasks = [task for task in list(_shadow_tasks) if not task.done()]
    if not tasks:
        return
    _done, pending = await asyncio.wait(
        tasks,
        timeout=max(0.0, float(timeout_seconds)),
    )
    if cancel_pending:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
