"""把现有 RoutingContext 适配成 Planner 可消费的裁剪上下文。"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from app.conversation.runtime_memory import PlannerMemoryView
from app.observability.trace import current_trace_id
from app.orchestrator.planning.actions import planner_capabilities
from app.orchestrator.planning.models import RecipeRef, TurnContext
from app.orchestrator.turn.runtime_models import DerivedTurnContext
from app.orchestrator.routing_models import RoutingContext

MemoryContextState = Literal["not_provided", "provided", "unavailable"]


def _clean_text(value: object, *, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _recipe_ref(value: Any, *, position: int | None = None) -> RecipeRef | None:
    if not isinstance(value, dict):
        return None
    recipe_id = _clean_text(
        value.get("cookId") or value.get("recipe_id") or value.get("id"),
        limit=128,
    )
    name = _clean_text(value.get("name") or value.get("topic"), limit=80)
    if not recipe_id and not name:
        return None
    return RecipeRef(recipe_id=recipe_id, name=name, position=position)


def _clean_constraints(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, list[str]] = {}
    for raw_key, raw_values in list(value.items())[:12]:
        key = _clean_text(raw_key, limit=40)
        if not key or not isinstance(raw_values, list):
            continue
        values = [
            _clean_text(item, limit=80)
            for item in raw_values[:8]
            if _clean_text(item, limit=80)
        ]
        if values:
            cleaned[key] = values
    return cleaned


def _memory_context_values(
    memory_context: Any,
) -> tuple[list[str], dict[str, list[str]], list[str], int]:
    if isinstance(memory_context, PlannerMemoryView):
        if not memory_context.policy_resolved:
            return [], {}, [], 0
        return (
            list(memory_context.recent_user_turns[-4:]),
            _clean_constraints(
                memory_context.effective_preferences.to_context_dict()
            ),
            [
                item.value
                for item in memory_context.effective_temporal_dietary_constraints[:8]
                if item.value
            ],
            max(0, min(20, memory_context.previous_candidate_count)),
        )
    if not isinstance(memory_context, dict):
        return [], {}, [], 0
    recent = [
        _clean_text(item, limit=180)
        for item in list(memory_context.get("recent_user_turns") or [])[-4:]
        if _clean_text(item, limit=180)
    ]
    stable = _clean_constraints(memory_context.get("preferences") or {})
    temporal = [
        _clean_text(item.get("value"), limit=80)
        for item in list(
            memory_context.get("temporal_dietary_constraints") or []
        )[:8]
        if isinstance(item, dict) and _clean_text(item.get("value"), limit=80)
    ]
    try:
        previous_candidate_count = max(
            0,
            min(20, int(memory_context.get("previous_candidate_count") or 0)),
        )
    except (TypeError, ValueError):
        previous_candidate_count = 0
    return recent, stable, temporal, previous_candidate_count


def _memory_context_state(memory_context: Any) -> MemoryContextState:
    """区分未加载、已加载为空和加载失败，避免用 truthy 代替来源语义。"""
    if isinstance(memory_context, PlannerMemoryView):
        if not memory_context.policy_resolved:
            return "unavailable"
        if memory_context.load_state.long_term_failed:
            return "unavailable"
        return "provided"
    if not isinstance(memory_context, dict):
        return "not_provided"
    status = _clean_text(
        memory_context.get("long_term_memory_status"),
        limit=24,
    ).lower()
    if status in {"unavailable", "invalid"}:
        return "unavailable"
    return "provided"


def build_turn_context(
    utterance: str,
    routing_context: RoutingContext,
    *,
    memory_context: PlannerMemoryView | dict[str, Any] | None = None,
    trace_id: str | None = None,
    derived_context: DerivedTurnContext | None = None,
) -> TurnContext:
    """构造只读 Planner 输入；不会访问存储、工具或设备。"""
    summary = routing_context.classifier_summary()
    (
        memory_recent,
        memory_stable,
        temporal_constraints,
        previous_candidate_count,
    ) = _memory_context_values(memory_context)
    memory_state = _memory_context_state(memory_context)
    if memory_state == "not_provided":
        # 未提供运行时记忆的调用只使用本轮 RoutingContext。
        recent_turns = [
            _clean_text(item, limit=180)
            for item in routing_context.recent_user_turns[-4:]
            if _clean_text(item, limit=180)
        ]
        stable_constraints = _clean_constraints(
            routing_context.stable_constraints
        )
    elif memory_state == "unavailable":
        # 后端不可用不是“没有资料”，更不能回退到可能陈旧的旧上下文。
        # 若短期快照仍提供了本轮可信值，则保留这些显式值。
        recent_turns = memory_recent
        stable_constraints = memory_stable
    else:
        # 已提供的空列表/空字典是权威的 loaded-empty，而不是回退信号。
        recent_turns = memory_recent
        stable_constraints = memory_stable

    candidates = [
        ref
        for position, item in enumerate(routing_context.latest_candidates[:10], 1)
        if (ref := _recipe_ref(item, position=position)) is not None
    ]
    focus = _recipe_ref(routing_context.latest_focus)
    selected = _recipe_ref(routing_context.selected_recipe)

    active_constraints = _clean_constraints(routing_context.active_constraints)
    if temporal_constraints:
        active_constraints["temporal_dietary_constraints"] = temporal_constraints

    return TurnContext(
        trace_id=(
            _clean_text(trace_id or current_trace_id(), limit=64)
            or uuid.uuid4().hex
        ),
        utterance=_clean_text(utterance, limit=2_000),
        channel=_clean_text(routing_context.channel or "unknown", limit=24).lower(),
        message_type=_clean_text(
            routing_context.message_type or "text",
            limit=24,
        ).lower(),
        pending_action=summary.get("pending_action"),
        pending_device_start=bool(summary.get("pending_device_start")),
        device_choice_required=bool(summary.get("device_choice_required")),
        active_cooking=bool(summary.get("active_cooking")),
        pending_clarification=summary.get("pending_clarification"),
        latest_candidates=candidates,
        previous_candidate_count=previous_candidate_count,
        latest_focus=focus,
        selected_recipe=selected,
        current_task=_clean_text(summary.get("current_task"), limit=80) or None,
        menu_task_active=summary.get("current_task") == "menu_plan",
        active_constraints=active_constraints,
        stable_constraints=stable_constraints,
        recent_user_turns=recent_turns,
        capabilities=planner_capabilities(),
        derived_context=derived_context,
    )
