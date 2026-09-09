"""Planner 上下文对运行时记忆加载状态的语义回归测试。"""

from __future__ import annotations

import pytest

from app.conversation.models import ConversationMemory
from app.conversation.runtime_memory import (
    LongTermMemorySnapshot,
    RuntimeMemorySnapshot,
    build_planner_memory_view,
)
from app.orchestrator.turn.context_loader import build_turn_context
from app.orchestrator.routing_models import RoutingContext


def _stale_routing_context() -> RoutingContext:
    return RoutingContext(
        recent_user_turns=["旧 RoutingContext 轮次"],
        stable_constraints={"allergens": ["旧过敏原"]},
        active_constraints={"taste": ["本轮少辣"]},
        latest_candidates=[{"cookId": "r1", "name": "真实候选"}],
        channel="qq",
    )


def test_missing_memory_context_preserves_legacy_routing_fallback():
    context = build_turn_context(
        "继续",
        _stale_routing_context(),
        memory_context=None,
        trace_id="memory-none",
    )

    assert context.recent_user_turns == ["旧 RoutingContext 轮次"]
    assert context.stable_constraints == {"allergens": ["旧过敏原"]}
    assert context.active_constraints == {"taste": ["本轮少辣"]}
    assert context.latest_candidates[0].recipe_id == "r1"


@pytest.mark.parametrize("status", ["loaded", "empty"])
def test_provided_loaded_empty_memory_does_not_revive_stale_routing_values(
    status: str,
):
    context = build_turn_context(
        "继续",
        _stale_routing_context(),
        memory_context={
            "long_term_memory_status": status,
            "recent_user_turns": [],
            "preferences": {},
        },
        trace_id=f"memory-{status}",
    )

    assert context.recent_user_turns == []
    assert context.stable_constraints == {}
    # 与记忆来源无关的 Planner 业务状态保持原行为。
    assert context.active_constraints == {"taste": ["本轮少辣"]}
    assert context.latest_candidates[0].recipe_id == "r1"


@pytest.mark.parametrize("status", ["unavailable", "invalid"])
def test_unavailable_memory_never_falls_back_to_stale_routing_values(
    status: str,
):
    context = build_turn_context(
        "继续",
        _stale_routing_context(),
        memory_context={"long_term_memory_status": status},
        trace_id=f"memory-{status}",
    )

    assert context.recent_user_turns == []
    assert context.stable_constraints == {}


def test_unavailable_long_term_memory_keeps_explicit_short_term_snapshot_values():
    context = build_turn_context(
        "这次没有其他忌口",
        _stale_routing_context(),
        memory_context={
            "long_term_memory_status": "unavailable",
            "recent_user_turns": ["本轮可信短期内容"],
            "preferences": {"allergens": ["本轮明确花生"]},
        },
        trace_id="memory-unavailable-with-short-term",
    )

    assert context.recent_user_turns == ["本轮可信短期内容"]
    assert context.stable_constraints == {"allergens": ["本轮明确花生"]}
    assert "旧过敏原" not in repr(context.stable_constraints)


def test_typed_loaded_empty_view_does_not_revive_stale_routing_values():
    memory = ConversationMemory(
        thread_id="qq:dm:typed-empty:typed-empty",
        channel="qq",
        user_id="typed-empty",
    )
    snapshot = RuntimeMemorySnapshot(
        thread_id=memory.thread_id,
        channel="qq",
        user_id="typed-empty",
        short_term=memory,
        short_term_status="new",
        long_term=LongTermMemorySnapshot(status="empty"),
    )
    view = build_planner_memory_view(
        snapshot,
        resolved_context={
            "preferences": {},
            "recent_user_turns": [],
            "temporal_dietary_constraints": [],
        },
    )

    context = build_turn_context(
        "继续",
        _stale_routing_context(),
        memory_context=view,
        trace_id="typed-memory-empty",
    )

    assert context.recent_user_turns == []
    assert context.stable_constraints == {}
