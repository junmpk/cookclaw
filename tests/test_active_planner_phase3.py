"""阶段 3.1 Planner active 灰度、校验和封闭 Executor 回归。"""

from __future__ import annotations

import asyncio

from app.core.config import settings
from app.orchestrator.planning.active_planner import evaluate_active_plan
from app.orchestrator.turn.context_loader import build_turn_context
from app.orchestrator.planning.models import PlannerCallResult, PlanStep, TurnPlan
from app.orchestrator.planning.plan_executor import BoundedPlanExecutor
from app.orchestrator.planning.planner_rollout import planner_rollout_decision
from app.orchestrator.turn.runtime_models import ResponseEnvelope
from app.orchestrator.routing_models import RoutingContext


def _context():
    return build_turn_context(
        "帮我找一道鸡肉菜",
        RoutingContext(channel="qq", message_type="text"),
        trace_id="phase3-active-trace",
    )


def _plan(action: str = "recipe.search") -> TurnPlan:
    return TurnPlan(
        goal="搜索真实菜谱",
        phase="search_or_recommend",
        known_facts=["用户要求搜索"],
        steps=[
            PlanStep(
                action=action,
                args={"query": "鸡肉"} if action == "recipe.search" else {},
                evidence_refs=["utterance"],
            )
        ],
        reply_act="recommend",
        risk="low",
        reason_code="ACTIVE_SEARCH",
        confidence=0.8,
    )


def test_active_rollout_requires_mode_channel_supported_media_and_user_selection(
    monkeypatch,
):
    thread_id = "qq:dm:planner-user:planner-user"
    monkeypatch.setattr(settings, "TURN_PLANNER_MODE", "off")
    assert planner_rollout_decision(thread_id).cohort == "mode_disabled"

    monkeypatch.setattr(settings, "TURN_PLANNER_MODE", "active")
    monkeypatch.setattr(settings, "TURN_PLANNER_ACTIVE_CHANNELS", "qq")
    monkeypatch.setattr(settings, "TURN_PLANNER_ACTIVE_QQ_ALLOW_FROM", "")
    monkeypatch.setattr(settings, "TURN_PLANNER_ACTIVE_ROLLOUT_PERCENT", 0)
    assert planner_rollout_decision(thread_id).cohort == "not_selected"
    assert planner_rollout_decision(thread_id, message_type="voice").cohort == (
        "message_type_excluded"
    )

    monkeypatch.setattr(
        settings,
        "TURN_PLANNER_ACTIVE_QQ_ALLOW_FROM",
        "planner-user",
    )
    selected = planner_rollout_decision(thread_id)
    assert selected.enabled is True
    assert selected.cohort == "qq_allowlist"
    image_selected = planner_rollout_decision(
        thread_id,
        message_type="image",
    )
    assert image_selected.enabled is True
    assert image_selected.cohort == "qq_allowlist"


def test_active_rollout_supports_namespaced_im_identity_allowlist(monkeypatch):
    monkeypatch.setattr(settings, "TURN_PLANNER_MODE", "active")
    monkeypatch.setattr(
        settings,
        "TURN_PLANNER_ACTIVE_CHANNELS",
        "qq,weixin,whatsapp",
    )
    monkeypatch.setattr(settings, "TURN_PLANNER_ACTIVE_QQ_ALLOW_FROM", "")
    monkeypatch.setattr(settings, "TURN_PLANNER_ACTIVE_ROLLOUT_PERCENT", 0)
    monkeypatch.setattr(
        settings,
        "TURN_PLANNER_ACTIVE_ALLOW_IDENTITIES",
        "weixin:wx-user,whatsapp:+111",
    )

    weixin = planner_rollout_decision("weixin:dm:bot-a:wx-user")
    whatsapp = planner_rollout_decision(
        "whatsapp:dm:111@s.whatsapp.net:+111"
    )
    other = planner_rollout_decision("weixin:dm:bot-a:other-user")

    assert weixin.enabled is True
    assert weixin.cohort == "identity_allowlist"
    assert whatsapp.enabled is True
    assert whatsapp.cohort == "identity_allowlist"
    assert other.enabled is False


def test_active_planner_accepts_one_valid_allowlisted_step(monkeypatch):
    class FakePlanner:
        async def plan(self, _context):
            return PlannerCallResult(
                status="success",
                plan=_plan(),
                model="fake",
            )

    monkeypatch.setattr(
        settings,
        "TURN_PLANNER_ACTIVE_ACTIONS",
        "recipe.search",
    )
    decision = asyncio.run(
        evaluate_active_plan(_context(), planner=FakePlanner())
    )

    assert decision.status == "ready"
    assert decision.execution_allowed is True
    assert decision.action == "recipe.search"
    assert decision.validation.valid is True


def test_active_planner_hands_high_risk_plan_back_to_deterministic_handler():
    handoff = TurnPlan(
        goal="停止当前设备",
        phase="running",
        steps=[],
        reply_act="report_state",
        risk="high",
        reason_code="DEFER_TO_DETERMINISTIC_HANDLER",
        confidence=1,
        requires_deterministic_handler=True,
    )

    class FakePlanner:
        async def plan(self, _context):
            return PlannerCallResult(
                status="success",
                plan=handoff,
                model="fake",
            )

    decision = asyncio.run(
        evaluate_active_plan(_context(), planner=FakePlanner())
    )

    assert decision.status == "deterministic_handoff"
    assert decision.execution_allowed is False
    assert decision.action is None


def test_active_planner_rejects_multiple_steps_even_when_validator_allows_them(
    monkeypatch,
):
    multi = _plan().model_copy(
        update={
            "steps": [
                PlanStep(
                    action="recipe.search",
                    args={"query": "鸡肉"},
                    evidence_refs=["utterance"],
                ),
                PlanStep(
                    action="conversation.respond",
                    evidence_refs=["utterance"],
                ),
            ]
        }
    )

    class FakePlanner:
        async def plan(self, _context):
            return PlannerCallResult(
                status="success",
                plan=multi,
                model="fake",
            )

    monkeypatch.setattr(
        settings,
        "TURN_PLANNER_ACTIVE_ACTIONS",
        "recipe.search,conversation.respond",
    )
    decision = asyncio.run(
        evaluate_active_plan(_context(), planner=FakePlanner())
    )

    assert decision.status == "fallback"
    assert decision.fallback_code == "PLANNER_ACTIVE_STEP_COUNT_UNSUPPORTED"


def test_bounded_executor_calls_only_registered_allowlisted_handler_once():
    calls: list[str] = []

    async def search_handler(step, plan, context):
        calls.append(step.action)
        assert plan.reason_code == "ACTIVE_SEARCH"
        assert context.trace_id == "phase3-active-trace"
        return ResponseEnvelope(message="真实搜索结果")

    executor = BoundedPlanExecutor(
        {"recipe.search": search_handler},
        allowed_actions={"recipe.search"},
    )
    result = asyncio.run(executor.execute(_plan(), _context()))

    assert result.status == "executed"
    assert result.envelope.message == "真实搜索结果"
    assert calls == ["recipe.search"]


def test_bounded_executor_rejects_missing_or_non_allowlisted_handler():
    blocked = BoundedPlanExecutor({}, allowed_actions={"recipe.search"})
    missing = asyncio.run(blocked.execute(_plan(), _context()))
    denied = asyncio.run(
        BoundedPlanExecutor(
            {"recipe.search": lambda *_args: None},
            allowed_actions=set(),
        ).execute(_plan(), _context())
    )

    assert missing.error_code == "ACTIVE_ACTION_HANDLER_MISSING"
    assert denied.error_code == "ACTIVE_ACTION_NOT_ALLOWED"
