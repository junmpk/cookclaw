import asyncio
import json

from app.orchestrator.intent import IntentCategory, IntentResult
from app.orchestrator.routing_models import RoutingContext
from app.orchestrator.routing_policy import decide_exact_route
import app.orchestrator.intent as intent_module
import app.orchestrator.router as router


def test_routing_context_only_exposes_compact_summary():
    context = RoutingContext(
        pending_action={"kind": "select_recipe", "payload": {"secret": "not-exposed"}},
        latest_candidates=[
            {"cookId": "1", "name": "番茄炒蛋", "recipe_detail": {"steps": ["hidden"]}},
        ],
        recent_user_turns=["第一轮", "第二轮"],
        stable_constraints={"allergens": ["花生"]},
        channel="qq",
    )

    summary = context.classifier_summary()

    assert summary["pending_action"] == "select_recipe"
    assert summary["candidate_count"] == 1
    assert summary["candidate_names"] == ["番茄炒蛋"]
    assert "payload" not in summary
    assert "recipe_detail" not in json.dumps(summary, ensure_ascii=False)


def test_state_rule_chooses_candidate_without_classifier():
    context = RoutingContext(
        latest_candidates=[
            {"cookId": "1", "name": "番茄炒蛋"},
            {"cookId": "2", "name": "红烧肉"},
        ],
    )

    decision = decide_exact_route("第二个", context)

    assert decision is not None
    assert decision.action == "choose_candidate"
    assert decision.source == "state_rule"
    assert decision.reason_code == "LATEST_CANDIDATE_ORDINAL"


def test_same_stop_text_depends_on_active_state():
    active = decide_exact_route(
        "停止",
        RoutingContext(active_cooking={"cookId": "1", "device_id": "office"}),
    )
    idle = decide_exact_route("停止", RoutingContext())

    assert active.action == "stop"
    assert active.reason_code == "ACTIVE_TASK_STOP_EXACT"
    assert idle.action == "ambiguous"
    assert idle.reason_code == "STOP_WITHOUT_ACTIVE_TASK"
    assert idle.needs_clarification is True


def test_no_state_confirm_and_continue_are_rejected_before_classifier():
    confirm = decide_exact_route("确认开始", RoutingContext())
    continuation = decide_exact_route("继续", RoutingContext())

    assert confirm is not None
    assert confirm.action == "ambiguous"
    assert confirm.reason_code == "AFFIRMATIVE_WITHOUT_PENDING_ACTION"
    assert confirm.needs_clarification is True
    assert continuation is not None
    assert continuation.action == "ambiguous"
    assert continuation.reason_code == "CONTINUE_WITHOUT_CONTEXT"
    assert continuation.needs_clarification is True


def test_route_rejects_ordinal_without_candidates_before_classifier(monkeypatch):
    async def must_not_classify(*_args, **_kwargs):
        raise AssertionError("无候选序号应由状态规则拒判")

    monkeypatch.setattr(router, "classify_intent", must_not_classify)
    outcome = asyncio.run(router.route_fast_path(
        "第二个",
        routing_context=RoutingContext(channel="qq"),
    ))

    assert outcome.kind == "ambiguous"
    assert outcome.decision.source == "ambiguous"
    assert outcome.decision.reason_code == "ORDINAL_WITHOUT_CANDIDATES"
    assert outcome.decision.classifier_called is False
    assert "先搜索菜谱" in outcome.direct_message


def test_route_rejects_no_state_continue_before_classifier(monkeypatch):
    async def must_not_classify(*_args, **_kwargs):
        raise AssertionError("无上下文继续应由状态规则拒判")

    monkeypatch.setattr(router, "classify_intent", must_not_classify)
    outcome = asyncio.run(router.route_fast_path(
        "继续",
        routing_context=RoutingContext(channel="qq"),
    ))

    assert outcome.kind == "ambiguous"
    assert outcome.decision.reason_code == "CONTINUE_WITHOUT_CONTEXT"
    assert outcome.decision.classifier_called is False
    assert "没有可继续的步骤" in outcome.direct_message


def test_classifier_receives_compact_context(monkeypatch):
    captured = {}

    class FakeResponse:
        content = json.dumps({
            "r": True,
            "c": "recipe_execute",
            "a": "choose_candidate",
            "k": ["第二个"],
            "slots": {"ordinal": 2},
        }, ensure_ascii=False)
        usage_metadata = None
        response_metadata = None

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["input"] = messages[-1].content
            return FakeResponse()

    monkeypatch.setattr(intent_module, "_intent_llm", FakeLLM())
    context = RoutingContext(
        pending_action={"kind": "select_recipe"},
        latest_candidates=[{"cookId": "1", "name": "番茄炒蛋"}],
        recent_user_turns=["推荐两个家常菜"],
        channel="web",
    )

    result = asyncio.run(intent_module.classify_intent("还是这个吧", context))
    payload = json.loads(captured["input"])

    assert payload["utterance"] == "还是这个吧"
    assert payload["context"]["pending_action"] == "select_recipe"
    assert payload["context"]["candidate_count"] == 1
    assert result.category is IntentCategory.recipe_execute
    assert result.action == "choose_candidate"
    assert result.confidence is None
    assert result.context_ref["candidate_count"] == 1


def test_parse_error_is_explicitly_rejected(monkeypatch):
    async def parse_error(_question, _context):
        return IntentResult(
            category=IntentCategory.unknown,
            source="parse_error",
            reason_code="CLASSIFIER_PARSE_ERROR",
            needs_clarification=True,
        )

    monkeypatch.setattr(router, "classify_intent", parse_error)
    outcome = asyncio.run(router.route_fast_path(
        "帮我弄一下",
        routing_context=RoutingContext(channel="web"),
    ))

    assert outcome.kind == "ambiguous"
    assert outcome.decision.source == "parse_error"
    assert outcome.decision.classifier_called is True
    assert outcome.decision.needs_clarification is True


def test_legacy_intent_without_action_gets_business_action():
    outcome = router.FastPathOutcome(
        kind="agent",
        lang="zh",
        intent=IntentResult(category=IntentCategory.recipe_execute),
    )

    assert outcome.decision.action == "prepare_device"
    assert outcome.decision.risk == "medium"
