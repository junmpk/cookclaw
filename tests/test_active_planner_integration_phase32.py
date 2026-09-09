"""阶段 3.2 Active Planner 接入业务编排入口的安全回归。"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

from app.core.config import settings
from app.observability.trace import (
    ensure_turn_trace,
    finish_turn_trace,
    record_tool_call,
)
from app.orchestrator.turn.recipe_adapter import RecipeHandlerResult
from app.orchestrator.turn.runtime_models import ResponseEnvelope
from app.conversation import task_state_workspace as session
from app.orchestrator.intent import IntentCategory


def _runtime(agent_module, question: str, *, thread_id: str):
    return agent_module._QQTurnRuntime(
        question=question,
        raw_question=question,
        thread_id=thread_id,
        on_search_start=None,
        source_message_type="text",
        total_start=time.monotonic(),
        channel_markdown=True,
        voice_input=False,
        clarification_stateful=True,
        pending_clarification=None,
    )


def _plan(agent_module, action: str, *, args=None, risk: str = "low"):
    return agent_module.TurnPlan(
        goal="完成当前用户请求",
        phase=("device_prepare" if action == "device.prepare" else "explore"),
        steps=[
            agent_module.PlanStep(
                action=action,
                args=dict(args or {}),
                evidence_refs=["utterance"],
            )
        ],
        reply_act=(
            "offer_next_step"
            if action == "device.prepare"
            else "answer"
        ),
        risk=risk,
        reason_code="PHASE32_TEST",
        confidence=0.9,
    )


def _ready(plan):
    return SimpleNamespace(
        status="ready",
        execution_allowed=True,
        plan=plan,
        action=plan.steps[0].action,
        fallback_code=None,
    )


def _enable_active(monkeypatch, *, user_id: str, actions: str):
    monkeypatch.setattr(settings, "TURN_PLANNER_MODE", "active")
    monkeypatch.setattr(settings, "TURN_PLANNER_ACTIVE_CHANNELS", "qq")
    monkeypatch.setattr(
        settings,
        "TURN_PLANNER_ACTIVE_QQ_ALLOW_FROM",
        user_id,
    )
    monkeypatch.setattr(settings, "TURN_PLANNER_ACTIVE_ROLLOUT_PERCENT", 0)
    monkeypatch.setattr(settings, "TURN_PLANNER_ACTIVE_ACTIONS", actions)


def _isolate_context(monkeypatch, agent_module):
    monkeypatch.setattr(agent_module, "supports_session_memory", lambda _thread: False)
    monkeypatch.setattr(agent_module, "_planner_exact_handoff", lambda *_args: False)


def test_active_recipe_search_uses_original_text_for_grounded_router(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module

    user_id = "phase32-search"
    thread_id = f"qq:dm:{user_id}:{user_id}"
    session.clear_thread(thread_id)
    _enable_active(monkeypatch, user_id=user_id, actions="recipe.search")
    _isolate_context(monkeypatch, agent_module)
    plan = _plan(
        agent_module,
        "recipe.search",
        args={"query": "Planner 不得替换成这个词"},
    )
    captured = {}

    async def fake_evaluate(_context):
        return _ready(plan)

    async def fake_route(question, thread, **kwargs):
        captured["question"] = question
        captured["thread_id"] = thread
        captured["intent"] = kwargs["intent_override"]
        return object()

    async def fake_recipe_handler(runtime, **_kwargs):
        assert runtime.route_outcome is not None
        return RecipeHandlerResult(
            envelope=ResponseEnvelope(
                response_type="recipe_search",
                message="真实检索结果",
            ),
            operation="search",
            success=True,
            result_code="RECIPE_SEARCH_SUCCESS",
            result_count=1,
        )

    monkeypatch.setattr(agent_module, "evaluate_active_plan", fake_evaluate)
    monkeypatch.setattr(agent_module, "_route_conversation_turn", fake_route)
    monkeypatch.setattr(agent_module, "_handle_qq_recipe_route", fake_recipe_handler)

    question = "家里有鸡腿和土豆，找一道不辣的菜"
    envelope = asyncio.run(
        agent_module._handle_qq_active_planner(
            _runtime(agent_module, question, thread_id=thread_id),
            trace_id="phase32-search-trace",
        )
    )

    assert envelope is not None
    assert envelope.handled_by == "planner"
    assert envelope.message == "真实检索结果"
    assert captured["question"] == question
    assert captured["thread_id"] == thread_id
    assert captured["intent"].category == IntentCategory.recipe_search
    assert captured["intent"].keywords == [question]
    assert captured["intent"].search_request.original_text == question
    assert captured["intent"].search_request.query == question


def test_active_planner_never_receives_exact_high_risk_command(monkeypatch):
    import app.agent.participle_agent as agent_module

    user_id = "phase32-stop"
    thread_id = f"qq:dm:{user_id}:{user_id}"
    _enable_active(monkeypatch, user_id=user_id, actions="conversation.respond")
    monkeypatch.setattr(agent_module, "supports_session_memory", lambda _thread: False)
    monkeypatch.setattr(agent_module, "_planner_exact_handoff", lambda *_args: True)

    async def forbidden(_context):
        raise AssertionError("高风险精确命令不应调用 Planner")

    monkeypatch.setattr(agent_module, "evaluate_active_plan", forbidden)
    result = asyncio.run(
        agent_module._handle_qq_active_planner(
            _runtime(agent_module, "立即停止设备", thread_id=thread_id),
        )
    )

    assert result is None


def test_active_planner_cannot_upgrade_chat_to_web_or_device_tool():
    import app.agent.participle_agent as agent_module

    assert agent_module._planner_action_compatible(
        "conversation.respond",
        "今天有点累，陪我聊聊",
    )
    assert not agent_module._planner_action_compatible(
        "web.search",
        "今天有点累，陪我聊聊",
    )
    assert not agent_module._planner_action_compatible(
        "device.status",
        "今天有点累，陪我聊聊",
    )
    assert not agent_module._planner_action_compatible(
        "device.prepare",
        "今天有点累，陪我聊聊",
    )
    assert agent_module._planner_action_compatible(
        "device.status",
        "我的设备在线吗",
    )
    assert agent_module._planner_action_compatible(
        "web.search",
        "帮我联网查一下今天上海的天气",
    )


def test_active_planner_conversation_cannot_answer_grounded_recipe_requests():
    import app.agent.participle_agent as agent_module

    assert not agent_module._planner_action_compatible(
        "conversation.respond",
        "我在杭州，今晚10个人吃饭，请推荐一个食谱清单给我",
    )
    assert not agent_module._planner_action_compatible(
        "conversation.respond",
        "10个人推荐五个菜太少了",
    )
    assert agent_module._planner_action_compatible(
        "recipe.recommend",
        "我在杭州，今晚10个人吃饭，请推荐一个食谱清单给我",
    )
    assert agent_module._planner_action_compatible(
        "conversation.respond",
        "今天有点累，陪我聊聊",
    )


def test_active_planner_hands_structured_pending_to_deterministic_router(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module

    user_id = "phase32-structured-pending"
    thread_id = f"qq:dm:{user_id}:{user_id}"
    _enable_active(monkeypatch, user_id=user_id, actions="conversation.respond")
    runtime = _runtime(
        agent_module,
        "我已经不减肥了。以后给我推荐菜谱不要参考这个标准",
        thread_id=thread_id,
    )
    runtime.pending_clarification = {
        "request": {
            "party_size": 11,
            "constraints_confirmed": True,
            "context_note": {
                "kind": "remembered_dietary_constraint",
                "term": "减肥",
            },
        },
        "dimension": "remembered_dietary_constraint",
    }

    async def forbidden(_context):
        raise AssertionError("结构化 pending 存在时不应调用 Planner")

    monkeypatch.setattr(agent_module, "evaluate_active_plan", forbidden)
    result = asyncio.run(
        agent_module._handle_qq_active_planner(runtime)
    )

    assert result is None


def test_active_planner_recipe_request_falls_through_before_free_answer(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module

    user_id = "phase32-grounded-fallthrough"
    thread_id = f"qq:dm:{user_id}:{user_id}"
    _enable_active(monkeypatch, user_id=user_id, actions="conversation.respond")
    _isolate_context(monkeypatch, agent_module)
    plan = _plan(agent_module, "conversation.respond")

    async def fake_evaluate(_context):
        return _ready(plan)

    async def forbidden_answer(*_args, **_kwargs):
        raise AssertionError("菜谱请求不得交给自由对话模型生成")

    monkeypatch.setattr(agent_module, "evaluate_active_plan", fake_evaluate)
    monkeypatch.setattr(
        agent_module,
        "_planner_conversation_answer",
        forbidden_answer,
    )

    result = asyncio.run(
        agent_module._handle_qq_active_planner(
            _runtime(
                agent_module,
                "今晚10个人吃饭，请推荐一个食谱清单",
                thread_id=thread_id,
            )
        )
    )

    assert result is None


def test_diet_retraction_resumes_pending_party_menu_through_recipe_handler(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    import app.orchestrator.router as router
    from app.conversation.preference_command import handle_preference_memory_turn
    from app.conversation.service import ConversationService
    from app.conversation.store import InMemoryConversationStore
    from app.orchestrator.intent import IntentResult
    from app.orchestrator.recommendation_response import RecommendationNarrative
    from app.orchestrator.search_request import SearchRequest

    user_id = "phase32-diet-party-resume"
    thread_id = f"qq:dm:{user_id}:{user_id}"
    service = ConversationService(InMemoryConversationStore())
    _enable_active(monkeypatch, user_id=user_id, actions="conversation.respond")
    monkeypatch.setattr(service_module, "_service", service)
    captured = {}

    async def classify(question, *_args):
        return IntentResult.from_raw(
            {
                "r": True,
                "c": "recipe_recommend",
                "a": "new_search",
                "k": [question],
                "s": {"q": question},
            },
            original_text=question,
        )

    async def build_menu_plan(request, **_kwargs):
        captured["request"] = request
        results = [
            {
                "id": f"recipe-{index}",
                "menu_role": "dish" if index <= 6 else "soup",
                "metadata": {
                    "recipe_id": f"recipe-{index}",
                    "name": f"真实菜谱{index}",
                    "ingredients": ["测试食材"],
                },
            }
            for index in range(1, 9)
        ]
        return {
            "success": True,
            "results": results,
            "_candidate_pool": results,
            "_search_request": request.public_dict(),
            "_menu_plan": {
                "requested": {"dish": 6, "soup": 2},
                "fulfilled": {"dish": 6, "soup": 2},
                "missing": {"dish": 0, "soup": 0},
                "complete": True,
                "errors": [],
                "queries": [{"query": request.retrieval_query()}],
            },
        }

    async def narrative(**_kwargs):
        return RecommendationNarrative("", "", {}, "")

    async def forbidden_planner(_context):
        raise AssertionError("结构化 pending 存在时不得执行 Planner")

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "build_menu_plan", build_menu_plan)
    monkeypatch.setattr(router, "generate_recommendation_narrative", narrative)
    monkeypatch.setattr(agent_module, "evaluate_active_plan", forbidden_planner)

    async def run():
        seeded = await handle_preference_memory_turn(
            "我现在正在减肥",
            thread_id,
        )
        assert seeded is not None and seeded.status == "saved"
        pending_request = SearchRequest(
            original_text=(
                "我在杭州，今晚有10个朋友来家里吃饭，都是江西人，"
                "要喝白酒，所有人都没忌口，请推荐食谱清单"
            ),
            query="晚餐 下酒",
            scenes=["下酒", "减肥"],
            meals=["晚餐"],
            party_size=11,
            constraints_confirmed=True,
            context_note={
                "kind": "remembered_dietary_constraint",
                "term": "减肥",
            },
        )
        await service.save_search_clarification(
            thread_id,
            {
                "request": pending_request.model_dump(),
                "dimension": "remembered_dietary_constraint",
                "asked_dimensions": ["remembered_dietary_constraint"],
                "round_count": 1,
                "lang": "zh",
                "ts": time.time(),
            },
        )

        envelope = await agent_module._run_turn_orchestrator(
            "我已经不减肥了。以后给我推荐菜谱不要参考这个标准",
            channel="qq",
            thread_id=thread_id,
            source_message_type="text",
            trace_id="phase32-diet-party-resume-trace",
        )

        request = captured["request"]
        assert request.party_size == 11
        assert request.constraints_confirmed is True
        assert request.menu_dish_count == 6
        assert request.menu_soup_count == 2
        assert "减肥" not in request.scenes
        assert request.exclude == []
        assert request.cuisines == []
        assert request.flavors == []
        retrieval_query = request.retrieval_query()
        assert "减肥" not in retrieval_query
        assert "参考" not in retrieval_query
        assert "标准" not in retrieval_query
        assert envelope.handled_by == "recipe_handler"
        assert envelope.response_type == "menu_plan"
        assert len(envelope.data["recipes"]) == 8
        assert "之前关于减肥的限制已经取消" in envelope.message
        assert [item.tool for item in envelope.tool_results] == [
            "preference_memory"
        ]
        assert await service.load_search_clarification(thread_id) is None
        memory = await service.load(thread_id)
        assert "减肥" not in memory.preferences["dietary_constraints"]

    asyncio.run(run())


def test_active_execution_without_effect_falls_back_once(monkeypatch):
    import app.agent.participle_agent as agent_module

    user_id = "phase32-safe-fallback"
    thread_id = f"qq:dm:{user_id}:{user_id}"
    session.clear_thread(thread_id)
    _enable_active(monkeypatch, user_id=user_id, actions="conversation.respond")
    _isolate_context(monkeypatch, agent_module)
    plan = _plan(agent_module, "conversation.respond")

    async def fake_evaluate(_context):
        return _ready(plan)

    async def fail_before_effect(_runtime, _plan):
        raise RuntimeError("reply failed before any effect")

    monkeypatch.setattr(agent_module, "evaluate_active_plan", fake_evaluate)
    monkeypatch.setattr(
        agent_module,
        "_planner_conversation_answer",
        fail_before_effect,
    )

    result = asyncio.run(
        agent_module._handle_qq_active_planner(
            _runtime(agent_module, "今天有点累", thread_id=thread_id),
        )
    )

    assert result is None
    assert session.snapshot_thread_state(thread_id).pending_action is None


def test_active_execution_after_tool_call_fails_closed_without_retry(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module

    user_id = "phase32-effect"
    thread_id = f"qq:dm:{user_id}:{user_id}"
    session.clear_thread(thread_id)
    _enable_active(monkeypatch, user_id=user_id, actions="web.search")
    _isolate_context(monkeypatch, agent_module)
    monkeypatch.setattr(agent_module, "_planner_action_compatible", lambda *_args: True)
    plan = _plan(
        agent_module,
        "web.search",
        args={"query": "联网信息"},
    )

    async def fake_evaluate(_context):
        return _ready(plan)

    async def fail_after_tool(_question, **_kwargs):
        record_tool_call(
            "web_search",
            duration_ms=2,
            success=False,
            result_code="UPSTREAM_UNKNOWN",
        )
        raise RuntimeError("tool result is unknown")

    monkeypatch.setattr(agent_module, "evaluate_active_plan", fake_evaluate)
    monkeypatch.setattr(agent_module, "search_web", fail_after_tool)
    _trace, token = ensure_turn_trace(channel="qq", thread_id=thread_id)
    try:
        envelope = asyncio.run(
            agent_module._handle_qq_active_planner(
                _runtime(agent_module, "帮我联网查一下", thread_id=thread_id),
            )
        )
    finally:
        trace = finish_turn_trace(token, success=True)

    assert envelope is not None
    assert envelope.handled_by == "planner"
    assert "不会自动再执行一次" in envelope.message
    assert trace["tool_call_count"] == 1
    assert "PLANNER_EXECUTION_FAIL_CLOSED_AFTER_EFFECT" in trace["fallback_reasons"]
    execution_event = next(
        event
        for event in trace["events"]
        if event.get("kind") == "planner_execution"
    )
    assert execution_event["status"] == "failed_closed"
    assert execution_event["action"] == "web.search"
    assert execution_event["result_code"] == "ACTIVE_ACTION_EXECUTION_ERROR"


def test_active_device_prepare_only_creates_confirmation_state(monkeypatch):
    import app.agent.participle_agent as agent_module

    user_id = "phase32-device"
    thread_id = f"qq:dm:{user_id}:{user_id}"
    session.clear_thread(thread_id)
    session.remember_candidates(
        thread_id,
        {
            "results": [
                {
                    "id": "recipe-1",
                    "metadata": {
                        "recipe_id": "recipe-1",
                        "name": "土豆炖鸡",
                    },
                }
            ]
        },
        lang="zh",
    )
    _enable_active(monkeypatch, user_id=user_id, actions="device.prepare")
    _isolate_context(monkeypatch, agent_module)
    monkeypatch.setattr(agent_module, "_planner_action_compatible", lambda *_args: True)
    plan = _plan(
        agent_module,
        "device.prepare",
        args={"recipe_id": "recipe-1"},
        risk="medium",
    )
    calls = {"prepare": 0, "cook": 0}

    async def fake_evaluate(_context):
        return _ready(plan)

    async def fake_prepare(thread, recipe, lang):
        calls["prepare"] += 1
        session.set_pending(
            thread,
            recipe["cookId"],
            recipe["name"],
            device_id="device-1",
            lang=lang,
        )
        return agent_module._cook_msg("设备已就绪，请确认后再开始。", lang)

    async def forbidden_cook(*_args, **_kwargs):
        calls["cook"] += 1
        raise AssertionError("device.prepare 不得发送开火命令")

    monkeypatch.setattr(agent_module, "evaluate_active_plan", fake_evaluate)
    monkeypatch.setattr(agent_module, "_prepare_recipe_for_device", fake_prepare)
    monkeypatch.setattr(agent_module, "_cook_and_format", forbidden_cook)

    envelope = asyncio.run(
        agent_module._handle_qq_active_planner(
            _runtime(agent_module, "用设备做第一道", thread_id=thread_id),
        )
    )

    assert envelope is not None
    assert calls == {"prepare": 1, "cook": 0}
    assert session.get_pending(thread_id)["cookId"] == "recipe-1"
    assert session.get_active_cooking(thread_id) is None


def test_facade_runs_pending_stage_before_active_planner_and_legacy(monkeypatch):
    import app.agent.participle_agent as agent_module

    user_id = "phase32-facade"
    thread_id = f"qq:dm:{user_id}:{user_id}"
    _enable_active(monkeypatch, user_id=user_id, actions="conversation.respond")
    calls = []

    async def fake_prelude(_question, **_kwargs):
        calls.append("memory")
        return agent_module._IMMemoryPrelude()

    async def fake_build(question, **_kwargs):
        calls.append("runtime")
        return _runtime(agent_module, question, thread_id=thread_id)

    async def skip_exact(_runtime):
        calls.append("exact")
        return None

    async def skip_device_pending(_runtime, **_kwargs):
        calls.append("device_pending")
        return None

    async def pending_skip(_runtime):
        calls.append("pending")
        return None

    async def planner_reply(_runtime, **_kwargs):
        calls.append("planner")
        return ResponseEnvelope(message="Planner 已处理")

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("Planner 命中后不得继续执行后续 handler")

    monkeypatch.setattr(agent_module, "_run_im_memory_prelude", fake_prelude)
    monkeypatch.setattr(agent_module, "_build_qq_turn_runtime", fake_build)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", skip_exact)
    monkeypatch.setattr(agent_module, "_handle_qq_device_pending", skip_device_pending)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", pending_skip)
    monkeypatch.setattr(agent_module, "_handle_qq_active_planner", planner_reply)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", forbidden)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "随便聊两句",
            channel="qq",
            thread_id=thread_id,
            source_message_type="text",
            trace_id="phase32-facade-trace",
        )
    )

    assert envelope.handled_by == "planner"
    assert envelope.message == "Planner 已处理"
    assert calls == [
        "memory",
        "runtime",
        "exact",
        "device_pending",
        "pending",
        "planner",
    ]


def test_active_planner_internal_fields_do_not_leak_to_public_json():
    from app.orchestrator.turn.response_renderer import IMResponseRenderer

    envelope = ResponseEnvelope(
        response_type="chat",
        message="可以，我们接着聊。",
        handled_by="planner",
        trace_id="private-trace",
    )
    public = json.loads(IMResponseRenderer().render(envelope))

    assert public["message"] == "可以，我们接着聊。"
    assert "handled_by" not in public
    assert "trace_id" not in public
