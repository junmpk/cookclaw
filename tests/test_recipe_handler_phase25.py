"""阶段 2.5 Recipe Handler、单次路由与事实失败回归。"""

from __future__ import annotations

import asyncio
import json

from app.observability.trace import ensure_turn_trace, finish_turn_trace
from app.orchestrator.turn.memory_adapter import MemoryHandlerResult
from app.orchestrator.turn.response_renderer import IMResponseRenderer
from app.orchestrator.intent import IntentCategory, IntentResult
from app.orchestrator.router import FastPathOutcome


def _intent(category: IntentCategory) -> IntentResult:
    return IntentResult(
        related=True,
        category=category,
        keywords=[],
        action="new_search" if category == IntentCategory.recipe_search else "casual",
        source="exact_rule",
        reason_code="PHASE25_TEST",
    )


def test_recipe_handler_owns_grounded_search_and_routes_once(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    thread_id = "phase25-recipe-search"
    session.clear_thread(thread_id)
    calls: list[str] = []

    async def empty_memory(_question, **_kwargs):
        return MemoryHandlerResult()

    async def skip(_runtime):
        return None

    async def fake_route(*_args, **_kwargs):
        calls.append("route")
        return FastPathOutcome(
            kind="search",
            lang="zh",
            intent=_intent(IntentCategory.recipe_search),
            search_query="私密检索词",
            search_result={
                "success": True,
                "results": [
                    {
                        "id": "private-recipe-id",
                        "metadata": {
                            "recipe_id": "private-recipe-id",
                            "name": "私密菜名",
                        },
                    }
                ],
            },
        )

    async def fake_commit(*_args, **_kwargs):
        calls.append("commit")

    async def fake_format(*_args, **_kwargs):
        return json.dumps(
            {
                "type": "recipe_search",
                "intent": "search",
                "lang": "zh",
                "data": {
                    "recipes": [
                        {"id": "private-recipe-id", "name": "私密菜名"}
                    ]
                },
                "message": "找到一道真实菜谱。",
            },
            ensure_ascii=False,
        )

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("Recipe 命中后不得进入 Legacy router")

    monkeypatch.setattr(agent_module, "_run_im_memory_prelude", empty_memory)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", skip)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", skip)
    monkeypatch.setattr(agent_module, "_route_conversation_turn", fake_route)
    monkeypatch.setattr(agent_module, "_commit_search_state", fake_commit)
    monkeypatch.setattr(agent_module, "_format_search_response", fake_format)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", forbidden)

    _trace, token = ensure_turn_trace(channel="qq", thread_id=thread_id)
    try:
        envelope = asyncio.run(
            agent_module._run_turn_orchestrator(
                "帮我找一道菜",
                channel="qq",
                thread_id=thread_id,
                source_message_type=None,
                trace_id=_trace.trace_id,
            )
        )
    finally:
        trace_payload = finish_turn_trace(token, success=True)

    assert calls == ["route", "commit"]
    assert envelope.handled_by == "recipe_handler"
    assert json.loads(IMResponseRenderer().render(envelope))["message"] == (
        "找到一道真实菜谱。"
    )
    domain_event = next(
        event
        for event in trace_payload["events"]
        if event.get("kind") == "domain_result"
    )
    assert domain_event["name"] == "recipe_flow"
    assert domain_event["operation"] == "search"
    assert domain_event["result_code"] == "RECIPE_SEARCH_READY"
    assert domain_event["result_count"] == 1
    serialized_trace = json.dumps(trace_payload, ensure_ascii=False)
    assert "私密检索词" not in serialized_trace
    assert "private-recipe-id" not in serialized_trace
    assert "私密菜名" not in serialized_trace
    session.clear_thread(thread_id)


def test_recipe_handler_search_failure_does_not_fall_through_to_free_reply(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module

    async def empty_memory(_question, **_kwargs):
        return MemoryHandlerResult()

    async def skip(_runtime):
        return None

    async def fake_route(*_args, **_kwargs):
        return FastPathOutcome(
            kind="agent",
            lang="zh",
            intent=_intent(IntentCategory.recipe_search),
        )

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("检索失败不得进入自由回复")

    monkeypatch.setattr(agent_module, "_run_im_memory_prelude", empty_memory)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", skip)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", skip)
    monkeypatch.setattr(agent_module, "_route_conversation_turn", fake_route)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", forbidden)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "帮我找一道不存在的菜",
            channel="qq",
            thread_id="phase25-search-failed",
            source_message_type=None,
            trace_id="trace-phase25-search-failed",
        )
    )
    rendered = json.loads(IMResponseRenderer().render(envelope))

    assert envelope.handled_by == "recipe_handler"
    assert "不凭空" in rendered["message"]


def test_non_recipe_flow_reuses_recipe_handler_route_outcome(monkeypatch):
    import app.agent.participle_agent as agent_module

    calls: list[str] = []

    async def fake_route(*_args, **_kwargs):
        calls.append("route")
        return FastPathOutcome(
            kind="greeting",
            lang="zh",
            intent=_intent(IntentCategory.greeting),
        )

    async def fake_greeting(*_args, **_kwargs):
        return agent_module._cook_msg("你好呀。", "zh")

    monkeypatch.setattr(agent_module, "_route_conversation_turn", fake_route)
    monkeypatch.setattr(agent_module, "_greeting_with_memory", fake_greeting)

    runtime = agent_module._QQTurnRuntime(
        question="你好",
        raw_question="你好",
        thread_id="phase25-route-cache",
        on_search_start=None,
        source_message_type="text",
        total_start=0.0,
        channel_markdown=False,
        voice_input=False,
        clarification_stateful=False,
        pending_clarification=None,
    )

    recipe_result = asyncio.run(agent_module._handle_qq_recipe_route(runtime))
    legacy_response = asyncio.run(agent_module._handle_qq_conversation_fallback(runtime))

    assert recipe_result is None
    assert json.loads(legacy_response)["message"] == "你好呀。"
    assert calls == ["route"]
