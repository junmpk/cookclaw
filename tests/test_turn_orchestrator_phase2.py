"""阶段 2 TurnOrchestrator facade、响应协议与 QQ 兼容回归。"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.orchestrator.turn.response_renderer import (
    IMResponseRenderer,
    response_to_envelope,
    prepend_response_ack,
    public_response_shape,
)
from app.orchestrator.turn.runtime_models import (
    ResponseEnvelope,
    StatePatch,
    ToolResult,
    TurnRequest,
)
from app.orchestrator.turn.turn_orchestrator import (
    TurnNotHandledError,
    TurnOrchestrator,
)


def _request() -> TurnRequest:
    return TurnRequest(
        utterance="帮我找红烧肉",
        thread_id="qq:dm:test:test",
        channel="qq",
        trace_id="trace-phase2",
    )


def test_turn_orchestrator_uses_fixed_first_handler_wins_order():
    calls: list[str] = []

    async def memory(_request):
        calls.append("memory_command")
        return None

    async def exact(_request):
        calls.append("exact_command")
        return None

    async def pending(_request):
        calls.append("pending_state")
        return None

    async def device_pending(_request):
        calls.append("device_pending")
        return None

    async def recipe(_request):
        calls.append("recipe_handler")
        return ResponseEnvelope(
            response_type="recipe_search",
            message="找到真实菜谱",
        )

    async def forbidden(_request):
        raise AssertionError("首个命中 handler 后不得继续执行")

    envelope = asyncio.run(
        TurnOrchestrator(
            memory_command_handler=memory,
            exact_command_handler=exact,
            device_pending_handler=device_pending,
            pending_state_handler=pending,
            recipe_handler=recipe,
            conversation_fallback_handler=forbidden,
            fallback_handler=forbidden,
        ).handle(_request())
    )

    assert calls == [
        "memory_command",
        "exact_command",
        "device_pending",
        "pending_state",
        "recipe_handler",
    ]
    assert envelope.handled_by == "recipe_handler"
    assert envelope.message == "找到真实菜谱"


def test_pending_state_preempts_planner_when_it_handles_the_turn():
    calls: list[str] = []

    async def pending(_request):
        calls.append("pending_state")
        return ResponseEnvelope(message="继续当前结构化任务")

    async def forbidden_planner(_request):
        raise AssertionError("pending 命中后 Planner 不得重解释当前轮")

    envelope = asyncio.run(
        TurnOrchestrator(
            pending_state_handler=pending,
            planner_handler=forbidden_planner,
        ).handle(_request())
    )

    assert calls == ["pending_state"]
    assert envelope.handled_by == "pending_state"
    assert envelope.message == "继续当前结构化任务"


def test_turn_request_safe_ref_and_repr_do_not_expose_private_text():
    request = _request()

    assert request.context_ref() == {
        "schema_version": "turn_request_v1",
        "channel": "qq",
        "message_type": "text",
        "input_chars": 6,
        "has_trace": True,
    }
    assert "帮我找红烧肉" not in repr(request)
    assert "qq:dm:test:test" not in repr(request)


def test_turn_orchestrator_fails_closed_when_no_handler_accepts_turn():
    async def skip(_request):
        return None

    with pytest.raises(TurnNotHandledError):
        asyncio.run(
            TurnOrchestrator(
                exact_command_handler=skip,
                pending_state_handler=skip,
                conversation_fallback_handler=skip,
                fallback_handler=skip,
            ).handle(_request())
        )


def test_legacy_response_round_trip_preserves_public_qq_payload():
    payload = {
        "type": "recipe_search",
        "intent": "search",
        "lang": "zh",
        "data": {
            "recipes": [
                {"id": "r1", "name": "红烧肉"},
            ],
        },
        "message": "给你找到一道菜。",
        "compat_field": {"version": 1},
    }
    raw = json.dumps(payload, ensure_ascii=False)

    envelope = response_to_envelope(
        raw,
        trace_id="trace-round-trip",
    )
    envelope.tool_results.append(
        ToolResult(tool="recipe_search", success=True, facts={"count": 1})
    )
    envelope.state_patches.append(
        StatePatch(
            scope="session",
            operation="set",
            keys=["candidate_recipes"],
            reason_code="SEARCH_SUCCEEDED",
        )
    )
    rendered = IMResponseRenderer().render(envelope)

    assert json.loads(rendered) == payload
    assert public_response_shape(envelope) == payload
    assert envelope.response_type == "recipe_search"
    assert envelope.handled_by == "conversation_fallback"
    assert "tool_results" not in json.loads(rendered)
    assert "state_patches" not in json.loads(rendered)
    assert "trace-round-trip" not in rendered
    assert "channel_payload" not in envelope.model_dump()


def test_plain_legacy_response_round_trip_stays_plain_text():
    envelope = response_to_envelope("普通文本回复")

    assert envelope.response_type == "text"
    assert IMResponseRenderer().render(envelope) == "普通文本回复"


def test_preference_ack_updates_envelope_without_exposing_internal_fields():
    payload = {
        "type": "recipe_search",
        "intent": "search",
        "lang": "zh",
        "data": {"opening": "给你找到两道。"},
        "message": "可以看看第二道。",
    }
    envelope = response_to_envelope(
        json.dumps(payload, ensure_ascii=False),
        trace_id="trace-preference-ack",
    )

    updated = prepend_response_ack(envelope, "记住了，你不吃花生。")
    rendered = json.loads(IMResponseRenderer().render(updated))

    assert rendered["message"] == "记住了，你不吃花生。 可以看看第二道。"
    assert rendered["data"]["opening"] == "记住了，你不吃花生。 给你找到两道。"
    assert "trace-preference-ack" not in json.dumps(rendered, ensure_ascii=False)


def test_qq_facade_calls_split_handlers_once_in_fixed_order(monkeypatch):
    import app.agent.participle_agent as agent_module

    calls: list[str] = []
    payload = {
        "type": "chat",
        "intent": "clarify",
        "lang": "zh",
        "data": {},
        "message": "你想继续哪一步？",
    }

    async def fake_prelude(_question, **_kwargs):
        calls.append("memory_prelude")
        return agent_module._IMMemoryPrelude()

    async def fake_build(_question, **_kwargs):
        calls.append("build_runtime")
        return object()

    async def fake_exact(_runtime):
        calls.append("exact_command")
        return None

    async def fake_pending(_runtime):
        calls.append("pending_state")
        return None

    async def fake_device_pending(_runtime, **_kwargs):
        calls.append("device_pending")
        return None

    async def fake_device(_runtime, **_kwargs):
        calls.append("device_handler")
        return None

    async def fake_recipe(_runtime, **_kwargs):
        calls.append("recipe_handler")
        return None

    async def fake_legacy(_runtime):
        calls.append("conversation_fallback")
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(agent_module, "_run_im_memory_prelude", fake_prelude)
    monkeypatch.setattr(agent_module, "_build_qq_turn_runtime", fake_build)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", fake_exact)
    monkeypatch.setattr(agent_module, "_handle_qq_device_pending", fake_device_pending)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", fake_pending)
    monkeypatch.setattr(agent_module, "_handle_qq_recipe_route", fake_recipe)
    monkeypatch.setattr(agent_module, "_handle_qq_device_route", fake_device)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", fake_legacy)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "继续",
            channel="qq",
            thread_id="qq:dm:test:test",
            source_message_type=None,
            trace_id="trace-facade",
        )
    )

    assert calls == [
        "memory_prelude",
        "build_runtime",
        "exact_command",
        "device_pending",
        "pending_state",
        "recipe_handler",
        "device_handler",
        "conversation_fallback",
    ]
    assert envelope.trace_id == "trace-facade"
    assert envelope.handled_by == "conversation_fallback"
    assert json.loads(IMResponseRenderer().render(envelope)) == payload


def test_qq_facade_runs_memory_prelude_once_and_keeps_preference_ack(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation.command_result import MemoryStateChange

    calls: list[str] = []

    class PreferenceReply:
        message = "记住了，你不吃花生。"
        lang = "zh"
        continue_routing = True
        status = "saved"
        success = True
        state_changes = (
            MemoryStateChange(
                scope="session",
                operation="set",
                keys=("dislikes",),
                reason_code="PREFERENCE_SESSION_UPDATED",
            ),
        )

    async def fake_profile(_question, _thread_id):
        calls.append("profile")
        return None

    async def fake_preference(_question, _thread_id):
        calls.append("preference")
        return PreferenceReply()

    async def fake_build(_question, **_kwargs):
        calls.append("build_runtime")
        return object()

    async def skip(_runtime):
        return None

    async def skip_device(_runtime, **_kwargs):
        return None

    async def pending_reply(_runtime):
        calls.append("pending_state")
        return json.dumps(
            {
                "type": "recipe_search",
                "intent": "search",
                "lang": "zh",
                "data": {"opening": "给你找到一道。"},
                "message": "这道可以看看。",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(agent_module, "handle_profile_memory_turn", fake_profile)
    monkeypatch.setattr(agent_module, "handle_preference_memory_turn", fake_preference)
    monkeypatch.setattr(agent_module, "_build_qq_turn_runtime", fake_build)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", skip)
    monkeypatch.setattr(agent_module, "_handle_qq_device_pending", skip_device)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", pending_reply)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "不吃花生，继续",
            channel="qq",
            thread_id="qq",
            source_message_type=None,
            trace_id="trace-prelude-once",
        )
    )
    rendered = json.loads(IMResponseRenderer().render(envelope))

    assert calls == ["profile", "preference", "build_runtime", "pending_state"]
    assert envelope.handled_by == "pending_state"
    assert [result.tool for result in envelope.tool_results] == [
        "preference_memory"
    ]
    assert envelope.tool_results[0].code == "PREFERENCE_SAVED"
    assert envelope.state_patches[0].scope == "session"
    assert envelope.state_patches[0].keys == ["dislikes"]
    assert rendered["message"] == "记住了，你不吃花生。 这道可以看看。"
    assert rendered["data"]["opening"] == "记住了，你不吃花生。 给你找到一道。"


def test_qq_facade_preference_write_failure_continues_without_false_patch(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    calls: list[str] = []
    thread_id = "phase24-preference-write-failed"
    session.clear_thread(thread_id)

    class PreferenceReply:
        message = "本轮我会照这个处理，但会话偏好没有保存成功。"
        lang = "zh"
        continue_routing = True
        status = "write_failed"
        success = False
        state_changes = ()

    async def fake_profile(_question, _thread_id):
        calls.append("profile")
        return None

    async def fake_preference(_question, _thread_id):
        calls.append("preference")
        return PreferenceReply()

    async def fake_build(_question, **_kwargs):
        calls.append("build_runtime")
        return object()

    async def skip(_runtime):
        return None

    async def skip_device(_runtime, **_kwargs):
        return None

    async def pending_reply(_runtime):
        calls.append("pending_state")
        return json.dumps(
            {
                "type": "recipe_search",
                "intent": "search",
                "lang": "zh",
                "data": {},
                "message": "继续按本轮条件找菜。",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(agent_module, "handle_profile_memory_turn", fake_profile)
    monkeypatch.setattr(agent_module, "handle_preference_memory_turn", fake_preference)
    monkeypatch.setattr(agent_module, "_build_qq_turn_runtime", fake_build)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", skip)
    monkeypatch.setattr(agent_module, "_handle_qq_device_pending", skip_device)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", pending_reply)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "不吃花生，继续找菜",
            channel="qq",
            thread_id=thread_id,
            source_message_type=None,
            trace_id="trace-preference-write-failed",
        )
    )
    rendered = json.loads(IMResponseRenderer().render(envelope))

    assert calls == ["profile", "preference", "build_runtime", "pending_state"]
    assert envelope.handled_by == "pending_state"
    assert len(envelope.tool_results) == 1
    assert envelope.tool_results[0].tool == "preference_memory"
    assert envelope.tool_results[0].success is False
    assert envelope.tool_results[0].code == "PREFERENCE_WRITE_FAILED"
    assert envelope.tool_results[0].error_type == "PREFERENCE_WRITE_FAILED"
    assert envelope.state_patches == []
    assert rendered["message"] == (
        "本轮我会照这个处理，但会话偏好没有保存成功。 "
        "继续按本轮条件找菜。"
    )
    session.clear_thread(thread_id)


def test_qq_facade_memory_command_short_circuits_and_returns_domain_result(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    from app.conversation.command_result import MemoryStateChange
    from app.conversation.profile_command import ProfileTurnReply

    calls: list[str] = []

    async def fake_build(_question, **_kwargs):
        calls.append("build_runtime")
        return object()

    async def fake_profile(_question, _thread_id):
        calls.append("profile")
        return ProfileTurnReply(
            "好，以后我就叫你周总。",
            "saved",
            state_changes=(
                MemoryStateChange(
                    scope="profile",
                    operation="set",
                    keys=("preferred_name",),
                    reason_code="PREFERRED_NAME_SAVED",
                ),
            ),
        )

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("Memory 命中后不得执行后续 handler")

    monkeypatch.setattr(agent_module, "_build_qq_turn_runtime", fake_build)
    monkeypatch.setattr(agent_module, "handle_profile_memory_turn", fake_profile)
    monkeypatch.setattr(agent_module, "handle_preference_memory_turn", forbidden)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", forbidden)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", forbidden)
    monkeypatch.setattr(agent_module, "_handle_qq_recipe_route", forbidden)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", forbidden)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "以后请叫我周总",
            channel="qq",
            thread_id="default",
            source_message_type=None,
            trace_id="trace-memory-command",
        )
    )
    rendered = json.loads(IMResponseRenderer().render(envelope))

    assert calls == ["profile"]
    assert envelope.handled_by == "memory_command"
    assert rendered["message"] == "好，以后我就叫你周总。"
    assert [result.tool for result in envelope.tool_results] == ["profile_memory"]
    assert envelope.tool_results[0].code == "PROFILE_SAVED"
    assert envelope.state_patches[0].scope == "profile"
    assert envelope.state_patches[0].keys == ["preferred_name"]
    internal = json.dumps(
        {
            "tool_results": [item.model_dump() for item in envelope.tool_results],
            "state_patches": [item.model_dump() for item in envelope.state_patches],
        },
        ensure_ascii=False,
    )
    assert "周总" not in internal


def test_qq_facade_reset_is_handled_before_pending_and_router(monkeypatch):
    import app.agent.participle_agent as agent_module

    async def fake_prelude(_question, **_kwargs):
        return agent_module._IMMemoryPrelude()

    async def fake_build(_question, **kwargs):
        return agent_module._QQTurnRuntime(
            question="给我一个新对话",
            raw_question="给我一个新对话",
            thread_id=kwargs["thread_id"],
            on_search_start=None,
            source_message_type=kwargs["source_message_type"],
            total_start=0.0,
            channel_markdown=True,
            voice_input=False,
            clarification_stateful=True,
            pending_clarification=None,
        )

    async def fake_reset(_thread_id, _question):
        return "已经开始一个新对话。", "zh"

    async def forbidden(_runtime):
        raise AssertionError("精确命令命中后不得继续执行")

    monkeypatch.setattr(agent_module, "_run_im_memory_prelude", fake_prelude)
    monkeypatch.setattr(agent_module, "_build_qq_turn_runtime", fake_build)
    monkeypatch.setattr(agent_module, "_reset_conversation_state", fake_reset)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", forbidden)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", forbidden)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "给我一个新对话",
            channel="qq",
            thread_id="qq",
            source_message_type=None,
            trace_id="trace-reset",
        )
    )

    assert envelope.handled_by == "exact_command"
    assert json.loads(IMResponseRenderer().render(envelope))["message"] == "已经开始一个新对话。"


def test_qq_facade_candidate_followup_is_handled_before_router(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    thread_id = "qq"
    session.clear_thread(thread_id)

    async def fake_prelude(_question, **_kwargs):
        return agent_module._IMMemoryPrelude()

    async def fake_build(_question, **kwargs):
        return agent_module._QQTurnRuntime(
            question="第二道有什么区别",
            raw_question="第二道有什么区别",
            thread_id=kwargs["thread_id"],
            on_search_start=None,
            source_message_type=kwargs["source_message_type"],
            total_start=0.0,
            channel_markdown=True,
            voice_input=False,
            clarification_stateful=True,
            pending_clarification=None,
        )

    async def skip(*_args, **_kwargs):
        return None

    async def candidate_reply(*_args, **_kwargs):
        return agent_module._cook_msg("第二道更清淡。", "zh")

    async def forbidden(_runtime):
        raise AssertionError("候选追问命中后不得进入 router")

    monkeypatch.setattr(agent_module, "_run_im_memory_prelude", fake_prelude)
    monkeypatch.setattr(agent_module, "_build_qq_turn_runtime", fake_build)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", skip)
    monkeypatch.setattr(agent_module, "_handle_pending_action", skip)
    monkeypatch.setattr(agent_module, "_recipe_condition_followup", skip)
    monkeypatch.setattr(agent_module, "_resume_recipe_task", skip)
    monkeypatch.setattr(agent_module, "_memory_correction_followup", skip)
    monkeypatch.setattr(agent_module, "_omitted_recipe_followup", skip)
    monkeypatch.setattr(agent_module, "_search_memory_followup", skip)
    monkeypatch.setattr(agent_module, "_recent_candidate_followup", candidate_reply)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", forbidden)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "第二道有什么区别",
            channel="qq",
            thread_id=thread_id,
            source_message_type=None,
            trace_id="trace-candidate",
        )
    )

    assert envelope.handled_by == "pending_state"
    assert json.loads(IMResponseRenderer().render(envelope))["message"] == "第二道更清淡。"
    session.clear_thread(thread_id)


def test_qq_facade_clarification_cancel_is_pending_state(monkeypatch):
    import app.agent.participle_agent as agent_module

    clear_calls: list[str] = []

    async def fake_prelude(_question, **_kwargs):
        return agent_module._IMMemoryPrelude()

    async def fake_build(_question, **kwargs):
        return agent_module._QQTurnRuntime(
            question="先不选了",
            raw_question="先不选了",
            thread_id=kwargs["thread_id"],
            on_search_start=None,
            source_message_type=kwargs["source_message_type"],
            total_start=0.0,
            channel_markdown=True,
            voice_input=False,
            clarification_stateful=True,
            pending_clarification={"dimension": "meal_size"},
        )

    async def skip(*_args, **_kwargs):
        return None

    async def fake_clear(thread_id):
        clear_calls.append(thread_id)

    async def forbidden(_runtime):
        raise AssertionError("澄清取消命中后不得进入 router")

    monkeypatch.setattr(agent_module, "_run_im_memory_prelude", fake_prelude)
    monkeypatch.setattr(agent_module, "_build_qq_turn_runtime", fake_build)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", skip)
    monkeypatch.setattr(agent_module, "_handle_pending_action", skip)
    monkeypatch.setattr(agent_module, "_clear_search_clarification_state", fake_clear)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", forbidden)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "先不选了",
            channel="qq",
            thread_id="qq",
            source_message_type=None,
            trace_id="trace-clarification-cancel",
        )
    )

    assert envelope.handled_by == "pending_state"
    assert clear_calls == ["qq"]
    assert "先不往下挑" in json.loads(IMResponseRenderer().render(envelope))["message"]


def test_device_confirmation_side_effect_executes_once_through_public_entry(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    thread_id = "qq"
    cook_calls: list[dict] = []

    async def fake_cook_and_format(*_args, **kwargs):
        cook_calls.append(dict(kwargs))
        return agent_module._cook_msg("已发送一次启动指令", "zh")

    monkeypatch.setattr(agent_module, "_cook_and_format", fake_cook_and_format)

    session.clear_pending(thread_id)
    session.set_pending(
        thread_id,
        "recipe-1",
        "红烧肉",
        device_id="test-device",
        lang="zh",
    )

    response = json.loads(asyncio.run(agent_module.qqbot_chat("确认", thread_id)))

    assert response["message"] == "已发送一次启动指令"
    assert len(cook_calls) == 1
    assert all(call["device_id"] == "test-device" for call in cook_calls)
    assert session.get_pending(thread_id) is None
    session.clear_thread(thread_id)
    session.clear_device_execution(thread_id)


def test_no_pending_confirmation_is_handled_by_unified_pending_stage(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    thread_id = "qq"
    session.clear_thread(thread_id)
    session.clear_device_execution(thread_id)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "可以",
            channel="qq",
            thread_id=thread_id,
            source_message_type=None,
            trace_id="trace-no-pending",
        )
    )
    response = json.loads(IMResponseRenderer().render(envelope))

    assert envelope.handled_by == "pending_state"
    assert "没有正在等确认的步骤" in response["message"]
    assert response["message"].count("？") == 1
    assert not any(
        word in response["message"]
        for word in ("已确认", "已开始", "已执行", "不能替你猜")
    )
    assert session.get_pending(thread_id) is None
    session.clear_thread(thread_id)
    session.clear_device_execution(thread_id)
