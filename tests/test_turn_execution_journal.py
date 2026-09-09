"""阶段 2.3 真实工具结果与状态差异账本回归。"""

from __future__ import annotations

import asyncio
import json
from app.observability.trace import (
    collect_turn_traces,
    ensure_turn_trace,
    finish_turn_trace,
    record_tool_call,
)
from app.orchestrator.turn.execution_journal import TurnExecutionJournal
from app.orchestrator.turn.response_renderer import (
    IMResponseRenderer,
    response_to_envelope,
)
from app.orchestrator.turn.runtime_models import ResponseEnvelope, StatePatch, ToolResult
from app.conversation import task_state_workspace as session


def test_execution_journal_uses_real_trace_and_only_exposes_state_keys(monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")
    thread_id = "journal-device-state"
    session.clear_thread(thread_id)
    _trace, token = ensure_turn_trace(channel="qq", thread_id=thread_id)
    try:
        journal = TurnExecutionJournal.start(thread_id)
        record_tool_call(
            "device_start",
            duration_ms=12.5,
            success=False,
            error_type="DEVICE_OFFLINE",
            backend="kitchen_idea",
        )
        session.set_pending(
            thread_id,
            "private-recipe-id",
            "私密菜名",
            device_id="private-device-id",
            lang="zh",
        )
        envelope = journal.enrich(
            response_to_envelope(
                json.dumps(
                    {
                        "type": "cooking",
                        "intent": "chat",
                        "lang": "zh",
                        "data": {},
                        "message": "设备当前离线。",
                    },
                    ensure_ascii=False,
                )
            ),
            handled_by="pending_state",
        )
    finally:
        trace_payload = finish_turn_trace(token, success=True)

    assert len(envelope.tool_results) == 1
    tool_result = envelope.tool_results[0]
    assert tool_result.tool == "device_start"
    assert tool_result.success is False
    assert tool_result.code == "DEVICE_OFFLINE"
    assert tool_result.facts == {
        "tool_kind": "tool",
        "backend": "kitchen_idea",
        "timeout": False,
    }
    changed_keys = {
        key
        for patch in envelope.state_patches
        for key in patch.keys
    }
    assert {
        "current_task",
        "pending_device_start",
        "selected_device_id",
        "language",
    }.issubset(changed_keys)
    assert all(
        patch.reason_code == "PENDING_STATE_COMPLETED"
        for patch in envelope.state_patches
    )

    public = IMResponseRenderer().render(envelope)
    internal_journal = json.dumps(
        {
            "tool_results": [item.model_dump() for item in envelope.tool_results],
            "state_patches": [item.model_dump() for item in envelope.state_patches],
        },
        ensure_ascii=False,
    )
    assert "tool_results" not in public
    assert "state_patches" not in public
    assert "private-recipe-id" not in internal_journal
    assert "private-device-id" not in internal_journal
    assert "私密菜名" not in internal_journal
    tool_event = next(
        event
        for event in trace_payload["events"]
        if event.get("kind") == "tool_call"
    )
    assert tool_event["result_code"] == "DEVICE_OFFLINE"
    state_events = [
        event
        for event in trace_payload["events"]
        if event.get("kind") == "state_patch"
    ]
    assert state_events
    assert any(
        "pending_device_start" in event.get("keys", "")
        and event.get("scope") == "device"
        for event in state_events
    )
    serialized_trace = json.dumps(trace_payload, ensure_ascii=False)
    assert "private-recipe-id" not in serialized_trace
    assert "private-device-id" not in serialized_trace
    assert "私密菜名" not in serialized_trace
    session.clear_pending(thread_id)
    session.clear_thread(thread_id)


def test_execution_journal_records_real_state_clear_without_tool_inference():
    thread_id = "journal-clear-pending"
    session.clear_active_cooking(thread_id)
    session.clear_thread(thread_id)
    session.set_pending(
        thread_id,
        "private-recipe-id",
        "私密菜名",
        device_id="private-device-id",
        lang="zh",
    )
    journal = TurnExecutionJournal.start(thread_id)

    session.clear_pending(thread_id)
    envelope = journal.enrich(
        response_to_envelope("已取消。"),
        handled_by="exact_command",
    )

    assert envelope.tool_results == []
    clear_keys = {
        key
        for patch in envelope.state_patches
        if patch.operation == "clear"
        for key in patch.keys
    }
    assert {
        "current_task",
        "pending_action",
        "pending_device_start",
        "selected_device_id",
        "language",
    }.issubset(clear_keys)
    assert all(
        patch.reason_code == "EXACT_COMMAND_COMPLETED"
        for patch in envelope.state_patches
    )
    session.clear_thread(thread_id)


def test_execution_journal_records_device_execution_as_device_state():
    thread_id = "journal-device-execution"
    session.clear_thread(thread_id)
    session.clear_device_execution(thread_id)
    journal = TurnExecutionJournal.start(thread_id)

    session.set_device_execution(
        thread_id,
        {
            "cookId": "private-recipe-id",
            "device_id": "private-device-id",
            "action_id": "private-action-id",
            "msg_id": 12345,
            "status": "dispatching",
        },
    )
    envelope = journal.enrich(
        response_to_envelope("正在处理。"),
        handled_by="device_pending",
    )

    matching = [
        patch
        for patch in envelope.state_patches
        if "device_execution" in patch.keys
    ]
    assert len(matching) == 1
    assert matching[0].scope == "device"
    serialized = json.dumps(matching[0].model_dump(), ensure_ascii=False)
    assert "private-recipe-id" not in serialized
    assert "private-device-id" not in serialized
    assert "private-action-id" not in serialized
    session.clear_device_execution(thread_id)
    session.clear_thread(thread_id)


def test_execution_journal_preserves_explicit_domain_results_and_logs_patches(
    monkeypatch,
):
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")
    thread_id = "journal-explicit-memory-result"
    session.clear_thread(thread_id)
    _trace, token = ensure_turn_trace(channel="qq", thread_id=thread_id)
    try:
        journal = TurnExecutionJournal.start(thread_id)
        record_tool_call(
            "recipe_search",
            duration_ms=7.0,
            success=True,
            kind="search",
            result_count=2,
        )
        envelope = journal.enrich(
            ResponseEnvelope(
                response_type="cooking",
                message="已保存并继续搜索。",
                tool_results=[
                    ToolResult(
                        tool="preference_memory",
                        success=True,
                        code="PREFERENCE_SAVED",
                        facts={"domain": "memory", "status": "saved"},
                    )
                ],
                state_patches=[
                    StatePatch(
                        scope="session",
                        operation="set",
                        keys=["dislikes"],
                        reason_code="PREFERENCE_SESSION_UPDATED",
                    )
                ],
            ),
            handled_by="conversation_fallback",
        )
    finally:
        trace_payload = finish_turn_trace(token, success=True)

    assert [result.tool for result in envelope.tool_results] == [
        "preference_memory",
        "recipe_search",
    ]
    assert envelope.state_patches[0].keys == ["dislikes"]
    assert any(
        event.get("kind") == "state_patch"
        and event.get("scope") == "session"
        and event.get("keys") == "dislikes"
        for event in trace_payload["events"]
    )
    session.clear_thread(thread_id)


def test_qq_facade_emits_memory_domain_result_without_exposing_profile_value(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation.command_result import MemoryStateChange
    from app.conversation.profile_command import ProfileTurnReply
    from app.conversation.service import ConversationService
    from app.conversation.store import InMemoryConversationStore

    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")
    monkeypatch.setattr(
        service_module,
        "_service",
        ConversationService(InMemoryConversationStore()),
    )
    thread_id = "qq:dm:journal-memory:journal-memory"
    session.clear_thread(thread_id)

    async def fake_profile(_question, _thread_id):
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
        raise AssertionError("Memory 命中后不得继续")

    monkeypatch.setattr(agent_module, "_build_qq_turn_runtime", forbidden)
    monkeypatch.setattr(agent_module, "handle_profile_memory_turn", fake_profile)
    monkeypatch.setattr(agent_module, "handle_preference_memory_turn", forbidden)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", forbidden)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", forbidden)
    monkeypatch.setattr(agent_module, "_handle_qq_recipe_route", forbidden)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", forbidden)

    with collect_turn_traces() as traces:
        raw = asyncio.run(agent_module.qqbot_chat("以后请叫我周总", thread_id))

    assert json.loads(raw)["message"] == "好，以后我就叫你周总。"
    assert len(traces) == 1
    domain_event = next(
        event
        for event in traces[0]["events"]
        if event.get("kind") == "domain_result"
    )
    assert domain_event["name"] == "profile_memory"
    assert domain_event["result_code"] == "PROFILE_SAVED"
    assert domain_event["state_patch_count"] == 1
    assert any(
        event.get("kind") == "state_patch"
        and event.get("scope") == "profile"
        and event.get("keys") == "preferred_name"
        for event in traces[0]["events"]
    )
    summary = next(
        event
        for event in traces[0]["events"]
        if event.get("kind") == "turn_orchestrator"
    )
    assert summary["handled_by"] == "memory_command"
    assert summary["tool_result_count"] == 1
    assert summary["state_patch_count"] == 1
    assert "周总" not in json.dumps(traces, ensure_ascii=False)
    session.clear_thread(thread_id)


def test_qq_facade_adds_journal_counts_to_trace_without_changing_reply(monkeypatch):
    import app.agent.participle_agent as agent_module

    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")
    thread_id = "qq"
    session.clear_thread(thread_id)

    async def fake_prelude(_question, **_kwargs):
        return agent_module._IMMemoryPrelude()

    async def skip(_runtime, **_kwargs):
        return None

    async def fake_router(runtime):
        record_tool_call(
            "recipe_search",
            duration_ms=8.0,
            success=True,
            kind="search",
            result_count=2,
            backend="inprocess",
            fallback=False,
            result_code="OK",
        )
        session.set_pending_action(
            runtime.thread_id,
            "select_recipe",
            payload={"candidate_count": 2},
            lang="zh",
        )
        return json.dumps(
            {
                "type": "recipe_search",
                "intent": "search",
                "lang": "zh",
                "data": {"recipes": []},
                "message": "找到两道真实候选。",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(agent_module, "_run_im_memory_prelude", fake_prelude)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", skip)
    monkeypatch.setattr(agent_module, "_handle_qq_device_pending", skip)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", skip)
    monkeypatch.setattr(agent_module, "_handle_qq_recipe_route", skip)
    monkeypatch.setattr(agent_module, "_handle_qq_device_route", skip)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", fake_router)

    with collect_turn_traces() as traces:
        raw = asyncio.run(agent_module.qqbot_chat("帮我找菜", thread_id))

    assert json.loads(raw)["message"] == "找到两道真实候选。"
    assert "tool_results" not in raw
    assert "state_patches" not in raw
    assert len(traces) == 1
    event = next(
        item
        for item in traces[0]["events"]
        if item.get("kind") == "turn_orchestrator"
    )
    assert event["handled_by"] == "conversation_fallback"
    assert event["tool_result_count"] == 1
    assert event["state_patch_count"] >= 1
    assert event["tool_failure_count"] == 0
    search_event = next(
        item
        for item in traces[0]["events"]
        if item.get("kind") == "tool_call"
    )
    assert search_event["result_code"] == "OK"
    assert any(
        item.get("kind") == "state_patch"
        and "pending_action" in item.get("keys", "")
        for item in traces[0]["events"]
    )
    session.clear_thread(thread_id)
