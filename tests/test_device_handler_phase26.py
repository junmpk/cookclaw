"""阶段 2.6 Device Handler、确认门禁和结构化工具事实回归。"""

from __future__ import annotations

import asyncio
import json

from app.observability.trace import (
    ensure_turn_trace,
    finish_turn_trace,
    record_tool_call,
)
from app.orchestrator.turn.response_renderer import IMResponseRenderer
from app.orchestrator.intent import IntentCategory, IntentResult
from app.orchestrator.router import FastPathOutcome


def _intent(category: IntentCategory) -> IntentResult:
    return IntentResult(
        related=True,
        category=category,
        keywords=[],
        action=(
            "device_status"
            if category == IntentCategory.device_manage
            else "prepare_device"
        ),
        source="exact_rule",
        reason_code="PHASE26_TEST",
    )


def _runtime(agent_module, *, question: str, thread_id: str, outcome):
    return agent_module._QQTurnRuntime(
        question=question,
        raw_question=question,
        thread_id=thread_id,
        on_search_start=None,
        source_message_type=None,
        total_start=0.0,
        channel_markdown=True,
        voice_input=False,
        clarification_stateful=True,
        pending_clarification=None,
        route_outcome=outcome,
    )


def test_device_status_handler_uses_structured_tool_result_not_reply_text(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module

    private_reply = "设备 private-device-id 状态暂时未知"

    async def fake_status(*_args, **_kwargs):
        record_tool_call(
            "device_status",
            duration_ms=8.0,
            success=False,
            error_type="DEVICE_STATUS_TIMEOUT",
            result_code="DEVICE_STATUS_TIMEOUT",
        )
        return agent_module._cook_msg(private_reply, "zh")

    monkeypatch.setattr(agent_module, "_device_status_and_format", fake_status)
    runtime = _runtime(
        agent_module,
        question="查看设备状态",
        thread_id="phase26-status",
        outcome=FastPathOutcome(
            kind="agent",
            lang="zh",
            intent=_intent(IntentCategory.device_manage),
        ),
    )

    trace, token = ensure_turn_trace(channel="qq", thread_id=runtime.thread_id)
    try:
        result = asyncio.run(
            agent_module._handle_qq_device_route(
                runtime,
                trace_id=trace.trace_id,
            )
        )
    finally:
        payload = finish_turn_trace(token, success=True)

    assert result is not None
    assert result.operation == "status"
    assert result.success is False
    assert result.result_code == "DEVICE_STATUS_TIMEOUT"
    assert json.loads(IMResponseRenderer().render(result.envelope))["message"] == (
        private_reply
    )
    domain = next(
        event
        for event in payload["events"]
        if event.get("kind") == "domain_result"
    )
    assert domain["name"] == "device_flow"
    assert domain["result_code"] == "DEVICE_STATUS_TIMEOUT"
    assert private_reply not in json.dumps(payload, ensure_ascii=False)
    assert "private-device-id" not in json.dumps(payload, ensure_ascii=False)


def test_pending_confirmation_claims_once_and_reports_unknown_start_truth(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    thread_id = "phase26-confirm"
    session.clear_thread(thread_id)
    session.set_pending(
        thread_id,
        "private-recipe-id",
        "私密菜名",
        device_id="private-device-id",
        lang="zh",
    )
    pending = session.get_pending(thread_id)
    calls: list[str] = []

    async def fake_claim(claim_thread_id, *, expected_action_id):
        calls.append("claim")
        assert claim_thread_id == thread_id
        assert expected_action_id == pending["action_id"]
        session.clear_pending(thread_id)
        return dict(pending)

    async def fake_cook(*_args, **_kwargs):
        calls.append("cook")
        record_tool_call(
            "device_start",
            duration_ms=12.0,
            success=False,
            error_type="DEVICE_STATE_UNVERIFIED",
            result_code="DEVICE_STATE_UNVERIFIED",
            command_sent=True,
            state_unknown=True,
        )
        return agent_module._cook_msg("启动结果待确认。", "zh")

    monkeypatch.setattr(agent_module, "claim_pending_device_start", fake_claim)
    monkeypatch.setattr(agent_module, "_cook_and_format", fake_cook)
    runtime = _runtime(
        agent_module,
        question="确认",
        thread_id=thread_id,
        outcome=None,
    )

    trace, token = ensure_turn_trace(channel="qq", thread_id=thread_id)
    try:
        result = asyncio.run(
            agent_module._handle_qq_device_pending(
                runtime,
                trace_id=trace.trace_id,
            )
        )
    finally:
        payload = finish_turn_trace(token, success=True)

    assert result is not None
    assert calls == ["claim", "cook"]
    assert result.operation == "confirm_start"
    assert result.success is False
    assert result.result_code == "DEVICE_STATE_UNVERIFIED"
    assert result.command_sent is True
    assert result.state_unknown is True
    assert session.get_pending(thread_id) is None
    domain = next(
        event
        for event in payload["events"]
        if event.get("kind") == "domain_result"
    )
    assert domain["risk"] == "high"
    assert domain["command_sent"] is True
    assert domain["state_unknown"] is True
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "private-recipe-id" not in serialized
    assert "private-device-id" not in serialized
    assert "私密菜名" not in serialized
    session.clear_thread(thread_id)


def test_prepare_device_stops_at_confirmation_and_never_starts(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    thread_id = "phase26-precheck"
    session.clear_thread(thread_id)
    session.remember_candidates(
        thread_id,
        {
            "success": True,
            "results": [
                {
                    "id": "recipe-1",
                    "metadata": {"recipe_id": "recipe-1", "name": "番茄炒蛋"},
                }
            ],
        },
        lang="zh",
    )

    async def fake_prepare(precheck_thread_id, recipe, lang):
        record_tool_call(
            "device_status",
            duration_ms=5.0,
            success=True,
            result_code="OK",
        )
        session.set_pending(
            precheck_thread_id,
            recipe["cookId"],
            recipe["name"],
            device_id="office",
            lang=lang,
        )
        return agent_module._cook_msg("请确认是否启动。", lang)

    async def must_not_start(*_args, **_kwargs):
        raise AssertionError("设备预检轮不得发送启动指令")

    monkeypatch.setattr(agent_module, "_prepare_recipe_for_device", fake_prepare)
    monkeypatch.setattr(agent_module, "_cook_and_format", must_not_start)
    runtime = _runtime(
        agent_module,
        question="用设备做第一道",
        thread_id=thread_id,
        outcome=FastPathOutcome(
            kind="agent",
            lang="zh",
            intent=_intent(IntentCategory.recipe_execute),
        ),
    )

    trace, token = ensure_turn_trace(channel="qq", thread_id=thread_id)
    try:
        result = asyncio.run(
            agent_module._handle_qq_device_route(
                runtime,
                trace_id=trace.trace_id,
            )
        )
    finally:
        payload = finish_turn_trace(token, success=True)

    assert result is not None
    assert result.operation == "precheck"
    assert result.result_code == "DEVICE_CONFIRMATION_REQUIRED"
    assert result.success is True
    assert session.get_pending(thread_id)["device_id"] == "office"
    assert not any(
        event.get("kind") == "tool_call" and event.get("name") == "device_start"
        for event in payload["events"]
    )
    session.clear_thread(thread_id)


def test_unknown_device_outcome_remains_durable_and_is_not_requeued(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.orchestrator.cook as cook_module
    from app.conversation import task_state_workspace as session

    thread_id = "phase26-outcome-unknown"
    action_id = "action-unknown"
    session.clear_thread(thread_id)
    session.clear_active_cooking(thread_id)
    session.clear_device_execution(thread_id)
    session.set_device_execution(
        thread_id,
        {
            "action_id": action_id,
            "msg_id": 12345,
            "cookId": "recipe-unknown",
            "name": "番茄炒蛋",
            "device_id": "office",
            "lang": "zh",
            "status": "dispatching",
        },
    )

    async def unknown_result(*_args, **_kwargs):
        return {
            "ok": False,
            "code": "DEVICE_OUTCOME_UNKNOWN",
            "reason": "设备指令结果未知，请不要重复启动。",
            "command_sent": None,
            "started": None,
            "retryable": False,
            "msg_id": 12345,
        }

    monkeypatch.setattr(cook_module, "execute_cook", unknown_result)
    response = asyncio.run(
        agent_module._cook_and_format(
            thread_id,
            "recipe-unknown",
            "番茄炒蛋",
            device_id="office",
            lang="zh",
            action_id=action_id,
            msg_id=12345,
        )
    )

    execution = session.get_device_execution(thread_id)
    assert execution["status"] == "outcome_unknown"
    assert execution["command_sent"] is None
    assert execution["started"] is None
    assert session.get_pending(thread_id) is None
    assert session.get_active_cooking(thread_id) is None
    assert "不要重复启动" in json.loads(response)["message"]
    session.clear_device_execution(thread_id)


def test_verified_device_start_links_active_task_to_execution_identity(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.orchestrator.cook as cook_module
    from app.conversation import task_state_workspace as session

    thread_id = "phase26-start-running"
    action_id = "action-running"
    session.clear_thread(thread_id)
    session.clear_active_cooking(thread_id)
    session.clear_device_execution(thread_id)
    session.set_device_execution(
        thread_id,
        {
            "action_id": action_id,
            "msg_id": 54321,
            "cookId": "recipe-running",
            "name": "红烧肉",
            "device_id": "office",
            "lang": "zh",
            "status": "dispatching",
        },
    )

    async def running_result(*_args, **_kwargs):
        return {
            "ok": True,
            "code": "OK",
            "reason": "",
            "command_sent": True,
            "started": True,
            "retryable": False,
            "msg_id": 54321,
        }

    monkeypatch.setattr(cook_module, "execute_cook", running_result)
    asyncio.run(
        agent_module._cook_and_format(
            thread_id,
            "recipe-running",
            "红烧肉",
            device_id="office",
            lang="zh",
            action_id=action_id,
            msg_id=54321,
        )
    )

    execution = session.get_device_execution(thread_id)
    active = session.get_active_cooking(thread_id)
    assert execution["status"] == "running"
    assert active["action_id"] == action_id
    assert active["msg_id"] == 54321
    assert active["verification_status"] == "running"
    session.clear_active_cooking(thread_id)
    session.clear_device_execution(thread_id)
