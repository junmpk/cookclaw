"""阶段 4.2 多 IM 通道统一 facade 回归。"""

from __future__ import annotations

import asyncio
import json

from app.orchestrator.turn.response_renderer import (
    IMResponseRenderer,
)
from app.orchestrator.turn.runtime_models import ResponseEnvelope


def test_qq_weixin_and_whatsapp_enter_same_im_facade(monkeypatch):
    import app.agent.participle_agent as agent_module

    captured = []

    async def fake_facade(question, *, thread_id, channel, **_kwargs):
        captured.append((question, thread_id, channel))
        return ResponseEnvelope(
            response_type="chat",
            intent="chat",
            lang="zh",
            message=f"{channel}-reply",
            handled_by="planner",
        )

    monkeypatch.setattr(agent_module, "_run_turn_orchestrator", fake_facade)

    thread_ids = (
        "qq:dm:qq-user:qq-user",
        "weixin:dm:bot-a:wx-user",
        "whatsapp:dm:111@s.whatsapp.net:+111",
    )
    responses = [
        json.loads(asyncio.run(agent_module.qqbot_chat("你好", thread_id)))
        for thread_id in thread_ids
    ]

    assert [item[2] for item in captured] == ["qq", "weixin", "whatsapp"]
    assert [item["message"] for item in responses] == [
        "qq-reply",
        "weixin-reply",
        "whatsapp-reply",
    ]


def test_im_renderer_keeps_private_fields_hidden():
    envelope = ResponseEnvelope(
        response_type="recipe_search",
        intent="search",
        lang="zh",
        message="找到两道菜。",
        data={"count": 2},
        handled_by="recipe_handler",
        trace_id="private-trace",
    )

    current = IMResponseRenderer().render(envelope)
    payload = json.loads(current)
    assert payload == {
        "type": "recipe_search",
        "intent": "search",
        "lang": "zh",
        "data": {"count": 2},
        "message": "找到两道菜。",
    }


def test_im_facade_builds_turn_request_with_actual_channel(monkeypatch):
    import app.agent.participle_agent as agent_module

    captured = {}

    async def fake_execute_turn(request, **_kwargs):
        captured["request"] = request
        return ResponseEnvelope(message="handled", handled_by="test")

    monkeypatch.setattr(
        agent_module,
        "execute_turn",
        fake_execute_turn,
    )

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "推荐一道菜",
            thread_id="weixin:dm:bot-a:wx-user",
            channel="weixin",
            source_message_type="text",
            trace_id="phase42-trace",
        )
    )

    assert envelope.message == "handled"
    assert captured["request"].channel == "weixin"
    assert captured["request"].message_type == "text"
    assert captured["request"].context_ref()["has_trace"] is True
