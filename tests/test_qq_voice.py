"""QQ 语音入站与设备确认回归测试；不访问网络、不调用真实设备。"""
import asyncio
import json

from app.agent.fast_path import detect_lang
from app.qqbot.adapter import MessageEvent, MessageSource, MessageType, QQAdapter


def test_qq_voice_attachment_keeps_clean_transcript_and_voice_type():
    async def run():
        events = []
        adapter = QQAdapter(extra={"dm_policy": "open"})

        async def collect(event):
            events.append(event)

        adapter.set_message_handler(collect)

        await adapter._handle_c2c_message({
            "id": "voice-message-1",
            "content": "",
            "author": {"user_openid": "voice-user"},
            "attachments": [{
                "content_type": "voice",
                "asr_refer_text": "确认。",
            }],
        })

        assert len(events) == 1
        assert events[0].text == "确认。"
        assert events[0].message_type is MessageType.VOICE
        assert "[Voice]" not in events[0].text

    asyncio.run(run())


def test_clean_qq_voice_transcript_is_detected_as_chinese():
    assert detect_lang("确认。") == "zh"
    assert detect_lang("停止。") == "zh"


def test_qq_handler_normalizes_voice_before_memory_and_agent(monkeypatch):
    import app.main as main_module
    import app.conversation.service as service_module
    from app.conversation.service import ConversationService, InMemoryConversationStore

    calls = []

    class Adapter:
        def __init__(self):
            self.sent = []

        async def send(self, chat_id, text, reply_to=""):
            self.sent.append((chat_id, text, reply_to))

    async def fake_chat(
        question,
        thread_id="default",
        on_search_start=None,
        source_message_type=None,
    ):
        calls.append((question, thread_id, source_message_type))
        return "started"

    async def run():
        service = ConversationService(InMemoryConversationStore())
        adapter = Adapter()
        monkeypatch.setattr(service_module, "_service", service)
        monkeypatch.setattr(main_module, "_qq_adapter", adapter)
        monkeypatch.setattr(main_module, "qqbot_chat", fake_chat)

        event = MessageEvent(
            source=MessageSource(
                chat_id="voice-user",
                user_id="voice-user",
                chat_type="dm",
            ),
            text="确认，嗯。",
            message_type=MessageType.VOICE,
            message_id="voice-message-2",
        )
        await main_module._qqbot_message_handler(event)
        await service.drain_pending()

        tid = "qq:dm:voice-user:voice-user"
        memory = await service.load(tid)
        assert calls == [("确认", tid, "voice")]
        assert [turn.content for turn in memory.recent_turns] == ["确认", "started"]
        assert adapter.sent == [("voice-user", "started", "voice-message-2")]

    asyncio.run(run())


def test_voice_confirmation_starts_only_after_exact_normalization(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation.service import ConversationService
    from app.conversation.store import InMemoryConversationStore

    calls = []
    service = ConversationService(InMemoryConversationStore())

    async def fake_cook(
        thread_id,
        cook_id,
        name,
        device_id=None,
        lang="zh",
        **kwargs,
    ):
        calls.append((thread_id, cook_id, name, device_id, lang))
        return agent_module._cook_msg("started", lang)

    async def must_not_route(*args, **kwargs):
        raise AssertionError("待确认语音不应重新进入意图路由")

    async def run():
        tid = "qq:dm:voice-confirm:voice-confirm"
        agent_module.clear_thread(tid)
        agent_module.set_pending(
            tid, "recipe-voice", "三和菜", device_id="showroom", lang="zh",
        )
        await service.save_task_state(
            tid,
            agent_module.snapshot_thread_state(tid),
        )
        agent_module.clear_thread(tid)

        response = json.loads(await agent_module.qqbot_chat(
            "[Voice] 确认，嗯。",
            tid,
            source_message_type="voice",
        ))

        assert response["message"] == "started"
        assert calls == [(tid, "recipe-voice", "三和菜", "showroom", "zh")]
        assert agent_module.get_pending(tid) is None
        agent_module.clear_thread(tid)

    monkeypatch.setattr(agent_module, "route_fast_path", must_not_route)
    monkeypatch.setattr(agent_module, "_cook_and_format", fake_cook)
    monkeypatch.setattr(
        service_module,
        "_service",
        service,
    )
    asyncio.run(run())


def test_ambiguous_voice_keeps_pending_and_never_operates_device(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation.service import ConversationService
    from app.conversation.store import InMemoryConversationStore

    service = ConversationService(InMemoryConversationStore())

    async def must_not_cook(*args, **kwargs):
        raise AssertionError("含糊语音绝不能操作设备")

    async def must_not_route(*args, **kwargs):
        raise AssertionError("含糊语音应留在待确认状态")

    async def run():
        tid = "qq:dm:voice-ambiguous:voice-ambiguous"
        agent_module.clear_thread(tid)
        agent_module.set_pending(
            tid, "recipe-voice", "三和菜", device_id="showroom", lang="zh",
        )
        await service.save_task_state(
            tid,
            agent_module.snapshot_thread_state(tid),
        )
        agent_module.clear_thread(tid)

        response = json.loads(await agent_module.qqbot_chat(
            "[Voice] 承认第三道。",
            tid,
            source_message_type="voice",
        ))

        assert "这次没有操作设备" in response["message"]
        assert "待确认仍保留" in response["message"]
        persisted = await service.load_task_state(tid)
        assert persisted.pending_device_start["cookId"] == "recipe-voice"
        agent_module.clear_thread(tid)

    monkeypatch.setattr(agent_module, "route_fast_path", must_not_route)
    monkeypatch.setattr(agent_module, "_cook_and_format", must_not_cook)
    monkeypatch.setattr(service_module, "_service", service)
    asyncio.run(run())


def test_voice_stop_uses_chinese_reply_and_the_recorded_device(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    import app.orchestrator.cook as cook_module
    from app.conversation.service import ConversationService
    from app.conversation.store import InMemoryConversationStore
    from app.orchestrator.intent import IntentCategory, IntentResult
    from app.orchestrator.router import FastPathOutcome

    calls = []
    service = ConversationService(InMemoryConversationStore())

    async def fake_route(question, **kwargs):
        assert question == "停止"
        return FastPathOutcome(
            kind="agent",
            lang=detect_lang(question),
            intent=IntentResult(related=True, category=IntentCategory.recipe_execute),
        )

    async def fake_stop(lang="zh", device_id=None):
        calls.append((lang, device_id))
        return {"ok": True}

    async def run():
        tid = "qq:dm:voice-stop:voice-stop"
        agent_module.clear_thread(tid)
        agent_module.set_active_cooking(
            tid, "recipe-voice", "三和菜", device_id="showroom", lang="zh",
        )
        await service.save_task_state(
            tid,
            agent_module.snapshot_thread_state(tid),
        )
        agent_module.clear_thread(tid)

        response = json.loads(await agent_module.qqbot_chat(
            "[Voice] 停止。",
            tid,
            source_message_type="voice",
        ))

        assert response["lang"] == "zh"
        assert "已发送停止" in response["message"]
        assert calls == [("zh", "showroom")]
        assert (await service.load_task_state(tid)).active_cooking is None
        agent_module.clear_thread(tid)

    monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
    monkeypatch.setattr(cook_module, "stop_cook", fake_stop)
    monkeypatch.setattr(service_module, "_service", service)
    asyncio.run(run())
