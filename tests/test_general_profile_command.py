import asyncio
import json

import app.agent.participle_agent as agent_module
import app.conversation.service as service_module
from app.conversation.general_profile_command import (
    handle_general_profile_memory_turn,
)
from app.conversation.preference_parser import extract_preference_mutations
from app.conversation.profile_command import handle_profile_memory_turn
from app.conversation.profile_fact_extractor import ProfileFactExtractorResult
from app.conversation.profile_facts import ProfileFactCandidate
from app.conversation import profile_rollout
from app.conversation.service import ConversationService, build_qq_thread_id
from app.conversation.store import InMemoryConversationStore
from app.observability.trace import collect_turn_traces


def _candidate(*, operation="upsert", value="AI应用开发工程师", evidence=None):
    return ProfileFactCandidate(
        operation=operation,
        category="work",
        key="occupation",
        value=value,
        subject="self",
        scope="stable",
        sensitivity="normal",
        evidence=evidence or (
            "忘掉我的职业" if operation == "delete" else "我是AI应用开发工程师"
        ),
    )


class FakeExtractor:
    def __init__(self, factory):
        self.factory = factory
        self.calls = []

    async def extract(self, text, *, force=False):
        self.calls.append((text, force))
        return self.factory(text)


class FakeReplyModel:
    def __init__(self, content):
        self.content = content
        self.messages = []

    async def ainvoke(self, messages):
        self.messages.append(messages)
        return type("Reply", (), {"content": self.content})()


def _enable_general_memory(monkeypatch):
    monkeypatch.setattr(
        profile_rollout.settings,
        "CONVERSATION_PROFILE_MEMORY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        profile_rollout.settings,
        "CONVERSATION_PROFILE_MEMORY_CHANNELS",
        "qq",
    )
    monkeypatch.setattr(
        profile_rollout.settings,
        "CONVERSATION_PROFILE_MEMORY_QQ_ALLOW_FROM",
        "user-a",
    )
    monkeypatch.setattr(
        profile_rollout.settings,
        "CONVERSATION_PROFILE_MEMORY_ROLLOUT_PERCENT",
        0,
    )
    monkeypatch.setattr(
        profile_rollout.settings,
        "CONVERSATION_GENERAL_MEMORY_ENABLED",
        True,
    )


def _install(monkeypatch, *, extractor):
    store = InMemoryConversationStore()
    service = ConversationService(
        store,
        profile_store=store,
        profile_fact_extractor=extractor,
    )
    monkeypatch.setattr(service_module, "_service", service)
    return service


def test_explicit_general_fact_is_saved_before_natural_ack_and_trace_has_no_value(
    monkeypatch,
):
    async def run():
        _enable_general_memory(monkeypatch)
        extractor = FakeExtractor(lambda _text: ProfileFactExtractorResult(
            status="success",
            facts=[_candidate()],
            candidate_count=1,
        ))
        service = _install(monkeypatch, extractor=extractor)
        reply_model = FakeReplyModel("行，这个职业背景我接住了，之后聊工作时会顺着它来。")
        monkeypatch.setattr(agent_module, "_qa_llm", reply_model)
        thread_id = build_qq_thread_id("dm", "user-a", "user-a")

        with collect_turn_traces() as traces:
            raw = await agent_module.qqbot_chat(
                "请记住，我是AI应用开发工程师",
                thread_id,
            )
        response = json.loads(raw)
        assert response["message"] == "行，这个职业背景我接住了，之后聊工作时会顺着它来。"
        assert [item["value"] for item in await service.general_profile_facts(thread_id)] == [
            "AI应用开发工程师"
        ]
        serialized_trace = json.dumps(traces, ensure_ascii=False)
        assert "AI应用开发工程师" not in serialized_trace
        event = next(
            item
            for item in traces[0]["events"]
            if item["kind"] == "general_profile_memory"
        )
        assert event["status"] == "saved"
        assert event["accepted_count"] == 1

    asyncio.run(run())


def test_sensitive_rejection_cannot_be_turned_into_false_saved_claim(monkeypatch):
    async def run():
        _enable_general_memory(monkeypatch)
        extractor = FakeExtractor(lambda _text: ProfileFactExtractorResult(
            status="success",
            facts=[],
            candidate_count=1,
            rejected_count=1,
        ))
        service = _install(monkeypatch, extractor=extractor)
        monkeypatch.setattr(
            agent_module,
            "_qa_llm",
            FakeReplyModel("好的，你的服务器密码我已经记下了。"),
        )
        thread_id = build_qq_thread_id("dm", "user-a", "user-a")

        raw = await agent_module.qqbot_chat(
            "请记住，我的服务器密码是abc123",
            thread_id,
        )
        message = json.loads(raw)["message"]
        assert "不会保存" in message
        assert "abc123" not in message
        assert await service.general_profile_facts(thread_id) == []
        assert extractor.calls == []

    asyncio.run(run())


def test_generic_delete_is_not_misrouted_as_food_preference(monkeypatch):
    async def run():
        _enable_general_memory(monkeypatch)

        def extraction(text):
            candidate = (
                _candidate(operation="delete", value="", evidence="忘掉我的职业")
                if "忘掉" in text
                else _candidate()
            )
            return ProfileFactExtractorResult(
                status="success",
                facts=[candidate],
                candidate_count=1,
            )

        service = _install(monkeypatch, extractor=FakeExtractor(extraction))
        monkeypatch.setattr(agent_module, "_qa_llm", FakeReplyModel("好，那段职业资料我已经放下了。"))
        thread_id = build_qq_thread_id("dm", "user-a", "user-a")
        await service.apply_general_profile_facts(thread_id, [_candidate()])

        assert extract_preference_mutations("忘掉我的职业") == []
        raw = await agent_module.qqbot_chat("忘掉我的职业", thread_id)
        assert json.loads(raw)["message"] == "好，那段职业资料我已经放下了。"
        assert await service.general_profile_facts(thread_id) == []

    asyncio.run(run())


def test_who_am_i_bypasses_fixed_name_reply_and_uses_all_confirmed_facts(
    monkeypatch,
):
    async def run():
        _enable_general_memory(monkeypatch)
        service = _install(
            monkeypatch,
            extractor=FakeExtractor(lambda _text: ProfileFactExtractorResult(status="skipped")),
        )
        thread_id = build_qq_thread_id("dm", "user-a", "user-a")
        await service.update_preferred_name(
            thread_id,
            "范老师",
            expected_current=None,
        )
        await service.apply_general_profile_facts(thread_id, [_candidate()])

        assert await handle_profile_memory_turn("你还记得我是谁吗", thread_id) is None
        context = await agent_module._conversation_context_for_qa(
            thread_id,
            "你还记得我是谁吗",
        )
        assert '用户已确认希望被称为："范老师"' in context
        assert "用户职业：AI应用开发工程师" in context

        reply_model = FakeReplyModel(
            "当然有印象。你是做 AI 应用开发的范老师。"
        )
        monkeypatch.setattr(agent_module, "_qa_llm", reply_model)
        answer = await agent_module._smalltalk_answer(
            "你还记得我是谁吗",
            "zh",
            context=context,
        )
        assert answer.startswith("当然有印象")
        assert "记得，你希望我叫你范老师" not in answer
        messages = reply_model.messages[0]
        assert "用户职业：AI应用开发工程师" not in messages[0].content
        assert "用户职业：AI应用开发工程师" in messages[1].content

    asyncio.run(run())


def test_qq_who_am_i_reaches_natural_reply_with_confirmed_profile_context(
    monkeypatch,
):
    from app.orchestrator.intent import IntentCategory, IntentResult
    from app.orchestrator.router import FastPathOutcome

    async def fake_route(_question, **_kwargs):
        return FastPathOutcome(
            kind="agent",
            lang="zh",
            intent=IntentResult(
                related=False,
                category=IntentCategory.off_topic,
            ),
        )

    async def run():
        _enable_general_memory(monkeypatch)
        service = _install(
            monkeypatch,
            extractor=FakeExtractor(
                lambda _text: ProfileFactExtractorResult(status="skipped")
            ),
        )
        thread_id = build_qq_thread_id("dm", "user-a", "user-a")
        await service.update_preferred_name(
            thread_id,
            "范老师",
            expected_current=None,
        )
        await service.apply_general_profile_facts(thread_id, [_candidate()])
        reply_model = FakeReplyModel("有印象，你是做 AI 应用开发的范老师。")
        monkeypatch.setattr(agent_module, "_qa_llm", reply_model)

        response = json.loads(
            await agent_module.qqbot_chat("你还记得我是谁吗", thread_id)
        )

        assert response["message"] == "有印象，你是做 AI 应用开发的范老师。"
        messages = reply_model.messages[-1]
        system_prompt = messages[0].content
        context_data = messages[1].content
        assert '用户已确认希望被称为：\\"范老师\\"' in context_data
        assert "用户职业：AI应用开发工程师" in context_data
        assert "用户职业：AI应用开发工程师" not in system_prompt
        assert "记得，你希望我叫你范老师" not in response["message"]

    monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
    asyncio.run(run())


def test_passive_capture_runs_after_assistant_and_explicit_command_is_not_repeated(
    monkeypatch,
):
    async def run():
        _enable_general_memory(monkeypatch)

        def extraction(text):
            candidate = ProfileFactCandidate(
                operation="upsert",
                category="interest",
                key="interest",
                value="摄影",
                subject="self",
                scope="stable",
                sensitivity="normal",
                evidence="我的兴趣是摄影",
            )
            return ProfileFactExtractorResult(
                status="success",
                facts=[candidate],
                candidate_count=1,
            )

        extractor = FakeExtractor(extraction)
        service = _install(monkeypatch, extractor=extractor)
        thread_id = build_qq_thread_id("dm", "user-a", "user-a")
        service.enqueue_turn(
            thread_id,
            "user",
            "我的兴趣是摄影",
            channel="qq",
            user_id="user-a",
        )
        assert await service.general_profile_facts(thread_id) == []
        service.enqueue_turn(
            thread_id,
            "assistant",
            "摄影挺有意思。",
            channel="qq",
            user_id="user-a",
        )
        await service.drain_pending(timeout_seconds=2)
        assert [item["value"] for item in await service.general_profile_facts(thread_id)] == [
            "摄影"
        ]
        assert extractor.calls == [("我的兴趣是摄影", False)]

        # 显式命令已同步处理后，assistant 后台落轮次只负责短期历史，不再调一次 Extractor。
        explicit = "请记住，我的兴趣是摄影"
        service.enqueue_turn(
            thread_id,
            "user",
            explicit,
            channel="qq",
            user_id="user-a",
        )
        result = await handle_general_profile_memory_turn(explicit, thread_id)
        assert result is not None and result.success is True
        service.enqueue_turn(
            thread_id,
            "assistant",
            "记下了。",
            channel="qq",
            user_id="user-a",
        )
        await service.drain_pending(timeout_seconds=2)
        assert [call[0] for call in extractor.calls].count(explicit) == 1

    asyncio.run(run())


def test_full_memory_clear_bypasses_generic_delete_and_removes_profile(monkeypatch):
    async def run():
        _enable_general_memory(monkeypatch)
        extractor = FakeExtractor(lambda _text: ProfileFactExtractorResult(
            status="success",
            facts=[_candidate(operation="delete", value="")],
            candidate_count=1,
        ))
        service = _install(monkeypatch, extractor=extractor)
        thread_id = build_qq_thread_id("dm", "user-a", "user-a")
        await service.update_preferred_name(
            thread_id,
            "范老师",
            expected_current=None,
        )
        await service.apply_general_profile_facts(thread_id, [_candidate()])

        assert await handle_general_profile_memory_turn("清除记忆", thread_id) is None
        raw = await agent_module.qqbot_chat("清除记忆", thread_id)
        assert "已清除长期保存" in json.loads(raw)["message"]
        profile = await service.profile_store.load("profile:qq:user-a")
        assert profile.preferred_name is None
        assert profile.long_term_facts == []
        assert extractor.calls == []

    asyncio.run(run())
