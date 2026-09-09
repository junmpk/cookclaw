import asyncio
import time

from app.conversation.models import ConversationMemory
from app.conversation.profile_facts import ProfileFactCandidate
from app.conversation.service import (
    ConversationService,
    build_qq_thread_id,
    build_weixin_thread_id,
)
from app.conversation.store import InMemoryConversationStore


def _fact(**updates) -> ProfileFactCandidate:
    data = {
        "operation": "upsert",
        "category": "work",
        "key": "occupation",
        "value": "AI应用开发工程师",
        "subject": "self",
        "scope": "stable",
        "expires_in_days": None,
        "sensitivity": "normal",
        "evidence": "我是AI应用开发工程师",
    }
    data.update(updates)
    return ProfileFactCandidate(**data)


def test_general_fact_persists_across_qq_threads_without_raw_thread_id():
    async def run():
        profile_store = InMemoryConversationStore()
        service = ConversationService(
            InMemoryConversationStore(),
            profile_store=profile_store,
        )
        group = build_qq_thread_id("group", "team", "user-a")
        dm = build_qq_thread_id("dm", "user-a", "user-a")

        applied = await service.apply_general_profile_facts(group, [_fact()])
        assert applied.changed is True
        assert len(applied.upserted) == 1

        facts = await service.general_profile_facts(dm)
        assert [(item["key"], item["value"]) for item in facts] == [
            ("occupation", "AI应用开发工程师")
        ]
        assert facts[0]["source_channel"] == "qq"
        assert "source_thread_id" not in facts[0]
        assert facts[0]["source_thread_hash"]

    asyncio.run(run())


def test_single_value_fact_is_replaced_and_old_value_is_physically_removed():
    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store, profile_store=store)
        tid = build_qq_thread_id("dm", "user-a", "user-a")
        await service.apply_general_profile_facts(tid, [_fact()])
        replacement = _fact(
            value="产品经理",
            evidence="我现在是产品经理",
        )
        result = await service.apply_general_profile_facts(tid, [replacement])

        assert result.changed is True
        facts = await service.general_profile_facts(tid)
        assert [item["value"] for item in facts] == ["产品经理"]
        profile = await store.load("profile:qq:user-a")
        assert not any(
            item.get("value") == "AI应用开发工程师"
            for item in profile.long_term_facts
        )

    asyncio.run(run())


def test_multi_value_interests_coexist_and_targeted_delete_removes_one():
    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store, profile_store=store)
        tid = build_qq_thread_id("dm", "user-a", "user-a")
        interests = [
            _fact(
                category="interest",
                key="interest",
                value=value,
                evidence=f"我喜欢{value}",
            )
            for value in ("摄影", "徒步")
        ]
        await service.apply_general_profile_facts(tid, interests[:1])
        await service.apply_general_profile_facts(tid, interests[1:])
        assert {item["value"] for item in await service.general_profile_facts(tid)} == {
            "摄影",
            "徒步",
        }

        deletion = _fact(
            operation="delete",
            category="interest",
            key="interest",
            value="摄影",
            evidence="忘掉我的摄影兴趣",
        )
        result = await service.apply_general_profile_facts(tid, [deletion])
        assert len(result.deleted_ids) == 1
        assert [item["value"] for item in await service.general_profile_facts(tid)] == [
            "徒步"
        ]

    asyncio.run(run())


def test_delete_with_empty_value_clears_the_whole_slot():
    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store, profile_store=store)
        tid = build_qq_thread_id("dm", "user-a", "user-a")
        await service.apply_general_profile_facts(tid, [_fact()])
        deletion = _fact(
            operation="delete",
            value="",
            evidence="忘掉我的职业",
        )
        result = await service.apply_general_profile_facts(tid, [deletion])
        assert result.changed is True
        assert result.deleted_ids
        assert await service.general_profile_facts(tid) == []

    asyncio.run(run())


def test_expired_general_fact_is_not_returned_and_is_purged_on_next_mutation():
    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store, profile_store=store)
        tid = build_qq_thread_id("dm", "user-a", "user-a")
        profile_key = "profile:qq:user-a"
        profile = ConversationMemory(
            thread_id=profile_key,
            channel="qq",
            user_id="user-a",
            long_term_facts=[{
                "id": "mem_expired",
                "type": "profile_fact",
                "category": "goal",
                "key": "temporary_goal",
                "value": "考架构师认证",
                "subject": "self",
                "scope": "temporal",
                "status": "active",
                "expires_at": time.time() - 1,
            }],
            version=1,
        )
        await store.save(profile, expires_at=time.time() + 3600)
        assert await service.general_profile_facts(tid) == []

        result = await service.apply_general_profile_facts(tid, [_fact()])
        assert "mem_expired" in result.deleted_ids
        saved = await store.load(profile_key)
        assert not any(item.get("id") == "mem_expired" for item in saved.long_term_facts)

    asyncio.run(run())


def test_clear_general_facts_does_not_remove_existing_food_preferences():
    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store, profile_store=store)
        tid = build_qq_thread_id("dm", "user-a", "user-a")
        await service.append_turn(
            tid,
            "user",
            "我不吃香菜",
            channel="qq",
            user_id="user-a",
        )
        await service.apply_general_profile_facts(tid, [_fact()])

        removed = await service.clear_general_profile_facts(tid)
        assert removed
        assert await service.general_profile_facts(tid) == []
        context = await service.account_memory_context(tid)
        assert context["preferences"]["dislikes"] == ["香菜"]

    asyncio.run(run())


def test_full_memory_clear_physically_removes_all_fact_values():
    async def run():
        short = InMemoryConversationStore()
        profile_store = InMemoryConversationStore()
        service = ConversationService(short, profile_store=profile_store)
        tid = build_qq_thread_id("dm", "user-a", "user-a")
        await service.apply_general_profile_facts(tid, [_fact()])
        await service.update_preferred_name(
            tid,
            "范老师",
            expected_current=None,
        )

        await service.clear(tid)
        profile = await profile_store.load("profile:qq:user-a")
        assert profile is not None
        assert profile.preferred_name is None
        assert profile.long_term_facts == []
        assert profile.summary.get("food_memory_cleared_at")

    asyncio.run(run())


def test_general_profile_facts_remain_isolated_by_channel_identity():
    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store, profile_store=store)
        qq = build_qq_thread_id("dm", "same", "same")
        weixin = build_weixin_thread_id("bot-a", "same")
        await service.apply_general_profile_facts(qq, [_fact()])
        assert await service.general_profile_facts(qq)
        assert await service.general_profile_facts(weixin) == []

    asyncio.run(run())
