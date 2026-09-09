"""单轮运行时记忆快照的加载、隔离与复用契约。"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.conversation.models import ConversationMemory, ConversationTurn
from app.conversation.runtime_memory import (
    LongTermMemorySnapshot,
    RuntimeMemorySnapshot,
)
from app.conversation.service import (
    ConversationService,
    build_qq_thread_id,
    build_weixin_thread_id,
)
from app.conversation.store import InMemoryConversationStore


class RecordingStore(InMemoryConversationStore):
    """记录读写次数，验证 hydrate 是只读的且快照不会重复读后端。"""

    def __init__(self) -> None:
        super().__init__()
        self.load_calls: list[str] = []
        self.save_calls: list[str] = []

    async def load(self, thread_id: str):
        self.load_calls.append(thread_id)
        return await super().load(thread_id)

    async def save(self, memory: ConversationMemory, expires_at: float) -> None:
        self.save_calls.append(memory.thread_id)
        await super().save(memory, expires_at)

    def reset_activity(self) -> None:
        self.load_calls.clear()
        self.save_calls.clear()


class UnavailableProfileStore(RecordingStore):
    async def load(self, thread_id: str):
        self.load_calls.append(thread_id)
        raise TimeoutError("postgres unavailable")


class UnavailableShortStore(RecordingStore):
    async def load(self, thread_id: str):
        self.load_calls.append(thread_id)
        raise TimeoutError("redis unavailable")


class InvalidProfileStore(RecordingStore):
    async def load(self, thread_id: str):
        self.load_calls.append(thread_id)
        memory = ConversationMemory(
            thread_id=thread_id,
            channel="qq",
            user_id="status-user",
        )
        memory.food_history = 1  # type: ignore[assignment]
        return memory


class TamperedShortStore(RecordingStore):
    async def load(self, thread_id: str):
        self.load_calls.append(thread_id)
        memory = ConversationMemory(
            thread_id="qq:dm:other-user:other-user",
            channel="weixin",
            user_id="other-user",
            preferences={"likes": ["不应串入"], "dislikes": []},
            recent_turns=[
                ConversationTurn(role="user", content="其他用户的对话")
            ],
        )
        memory.task_state.candidate_recipes = [
            {"cookId": "foreign", "name": "其他用户候选"}
        ]
        return memory


def _general_fact(
    fact_id: str,
    value: str,
    *,
    updated_at: float,
    expires_at: float = 0.0,
    status: str = "active",
) -> dict:
    return {
        "id": fact_id,
        "type": "profile_fact",
        "category": "work",
        "key": "occupation",
        "value": value,
        "subject": "self",
        "scope": "stable",
        "status": status,
        "source": "user_explicit",
        "updated_at": updated_at,
        "expires_at": expires_at,
    }


def _temporal_dietary_fact(
    fact_id: str,
    value: str,
    *,
    updated_at: float,
    expires_at: float,
) -> dict:
    return {
        "id": fact_id,
        "type": "dietary_constraint",
        "value": value,
        "status": "active",
        "source": "user_explicit",
        "last_confirmed_at": updated_at,
        "updated_at": updated_at,
        "expires_at": expires_at,
    }


async def _seed_profile(
    store: RecordingStore,
    profile_key: str,
    *,
    channel: str,
    user_id: str,
    preferred_name: str | None = None,
    preferences: dict[str, list[str]] | None = None,
    facts: list[dict] | None = None,
    version: int = 1,
) -> None:
    now = time.time()
    await store.save(
        ConversationMemory(
            thread_id=profile_key,
            channel=channel,
            user_id=user_id,
            preferred_name=preferred_name,
            preferences=preferences
            or {
                "likes": [],
                "dislikes": [],
                "allergens": [],
                "dietary_constraints": [],
                "available_ingredients": [],
            },
            long_term_facts=list(facts or []),
            version=version,
            updated_at=now,
        ),
        expires_at=now + 3600,
    )
    store.reset_activity()


def test_new_thread_hydrates_account_memory_without_writing_short_term_store():
    async def run() -> None:
        now = time.time()
        short_store = RecordingStore()
        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:user-a",
            channel="qq",
            user_id="user-a",
            preferred_name="范老师",
            preferences={
                "likes": ["鸡肉"],
                "dislikes": ["香菜"],
                "allergens": [],
                "dietary_constraints": ["减脂"],
                # 模拟旧版 PG 遗留；运行时长期投影必须丢弃这类会话条件。
                "available_ingredients": ["隔夜鸡蛋"],
            },
            facts=[
                _temporal_dietary_fact(
                    "diet-active",
                    "减脂",
                    updated_at=now - 60,
                    expires_at=now + 3600,
                ),
                _general_fact(
                    "fact-job",
                    "AI应用开发工程师",
                    updated_at=now - 30,
                ),
            ],
            version=7,
        )
        service = ConversationService(short_store, profile_store=profile_store)
        new_thread = build_qq_thread_id("dm", "user-a", "user-a")

        snapshot = await service.load_runtime_memory(new_thread)

        assert isinstance(snapshot, RuntimeMemorySnapshot)
        assert isinstance(snapshot.long_term, LongTermMemorySnapshot)
        assert snapshot.short_term.thread_id == new_thread
        assert snapshot.short_term.recent_turns == []
        assert snapshot.short_term_status == "new"
        assert snapshot.is_new_session is True
        assert snapshot.long_term.status == "loaded"
        assert snapshot.long_term.profile_version == 7
        assert snapshot.long_term.preferred_name == "范老师"
        assert snapshot.long_term.preferences["likes"] == ["鸡肉"]
        assert snapshot.long_term.preferences["dislikes"] == ["香菜"]
        assert snapshot.long_term.preferences["dietary_constraints"] == []
        assert snapshot.long_term.preferences["available_ingredients"] == []
        assert [
            item["value"]
            for item in snapshot.long_term.temporal_dietary_constraints
        ] == ["减脂"]
        assert [item["id"] for item in snapshot.long_term.general_facts] == [
            "fact-job"
        ]

        assert short_store.load_calls == [new_thread]
        assert short_store.save_calls == []
        assert profile_store.load_calls == ["profile:qq:user-a"]
        assert profile_store.save_calls == []

        ref = snapshot.context_ref()
        assert ref["short_term_status"] == "new"
        assert ref["long_term_status"] == "loaded"
        assert ref["long_term_profile_version"] == 7
        assert ref["long_term_general_fact_count"] == 1
        assert "范老师" not in repr(ref)
        assert "AI应用开发工程师" not in repr(ref)
        assert new_thread not in repr(ref)
        assert "范老师" not in repr(snapshot)
        assert "AI应用开发工程师" not in repr(snapshot)
        assert "user-a" not in repr(snapshot)
        assert new_thread not in repr(snapshot)

    asyncio.run(run())


def test_web_runtime_restores_session_without_touching_account_profile():
    async def run() -> None:
        short_store = RecordingStore()
        profile_store = RecordingStore()
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )
        thread_id = "web:qq:dm:forged-user:forged-user"

        await service.append_turn(
            thread_id,
            "user",
            "我最近在学 Python",
            channel="web",
            persist_account_memory=False,
        )
        await service.append_turn(
            thread_id,
            "assistant",
            "你学到哪一部分了？",
            channel="web",
            persist_account_memory=False,
        )
        profile_store.reset_activity()

        recreated = ConversationService(
            short_store,
            profile_store=profile_store,
        )
        snapshot = await recreated.load_runtime_memory(
            thread_id,
            channel="web",
            current_message="装饰器是什么？",
        )

        assert snapshot.channel == "web"
        assert snapshot.user_id is None
        assert snapshot.short_term_status == "loaded"
        assert snapshot.long_term.status == "unsupported"
        assert [turn.content for turn in snapshot.short_term.recent_turns] == [
            "我最近在学 Python",
            "你学到哪一部分了？",
        ]
        assert profile_store.load_calls == []
        assert profile_store.save_calls == []

        with pytest.raises(ValueError, match="cannot accept an account user id"):
            await recreated.load_runtime_memory(
                thread_id,
                channel="web",
                user_id="forged-user",
            )

    asyncio.run(run())


def test_reset_session_clears_short_term_then_rehydrates_long_term_profile():
    async def run() -> None:
        now = time.time()
        thread_id = build_qq_thread_id(
            "dm",
            "reset-user",
            "reset-user",
        )
        short_store = RecordingStore()
        old_memory = ConversationMemory(
            thread_id=thread_id,
            channel="qq",
            user_id="reset-user",
            recent_turns=[
                ConversationTurn(role="user", content="旧会话内容")
            ],
            version=3,
        )
        old_memory.task_state.candidate_recipes = [
            {"cookId": "old-recipe", "name": "旧候选"}
        ]
        old_memory.task_state.active_cooking = {
            "cookId": "running-recipe",
            "name": "正在烹饪",
        }
        await short_store.save(old_memory, expires_at=now + 3600)
        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:reset-user",
            channel="qq",
            user_id="reset-user",
            preferred_name="重开用户",
            version=6,
        )
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )

        await service.reset_session(thread_id)
        short_store.reset_activity()
        profile_store.reset_activity()
        snapshot = await service.load_runtime_memory(thread_id)

        assert snapshot.short_term_status == "loaded"
        assert snapshot.is_new_session is False
        assert snapshot.short_term.recent_turns == []
        assert snapshot.short_term.task_state.candidate_recipes == []
        assert snapshot.short_term.task_state.active_cooking == {
            "cookId": "running-recipe",
            "name": "正在烹饪",
        }
        assert snapshot.long_term.status == "loaded"
        assert snapshot.long_term.preferred_name == "重开用户"
        assert snapshot.long_term.profile_version == 6

    asyncio.run(run())


def test_runtime_memory_isolates_users_and_weixin_bot_accounts():
    async def run() -> None:
        short_store = RecordingStore()
        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:user-a",
            channel="qq",
            user_id="user-a",
            preferred_name="QQ-A",
        )
        await _seed_profile(
            profile_store,
            "profile:weixin:bot-a:shared-user",
            channel="weixin",
            user_id="shared-user",
            preferred_name="微信-A",
        )
        service = ConversationService(short_store, profile_store=profile_store)

        qq_a = await service.load_runtime_memory(
            build_qq_thread_id("group", "room", "user-a")
        )
        qq_b = await service.load_runtime_memory(
            build_qq_thread_id("group", "room", "user-b")
        )
        weixin_a = await service.load_runtime_memory(
            build_weixin_thread_id("bot-a", "shared-user")
        )
        weixin_b = await service.load_runtime_memory(
            build_weixin_thread_id("bot-b", "shared-user")
        )

        assert qq_a.long_term.preferred_name == "QQ-A"
        assert qq_a.long_term.status == "loaded"
        assert qq_b.long_term.preferred_name is None
        assert qq_b.long_term.status == "empty"
        assert weixin_a.long_term.preferred_name == "微信-A"
        assert weixin_a.long_term.status == "loaded"
        assert weixin_b.long_term.preferred_name is None
        assert weixin_b.long_term.status == "empty"

    asyncio.run(run())


def test_runtime_memory_derives_identity_from_thread_not_short_payload():
    async def run() -> None:
        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:expected-user",
            channel="qq",
            user_id="expected-user",
            preferred_name="正确账号",
        )
        await _seed_profile(
            profile_store,
            "profile:weixin:dm:other-user",
            channel="weixin",
            user_id="other-user",
            preferred_name="错误账号",
        )
        service = ConversationService(
            TamperedShortStore(),
            profile_store=profile_store,
        )
        thread_id = build_qq_thread_id(
            "dm",
            "expected-user",
            "expected-user",
        )

        snapshot = await service.load_runtime_memory(thread_id)

        assert snapshot.thread_id == thread_id
        assert snapshot.channel == "qq"
        assert snapshot.user_id == "expected-user"
        assert snapshot.short_term.thread_id == thread_id
        assert snapshot.short_term.channel == "qq"
        assert snapshot.short_term.user_id == "expected-user"
        assert snapshot.short_term_status == "invalid"
        assert snapshot.is_new_session is None
        assert snapshot.short_term.preferences["likes"] == []
        assert snapshot.short_term.recent_turns == []
        assert snapshot.short_term.task_state.candidate_recipes == []
        assert snapshot.long_term.preferred_name == "正确账号"
        assert snapshot.long_term.preferences["likes"] == []
        assert profile_store.load_calls == ["profile:qq:expected-user"]

    asyncio.run(run())


def test_runtime_memory_filters_expired_general_and_temporal_facts():
    async def run() -> None:
        now = time.time()
        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:expiry-user",
            channel="qq",
            user_id="expiry-user",
            preferences={
                "likes": [],
                "dislikes": [],
                "allergens": [],
                "dietary_constraints": ["减脂", "控卡"],
                "available_ingredients": [],
            },
            facts=[
                _general_fact(
                    "fact-active",
                    "产品经理",
                    updated_at=now - 10,
                ),
                _general_fact(
                    "fact-expired",
                    "旧职业",
                    updated_at=now - 20,
                    expires_at=now - 1,
                ),
                _general_fact(
                    "fact-deleted",
                    "已删除事实",
                    updated_at=now - 30,
                    status="deleted",
                ),
                {
                    **_general_fact(
                        "fact-invalid-expiry",
                        "过期时间损坏的事实",
                        updated_at=now - 40,
                    ),
                    "expires_at": "not-a-timestamp",
                },
                _temporal_dietary_fact(
                    "diet-active",
                    "减脂",
                    updated_at=now - 30,
                    expires_at=now + 3600,
                ),
                {
                    **_temporal_dietary_fact(
                        "diet-old-copy",
                        "减脂",
                        updated_at=now - 300,
                        expires_at=now - 200,
                    ),
                    "status": "deleted",
                },
                _temporal_dietary_fact(
                    "diet-expired",
                    "控卡",
                    updated_at=now - 90,
                    expires_at=now - 1,
                ),
            ],
            version=4,
        )
        service = ConversationService(
            RecordingStore(),
            profile_store=profile_store,
        )
        thread_id = build_qq_thread_id(
            "dm",
            "expiry-user",
            "expiry-user",
        )

        snapshot = await service.load_runtime_memory(thread_id)

        assert [item["id"] for item in snapshot.long_term.general_facts] == [
            "fact-active"
        ]
        assert [
            item["value"]
            for item in snapshot.long_term.temporal_dietary_constraints
        ] == ["减脂"]
        assert snapshot.long_term.preferences["dietary_constraints"] == []
        assert await service.general_profile_facts(
            thread_id,
            runtime_memory=snapshot,
        ) == snapshot.long_term.general_facts

    asyncio.run(run())


def test_runtime_memory_distinguishes_empty_unavailable_and_invalid_profiles():
    async def run() -> None:
        thread_id = build_qq_thread_id("dm", "status-user", "status-user")
        empty_service = ConversationService(
            RecordingStore(),
            profile_store=RecordingStore(),
        )
        unavailable_store = UnavailableProfileStore()
        unavailable_service = ConversationService(
            RecordingStore(),
            profile_store=unavailable_store,
            circuit_cooldown_seconds=1,
        )
        invalid_store = InvalidProfileStore()
        invalid_service = ConversationService(
            RecordingStore(),
            profile_store=invalid_store,
        )

        empty = await empty_service.load_runtime_memory(thread_id)
        unavailable = await unavailable_service.load_runtime_memory(thread_id)
        invalid = await invalid_service.load_runtime_memory(thread_id)

        assert empty.long_term.status == "empty"
        assert empty.long_term.profile_version == 0
        assert empty.long_term.general_facts == []
        assert unavailable.long_term.status == "unavailable"
        assert unavailable.long_term.profile_version == 0
        assert unavailable.long_term.general_facts == []
        assert unavailable.is_new_session is True
        assert unavailable_store.load_calls == ["profile:qq:status-user"]
        assert invalid.long_term.status == "invalid"
        assert invalid.long_term.general_facts == []
        assert invalid.is_new_session is True
        assert invalid_store.load_calls == ["profile:qq:status-user"]

    asyncio.run(run())


def test_runtime_memory_loads_long_term_when_short_term_backend_is_unavailable():
    async def run() -> None:
        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:short-down-user",
            channel="qq",
            user_id="short-down-user",
            preferred_name="长期仍可读",
            version=8,
        )
        short_store = UnavailableShortStore()
        service = ConversationService(
            short_store,
            profile_store=profile_store,
            circuit_cooldown_seconds=1,
        )
        thread_id = build_qq_thread_id(
            "dm",
            "short-down-user",
            "short-down-user",
        )

        snapshot = await service.load_runtime_memory(thread_id)

        assert snapshot.short_term_status == "unavailable"
        assert snapshot.is_new_session is None
        assert snapshot.short_term.recent_turns == []
        assert snapshot.long_term.status == "loaded"
        assert snapshot.long_term.preferred_name == "长期仍可读"
        assert profile_store.load_calls == ["profile:qq:short-down-user"]

    asyncio.run(run())


def test_cleared_profile_does_not_revive_preferences_from_an_old_thread():
    async def run() -> None:
        now = time.time()
        short_store = RecordingStore()
        profile_store = RecordingStore()
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )
        clear_thread = build_qq_thread_id(
            "dm",
            "clear-user",
            "clear-user",
        )
        old_thread = build_qq_thread_id(
            "group",
            "old-room",
            "clear-user",
        )
        await short_store.save(
            ConversationMemory(
                thread_id=clear_thread,
                channel="qq",
                user_id="clear-user",
                created_at=now - 3600,
                updated_at=now - 60,
                version=1,
            ),
            expires_at=now + 3600,
        )
        await _seed_profile(
            profile_store,
            "profile:qq:clear-user",
            channel="qq",
            user_id="clear-user",
            preferences={
                "likes": ["香菜"],
                "dislikes": [],
                "allergens": [],
                "dietary_constraints": [],
                "available_ingredients": [],
            },
            facts=[
                {
                    "id": "like-coriander",
                    "type": "preference_like",
                    "value": "香菜",
                    "status": "active",
                    "updated_at": now - 60,
                }
            ],
            version=2,
        )

        await service.clear(clear_thread)
        cleared_at = time.time()
        await short_store.save(
            ConversationMemory(
                thread_id=old_thread,
                channel="qq",
                user_id="clear-user",
                preferences={
                    "likes": ["香菜"],
                    "dislikes": [],
                    "allergens": [],
                    "dietary_constraints": [],
                    "available_ingredients": [],
                },
                # 模拟清除前创建、清除后又收到普通消息的旧群聊 thread。
                created_at=now - 7200,
                updated_at=cleared_at + 1,
                version=4,
            ),
            expires_at=now + 3600,
        )
        short_store.reset_activity()
        profile_store.reset_activity()

        snapshot = await service.load_runtime_memory(
            old_thread,
            current_message="随便推荐一道菜",
        )
        context = await service.routing_context(
            old_thread,
            scope="full",
            current_message="随便推荐一道菜",
            runtime_memory=snapshot,
        )
        planner_view = await service.planner_memory_view(
            old_thread,
            current_message="随便推荐一道菜",
            runtime_memory=snapshot,
        )
        assert context["preferences"]["likes"] == []
        assert planner_view.policy_resolved is True
        assert planner_view.effective_preferences.likes == ()
        assert snapshot.long_term.status == "empty"
        assert snapshot.long_term.food_memory_cleared_at > 0

        reaffirmed = await service.load_runtime_memory(
            old_thread,
            current_message="我现在喜欢鸡肉，推荐一道菜",
        )
        reaffirmed_context = await service.routing_context(
            old_thread,
            scope="full",
            current_message="我现在喜欢鸡肉，推荐一道菜",
            runtime_memory=reaffirmed,
        )
        reaffirmed_view = await service.planner_memory_view(
            old_thread,
            current_message="我现在喜欢鸡肉，推荐一道菜",
            runtime_memory=reaffirmed,
        )
        assert reaffirmed_context["preferences"]["likes"] == ["鸡肉"]
        assert reaffirmed_view.effective_preferences.likes == ("鸡肉",)
        assert "香菜" not in repr(reaffirmed_context)

    asyncio.run(run())


def test_current_turn_correction_overrides_loaded_long_term_preference():
    async def run() -> None:
        now = time.time()
        thread_id = build_qq_thread_id(
            "dm",
            "correction-user",
            "correction-user",
        )
        short_store = RecordingStore()
        await short_store.save(
            ConversationMemory(
                thread_id=thread_id,
                channel="qq",
                user_id="correction-user",
                preferences={
                    "likes": [],
                    "dislikes": ["香菜"],
                    "allergens": [],
                    "dietary_constraints": [],
                    "available_ingredients": [],
                },
                version=3,
            ),
            expires_at=now + 3600,
        )
        short_store.reset_activity()
        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:correction-user",
            channel="qq",
            user_id="correction-user",
            preferences={
                "likes": [],
                "dislikes": ["香菜"],
                "allergens": [],
                "dietary_constraints": [],
                "available_ingredients": [],
            },
            facts=[
                {
                    "id": "dislike-coriander",
                    "type": "preference_dislike",
                    "value": "香菜",
                    "status": "active",
                    "updated_at": now,
                }
            ],
            version=4,
        )
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )
        question = "我没有不喜欢香菜了，帮我推荐一道菜"

        snapshot = await service.load_runtime_memory(
            thread_id,
            current_message=question,
        )
        context = await service.routing_context(
            thread_id,
            scope="full",
            current_message=question,
            runtime_memory=snapshot,
        )
        preference_context = await service.routing_context(
            thread_id,
            scope="preferences",
            current_message=question,
            runtime_memory=snapshot,
        )
        safety_view = await service.safety_memory_view(
            thread_id,
            current_message=question,
            runtime_memory=snapshot,
        )
        planner_view = await service.planner_memory_view(
            thread_id,
            current_message=question,
            runtime_memory=snapshot,
        )
        qa_view = await service.qa_memory_view(
            thread_id,
            current_message=question,
            runtime_memory=snapshot,
        )

        assert context["preferences"]["dislikes"] == []
        assert preference_context["preferences"]["dislikes"] == []
        assert safety_view.effective_preferences.dislikes == ()
        assert planner_view.effective_preferences.dislikes == ()
        assert qa_view.effective_preferences.dislikes == ()

    asyncio.run(run())


def test_runtime_snapshot_is_reused_without_reading_stores_again():
    async def run() -> None:
        now = time.time()
        thread_id = build_qq_thread_id("dm", "reuse-user", "reuse-user")
        short_store = RecordingStore()
        profile_store = RecordingStore()
        await short_store.save(
            ConversationMemory(
                thread_id=thread_id,
                channel="qq",
                user_id="reuse-user",
                recent_turns=[ConversationTurn(role="user", content="我做什么工作")],
                version=3,
            ),
            expires_at=now + 3600,
        )
        short_store.reset_activity()
        await _seed_profile(
            profile_store,
            "profile:qq:reuse-user",
            channel="qq",
            user_id="reuse-user",
            preferred_name="复用用户",
            preferences={
                "likes": ["清淡"],
                "dislikes": [],
                "allergens": [],
                "dietary_constraints": [],
                "available_ingredients": [],
            },
            facts=[
                _general_fact(
                    "fact-reuse",
                    "后端工程师",
                    updated_at=now,
                )
            ],
            version=9,
        )
        service = ConversationService(short_store, profile_store=profile_store)

        snapshot = await service.load_runtime_memory(thread_id)
        context = await service.routing_context(
            thread_id,
            scope="full",
            current_message="你还记得吗",
            runtime_memory=snapshot,
        )
        preference_context = await service.routing_context(
            thread_id,
            scope="preferences",
            runtime_memory=snapshot,
        )
        route_context = await service.routing_context(
            thread_id,
            scope="route",
            runtime_memory=snapshot,
        )
        safety_view = await service.safety_memory_view(
            thread_id,
            runtime_memory=snapshot,
        )
        planner_view = await service.planner_memory_view(
            thread_id,
            runtime_memory=snapshot,
        )
        qa_view = await service.qa_memory_view(
            thread_id,
            runtime_memory=snapshot,
        )
        facts = await service.general_profile_facts(
            thread_id,
            runtime_memory=snapshot,
        )
        recalled = await service.recall_general_profile_facts(
            thread_id,
            "我的职业",
            limit=1,
            runtime_memory=snapshot,
        )

        assert snapshot.short_term_status == "loaded"
        assert snapshot.is_new_session is False
        assert context["preferred_name"] == "复用用户"
        assert context["preferences"]["likes"] == ["清淡"]
        assert context["recent_user_turns"] == ["我做什么工作"]
        assert safety_view.effective_preferences.likes == ("清淡",)
        assert planner_view.effective_preferences.likes == ("清淡",)
        assert qa_view.effective_preferences.likes == ("清淡",)
        assert qa_view.preferred_name == "复用用户"
        assert (
            safety_view.to_router_context_dict()["preferences"]
            == preference_context["preferences"]
        )
        assert (
            planner_view.to_planner_context_dict()["preferences"]
            == route_context["preferences"]
        )
        assert list(planner_view.recent_user_turns) == route_context[
            "recent_user_turns"
        ]
        assert (
            qa_view.to_qa_context_dict()["preferences"]
            == context["preferences"]
        )
        context["recent_turns"][0].content = "污染共享快照"
        fresh_context = await service.routing_context(
            thread_id,
            scope="full",
            runtime_memory=snapshot,
        )
        fresh_qa_view = await service.qa_memory_view(
            thread_id,
            runtime_memory=snapshot,
        )
        assert fresh_context["recent_turns"][0].content == "我做什么工作"
        assert fresh_qa_view.recent_turns[0].content == "我做什么工作"
        assert [item["id"] for item in facts] == ["fact-reuse"]
        assert [item["id"] for item in recalled] == ["fact-reuse"]
        assert "general_facts" not in context
        assert "后端工程师" not in repr(context)
        assert short_store.load_calls == [thread_id]
        assert profile_store.load_calls == ["profile:qq:reuse-user"]

    asyncio.run(run())


def test_im_turn_runtime_hydrates_once_and_reuses_snapshot(monkeypatch):
    async def run() -> None:
        import app.agent.participle_agent as agent_module
        import app.conversation.service as service_module
        from app.conversation.task_state_repository import (
            RedisDialogueTaskStateRepository,
        )

        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:runtime-user",
            channel="qq",
            user_id="runtime-user",
            preferred_name="运行时用户",
            preferences={
                "likes": ["清淡"],
                "dislikes": [],
                "allergens": [],
                "dietary_constraints": [],
                "available_ingredients": [],
            },
            version=5,
        )
        short_store = RecordingStore()
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )
        monkeypatch.setattr(service_module, "_service", service)
        monkeypatch.setattr(
            agent_module,
            "profile_memory_enabled",
            lambda _thread_id: True,
        )
        monkeypatch.setattr(
            agent_module,
            "general_profile_memory_enabled",
            lambda _thread_id: False,
        )
        thread_id = build_qq_thread_id(
            "dm",
            "runtime-user",
            "runtime-user",
        )
        async with RedisDialogueTaskStateRepository().turn_scope(
            thread_id
        ) as short_term_snapshot:
            runtime = await agent_module._build_qq_turn_runtime(
                "你还记得我吗",
                thread_id=thread_id,
                source_message_type="text",
                short_term_snapshot=short_term_snapshot,
            )
        assert runtime.memory_snapshot is not None
        assert runtime.memory_snapshot.is_new_session is True
        assert runtime.memory_snapshot.long_term.preferred_name == "运行时用户"

        context = await agent_module._conversation_context_for_qa(
            thread_id,
            "你还记得我吗",
            runtime_memory=runtime.memory_snapshot,
        )

        assert context is not None
        assert "运行时用户" in context
        assert "清淡" in context
        assert short_store.load_calls == [thread_id]
        assert profile_store.load_calls == ["profile:qq:runtime-user"]

    asyncio.run(run())


def test_memory_reset_commands_run_before_runtime_hydration(monkeypatch):
    async def run() -> None:
        import app.agent.participle_agent as agent_module

        calls: list[tuple[str, str]] = []

        async def fake_reset(thread_id: str, question: str):
            calls.append((thread_id, question))
            return "新的短期会话已经开始。", "zh"

        async def forbidden_runtime(*_args, **_kwargs):
            raise AssertionError("记忆重置命令不应先构造运行时记忆快照")

        async def forbidden_prelude(*_args, **_kwargs):
            raise AssertionError("记忆重置命令不应先进入记忆读写 prelude")

        monkeypatch.setattr(
            agent_module,
            "_reset_conversation_state",
            fake_reset,
        )
        monkeypatch.setattr(
            agent_module,
            "_build_qq_turn_runtime",
            forbidden_runtime,
        )
        monkeypatch.setattr(
            agent_module,
            "_run_im_memory_prelude",
            forbidden_prelude,
        )
        thread_id = build_qq_thread_id(
            "dm",
            "reset-before-hydrate",
            "reset-before-hydrate",
        )

        for command in ("给我一个新对话", "清空会话记录", "清除记忆"):
            envelope = await agent_module._run_turn_orchestrator(
                command,
                thread_id=thread_id,
                channel="qq",
                source_message_type="text",
                trace_id="reset-before-hydrate-trace",
            )

            assert envelope.handled_by == "exact_command"
            assert envelope.message == "新的短期会话已经开始。"
        assert calls == [
            (thread_id, "给我一个新对话"),
            (thread_id, "清空会话记录"),
            (thread_id, "清除记忆"),
        ]

    asyncio.run(run())


def test_explicit_chat_history_commands_use_session_reset_without_clearing_profile(
    monkeypatch,
):
    async def run() -> None:
        import app.agent.participle_agent as agent_module
        import app.conversation.service as service_module

        reset_calls: list[str] = []

        class ResetService:
            async def reset_session(self, thread_id: str):
                reset_calls.append(thread_id)
                return None

            async def clear(self, _thread_id: str):
                raise AssertionError("清空会话记录不得删除长期画像")

        thread_id = build_qq_thread_id(
            "dm",
            "session-history-reset",
            "session-history-reset",
        )
        monkeypatch.setattr(
            agent_module,
            "supports_session_memory",
            lambda _thread_id: True,
        )
        monkeypatch.setattr(
            agent_module,
            "supports_account_profile",
            lambda _thread_id: True,
        )
        monkeypatch.setattr(
            service_module,
            "get_conversation_service",
            lambda: ResetService(),
        )

        explicit_commands = (
            "清空会话记录",
            "清除会话记录。",
            "清空聊天记录",
            "清除聊天记录！",
        )
        for command in explicit_commands:
            reply = await agent_module._reset_conversation_state(thread_id, command)
            assert reply is not None
            assert "长期保存的偏好仍然保留" in reply[0]

        assert reset_calls == [thread_id] * len(explicit_commands)

        # 只接受完整、明确的命令，描述或否定句不能误触发清空。
        assert (
            await agent_module._reset_conversation_state(
                thread_id,
                "怎么清空会话记录？",
            )
            is None
        )
        assert (
            await agent_module._reset_conversation_state(
                thread_id,
                "不要清空会话记录",
            )
            is None
        )
        assert reset_calls == [thread_id] * len(explicit_commands)

    asyncio.run(run())


def test_tampered_short_payload_never_writes_back_from_task_scope(monkeypatch):
    async def run() -> None:
        import app.conversation.service as service_module
        from app.conversation.service import ConversationBackendUnavailable
        from app.conversation.task_state_repository import (
            RedisDialogueTaskStateRepository,
        )
        from app.ports.dialogue_state import remember_candidates

        short_store = TamperedShortStore()
        service = ConversationService(
            short_store,
            profile_store=RecordingStore(),
        )
        monkeypatch.setattr(service_module, "_service", service)
        thread_id = build_qq_thread_id(
            "dm",
            "scope-owner",
            "scope-owner",
        )
        with pytest.raises(ConversationBackendUnavailable):
            async with RedisDialogueTaskStateRepository().turn_scope(thread_id):
                remember_candidates(
                    thread_id,
                    {
                        "results": [
                            {
                                "metadata": {
                                    "recipe_id": "local-safe",
                                    "name": "本地候选",
                                }
                            }
                        ]
                    },
                    lang="zh",
                )

        assert short_store.load_calls == [thread_id]
        assert short_store.save_calls == []

    asyncio.run(run())


def test_profile_query_reuses_scope_snapshot_for_one_short_and_profile_read(
    monkeypatch,
):
    async def run() -> None:
        import app.conversation.profile_command as profile_command_module
        import app.conversation.service as service_module
        from app.conversation.task_state_repository import (
            RedisDialogueTaskStateRepository,
        )

        thread_id = build_qq_thread_id(
            "dm",
            "profile-query-user",
            "profile-query-user",
        )
        short_store = RecordingStore()
        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:profile-query-user",
            channel="qq",
            user_id="profile-query-user",
            preferred_name="范老师",
            version=4,
        )
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )
        monkeypatch.setattr(service_module, "_service", service)
        monkeypatch.setattr(
            profile_command_module,
            "profile_memory_enabled",
            lambda _thread_id: True,
        )
        async with RedisDialogueTaskStateRepository().turn_scope(
            thread_id
        ) as short_term_snapshot:
            reply = await profile_command_module.handle_profile_memory_turn(
                "你该叫我什么？",
                thread_id,
                short_term_snapshot=short_term_snapshot,
            )

        assert reply is not None
        assert reply.status == "found"
        assert "范老师" in reply.message
        assert short_store.load_calls == [thread_id]
        assert profile_store.load_calls == [
            "profile:qq:profile-query-user"
        ]

    asyncio.run(run())


def test_memory_queries_do_not_report_empty_when_profile_store_is_unavailable(
    monkeypatch,
):
    async def run() -> None:
        import app.conversation.preference_command as preference_command_module
        import app.conversation.profile_command as profile_command_module
        import app.conversation.service as service_module

        monkeypatch.setattr(
            profile_command_module,
            "profile_memory_enabled",
            lambda _thread_id: True,
        )
        thread_id = build_qq_thread_id(
            "dm",
            "profile-down-user",
            "profile-down-user",
        )

        for question, handler in (
            ("你记住了我哪些偏好", preference_command_module.handle_preference_memory_turn),
            ("你该叫我什么？", profile_command_module.handle_profile_memory_turn),
            ("清除我的称呼", profile_command_module.handle_profile_memory_turn),
        ):
            profile_store = UnavailableProfileStore()
            service = ConversationService(
                RecordingStore(),
                profile_store=profile_store,
            )
            monkeypatch.setattr(service_module, "_service", service)

            reply = await handler(question, thread_id)

            assert reply is not None
            assert reply.status == "read_failed"
            assert reply.success is False
            assert "没有保存" not in reply.message
            assert "原本就没有" not in reply.message
            assert profile_store.load_calls == [
                "profile:qq:profile-down-user"
            ]

    asyncio.run(run())


def test_candidate_and_correction_followups_reuse_loaded_runtime_snapshot(
    monkeypatch,
):
    async def run() -> None:
        import app.agent.participle_agent as agent_module
        import app.conversation.service as service_module
        from app.conversation import task_state_workspace as session

        thread_id = build_qq_thread_id(
            "dm",
            "followup-reuse-user",
            "followup-reuse-user",
        )
        now = time.time()
        short_store = RecordingStore()
        memory = ConversationMemory(
            thread_id=thread_id,
            channel="qq",
            user_id="followup-reuse-user",
            version=2,
        )
        memory.latest_search = {
            "original_question": "想吃清淡的菜",
            "search_query": "清淡",
            "recipes": [
                {"id": "r1", "name": "清蒸鲈鱼", "score": 0.9}
            ],
            "candidate_pool": [
                {"id": "r1", "name": "清蒸鲈鱼", "score": 0.9}
            ],
        }
        await short_store.save(memory, expires_at=now + 3600)
        short_store.reset_activity()
        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:followup-reuse-user",
            channel="qq",
            user_id="followup-reuse-user",
            version=1,
        )
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )
        monkeypatch.setattr(service_module, "_service", service)
        session.clear_thread(thread_id)

        runtime = await service.load_runtime_memory(thread_id)
        resumed = await agent_module._resume_recipe_task(
            thread_id,
            "继续刚才的菜",
            runtime_memory=runtime,
        )
        omitted = await agent_module._omitted_recipe_followup(
            thread_id,
            "为什么没有推荐麻婆豆腐",
            runtime_memory=runtime,
        )
        correction = await agent_module._memory_correction_followup(
            thread_id,
            "我没说过要素食",
            runtime_memory=runtime,
        )

        assert resumed is not None
        assert omitted is not None
        assert correction is not None
        assert short_store.load_calls == [thread_id]
        assert profile_store.load_calls == [
            "profile:qq:followup-reuse-user"
        ]
        session.clear_thread(thread_id)

    asyncio.run(run())


def test_name_change_fails_closed_when_short_term_store_is_unavailable(
    monkeypatch,
):
    async def run() -> None:
        import app.conversation.profile_command as profile_command_module
        import app.conversation.service as service_module

        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:name-change-short-down",
            channel="qq",
            user_id="name-change-short-down",
            preferred_name="原称呼",
            version=2,
        )
        service = ConversationService(
            UnavailableShortStore(),
            profile_store=profile_store,
        )
        monkeypatch.setattr(service_module, "_service", service)
        monkeypatch.setattr(
            profile_command_module,
            "profile_memory_enabled",
            lambda _thread_id: True,
        )
        thread_id = build_qq_thread_id(
            "dm",
            "name-change-short-down",
            "name-change-short-down",
        )

        reply = await profile_command_module.handle_profile_memory_turn(
            "以后叫我新称呼",
            thread_id,
        )

        assert reply is not None
        assert reply.status == "write_failed"
        assert reply.success is False
        assert "没有修改" in reply.message or "没有改动" in reply.message
        profile = await profile_store.load(
            "profile:qq:name-change-short-down"
        )
        assert profile is not None
        assert profile.preferred_name == "原称呼"

    asyncio.run(run())


def test_recipe_route_requires_current_safety_facts_when_profile_is_unreadable(
    monkeypatch,
):
    async def run() -> None:
        import app.orchestrator.router as router_module
        from app.orchestrator.intent import IntentCategory, IntentResult
        from app.orchestrator.search_request import SearchRequest

        loaded_scopes: list[str] = []
        search_calls: list[str] = []

        async def unavailable_memory(scope: str) -> dict:
            loaded_scopes.append(scope)
            return {
                "preferences": {},
                "temporal_dietary_constraints": [],
                "long_term_memory_status": "unavailable",
            }

        async def fake_search(query: str, **_kwargs):
            search_calls.append(query)
            return {"success": False, "results": []}

        monkeypatch.setattr(
            router_module,
            "_run_search_subprocess",
            fake_search,
        )
        unknown_safety = SearchRequest(
            original_text="红烧肉怎么做",
            query="红烧肉",
            dishes=["红烧肉"],
            task_operation="exact_search",
        )
        blocked = await router_module.route_fast_path(
            "红烧肉怎么做",
            memory_context_loader=unavailable_memory,
            intent_override=IntentResult(
                related=True,
                category=IntentCategory.recipe_search,
                keywords=["红烧肉"],
                search_request=unknown_safety,
            ),
        )

        assert blocked.kind == "clarify"
        assert blocked.clarification_dimension == "party_constraints"
        assert "读不到" in blocked.clarification_message
        assert search_calls == []

        confirmed_safety = unknown_safety.model_copy(update={
            "original_text": "这次没有过敏和忌口，红烧肉怎么做",
            "constraints_confirmed": True,
        })
        continued = await router_module.route_fast_path(
            "这次没有过敏和忌口，红烧肉怎么做",
            memory_context_loader=unavailable_memory,
            intent_override=IntentResult(
                related=True,
                category=IntentCategory.recipe_search,
                keywords=["红烧肉"],
                search_request=confirmed_safety,
            ),
        )

        assert continued.kind != "clarify"
        assert search_calls
        assert loaded_scopes == ["preferences", "preferences"]

    asyncio.run(run())


def test_full_memory_clear_preserves_only_active_device_state():
    async def run() -> None:
        thread_id = build_qq_thread_id(
            "dm",
            "clear-active-user",
            "clear-active-user",
        )
        short_store = RecordingStore()
        profile_store = RecordingStore()
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )
        memory = ConversationMemory(
            thread_id=thread_id,
            channel="qq",
            user_id="clear-active-user",
            recent_turns=[ConversationTurn(role="user", content="旧对话")],
            preferences={"likes": ["鸡肉"]},
        )
        memory.task_state.candidate_recipes = [
            {"cookId": "old-candidate", "name": "旧候选"}
        ]
        memory.task_state.active_cooking = {
            "cookId": "running-recipe",
            "name": "正在烹饪的菜",
            "device_id": "office",
            "lang": "zh",
            "ts": time.time(),
        }
        await short_store.save(memory, expires_at=time.time() + 3600)
        await _seed_profile(
            profile_store,
            "profile:qq:clear-active-user",
            channel="qq",
            user_id="clear-active-user",
            preferred_name="旧称呼",
            preferences={
                "likes": ["鸡肉"],
                "dislikes": [],
                "allergens": [],
                "dietary_constraints": [],
                "available_ingredients": [],
            },
            version=3,
        )

        preserved = await service.clear(thread_id)
        cleared_short = await service.load(thread_id)
        cleared_profile = await profile_store.load(
            "profile:qq:clear-active-user"
        )

        assert preserved.active_cooking["cookId"] == "running-recipe"
        assert cleared_short.recent_turns == []
        assert cleared_short.preferences["likes"] == []
        assert cleared_short.task_state.candidate_recipes == []
        assert cleared_short.task_state.active_cooking["cookId"] == (
            "running-recipe"
        )
        assert cleared_profile is not None
        assert cleared_profile.preferred_name is None
        assert cleared_profile.preferences["likes"] == []

    asyncio.run(run())


def test_full_memory_clear_drains_older_turn_queue_before_clear_marker():
    async def run() -> None:
        thread_id = build_qq_thread_id(
            "dm",
            "clear-queue-user",
            "clear-queue-user",
        )
        short_store = RecordingStore()
        profile_store = RecordingStore()
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )

        service.enqueue_turn(
            thread_id,
            "user",
            "我喜欢香菜",
            channel="qq",
            user_id="clear-queue-user",
        )
        await service.clear(thread_id)
        service.enqueue_turn(
            thread_id,
            "assistant",
            "已清除长期保存的资料。",
            channel="qq",
            user_id="clear-queue-user",
        )
        await service.drain_pending(timeout_seconds=2)

        short = await service.load(thread_id)
        profile = await profile_store.load(
            "profile:qq:clear-queue-user"
        )
        assert [turn.content for turn in short.recent_turns] == [
            "已清除长期保存的资料。"
        ]
        assert profile is not None
        assert profile.preferences["likes"] == []
        assert profile.long_term_facts == []
        assert profile.summary.get("food_memory_cleared_at")

    asyncio.run(run())


def test_recall_uses_postgres_facts_when_milvus_index_is_not_configured():
    async def run() -> None:
        now = time.time()
        profile_store = RecordingStore()
        await _seed_profile(
            profile_store,
            "profile:qq:pg-fallback-user",
            channel="qq",
            user_id="pg-fallback-user",
            facts=[
                _general_fact(
                    "fact-newest",
                    "数据工程师",
                    updated_at=now,
                ),
                _general_fact(
                    "fact-older",
                    "喜欢露营",
                    updated_at=now - 60,
                ),
            ],
            version=2,
        )
        service = ConversationService(
            RecordingStore(),
            profile_store=profile_store,
            profile_memory_index=None,
        )
        thread_id = build_qq_thread_id(
            "dm",
            "pg-fallback-user",
            "pg-fallback-user",
        )

        recalled = await service.recall_general_profile_facts(
            thread_id,
            "我的职业",
            limit=1,
        )

        assert [item["id"] for item in recalled] == ["fact-newest"]
        assert profile_store.load_calls == ["profile:qq:pg-fallback-user"]

    asyncio.run(run())
