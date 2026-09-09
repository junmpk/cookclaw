"""QQ 会话记忆、共享存储契约与摘要压缩测试。"""
import asyncio
import gc
import json
import time

import pytest

import app.conversation.service as conversation_service_module
from app.conversation.models import ConversationMemory, ConversationTaskState
from app.conversation.preference_command import handle_preference_memory_turn
from app.conversation.postgres_profile_store import parse_profile_key
from app.conversation.service import (
    ConversationService,
    build_qq_thread_id,
    build_whatsapp_thread_id,
    build_weixin_thread_id,
    should_persist_exchange,
    should_persist_session_exchange,
    supports_account_profile,
    supports_persistent_memory,
    supports_session_memory,
)
from app.conversation.store import InMemoryConversationStore
from app.agent.participle_agent import (
    _detail_candidate_indexes,
    _explanation_recipe_index,
    _referenced_recipe_indexes,
    _is_candidate_choice_request,
    _has_region_evidence,
    _search_memory_action,
    _snapshot_to_search_result,
    _extract_omitted_recipe,
    _omitted_recipe_followup,
    _memory_correction_followup,
    _recent_candidate_followup,
    _query_from_execute_intent,
)
from app.conversation.summarizer import update_preferences
from app.orchestrator.semantic_rewrite import rewrite_user_utterance


def test_qq_thread_id_isolates_group_users():
    a = build_qq_thread_id("group", "g1", "u1")
    b = build_qq_thread_id("group", "g1", "u2")
    assert a == "qq:group:g1:u1"
    assert a != b


def test_whatsapp_thread_id_isolates_group_members_and_enables_memory():
    a = build_whatsapp_thread_id("group", "family@g.us", "111@s.whatsapp.net")
    b = build_whatsapp_thread_id("group", "family@g.us", "222@s.whatsapp.net")
    assert a == "whatsapp:group:family@g.us:111@s.whatsapp.net"
    assert a != b
    assert supports_persistent_memory(a)
    assert not supports_persistent_memory("111@s.whatsapp.net")


def test_weixin_thread_id_isolates_bot_accounts_and_enables_memory():
    a = build_weixin_thread_id("bot-a", "wx-user")
    b = build_weixin_thread_id("bot-b", "wx-user")
    other_user = build_weixin_thread_id("bot-a", "wx-user-b")
    assert a == "weixin:dm:bot-a:wx-user"
    assert a != b
    assert a != other_user
    assert supports_persistent_memory(a)


def test_web_session_memory_never_grants_account_profile_identity():
    thread_id = "web:qq:dm:forged-user:forged-user"

    assert supports_session_memory(thread_id)
    assert not supports_account_profile(thread_id)
    assert not supports_persistent_memory(thread_id)
    assert conversation_service_module.conversation_identity(thread_id) == (
        "web",
        None,
    )


def test_incomplete_im_thread_never_grants_account_profile_identity():
    for thread_id in (
        "qq:",
        "qq:dm::user",
        "whatsapp:dm:chat:",
        "weixin:group:bot:user",
        "weixin:dm::user",
    ):
        assert not supports_account_profile(thread_id)


def test_profile_key_rejects_identity_mismatch():
    memory = ConversationMemory(
        thread_id="qq:dm:chat:actual-user",
        channel="qq",
        user_id="other-user",
    )
    assert ConversationService._profile_key(memory) is None


def test_ephemeral_thread_locks_are_reclaimed_when_unused():
    service = ConversationService(InMemoryConversationStore())

    storage_lock = service._lock("web:ephemeral-storage")
    turn_lock = service.turn_lock("web:ephemeral-turn")
    assert len(service._locks) == 1
    assert len(service._turn_locks) == 1

    del storage_lock, turn_lock
    gc.collect()

    assert len(service._locks) == 0
    assert len(service._turn_locks) == 0


def test_weixin_profile_key_isolates_same_peer_across_bot_accounts():
    first = ConversationMemory(
        thread_id=build_weixin_thread_id("bot-a", "wx-user"),
        channel="weixin",
        user_id="wx-user",
    )
    second = ConversationMemory(
        thread_id=build_weixin_thread_id("bot-b", "wx-user"),
        channel="weixin",
        user_id="wx-user",
    )
    assert ConversationService._profile_key(first) == "profile:weixin:bot-a:wx-user"
    assert ConversationService._profile_key(second) == "profile:weixin:bot-b:wx-user"


def test_legacy_top_level_clarification_migrates_into_task_state():
    memory = ConversationMemory.from_dict({
        "thread_id": "qq:dm:legacy:legacy",
        "channel": "qq",
        "pending_search_clarification": {
            "request": {"query": "鸡肉"},
            "dimension": "flavor_preferences",
            "ts": time.time(),
        },
    })

    assert memory.task_state.pending_search_clarification["request"]["query"] == "鸡肉"
    assert "pending_search_clarification" not in memory.to_dict()


def test_task_state_persists_across_service_recreation_and_isolates_users():
    async def run():
        store = InMemoryConversationStore()
        first_service = ConversationService(store)
        first = build_qq_thread_id("dm", "user-a", "user-a")
        second = build_qq_thread_id("dm", "user-b", "user-b")
        await first_service.save_task_state(first, ConversationTaskState(
            current_task="recipe_selection",
            candidate_recipes=[
                {"cookId": "r1", "name": "菜一"},
                {"cookId": "r2", "name": "菜二"},
            ],
            selected_recipe_id="r2",
            selected_recipe_name="菜二",
            selected_recipe={"cookId": "r2", "name": "菜二"},
            excluded_recipe_ids=["r1"],
            updated_at=time.time(),
        ))

        recreated = ConversationService(store)
        restored = await recreated.load_task_state(first)
        untouched = await recreated.load_task_state(second)

        assert [item["cookId"] for item in restored.candidate_recipes] == ["r1", "r2"]
        assert restored.selected_recipe_id == "r2"
        assert restored.excluded_recipe_ids == ["r1"]
        assert untouched.candidate_recipes == []
        assert untouched.selected_recipe_id is None

    asyncio.run(run())


def test_active_search_filters_persist_when_candidates_are_invalidated():
    async def run():
        service = ConversationService(InMemoryConversationStore())
        thread_id = build_qq_thread_id("dm", "active-filter", "active-filter")
        await service.save_task_state(thread_id, ConversationTaskState(
            current_task="recipe_search",
            active_search_request={
                "original_text": "推荐鸡肉菜，不要辣",
                "query": "鸡肉",
                "ingredients": ["鸡肉"],
                "avoid": ["辣"],
            },
            language="zh",
            updated_at=time.time(),
        ))

        restored = await service.load_task_state(thread_id)
        assert restored.current_task == "recipe_search"
        assert restored.candidate_recipes == []
        assert restored.active_search_request["ingredients"] == ["鸡肉"]
        assert restored.active_search_request["avoid"] == ["辣"]
        assert restored.language == "zh"

    asyncio.run(run())


def test_pending_device_action_is_consumed_once_from_shared_task_state():
    async def run():
        service = ConversationService(InMemoryConversationStore())
        thread_id = build_qq_thread_id("dm", "atomic-device", "atomic-device")
        pending = {
            "cookId": "r1",
            "name": "菜一",
            "device_id": "office",
            "action_id": "action-once",
            "msg_id": 123456,
            "lang": "zh",
            "ts": time.time(),
        }
        await service.save_task_state(thread_id, ConversationTaskState(
            current_task="device_start",
            candidate_recipes=[{"cookId": "r1", "name": "菜一"}],
            pending_device_start=pending,
            pending_action={
                "kind": "confirm_device_start",
                "payload": {"action_id": "action-once"},
                "ts": time.time(),
            },
            updated_at=time.time(),
        ))

        claims = await asyncio.gather(
            service.consume_pending_device_start(
                thread_id,
                expected_action_id="action-once",
            ),
            service.consume_pending_device_start(
                thread_id,
                expected_action_id="action-once",
            ),
        )
        assert sum(item is not None for item in claims) == 1
        state = await service.load_task_state(thread_id)
        assert state.pending_device_start is None
        assert state.pending_action is None

    asyncio.run(run())


def test_stale_task_patch_cannot_erase_new_active_device_state():
    async def run():
        service = ConversationService(InMemoryConversationStore())
        thread_id = build_qq_thread_id("dm", "task-race", "task-race")
        pending = {
            "cookId": "r1",
            "name": "菜一",
            "device_id": "office",
            "action_id": "race-action",
            "msg_id": 777,
            "lang": "zh",
            "ts": time.time(),
        }
        initial = ConversationTaskState(
            current_task="device_start",
            candidate_recipes=[{"cookId": "r1", "name": "菜一"}],
            pending_device_start=pending,
            pending_action={
                "kind": "confirm_device_start",
                "payload": {"action_id": "race-action"},
                "ts": time.time(),
            },
            updated_at=time.time(),
        )
        await service.save_task_state(thread_id, initial)
        stale_record = await service.load_task_state_record(thread_id)
        stale_before = stale_record.state

        claimed = await service.consume_pending_device_start(
            thread_id,
            expected_action_id="race-action",
        )
        assert claimed is not None
        active_record = await service.load_task_state_record(thread_id)
        active_before = active_record.state
        active_after = ConversationTaskState.from_dict(active_before.to_dict())
        active_after.active_cooking = {
            "cookId": "r1",
            "name": "菜一",
            "device_id": "office",
            "lang": "zh",
            "ts": time.time(),
        }
        await service.patch_task_state(
            thread_id,
            active_before,
            active_after,
            expected_revision=active_record.revision,
            expected_generation=active_record.generation,
        )

        stale_after = ConversationTaskState.from_dict(stale_before.to_dict())
        stale_after.pending_device_start = None
        stale_after.pending_action = None
        await service.patch_task_state(
            thread_id,
            stale_before,
            stale_after,
            expected_revision=stale_record.revision,
            expected_generation=stale_record.generation,
        )

        final = await service.load_task_state(thread_id)
        assert final.active_cooking["device_id"] == "office"
        assert final.current_task == "device_cooking"
        assert final.pending_device_start is None

    asyncio.run(run())


def test_redis_store_pending_claim_uses_atomic_script_and_returns_payload():
    from app.conversation.redis_store import RedisConversationStore

    class Client:
        def __init__(self):
            self.args = None

        async def eval(self, *args):
            self.args = args
            return [
                1,
                json.dumps({
                    "action_id": "redis-action",
                    "cookId": "r1",
                    "msg_id": 8899,
                }),
            ]

    async def run():
        client = Client()
        store = RedisConversationStore(client, key_prefix="cookclaw:test")
        claimed = await store.consume_pending_device_start(
            "qq:dm:redis-user:redis-user",
            "redis-action",
        )

        assert claimed["action_id"] == "redis-action"
        assert claimed["msg_id"] == 8899
        assert client.args[1] == 1
        assert client.args[3] == "redis-action"
        assert "qq:dm:redis-user:redis-user" not in client.args[2]

    asyncio.run(run())


def test_task_state_repository_restores_state_after_new_turn(monkeypatch):
    import app.conversation.service as service_module
    from app.conversation.task_state_repository import RedisDialogueTaskStateRepository
    from app.ports.dialogue_state import (
        get_selected_recipe,
        remember_candidates,
        resolve_selection,
        set_selected_recipe,
    )

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        thread_id = build_qq_thread_id("dm", "restart-user", "restart-user")
        repository = RedisDialogueTaskStateRepository()
        async with repository.turn_scope(thread_id):
            remember_candidates(
                thread_id,
                {
                    "results": [
                        {"metadata": {"recipe_id": "r1", "name": "菜一"}},
                        {"metadata": {"recipe_id": "r2", "name": "菜二"}},
                    ],
                },
                lang="zh",
            )
            set_selected_recipe(
                thread_id,
                {"cookId": "r2", "name": "菜二"},
                lang="zh",
            )
        # 新 Turn 创建新的隔离工作区，并从 TaskStateStore 重新水合。
        async with repository.turn_scope(thread_id):
            assert resolve_selection(thread_id, "第二个")["cookId"] == "r2"
            assert get_selected_recipe(thread_id)["cookId"] == "r2"

    asyncio.run(run())


def test_plain_recipe_selection_opens_detail_without_device_precheck(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    async def detail(recipe, lang):
        return f"DETAIL:{recipe['name']}"

    async def forbidden_device(*_args, **_kwargs):
        raise AssertionError("plain selection must not enter device precheck")

    async def run():
        thread_id = "selection-detail"
        session.clear_thread(thread_id)
        session.remember_candidates(thread_id, {
            "results": [{
                "id": "r1",
                "metadata": {"recipe_id": "r1", "name": "番茄炒蛋"},
            }],
        }, lang="zh")
        session.set_pending_action(thread_id, "select_recipe", lang="zh")
        raw = await agent_module._handle_pending_action(thread_id, "第一道")
        message = json.loads(raw)["message"]
        assert "DETAIL:番茄炒蛋" in message
        assert "设备" in message
        assert session.get_pending_action(thread_id)["kind"] == "device_precheck"

    monkeypatch.setattr(agent_module, "_recipe_detail_response", detail)
    monkeypatch.setattr(agent_module, "_prepare_recipe_for_device", forbidden_device)
    asyncio.run(run())


def test_pending_action_affirmative_does_not_override_search_clarification():
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    async def run():
        thread_id = "pending-action-with-search-clarification"
        session.clear_thread(thread_id)
        session.set_pending_action(
            thread_id,
            "select_recipe",
            payload={"candidate_count": 3},
            lang="zh",
        )

        result = await agent_module._handle_pending_action(
            thread_id,
            "可以",
            pending_clarification={
                "dimension": "available_ingredients",
                "request": {"query": "晚饭"},
            },
        )

        assert result is None
        assert session.get_pending_action(thread_id)["kind"] == "select_recipe"
        ordinal = await agent_module._handle_pending_action(
            thread_id,
            "第一道",
            pending_clarification={
                "dimension": "available_ingredients",
                "request": {"query": "晚饭"},
            },
        )
        assert ordinal is None
        session.clear_thread(thread_id)

    asyncio.run(run())


def test_menu_quantity_complaint_replans_current_menu_without_safety_reask(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    import app.orchestrator.menu_plan as menu_plan_module
    from app.conversation import task_state_workspace as session
    from app.orchestrator.search_request import SearchRequest

    captured = {}

    async def build_menu_plan(request, **_kwargs):
        captured["request"] = request
        results = [
            {
                "id": f"menu-{index}",
                "menu_role": "dish" if index <= 6 else "soup",
                "metadata": {
                    "recipe_id": f"menu-{index}",
                    "name": f"菜单菜谱{index}",
                    "ingredients": ["测试食材"],
                },
            }
            for index in range(1, 9)
        ]
        return {
            "success": True,
            "results": results,
            "_menu_plan": {
                "requested": {"dish": 6, "soup": 2},
                "fulfilled": {"dish": 6, "soup": 2},
                "missing": {"dish": 0, "soup": 0},
                "complete": True,
                "errors": [],
            },
        }

    async def run():
        thread_id = "menu-quantity-complaint"
        session.clear_thread(thread_id)
        previous = SearchRequest(
            original_text="8个人吃饭，没有忌口，请推荐菜单",
            query="晚餐 下酒",
            scenes=["下酒"],
            party_size=8,
            constraints_confirmed=True,
            result_limit=6,
            menu_dish_count=5,
            menu_soup_count=1,
        )
        session.set_menu_task(
            thread_id,
            previous.public_dict(),
            {"success": True, "results": []},
            lang="zh",
        )
        session.set_pending_action(
            thread_id,
            "select_recipe",
            payload={"context_kind": "menu_plan", "candidate_count": 5},
            lang="zh",
        )

        raw = await agent_module._handle_pending_action(
            thread_id,
            "10个人推荐五个菜。。。太少了",
        )

        assert raw is not None
        payload = json.loads(raw)
        request = captured["request"]
        assert request.party_size == 10
        assert request.constraints_confirmed is True
        assert request.menu_dish_count == 6
        assert request.menu_soup_count == 2
        assert request.result_limit == 8
        assert request.explicit_result_limit is False
        assert request.scenes == ["下酒"]
        assert "五个菜" not in request.retrieval_query()
        assert payload["type"] == "menu_plan"
        assert len(payload["data"]["recipes"]) == 8
        assert session.get_menu_task(thread_id)["request"]["party_size"] == 10
        session.clear_thread(thread_id)

    monkeypatch.setattr(menu_plan_module, "build_menu_plan", build_menu_plan)
    asyncio.run(run())


def test_candidate_number_positive_feedback_keeps_current_batch_without_search(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    async def forbidden_search(*_args, **_kwargs):
        raise AssertionError("候选积极反馈不能触发新搜索")

    async def run():
        thread_id = "candidate-positive-feedback"
        session.clear_thread(thread_id)
        session.remember_candidates(
            thread_id,
            {
                "results": [
                    {"metadata": {"recipe_id": "r1", "name": "凉拌黄瓜"}},
                    {"metadata": {"recipe_id": "r2", "name": "清蒸鲈鱼"}},
                    {"metadata": {"recipe_id": "r3", "name": "香菇蒸鸡"}},
                ],
            },
            lang="zh",
        )
        session.set_pending_action(thread_id, "select_recipe", lang="zh")

        response = json.loads(
            await agent_module._handle_pending_action(
                thread_id,
                "菜谱1看着不错呢",
            )
        )

        assert "凉拌黄瓜" in response["message"]
        assert session.get_selected_recipe(thread_id)["cookId"] == "r1"
        assert session.get_pending_action(thread_id)["kind"] == "select_recipe"
        assert [
            item["cookId"]
            for item in session.recall_candidate_context(thread_id)["items"]
        ] == ["r1", "r2", "r3"]
        session.clear_thread(thread_id)

    monkeypatch.setattr(agent_module, "_run_search_subprocess", forbidden_search)
    asyncio.run(run())


def test_candidate_exclusion_and_reconsideration_keep_original_indexes(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    async def detail(recipe, lang):
        return f"DETAIL:{recipe['name']}"

    async def run():
        thread_id = "candidate-reconsideration"
        session.clear_thread(thread_id)
        session.remember_candidates(
            thread_id,
            {
                "results": [
                    {"metadata": {"recipe_id": "r1", "name": "菜一"}},
                    {"metadata": {"recipe_id": "r2", "name": "菜二"}},
                    {"metadata": {"recipe_id": "r3", "name": "菜三"}},
                ],
            },
            lang="zh",
        )
        session.set_pending_action(thread_id, "select_recipe", lang="zh")

        second_removed = json.loads(
            await agent_module._handle_pending_action(thread_id, "第二个不要")
        )
        assert "菜二" in second_removed["message"]
        assert session.get_excluded_recipe_ids(thread_id) == ["r2"]

        first_removed = json.loads(
            await agent_module._handle_pending_action(thread_id, "第一个也太麻烦了")
        )
        assert "菜一" in first_removed["message"]
        assert session.get_excluded_recipe_ids(thread_id) == ["r2", "r1"]
        assert [
            item["cookId"]
            for item in session.recall_candidate_context(thread_id)["items"]
        ] == ["r1", "r2", "r3"]

        reconsidered = json.loads(
            await agent_module._handle_pending_action(
                thread_id,
                "还是刚才第二个吧",
            )
        )
        assert "DETAIL:菜二" in reconsidered["message"]
        assert session.get_selected_recipe(thread_id)["cookId"] == "r2"
        assert session.get_excluded_recipe_ids(thread_id) == ["r1"]
        session.clear_thread(thread_id)

    monkeypatch.setattr(agent_module, "_recipe_detail_response", detail)
    asyncio.run(run())


def test_candidate_focus_supports_this_one_selection(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    async def detail(recipe, lang):
        return f"DETAIL:{recipe['name']}"

    async def run():
        thread_id = "candidate-this-one"
        session.clear_thread(thread_id)
        session.remember_candidates(
            thread_id,
            {
                "results": [
                    {
                        "id": "r1",
                        "metadata": {
                            "recipe_id": "r1",
                            "name": "菜一",
                            "tags": ["简单"],
                        },
                    },
                    {
                        "id": "r2",
                        "metadata": {
                            "recipe_id": "r2",
                            "name": "菜二",
                            "tags": ["清淡"],
                        },
                    },
                    {
                        "id": "r3",
                        "metadata": {
                            "recipe_id": "r3",
                            "name": "菜三",
                            "tags": ["快手"],
                        },
                    },
                ],
            },
            lang="zh",
        )
        session.set_pending_action(thread_id, "select_recipe", lang="zh")
        explained = await agent_module._recent_candidate_followup(
            thread_id,
            "第二个怎么样",
        )
        assert "菜二" in json.loads(explained)["message"]

        selected = json.loads(
            await agent_module._handle_pending_action(thread_id, "就这个吧")
        )
        assert "DETAIL:菜二" in selected["message"]
        assert session.get_selected_recipe(thread_id)["cookId"] == "r2"
        session.clear_thread(thread_id)

    monkeypatch.setattr(agent_module, "_recipe_detail_response", detail)
    asyncio.run(run())


def test_controlled_deep_agent_selects_only_from_current_candidates(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    calls = []

    async def fake_deep_choice(**kwargs):
        calls.append(kwargs)
        return "r2"

    async def forbidden_legacy_choice(*_args, **_kwargs):
        raise AssertionError("受控 Deep Agent 成功后不应再调用旧单次选择模型")

    async def run():
        thread_id = "qq:controlled-deep-agent-choice"
        session.clear_thread(thread_id)
        recipes = [
            {"cookId": "r1", "name": "香辣鸡丁", "tags": ["香辣"]},
            {"cookId": "r2", "name": "清蒸鸡肉", "tags": ["清淡"]},
            {"cookId": "r3", "name": "油炸鸡排", "tags": ["油炸"]},
        ]
        answer = await agent_module._candidate_choice_answer(
            thread_id,
            "这三道里帮我选一个清淡点的",
            recipes,
            "zh",
        )
        assert "清蒸鸡肉" in answer
        assert calls[0]["recipes"] == recipes
        assert calls[0]["question"] == "这三道里帮我选一个清淡点的"
        assert calls[0]["conversation_state"]["current_task"] is None
        session.clear_thread(thread_id)

    monkeypatch.setattr(
        agent_module.settings,
        "CONVERSATION_DEEP_AGENT_ENABLED",
        True,
    )
    monkeypatch.setattr(
        agent_module.settings,
        "CONVERSATION_DEEP_AGENT_CHANNELS",
        "qq",
    )
    monkeypatch.setattr(
        agent_module.settings,
        "CONVERSATION_DEEP_AGENT_ROLLOUT_PERCENT",
        100,
    )
    monkeypatch.setattr(
        agent_module,
        "choose_candidate_with_deep_agent",
        fake_deep_choice,
    )
    monkeypatch.setattr(
        agent_module,
        "_qa_llm",
        type("ForbiddenLegacyChoice", (), {"ainvoke": forbidden_legacy_choice})(),
    )
    asyncio.run(run())


def test_controlled_deep_agent_is_scoped_to_qq_and_receives_sanitized_task_state(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    captured = []

    async def fake_route(question, **kwargs):
        captured.append((question, kwargs))
        return "routed"

    async def run():
        qq_thread = "qq:dm:deep-context:deep-context"
        web_thread = "web:deep-context"
        session.clear_thread(qq_thread)
        session.remember_candidates(qq_thread, {
            "success": True,
            "results": [
                {
                    "id": "r1",
                    "metadata": {
                        "recipe_id": "r1",
                        "name": "鸡肉菜一",
                    },
                },
            ],
            "_search_request": {
                "ingredients": ["鸡肉"],
                "exclude": ["辣"],
            },
        }, lang="zh")
        session.set_selected_recipe(
            qq_thread,
            {"cookId": "r1", "name": "鸡肉菜一"},
            lang="zh",
            source="test",
        )
        session.set_pending(
            qq_thread,
            "r1",
            "鸡肉菜一",
            device_id="must-not-leak",
            lang="zh",
        )

        assert await agent_module._route_conversation_turn(
            "不要辣",
            qq_thread,
        ) == "routed"
        assert await agent_module._route_conversation_turn(
            "不要辣",
            web_thread,
        ) == "routed"

        qq_kwargs = captured[0][1]
        assert qq_kwargs["recommendation_deep_agent_enabled"] is True
        context = qq_kwargs["recommendation_agent_context"]
        assert context["schema_version"] == "controlled_context_v1"
        assert context["active_search_request"]["ingredients"] == ["鸡肉"]
        assert context["selected_recipe"]["recipe_id"] == "r1"
        assert context["pending_device_start"]["waiting_confirmation"] is True
        assert "must-not-leak" not in json.dumps(context, ensure_ascii=False)

        web_kwargs = captured[1][1]
        assert web_kwargs["recommendation_deep_agent_enabled"] is False
        assert web_kwargs["recommendation_agent_context"] is None
        session.clear_thread(qq_thread)
        session.clear_thread(web_thread)

    monkeypatch.setattr(
        agent_module.settings,
        "CONVERSATION_DEEP_AGENT_ENABLED",
        True,
    )
    monkeypatch.setattr(
        agent_module.settings,
        "CONVERSATION_DEEP_AGENT_CHANNELS",
        "qq",
    )
    monkeypatch.setattr(
        agent_module.settings,
        "CONVERSATION_DEEP_AGENT_ROLLOUT_PERCENT",
        100,
    )
    monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
    monkeypatch.setattr(agent_module, "supports_session_memory", lambda _tid: False)
    asyncio.run(run())


def test_controlled_candidate_agent_failure_does_not_trigger_second_choice_model(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module

    async def failed_deep_choice(**_kwargs):
        return None

    class ForbiddenLegacyChoice:
        async def ainvoke(self, _messages):
            raise AssertionError("受控 Agent 失败后不应再调用第二个候选选择模型")

    monkeypatch.setattr(
        agent_module.settings,
        "CONVERSATION_DEEP_AGENT_ENABLED",
        True,
    )
    monkeypatch.setattr(
        agent_module.settings,
        "CONVERSATION_DEEP_AGENT_CHANNELS",
        "qq",
    )
    monkeypatch.setattr(
        agent_module.settings,
        "CONVERSATION_DEEP_AGENT_ROLLOUT_PERCENT",
        100,
    )
    monkeypatch.setattr(
        agent_module,
        "choose_candidate_with_deep_agent",
        failed_deep_choice,
    )
    monkeypatch.setattr(agent_module, "_qa_llm", ForbiddenLegacyChoice())

    answer = asyncio.run(agent_module._candidate_choice_answer(
        "qq:dm:deep-fallback:deep-fallback",
        "帮我选一道",
        [
            {"cookId": "r1", "name": "真实菜一", "tags": ["清淡"]},
            {"cookId": "r2", "name": "真实菜二", "tags": ["家常"]},
        ],
        "zh",
    ))
    assert "真实菜一" in answer


def test_condition_followups_merge_and_research_without_reclassifying(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session
    from app.orchestrator.recommendation_response import RecommendationNarrative

    searches = []

    async def fake_search(query, top_k, lang):
        searches.append((query, top_k, lang))
        batch = len(searches)
        return {
            "success": True,
            "results": [
                {
                    "id": f"r{batch}-{index}",
                    "score": 0.9,
                    "metadata": {
                        "recipe_id": f"r{batch}-{index}",
                        "name": f"鸡肉菜{batch}-{index}",
                        "ingredients": ["鸡肉"],
                        "tags": ["家常"],
                    },
                }
                for index in range(1, 4)
            ],
        }

    async def fake_narrative(*_args, **_kwargs):
        return RecommendationNarrative("按你刚补充的条件重新挑了。", "", {}, "")

    async def forbidden_route(*_args, **_kwargs):
        raise AssertionError("明确的条件跟进不应重新调用意图分类")

    async def run():
        thread_id = "condition-followups"
        session.clear_thread(thread_id)
        initial_request = {
            "original_text": "推荐几个鸡肉菜",
            "query": "鸡肉",
            "ingredients": ["鸡肉"],
        }
        session.set_active_search_request(thread_id, initial_request, lang="zh")
        session.remember_candidates(thread_id, {
            "success": True,
            "results": [{
                "id": "old",
                "metadata": {
                    "recipe_id": "old",
                    "name": "旧候选",
                    "ingredients": ["鸡肉"],
                },
            }],
            "_search_request": initial_request,
        }, lang="zh")
        session.set_pending_action(thread_id, "select_recipe", lang="zh")

        for followup in ("不要辣的", "孩子也能吃", "简单一点"):
            response = json.loads(await agent_module.qqbot_chat(
                followup,
                thread_id,
            ))
            assert response["type"] == "recipe_search"
            assert len(response["data"]["recipes"]) == 3
            assert response["data"]["opening"] == "按你刚补充的条件重新挑了。"

        request = session.get_active_search_request(thread_id)["request"]
        assert request["ingredients"] == ["鸡肉"]
        assert request["avoid"] == ["辣"]
        assert request["scenes"] == ["儿童", "快手"]
        assert len(searches) == 3
        assert all(call[1:] == (12, "zh") for call in searches)
        assert [item["cookId"] for item in session.recall_candidate_context(thread_id)["items"]] == [
            "r3-1", "r3-2", "r3-3",
        ]
        assert session.get_pending_action(thread_id)["kind"] == "select_recipe"
        session.clear_thread(thread_id)

    monkeypatch.setattr(agent_module, "_run_search_subprocess", fake_search)
    monkeypatch.setattr(agent_module, "generate_recommendation_narrative", fake_narrative)
    monkeypatch.setattr(agent_module, "route_fast_path", forbidden_route)
    asyncio.run(run())


def test_condition_revision_replaces_conflict_and_keeps_prior_subject(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session
    from app.orchestrator.recommendation_response import RecommendationNarrative

    searched_queries = []

    async def fake_search(query, top_k, lang):
        searched_queries.append(query)
        return {
            "success": True,
            "results": [{
                "id": "mild-chicken",
                "score": 0.9,
                "metadata": {
                    "recipe_id": "mild-chicken",
                    "name": "微辣鸡肉",
                    "ingredients": ["鸡肉"],
                    "tags": ["微辣"],
                },
            }],
        }

    async def fake_narrative(*_args, **_kwargs):
        return RecommendationNarrative("", "", {}, "")

    async def forbidden_route(*_args, **_kwargs):
        raise AssertionError("明确的条件修改不应重新调用意图分类")

    async def run():
        thread_id = "condition-revision"
        session.clear_thread(thread_id)
        session.set_active_search_request(thread_id, {
            "original_text": "推荐不辣的鸡肉菜",
            "query": "鸡肉 不辣",
            "ingredients": ["鸡肉"],
            "avoid": ["辣"],
        }, lang="zh")

        await agent_module.qqbot_chat(
            "算了，稍微辣一点也可以",
            thread_id,
        )
        await agent_module.qqbot_chat(
            "不要牛肉",
            thread_id,
        )

        request = session.get_active_search_request(thread_id)["request"]
        assert request["ingredients"] == ["鸡肉"]
        assert request["flavors"] == ["微辣"]
        assert request["avoid"] == ["太辣"]
        assert request["exclude"] == ["牛肉"]
        assert all("不辣" not in query for query in searched_queries)
        assert all("鸡肉" in query for query in searched_queries)
        session.clear_thread(thread_id)

    monkeypatch.setattr(agent_module, "_run_search_subprocess", fake_search)
    monkeypatch.setattr(agent_module, "generate_recommendation_narrative", fake_narrative)
    monkeypatch.setattr(agent_module, "route_fast_path", forbidden_route)
    asyncio.run(run())


def test_failed_condition_search_keeps_filters_for_deterministic_retry(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session
    from app.orchestrator.recommendation_response import RecommendationNarrative

    calls = []

    async def fake_search(query, top_k, lang):
        calls.append(query)
        if len(calls) == 1:
            return None
        return {
            "success": True,
            "results": [{
                "id": "retry-r1",
                "score": 0.9,
                "metadata": {
                    "recipe_id": "retry-r1",
                    "name": "清蒸鸡肉",
                    "ingredients": ["鸡肉"],
                    "tags": ["清淡"],
                },
            }],
        }

    async def fake_narrative(*_args, **_kwargs):
        return RecommendationNarrative("按刚才保存的条件重新搜到了。", "", {}, "")

    async def forbidden_route(*_args, **_kwargs):
        raise AssertionError("保存条件后的明确重试不应重新调用意图分类")

    async def run():
        thread_id = "condition-retry"
        session.clear_thread(thread_id)
        session.set_active_search_request(thread_id, {
            "original_text": "推荐鸡肉菜",
            "query": "鸡肉",
            "ingredients": ["鸡肉"],
        }, lang="zh")

        failed = json.loads(await agent_module.qqbot_chat("不要辣的", thread_id))
        assert failed["type"] == "cooking"
        assert "条件已经合并保存" in failed["message"]
        assert session.recall_candidate_context(thread_id) is None
        assert session.get_active_search_request(thread_id)["request"]["avoid"] == ["辣"]

        retried = json.loads(await agent_module.qqbot_chat("重新搜索", thread_id))
        assert retried["type"] == "recipe_search"
        assert retried["data"]["recipes"][0]["id"] == "retry-r1"
        assert calls == ["鸡肉", "鸡肉"]
        session.clear_thread(thread_id)

    monkeypatch.setattr(agent_module, "_run_search_subprocess", fake_search)
    monkeypatch.setattr(agent_module, "generate_recommendation_narrative", fake_narrative)
    monkeypatch.setattr(agent_module, "route_fast_path", forbidden_route)
    asyncio.run(run())


def test_smalltalk_does_not_clear_recipe_candidates(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation import task_state_workspace as session
    from app.orchestrator.intent import IntentCategory, IntentResult
    from app.orchestrator.router import FastPathOutcome

    route_calls = []

    async def fake_route(question, **_kwargs):
        route_calls.append(question)
        return FastPathOutcome(
            kind="agent",
            lang="zh",
            intent=IntentResult(related=False, category=IntentCategory.off_topic),
        )

    async def fake_smalltalk(*_args, **_kwargs):
        return "不客气，慢慢挑。"

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        thread_id = build_qq_thread_id("dm", "smalltalk-resume", "smalltalk-resume")
        session.clear_thread(thread_id)
        session.remember_candidates(thread_id, {
            "success": True,
            "results": [
                {"id": "r1", "metadata": {"recipe_id": "r1", "name": "鸡肉菜一"}},
                {"id": "r2", "metadata": {"recipe_id": "r2", "name": "鸡肉菜二"}},
                {"id": "r3", "metadata": {"recipe_id": "r3", "name": "鸡肉菜三"}},
            ],
            "_search_request": {
                "original_text": "推荐几个鸡肉菜",
                "query": "鸡肉",
                "ingredients": ["鸡肉"],
            },
        }, lang="zh")
        session.set_pending_action(thread_id, "select_recipe", lang="zh")
        await service.save_task_state(thread_id, session.snapshot_thread_state(thread_id))
        session.clear_thread(thread_id)

        thanked = json.loads(await agent_module.qqbot_chat("谢谢你", thread_id))
        assert thanked["message"] == "不客气，慢慢挑。"

        second = json.loads(await agent_module.qqbot_chat("第二个怎么样", thread_id))
        assert "鸡肉菜二" in second["message"]
        assert route_calls == ["谢谢你"]
        persisted = await service.load_task_state(thread_id)
        assert [item["cookId"] for item in persisted.candidate_recipes] == [
            "r1", "r2", "r3",
        ]
        session.clear_thread(thread_id)

    monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
    monkeypatch.setattr(agent_module, "_smalltalk_answer", fake_smalltalk)
    asyncio.run(run())


def test_device_status_interrupt_can_resume_same_recipe_list_without_search(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation import task_state_workspace as session
    from app.orchestrator.intent import IntentCategory, IntentResult
    from app.orchestrator.router import FastPathOutcome

    route_calls = []

    async def fake_route(question, **_kwargs):
        route_calls.append(question)
        return FastPathOutcome(
            kind="agent",
            lang="zh",
            intent=IntentResult(related=True, category=IntentCategory.device_manage),
        )

    async def fake_device_status(*_args, **_kwargs):
        return json.dumps({"message": "测试设备在线"}, ensure_ascii=False)

    async def forbidden_search(*_args, **_kwargs):
        raise AssertionError("恢复菜谱任务不应重复检索")

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        thread_id = build_qq_thread_id("dm", "device-interrupt", "device-interrupt")
        session.clear_thread(thread_id)
        session.remember_candidates(thread_id, {
            "success": True,
            "results": [
                {"id": "r1", "metadata": {"recipe_id": "r1", "name": "鸡肉菜一"}},
                {"id": "r2", "metadata": {"recipe_id": "r2", "name": "鸡肉菜二"}},
                {"id": "r3", "metadata": {"recipe_id": "r3", "name": "鸡肉菜三"}},
            ],
            "_search_request": {
                "original_text": "推荐几个鸡肉菜",
                "query": "鸡肉",
                "ingredients": ["鸡肉"],
            },
        }, lang="zh")
        session.set_pending_action(thread_id, "select_recipe", lang="zh")
        await service.save_task_state(thread_id, session.snapshot_thread_state(thread_id))
        session.clear_thread(thread_id)

        status = json.loads(await agent_module.qqbot_chat("查一下设备状态", thread_id))
        assert status["message"] == "测试设备在线"

        resumed = json.loads(await agent_module.qqbot_chat("继续说刚才的菜", thread_id))
        assert [item["id"] for item in resumed["data"]["recipes"]] == ["r1", "r2", "r3"]
        assert route_calls == ["查一下设备状态"]
        persisted = await service.load_task_state(thread_id)
        assert persisted.current_task == "recipe_selection"
        assert persisted.pending_action["kind"] == "select_recipe"
        session.clear_thread(thread_id)

    monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
    monkeypatch.setattr(agent_module, "_device_status_and_format", fake_device_status)
    monkeypatch.setattr(agent_module, "_run_search_subprocess", forbidden_search)
    asyncio.run(run())


def test_group_detail_reference_selects_all_candidates_and_never_becomes_search_query():
    recipes = [
        {"cookId": f"r{index}", "name": f"真实菜{index}"}
        for index in range(1, 7)
    ]
    assert _detail_candidate_indexes("把这六道菜的情况详情都发给我", recipes) == list(range(6))
    assert _detail_candidate_indexes("这些菜的具体步骤都给我", recipes) == list(range(6))
    assert _query_from_execute_intent(
        "把这六道菜的情况详情发给我",
        ["六道菜", "详情"],
    ) == ""
    assert _query_from_execute_intent(
        "红烧肉详情",
        ["红烧肉", "详情"],
    ) == "红烧肉"


def test_group_details_restore_persisted_candidates_without_new_search(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation import task_state_workspace as session

    async def detail(recipe, lang):
        return f"DETAIL:{recipe['name']}"

    async def forbidden_search(*_args, **_kwargs):
        raise AssertionError("group detail reference must not trigger a new recipe search")

    async def run():
        thread_id = build_qq_thread_id("dm", "detail-user", "detail-user")
        service = ConversationService(InMemoryConversationStore())
        previous_service = service_module._service
        service_module._service = service
        try:
            await service.save_search(
                thread_id,
                original_question="推荐六道菜",
                search_query="家常菜",
                search_result={
                    "success": True,
                    "results": [
                        {
                            "id": f"r{index}",
                            "score": 0.9,
                            "metadata": {
                                "recipe_id": f"r{index}",
                                "name": f"真实菜{index}",
                            },
                        }
                        for index in range(1, 7)
                    ],
                },
            )
            session.clear_thread(thread_id)
            raw = await agent_module.qqbot_chat(
                "把这六道菜的情况详情发给我",
                thread_id=thread_id,
            )
            message = json.loads(raw)["message"]
            for index in range(1, 7):
                assert f"DETAIL:真实菜{index}" in message
            assert message.count("DETAIL:") == 6
        finally:
            session.clear_thread(thread_id)
            service_module._service = previous_service

    monkeypatch.setattr(agent_module, "_recipe_detail_response", detail)
    monkeypatch.setattr(agent_module, "_run_search_subprocess", forbidden_search)
    asyncio.run(run())


def test_explicit_device_selection_enters_device_precheck(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    called = []

    async def device(thread_id, recipe, lang):
        called.append((thread_id, recipe["cookId"], lang))
        return json.dumps({"message": "PRECHECK"}, ensure_ascii=False)

    async def run():
        thread_id = "selection-device"
        session.clear_thread(thread_id)
        session.remember_candidates(thread_id, {
            "results": [{
                "id": "r1",
                "metadata": {"recipe_id": "r1", "name": "番茄炒蛋"},
            }],
        }, lang="zh")
        session.set_pending_action(thread_id, "select_recipe", lang="zh")
        raw = await agent_module._handle_pending_action(thread_id, "用设备做第一道")
        assert json.loads(raw)["message"] == "PRECHECK"
        assert called == [(thread_id, "r1", "zh")]

    monkeypatch.setattr(agent_module, "_prepare_recipe_for_device", device)
    asyncio.run(run())


def test_profile_key_maps_channel_account_and_user():
    assert parse_profile_key("profile:qq:user-1") == ("qq", "default", "user-1")
    assert parse_profile_key("profile:whatsapp:+86123") == (
        "whatsapp",
        "default",
        "+86123",
    )
    assert parse_profile_key("profile:weixin:bot-a:user:with:colon") == (
        "weixin",
        "bot-a",
        "user:with:colon",
    )
    assert parse_profile_key("profile:weixin:legacy-user") == (
        "weixin",
        "default",
        "legacy-user",
    )


def test_separate_profile_store_shares_long_term_memory_across_threads():
    async def run():
        short_store = InMemoryConversationStore()
        profile_store = InMemoryConversationStore()
        service = ConversationService(short_store, profile_store=profile_store)
        group = build_qq_thread_id("group", "family", "u-shared")
        dm = build_qq_thread_id("dm", "u-shared", "u-shared")

        await service.append_turn(
            group,
            "user",
            "我喜欢蒜香，也不吃香菜",
            channel="qq",
            user_id="u-shared",
        )
        await service.append_turn(
            dm,
            "user",
            "今天吃什么",
            channel="qq",
            user_id="u-shared",
        )

        context = await service.account_memory_context(dm)
        profile = await profile_store.load("profile:qq:u-shared")
        assert context["preferences"]["likes"] == ["蒜香"]
        assert context["preferences"]["dislikes"] == ["香菜"]
        assert profile is not None
        active = {
            (fact["type"], fact["value"])
            for fact in profile.long_term_facts
            if fact["status"] == "active"
        }
        assert ("preference_like", "蒜香") in active
        assert ("preference_dislike", "香菜") in active
        assert await short_store.load("profile:qq:u-shared") is None

    asyncio.run(run())


def test_removed_preference_keeps_a_deleted_fact_for_traceability():
    async def run():
        profile_store = InMemoryConversationStore()
        service = ConversationService(
            InMemoryConversationStore(),
            profile_store=profile_store,
        )
        tid = build_weixin_thread_id("bot-a", "wx-delete")
        await service.append_turn(
            tid,
            "user",
            "我不吃香菜",
            channel="weixin",
            user_id="wx-delete",
        )
        await service.remove_preferences(tid, ["香菜"])

        profile = await profile_store.load("profile:weixin:bot-a:wx-delete")
        fact = next(item for item in profile.long_term_facts if item["value"] == "香菜")
        assert fact["status"] == "deleted"
        assert fact["deleted_at"] >= fact["created_at"]

    asyncio.run(run())


def test_sqlite_conversation_mode_is_rejected(monkeypatch):
    monkeypatch.setattr(conversation_service_module, "_service", None)
    monkeypatch.setenv("CONVERSATION_STORE", "sqlite")
    monkeypatch.delenv("PROFILE_STORE", raising=False)

    with pytest.raises(ValueError, match="expected 'redis'"):
        conversation_service_module.get_conversation_service()


def test_account_profile_carries_preferences_and_digest_across_chat_windows():
    async def run():
        service = ConversationService(InMemoryConversationStore())
        group = build_qq_thread_id("group", "family", "u-long")
        dm = build_qq_thread_id("dm", "u-long", "u-long")
        await service.append_turn(
            group, "user", "我喜欢蒜香，也不吃香菜",
            channel="qq", user_id="u-long",
        )
        await service.append_turn(
            dm, "user", "今天给我推荐点菜",
            channel="qq", user_id="u-long",
        )

        context = await service.account_memory_context(dm)
        assert context["preferences"]["likes"] == ["蒜香"]
        assert context["preferences"]["dislikes"] == ["香菜"]
        assert context["account_digest"] == [
            "饮食偏好-喜欢：蒜香",
            "饮食偏好-不喜欢或排除：香菜",
        ]

    asyncio.run(run())


def test_preference_correction_removes_account_level_memory():
    async def run():
        service = ConversationService(InMemoryConversationStore())
        tid = build_weixin_thread_id("bot-a", "wx-long")
        await service.append_turn(
            tid, "user", "我最近正在减肥",
            channel="weixin", user_id="wx-long",
        )
        context = await service.account_memory_context(tid)
        assert "减肥" not in context["preferences"]["dietary_constraints"]
        assert [item["value"] for item in context["temporal_dietary_constraints"]] == ["减肥"]
        removed = await service.remove_preferences(tid, ["减肥"])
        assert removed == ["减肥"]
        context = await service.account_memory_context(tid)
        assert "减肥" not in context["preferences"]["dietary_constraints"]
        assert context["temporal_dietary_constraints"] == []

    asyncio.run(run())


def test_explicit_diet_retraction_deletes_account_temporal_fact():
    async def run():
        profile_store = InMemoryConversationStore()
        service = ConversationService(
            InMemoryConversationStore(),
            profile_store=profile_store,
        )
        tid = build_qq_thread_id("dm", "diet-delete", "diet-delete")
        await service.append_turn(
            tid,
            "user",
            "我最近正在减肥",
            channel="qq",
            user_id="diet-delete",
        )
        before = await service.account_memory_context(tid)
        assert [
            item["value"]
            for item in before["temporal_dietary_constraints"]
        ] == ["减肥"]

        await service.append_turn(
            tid,
            "user",
            "我已经不减肥了。以后给我推荐菜谱不要参考这个标准",
            channel="qq",
            user_id="diet-delete",
        )

        after = await service.account_memory_context(tid)
        assert after["temporal_dietary_constraints"] == []
        profile = await profile_store.load("profile:qq:diet-delete")
        fact = next(
            item
            for item in profile.long_term_facts
            if item["value"] == "减肥"
        )
        assert fact["status"] == "deleted"
        assert fact["deleted_at"] >= fact["created_at"]

    asyncio.run(run())


def test_transient_diet_profile_fact_expires_out_of_routing_context():
    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(
            store,
            transient_diet_ttl_seconds=60,
        )
        tid = build_qq_thread_id("dm", "diet-ttl", "diet-ttl")
        await service.append_turn(
            tid,
            "user",
            "我最近正在减肥",
            channel="qq",
            user_id="diet-ttl",
        )

        profile = await store.load("profile:qq:diet-ttl")
        fact = next(
            item
            for item in profile.long_term_facts
            if item.get("type") == "dietary_constraint"
            and item.get("value") == "减肥"
        )
        assert fact["expires_at"] > fact["last_confirmed_at"]
        fact["expires_at"] = time.time() - 1
        await store.save(profile, expires_at=time.time() + 3600)

        context = await service.account_memory_context(tid)
        assert context["preferences"]["dietary_constraints"] == []
        assert context["temporal_dietary_constraints"] == []

    asyncio.run(run())


def test_conversation_compacts_and_keeps_recent_turns():
    async def run():
        service = ConversationService(
            InMemoryConversationStore(),
            max_turns=2,
            keep_recent_turns=1,
            max_tokens=99999,
        )
        tid = "qq:c2c:u1:u1"
        await service.append_turn(tid, "user", "我喜欢辣", user_id="u1")
        await service.append_turn(tid, "assistant", "记住了")
        await service.append_turn(tid, "user", "今天想吃鸡肉", user_id="u1")
        await service.append_turn(tid, "assistant", "我来找找")
        memory = await service.load(tid)
        assert len(memory.recent_turns) == 2
        assert memory.summary["summary_version"] == 1
        assert any("我喜欢辣" in line for line in memory.summary["conversation_digest"])
        assert memory.preferences["likes"]

    asyncio.run(run())


def test_save_search_snapshot_keeps_real_evidence():
    async def run():
        service = ConversationService(InMemoryConversationStore())
        tid = "qq:c2c:u1:u1"
        result = {
            "results": [{
                "id": "1",
                "metadata": {
                    "recipe_id": "r1",
                    "name": "剁椒鸡胸肉",
                    "ingredients": ["鸡胸肉", "剁椒"],
                    "tags": ["香辣", "家常"],
                },
            }],
            "_recommendation": {
                "recipe_reasons": {"r1": "风味更足，适合想吃辣的时候。"},
            },
        }
        await service.save_search(tid, "想吃辣的鸡肉", "辣 鸡肉", result)
        memory = await service.load(tid)
        recipe = memory.latest_search["recipes"][0]
        assert recipe["name"] == "剁椒鸡胸肉"
        assert recipe["recommendation_reason"] == ""
        assert recipe["ingredients"] == ["鸡胸肉", "剁椒"]
        assert len(memory.search_history) == 1
        assert memory.latest_search["candidate_pool"][0]["name"] == "剁椒鸡胸肉"

    asyncio.run(run())


def test_account_food_history_is_shared_across_conversations_and_cleared_together():
    async def run():
        service = ConversationService(InMemoryConversationStore(), profile_ttl_seconds=99999)
        first = build_qq_thread_id("group", "g1", "u1")
        second = build_qq_thread_id("dm", "u1", "u1")
        await service.append_turn(first, "user", "推荐鱼", channel="qq", user_id="u1")
        await service.save_search(first, "推荐鱼", "鱼", {
            "results": [{
                "id": "fish-1",
                "metadata": {
                    "recipe_id": "fish-1", "name": "清蒸鲈鱼",
                    "ingredients": ["鲈鱼", "姜"], "tags": ["清蒸"],
                },
            }],
        })

        await service.append_turn(second, "user", "我想吃大蒜多的菜", channel="qq", user_id="u1")
        context = await service.food_memory_context(second)
        assert context["events"][-1]["recipes"][0]["name"] == "清蒸鲈鱼"

        await service.clear(second)
        assert (await service.food_memory_context(first))["events"] == []

    asyncio.run(run())


def test_new_conversation_clears_short_state_but_keeps_profile_and_active_task(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation import task_state_workspace as session

    async def run():
        short_store = InMemoryConversationStore()
        profile_store = InMemoryConversationStore()
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )
        monkeypatch.setattr(service_module, "_service", service)
        thread_id = build_qq_thread_id("dm", "session-reset", "session-reset")
        await service.update_preferred_name(
            thread_id,
            "范老师",
            expected_current=None,
        )
        state = ConversationTaskState(
            candidate_recipes=[
                {"cookId": "r1", "name": "番茄炒蛋"},
                {"cookId": "r2", "name": "清蒸鲈鱼"},
            ],
            pending_action={"kind": "select_recipe", "lang": "zh"},
            active_cooking={
                "cookId": "r-running",
                "name": "红烧肉",
                "device_id": "office",
                "lang": "zh",
                "ts": time.time(),
            },
        )
        await service.save_task_state(thread_id, state)
        await service.append_turn(
            thread_id,
            "user",
            "刚才推荐了什么",
            channel="qq",
            user_id="session-reset",
            persist_account_memory=False,
        )

        response = json.loads(
            await agent_module.qqbot_chat("给我一个新对话", thread_id)
        )

        assert "长期保存的偏好仍然保留" in response["message"]
        profile = await profile_store.load("profile:qq:session-reset")
        assert profile is not None
        assert profile.preferred_name == "范老师"
        memory = await service.load(thread_id)
        assert memory.recent_turns == []
        assert memory.latest_search is None
        assert memory.task_state.candidate_recipes == []
        assert memory.task_state.pending_action is None
        assert memory.task_state.active_cooking["cookId"] == "r-running"
        persisted_task = await service.load_task_state(thread_id)
        assert persisted_task.active_cooking["cookId"] == "r-running"
        assert session.recall_candidates(thread_id) is None
        session.clear_active_cooking(thread_id)
        session.clear_thread(thread_id)

    asyncio.run(run())


def test_legacy_search_history_is_backfilled_into_account_profile():
    async def run():
        service = ConversationService(InMemoryConversationStore())
        tid = build_qq_thread_id("dm", "u2", "u2")
        memory = ConversationMemory(thread_id=tid, channel="qq", user_id="u2")
        memory.search_history = [{
            "original_question": "想吃辣的下饭菜",
            "search_query": "辣 下饭菜",
            "search_request": {"flavors": ["辣"], "scenes": ["下饭"]},
            "recipes": [{"id": "r1", "name": "酸辣白菜"}],
            "created_at": 100.0,
        }]
        await service.store.save(memory, expires_at=9_999_999_999)

        context = await service.food_memory_context(tid)
        assert context["events"][0]["query"] == "辣 下饭菜"
        profile = await service.store.load("profile:qq:u2")
        assert profile is not None
        assert profile.food_history[0]["kind"] == "searched"

    asyncio.run(run())


def test_successful_cooking_is_recorded_with_real_recipe_evidence():
    async def run():
        service = ConversationService(InMemoryConversationStore())
        tid = build_whatsapp_thread_id("dm", "111@s.whatsapp.net", "+111")
        await service.append_turn(tid, "user", "fish please", channel="whatsapp", user_id="+111")
        await service.save_search(tid, "fish please", "fish", {
            "results": [{
                "id": "fish-2",
                "metadata": {
                    "recipe_id": "fish-2", "name": "Garlic Steamed Fish",
                    "ingredients": ["fish", "garlic"], "tags": ["steamed"],
                },
            }],
        })
        await service.record_cooked_recipe(tid, "fish-2", "Garlic Steamed Fish")
        event = (await service.food_memory_context(tid))["events"][-1]
        assert event["kind"] == "cooked"
        assert event["recipe"]["ingredients"] == ["fish", "garlic"]

    asyncio.run(run())


def test_search_history_can_activate_previous_without_new_search():
    async def run():
        service = ConversationService(InMemoryConversationStore())
        tid = "qq:c2c:u1:u1"

        def result(recipe_id, name):
            return {
                "success": True,
                "results": [{"id": recipe_id, "metadata": {"recipe_id": recipe_id, "name": name}}],
                "_recommendation": {"recipe_reasons": {recipe_id: f"推荐{name}"}},
            }

        await service.save_search(tid, "湖南口味且减脂", "湖南 减脂", result("r1", "剁椒鸡胸"))
        await service.save_search(tid, "湖南口味且减脂", "湖南 减脂", result("r2", "小炒牛肉"))
        routing_context = await service.routing_context(tid, scope="route")
        assert routing_context["previous_candidate_count"] == 1
        previous = await service.activate_previous_search(tid)
        assert previous["recipes"][0]["name"] == "剁椒鸡胸"
        memory = await service.load(tid)
        assert len(memory.search_history) == 2
        assert memory.latest_search["recipes"][0]["id"] == "r1"
        rewound_context = await service.routing_context(tid, scope="route")
        assert rewound_context["previous_candidate_count"] == 0

    asyncio.run(run())


def test_search_memory_actions_are_not_recipe_queries():
    assert _search_memory_action("换一批吧，这几个不喜欢") == "change"
    assert _search_memory_action("不喜欢，再来几道吧") == "change"
    assert _search_memory_action("还是之前推荐的更符合，上一次是什么？") == "previous"
    assert _search_memory_action("推荐三道湖南菜") is None
    assert _search_memory_action("Show me more options") == "change"
    assert _search_memory_action("Go back to the previous recommendations") == "previous"
    assert _search_memory_action("上一批推荐中的菜哪一道最省事") is None


def test_omitted_recipe_question_extracts_named_dish():
    assert _extract_omitted_recipe("为什么没有走西红柿炒鸡蛋这道菜") == "西红柿炒鸡蛋"
    assert _extract_omitted_recipe("有没有西红柿炒鸡蛋这个菜") == "西红柿炒鸡蛋"


def test_snapshot_can_be_rendered_without_searching_again():
    snapshot = {
        "recipes": [{
            "id": "r1", "name": "剁椒鸡胸", "image_url": "https://img/r1.jpg",
            "ingredients": ["鸡胸肉"], "tags": ["湘菜"], "recommendation_reason": "香辣但不厚重",
        }],
    }
    result = _snapshot_to_search_result(snapshot)
    assert result["results"][0]["metadata"]["name"] == "剁椒鸡胸"
    assert result["_recommendation"]["recipe_reasons"]["r1"] == "香辣但不厚重"


def test_change_batch_keeps_query_and_previous_batch_does_not_search(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        from app.orchestrator.recommendation_response import RecommendationNarrative

        async def fake_narrative(original_question, *_args, **_kwargs):
            return RecommendationNarrative(
                "已经切回上一批。" if "上一批" in original_question else "给你换了几个不同方向。",
                "", {}, "",
            )

        monkeypatch.setattr(agent_module, "generate_recommendation_narrative", fake_narrative)
        tid = "qq:c2c:u1:u1"

        first = {
            "success": True,
            "results": [
                {"id": "r1", "metadata": {"recipe_id": "r1", "name": "剁椒鸡胸"}},
                {"id": "r2", "metadata": {"recipe_id": "r2", "name": "小炒大红椒"}},
                {"id": "r3", "metadata": {"recipe_id": "r3", "name": "剁椒藕片"}},
            ],
            "_recommendation": {"recipe_reasons": {}},
        }
        await service.save_search(tid, "湖南口味且减脂", "湖南 减脂", first)

        calls = []

        async def fake_search(query, top_k, lang):
            calls.append((query, top_k, lang))
            return {
                "success": True,
                "results": [
                    *first["results"],
                    {"id": "r4", "metadata": {"recipe_id": "r4", "name": "香辣蒸鱼"}},
                    {"id": "r5", "metadata": {"recipe_id": "r5", "name": "辣椒炒菌菇"}},
                    {"id": "r6", "metadata": {"recipe_id": "r6", "name": "湘味蒸茄子"}},
                ],
            }

        monkeypatch.setattr(agent_module, "_run_search_subprocess", fake_search)
        changed = json.loads(await agent_module._search_memory_followup(tid, "换一批吧"))
        assert calls == [("湖南 减脂", 12, "zh")]
        assert changed["data"]["query"] == "湖南 减脂"
        assert [item["name"] for item in changed["data"]["recipes"]] == ["香辣蒸鱼", "辣椒炒菌菇", "湘味蒸茄子"]

        recalled = json.loads(await agent_module._search_memory_followup(tid, "还是上一批更合适"))
        assert len(calls) == 1
        assert [item["name"] for item in recalled["data"]["recipes"]] == ["剁椒鸡胸", "小炒大红椒", "剁椒藕片"]
        assert recalled["data"]["opening"] == "已经切回上一批。"

    asyncio.run(run())


def test_whatsapp_english_previous_options_use_persisted_search(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        from app.orchestrator.recommendation_response import RecommendationNarrative

        async def fake_narrative(*_args, **_kwargs):
            return RecommendationNarrative("Here are the earlier dishes again.", "", {}, "")

        monkeypatch.setattr(agent_module, "generate_recommendation_narrative", fake_narrative)
        tid = build_whatsapp_thread_id("dm", "111@s.whatsapp.net", "+111")

        def result(recipe_id, name):
            return {
                "success": True,
                "results": [{"id": recipe_id, "metadata": {"recipe_id": recipe_id, "name": name}}],
                "_recommendation": {"recipe_reasons": {}},
            }

        await service.save_search(tid, "easy chicken dinner", "easy chicken dinner", result("r1", "Chicken Soup"))
        await service.save_search(tid, "show me more", "easy chicken dinner", result("r2", "Chicken Salad"))

        async def should_not_search(*args, **kwargs):
            raise AssertionError("previous options must not trigger a new search")

        monkeypatch.setattr(agent_module, "_run_search_subprocess", should_not_search)
        response = json.loads(await agent_module._search_memory_followup(tid, "Go back to the previous recommendations"))
        assert response["lang"] == "en"
        assert response["data"]["recipes"][0]["name"] == "Chicken Soup"
        assert response["data"]["opening"] == "Here are the earlier dishes again."

    asyncio.run(run())


def test_change_batch_uses_saved_candidate_pool_without_search(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        tid = "qq:c2c:u2:u2"

        def item(number):
            return {
                "id": f"r{number}",
                "score": 1 - number / 100,
                "metadata": {
                    "recipe_id": f"r{number}",
                    "name": f"候选菜{number}",
                    "tags": ["湘菜", "家常", "炒" if number % 2 else "蒸"],
                },
            }

        pool = [item(number) for number in range(1, 7)]
        first = {
            "success": True,
            "results": pool[:3],
            "_candidate_pool": pool,
            "_search_request": {
                "original_text": "湖南减脂菜", "query": "湖南 减脂", "cuisines": ["湘菜"], "scenes": ["减脂"]
            },
            "_recommendation": {"recipe_reasons": {}},
        }
        await service.save_search(tid, "湖南减脂菜", "湖南 减脂", first)

        async def should_not_search(*args, **kwargs):
            raise AssertionError("候选池足够时不应再次请求搜索服务")

        monkeypatch.setattr(agent_module, "_run_search_subprocess", should_not_search)
        changed = json.loads(
            await agent_module._search_memory_followup(
                tid,
                "不喜欢，再来几道吧",
            )
        )
        assert [item["id"] for item in changed["data"]["recipes"]] == ["r4", "r5", "r6"]

    asyncio.run(run())


def test_change_batch_with_new_constraint_researches_using_refined_request(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        from app.orchestrator.recommendation_response import RecommendationNarrative

        async def fake_narrative(*_args, **_kwargs):
            return RecommendationNarrative("这批按新条件重新挑过。", "", {}, "")

        monkeypatch.setattr(agent_module, "generate_recommendation_narrative", fake_narrative)
        tid = "qq:c2c:u3:u3"

        def item(number, name):
            return {"id": f"r{number}", "metadata": {"recipe_id": f"r{number}", "name": name, "tags": ["湘菜"]}}

        pool = [item(number, f"原候选{number}") for number in range(1, 7)]
        await service.save_search(tid, "湖南减脂菜", "湖南 减脂", {
            "success": True,
            "results": pool[:3],
            "_candidate_pool": pool,
            "_search_request": {
                "original_text": "湖南减脂菜", "query": "湖南 减脂", "cuisines": ["湘菜"], "scenes": ["减脂"]
            },
            "_recommendation": {"recipe_reasons": {}},
        })
        searched = []

        async def fake_search(query, top_k, lang):
            searched.append(query)
            return {
                "success": True,
                "results": [item(7, "清蒸辣椒鱼"), item(8, "清炒菌菇"), item(9, "湘味蒸茄子")],
            }

        monkeypatch.setattr(agent_module, "_run_search_subprocess", fake_search)
        changed = json.loads(await agent_module._search_memory_followup(tid, "换一批，可以稍微清淡一点"))
        assert len(searched) == 1
        assert "湖南 减脂" in searched[0]
        assert "清淡" in searched[0]
        assert changed["data"]["opening"] == "这批按新条件重新挑过。"

    asyncio.run(run())


def test_candidate_choice_uses_recent_options_and_conversation_context(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        tid = "qq:c2c:warm:warm"
        await service.append_turn(tid, "user", "一点都不下饭，换一批", channel="qq", user_id="warm")
        await service.save_search(tid, "一点都不下饭，换一批", "牛肉 下饭", {
            "success": True,
            "results": [
                {"id": "r1", "metadata": {"recipe_id": "r1", "name": "牛肉烩白菜", "tags": ["咸鲜", "炖"]}},
                {"id": "r2", "metadata": {"recipe_id": "r2", "name": "干拌牛肉", "tags": ["香辣", "凉拌"]}},
                {"id": "r3", "metadata": {"recipe_id": "r3", "name": "花生酱焖牛肉", "tags": ["焖"]}},
            ],
            "_recommendation": {"recipe_reasons": {}},
        })
        await service.append_turn(
            tid, "user", "昨天晚上喝酒了，这三道菜选哪一道推荐给我。", channel="qq", user_id="warm",
        )

        response = json.loads(await agent_module._recent_candidate_followup(
            tid, "昨天晚上喝酒了，这三道菜选哪一道推荐给我。"
        ))
        assert response["type"] == "cooking"
        assert "牛肉烩白菜" in response["message"]
        assert "真实菜谱库" not in response["message"]
        assert "健康作用" in response["message"]
        assert not any(word in response["message"] for word in ("解酒", "暖胃", "养胃"))

    asyncio.run(run())


def test_three_consecutive_greetings_use_different_llm_replies(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    replies = iter([
        "早呀，今天这声招呼很有精神。",
        "又一声早安收到，看来今天要双倍清醒了。",
        "三连早上好达成，我已经完全醒透啦 😄",
    ])

    class Reply:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        prompts = []

        async def ainvoke(self, messages):
            self.prompts.append(messages[0].content)
            return Reply(next(replies))

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        monkeypatch.setattr(agent_module, "_qa_llm", FakeLLM())
        monkeypatch.setattr(agent_module, "greeting_time_context", lambda *_args, **_kwargs: {
            "timezone": "Asia/Shanghai",
            "period": "morning",
            "period_zh": "上午",
            "period_en": "morning",
            "user_period": "morning",
            "mismatch": False,
        })
        tid = "qq:c2c:returning:returning"
        answers = []
        for question in ("早上好", "早上好呀", "早上好"):
            await service.append_turn(tid, "user", question, channel="qq", user_id="returning")
            response = json.loads(await agent_module._greeting_with_memory(tid, "zh", question))
            answers.append(response["message"])
            await service.append_turn(
                tid, "assistant", response["message"], channel="qq", user_id="returning",
            )

        assert len(set(answers)) == 3
        assert answers[0] in FakeLLM.prompts[1]
        assert answers[1] in FakeLLM.prompts[2]
        assert all("CookClaw" not in answer for answer in answers)

    asyncio.run(run())


def test_time_mismatched_greeting_requires_actual_period_and_has_safe_fallback():
    import app.agent.participle_agent as agent_module

    context = {
        "period": "afternoon",
        "period_zh": "下午",
        "period_en": "afternoon",
        "mismatch": True,
    }
    assert not agent_module._greeting_matches_time("早上好呀！", "zh", context)
    assert agent_module._greeting_matches_time(
        "下午好呀，这声早上好来得有点晚。",
        "zh",
        context,
    )
    assert "下午" in agent_module._greeting_fallback("zh", [], context)


def test_repeated_greeting_generation_retries_once(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    class Reply:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            return Reply("早呀，今天也很精神。" if self.calls == 1 else "第二声早安收到，我换个姿势跟你打招呼。")

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        fake_llm = FakeLLM()
        monkeypatch.setattr(agent_module, "_qa_llm", fake_llm)
        monkeypatch.setattr(agent_module, "greeting_time_context", lambda *_args, **_kwargs: {
            "timezone": "Asia/Shanghai",
            "period": "morning",
            "period_zh": "上午",
            "period_en": "morning",
            "user_period": "morning",
            "mismatch": False,
        })
        tid = "weixin:dm:bot:greeting-retry"
        await service.append_turn(tid, "assistant", "早呀，今天也很精神。")
        await service.append_turn(tid, "user", "早上好呀")

        answer = await agent_module._greeting_answer("早上好呀", tid, "zh")

        assert answer == "第二声早安收到，我换个姿势跟你打招呼。"
        assert fake_llm.calls == 2

    asyncio.run(run())


def test_omitted_recipe_followup_directly_says_not_in_candidate_pool(monkeypatch):
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        tid = "qq:c2c:u4:u4"
        result = {
            "success": True,
            "results": [{
                "id": "r1", "metadata": {"recipe_id": "r1", "name": "番茄虾仁鸡蛋汤"}
            }],
            "_candidate_pool": [{
                "id": "r1", "metadata": {"recipe_id": "r1", "name": "番茄虾仁鸡蛋汤"}
            }],
            "_search_request": {
                "original_text": "我想吃鸡蛋西红柿", "query": "鸡蛋西红柿",
                "dietary_constraints": ["减肥"], "flavors": ["清淡"],
            },
            "_recommendation": {"recipe_reasons": {}},
        }
        await service.save_search(tid, "我想吃鸡蛋西红柿", "鸡蛋西红柿 减肥 清淡", result)
        response = json.loads(await _omitted_recipe_followup(tid, "为什么没有走西红柿炒鸡蛋这道菜"))
        assert response["message"].startswith("刚才那几道里确实没有【西红柿炒鸡蛋】，我给你找偏了。")
        assert "减肥、清淡" in response["message"]
        assert "第1道" not in response["message"]
        assert "召回" not in response["message"]
        assert "候选" not in response["message"]
        assert "不给你绕别的" in response["message"]

    asyncio.run(run())


def test_memory_correction_removes_denied_preferences_before_intent_router(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.orchestrator.intent import IntentCategory, IntentResult
    from app.orchestrator.router import FastPathOutcome

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        tid = "qq:c2c:u5:u5"
        memory = await service.load(tid)
        memory.preferences["dietary_constraints"] = ["减肥", "清淡"]
        memory.latest_search = {
            "original_question": "我想吃鸡蛋西红柿",
            "created_at": time.time(),
            "search_query": "鸡蛋西红柿 减肥 清淡",
            "recipes": [{"id": "r1", "name": "番茄汤"}],
        }
        await service._save(memory)
        routed = []

        async def fake_route(question, **kwargs):
            routed.append((question, kwargs.get("preferences")))
            return FastPathOutcome(
                kind="agent",
                lang="zh",
                intent=IntentResult(related=False, category=IntentCategory.off_topic),
            )

        monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
        response = json.loads(await _memory_correction_followup(tid, "我没说过要减肥清淡"))
        updated = await service.load(tid)
        assert updated.preferences["dietary_constraints"] == []
        assert routed == []
        assert "你说得对" in response["message"]
        assert "不会擅自重新搜索" in response["message"]

    asyncio.run(run())


def test_whatsapp_english_memory_correction_removes_stale_preference(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.orchestrator.intent import IntentCategory, IntentResult
    from app.orchestrator.router import FastPathOutcome

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        tid = build_whatsapp_thread_id("dm", "111@s.whatsapp.net", "+111")
        memory = await service.load(tid, channel="whatsapp", user_id="+111")
        memory.preferences["dietary_constraints"] = ["vegetarian"]
        memory.latest_search = {
            "original_question": "Recommend something for dinner",
            "created_at": time.time(),
            "search_query": "vegetarian dinner",
            "recipes": [{"id": "r1", "name": "Vegetable soup"}],
        }
        await service._save(memory)

        async def fake_route(question, **kwargs):
            return FastPathOutcome(
                kind="agent",
                lang="en",
                intent=IntentResult(related=False, category=IntentCategory.off_topic),
            )

        monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
        response = json.loads(await _memory_correction_followup(tid, "I didn't say I was vegetarian"))
        updated = await service.load(tid)
        assert updated.preferences["dietary_constraints"] == []
        assert "I've removed it" in response["message"]

    asyncio.run(run())


def test_candidate_followup_resolves_second_recipe():
    recipes = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    assert _explanation_recipe_index("第二道为什么适合我？", recipes) == 1
    assert _explanation_recipe_index("第一道菜什么风格？", recipes) == 0


def test_candidate_group_quantity_is_not_mistaken_for_third_recipe():
    recipes = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    question = "这三道里面哪一道最省事？"
    assert _referenced_recipe_indexes(question, recipes) == []
    assert _explanation_recipe_index(question, recipes) is None
    assert _is_candidate_choice_request(question)


def test_semantic_rewrite_normalizes_candidate_typo_without_inventing_constraints():
    raw = "对比一下这三个菜哪知道最省事，我怕麻烦"
    rewrite = rewrite_user_utterance(raw)
    assert rewrite.raw_text == raw
    assert rewrite.normalized_text == "对比一下这三道菜哪一道最省事，希望步骤简单"
    assert rewrite.action_hint == "compare_candidates"
    assert rewrite.changes == (
        "candidate_group_reference", "candidate_ordinal_typo", "effort_preference",
    )


def test_semantic_rewrite_does_not_change_unrelated_mazhi_expression():
    raw = "谁知道这件事？不要麻烦你了"
    rewrite = rewrite_user_utterance(raw)
    assert rewrite.normalized_text == raw
    assert not rewrite.changed


def test_semantic_rewrite_normalizes_suitability_question_typo():
    rewrite = rewrite_user_utterance("我是说这三道菜那一道最适合减肥的人吃")
    assert rewrite.normalized_text == "我是说这三道菜哪一道最适合减肥的人吃"
    assert rewrite.action_hint == "compare_candidates"
    assert _is_candidate_choice_request(rewrite.normalized_text)


def test_effort_comparison_is_shared_by_qq_weixin_and_whatsapp(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)

        async def must_not_search(*args, **kwargs):
            raise AssertionError("候选比较不应该重新进入意图分类或菜谱搜索")

        monkeypatch.setattr(agent_module, "route_fast_path", must_not_search)
        snapshot = {
            "original_question": "我想吃土豆丝",
            "search_query": "土豆丝",
            "recipes": [
                {"id": "1", "name": "慢炖土豆", "tags": ["炖煮", "中等烹饪难度"], "ingredients": ["土豆", "牛肉", "洋葱"]},
                {"id": "2", "name": "清炒土豆丝", "tags": ["快手", "快炒", "家常简餐"], "ingredients": ["土豆", "青椒"]},
                {"id": "3", "name": "焗烤土豆", "tags": ["烘焙", "中等难度"], "ingredients": ["土豆", "奶酪", "黄油"]},
            ],
            "created_at": time.time(),
        }
        thread_ids = (
            build_qq_thread_id("dm", "qq-user", "qq-user"),
            build_weixin_thread_id("bot-default", "wx-user"),
            build_whatsapp_thread_id("dm", "111@s.whatsapp.net", "+111"),
        )
        for thread_id in thread_ids:
            memory = await service.load(thread_id)
            memory.latest_search = dict(snapshot)
            memory.search_history = [dict(snapshot)]
            await service._save(memory)
            response = json.loads(await agent_module.qqbot_chat(
                "对比一下这三个菜哪知道最省事，我怕麻烦", thread_id,
            ))
            message = response["message"]
            assert "清炒土豆丝" in message
            assert "慢炖土豆" in message
            assert "焗烤土豆" in message
            assert "第3道是" not in message
            assert "相对判断" in message

    asyncio.run(run())


def test_web_candidate_typo_is_rewritten_before_intent_router(monkeypatch):
    import app.agent.participle_agent as agent_module

    async def must_not_search(*args, **kwargs):
        raise AssertionError("Web 候选比较不应该重新进入意图分类或菜谱搜索")

    async def run():
        thread_id = "web-semantic-rewrite"
        agent_module.clear_thread(thread_id)
        agent_module.remember_candidates(thread_id, {
            "results": [
                {"id": "1", "metadata": {"recipe_id": "1", "name": "慢炖土豆", "tags": ["炖煮", "中等难度"], "ingredients": ["土豆", "牛肉"]}},
                {"id": "2", "metadata": {"recipe_id": "2", "name": "清炒土豆丝", "tags": ["快手", "快炒"], "ingredients": ["土豆"]}},
                {"id": "3", "metadata": {"recipe_id": "3", "name": "焗烤土豆", "tags": ["烘焙"], "ingredients": ["土豆", "奶酪"]}},
            ],
        }, lang="zh")
        chunks = [chunk async for chunk in agent_module.chat_stream(
            "对比一下这三个菜哪知道最省事，我怕麻烦", thread_id,
        )]
        assert len(chunks) == 1
        assert "清炒土豆丝" in chunks[0]
        assert "相对判断" in chunks[0]
        agent_module.clear_thread(thread_id)

    monkeypatch.setattr(agent_module, "route_fast_path", must_not_search)
    asyncio.run(run())


def test_open_chat_handles_off_topic_on_im_without_forcing_cooking(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.orchestrator.intent import IntentCategory, IntentResult
    from app.orchestrator.router import FastPathOutcome

    async def fake_route(question, **kwargs):
        return FastPathOutcome(
            kind="agent",
            lang="zh",
            intent=IntentResult(related=False, category=IntentCategory.off_topic),
        )

    async def fake_open_chat(question, lang, context=None, **_kwargs):
        assert question == "Python 的装饰器是什么？"
        assert "最近对话" in context
        assert "用户：我最近在学 Python" in context
        return "装饰器是在不修改原函数主体的前提下扩展行为的一种方式。"

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        thread_id = "qq:dm:open-chat:open-chat"
        await service.append_turn(thread_id, "user", "我最近在学 Python")
        await service.append_turn(thread_id, "assistant", "你学到哪部分了？")
        await service.append_turn(thread_id, "user", "Python 的装饰器是什么？")
        response = json.loads(await agent_module.qqbot_chat(
            "Python 的装饰器是什么？", thread_id,
        ))
        assert "装饰器" in response["message"]
        assert "问我菜谱" not in response["message"]

    monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
    monkeypatch.setattr(agent_module, "_smalltalk_answer", fake_open_chat)
    asyncio.run(run())


def test_open_chat_handles_unknown_on_web_with_no_tool_llm(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.orchestrator.intent import IntentCategory, IntentResult
    from app.orchestrator.router import FastPathOutcome

    async def fake_route(question, **kwargs):
        return FastPathOutcome(
            kind="agent",
            lang="zh",
            intent=IntentResult(related=True, category=IntentCategory.unknown),
        )

    async def fake_open_chat(question, lang, context=None, **_kwargs):
        return "可以，我们接着聊这个话题。"

    async def run():
        chunks = [chunk async for chunk in agent_module.chat_stream(
            "接着刚才的话题聊聊", "web-open-chat",
        )]
        assert chunks == ["可以，我们接着聊这个话题。"]

    monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
    monkeypatch.setattr(agent_module, "_smalltalk_answer", fake_open_chat)
    asyncio.run(run())


def test_abandonment_is_resolved_by_state_without_intent_or_device_calls(monkeypatch):
    import app.agent.participle_agent as agent_module

    async def must_not_route(*args, **kwargs):
        raise AssertionError("口头放弃必须在意图模型前按状态处理")

    monkeypatch.setattr(agent_module, "route_fast_path", must_not_route)

    async def run():
        # 无任务：自然收口。
        idle_tid = "abandon-idle"
        agent_module.clear_thread(idle_tid)
        agent_module.clear_active_cooking(idle_tid)
        idle = json.loads(await agent_module.qqbot_chat("好吧，我放弃", idle_tid))
        assert idle["message"] == "好，那先到这里。"

        # 等待搜索条件：只取消本次澄清。
        clarify_tid = "abandon-clarify"
        agent_module.clear_thread(clarify_tid)
        agent_module.set_search_clarification(
            clarify_tid,
            {"original_text": "晚饭吃什么", "query": "晚饭"},
            dimension="available_ingredients",
            lang="zh",
        )
        clarified = json.loads(await agent_module.qqbot_chat("好吧，我放弃", clarify_tid))
        assert "已取消这次选菜" in clarified["message"]
        assert agent_module.get_search_clarification(clarify_tid) is None

        # 等待开火确认：只撤销待确认，不发送设备命令。
        pending_tid = "abandon-pending"
        agent_module.clear_thread(pending_tid)
        agent_module.set_pending(pending_tid, "recipe-1", "红烧肉", lang="zh")
        pending = json.loads(await agent_module.qqbot_chat("好吧，我放弃", pending_tid))
        assert "已取消待确认的设备启动" in pending["message"]
        assert agent_module.get_pending(pending_tid) is None

        # 已运行：口头放弃不能当停止命令，必须要求明确“停止烹饪”。
        active_tid = "abandon-active"
        agent_module.clear_thread(active_tid)
        agent_module.clear_active_cooking(active_tid)
        agent_module.set_active_cooking(active_tid, "recipe-2", "番茄炒蛋", device_id="office", lang="zh")
        active = json.loads(await agent_module.qqbot_chat("好吧，我放弃", active_tid))
        assert "仍在运行" in active["message"]
        assert "停止烹饪" in active["message"]
        assert agent_module.get_active_cooking(active_tid) is not None
        agent_module.clear_active_cooking(active_tid)

    asyncio.run(run())


def test_device_product_question_never_queries_account_device(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.orchestrator.router as router_module

    async def must_not_classify(_question):
        raise AssertionError("产品知识应在意图模型前分流")

    async def must_not_query_device(*args, **kwargs):
        raise AssertionError("产品知识不得查询账号设备实例")

    monkeypatch.setattr(router_module, "classify_intent", must_not_classify)
    monkeypatch.setattr(agent_module, "_device_status_and_format", must_not_query_device)
    monkeypatch.setattr(agent_module, "_device_status_text", must_not_query_device)

    async def run():
        response = json.loads(await agent_module.qqbot_chat(
            "田螺云厨有哪些设备、哪个好、怎么用",
            "device-product-knowledge",
        ))
        assert "Mock" in response["message"]
        assert "不能据此判断哪款最好" in response["message"]
        assert "在线，空闲" not in response["message"]

    asyncio.run(run())


def test_diet_suitability_question_chooses_from_current_candidates(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    class Reply:
        content = "牛肉萝卜汤"

    class FakeLLM:
        async def ainvoke(self, messages):
            return Reply()

    async def must_not_search(*args, **kwargs):
        raise AssertionError("适合度比较不应该重新进入意图分类或菜谱搜索")

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        monkeypatch.setattr(agent_module, "_qa_llm", FakeLLM())
        monkeypatch.setattr(agent_module, "route_fast_path", must_not_search)
        thread_id = build_qq_thread_id("dm", "diet-user", "diet-user")
        memory = await service.load(thread_id)
        memory.latest_search = {
            "original_question": "想喝点儿汤",
            "created_at": time.time(),
            "recipes": [
                {"id": "1", "name": "牛肉萝卜汤", "tags": ["粤菜", "清淡", "炖"], "ingredients": ["牛肉", "萝卜"]},
                {"id": "2", "name": "黄姜牛尾汤", "tags": ["泰式菜", "咸鲜", "炖"], "ingredients": ["牛尾", "黄姜"]},
                {"id": "3", "name": "赤小豆牛肉汤", "tags": ["家常菜", "汤羹"], "ingredients": ["牛肉", "赤小豆"]},
            ],
        }
        await service._save(memory)
        response = json.loads(await agent_module.qqbot_chat(
            "我是说这三道菜那一道最适合减肥的人吃", thread_id,
        ))
        assert "牛肉萝卜汤" in response["message"]
        assert response["type"] == "cooking"

    asyncio.run(run())


def test_previous_batch_effort_question_switches_context_without_resending_cards(monkeypatch):
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        tid = build_weixin_thread_id("bot-default", "wx-previous")

        def result(recipe_id, name, tags, ingredients):
            return {"success": True, "results": [{
                "id": recipe_id,
                "metadata": {"recipe_id": recipe_id, "name": name, "tags": tags, "ingredients": ingredients},
            }]}

        await service.save_search(tid, "我想吃土豆丝", "土豆丝", result(
            "easy", "清炒土豆丝", ["快手", "快炒"], ["土豆", "青椒"],
        ))
        await service.save_search(tid, "省事土豆", "省事 土豆", result(
            "slow", "慢炖土豆", ["炖煮", "中等难度"], ["土豆", "牛肉", "洋葱"],
        ))

        response = json.loads(await _recent_candidate_followup(tid, "上一批推荐中的菜哪一道最省事"))
        assert "清炒土豆丝" in response["message"]
        memory = await service.load(tid)
        assert memory.latest_search["recipes"][0]["id"] == "easy"

    asyncio.run(run())


def test_stale_latest_search_does_not_revive_candidates_for_plain_reference(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation import task_state_workspace as session

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        tid = build_qq_thread_id("dm", "stale-candidate", "stale-candidate")
        memory = await service.load(tid)
        memory.latest_search = {
            "original_question": "推荐两个鸡肉菜",
            "created_at": time.time() - 601,
            "recipes": [
                {"id": "old-1", "name": "旧候选一"},
                {"id": "old-2", "name": "旧候选二"},
            ],
        }
        await service._save(memory)
        session.clear_thread(tid)

        assert await agent_module._hydrate_candidate_context(tid) is None
        assert await _recent_candidate_followup(tid, "为什么推荐第二道？") is None
        assert session.recall_candidate_context(tid) is None

    asyncio.run(run())


def test_why_recommend_second_explains_instead_of_retracting(monkeypatch):
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        tid = "qq:dm:u-explain:u-explain"
        memory = await service.load(tid)
        memory.latest_search = {
            "original_question": "推荐三道简单的鸡肉菜",
            "created_at": time.time(),
            "recipes": [
                {"id": "1", "name": "爆炒鸡肉", "tags": ["香辣", "炒"]},
                {
                    "id": "2", "name": "鸡肉烧猴头菇", "tags": ["家常菜", "咸鲜", "烧"],
                    "ingredients": ["鸡肉", "猴头菇"],
                    "recommendation_reason": "做法方向比较家常，适合日常吃。",
                },
            ],
        }
        await service._save(memory)
        response = json.loads(await _recent_candidate_followup(tid, "为什么推荐第二道？"))
        message = response["message"]
        assert "第2道" in message
        assert "鸡肉烧猴头菇" in message
        assert "咸鲜" in message
        assert "撤掉" not in message
        assert "质疑" not in message

    asyncio.run(run())


def test_english_candidate_choice_is_recognized_before_intent_router(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    class Reply:
        content = "I'd pick [Chicken Soup]. It is the calmer option for this meal, while the fried dish is bolder."

    class FakeLLM:
        async def ainvoke(self, messages):
            return Reply()

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        monkeypatch.setattr(agent_module, "_qa_llm", FakeLLM())
        tid = "qq:dm:u-en-choice:u-en-choice"
        memory = await service.load(tid)
        memory.latest_search = {
            "original_question": "Recommend two chicken dishes",
            "created_at": time.time(),
            "recipes": [
                {"id": "1", "name": "Chicken Soup", "tags": ["soup", "savory"]},
                {"id": "2", "name": "Fried Chicken", "tags": ["fried", "savory"]},
            ],
        }
        await service._save(memory)
        response = json.loads(await _recent_candidate_followup(tid, "Which one would you pick for me?"))
        assert "Chicken Soup" in response["message"]
        assert response["lang"] == "en"
        assert "recipe data" in response["message"]
        assert "calmer option" not in response["message"]

    asyncio.run(run())


def test_candidate_followup_compares_two_recipes_instead_of_explaining_only_first(monkeypatch):
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        tid = "qq:dm:u-compare:u-compare"
        memory = await service.load(tid)
        memory.latest_search = {
            "original_question": "我想吃点家常菜",
            "created_at": time.time(),
            "recipes": [
                {"id": "1", "name": "番茄炒蛋", "tags": ["家常", "炒"], "ingredients": ["番茄", "鸡蛋"]},
                {"id": "2", "name": "番茄鸡蛋汤", "tags": ["家常", "汤"], "ingredients": ["番茄", "鸡蛋", "水"]},
            ],
        }
        await service._save(memory)
        response = json.loads(await _recent_candidate_followup(tid, "前两道菜有什么区别？"))
        assert "番茄炒蛋" in response["message"]
        assert "番茄鸡蛋汤" in response["message"]
        assert "| 对比维度 |" in response["message"]
        assert "### 🍳 小田帮你选" in response["message"]
        assert "最明显的区别" in response["message"]
        assert "菜谱字段" not in response["message"]

    asyncio.run(run())


def test_english_candidate_comparison_never_mixes_chinese(monkeypatch):
    import app.conversation.service as service_module

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        tid = "qq:dm:u-en-compare:u-en-compare"
        memory = await service.load(tid)
        memory.latest_search = {
            "original_question": "Recommend two easy chicken dishes",
            "created_at": time.time(),
            "recipes": [
                {
                    "id": "1", "name": "Chicken Soup",
                    "tags": ["清淡", "light", "soup"],
                    "ingredients": ["chicken", "生姜"],
                },
                {
                    "id": "2", "name": "Grilled Chicken",
                    "tags": ["烧烤", "savory", "grilled"],
                    "ingredients": ["chicken", "pepper"],
                },
            ],
        }
        await service._save(memory)
        response = json.loads(await _recent_candidate_followup(
            tid, "What is the difference between the first and second?",
        ))
        assert response["lang"] == "en"
        assert "Chicken Soup" in response["message"]
        assert "Grilled Chicken" in response["message"]
        assert "| Category |" in response["message"]
        assert "### 🍳 My pick" in response["message"]
        assert not any("\u4e00" <= char <= "\u9fff" for char in response["message"])

    asyncio.run(run())


def test_english_qa_rewrites_mixed_language_before_return(monkeypatch):
    import app.agent.participle_agent as agent_module

    class Reply:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return Reply("Use medium heat，然后翻面。")
            return Reply("Use medium heat, then turn it over.")

    async def run():
        llm = FakeLLM()
        monkeypatch.setattr(agent_module, "_qa_llm", llm)
        answer = await agent_module._qa_answer("How should I pan-fry it?", "en")
        assert answer == "Use medium heat, then turn it over."
        assert llm.calls == 2

    asyncio.run(run())


def test_qa_requests_natural_markdown_for_qq_and_web_without_changing_plain_default(monkeypatch):
    import app.agent.participle_agent as agent_module

    class Reply:
        content = "## 煎制要点\n\n- 中火预热\n- 下锅后别急着翻面"

    class FakeLLM:
        def __init__(self):
            self.system_prompts = []

        async def ainvoke(self, messages):
            self.system_prompts.append(str(messages[0].content))
            return Reply()

    async def run():
        llm = FakeLLM()
        monkeypatch.setattr(agent_module, "_qa_llm", llm)
        web_answer = await agent_module._qa_answer("怎么煎不粘锅？", "zh", markdown=True)
        assert web_answer.startswith("## 煎制要点")
        assert "支持 Markdown 的 QQ 或 Web 对话" in llm.system_prompts[-1]
        assert "`- **要点名**：具体说明`" in llm.system_prompts[-1]
        assert "只有一两句就直接说" in llm.system_prompts[-1]
        assert "不夹杂没有必要的英文术语" in llm.system_prompts[-1]

        await agent_module._qa_answer("怎么煎不粘锅？", "zh")
        assert "支持 Markdown 的 QQ 或 Web 对话" not in llm.system_prompts[-1]

    asyncio.run(run())


def test_qa_keeps_recent_history_roles_and_current_question_once(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    class Reply:
        content = "装饰器会包裹原函数，并在调用前后增加行为。"

    class FakeLLM:
        def __init__(self):
            self.messages = []

        async def ainvoke(self, messages):
            self.messages = messages
            return Reply()

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        thread_id = build_qq_thread_id("dm", "role-fidelity", "role-fidelity")
        await service.append_turn(thread_id, "user", "我最近在学 Python")
        await service.append_turn(thread_id, "assistant", "你学到哪一部分了？")
        raw_current = "对比一下这三个菜哪知道最省事，我怕麻烦"
        normalized_current = rewrite_user_utterance(raw_current).normalized_text
        assert normalized_current != raw_current
        await service.append_turn(thread_id, "user", raw_current)
        context = await agent_module._conversation_context_for_qa(
            thread_id,
            normalized_current,
            current_turn_text=raw_current,
        )
        llm = FakeLLM()
        monkeypatch.setattr(agent_module, "_qa_llm", llm)

        await agent_module._qa_answer(
            normalized_current,
            "zh",
            context=context,
        )

        assert isinstance(llm.messages[0], SystemMessage)
        assert "我最近在学 Python" not in llm.messages[0].content
        assert "你学到哪一部分了" not in llm.messages[0].content
        assert isinstance(llm.messages[1], HumanMessage)
        assert llm.messages[1].content == "我最近在学 Python"
        assert isinstance(llm.messages[2], AIMessage)
        assert llm.messages[2].content == "你学到哪一部分了？"
        assert isinstance(llm.messages[-1], HumanMessage)
        assert llm.messages[-1].content == normalized_current
        assert all(message.content != raw_current for message in llm.messages)
        assert sum(
            message.content == normalized_current
            for message in llm.messages
        ) == 1

    asyncio.run(run())


def test_qa_filters_legacy_assistant_digest_before_prompt(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store)
        monkeypatch.setattr(service_module, "_service", service)
        thread_id = build_qq_thread_id("dm", "old-digest", "old-digest")
        memory = await service.load(thread_id)
        memory.summary = {
            "conversation_digest": [
                "用户：我喜欢清淡一点",
                "助手：你最喜欢红烧肉",
                "Assistant: You always want beef.",
            ],
        }
        await service._save(memory)

        context = await agent_module._conversation_context_for_qa(
            thread_id,
            "今天吃什么？",
        )

        assert context is not None
        assert context.earlier_digest == ("用户：我喜欢清淡一点",)

    asyncio.run(run())


def test_smalltalk_can_use_same_natural_markdown_rhythm(monkeypatch):
    import app.agent.participle_agent as agent_module

    class Reply:
        content = "听起来你今天确实有点累，晚饭可以先选省心的。\n\n## 今晚怎么选\n\n- **想吃热乎的**：选一份汤面。"

    class FakeLLM:
        def __init__(self):
            self.system_prompts = []

        async def ainvoke(self, messages):
            self.system_prompts.append(str(messages[0].content))
            return Reply()

    async def run():
        llm = FakeLLM()
        monkeypatch.setattr(agent_module, "_qa_llm", llm)
        answer = await agent_module._smalltalk_answer("今天好累，不知道吃什么", "zh", markdown=True)
        assert answer.startswith("听起来你今天确实有点累")
        assert "第一段先直接回应用户" in llm.system_prompts[-1]
        assert "结尾才问一个" in llm.system_prompts[-1]

        await agent_module._smalltalk_answer("今天好累", "zh")
        assert "支持 Markdown 的 QQ 或 Web 对话" not in llm.system_prompts[-1]

    asyncio.run(run())


def test_pending_device_selection_requires_separate_final_confirmation(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    async def must_not_route(*args, **kwargs):
        raise AssertionError("选择设备阶段不应重新进入意图路由")

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
        calls.append((cook_id, device_id))
        return agent_module._cook_msg("started", lang)

    async def run():
        tid = "qq:dm:device-two-stage:device-two-stage"
        agent_module.clear_thread(tid)
        agent_module.set_pending(
            tid, "recipe-1", "金钱蛋",
            devices=[("office", "办公室设备"), ("showroom", "展厅设备")],
            lang="zh",
        )
        await service.save_task_state(
            tid,
            agent_module.snapshot_thread_state(tid),
        )
        agent_module.clear_thread(tid)
        selected = json.loads(await agent_module.qqbot_chat("1", tid))
        assert "再回复“确认”" in selected["message"]
        assert calls == []
        selected_state = await service.load_task_state(tid)
        assert selected_state.pending_device_start["device_id"] == "office"

        confirmed = json.loads(await agent_module.qqbot_chat("确认", tid))
        assert confirmed["message"] == "started"
        assert calls == [("recipe-1", "office")]
        agent_module.clear_thread(tid)

    monkeypatch.setattr(agent_module, "route_fast_path", must_not_route)
    monkeypatch.setattr(agent_module, "_cook_and_format", fake_cook)
    monkeypatch.setattr(
        service_module,
        "_service",
        service,
    )
    asyncio.run(run())


def test_concurrent_affirmations_claim_pending_device_action_once(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    entered = asyncio.Event()
    release = asyncio.Event()
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
        calls.append((cook_id, device_id, kwargs.get("msg_id")))
        entered.set()
        await release.wait()
        return agent_module._cook_msg("started", lang)

    async def run():
        tid = "qq:dm:device-idempotent:device-idempotent"
        agent_module.clear_thread(tid)
        agent_module.clear_device_execution(tid)
        agent_module.set_pending(
            tid,
            "recipe-1",
            "金钱蛋",
            device_id="office",
            lang="zh",
        )
        await service.save_task_state(
            tid,
            agent_module.snapshot_thread_state(tid),
        )
        agent_module.clear_thread(tid)

        first = asyncio.create_task(agent_module.qqbot_chat("嗯", tid))
        await entered.wait()
        duplicate = json.loads(await agent_module.qqbot_chat("嗯", tid))
        release.set()
        completed = json.loads(await first)

        assert completed["message"] == "started"
        assert "在处理中" in duplicate["message"]
        assert len(calls) == 1
        assert calls[0][2] is not None
        agent_module.clear_thread(tid)
        agent_module.clear_device_execution(tid)

    monkeypatch.setattr(agent_module, "_cook_and_format", fake_cook)
    monkeypatch.setattr(
        service_module,
        "_service",
        service,
    )
    asyncio.run(run())


def test_english_device_status_uses_structured_localized_text():
    import app.agent.participle_agent as agent_module

    response = agent_module._format_device_status({
        "ok": True,
        "devices": [
            {
                "ok": True,
                "device_id": "office",
                "name": "办公室设备",
                "status": "在线，运行中(status=2)",
                "data": {"isOnline": 1, "attributes": {"status": 2}},
            },
            {
                "ok": True,
                "device_id": "showroom",
                "name": "展厅设备",
                "status": "在线，空闲",
                "data": {"isOnline": 1, "attributes": {"status": 0}},
            },
        ],
    }, "en")

    assert "Office Device: online, running (status=2)" in response
    assert "Showroom Device: online, idle" in response
    assert not any("一" <= char <= "鿿" for char in response)


def test_confirmed_device_not_ready_returns_fact_and_keeps_confirmation(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    import app.orchestrator.cook as cook_module

    responses = [
        {
            "ok": False,
            "code": "DEVICE_OFFLINE",
            "reason": "设备离线",
            "command_sent": False,
            "readiness": {
                "state": "offline",
                "device_id": "office",
                "name": "办公室设备",
                "status": "离线",
            },
        },
        {
            "ok": False,
            "code": "DEVICE_BUSY",
            "reason": "设备正忙",
            "command_sent": False,
            "readiness": {
                "state": "busy",
                "device_id": "office",
                "name": "办公室设备",
                "status": "在线，运行中(status=2)",
            },
        },
        {
            "ok": False,
            "code": "DEVICE_STATUS_UNKNOWN",
            "reason": "状态未知",
            "command_sent": False,
            "readiness": {
                "state": "unknown",
                "device_id": "office",
                "name": "办公室设备",
                "status": "在线，但忙闲状态未知",
            },
        },
    ]
    service = ConversationService(InMemoryConversationStore())

    async def fake_execute(*args, **kwargs):
        return responses.pop(0)

    async def run():
        tid = "qq:dm:device-readiness:device-readiness"
        expected = [
            ("设备事实状态", "离线", "开启设备"),
            ("设备事实状态", "运行中", "等设备空闲"),
            ("无法确认", "忙闲状态未知", "避免误启动"),
        ]
        for fragments in expected:
            await service.reset_session(tid)
            agent_module.clear_thread(tid)
            agent_module.set_pending(
                tid, "recipe-1", "金钱蛋", device_id="office", lang="zh",
            )
            baseline = await service.load_task_state_record(tid)
            await service.save_task_state(
                tid,
                agent_module.snapshot_thread_state(tid),
                expected_revision=baseline.revision,
                expected_generation=baseline.generation,
            )
            agent_module.clear_thread(tid)
            reply = json.loads(await agent_module.qqbot_chat("确认", tid))
            assert all(fragment in reply["message"] for fragment in fragments)
            assert "没有发送" in reply["message"]
            persisted = await service.load_task_state(tid)
            assert persisted.pending_device_start["cookId"] == "recipe-1"
        agent_module.clear_thread(tid)

    monkeypatch.setattr(cook_module, "execute_cook", fake_execute)
    monkeypatch.setattr(
        service_module,
        "_service",
        service,
    )
    asyncio.run(run())


def test_first_step_question_never_selects_device_or_starts_cooking(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    import app.orchestrator.cook as cook_module
    from app.orchestrator.intent import IntentCategory, IntentResult
    from app.orchestrator.router import FastPathOutcome

    service = ConversationService(InMemoryConversationStore())

    async def fake_route(question, **kwargs):
        return FastPathOutcome(
            kind="agent", lang="zh",
            intent=IntentResult(related=True, category=IntentCategory.recipe_execute),
        )

    async def must_not_qa(*args, **kwargs):
        raise AssertionError("菜谱步骤缺失时不得调用 QA 模型补写")

    async def no_remote_steps(*args, **kwargs):
        return {"ok": True, "code": "OK", "recipe": {"name": "金钱蛋"}}

    async def must_not_touch_device(*args, **kwargs):
        raise AssertionError("教学问题绝不能查询或操作设备")

    async def run():
        tid = "qq:dm:first-step-safe:first-step-safe"
        agent_module.clear_thread(tid)
        agent_module.set_pending(
            tid, "recipe-1", "金钱蛋",
            devices=[("office", "办公室设备"), ("showroom", "展厅设备")],
            lang="zh",
        )
        await service.save_task_state(
            tid,
            agent_module.snapshot_thread_state(tid),
        )
        agent_module.clear_thread(tid)
        response = json.loads(await agent_module.qqbot_chat("一步一步告诉我，第一步做什么？", tid))
        assert "详细用量、火候和步骤" in response["message"]
        assert "不会补写" in response["message"]
        assert "设备启动确认" not in response["message"]
        assert agent_module.get_pending(tid) is None
        agent_module.clear_thread(tid)

    monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
    monkeypatch.setattr(agent_module, "_qa_answer", must_not_qa)
    monkeypatch.setattr(cook_module, "fetch_recipe_details", no_remote_steps)
    monkeypatch.setattr(agent_module, "_cook_and_format", must_not_touch_device)
    monkeypatch.setattr(cook_module, "check_device_status", must_not_touch_device)
    monkeypatch.setattr(cook_module, "execute_cook", must_not_touch_device)
    monkeypatch.setattr(service_module, "_service", service)
    asyncio.run(run())


def test_stop_without_active_task_never_calls_device(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    import app.orchestrator.cook as cook_module
    from app.orchestrator.intent import IntentCategory, IntentResult
    from app.orchestrator.router import FastPathOutcome

    async def fake_route(question, **kwargs):
        return FastPathOutcome(
            kind="agent", lang="en",
            intent=IntentResult(related=True, category=IntentCategory.recipe_execute),
        )

    async def must_not_stop(*args, **kwargs):
        raise AssertionError("无活动任务时不能发送 stop")

    async def run():
        tid = "qq:dm:no-active-stop:no-active-stop"
        agent_module.clear_thread(tid)
        response = json.loads(await agent_module.qqbot_chat("stop cooking", tid))
        assert "did not send a stop command" in response["message"]

    monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
    monkeypatch.setattr(cook_module, "stop_cook", must_not_stop)
    monkeypatch.setattr(
        service_module,
        "_service",
        ConversationService(InMemoryConversationStore()),
    )
    asyncio.run(run())


def test_region_evidence_requires_real_recipe_fields():
    assert _has_region_evidence({"name": "定阳猴头菇", "tags": ["晋菜", "鲜香"]}, "山西")
    assert not _has_region_evidence({"name": "三和菜", "tags": ["家常菜", "清淡"]}, "山西")


def test_questions_are_not_saved_as_user_preferences():
    prefs = update_preferences({}, "山西人喜欢什么口味？")
    assert prefs["likes"] == []
    prefs = update_preferences(prefs, "为什么推荐这几道？山西人喜欢吃这种东西？")
    assert prefs["likes"] == []
    prefs = update_preferences(prefs, "我喜欢辣，但是不想太油")
    assert prefs["likes"] == ["辣"]
    explicit_dislike = update_preferences({}, "我喜欢辣，但是不吃香菜。")
    assert explicit_dislike["likes"] == ["辣"]
    assert explicit_dislike["dislikes"] == ["香菜"]
    prefs = update_preferences({}, "我想吃点适合湖南人的菜，但是最近在减肥")
    assert prefs["likes"] == []
    assert prefs["dietary_constraints"] == ["减肥"]
    cleaned = update_preferences({"likes": ["点适合湖南人的菜"]}, "好的")
    assert cleaned["likes"] == []
    disputed = update_preferences({}, "我到底在哪里提到过减肥啊？")
    assert disputed["dietary_constraints"] == []
    consultation = update_preferences({}, "第二道菜都有啥材料，减肥可以吃吗？")
    assert consultation["dietary_constraints"] == []
    explicit = update_preferences({}, "我最近正在减肥，想吃点简单的")
    assert explicit["dietary_constraints"] == ["减肥"]
    party = update_preferences({}, "我家里有四口人，帮我推荐几个菜")
    assert party["available_ingredients"] == []


def test_standalone_exact_how_to_is_not_persisted_but_followups_and_facts_are():
    assert not should_persist_exchange("烤红薯怎么做")
    assert not should_persist_exchange("How to make baked sweet potato?")
    assert should_persist_exchange("刚才这道菜怎么做")
    assert should_persist_exchange("烤红薯怎么做？另外我对花生过敏")
    assert should_persist_session_exchange("烤红薯怎么做")
    assert not should_persist_session_exchange("   ")


def test_plain_recipe_question_does_not_enter_account_long_term_digest():
    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store)
        tid = build_qq_thread_id("dm", "plain-question", "plain-question")

        await service.append_turn(
            tid,
            "user",
            "烤红薯怎么做",
            channel="qq",
            user_id="plain-question",
        )

        memory = await service.load(tid)
        profile = await store.load("profile:qq:plain-question")
        assert [turn.content for turn in memory.recent_turns] == ["烤红薯怎么做"]
        assert profile is None

    asyncio.run(run())


def test_preferences_only_routing_context_does_not_load_short_term_store():
    class MustNotLoadConversationStore(InMemoryConversationStore):
        async def load(self, thread_id):
            raise AssertionError("preferences-only scope must not read short-term conversation")

    async def run():
        profile_store = InMemoryConversationStore()
        profile = ConversationMemory(
            thread_id="profile:qq:preference-only",
            channel="qq",
            user_id="preference-only",
        )
        profile.preferences["allergens"] = ["花生"]
        await profile_store.save(profile, expires_at=9_999_999_999)
        service = ConversationService(
            MustNotLoadConversationStore(),
            profile_store=profile_store,
        )

        context = await service.routing_context(
            build_qq_thread_id("dm", "preference-only", "preference-only"),
            scope="preferences",
        )

        assert context["preferences"]["allergens"] == ["花生"]
        assert context["recent_turns"] == []

    asyncio.run(run())


def test_search_clarification_survives_service_recreation_and_can_be_cleared():
    async def run():
        store = InMemoryConversationStore()
        first_service = ConversationService(store)
        tid = build_qq_thread_id("dm", "clarify-persist", "clarify-persist")
        state = {
            "request": {
                "original_text": "午餐吃什么呢",
                "query": "午餐",
                "meals": ["午餐"],
            },
            "dimension": "recommendation_basics",
            "asked_dimensions": ["recommendation_basics"],
            "round_count": 1,
            "lang": "zh",
            "ts": time.time(),
        }

        await first_service.save_search_clarification(tid, state)
        recreated_service = ConversationService(store)

        assert await recreated_service.load_search_clarification(tid) == state
        await recreated_service.clear_search_clarification(tid)
        assert await first_service.load_search_clarification(tid) is None

    asyncio.run(run())


def test_agent_recovers_clarification_state_from_shared_store(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module

    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store)
        monkeypatch.setattr(service_module, "_service", service)
        tid = build_qq_thread_id("dm", "clarify-agent", "clarify-agent")
        state = {
            "request": {"original_text": "晚饭吃什么", "query": "晚饭"},
            "dimension": "available_ingredients",
            "asked_dimensions": ["available_ingredients"],
            "round_count": 1,
            "lang": "zh",
            "ts": time.time(),
        }
        await service.save_search_clarification(tid, state)
        agent_module.clear_search_clarification(tid)

        loaded = await agent_module._load_search_clarification_state(tid)

        assert loaded == state
        assert agent_module.get_search_clarification(tid) == state
        await agent_module._clear_search_clarification_state(tid)
        assert await service.load_search_clarification(tid) is None

    asyncio.run(run())


def test_weixin_party_constraints_reply_uses_shared_task_state_context(monkeypatch):
    # 该测试 pin 旧的固定澄清话术，需关闭 LLM 澄清
    import app.orchestrator.recommendation_response as rec_response
    monkeypatch.setattr(rec_response, "_CLARIFICATION_LLM_ENABLED", False)
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    import app.main as main_module
    import app.orchestrator.router as router
    from app.orchestrator.intent import IntentResult
    from app.orchestrator.recommendation_response import RecommendationNarrative
    from app.conversation import task_state_workspace as session

    class Event:
        from_user_id = "wx-context-user"
        account_id = "bot-context"
        image_path = ""
        voice_path = ""
        file_path = ""
        video_path = ""
        context_token = "ctx-context"

        def __init__(self, text, message_id):
            self.text = text
            self.message_id = message_id

    class Adapter:
        def __init__(self):
            self.texts = []

        async def send_message(self, **kwargs):
            self.texts.append(kwargs)
            return {"success": True}

    async def classify(question, _routing_context=None):
        if "两个人" in question:
            raw = {
                "r": True,
                "c": "recipe_recommend",
                "s": {
                    "q": "晚饭 辣",
                    "meal": ["晚饭"],
                    "flavor": ["辣"],
                },
            }
        else:
            raw = {"r": False, "c": "off_topic", "s": {}}
        return IntentResult.from_raw(raw, original_text=question)

    async def menu_plan(request, **_kwargs):
        return {
            "success": True,
            "results": [{
                "id": "spicy-pork-1",
                "metadata": {
                    "recipe_id": "spicy-pork-1",
                    "name": "辣椒炒肉",
                    "ingredients": ["辣椒", "猪肉"],
                    "tags": ["香辣", "下饭"],
                },
            }],
            "_candidate_pool": [],
            "_menu_plan": {
                "queries": [{"query": request.retrieval_query()}],
                "complete": True,
                "errors": [],
            },
        }

    async def empty_narrative(*_args, **_kwargs):
        return RecommendationNarrative("", "", {}, "")

    async def run():
        service = ConversationService(InMemoryConversationStore())
        adapter = Adapter()
        thread_id = build_weixin_thread_id("bot-context", "wx-context-user")
        session.clear_thread(thread_id)

        monkeypatch.setattr(service_module, "_service", service)
        monkeypatch.setattr(main_module, "_wx_adapter", adapter)
        monkeypatch.setattr(router, "classify_intent", classify)
        monkeypatch.setattr(router, "build_menu_plan", menu_plan)
        monkeypatch.setattr(
            router,
            "generate_recommendation_narrative",
            empty_narrative,
        )

        try:
            await main_module._weixin_message_handler(
                Event("两个人吃，都喜欢吃辣的，准备晚饭", 201),
            )
            await service.drain_pending()

            first_state = await service.load_search_clarification(thread_id)
            assert first_state["dimension"] == "party_constraints"
            assert first_state["request"]["party_size"] == 2
            assert first_state["request"]["flavors"] == ["辣"]

            # 清掉进程内副本，证明第二轮是从共享任务存储恢复。
            agent_module.clear_search_clarification(thread_id)
            restored = await agent_module._load_search_clarification_state(thread_id)
            assert restored == first_state

            await main_module._weixin_message_handler(
                Event("没有什么忌口，请推荐", 202)
            )
            await service.drain_pending()

            assert await service.load_search_clarification(thread_id) is None
            persisted = await service.load_task_state(thread_id)
            assert persisted.pending_search_clarification is None

            sent_text = "\n".join(
                str(item.get("text") or "") for item in adapter.texts
            )
            constraint_question = "你们中有人过敏、忌口，或有素食、清真等饮食要求吗？"
            assert sent_text.count(constraint_question) == 1
            assert "辣椒炒肉" in sent_text
        finally:
            session.clear_thread(thread_id)
            await service.close()

    asyncio.run(run())


def test_conversation_backend_timeout_opens_fast_circuit():
    class HangingStore(InMemoryConversationStore):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def load(self, thread_id):
            self.calls += 1
            await asyncio.sleep(1)

    async def run():
        store = HangingStore()
        service = ConversationService(
            store,
            read_timeout_seconds=0.05,
            circuit_failure_threshold=1,
            circuit_cooldown_seconds=10,
        )

        started = time.monotonic()
        try:
            await service.load("qq:dm:slow:slow")
        except TimeoutError:
            pass
        first_elapsed = time.monotonic() - started

        started = time.monotonic()
        try:
            await service.load("qq:dm:slow:slow")
        except Exception as exc:
            assert type(exc).__name__ == "ConversationBackendUnavailable"
        second_elapsed = time.monotonic() - started

        assert store.calls == 1
        assert first_elapsed < 0.2
        assert second_elapsed < 0.02

    asyncio.run(run())


def test_default_conversation_circuit_requires_two_consecutive_failures():
    class HangingStore(InMemoryConversationStore):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def load(self, thread_id):
            self.calls += 1
            await asyncio.sleep(1)

    async def run():
        store = HangingStore()
        service = ConversationService(
            store,
            read_timeout_seconds=0.01,
            circuit_cooldown_seconds=10,
        )

        for _ in range(2):
            try:
                await service.load("qq:dm:slow-default:slow-default")
            except TimeoutError:
                pass

        try:
            await service.load("qq:dm:slow-default:slow-default")
        except Exception as exc:
            assert type(exc).__name__ == "ConversationBackendUnavailable"

        assert store.calls == 2

    asyncio.run(run())


def test_redis_store_disables_long_default_retry_chain(monkeypatch):
    from app.conversation.redis_store import RedisConversationStore

    monkeypatch.setenv("DATA_SERVICE_PROFILE", "local")
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_RETRY_ATTEMPTS", "0")
    store = RedisConversationStore.from_env()
    retry = store.client.connection_pool.connection_kwargs["retry"]
    assert retry._retries == 0
    assert type(retry._backoff).__name__ == "NoBackoff"
    asyncio.run(store.close())


def test_background_turn_queue_is_ordered_and_drainable():
    async def run():
        service = ConversationService(InMemoryConversationStore())
        tid = build_qq_thread_id("dm", "queued", "queued")
        service.enqueue_turn(tid, "user", "推荐鸡肉", channel="qq", user_id="queued")
        service.enqueue_turn(tid, "assistant", "给你三个选择", channel="qq", user_id="queued")
        await service.drain_pending()

        memory = await service.load(tid)
        assert [(turn.role, turn.content) for turn in memory.recent_turns] == [
            ("user", "推荐鸡肉"),
            ("assistant", "给你三个选择"),
        ]

    asyncio.run(run())


def test_background_profile_write_waits_for_assistant_turn():
    async def run():
        short_store = InMemoryConversationStore()
        profile_store = InMemoryConversationStore()
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )
        tid = build_qq_thread_id("dm", "async-pref", "async-pref")
        user_task = service.enqueue_turn(
            tid,
            "user",
            "我不喜欢吃西餐",
            channel="qq",
            user_id="async-pref",
        )
        assert user_task is not None
        await user_task

        session = await short_store.load(tid)
        assert session.preferences["dislikes"] == ["西餐"]
        assert await profile_store.load("profile:qq:async-pref") is None

        service.enqueue_turn(
            tid,
            "assistant",
            "明白了，后续推荐会避开。",
            channel="qq",
            user_id="async-pref",
        )
        await service.drain_pending()
        profile = await profile_store.load("profile:qq:async-pref")
        assert profile.preferences["dislikes"] == ["西餐"]

    asyncio.run(run())


def test_session_write_wait_reports_backend_failure():
    class FailingSaveStore(InMemoryConversationStore):
        async def save(self, memory, expires_at=None):
            raise OSError("redis unavailable")

    async def run():
        service = ConversationService(FailingSaveStore())
        tid = build_qq_thread_id("dm", "failed-write", "failed-write")
        service.enqueue_turn(
            tid,
            "user",
            "继续刚才的菜",
            channel="qq",
            user_id="failed-write",
        )

        assert await service.wait_for_session_writes(tid) is False

    asyncio.run(run())


def test_slow_account_promotion_does_not_block_next_session_user_turn():
    class SlowProfileStore(InMemoryConversationStore):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def save(self, memory, expires_at=None):
            if str(memory.thread_id).startswith("profile:"):
                self.started.set()
                await self.release.wait()
            await super().save(memory, expires_at=expires_at)

    async def run():
        profile_store = SlowProfileStore()
        service = ConversationService(
            InMemoryConversationStore(),
            profile_store=profile_store,
        )
        tid = build_qq_thread_id("dm", "slow-profile", "slow-profile")
        service.enqueue_turn(
            tid,
            "user",
            "我喜欢香菜",
            channel="qq",
            user_id="slow-profile",
        )
        service.enqueue_turn(
            tid,
            "assistant",
            "记住了。",
            channel="qq",
            user_id="slow-profile",
        )
        assert await service.wait_for_session_writes(tid) is True
        await asyncio.wait_for(profile_store.started.wait(), timeout=0.2)

        service.enqueue_turn(
            tid,
            "user",
            "那今天吃什么？",
            channel="qq",
            user_id="slow-profile",
            promote_account_memory=False,
        )
        assert await asyncio.wait_for(
            service.wait_for_session_writes(tid),
            timeout=0.2,
        ) is True
        session = await service.load(tid)
        assert session.recent_turns[-1].content == "那今天吃什么？"

        profile_store.release.set()
        await service.drain_pending()

    asyncio.run(run())


def test_preference_command_updates_session_and_allows_user_query(monkeypatch):
    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(conversation_service_module, "_service", service)
        tid = build_qq_thread_id("dm", "pref-query", "pref-query")

        reply = await handle_preference_memory_turn(
            "我不喜欢吃西餐",
            tid,
        )
        assert reply is not None
        assert "西餐" in reply.message
        assert reply.continue_routing is False
        assert reply.status == "saved"
        assert reply.success is True
        assert reply.state_changes[0].keys == ("dislikes",)
        memory = await service.load(tid)
        assert memory.preferences["dislikes"] == ["西餐"]

        query = await handle_preference_memory_turn(
            "你记住了我哪些偏好",
            tid,
        )
        assert query is not None
        assert "不喜欢或不吃：西餐" in query.message

    asyncio.run(run())


def test_diet_goal_retraction_resumes_matching_pending_recipe_request(monkeypatch):
    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(conversation_service_module, "_service", service)
        tid = build_qq_thread_id("dm", "diet-resume", "diet-resume")

        seeded = await handle_preference_memory_turn(
            "我现在正在减肥",
            tid,
        )
        assert seeded is not None
        await service.save_search_clarification(
            tid,
            {
                "request": {
                    "original_text": "今晚10个人吃饭，请推荐食谱清单",
                    "query": "晚饭 下酒",
                    "party_size": 10,
                    "constraints_confirmed": True,
                    "context_note": {
                        "kind": "remembered_dietary_constraint",
                        "term": "减肥",
                    },
                },
                "dimension": "remembered_dietary_constraint",
                "asked_dimensions": ["remembered_dietary_constraint"],
                "round_count": 1,
                "lang": "zh",
                "ts": time.time(),
            },
        )
        snapshot = await service.load_short_term_runtime(tid)

        reply = await handle_preference_memory_turn(
            "我已经不减肥了。以后给我推荐菜谱不要参考这个标准",
            tid,
            short_term_snapshot=snapshot,
        )

        assert reply is not None
        assert reply.status == "saved"
        assert reply.continue_routing is True
        assert "之前关于减肥的限制已经取消" in reply.message
        memory = await service.load(tid)
        assert "减肥" not in memory.preferences["dietary_constraints"]

        standalone_tid = build_qq_thread_id(
            "dm",
            "diet-standalone",
            "diet-standalone",
        )
        standalone = await handle_preference_memory_turn(
            "我已经不减肥了。以后给我推荐菜谱不要参考这个标准",
            standalone_tid,
            short_term_snapshot=(
                await service.load_short_term_runtime(standalone_tid)
            ),
        )
        assert standalone is not None
        assert standalone.continue_routing is False

    asyncio.run(run())


def test_current_session_preference_overrides_opposite_account_preference():
    merged = ConversationService._merge_preferences(
        {"likes": ["西餐"]},
        {"dislikes": ["西餐"]},
    )
    assert merged["likes"] == []
    assert merged["dislikes"] == ["西餐"]


def test_long_term_preference_correction_updates_fact_instead_of_unioning():
    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store)
        tid = build_qq_thread_id("dm", "pref-correction", "pref-correction")

        service.enqueue_turn(
            tid,
            "user",
            "我不喜欢西餐",
            channel="qq",
            user_id="pref-correction",
        )
        service.enqueue_turn(
            tid,
            "assistant",
            "知道了。",
            channel="qq",
            user_id="pref-correction",
        )
        await service.drain_pending()

        service.enqueue_turn(
            tid,
            "user",
            "我喜欢西餐了",
            channel="qq",
            user_id="pref-correction",
        )
        service.enqueue_turn(
            tid,
            "assistant",
            "好。",
            channel="qq",
            user_id="pref-correction",
        )
        await service.drain_pending()

        profile = await store.load("profile:qq:pref-correction")
        assert profile.preferences["likes"] == ["西餐"]
        assert profile.preferences["dislikes"] == []
        active = {
            (item["type"], item["value"])
            for item in profile.long_term_facts
            if item["status"] == "active"
        }
        assert ("preference_like", "西餐") in active
        assert ("preference_dislike", "西餐") not in active

    asyncio.run(run())


def test_english_whatsapp_preferences_are_extracted_without_negation_pollution():
    prefs = update_preferences({}, "I like spicy food, but I don't eat peanuts. I am vegetarian.")
    assert prefs["likes"] == ["spicy food"]
    assert prefs["dislikes"] == ["peanuts"]
    assert "vegetarian" in prefs["dietary_constraints"]
    corrected = update_preferences(prefs, "I did not say I was vegan")
    assert "vegan" not in corrected["dietary_constraints"]
    transient = update_preferences(prefs, "I don't like these options")
    assert "these options" not in transient["dislikes"]


def test_allergy_statements_are_saved_as_exact_hard_preferences():
    chinese = update_preferences({}, "我花生过敏")
    assert chinese["dislikes"] == ["花生"] and chinese["allergens"] == ["花生"]
    severe = update_preferences({}, "我对花生严重过敏，以后都不能有花生")
    assert severe["dislikes"] == ["花生"] and severe["allergens"] == ["花生"]
    english = update_preferences({}, "I'm severely allergic to peanuts")
    assert english["dislikes"] == ["peanuts"] and english["allergens"] == ["peanuts"]


def test_allergy_is_persisted_to_account_profile(monkeypatch):
    async def run():
        service = ConversationService(InMemoryConversationStore())
        tid = build_qq_thread_id("dm", "allergy-user", "allergy-user")
        await service.append_turn(
            tid, "user", "我对花生严重过敏，以后推荐都不能有花生",
            channel="qq", user_id="allergy-user",
        )
        context = await service.account_memory_context(tid)
        assert context["preferences"]["dislikes"] == ["花生"]
        assert context["preferences"]["allergens"] == ["花生"]

    asyncio.run(run())


def test_whatsapp_handler_writes_user_and_assistant_turns(monkeypatch):
    import app.main as main_module
    import app.conversation.service as service_module

    class Event:
        chat_type = "dm"
        from_jid = "111@s.whatsapp.net"
        from_number = "+111"
        participant = ""
        group_jid = ""
        message_id = "m1"
        media_urls = []
        media_types = []
        text = "I want chicken"

    class Adapter:
        def __init__(self):
            self.sent = []

        async def send_message(self, **kwargs):
            self.sent.append(kwargs)
            return {"success": True}

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        adapter = Adapter()
        monkeypatch.setattr(main_module, "_wa_adapter", adapter)

        async def fake_chat(question, thread_id="default", on_search_start=None):
            assert question == "I want chicken"
            assert thread_id == "whatsapp:dm:111@s.whatsapp.net:+111"
            memory = await service.load(thread_id)
            assert [turn.content for turn in memory.recent_turns] == [
                "I want chicken",
            ]
            return "Here are a few chicken dishes."

        monkeypatch.setattr(main_module, "qqbot_chat", fake_chat)
        await main_module._whatsapp_message_handler(Event())
        await service.drain_pending()

        memory = await service.load("whatsapp:dm:111@s.whatsapp.net:+111")
        assert memory.channel == "whatsapp"
        assert memory.user_id == "+111"
        assert [turn.content for turn in memory.recent_turns] == [
            "I want chicken",
            "Here are a few chicken dishes.",
        ]
        assert adapter.sent[0]["to"] == "111@s.whatsapp.net"

    asyncio.run(run())


def test_qq_exact_how_to_keeps_session_without_promoting_account(monkeypatch):
    import app.main as main_module
    import app.conversation.service as service_module
    from app.qqbot.adapter import MessageEvent, MessageSource, MessageType

    class Adapter:
        def __init__(self):
            self.sent = []
            self.typing = 0

        async def send_typing(self, chat_id):
            self.typing += 1

        async def send(self, chat_id, text, reply_to=None):
            self.sent.append((chat_id, text, reply_to))

    async def run():
        store = InMemoryConversationStore()
        service = ConversationService(store)
        monkeypatch.setattr(service_module, "_service", service)
        adapter = Adapter()
        monkeypatch.setattr(main_module, "_qq_adapter", adapter)

        async def fake_chat(*_args, **_kwargs):
            memory = await service.load("qq:dm:sweet-potato:sweet-potato")
            assert [turn.content for turn in memory.recent_turns] == [
                "烤红薯怎么做",
            ]
            assert await store.load("profile:qq:sweet-potato") is None
            return "真实做法结果"

        monkeypatch.setattr(main_module, "qqbot_chat", fake_chat)
        event = MessageEvent(
            source=MessageSource(
                chat_id="sweet-potato",
                user_id="sweet-potato",
                chat_type="dm",
            ),
            text="烤红薯怎么做",
            message_type=MessageType.TEXT,
            message_id="sweet-potato-message",
        )
        await main_module._qqbot_message_handler(event)
        await service.drain_pending()

        assert adapter.typing == 1
        assert adapter.sent == [
            ("sweet-potato", "真实做法结果", "sweet-potato-message"),
        ]
        memory = await store.load("qq:dm:sweet-potato:sweet-potato")
        assert memory is not None
        assert [turn.content for turn in memory.recent_turns] == [
            "烤红薯怎么做",
            "真实做法结果",
        ]
        assert await store.load("profile:qq:sweet-potato") is None

    asyncio.run(run())


def test_weixin_handler_writes_user_and_assistant_turns(monkeypatch):
    import app.main as main_module
    import app.conversation.service as service_module

    class Event:
        from_user_id = "wx-user"
        account_id = "bot-default"
        message_id = 101
        image_path = ""
        voice_path = ""
        file_path = ""
        video_path = ""
        context_token = "ctx-1"
        text = "我想吃鸡肉"

    class Adapter:
        def __init__(self):
            self.sent = []

        async def send_message(self, **kwargs):
            self.sent.append(kwargs)
            return {"success": True}

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        adapter = Adapter()
        monkeypatch.setattr(main_module, "_wx_adapter", adapter)

        async def fake_chat(question, thread_id="default", on_search_start=None):
            assert question == "我想吃鸡肉"
            assert thread_id == "weixin:dm:bot-default:wx-user"
            memory = await service.load(thread_id)
            assert [turn.content for turn in memory.recent_turns] == [
                "我想吃鸡肉",
            ]
            return "给你推荐几道鸡肉菜。"

        monkeypatch.setattr(main_module, "qqbot_chat", fake_chat)
        await main_module._weixin_message_handler(Event())
        await service.drain_pending()

        memory = await service.load("weixin:dm:bot-default:wx-user")
        assert memory.channel == "weixin"
        assert memory.user_id == "wx-user"
        assert [turn.content for turn in memory.recent_turns] == [
            "我想吃鸡肉",
            "给你推荐几道鸡肉菜。",
        ]
        assert adapter.sent == [{
            "to": "wx-user",
            "text": "给你推荐几道鸡肉菜。",
            "context_token": "ctx-1",
            "account_id": "bot-default",
        }]

    asyncio.run(run())


def _channel_recipe_response():
    return {
        "type": "recipe_search",
        "lang": "zh",
        "data": {
            "opening": "按鸡蛋给你挑了两道。",
            "recipes": [
                {
                    "id": "r1",
                    "name": "番茄炒蛋",
                    "image": "https://images.example.invalid/cook_platform/a.jpg",
                    "ingredients": ["鸡蛋", "番茄"],
                    "tags": ["家常菜", "酸甜"],
                    "recommendation_reason": "省事又下饭。",
                },
                {
                    "id": "r2",
                    "name": "蒸水蛋",
                    "image": "https://images.example.invalid/cook_platform/b.png",
                    "ingredients": ["鸡蛋", "水"],
                    "tags": ["清淡", "蒸"],
                    "recommendation_reason": "口感更轻松。",
                },
            ],
            "closing": "回复 1 或 2 选择。",
        },
    }


def test_qq_recipe_delivery_sends_text_before_image_cards(monkeypatch):
    import app.main as main_module

    class Result:
        success = True

    class Adapter:
        def __init__(self):
            self.events = []

        async def send(self, chat_id, content, reply_to=None):
            self.events.append(("text", content, reply_to))
            return Result()

        async def send_image(self, chat_id, image, caption="", reply_to=None):
            self.events.append(("image", image, caption, reply_to))
            return Result()

    async def no_sleep(_seconds):
        return None

    async def run():
        adapter = Adapter()
        monkeypatch.setattr(main_module, "_qq_adapter", adapter)
        monkeypatch.setattr(main_module.asyncio, "sleep", no_sleep)
        event = type("Event", (), {
            "source": type("Source", (), {"chat_id": "qq-fish-user"})(),
            "message_id": "qq-fish-message",
        })()

        delivered = await main_module._send_qq_recipe_search(
            event,
            _channel_recipe_response(),
        )

        assert delivered is not None
        assert [item[0] for item in adapter.events] == ["text"]
        document = adapter.events[0][1]
        assert document.startswith("## 🍳 推荐菜谱")
        assert document.count("按鸡蛋给你挑了两道。") == 1
        assert document.count("![") == 2
        assert document.index("**1. 番茄炒蛋**") < document.index("![番茄炒蛋")
        assert document.index("![番茄炒蛋") < document.index("**主要食材**：")
        assert "**2. 蒸水蛋**" in document
        assert "{{QQ_RECIPE_IMAGE}}" not in document
        assert "## 怎么选" in document
        assert "回复 1 或 2 选择。" in document

    asyncio.run(run())


def _channel_menu_plan_response():
    roles = ("dish", "dish", "scoped_dish", "soup")
    names = ("芹菜炒牛肉", "番茄炒蛋", "清蒸鲈鱼", "菌菇汤")
    recipes = []
    for index, (name, role) in enumerate(zip(names, roles), 1):
        recipes.append({
            "id": f"menu-{index}",
            "name": name,
            "menu_role": role,
            "image": f"https://images.example.invalid/cook_platform/menu-{index}.jpg",
            "ingredients": [name[:2], "辅料"],
            "tags": ["家常"],
            "recommendation_reason": "来自当前菜单检索结果。",
        })
    return {
        "type": "menu_plan",
        "intent": "menu_plan",
        "lang": "zh",
        "data": {
            "query": "三菜一汤",
            "requested": {"dish": 3, "soup": 1},
            "fulfilled": {"dish": 3, "soup": 1},
            "missing": {"dish": 0, "soup": 0},
            "complete": True,
            "validation_errors": [],
            "recipes": recipes,
            "closing": "想做哪一道，回复编号就行。",
            "lang": "zh",
        },
        "message": "三菜一汤已经按真实菜谱配齐了。",
    }


def test_menu_plan_renderers_keep_all_recipes_and_roles():
    import app.main as main_module

    data = _channel_menu_plan_response()
    markdown = main_module._json_to_markdown(data)
    plaintext = main_module._json_to_plaintext(data)
    delivery = main_module._recipe_search_delivery(data)

    for name in ("芹菜炒牛肉", "番茄炒蛋", "清蒸鲈鱼", "菌菇汤"):
        assert name in markdown
        assert name in plaintext
    assert markdown.startswith("## 🍽️ 推荐菜单")
    assert "### 4. 菌菇汤" in markdown
    assert "- **🍽️ 菜单定位**：汤" in markdown
    assert "（照顾局部偏好的菜）" in plaintext
    assert "想做哪一道" in markdown
    assert delivery is not None
    assert len(delivery["cards"]) == 4
    assert "菌菇汤" in delivery["cards"][-1]["text"]


def test_qq_handler_sends_menu_plan_recipes_instead_of_summary_only(monkeypatch):
    import app.main as main_module
    import app.conversation.service as service_module
    from app.qqbot.adapter import MessageEvent, MessageSource, MessageType

    class Result:
        success = True

    class Adapter:
        def __init__(self):
            self.sent = []
            self.typing = 0

        async def send(self, chat_id, content, reply_to=None):
            self.sent.append({
                "kind": "text",
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
            })
            return Result()

        async def send_image(self, chat_id, image, caption="", reply_to=None):
            self.sent.append({
                "kind": "image",
                "chat_id": chat_id,
                "content": caption,
                "image": image,
                "reply_to": reply_to,
            })
            return Result()

        async def send_typing(self, chat_id):
            self.typing += 1

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        adapter = Adapter()
        monkeypatch.setattr(main_module, "_qq_adapter", adapter)

        async def fake_chat(*_args, on_search_start=None, **_kwargs):
            assert on_search_start is not None
            await on_search_start("三菜一汤", "zh", "我先按三菜一汤给你配。")
            return json.dumps(_channel_menu_plan_response(), ensure_ascii=False)

        monkeypatch.setattr(main_module, "qqbot_chat", fake_chat)
        event = MessageEvent(
            source=MessageSource(
                chat_id="qq-menu-user",
                user_id="qq-menu-user",
                chat_type="dm",
            ),
            text="帮我推荐三菜一汤",
            message_type=MessageType.TEXT,
            message_id="qq-menu-message",
        )
        await main_module._qqbot_message_handler(event)
        await service.drain_pending()

        assert adapter.typing == 1
        assert [item["kind"] for item in adapter.sent] == ["text"]
        assert adapter.sent[0]["content"].count("三菜一汤已经按真实菜谱配齐了") == 1
        assert "想做哪一道，回复编号就行。" in adapter.sent[0]["content"]
        assert adapter.sent[0]["content"].count("![") == 4
        final = "\n".join(item["content"] for item in adapter.sent)
        for name in ("芹菜炒牛肉", "番茄炒蛋", "清蒸鲈鱼", "菌菇汤"):
            assert name in final
        assert "1. 【芹菜炒牛肉】" not in adapter.sent[0]["content"]

        memory = await service.load(
            "qq:dm:qq-menu-user:qq-menu-user",
            channel="qq",
            user_id="qq-menu-user",
        )
        assert "菌菇汤" in memory.recent_turns[-1].content

    asyncio.run(run())


def test_whatsapp_recipe_search_sends_native_image_cards(monkeypatch):
    import app.main as main_module
    import app.conversation.service as service_module

    class Event:
        chat_type = "dm"
        from_jid = "111@s.whatsapp.net"
        from_number = "+111"
        participant = ""
        group_jid = ""
        message_id = "m-card"
        media_urls = []
        media_types = []
        text = "推荐鸡蛋菜"

    class Adapter:
        def __init__(self):
            self.sent = []

        async def send_message(self, **kwargs):
            self.sent.append(kwargs)
            return {"success": True}

    async def run():
        service = ConversationService(InMemoryConversationStore())
        monkeypatch.setattr(service_module, "_service", service)
        adapter = Adapter()
        monkeypatch.setattr(main_module, "_wa_adapter", adapter)

        async def fake_chat(*_args, **_kwargs):
            return json.dumps(_channel_recipe_response(), ensure_ascii=False)

        monkeypatch.setattr(main_module, "qqbot_chat", fake_chat)
        await main_module._whatsapp_message_handler(Event())
        await service.drain_pending()

        assert len(adapter.sent) == 4  # 导语、两张图文卡、结尾
        cards = [item for item in adapter.sent if item.get("media_url")]
        assert [item["media_url"] for item in cards] == [
            "https://images.example.invalid/cook_platform/a.jpg",
            "https://images.example.invalid/cook_platform/b.png",
        ]
        assert "%5F" not in "".join(item["media_url"] for item in cards)
        assert "番茄炒蛋" in cards[0]["text"]
        assert "鸡蛋、番茄" in cards[0]["text"]

        memory = await service.load("whatsapp:dm:111@s.whatsapp.net:+111")
        assert len(memory.recent_turns) == 2
        assert "番茄炒蛋" in memory.recent_turns[-1].content

    asyncio.run(run())


def test_weixin_recipe_search_uploads_images_and_degrades_one_card(monkeypatch, tmp_path):
    import app.main as main_module

    class Event:
        from_user_id = "wx-user"
        account_id = "bot-recipe"
        context_token = "ctx"

    class Adapter:
        def __init__(self):
            self.texts = []
            self.media = []

        async def send_message(self, **kwargs):
            self.texts.append(kwargs)
            return {"success": True}

        async def send_media(self, **kwargs):
            assert __import__("os").path.exists(kwargs["file_path"])
            self.media.append(kwargs)
            return {"success": True}

    downloaded = []

    async def fake_download(url):
        if url.endswith("b.png"):
            raise RuntimeError("download failed")
        path = tmp_path / "recipe-a.jpg"
        path.write_bytes(b"fake-image")
        downloaded.append(str(path))
        return str(path)

    async def run():
        adapter = Adapter()
        monkeypatch.setattr(main_module, "_wx_adapter", adapter)
        monkeypatch.setattr(main_module, "_download_recipe_image", fake_download)

        full_text = await main_module._send_weixin_recipe_search(
            Event(), _channel_recipe_response()
        )

        assert len(adapter.media) == 1
        assert adapter.media[0]["account_id"] == "bot-recipe"
        assert adapter.media[0]["text"] == ""
        assert any("番茄炒蛋" in item["text"] for item in adapter.texts)
        assert any("蒸水蛋" in item["text"] for item in adapter.texts)
        assert any("cook%5Fplatform/b.png" in item["text"] for item in adapter.texts)
        assert "番茄炒蛋" in full_text and "蒸水蛋" in full_text
        assert downloaded and not __import__("os").path.exists(downloaded[0])

    asyncio.run(run())


def test_weixin_recipe_media_timeout_does_not_duplicate_card_text(monkeypatch, tmp_path):
    import httpx
    import app.main as main_module

    class Event:
        from_user_id = "wx-user"
        account_id = "bot-recipe"
        context_token = "ctx"

    class Adapter:
        def __init__(self):
            self.texts = []
            self.media = []

        async def send_message(self, **kwargs):
            self.texts.append(kwargs)
            return {"success": True}

        async def send_media(self, **kwargs):
            # 模拟 sidecar 已接收媒体请求并产生副作用，但 Python 等响应时超时。
            self.media.append(kwargs)
            raise httpx.ReadTimeout("sidecar response arrived too late")

    image_path = tmp_path / "recipe.jpg"
    image_path.write_bytes(b"fake-image")

    async def fake_download(_url):
        return str(image_path)

    data = {
        "type": "recipe_search",
        "lang": "zh",
        "data": {
            "opening": "给你挑了一道。",
            "recipes": [{
                "id": "1524282326738632706",
                "name": "胡萝卜芹菜粥",
                "image": "https://images.example.invalid/cook_platform/porridge.jpg",
                "ingredients": ["胡萝卜", "芹菜", "大米"],
            }],
        },
    }

    async def run():
        adapter = Adapter()
        monkeypatch.setattr(main_module, "_wx_adapter", adapter)
        monkeypatch.setattr(main_module, "_download_recipe_image", fake_download)

        full_text = await main_module._send_weixin_recipe_search(Event(), data)

        matching_texts = [
            item for item in adapter.texts
            if "胡萝卜芹菜粥" in str(item.get("text") or "")
        ]
        assert len(matching_texts) == 1
        assert len(adapter.media) == 1
        assert adapter.media[0]["text"] == ""
        assert "胡萝卜芹菜粥" in full_text
        assert not image_path.exists()

    asyncio.run(run())


def test_whatsapp_login_page_and_qr_image_are_browser_friendly(monkeypatch):
    import app.main as main_module

    class Status:
        connected = False

    class Adapter:
        login_started = False

        async def get_status(self):
            return Status()

        async def get_qr_code(self):
            return "test-whatsapp-qr-payload"

        async def start_qr_login(self):
            self.login_started = True
            return {"success": True}

    async def run():
        adapter = Adapter()
        monkeypatch.setattr(main_module, "_wa_adapter", adapter)
        page = await main_module.whatsapp_login_page()
        assert page.media_type == "text/html"
        assert b"./qr/image" in page.body
        assert b"./status" in page.body
        assert b"./login/qr/start" in page.body
        assert "更换账号".encode() in page.body

        image = await main_module.whatsapp_qr_image()
        assert image.media_type == "image/png"
        assert image.headers["cache-control"].startswith("no-store")

        started = await main_module.whatsapp_login_qr_start()
        assert started == {"success": True}
        assert adapter.login_started is True

    asyncio.run(run())


def test_weixin_adapter_exposes_real_connection_state_and_auth_expiry():
    from app.weixinbot.adapter import WeixinAdapter, WeixinAuthExpiredError

    class Response:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data

        def json(self):
            return self._data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class Client:
        async def get(self, url):
            return Response(200, {
                "status": "ok",
                "started": False,
                "connected": False,
                "authExpired": True,
                "lastError": "session timeout",
                "token": "configured",
            })

        async def post(self, url, json=None):
            return Response(401, {
                "error": "auth_expired",
                "code": -14,
                "message": "session timeout",
            })

    async def run():
        adapter = WeixinAdapter()
        original_client = adapter._client
        adapter._client = Client()
        await original_client.aclose()
        try:
            status = await adapter.get_status()
            assert status.started is False
            assert status.connected is False
            assert status.auth_expired is True
            assert status.last_error == "session timeout"

            try:
                await adapter.start()
                raise AssertionError("过期 token 不应启动成功")
            except WeixinAuthExpiredError:
                pass
        finally:
            await adapter._long_client.aclose()

    asyncio.run(run())


def test_weixin_status_api_returns_connection_details(monkeypatch):
    import app.main as main_module

    class Status:
        started = False
        connected = False
        auth_expired = True
        last_error = "session timeout"
        token = "configured"
        account_count = 2
        started_count = 1
        connected_count = 1
        auth_expired_count = 1
        max_accounts = 10
        accounts = [
            {"accountId": "bot-a", "connected": True},
            {"accountId": "bot-b", "authExpired": True},
        ]

    class Adapter:
        async def get_status(self):
            return Status()

    async def run():
        monkeypatch.setattr(main_module, "_wx_adapter", Adapter())
        result = await main_module.weixin_status()
        assert result == {
            "enabled": True,
            "started": False,
            "connected": False,
            "auth_expired": True,
            "last_error": "session timeout",
            "token": "configured",
            "account_count": 2,
            "started_count": 1,
            "connected_count": 1,
            "auth_expired_count": 1,
            "max_accounts": 10,
            "accounts": [
                {"accountId": "bot-a", "connected": True},
                {"accountId": "bot-b", "authExpired": True},
            ],
        }

    asyncio.run(run())


def test_weixin_session_persists_gateway_and_account(monkeypatch, tmp_path):
    import app.main as main_module

    session_file = tmp_path / ".wx_session.json"
    monkeypatch.setattr(main_module, "_WX_TOKEN_FILE", str(session_file))

    main_module._save_wx_session(
        "token-1",
        base_url="https://wx-gateway.example",
        account_id="bot-1",
        user_id="user-1",
    )

    assert main_module._load_wx_session() == {
        "token": "token-1",
        "base_url": "https://wx-gateway.example",
        "account_id": "bot-1",
        "user_id": "user-1",
        "saved_at": main_module._load_wx_session()["saved_at"],
    }
    assert main_module._load_wx_token() == "token-1"


def test_weixin_session_persists_multiple_accounts(monkeypatch, tmp_path):
    import app.main as main_module

    session_file = tmp_path / ".wx_session.json"
    monkeypatch.setattr(main_module, "_WX_TOKEN_FILE", str(session_file))
    for index in range(10):
        main_module._save_wx_session(
            f"token-{index}",
            account_id=f"bot-{index}",
            user_id=f"owner-{index}",
        )

    sessions = main_module._load_wx_sessions()
    assert len(sessions) == 10
    assert sessions["bot-0"]["token"] == "token-0"
    assert sessions["bot-9"]["user_id"] == "owner-9"
    assert session_file.stat().st_mode & 0o777 == 0o600


def test_weixin_adapter_switches_complete_session():
    from app.weixinbot.adapter import WeixinAdapter

    class Response:
        def raise_for_status(self):
            return None

    class Client:
        payload = None

        async def post(self, url, json):
            self.payload = json
            return Response()

    async def run():
        adapter = WeixinAdapter()
        original_client = adapter._client
        client = Client()
        adapter._client = client
        await original_client.aclose()
        try:
            await adapter.set_token(
                "token-2",
                "https://wx-gateway.example",
                "bot-2",
                "user-2",
            )
            assert client.payload == {
                "token": "token-2",
                "baseUrl": "https://wx-gateway.example",
                "accountId": "bot-2",
                "userId": "user-2",
            }
        finally:
            await adapter._long_client.aclose()

    asyncio.run(run())


def test_weixin_adapter_routes_reply_to_explicit_account():
    from app.weixinbot.adapter import WeixinAdapter

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"messageId": "m-1"}

    class Client:
        payload = None

        async def post(self, url, json):
            self.payload = json
            return Response()

    async def run():
        adapter = WeixinAdapter()
        original_client = adapter._client
        client = Client()
        adapter._client = client
        await original_client.aclose()
        try:
            result = await adapter.send_message(
                "wx-user",
                "hello",
                context_token="ctx",
                account_id="bot-7",
            )
            assert result == {"messageId": "m-1"}
            assert client.payload == {
                "to": "wx-user",
                "text": "hello",
                "contextToken": "ctx",
                "accountId": "bot-7",
            }
        finally:
            await adapter._long_client.aclose()

    asyncio.run(run())
