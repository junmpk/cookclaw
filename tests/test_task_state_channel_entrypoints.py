"""Web 与图片入口接入独立 task-state UoW 的回归测试。"""

from __future__ import annotations

import pytest

from app import main
from app.conversation.service import ConversationService
from app.conversation.store import InMemoryConversationStore
from app.conversation.task_state_repository import (
    RedisDialogueTaskStateRepository,
)
from app.conversation import task_state_workspace as task_workspace
from app.ports import dialogue_state


def _candidate_result(*items: tuple[str, str]) -> dict:
    return {
        "results": [
            {
                "id": recipe_id,
                "metadata": {
                    "recipe_id": recipe_id,
                    "name": name,
                },
            }
            for recipe_id, name in items
        ]
    }


def _service(
    store: InMemoryConversationStore,
) -> ConversationService:
    return ConversationService(
        store,
        profile_store=InMemoryConversationStore(),
    )


@pytest.mark.asyncio
async def test_web_two_turns_restore_independent_task_state_without_legacy_leak():
    store = InMemoryConversationStore()
    thread_id = "web:task-state-entrypoint"
    task_workspace.clear_thread(thread_id)
    task_workspace.clear_active_cooking(thread_id)

    try:
        first_repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: _service(store)
        )
        async with first_repository.turn_scope(thread_id) as first_snapshot:
            assert first_snapshot is not None
            assert first_snapshot.status == "new"
            assert first_snapshot.task_state_status == "new"
            dialogue_state.remember_candidates(
                thread_id,
                _candidate_result(
                    ("web-r1", "番茄炒蛋"),
                    ("web-r2", "清蒸鲈鱼"),
                ),
                lang="zh",
            )

        persisted = await store.load_task_state_record(thread_id)
        assert persisted is not None
        assert persisted.revision == 1
        assert [
            item["cookId"] for item in persisted.state.candidate_recipes
        ] == ["web-r1", "web-r2"]
        assert task_workspace.recall_candidates(thread_id) is None

        # 模拟下一轮由另一个 Service/Repository 实例处理，只共享持久化 Store。
        second_repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: _service(store)
        )
        async with second_repository.turn_scope(thread_id) as second_snapshot:
            assert second_snapshot is not None
            # 本测试只写 task-state，没有写 transcript，因此会话仍是 new；
            # task-state 则由独立 key 正确恢复为 loaded。
            assert second_snapshot.status == "new"
            assert second_snapshot.task_state_status == "loaded"
            selected = dialogue_state.resolve_selection(thread_id, "第二个")
            assert selected is not None
            assert selected["cookId"] == "web-r2"

        assert task_workspace.recall_candidates(thread_id) is None
    finally:
        task_workspace.clear_thread(thread_id)
        task_workspace.clear_active_cooking(thread_id)


@pytest.mark.asyncio
async def test_image_wrapper_commits_task_state_and_cleans_legacy_thread(
    monkeypatch,
):
    import app.conversation.service as service_module

    store = InMemoryConversationStore()
    service = _service(store)
    monkeypatch.setattr(service_module, "_service", service)
    thread_id = "qq:dm:image-uow-success:image-uow-success"
    task_workspace.clear_thread(thread_id)
    task_workspace.clear_active_cooking(thread_id)

    async def fake_image_impl(
        request,
        media_urls,
        short_term_snapshot=None,
    ):
        assert media_urls == ["https://example.com/ingredients.jpg"]
        assert request.thread_id == thread_id
        assert request.utterance == "看看能做什么"
        assert short_term_snapshot is not None
        assert short_term_snapshot.belongs_to(thread_id)
        dialogue_state.remember_candidates(
            thread_id,
            _candidate_result(("image-r1", "番茄炒蛋")),
            lang="zh",
        )
        return "image-ok"

    monkeypatch.setattr(main, "_handle_image_media_impl", fake_image_impl)

    try:
        result = await main._handle_image_media(
            ["https://example.com/ingredients.jpg"],
            chat_id=thread_id,
            user_text="看看能做什么",
        )

        assert result == "image-ok"
        persisted = await store.load_task_state_record(thread_id)
        assert persisted is not None
        assert persisted.revision == 1
        assert persisted.state.candidate_recipes == [
            {"cookId": "image-r1", "name": "番茄炒蛋"}
        ]
        assert task_workspace.recall_candidates(thread_id) is None
    finally:
        task_workspace.clear_thread(thread_id)
        task_workspace.clear_active_cooking(thread_id)


@pytest.mark.asyncio
async def test_image_wrapper_discards_partial_task_state_when_impl_raises(
    monkeypatch,
):
    import app.conversation.service as service_module

    store = InMemoryConversationStore()
    service = _service(store)
    monkeypatch.setattr(service_module, "_service", service)
    thread_id = "qq:dm:image-uow-error:image-uow-error"
    task_workspace.clear_thread(thread_id)
    task_workspace.clear_active_cooking(thread_id)

    async def failing_image_impl(
        request,
        _media_urls,
        short_term_snapshot=None,
    ):
        assert request.thread_id == thread_id
        assert request.utterance == "看看能做什么"
        assert short_term_snapshot is not None
        dialogue_state.remember_candidates(
            thread_id,
            _candidate_result(("partial-r1", "不应提交的菜")),
            lang="zh",
        )
        raise RuntimeError("image implementation failed")

    monkeypatch.setattr(main, "_handle_image_media_impl", failing_image_impl)

    try:
        with pytest.raises(RuntimeError, match="image implementation failed"):
            await main._handle_image_media(
                ["https://example.com/ingredients.jpg"],
                chat_id=thread_id,
                user_text="看看能做什么",
            )

        assert await store.load_task_state_record(thread_id) is None
        assert task_workspace.recall_candidates(thread_id) is None
    finally:
        task_workspace.clear_thread(thread_id)
        task_workspace.clear_active_cooking(thread_id)
