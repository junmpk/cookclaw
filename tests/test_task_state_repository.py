"""独立任务状态 Repository 的 CAS、UoW 与故障边界回归。"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.conversation import task_state_workspace as task_workspace
from app.conversation.models import (
    ConversationMemory,
    ConversationTaskState,
    ConversationTurn,
)
from app.conversation.service import (
    ConversationBackendUnavailable,
    ConversationService,
)
from app.conversation.store import (
    ConversationStoreConflict,
    InMemoryConversationStore,
)
from app.conversation.task_state_repository import (
    RedisDialogueTaskStateRepository,
)
from app.conversation.task_state_store import TaskStateStoreConflict
from app.orchestrator.turn.application_service import TurnApplicationService
from app.orchestrator.turn.runtime_models import TurnRequest
from app.ports import dialogue_state


def _service(store: InMemoryConversationStore) -> ConversationService:
    return ConversationService(store, profile_store=InMemoryConversationStore())


def _candidate_result(*items: tuple[str, str]) -> dict:
    return {
        "results": [
            {
                "id": recipe_id,
                "metadata": {"recipe_id": recipe_id, "name": name},
            }
            for recipe_id, name in items
        ]
    }


def test_strict_cas_conflict_does_not_overwrite_committed_state():
    async def run() -> None:
        store = InMemoryConversationStore()
        first_service = _service(store)
        second_service = _service(store)
        thread_id = "qq:dm:task-cas:task-cas"

        seeded = await first_service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[{"cookId": "seed", "name": "初始菜"}],
                candidate_language="zh",
            ),
            expected_revision=0,
        )
        first_snapshot = await first_service.load_task_state_record(thread_id)
        stale_snapshot = await second_service.load_task_state_record(thread_id)
        assert first_snapshot.revision == stale_snapshot.revision == seeded.revision

        winner = ConversationTaskState.from_dict(first_snapshot.state.to_dict())
        winner.candidate_recipes = [{"cookId": "winner", "name": "赢家菜"}]
        committed = await first_service.commit_task_state(
            thread_id,
            winner,
            expected_revision=first_snapshot.revision,
        )

        stale = ConversationTaskState.from_dict(stale_snapshot.state.to_dict())
        stale.candidate_recipes = [{"cookId": "stale", "name": "旧写入"}]
        with pytest.raises(TaskStateStoreConflict):
            await second_service.commit_task_state(
                thread_id,
                stale,
                expected_revision=stale_snapshot.revision,
            )

        final = await second_service.load_task_state_record(thread_id)
        assert final.revision == committed.revision
        assert final.state.candidate_recipes == [
            {"cookId": "winner", "name": "赢家菜"}
        ]

    asyncio.run(run())


def test_task_state_recovers_across_service_recreation():
    async def run() -> None:
        store = InMemoryConversationStore()
        thread_id = "qq:dm:task-restart:task-restart"
        task_workspace.clear_thread(thread_id)

        first_service = _service(store)
        first_repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: first_service
        )
        async with first_repository.turn_scope(thread_id):
            dialogue_state.remember_candidates(
                thread_id,
                _candidate_result(("r1", "菜一"), ("r2", "菜二")),
                lang="zh",
            )

        # 模拟 worker/进程重建：新 Service 和 Repository 只共享持久化 Store。
        task_workspace.clear_thread(thread_id)
        second_service = _service(store)
        second_repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: second_service
        )
        async with second_repository.turn_scope(thread_id):
            selected = dialogue_state.resolve_selection(thread_id, "第二个")
            assert selected is not None
            assert selected["cookId"] == "r2"
            dialogue_state.set_selected_recipe(thread_id, selected, lang="zh")

        third_service = _service(store)
        restored = await third_service.load_task_state_record(thread_id)
        assert restored.revision == 2
        assert [
            item["cookId"] for item in restored.state.candidate_recipes
        ] == ["r1", "r2"]
        assert restored.state.selected_recipe_id == "r2"
        assert task_workspace.recall_candidates(thread_id) is None

    asyncio.run(run())


def test_turn_uow_does_not_commit_when_handler_raises():
    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        application = TurnApplicationService(task_state_repository=repository)
        thread_id = "qq:dm:task-handler-error:task-handler-error"
        seeded = await service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[{"cookId": "seed", "name": "初始菜"}],
                candidate_language="zh",
            ),
            expected_revision=0,
        )

        async def runtime_loader(_context):
            raise AssertionError("异常 handler 不应加载额外运行时")

        async def failing_handler(_context):
            dialogue_state.remember_candidates(
                thread_id,
                _candidate_result(("partial", "半轮状态")),
                lang="zh",
            )
            raise RuntimeError("handler failed")

        request = TurnRequest(
            utterance="触发异常",
            thread_id=thread_id,
            channel="qq",
            trace_id="task-handler-error",
        )
        with pytest.raises(RuntimeError, match="handler failed"):
            await application.handle(
                request,
                runtime_loader=runtime_loader,
                handlers={"exact_command": failing_handler},
            )

        final = await service.load_task_state_record(thread_id)
        assert final.revision == seeded.revision
        assert final.state.candidate_recipes == [
            {"cookId": "seed", "name": "初始菜"}
        ]

    asyncio.run(run())


def test_two_repositories_claim_pending_device_start_exactly_once():
    async def run() -> None:
        store = InMemoryConversationStore()
        seed_service = _service(store)
        first_service = _service(store)
        second_service = _service(store)
        first_repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: first_service
        )
        second_repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: second_service
        )
        thread_id = "qq:dm:task-claim:task-claim"
        pending = {
            "cookId": "r1",
            "name": "菜一",
            "device_id": "office",
            "action_id": "claim-once",
            "msg_id": 1001,
            "lang": "zh",
            "ts": time.time(),
        }
        seeded = await seed_service.commit_task_state(
            thread_id,
            ConversationTaskState(
                pending_device_start=pending,
                pending_action={
                    "kind": "confirm_device_start",
                    "payload": {"action_id": "claim-once"},
                    "ts": time.time(),
                },
            ),
            expected_revision=0,
        )
        barrier = asyncio.Barrier(2)

        async def claim(repository: RedisDialogueTaskStateRepository):
            async with repository.turn_scope(thread_id):
                await barrier.wait()
                return await repository.claim_pending_device_start(
                    thread_id,
                    expected_action_id="claim-once",
                )

        claims = await asyncio.gather(
            claim(first_repository),
            claim(second_repository),
        )

        assert sum(result.claimed for result in claims) == 1
        assert sorted(result.status for result in claims) == [
            "claimed",
            "in_progress",
        ]
        claimed = next(result for result in claims if result.claimed)
        assert claimed.payload == pending
        final = await seed_service.load_task_state_record(thread_id)
        assert final.revision == seeded.revision + 1
        assert final.state.pending_device_start is None
        assert final.state.pending_action is None
        assert final.state.current_task == "device_execution"
        assert final.state.device_execution is not None
        assert final.state.device_execution["status"] == "dispatching"
        assert final.state.device_execution["action_id"] == "claim-once"
        assert final.state.device_execution["cookId"] == "r1"
        assert final.state.device_execution["device_id"] == "office"
        assert final.state.device_execution["msg_id"] == 1001
        assert final.state.device_execution["confirmation_created_at"] == pending["ts"]
        assert final.state.device_execution["claimed_at"] >= pending["ts"]

    asyncio.run(run())


def test_task_state_presence_includes_device_execution():
    state = ConversationTaskState(
        device_execution={
            "action_id": "presence-action",
            "status": "outcome_unknown",
        }
    )

    assert state.has_data() is True
    assert task_workspace.task_state_has_data(state) is True


def test_unresolved_device_execution_blocks_another_pending_claim_without_write():
    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        thread_id = "qq:dm:task-in-progress:task-in-progress"
        execution = {
            "cookId": "running-recipe",
            "name": "正在下发的菜",
            "device_id": "office",
            "action_id": "running-action",
            "msg_id": 2001,
            "lang": "zh",
            "status": "submitted_unverified",
            "updated_at": time.time(),
            "ts": time.time(),
        }
        pending = {
            "cookId": "next-recipe",
            "name": "另一道菜",
            "device_id": "showroom",
            "action_id": "next-action",
            "msg_id": 2002,
            "lang": "zh",
            "ts": time.time(),
        }
        seeded = await service.commit_task_state(
            thread_id,
            ConversationTaskState(
                pending_device_start=pending,
                pending_action={
                    "kind": "confirm_device_start",
                    "payload": {"action_id": "next-action"},
                    "ts": time.time(),
                },
                device_execution=execution,
            ),
            expected_revision=0,
        )

        claim = await store.claim_pending_device_start_record(
            thread_id,
            expected_action_id="next-action",
            max_age_seconds=300,
            now=time.time(),
        )

        assert claim.status == "in_progress"
        assert claim.payload is None
        assert claim.revision == seeded.revision
        final = await service.load_task_state_record(thread_id)
        assert final.revision == seeded.revision
        assert final.state.pending_device_start == pending
        assert final.state.device_execution == execution

    asyncio.run(run())


def test_expired_pending_claim_does_not_create_device_execution():
    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        thread_id = "qq:dm:task-expired-claim:task-expired-claim"
        pending = {
            "cookId": "expired-recipe",
            "name": "过期菜",
            "device_id": "office",
            "action_id": "expired-action",
            "msg_id": 2003,
            "lang": "zh",
            "ts": time.time() - 301,
        }
        seeded = await service.commit_task_state(
            thread_id,
            ConversationTaskState(pending_device_start=pending),
            expected_revision=0,
        )

        claim = await store.claim_pending_device_start_record(
            thread_id,
            expected_action_id="expired-action",
            max_age_seconds=300,
            now=time.time(),
        )

        assert claim.status == "expired"
        assert claim.payload is None
        final = await service.load_task_state_record(thread_id)
        assert final.revision == seeded.revision + 1
        assert final.state.pending_device_start is None
        assert final.state.device_execution is None

    asyncio.run(run())


def test_nonpersistent_claim_migrates_pending_and_blocks_duplicate(monkeypatch):
    import app.conversation.task_state_repository as repository_module

    monkeypatch.setattr(
        repository_module,
        "supports_persistent_task_state",
        lambda _thread_id: False,
    )
    thread_id = "local-device-execution-claim"
    dialogue_state.clear_thread(thread_id)
    dialogue_state.clear_active_cooking(thread_id)
    dialogue_state.clear_device_execution(thread_id)
    dialogue_state.set_pending(
        thread_id,
        "local-recipe",
        "本地菜",
        device_id="office",
        action_id="local-action",
        msg_id=2004,
        lang="zh",
    )
    repository = RedisDialogueTaskStateRepository()

    async def run() -> None:
        try:
            async with repository.turn_scope(thread_id):
                first = await repository.claim_pending_device_start(
                    thread_id,
                    expected_action_id="local-action",
                )
                duplicate = await repository.claim_pending_device_start(
                    thread_id,
                    expected_action_id="local-action",
                )

                assert first.claimed is True
                assert duplicate.status == "in_progress"
                assert duplicate.payload is None
                execution = dialogue_state.get_device_execution(thread_id)
                assert execution is not None
                assert execution["action_id"] == "local-action"
                assert execution["status"] == "dispatching"
                assert dialogue_state.get_pending(thread_id) is None
        finally:
            dialogue_state.clear_thread(thread_id)
            dialogue_state.clear_active_cooking(thread_id)
            dialogue_state.clear_device_execution(thread_id)

    asyncio.run(run())


def test_dialogue_port_snapshot_preserves_profile_task_fields():
    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        thread_id = "qq:dm:task-profile-fields:task-profile-fields"
        pending_profile_update = {
            "kind": "preferred_name",
            "value": "范老师",
            "ts": time.time(),
        }
        temporary_profile = {
            "preferred_name_set": True,
            "preferred_name": "临时称呼",
            "updated_at": time.time(),
        }
        await service.commit_task_state(
            thread_id,
            ConversationTaskState(
                pending_profile_update=pending_profile_update,
                temporary_profile=temporary_profile,
            ),
            expected_revision=0,
        )

        async with repository.turn_scope(thread_id):
            dialogue_state.remember_candidates(
                thread_id,
                _candidate_result(("r-profile", "画像保护菜")),
                lang="zh",
            )

        final = await service.load_task_state_record(thread_id)
        assert final.state.pending_profile_update == pending_profile_update
        assert final.state.temporary_profile == temporary_profile
        assert final.state.candidate_recipes[0]["cookId"] == "r-profile"

    asyncio.run(run())


@pytest.mark.parametrize("conversation_unavailable", [False, True])
def test_unavailable_task_state_never_falls_back_to_real_legacy_thread(
    conversation_unavailable: bool,
):
    class UnavailableStore(InMemoryConversationStore):
        def __init__(self) -> None:
            super().__init__()
            self.commit_calls = 0

        async def load(self, thread_id: str):
            if conversation_unavailable:
                raise TimeoutError("conversation Redis unavailable")
            return await super().load(thread_id)

        async def load_task_state_record(self, thread_id: str):
            raise TimeoutError("task-state Redis unavailable")

        async def commit_task_state_record(self, *args, **kwargs):
            self.commit_calls += 1
            return await super().commit_task_state_record(*args, **kwargs)

    async def run() -> None:
        store = UnavailableStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        thread_id = "qq:dm:task-unavailable:task-unavailable"
        task_workspace.clear_thread(thread_id)
        task_workspace.remember_candidates(
            thread_id,
            _candidate_result(("legacy-only", "进程旧候选")),
            lang="zh",
        )

        try:
            async with repository.turn_scope(thread_id) as snapshot:
                assert snapshot.task_state_status == "unavailable"
                assert dialogue_state.recall_candidate_context(thread_id) is None
                assert dialogue_state.snapshot_thread_state(
                    thread_id
                ).candidate_recipes == []

            assert store.commit_calls == 0
            legacy = task_workspace.recall_candidates(thread_id)
            assert legacy is not None
            assert legacy[0]["cookId"] == "legacy-only"
        finally:
            task_workspace.clear_thread(thread_id)

    asyncio.run(run())


def test_noop_turn_does_not_advance_existing_task_state_revision():
    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        thread_id = "qq:dm:task-noop:task-noop"
        seeded = await service.commit_task_state(
            thread_id,
            ConversationTaskState(),
            expected_revision=0,
        )

        async with repository.turn_scope(thread_id):
            assert dialogue_state.snapshot_thread_state(
                thread_id
            ).candidate_recipes == []

        final = await store.load_task_state_record(thread_id)
        assert final is not None
        assert final.revision == seeded.revision
        assert final.state.to_dict() == seeded.state.to_dict()

    asyncio.run(run())


def test_deleted_independent_key_never_revives_version_two_conversation_mirror():
    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        thread_id = "qq:dm:task-expired-key:task-expired-key"
        mirrored_state = ConversationTaskState(
            candidate_recipes=[{"cookId": "expired", "name": "过期候选"}],
            candidate_language="zh",
            candidate_updated_at=time.time(),
        )
        memory = ConversationMemory(
            thread_id=thread_id,
            channel="qq",
            user_id="task-expired-key",
            task_state=mirrored_state,
            task_state_storage_version=2,
        )
        await store.save(memory, expires_at=time.time() + 3_600)
        await service.commit_task_state(
            thread_id,
            mirrored_state,
            expected_revision=0,
        )
        await store.delete_task_state_record(thread_id)

        stale_conversation = await store.load(thread_id)
        assert stale_conversation is not None
        assert stale_conversation.task_state_storage_version == 2
        assert stale_conversation.task_state.candidate_recipes[0]["cookId"] == (
            "expired"
        )

        async with repository.turn_scope(thread_id) as snapshot:
            assert snapshot.task_state_status == "new"
            assert snapshot.task_state_revision == 0
            assert snapshot.memory.task_state.candidate_recipes == []
            assert dialogue_state.recall_candidate_context(thread_id) is None

        assert await store.load_task_state_record(thread_id) is None

    asyncio.run(run())


def test_conversation_get_failure_with_missing_task_key_fails_closed_on_mutation():
    class ConversationReadFailureStore(InMemoryConversationStore):
        def __init__(self) -> None:
            super().__init__()
            self.commit_calls = 0

        async def load(self, thread_id: str):
            raise TimeoutError("conversation GET failed")

        async def commit_task_state_record(self, *args, **kwargs):
            self.commit_calls += 1
            return await super().commit_task_state_record(*args, **kwargs)

    async def run() -> None:
        store = ConversationReadFailureStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        thread_id = "qq:dm:task-split-read-failure:task-split-read-failure"

        with pytest.raises(
            ConversationBackendUnavailable,
            match="task state changed while Redis state was unavailable",
        ):
            async with repository.turn_scope(thread_id) as snapshot:
                assert snapshot.status == "unavailable"
                assert snapshot.task_state_status == "unavailable"
                dialogue_state.remember_candidates(
                    thread_id,
                    _candidate_result(("must-not-save", "不能保存")),
                    lang="zh",
                )

        assert store.commit_calls == 0
        assert await store.load_task_state_record(thread_id) is None

    asyncio.run(run())


def test_aborted_turn_scope_never_flushes_partial_state():
    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        thread_id = "qq:dm:abort-uow:abort-uow"
        baseline = ConversationTaskState(
            candidate_recipes=[{"cookId": "keep", "name": "保留候选"}],
            candidate_language="zh",
            candidate_updated_at=time.time(),
        )
        await service.commit_task_state(
            thread_id,
            baseline,
            expected_revision=0,
        )

        async with repository.turn_scope(thread_id):
            dialogue_state.clear_candidate_context(thread_id)
            await repository.abort_current_scope()

        record = await service.load_task_state_record(thread_id)
        assert record.state.candidate_recipes == [
            {"cookId": "keep", "name": "保留候选"}
        ]

    asyncio.run(run())


@pytest.mark.parametrize("task_read_failure", ["unavailable", "invalid"])
def test_split_task_read_failure_hides_version_two_session_override(
    task_read_failure: str,
):
    from app.conversation.task_state_store import TaskStateRecord

    class SplitReadStore(InMemoryConversationStore):
        async def load_task_state_record(self, thread_id: str):
            if task_read_failure == "unavailable":
                raise TimeoutError("independent task GET unavailable")
            return TaskStateRecord(
                thread_id=f"{thread_id}:wrong-key",
                state=ConversationTaskState(
                    temporary_profile={
                        "preferred_name_set": True,
                        "preferred_name": "非法独立任务称呼",
                    }
                ),
                revision=9,
                generation=2,
            )

    async def run() -> None:
        short_store = SplitReadStore()
        profile_store = InMemoryConversationStore()
        user_id = f"split-profile-{task_read_failure}"
        thread_id = f"qq:dm:{user_id}:{user_id}"
        stale_override = {
            "preferred_name_set": True,
            "preferred_name": "旧会话覆盖称呼",
            "updated_at": time.time() - 60,
        }
        await short_store.save(
            ConversationMemory(
                thread_id=thread_id,
                channel="qq",
                user_id=user_id,
                task_state_storage_version=2,
                task_state=ConversationTaskState(
                    temporary_profile=stale_override,
                    candidate_recipes=[
                        {"cookId": "stale-mirror", "name": "旧镜像候选"}
                    ],
                ),
            ),
            expires_at=time.time() + 3_600,
        )
        await profile_store.save(
            ConversationMemory(
                thread_id=f"profile:qq:{user_id}",
                channel="qq",
                user_id=user_id,
                preferred_name="PostgreSQL 称呼",
                version=5,
            ),
            expires_at=time.time() + 3_600,
        )
        service = ConversationService(
            short_store,
            profile_store=profile_store,
        )

        short_snapshot = await service.load_short_term_runtime(thread_id)

        assert short_snapshot.status == "loaded"
        assert short_snapshot.task_state_status == task_read_failure
        assert short_snapshot.task_state_revision == 0
        assert short_snapshot.task_state_generation == 0
        assert (
            short_snapshot.memory.task_state.to_dict()
            == ConversationTaskState().to_dict()
        )

        runtime = await service.load_runtime_memory(
            thread_id,
            short_term_snapshot=short_snapshot,
        )
        profile_context = await service.user_profile_context(
            thread_id,
            runtime_memory=runtime,
        )
        assert runtime.short_term.task_state.temporary_profile == {}
        assert profile_context["persistent_preferred_name"] == "PostgreSQL 称呼"
        assert profile_context["preferred_name"] == "PostgreSQL 称呼"
        assert profile_context["preferred_name_source"] == "postgres"
        assert "旧会话覆盖称呼" not in repr(profile_context)
        assert "非法独立任务称呼" not in repr(profile_context)

        # hydrate 只做失败关闭投影，不篡改 conversation 中待 TTL 回收的兼容镜像。
        persisted = await short_store.load(thread_id)
        assert persisted is not None
        assert persisted.task_state_storage_version == 2
        assert persisted.task_state.temporary_profile == stale_override

    asyncio.run(run())


def test_uow_cas_conflict_three_way_merges_disjoint_fields():
    async def run() -> None:
        store = InMemoryConversationStore()
        local_service = _service(store)
        remote_service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: local_service
        )
        thread_id = "qq:dm:task-three-way-merge:task-three-way-merge"
        seeded = await local_service.commit_task_state(
            thread_id,
            ConversationTaskState(),
            expected_revision=0,
        )
        remote_tool_result = {
            "tool": "recipe_search",
            "success": True,
            "code": "OK",
        }

        async with repository.turn_scope(thread_id):
            dialogue_state.set_active_cooking(
                thread_id,
                "r-active",
                "正在烹饪",
                device_id="office",
                lang="zh",
            )

            remote = await remote_service.load_task_state_record(thread_id)
            remote_after = ConversationTaskState.from_dict(remote.state.to_dict())
            remote_after.last_tool_result = remote_tool_result
            remote_commit = await remote_service.commit_task_state(
                thread_id,
                remote_after,
                expected_revision=remote.revision,
            )

        final = await local_service.load_task_state_record(thread_id)
        assert remote_commit.revision == seeded.revision + 1
        assert final.revision == remote_commit.revision + 1
        assert final.state.last_tool_result == remote_tool_result
        assert final.state.active_cooking is not None
        assert final.state.active_cooking["cookId"] == "r-active"
        assert final.state.active_cooking["device_id"] == "office"
        assert final.state.current_task == "device_cooking"

    asyncio.run(run())


def test_uow_same_field_divergence_raises_explicit_conflict():
    async def run() -> None:
        store = InMemoryConversationStore()
        local_service = _service(store)
        remote_service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: local_service
        )
        thread_id = "qq:dm:task-field-conflict:task-field-conflict"
        await local_service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[
                    {"cookId": "seed-1", "name": "初始一"},
                    {"cookId": "seed-2", "name": "初始二"},
                ],
                candidate_language="zh",
                candidate_updated_at=time.time(),
            ),
            expected_revision=0,
        )

        with pytest.raises(
            TaskStateStoreConflict,
            match="task state field conflict: candidate_recipes",
        ):
            async with repository.turn_scope(thread_id):
                dialogue_state.remember_candidates(
                    thread_id,
                    _candidate_result(
                        ("local-1", "本地一"),
                        ("local-2", "本地二"),
                    ),
                    lang="zh",
                )

                remote = await remote_service.load_task_state_record(thread_id)
                remote_after = ConversationTaskState.from_dict(
                    remote.state.to_dict()
                )
                remote_after.candidate_recipes = [
                    {"cookId": "remote", "name": "远端候选"}
                ]
                await remote_service.commit_task_state(
                    thread_id,
                    remote_after,
                    expected_revision=remote.revision,
                )

        final = await local_service.load_task_state_record(thread_id)
        assert final.revision == 2
        assert final.state.candidate_recipes == [
            {"cookId": "remote", "name": "远端候选"}
        ]

    asyncio.run(run())


def test_stale_turn_cannot_restore_pre_reset_state_after_generation_changes():
    async def run() -> None:
        store = InMemoryConversationStore()
        stale_turn_service = _service(store)
        reset_service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: stale_turn_service
        )
        thread_id = "qq:dm:task-reset-generation:task-reset-generation"
        active_cooking = {
            "cookId": "running-recipe",
            "name": "正在烹饪的菜",
            "device_id": "office",
            "lang": "zh",
            "ts": time.time(),
        }
        seeded = await stale_turn_service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[
                    {"cookId": "candidate-1", "name": "旧候选一"},
                    {"cookId": "candidate-2", "name": "旧候选二"},
                ],
                candidate_language="zh",
                candidate_updated_at=time.time(),
                active_cooking=active_cooking,
            ),
            expected_revision=0,
        )
        assert seeded.generation == 0

        with pytest.raises(
            TaskStateStoreConflict,
            match="task state generation changed; stale turn cannot cross reset",
        ):
            async with repository.turn_scope(thread_id):
                dialogue_state.set_selected_recipe(
                    thread_id,
                    {"cookId": "candidate-2", "name": "旧候选二"},
                    lang="zh",
                )

                reset_state = await reset_service.reset_session(thread_id)
                assert reset_state.active_cooking == active_cooking
                assert reset_state.candidate_recipes == []
                assert reset_state.selected_recipe_id is None

        final = await reset_service.load_task_state_record(thread_id)
        assert final.revision == seeded.revision + 1
        assert final.generation == seeded.generation + 1
        assert final.state.active_cooking == active_cooking
        assert final.state.current_task == "device_cooking"
        assert final.state.candidate_recipes == []
        assert final.state.selected_recipe_id is None
        assert final.state.selected_recipe is None

    asyncio.run(run())


@pytest.mark.parametrize("reset_operation", ["reset_session", "clear"])
def test_stale_conversation_copy_cannot_cross_reset_tombstone(
    reset_operation: str,
):
    async def run() -> None:
        store = InMemoryConversationStore()
        stale_service = _service(store)
        reset_service = _service(store)
        thread_id = "qq:dm:conversation-generation:conversation-generation"
        initial_task_state = ConversationTaskState(
            candidate_recipes=[{"cookId": "old", "name": "旧候选"}],
            candidate_language="zh",
            candidate_updated_at=time.time(),
        )
        await store.save(
            ConversationMemory(
                thread_id=thread_id,
                channel="qq",
                user_id="conversation-generation",
                recent_turns=[
                    ConversationTurn(role="user", content="重置前的聊天")
                ],
                task_state=initial_task_state,
                task_state_storage_version=2,
                version=1,
            ),
            expires_at=time.time() + 3_600,
        )
        await stale_service.commit_task_state(
            thread_id,
            initial_task_state,
            expected_revision=0,
        )

        stale_memory = await stale_service.load(thread_id)
        assert stale_memory.conversation_generation == 0
        assert stale_memory.recent_turns[0].content == "重置前的聊天"
        assert stale_memory.task_state.candidate_recipes[0]["cookId"] == "old"

        await getattr(reset_service, reset_operation)(thread_id)

        stale_memory.recent_turns.append(
            ConversationTurn(role="assistant", content="旧 worker 的迟到回复")
        )
        stale_memory.task_state.candidate_recipes = [
            {"cookId": "revived", "name": "不应复活的候选"}
        ]
        with pytest.raises(
            ConversationStoreConflict,
            match="conversation generation conflict",
        ):
            await stale_service._save(stale_memory)

        tombstone = await store.load(thread_id)
        assert tombstone is not None
        assert tombstone.conversation_generation == 1
        assert tombstone.recent_turns == []
        assert tombstone.task_state.candidate_recipes == []
        task_record = await store.load_task_state_record(thread_id)
        assert task_record is not None
        assert task_record.generation == 1
        assert task_record.state.candidate_recipes == []

    asyncio.run(run())


def test_stale_uow_claim_is_rejected_before_atomic_claim_after_reset():
    class RecordingClaimStore(InMemoryConversationStore):
        def __init__(self) -> None:
            super().__init__()
            self.claim_calls = 0

        async def claim_pending_device_start_record(self, *args, **kwargs):
            self.claim_calls += 1
            return await super().claim_pending_device_start_record(
                *args,
                **kwargs,
            )

    async def run() -> None:
        store = RecordingClaimStore()
        stale_turn_service = _service(store)
        reset_service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: stale_turn_service
        )
        thread_id = "qq:dm:claim-reset-generation:claim-reset-generation"
        pending = {
            "cookId": "candidate-2",
            "name": "旧候选二",
            "device_id": "office",
            "action_id": "pre-reset-action",
            "msg_id": 2002,
            "lang": "zh",
            "ts": time.time(),
        }
        seeded = await stale_turn_service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[
                    {"cookId": "candidate-1", "name": "旧候选一"},
                    {"cookId": "candidate-2", "name": "旧候选二"},
                ],
                candidate_language="zh",
                candidate_updated_at=time.time(),
                pending_device_start=pending,
                pending_action={
                    "kind": "confirm_device_start",
                    "payload": {"action_id": "pre-reset-action"},
                    "ts": time.time(),
                },
            ),
            expected_revision=0,
        )

        with pytest.raises(
            TaskStateStoreConflict,
            match="task state generation changed; stale turn cannot cross reset",
        ):
            async with repository.turn_scope(thread_id):
                dialogue_state.set_selected_recipe(
                    thread_id,
                    {"cookId": "candidate-2", "name": "旧候选二"},
                    lang="zh",
                )
                assert dialogue_state.get_selected_recipe(thread_id) is not None

                await reset_service.reset_session(thread_id)
                await repository.claim_pending_device_start(
                    thread_id,
                    expected_action_id="pre-reset-action",
                )

        final = await reset_service.load_task_state_record(thread_id)
        assert store.claim_calls == 0
        assert final.revision == seeded.revision + 1
        assert final.generation == seeded.generation + 1
        assert final.state.candidate_recipes == []
        assert final.state.selected_recipe_id is None
        assert final.state.selected_recipe is None
        assert final.state.pending_device_start is None
        assert final.state.pending_action is None

    asyncio.run(run())


def test_patch_with_pre_reset_generation_is_rejected_without_reviving_state():
    async def run() -> None:
        store = InMemoryConversationStore()
        stale_patch_service = _service(store)
        reset_service = _service(store)
        thread_id = "qq:dm:patch-reset-generation:patch-reset-generation"
        seeded = await stale_patch_service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[
                    {"cookId": "candidate-1", "name": "旧候选一"},
                    {"cookId": "candidate-2", "name": "旧候选二"},
                ],
                candidate_language="zh",
                candidate_updated_at=time.time(),
            ),
            expected_revision=0,
        )
        stale = await stale_patch_service.load_task_state_record(thread_id)
        stale_after = ConversationTaskState.from_dict(stale.state.to_dict())
        stale_after.selected_recipe_id = "candidate-2"
        stale_after.selected_recipe_name = "旧候选二"
        stale_after.selected_recipe = {
            "cookId": "candidate-2",
            "name": "旧候选二",
        }

        await reset_service.reset_session(thread_id)

        with pytest.raises(
            TaskStateStoreConflict,
            match="task state generation changed; stale patch cannot cross reset",
        ):
            await stale_patch_service.patch_task_state(
                thread_id,
                stale.state,
                stale_after,
                expected_revision=stale.revision,
                expected_generation=stale.generation,
            )

        final = await reset_service.load_task_state_record(thread_id)
        assert final.revision == seeded.revision + 1
        assert final.generation == seeded.generation + 1
        assert final.state.candidate_recipes == []
        assert final.state.selected_recipe_id is None
        assert final.state.selected_recipe is None

    asyncio.run(run())


def test_patch_three_way_merges_non_overlapping_concurrent_changes():
    async def run() -> None:
        store = InMemoryConversationStore()
        local_service = _service(store)
        remote_service = _service(store)
        thread_id = "qq:dm:patch-three-way:patch-three-way"
        seeded = await local_service.commit_task_state(
            thread_id,
            ConversationTaskState(),
            expected_revision=0,
        )
        local_baseline = await local_service.load_task_state_record(thread_id)
        remote_baseline = await remote_service.load_task_state_record(thread_id)
        remote_tool_result = {
            "tool": "recipe_search",
            "success": True,
            "code": "OK",
        }

        remote_after = ConversationTaskState.from_dict(
            remote_baseline.state.to_dict()
        )
        remote_after.last_tool_result = remote_tool_result
        await remote_service.patch_task_state(
            thread_id,
            remote_baseline.state,
            remote_after,
            expected_revision=remote_baseline.revision,
            expected_generation=remote_baseline.generation,
        )

        local_after = ConversationTaskState.from_dict(
            local_baseline.state.to_dict()
        )
        local_after.active_cooking = {
            "cookId": "active-recipe",
            "name": "并发烹饪",
            "device_id": "office",
            "lang": "zh",
            "ts": time.time(),
        }
        merged = await local_service.patch_task_state(
            thread_id,
            local_baseline.state,
            local_after,
            expected_revision=local_baseline.revision,
            expected_generation=local_baseline.generation,
        )

        final = await local_service.load_task_state_record(thread_id)
        assert final.revision == seeded.revision + 2
        assert final.generation == seeded.generation
        assert merged.last_tool_result == remote_tool_result
        assert merged.active_cooking is not None
        assert merged.active_cooking["cookId"] == "active-recipe"
        assert final.state.last_tool_result == remote_tool_result
        assert final.state.active_cooking == merged.active_cooking

    asyncio.run(run())


def test_patch_rejects_same_field_concurrent_divergence():
    async def run() -> None:
        store = InMemoryConversationStore()
        local_service = _service(store)
        remote_service = _service(store)
        thread_id = "qq:dm:patch-field-conflict:patch-field-conflict"
        await local_service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[{"cookId": "seed", "name": "初始候选"}],
                candidate_language="zh",
                candidate_updated_at=time.time(),
            ),
            expected_revision=0,
        )
        local_baseline = await local_service.load_task_state_record(thread_id)
        remote_baseline = await remote_service.load_task_state_record(thread_id)

        remote_after = ConversationTaskState.from_dict(
            remote_baseline.state.to_dict()
        )
        remote_after.candidate_recipes = [
            {"cookId": "remote", "name": "远端候选"}
        ]
        await remote_service.patch_task_state(
            thread_id,
            remote_baseline.state,
            remote_after,
            expected_revision=remote_baseline.revision,
            expected_generation=remote_baseline.generation,
        )

        local_after = ConversationTaskState.from_dict(
            local_baseline.state.to_dict()
        )
        local_after.candidate_recipes = [
            {"cookId": "local", "name": "本地候选"}
        ]
        with pytest.raises(
            TaskStateStoreConflict,
            match="task state field conflict: candidate_recipes",
        ):
            await local_service.patch_task_state(
                thread_id,
                local_baseline.state,
                local_after,
                expected_revision=local_baseline.revision,
                expected_generation=local_baseline.generation,
            )

        final = await remote_service.load_task_state_record(thread_id)
        assert final.revision == 2
        assert final.state.candidate_recipes == [
            {"cookId": "remote", "name": "远端候选"}
        ]

    asyncio.run(run())


def test_claim_flush_rejects_field_conflict_before_atomic_claim():
    class RecordingClaimStore(InMemoryConversationStore):
        def __init__(self) -> None:
            super().__init__()
            self.claim_calls = 0

        async def claim_pending_device_start_record(self, *args, **kwargs):
            self.claim_calls += 1
            return await super().claim_pending_device_start_record(
                *args,
                **kwargs,
            )

    async def run() -> None:
        store = RecordingClaimStore()
        local_service = _service(store)
        remote_service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: local_service
        )
        thread_id = "qq:dm:claim-field-conflict:claim-field-conflict"
        pending = {
            "cookId": "seed-1",
            "name": "初始候选一",
            "device_id": "office",
            "action_id": "same-generation-action",
            "msg_id": 3003,
            "lang": "zh",
            "ts": time.time(),
        }
        seeded = await local_service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[
                    {"cookId": "seed-1", "name": "初始候选一"},
                    {"cookId": "seed-2", "name": "初始候选二"},
                ],
                candidate_language="zh",
                candidate_updated_at=time.time(),
                pending_device_start=pending,
                pending_action={
                    "kind": "confirm_device_start",
                    "payload": {"action_id": "same-generation-action"},
                    "ts": time.time(),
                },
            ),
            expected_revision=0,
        )

        with pytest.raises(
            TaskStateStoreConflict,
            match="task state field conflict: candidate_recipes",
        ):
            async with repository.turn_scope(thread_id):
                dialogue_state.remember_candidates(
                    thread_id,
                    _candidate_result(
                        ("local-1", "本地候选一"),
                        ("local-2", "本地候选二"),
                    ),
                    lang="zh",
                )

                remote = await remote_service.load_task_state_record(thread_id)
                remote_after = ConversationTaskState.from_dict(
                    remote.state.to_dict()
                )
                remote_after.candidate_recipes = [
                    {"cookId": "remote", "name": "权威候选"}
                ]
                await remote_service.commit_task_state(
                    thread_id,
                    remote_after,
                    expected_revision=remote.revision,
                    expected_generation=remote.generation,
                )

                await repository.claim_pending_device_start(
                    thread_id,
                    expected_action_id="same-generation-action",
                )

        final = await remote_service.load_task_state_record(thread_id)
        assert store.claim_calls == 0
        assert final.revision == seeded.revision + 1
        assert final.generation == seeded.generation
        assert final.state.candidate_recipes == [
            {"cookId": "remote", "name": "权威候选"}
        ]
        assert final.state.pending_device_start == pending
        assert final.state.pending_action is not None
        assert final.state.pending_action["kind"] == "confirm_device_start"
        assert final.state.pending_action["payload"] == {
            "action_id": "same-generation-action"
        }

    asyncio.run(run())


def test_successful_claim_flushes_once_then_does_not_recommit_or_revive_pending():
    class RecordingStore(InMemoryConversationStore):
        def __init__(self) -> None:
            super().__init__()
            self.commit_calls = 0
            self.claim_calls = 0

        async def commit_task_state_record(self, *args, **kwargs):
            self.commit_calls += 1
            return await super().commit_task_state_record(*args, **kwargs)

        async def claim_pending_device_start_record(self, *args, **kwargs):
            self.claim_calls += 1
            return await super().claim_pending_device_start_record(
                *args,
                **kwargs,
            )

    async def run() -> None:
        store = RecordingStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        thread_id = "qq:dm:claim-preflush-once:claim-preflush-once"
        pending = {
            "cookId": "candidate-2",
            "name": "候选二",
            "device_id": "office",
            "action_id": "preflush-once-action",
            "msg_id": 4004,
            "lang": "zh",
            "ts": time.time(),
        }
        seeded = await service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[
                    {"cookId": "candidate-1", "name": "候选一"},
                    {"cookId": "candidate-2", "name": "候选二"},
                ],
                candidate_language="zh",
                candidate_updated_at=time.time(),
                pending_device_start=pending,
                pending_action={
                    "kind": "confirm_device_start",
                    "payload": {"action_id": "preflush-once-action"},
                    "ts": time.time(),
                },
            ),
            expected_revision=0,
        )
        store.commit_calls = 0

        async with repository.turn_scope(thread_id):
            dialogue_state.set_selected_recipe(
                thread_id,
                {"cookId": "candidate-2", "name": "候选二"},
                lang="zh",
            )
            claim = await repository.claim_pending_device_start(
                thread_id,
                expected_action_id="preflush-once-action",
            )
            assert claim.claimed is True
            assert claim.payload == pending
            scoped = dialogue_state.snapshot_thread_state(thread_id)
            assert scoped.selected_recipe_id == "candidate-2"
            assert scoped.pending_device_start is None
            assert scoped.pending_action is None
            assert scoped.device_execution is not None
            assert scoped.device_execution["status"] == "dispatching"
            assert scoped.device_execution["action_id"] == "preflush-once-action"

        final = await service.load_task_state_record(thread_id)
        assert store.commit_calls == 1
        assert store.claim_calls == 1
        assert final.revision == seeded.revision + 2
        assert final.generation == seeded.generation
        assert final.state.selected_recipe_id == "candidate-2"
        assert final.state.pending_device_start is None
        assert final.state.pending_action is None
        assert final.state.device_execution is not None
        assert final.state.device_execution["status"] == "dispatching"
        assert final.state.device_execution["action_id"] == "preflush-once-action"
        assert final.state.current_task == "device_execution"

    asyncio.run(run())


def test_turn_repository_does_not_backfill_stale_cache_after_redis_reset():
    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        thread_id = "qq:dm:legacy-reset-empty:legacy-reset-empty"
        task_workspace.clear_thread(thread_id)
        seeded = await service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[
                    {"cookId": "old-1", "name": "旧候选一"},
                    {"cookId": "old-2", "name": "旧候选二"},
                ],
                candidate_language="zh",
                candidate_updated_at=time.time(),
            ),
            expected_revision=0,
        )
        await service.reset_session(thread_id)
        reset_record = await service.load_task_state_record(thread_id)
        assert reset_record.revision == seeded.revision + 1
        assert reset_record.generation == seeded.generation + 1
        assert reset_record.state.candidate_recipes == []

        # 模拟另一 worker 仍持有 reset 前的模块级进程缓存。
        task_workspace.remember_candidates(
            thread_id,
            _candidate_result(("old-1", "旧候选一"), ("old-2", "旧候选二")),
            lang="zh",
        )
        task_workspace.set_selected_recipe(
            thread_id,
            {"cookId": "old-2", "name": "旧候选二"},
            lang="zh",
        )

        try:
            async with repository.turn_scope(thread_id) as snapshot:
                assert snapshot.task_state_generation == reset_record.generation
                assert dialogue_state.recall_candidate_context(thread_id) is None
                assert dialogue_state.get_selected_recipe(thread_id) is None

            final = await service.load_task_state_record(thread_id)
            assert final.revision == reset_record.revision
            assert final.generation == reset_record.generation
            assert final.state.candidate_recipes == []
            assert final.state.selected_recipe_id is None
        finally:
            task_workspace.clear_thread(thread_id)

    asyncio.run(run())


def test_expired_active_search_is_not_revived_by_unrelated_task_patch():
    async def run() -> None:
        store = InMemoryConversationStore()
        service = _service(store)
        repository = RedisDialogueTaskStateRepository(
            service_factory=lambda: service
        )
        thread_id = "qq:dm:active-search-ttl:active-search-ttl"
        stale_updated_at = time.time() - (6 * 60 * 60) - 60
        stale_request = {
            "original_text": "推荐鸡肉菜，不要辣",
            "query": "鸡肉",
            "ingredients": ["鸡肉"],
            "avoid": ["辣"],
        }

        seeded = await service.commit_task_state(
            thread_id,
            ConversationTaskState(
                active_search_request=stale_request,
                active_search_updated_at=stale_updated_at,
            ),
            expected_revision=0,
        )

        # 先从存储层读取，证明专用时间字段没有被通用 updated_at 覆盖。
        raw_seeded = await store.load_task_state_record(thread_id)
        assert raw_seeded is not None
        assert raw_seeded.state.active_search_request == stale_request
        assert raw_seeded.state.active_search_updated_at == stale_updated_at
        assert raw_seeded.state.updated_at > stale_updated_at

        hydrated = await service.load_short_term_runtime(thread_id)
        assert hydrated.memory.task_state.active_search_request == {}
        assert hydrated.memory.task_state.active_search_updated_at is None
        assert hydrated.memory.task_state.current_task is None

        # 画像临时值是无关 patch；它可以推进整份状态的更新时间，但不能让
        # 六小时前的 active search 再次进入运行时。
        await service.save_temporary_preferred_name(thread_id, "小范")
        after_profile_patch = await store.load_task_state_record(thread_id)
        assert after_profile_patch is not None
        assert after_profile_patch.revision == seeded.revision + 1
        assert after_profile_patch.state.temporary_profile["preferred_name"] == "小范"
        assert after_profile_patch.state.updated_at >= raw_seeded.state.updated_at

        hydrated_after_patch = await service.load_short_term_runtime(thread_id)
        assert hydrated_after_patch.memory.task_state.active_search_request == {}
        assert hydrated_after_patch.memory.task_state.active_search_updated_at is None

        # 真正更新 active search 时才推进它自己的时间戳，并保留无关画像字段。
        fresh_request = {
            "original_text": "换成牛肉菜",
            "query": "牛肉",
            "ingredients": ["牛肉"],
        }
        started_at = time.time()
        async with repository.turn_scope(thread_id):
            assert dialogue_state.get_active_search_request(thread_id) is None
            dialogue_state.set_active_search_request(
                thread_id,
                fresh_request,
                lang="zh",
            )

        raw_fresh = await store.load_task_state_record(thread_id)
        assert raw_fresh is not None
        assert raw_fresh.state.active_search_request == fresh_request
        assert raw_fresh.state.active_search_updated_at is not None
        assert raw_fresh.state.active_search_updated_at >= started_at
        assert raw_fresh.state.active_search_updated_at > stale_updated_at
        assert raw_fresh.state.temporary_profile["preferred_name"] == "小范"

        hydrated_fresh = await service.load_short_term_runtime(thread_id)
        assert hydrated_fresh.memory.task_state.active_search_request == fresh_request
        assert (
            hydrated_fresh.memory.task_state.active_search_updated_at
            == raw_fresh.state.active_search_updated_at
        )

    asyncio.run(run())


@pytest.mark.parametrize("reset_operation", ["reset_session", "clear"])
@pytest.mark.parametrize("conflict_source", ["conversation", "task"])
def test_short_term_reset_retries_whole_pair_after_precommit_conflict(
    reset_operation: str,
    conflict_source: str,
):
    class ConflictOnceStore(InMemoryConversationStore):
        def __init__(self) -> None:
            super().__init__()
            self.reset_calls = 0
            self.failed_pair_was_unchanged = False

        async def reset_with_task_state(
            self,
            memory,
            task_state,
            **kwargs,
        ):
            self.reset_calls += 1
            if self.reset_calls != 1:
                return await super().reset_with_task_state(
                    memory,
                    task_state,
                    **kwargs,
                )

            before_memory = await super().load(memory.thread_id)
            before_task = await super().load_task_state_record(memory.thread_id)
            injected = dict(kwargs)
            expected_error: type[Exception]
            if conflict_source == "conversation":
                injected["expected_version"] += 1
                expected_error = ConversationStoreConflict
            else:
                injected["expected_task_revision"] += 1
                expected_error = TaskStateStoreConflict

            try:
                await super().reset_with_task_state(
                    memory,
                    task_state,
                    **injected,
                )
            except expected_error:
                after_memory = await super().load(memory.thread_id)
                after_task = await super().load_task_state_record(
                    memory.thread_id
                )
                self.failed_pair_was_unchanged = bool(
                    before_memory is not None
                    and after_memory is not None
                    and before_memory.to_dict() == after_memory.to_dict()
                    and before_task is not None
                    and after_task is not None
                    and before_task == after_task
                )
                raise
            raise AssertionError("injected reset baseline conflict was not raised")

    async def run() -> None:
        store = ConflictOnceStore()
        service = _service(store)
        thread_id = (
            f"qq:dm:atomic-reset-{reset_operation}-{conflict_source}:"
            f"atomic-reset-{reset_operation}-{conflict_source}"
        )
        await store.save(
            ConversationMemory(
                thread_id=thread_id,
                channel="qq",
                user_id=f"atomic-reset-{reset_operation}-{conflict_source}",
                recent_turns=[
                    ConversationTurn(role="user", content="重置前的聊天")
                ],
                version=4,
            ),
            expires_at=time.time() + 3_600,
        )
        seeded_task = await service.commit_task_state(
            thread_id,
            ConversationTaskState(
                candidate_recipes=[
                    {"cookId": "before-reset", "name": "重置前候选"}
                ],
                candidate_language="zh",
                candidate_updated_at=time.time(),
            ),
            expected_revision=0,
        )
        before_memory = await store.load(thread_id)
        assert before_memory is not None

        reset_state = await getattr(service, reset_operation)(thread_id)

        assert store.reset_calls == 2
        assert store.failed_pair_was_unchanged is True
        assert reset_state.candidate_recipes == []
        after_memory = await store.load(thread_id)
        after_task = await store.load_task_state_record(thread_id)
        assert after_memory is not None
        assert after_task is not None
        assert after_memory.version == before_memory.version + 1
        assert (
            after_memory.conversation_generation
            == before_memory.conversation_generation + 1
        )
        assert after_memory.recent_turns == []
        assert after_memory.task_state.candidate_recipes == []
        assert after_task.revision == seeded_task.revision + 1
        assert after_task.generation == seeded_task.generation + 1
        assert after_task.state.candidate_recipes == []

    asyncio.run(run())


def test_redis_task_commit_uses_lua_even_when_version_cas_is_disabled():
    import json

    from app.conversation.redis_store import RedisConversationStore
    from app.conversation.task_state_store import TaskStateRecord

    class RecordingClient:
        def __init__(self) -> None:
            self.eval_calls: list[tuple] = []
            self.get_calls = 0
            self.set_calls = 0

        async def eval(self, *args):
            self.eval_calls.append(args)
            return [1, 8, 3]

        async def get(self, *_args):
            self.get_calls += 1
            raise AssertionError("task CAS must not fall back to GET")

        async def set(self, *_args, **_kwargs):
            self.set_calls += 1
            raise AssertionError("task CAS must not fall back to SET")

    async def run() -> None:
        client = RecordingClient()
        store = RedisConversationStore(
            client,
            key_prefix="cookclaw:test",
            enforce_version=False,
        )
        thread_id = "qq:dm:redis-task-lua:redis-task-lua"
        committed = await store.commit_task_state_record(
            TaskStateRecord(
                thread_id=thread_id,
                state=ConversationTaskState(
                    candidate_recipes=[
                        {"cookId": "lua-1", "name": "Lua 候选"}
                    ],
                    candidate_language="zh",
                ),
                revision=7,
                generation=3,
            ),
            expected_revision=7,
            expected_generation=3,
            new_generation=None,
            expires_at=time.time() + 3_600,
        )

        assert client.get_calls == 0
        assert client.set_calls == 0
        assert len(client.eval_calls) == 1
        script, key_count, task_key, conversation_key, payload, *arguments = (
            client.eval_calls[0]
        )
        assert key_count == 2
        assert "local next_revision = expected_revision + 1" in script
        assert task_key == store.task_state_key_for(thread_id)
        assert conversation_key == store.key_for(thread_id)
        assert json.loads(payload)["state"]["candidate_recipes"][0]["cookId"] == "lua-1"
        assert arguments[0] == 7
        assert arguments[3] == 3
        assert arguments[4] == -1
        assert committed.revision == 8
        assert committed.generation == 3

    asyncio.run(run())


def test_claim_lua_empty_mappings_match_python_derived_state_contract():
    from app.conversation.redis_store import _CLAIM_TASK_STATE_PENDING_DEVICE
    from app.conversation.task_state_store import refresh_task_state_derived

    now = time.time()
    pending = {
        "action_id": "empty-mapping-parity",
        "device_id": "office",
        "lang": "zh",
        "ts": now,
    }
    python_state = ConversationTaskState(
        pending_device_start=pending,
        pending_action={
            "kind": "confirm_device_start",
            "payload": {"action_id": "empty-mapping-parity"},
        },
        active_cooking={},
        pending_search_clarification={},
        menu_task={},
        active_search_request={},
        focus={},
        selected_recipe={},
    )
    python_state.device_execution = {
        **pending,
        "status": "dispatching",
        "confirmation_created_at": now,
        "claimed_at": now,
        "updated_at": now,
        "ts": now,
        "result_code": "",
        "command_sent": None,
        "started": None,
    }
    python_state.pending_device_start = None
    python_state.pending_action = None
    refresh_task_state_derived(python_state, now=now)
    assert python_state.current_task == "device_execution"
    assert python_state.selected_device_id == "office"
    assert python_state.language == "zh"

    script = _CLAIM_TASK_STATE_PENDING_DEVICE
    execution_position = script.index(
        "state['device_execution'] = next_execution"
    )
    clear_position = script.index(
        "state['pending_device_start'] = cjson.null"
    )
    derived_position = script.index(
        "if type(state['active_cooking']) == 'table'"
    )
    assert execution_position < clear_position < derived_position
    assert "return {4, revision, '', cjson.encode(state), generation}" in script
    assert "next_execution['status'] = 'dispatching'" in script
    assert "state['current_task'] = 'device_execution'" in script
    for field_name in (
        "active_cooking",
        "pending_search_clarification",
        "menu_task",
        "active_search_request",
        "focus",
    ):
        assert (
            f"type(state['{field_name}']) == 'table' "
            f"and next(state['{field_name}']) ~= nil"
        ) in script
    assert "state['current_task'] = cjson.null" in script
    assert "selected_device_id ~= '' and selected_device_id or cjson.null" in script
    assert "(language == 'zh' or language == 'en') and language or cjson.null" in script


def test_redis_task_claim_maps_unresolved_execution_to_in_progress():
    import json

    from app.conversation.redis_store import RedisConversationStore

    execution = {
        "action_id": "already-running",
        "device_id": "office",
        "status": "outcome_unknown",
        "updated_at": time.time(),
    }

    class Client:
        async def eval(self, *_args):
            return [
                4,
                9,
                "",
                json.dumps({"device_execution": execution}),
                3,
            ]

    async def run() -> None:
        store = RedisConversationStore(Client(), key_prefix="cookclaw:test")
        claim = await store.claim_pending_device_start_record(
            "qq:dm:redis-in-progress:redis-in-progress",
            expected_action_id="another-action",
            max_age_seconds=300,
            now=time.time(),
        )

        assert claim.status == "in_progress"
        assert claim.payload is None
        assert claim.revision == 9
        assert claim.generation == 3
        assert claim.state is not None
        assert claim.state.device_execution == execution

    asyncio.run(run())
