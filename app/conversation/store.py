"""会话存储协议与仅供测试使用的进程内实现。"""
from __future__ import annotations

import time
from typing import Protocol

from app.conversation.models import ConversationMemory, ConversationTaskState
from app.conversation.task_state_store import (
    DeviceStartClaim,
    TaskStateRecord,
    TaskStateStoreConflict,
    refresh_task_state_derived,
)

_UNRESOLVED_DEVICE_EXECUTION_STATUSES = frozenset({
    "dispatching",
    "submitted_unverified",
    "outcome_unknown",
})


def _is_unresolved_device_execution(value: object) -> bool:
    return (
        isinstance(value, dict)
        and str(value.get("status") or "").strip().lower()
        in _UNRESOLVED_DEVICE_EXECUTION_STATUSES
    )


def _execution_from_pending(pending: dict, *, now: float) -> dict:
    execution = dict(pending)
    execution.update(
        status="dispatching",
        confirmation_created_at=float(pending.get("ts") or 0),
        claimed_at=float(now),
        updated_at=float(now),
        ts=float(now),
        result_code="",
        command_sent=None,
        started=None,
    )
    return execution


class ConversationStoreConflict(RuntimeError):
    """共享存储中的版本已变化，拒绝用旧状态覆盖新状态。"""


class ConversationStore(Protocol):
    async def load(self, thread_id: str) -> ConversationMemory | None: ...
    async def save(self, memory: ConversationMemory, expires_at: float) -> None: ...
    async def reset(
        self,
        memory: ConversationMemory,
        *,
        expected_version: int,
        expected_generation: int,
        expires_at: float,
    ) -> None: ...
    async def reset_with_task_state(
        self,
        memory: ConversationMemory,
        task_state: ConversationTaskState,
        *,
        expected_version: int,
        expected_generation: int,
        expected_task_revision: int,
        expected_task_generation: int,
        expires_at: float,
    ) -> TaskStateRecord: ...
    async def delete(self, thread_id: str) -> None: ...
    async def cleanup_expired(self, now: float | None = None) -> int: ...


class InMemoryConversationStore:
    """单元测试/本地隔离回放使用；生产必须使用 Redis。"""

    def __init__(self) -> None:
        self._items: dict[str, tuple[ConversationMemory, float]] = {}
        self._task_states: dict[str, tuple[TaskStateRecord, float]] = {}

    async def load(self, thread_id: str) -> ConversationMemory | None:
        item = self._items.get(thread_id)
        if not item:
            return None
        memory, expires_at = item
        if expires_at <= time.time():
            self._items.pop(thread_id, None)
            return None
        return ConversationMemory.from_dict(memory.to_dict())

    async def save(self, memory: ConversationMemory, expires_at: float) -> None:
        incoming = ConversationMemory.from_dict(memory.to_dict())
        current_item = self._items.get(memory.thread_id)
        if current_item is not None:
            current, current_expires_at = current_item
            if (
                current_expires_at > time.time()
                and incoming.conversation_generation
                != current.conversation_generation
            ):
                raise ConversationStoreConflict(
                    "conversation generation conflict"
                )
            if (
                current_expires_at > time.time()
                and current.task_state_storage_version >= 2
            ):
                incoming.task_state_storage_version = 2
                # 兼容旧只读调用保留最新镜像；运行时是否可用仍只看独立 key。
                incoming.task_state = ConversationTaskState.from_dict(
                    current.task_state.to_dict()
                )
        self._items[memory.thread_id] = (incoming, expires_at)

    async def reset(
        self,
        memory: ConversationMemory,
        *,
        expected_version: int,
        expected_generation: int,
        expires_at: float,
    ) -> None:
        current_item = self._items.get(memory.thread_id)
        if current_item is None or current_item[1] <= time.time():
            current_version = 0
            current_generation = 0
        else:
            current_version = int(current_item[0].version)
            current_generation = int(
                current_item[0].conversation_generation
            )
        if (
            current_version != max(0, int(expected_version))
            or current_generation != max(0, int(expected_generation))
        ):
            raise ConversationStoreConflict(
                "conversation reset baseline changed"
            )
        incoming = ConversationMemory.from_dict(memory.to_dict())
        incoming.version = current_version + 1
        incoming.conversation_generation = current_generation + 1
        self._items[memory.thread_id] = (incoming, float(expires_at))

    async def reset_with_task_state(
        self,
        memory: ConversationMemory,
        task_state: ConversationTaskState,
        *,
        expected_version: int,
        expected_generation: int,
        expected_task_revision: int,
        expected_task_generation: int,
        expires_at: float,
    ) -> TaskStateRecord:
        """同时写会话与任务代际墓碑，测试实现也不暴露半重置。"""
        now = time.time()
        conversation_item = self._items.get(memory.thread_id)
        if conversation_item is None or conversation_item[1] <= now:
            current_version = 0
            current_generation = 0
        else:
            current_version = int(conversation_item[0].version)
            current_generation = int(
                conversation_item[0].conversation_generation
            )
        if (
            current_version != max(0, int(expected_version))
            or current_generation != max(0, int(expected_generation))
        ):
            raise ConversationStoreConflict(
                "conversation reset baseline changed"
            )

        task_item = self._task_states.get(memory.thread_id)
        if task_item is None or task_item[1] <= now:
            current_task_revision = 0
            current_task_generation = 0
        else:
            current_task_revision = int(task_item[0].revision)
            current_task_generation = int(task_item[0].generation)
        if (
            current_task_revision != max(0, int(expected_task_revision))
            or current_task_generation
            != max(0, int(expected_task_generation))
        ):
            raise TaskStateStoreConflict(
                "task state reset baseline changed"
            )

        committed_task = TaskStateRecord(
            thread_id=memory.thread_id,
            state=ConversationTaskState.from_dict(task_state.to_dict()),
            revision=current_task_revision + 1,
            generation=current_task_generation + 1,
            updated_at=now,
        )
        incoming = ConversationMemory.from_dict(memory.to_dict())
        incoming.version = current_version + 1
        incoming.conversation_generation = current_generation + 1
        incoming.task_state_storage_version = 2
        incoming.task_state = ConversationTaskState.from_dict(
            committed_task.state.to_dict()
        )
        # 校验都完成后才一起替换两份记录。
        self._items[memory.thread_id] = (incoming, float(expires_at))
        self._task_states[memory.thread_id] = (
            committed_task.copy(),
            float(expires_at),
        )
        return committed_task.copy()

    async def delete(self, thread_id: str) -> None:
        self._items.pop(thread_id, None)

    async def cleanup_expired(self, now: float | None = None) -> int:
        now = now or time.time()
        expired = [key for key, (_, expires_at) in self._items.items() if expires_at <= now]
        for key in expired:
            self._items.pop(key, None)
        expired_task_states = [
            key
            for key, (_, expires_at) in self._task_states.items()
            if expires_at <= now
        ]
        for key in expired_task_states:
            self._task_states.pop(key, None)
        return len(expired) + len(expired_task_states)

    async def load_task_state_record(
        self,
        thread_id: str,
    ) -> TaskStateRecord | None:
        item = self._task_states.get(thread_id)
        if item is None:
            return None
        record, expires_at = item
        if expires_at <= time.time():
            self._task_states.pop(thread_id, None)
            return None
        return record.copy()

    async def commit_task_state_record(
        self,
        record: TaskStateRecord,
        *,
        expected_revision: int,
        expected_generation: int,
        new_generation: int | None,
        expires_at: float,
    ) -> TaskStateRecord:
        item = self._task_states.get(record.thread_id)
        current = None
        if item is not None:
            stored, stored_expires_at = item
            if stored_expires_at > time.time():
                current = stored.copy()
            else:
                self._task_states.pop(record.thread_id, None)
        current_revision = current.revision if current is not None else 0
        current_generation = current.generation if current is not None else 0
        if current_revision != max(0, int(expected_revision)):
            raise TaskStateStoreConflict(
                f"task state revision conflict: expected={expected_revision} "
                f"actual={current_revision}"
            )
        if current_generation != max(0, int(expected_generation)):
            raise TaskStateStoreConflict(
                "task state generation conflict: "
                f"expected={expected_generation} actual={current_generation}"
            )
        target_generation = (
            current_generation
            if new_generation is None
            else max(0, int(new_generation))
        )
        if target_generation not in {
            current_generation,
            current_generation + 1,
        }:
            raise ValueError("task state generation can only stay or advance once")
        committed = TaskStateRecord(
            thread_id=record.thread_id,
            state=ConversationTaskState.from_dict(record.state.to_dict()),
            revision=current_revision + 1,
            generation=target_generation,
            updated_at=max(time.time(), float(record.updated_at or 0)),
        )
        self._task_states[record.thread_id] = (
            committed.copy(),
            float(expires_at),
        )
        conversation_item = self._items.get(record.thread_id)
        if conversation_item is not None:
            memory, conversation_expires_at = conversation_item
            if conversation_expires_at > time.time():
                migrated = ConversationMemory.from_dict(memory.to_dict())
                migrated.task_state_storage_version = 2
                migrated.task_state = ConversationTaskState.from_dict(
                    committed.state.to_dict()
                )
                self._items[record.thread_id] = (
                    migrated,
                    conversation_expires_at,
                )
        return committed.copy()

    async def claim_pending_device_start_record(
        self,
        thread_id: str,
        *,
        expected_action_id: str,
        max_age_seconds: int,
        now: float,
    ) -> DeviceStartClaim:
        item = self._task_states.get(thread_id)
        current = None
        expires_at = float(now) + 86_400
        if item is not None:
            stored, stored_expires_at = item
            if stored_expires_at > float(now):
                current = stored.copy()
                expires_at = stored_expires_at
            else:
                self._task_states.pop(thread_id, None)
        if current is None:
            return DeviceStartClaim("missing", None, 0, None, 0)
        if _is_unresolved_device_execution(current.state.device_execution):
            return DeviceStartClaim(
                "in_progress",
                None,
                current.revision,
                ConversationTaskState.from_dict(current.state.to_dict()),
                current.generation,
            )
        pending = current.state.pending_device_start
        if not isinstance(pending, dict):
            return DeviceStartClaim(
                "missing",
                None,
                current.revision,
                ConversationTaskState.from_dict(current.state.to_dict()),
                current.generation,
            )
        if str(pending.get("action_id") or "") != str(expected_action_id or ""):
            return DeviceStartClaim(
                "mismatch",
                None,
                current.revision,
                ConversationTaskState.from_dict(current.state.to_dict()),
                current.generation,
            )
        if float(now) - float(pending.get("ts") or 0) > max(1, int(max_age_seconds)):
            status = "expired"
            payload = None
        else:
            status = "claimed"
            payload = dict(pending)
        updated = current.copy()
        if status == "claimed":
            updated.state.device_execution = _execution_from_pending(
                pending,
                now=float(now),
            )
        updated.state.pending_device_start = None
        if (
            isinstance(updated.state.pending_action, dict)
            and updated.state.pending_action.get("kind")
            in {"choose_device", "confirm_device_start"}
        ):
            updated.state.pending_action = None
        refresh_task_state_derived(updated.state, now=float(now))
        committed = TaskStateRecord(
            thread_id=thread_id,
            state=ConversationTaskState.from_dict(updated.state.to_dict()),
            revision=current.revision + 1,
            generation=current.generation,
            updated_at=float(now),
        )
        self._task_states[thread_id] = (committed.copy(), expires_at)
        return DeviceStartClaim(
            status,
            payload,
            committed.revision,
            ConversationTaskState.from_dict(committed.state.to_dict()),
            committed.generation,
        )

    async def delete_task_state_record(self, thread_id: str) -> None:
        self._task_states.pop(thread_id, None)
