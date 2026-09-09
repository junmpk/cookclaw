"""基于独立 Redis task-state key 的单轮任务状态 Repository。"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from app.conversation.models import ConversationTaskState
from app.conversation.dialogue_state_workspace import TurnScopedDialogueStatePort
from app.conversation.runtime_memory import ShortTermRuntimeSnapshot
from app.conversation.service import (
    ConversationBackendUnavailable,
    get_conversation_service,
    supports_persistent_task_state,
)
from app.conversation.task_state_store import (
    DeviceStartClaim,
    TaskStateStoreConflict,
)
from app.ports.dialogue_state import (
    dialogue_state_port_scope,
    get_dialogue_state_port,
)
from app.ports.task_state import (
    current_task_state_repository,
    task_state_repository_scope,
)

_MANAGED_FIELDS = (
    "candidate_recipes",
    "candidate_language",
    "candidate_decision_id",
    "candidate_updated_at",
    "active_search_request",
    "active_search_updated_at",
    "selected_recipe_id",
    "selected_recipe_name",
    "selected_recipe",
    "selected_recipe_updated_at",
    "excluded_recipe_ids",
    "excluded_recipe_updated_at",
    "focus",
    "pending_search_clarification",
    "pending_action",
    "pending_device_start",
    "device_execution",
    "menu_task",
    "active_cooking",
)
_DERIVED_FIELDS = {
    "schema_version",
    "current_task",
    "selected_device_id",
    "language",
    "updated_at",
}
_UNRESOLVED_DEVICE_EXECUTION_STATUSES = frozenset({
    "dispatching",
    "submitted_unverified",
    "outcome_unknown",
})


def _copy_state(state: ConversationTaskState) -> ConversationTaskState:
    return ConversationTaskState.from_dict(state.to_dict())


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


def _overlay_managed_state(
    baseline: ConversationTaskState,
    managed: ConversationTaskState,
) -> ConversationTaskState:
    """保留画像命令字段，只覆盖同步 DialogueStatePort 管理的字段。"""
    merged = _copy_state(baseline)
    managed_copy = _copy_state(managed)
    for field_name in _MANAGED_FIELDS:
        setattr(merged, field_name, getattr(managed_copy, field_name))
    return merged


def _merge_after_conflict(
    baseline: ConversationTaskState,
    local_after: ConversationTaskState,
    latest: ConversationTaskState,
    *,
    baseline_generation: int,
    latest_generation: int,
) -> ConversationTaskState:
    """三方合并非重叠字段；同字段分叉时拒绝后写覆盖。"""
    if int(latest_generation) != int(baseline_generation):
        raise TaskStateStoreConflict(
            "task state generation changed; stale turn cannot cross reset"
        )
    before_values = baseline.to_dict()
    local_values = local_after.to_dict()
    latest_values = latest.to_dict()
    merged = _copy_state(latest)
    conflicting: list[str] = []
    for field_name in ConversationTaskState.__dataclass_fields__:
        if field_name in _DERIVED_FIELDS:
            continue
        local_changed = before_values.get(field_name) != local_values.get(field_name)
        if not local_changed:
            continue
        remote_changed = before_values.get(field_name) != latest_values.get(field_name)
        if remote_changed and local_values.get(field_name) != latest_values.get(field_name):
            conflicting.append(field_name)
            continue
        setattr(merged, field_name, getattr(_copy_state(local_after), field_name))
    if conflicting:
        raise TaskStateStoreConflict(
            "task state field conflict: " + ",".join(sorted(conflicting))
        )
    return merged


@dataclass(slots=True)
class _TaskStateUnitOfWork:
    thread_id: str
    service: Any
    port: TurnScopedDialogueStatePort | None
    baseline: ConversationTaskState
    revision: int
    generation: int
    short_term_snapshot: ShortTermRuntimeSnapshot | None
    persistent: bool
    trustworthy: bool
    aborted: bool = False


_current_uow: ContextVar[_TaskStateUnitOfWork | None] = ContextVar(
    "cookclaw_task_state_uow",
    default=None,
)


class RedisDialogueTaskStateRepository:
    """统一 Turn 链路的任务事实源。"""

    def __init__(self, *, service_factory=get_conversation_service) -> None:
        self._service_factory = service_factory

    @asynccontextmanager
    async def turn_scope(self, thread_id: str):
        value = str(thread_id or "")
        persistent = supports_persistent_task_state(value)
        if not persistent:
            uow = _TaskStateUnitOfWork(
                thread_id=value,
                service=None,
                port=None,
                baseline=ConversationTaskState(),
                revision=0,
                generation=0,
                short_term_snapshot=None,
                persistent=False,
                trustworthy=True,
            )
            uow_token = _current_uow.set(uow)
            with task_state_repository_scope(self):
                try:
                    yield None
                finally:
                    _current_uow.reset(uow_token)
            return

        service = self._service_factory()
        snapshot = await service.load_short_term_runtime(value)
        trustworthy = snapshot.task_state_status not in {
            "unavailable",
            "invalid",
        }
        baseline = (
            _copy_state(snapshot.memory.task_state)
            if trustworthy
            else ConversationTaskState()
        )
        port = TurnScopedDialogueStatePort(value, baseline)
        uow = _TaskStateUnitOfWork(
            thread_id=value,
            service=service,
            port=port,
            baseline=baseline,
            revision=max(0, int(snapshot.task_state_revision)),
            generation=max(0, int(snapshot.task_state_generation)),
            short_term_snapshot=snapshot,
            persistent=True,
            trustworthy=trustworthy,
        )
        uow_token = _current_uow.set(uow)
        try:
            with task_state_repository_scope(self), dialogue_state_port_scope(port):
                # yield 抛错时下面的 flush 不会执行，半轮工作区会直接丢弃。
                yield snapshot
                await self._flush_uow(uow)
        finally:
            _current_uow.reset(uow_token)
            port.close()

    async def _flush_uow(
        self,
        uow: _TaskStateUnitOfWork,
    ) -> None:
        if uow.aborted or not uow.persistent or uow.port is None:
            return
        managed_after = uow.port.snapshot()
        local_after = _overlay_managed_state(uow.baseline, managed_after)
        if local_after.to_dict() == uow.baseline.to_dict():
            return
        if not uow.trustworthy:
            raise ConversationBackendUnavailable(
                "task state changed while Redis state was unavailable"
            )

        baseline = _copy_state(uow.baseline)
        candidate = local_after
        expected_revision = uow.revision
        expected_generation = uow.generation
        last_conflict: TaskStateStoreConflict | None = None
        for attempt in range(3):
            try:
                committed = await uow.service.commit_task_state(
                    uow.thread_id,
                    candidate,
                    expected_revision=expected_revision,
                    expected_generation=expected_generation,
                )
            except TaskStateStoreConflict as exc:
                last_conflict = exc
                if attempt >= 2:
                    break
                latest = await uow.service.load_task_state_record(
                    uow.thread_id
                )
                candidate = _merge_after_conflict(
                    baseline,
                    local_after,
                    latest.state,
                    baseline_generation=uow.generation,
                    latest_generation=latest.generation,
                )
                if candidate.to_dict() == latest.state.to_dict():
                    uow.baseline = _copy_state(latest.state)
                    uow.revision = latest.revision
                    uow.generation = latest.generation
                    uow.port.replace(latest.state)
                    return
                expected_revision = latest.revision
                expected_generation = latest.generation
                continue
            uow.baseline = _copy_state(committed.state)
            uow.revision = committed.revision
            uow.generation = committed.generation
            uow.port.replace(committed.state)
            return
        if last_conflict is not None:
            raise last_conflict
        raise TaskStateStoreConflict("task state could not be committed")

    async def flush_current_scope(self) -> None:
        uow = _current_uow.get()
        if uow is None:
            raise RuntimeError("task state flush requires an active scope")
        await self._flush_uow(uow)

    async def abort_current_scope(self) -> None:
        """放弃本轮工作区修改；用于 reset/refresh 失败后的 fail-closed。"""
        uow = _current_uow.get()
        if uow is None:
            return
        uow.aborted = True
        if uow.port is not None:
            uow.port.replace(uow.baseline)

    async def refresh_current_scope(self) -> ShortTermRuntimeSnapshot | None:
        """Memory handler 直接修改任务态后显式重建本轮 baseline。"""
        uow = _current_uow.get()
        if uow is None or not uow.persistent:
            return None
        snapshot = await uow.service.load_short_term_runtime(uow.thread_id)
        if snapshot.task_state_status in {"unavailable", "invalid"}:
            uow.trustworthy = False
            raise ConversationBackendUnavailable("task state refresh failed")
        state = _copy_state(snapshot.memory.task_state)
        uow.port.replace(state)
        uow.baseline = state
        uow.revision = max(0, int(snapshot.task_state_revision))
        uow.generation = max(0, int(snapshot.task_state_generation))
        uow.short_term_snapshot = snapshot
        uow.trustworthy = True
        return snapshot

    async def claim_pending_device_start(
        self,
        thread_id: str,
        *,
        expected_action_id: str,
    ) -> DeviceStartClaim:
        uow = _current_uow.get()
        if uow is None or uow.thread_id != str(thread_id or ""):
            raise RuntimeError("device claim requires an active task state scope")
        if not uow.persistent:
            port = get_dialogue_state_port()
            execution = port.get_device_execution(thread_id)
            if _is_unresolved_device_execution(execution):
                return DeviceStartClaim(
                    "in_progress",
                    None,
                    0,
                    port.snapshot_thread_state(thread_id),
                    0,
                )
            payload = port.consume_pending(
                thread_id,
                expected_action_id=expected_action_id,
            )
            if isinstance(payload, dict):
                port.set_device_execution(
                    thread_id,
                    _execution_from_pending(payload, now=time.time()),
                )
            return DeviceStartClaim(
                "claimed" if payload else "missing",
                dict(payload) if isinstance(payload, dict) else None,
                0,
                port.snapshot_thread_state(thread_id),
                0,
            )
        if not uow.trustworthy:
            raise ConversationBackendUnavailable("task state is unavailable")

        # 先把本轮增量用 strict CAS 提交。如果同字段分叉，必须在
        # Redis claim 之前失败，保留 pending 供用户重试。
        await self._flush_uow(uow)
        prior_generation = uow.generation
        result = await uow.service.consume_pending_device_start_with_revision(
            thread_id,
            expected_action_id=expected_action_id,
        )
        authoritative = (
            _copy_state(result.state)
            if result.state is not None
            else ConversationTaskState()
        )
        if int(result.generation) != int(prior_generation):
            # Reset 是代际屏障。旧轮次不得借一次 missing/mismatch claim
            # “领养”新 generation 后再把 reset 前的局部选择叠回去。
            uow.port.replace(authoritative)
            uow.baseline = authoritative
            uow.revision = max(0, int(result.revision))
            uow.generation = max(0, int(result.generation))
            return DeviceStartClaim(
                "missing",
                None,
                result.revision,
                _copy_state(authoritative),
                result.generation,
            )
        if result.status in {"claimed", "expired"}:
            local_payload = uow.port.consume_pending(
                thread_id,
                expected_action_id=expected_action_id,
            )
        else:
            local_payload = None

        # claim 返回的完整状态是新基线；本轮增量已在 claim 前提交，
        # 因此不再执行“消费后合并”。
        uow.port.replace(authoritative)
        uow.revision = max(0, int(result.revision))
        uow.generation = max(0, int(result.generation))
        # claim 是一次独立 Redis 提交，后续普通提交必须从它返回的完整状态继续。
        uow.baseline = authoritative
        if result.claimed:
            return DeviceStartClaim(
                "claimed",
                (
                    dict(local_payload)
                    if isinstance(local_payload, dict)
                    else dict(result.payload or {})
                ),
                result.revision,
                _copy_state(authoritative),
                result.generation,
            )
        return DeviceStartClaim(
            result.status,
            None,
            result.revision,
            _copy_state(authoritative) if result.state is not None else None,
            result.generation,
        )


async def claim_pending_device_start(
    thread_id: str,
    *,
    expected_action_id: str,
) -> dict | None:
    """在当前 Turn UoW 内原子领取一次设备启动确认。"""
    repository = current_task_state_repository()
    if repository is None:
        raise RuntimeError("device claim requires the unified turn task repository")
    result = await repository.claim_pending_device_start(
        thread_id,
        expected_action_id=expected_action_id,
    )
    return dict(result.payload) if result.claimed else None
