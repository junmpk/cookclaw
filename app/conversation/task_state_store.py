"""独立任务状态存储协议。

任务状态使用独立 revision，不与聊天历史、摘要和搜索历史的
``ConversationMemory.version`` 共用冲突域。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.conversation.models import ConversationTaskState
from app.ports.task_state import DeviceStartClaim


def refresh_task_state_derived(
    state: ConversationTaskState,
    *,
    now: float,
) -> None:
    """从领域事实重算冗余路由字段，存储实现不得各自保留旧值。"""
    if state.active_cooking:
        state.current_task = "device_cooking"
    elif state.pending_device_start:
        state.current_task = "device_start"
    elif isinstance(state.device_execution, dict) and state.device_execution.get(
        "status"
    ) in {"dispatching", "submitted_unverified", "outcome_unknown"}:
        state.current_task = "device_execution"
    elif state.pending_search_clarification:
        state.current_task = "recipe_search"
    elif state.menu_task:
        state.current_task = "menu_plan"
    elif state.candidate_recipes:
        state.current_task = "recipe_selection"
    elif state.active_search_request:
        state.current_task = "recipe_search"
    elif state.focus:
        state.current_task = "recipe_discussion"
    else:
        state.current_task = None
    state.selected_device_id = str(
        (state.pending_device_start or {}).get("device_id")
        or (state.active_cooking or {}).get("device_id")
        or (state.device_execution or {}).get("device_id")
        or ""
    ) or None
    state.language = next(
        (
            value
            for value in (
                (state.pending_device_start or {}).get("lang"),
                (state.device_execution or {}).get("lang"),
                (state.selected_recipe or {}).get("lang"),
                state.candidate_language,
                (state.focus or {}).get("lang"),
                (state.active_cooking or {}).get("lang"),
                state.language,
            )
            if value in ("zh", "en")
        ),
        None,
    )
    state.updated_at = float(now)


class TaskStateStoreConflict(RuntimeError):
    """任务状态 revision 已变化，拒绝用旧快照覆盖。"""


@dataclass(frozen=True, slots=True)
class TaskStateRecord:
    thread_id: str
    state: ConversationTaskState
    revision: int = 0
    generation: int = 0
    updated_at: float = 0.0

    def copy(self) -> TaskStateRecord:
        return TaskStateRecord(
            thread_id=self.thread_id,
            state=ConversationTaskState.from_dict(self.state.to_dict()),
            revision=max(0, int(self.revision)),
            generation=max(0, int(self.generation)),
            updated_at=max(0.0, float(self.updated_at)),
        )


class TaskStateStore(Protocol):
    async def load_task_state_record(
        self,
        thread_id: str,
    ) -> TaskStateRecord | None: ...

    async def commit_task_state_record(
        self,
        record: TaskStateRecord,
        *,
        expected_revision: int,
        expected_generation: int,
        new_generation: int | None,
        expires_at: float,
    ) -> TaskStateRecord: ...

    async def claim_pending_device_start_record(
        self,
        thread_id: str,
        *,
        expected_action_id: str,
        max_age_seconds: int,
        now: float,
    ) -> DeviceStartClaim: ...

    async def delete_task_state_record(self, thread_id: str) -> None: ...
