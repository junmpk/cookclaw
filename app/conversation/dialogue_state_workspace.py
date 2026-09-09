"""DialogueStatePort 的会话工作区实现。

本模块属于会话基础设施：复用 ``task_state_workspace`` 的同步状态语义，并为每个
Turn 提供随机内部 key。编排层只通过 ``app.ports.dialogue_state`` 使用它。
"""

from __future__ import annotations

import uuid
from typing import Any

from app.conversation import task_state_workspace as task_workspace
from app.conversation.models import ConversationTaskState
from app.ports.dialogue_state import configure_default_dialogue_state_port


class WorkspaceDialogueStatePort:
    """默认工作区 adapter；不复制 ``task_state_workspace`` 的状态语义。"""

    def __getattr__(self, operation: str):
        return getattr(task_workspace, operation)


_THREAD_STATE_OPERATIONS = frozenset(
    {
        "remember_candidates",
        "recall_candidate_context",
        "recall_candidate_lang",
        "resolve_selection",
        "remember_focus",
        "remember_recipe_focus",
        "recall_focus",
        "set_pending",
        "get_pending",
        "clear_pending",
        "consume_pending",
        "set_active_cooking",
        "get_active_cooking",
        "clear_active_cooking",
        "set_device_execution",
        "get_device_execution",
        "update_device_execution",
        "clear_device_execution",
        "set_search_clarification",
        "get_search_clarification",
        "clear_search_clarification",
        "set_pending_action",
        "get_pending_action",
        "clear_pending_action",
        "consume_pending_action",
        "set_menu_task",
        "get_menu_task",
        "clear_menu_task",
        "set_selected_recipe",
        "get_selected_recipe",
        "clear_selected_recipe",
        "set_candidate_excluded",
        "get_excluded_recipe_ids",
        "clear_candidate_decisions",
        "set_active_search_request",
        "get_active_search_request",
        "clear_candidate_context",
        "snapshot_thread_state",
        "restore_thread_state",
        "routing_state_snapshot",
        "clear_thread",
    }
)


class TurnScopedDialogueStatePort:
    """每轮隔离的同步状态工作区；退出后完整清理，不是跨轮事实源。"""

    def __init__(
        self,
        thread_id: str,
        initial_state: ConversationTaskState | dict | None,
    ) -> None:
        self.thread_id = str(thread_id)
        self._internal_thread_id = f"turn:{uuid.uuid4().hex}"
        self._closed = False
        task_workspace.restore_thread_state(
            self._internal_thread_id,
            initial_state,
        )

    def _translated_thread_id(self, thread_id: str) -> str:
        if self._closed:
            raise RuntimeError("turn-scoped dialogue state is closed")
        if str(thread_id) != self.thread_id:
            raise ValueError("turn-scoped dialogue state cannot access another thread")
        return self._internal_thread_id

    def __getattr__(self, operation: str):
        target = getattr(task_workspace, operation)
        if operation not in _THREAD_STATE_OPERATIONS:
            return target

        def call(thread_id: str, *args: Any, **kwargs: Any):
            return target(
                self._translated_thread_id(thread_id),
                *args,
                **kwargs,
            )

        return call

    def snapshot(self) -> ConversationTaskState:
        if self._closed:
            raise RuntimeError("turn-scoped dialogue state is closed")
        return task_workspace.snapshot_thread_state(self._internal_thread_id)

    def replace(self, state: ConversationTaskState | dict | None) -> None:
        if self._closed:
            raise RuntimeError("turn-scoped dialogue state is closed")
        self._clear_internal()
        task_workspace.restore_thread_state(self._internal_thread_id, state)

    def _clear_internal(self) -> None:
        task_workspace.clear_thread(self._internal_thread_id)
        task_workspace.clear_active_cooking(self._internal_thread_id)
        task_workspace.clear_device_execution(self._internal_thread_id)

    def close(self) -> None:
        if self._closed:
            return
        self._clear_internal()
        self._closed = True


configure_default_dialogue_state_port(WorkspaceDialogueStatePort())


__all__ = ["TurnScopedDialogueStatePort", "WorkspaceDialogueStatePort"]
