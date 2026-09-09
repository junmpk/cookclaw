"""对话任务状态端口。

编排层只依赖本模块。具体工作区 adapter 由 ``app.conversation`` 在组合边界注册，
测试和其他存储实现可在当前 Context 中替换端口。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol


class DialogueStatePort(Protocol):
    """编排层可使用的稳定状态与状态语义接口。"""

    def remember_candidates(
        self,
        thread_id: str,
        search_result: dict,
        lang: str | None = None,
    ) -> None: ...

    def recall_candidate_context(
        self,
        thread_id: str,
        max_age: int = ...,
    ) -> dict | None: ...

    def recall_candidate_lang(
        self,
        thread_id: str,
        max_age: int = ...,
    ) -> str | None: ...

    def resolve_selection(self, thread_id: str, text: str) -> dict | None: ...

    def remember_focus(
        self,
        thread_id: str,
        topic: str,
        lang: str | None = None,
        source: str = "chat",
        focus_kind: str = "topic",
    ) -> None: ...

    def remember_recipe_focus(
        self,
        thread_id: str,
        recipe: dict,
        lang: str | None = None,
        source: str = "selection",
    ) -> None: ...

    def recall_focus(self, thread_id: str, max_age: int = ...) -> dict | None: ...

    def set_pending(self, thread_id: str, cook_id: str, name: str, **kwargs: Any) -> None: ...
    def get_pending(self, thread_id: str, max_age: int = ...) -> dict | None: ...
    def clear_pending(self, thread_id: str) -> None: ...
    def consume_pending(self, thread_id: str, **kwargs: Any) -> dict | None: ...

    def set_active_cooking(
        self,
        thread_id: str,
        cook_id: str,
        name: str,
        **kwargs: Any,
    ) -> None: ...

    def get_active_cooking(self, thread_id: str, max_age: int = ...) -> dict | None: ...
    def clear_active_cooking(self, thread_id: str) -> None: ...

    def set_device_execution(self, thread_id: str, execution: dict) -> None: ...
    def get_device_execution(
        self,
        thread_id: str,
        max_age: int = ...,
    ) -> dict | None: ...
    def update_device_execution(
        self,
        thread_id: str,
        *,
        expected_action_id: str,
        status: str,
        result_code: str | None = None,
        command_sent: bool | None = None,
        started: bool | None = None,
    ) -> bool: ...
    def clear_device_execution(self, thread_id: str) -> None: ...

    def set_search_clarification(
        self,
        thread_id: str,
        request: dict,
        **kwargs: Any,
    ) -> None: ...

    def get_search_clarification(
        self,
        thread_id: str,
        max_age: int = ...,
    ) -> dict | None: ...

    def clear_search_clarification(self, thread_id: str) -> None: ...
    def set_pending_action(self, thread_id: str, kind: str, **kwargs: Any) -> None: ...
    def get_pending_action(self, thread_id: str, max_age: int = ...) -> dict | None: ...
    def clear_pending_action(self, thread_id: str) -> None: ...
    def consume_pending_action(self, thread_id: str, **kwargs: Any) -> dict | None: ...
    def set_menu_task(self, thread_id: str, request: dict, search_result: dict, **kwargs: Any) -> None: ...
    def get_menu_task(self, thread_id: str, max_age: int = ...) -> dict | None: ...
    def clear_menu_task(self, thread_id: str) -> None: ...
    def set_selected_recipe(self, thread_id: str, recipe: dict, **kwargs: Any) -> None: ...
    def get_selected_recipe(self, thread_id: str, max_age: int = ...) -> dict | None: ...
    def clear_selected_recipe(self, thread_id: str) -> None: ...
    def set_candidate_excluded(self, thread_id: str, recipe_id: str, **kwargs: Any) -> None: ...
    def get_excluded_recipe_ids(self, thread_id: str, max_age: int = ...) -> list[str]: ...
    def clear_candidate_decisions(self, thread_id: str) -> None: ...
    def set_active_search_request(self, thread_id: str, request: dict, **kwargs: Any) -> None: ...
    def get_active_search_request(self, thread_id: str, max_age: int = ...) -> dict | None: ...
    def clear_candidate_context(self, thread_id: str, **kwargs: Any) -> None: ...

    def snapshot_thread_state(self, thread_id: str) -> Any: ...
    def restore_thread_state(
        self,
        thread_id: str,
        state: Any,
    ) -> None: ...

    def task_state_has_data(
        self,
        state: Any,
    ) -> bool: ...

    def routing_state_snapshot(
        self,
        thread_id: str,
        **kwargs: Any,
    ) -> dict: ...

    def clear_thread(self, thread_id: str) -> None: ...
    def resolve_device_choice(self, text: str, devices: list) -> Any: ...
    def normalize_voice_control_text(self, text: str) -> str: ...
    def is_confirm(self, text: str) -> bool: ...
    def is_cancel(self, text: str) -> bool: ...
    def is_abandonment(self, text: str) -> bool: ...
    def is_stop(self, text: str) -> bool: ...
    def is_progress(self, text: str) -> bool: ...


_default_port: DialogueStatePort | None = None
_current_port: ContextVar[DialogueStatePort | None] = ContextVar(
    "cookclaw_dialogue_state_port",
    default=None,
)


def configure_default_dialogue_state_port(port: DialogueStatePort) -> None:
    """由应用组合边界注册默认实现。"""
    global _default_port
    _default_port = port


def get_dialogue_state_port() -> DialogueStatePort:
    port = _current_port.get() or _default_port
    if port is None:
        raise RuntimeError("default dialogue state port is not configured")
    return port


@contextmanager
def dialogue_state_port_scope(port: DialogueStatePort) -> Iterator[None]:
    """在当前同步/异步 Context 中替换状态实现；退出时自动恢复。"""
    token = _current_port.set(port)
    try:
        yield
    finally:
        _current_port.reset(token)


def _delegate(operation: str):
    def call(*args: Any, **kwargs: Any):
        target = getattr(get_dialogue_state_port(), operation)
        return target(*args, **kwargs)

    call.__name__ = operation
    return call


# 模块级函数是稳定调用边界；真实目标在每次调用时由当前 Context Port 决定。
remember_candidates = _delegate("remember_candidates")
recall_candidate_context = _delegate("recall_candidate_context")
recall_candidate_lang = _delegate("recall_candidate_lang")
resolve_selection = _delegate("resolve_selection")
remember_focus = _delegate("remember_focus")
remember_recipe_focus = _delegate("remember_recipe_focus")
recall_focus = _delegate("recall_focus")
set_pending = _delegate("set_pending")
get_pending = _delegate("get_pending")
clear_pending = _delegate("clear_pending")
consume_pending = _delegate("consume_pending")
set_active_cooking = _delegate("set_active_cooking")
get_active_cooking = _delegate("get_active_cooking")
clear_active_cooking = _delegate("clear_active_cooking")
set_device_execution = _delegate("set_device_execution")
get_device_execution = _delegate("get_device_execution")
update_device_execution = _delegate("update_device_execution")
clear_device_execution = _delegate("clear_device_execution")
set_search_clarification = _delegate("set_search_clarification")
get_search_clarification = _delegate("get_search_clarification")
clear_search_clarification = _delegate("clear_search_clarification")
set_pending_action = _delegate("set_pending_action")
get_pending_action = _delegate("get_pending_action")
clear_pending_action = _delegate("clear_pending_action")
consume_pending_action = _delegate("consume_pending_action")
set_menu_task = _delegate("set_menu_task")
get_menu_task = _delegate("get_menu_task")
clear_menu_task = _delegate("clear_menu_task")
set_selected_recipe = _delegate("set_selected_recipe")
get_selected_recipe = _delegate("get_selected_recipe")
clear_selected_recipe = _delegate("clear_selected_recipe")
set_candidate_excluded = _delegate("set_candidate_excluded")
get_excluded_recipe_ids = _delegate("get_excluded_recipe_ids")
clear_candidate_decisions = _delegate("clear_candidate_decisions")
set_active_search_request = _delegate("set_active_search_request")
get_active_search_request = _delegate("get_active_search_request")
clear_candidate_context = _delegate("clear_candidate_context")
snapshot_thread_state = _delegate("snapshot_thread_state")
restore_thread_state = _delegate("restore_thread_state")
task_state_has_data = _delegate("task_state_has_data")
routing_state_snapshot = _delegate("routing_state_snapshot")
clear_thread = _delegate("clear_thread")
resolve_device_choice = _delegate("resolve_device_choice")
normalize_voice_control_text = _delegate("normalize_voice_control_text")
is_confirm = _delegate("is_confirm")
is_cancel = _delegate("is_cancel")
is_abandonment = _delegate("is_abandonment")
is_stop = _delegate("is_stop")
is_progress = _delegate("is_progress")


__all__ = [
    "DialogueStatePort",
    "configure_default_dialogue_state_port",
    "dialogue_state_port_scope",
    "get_dialogue_state_port",
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
    "task_state_has_data",
    "routing_state_snapshot",
    "clear_thread",
    "resolve_device_choice",
    "normalize_voice_control_text",
    "is_confirm",
    "is_cancel",
    "is_abandonment",
    "is_stop",
    "is_progress",
]
