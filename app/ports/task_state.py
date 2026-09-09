"""应用层可用的异步任务状态 Repository 端口。"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol

TaskStateClaimStatus = Literal[
    "claimed",
    "in_progress",
    "missing",
    "mismatch",
    "expired",
]


@dataclass(frozen=True, slots=True)
class DeviceStartClaim:
    """设备确认原子领取结果；端口 DTO 不依赖具体存储实现。"""

    status: TaskStateClaimStatus
    payload: dict | None
    revision: int
    state: Any | None = None
    generation: int = 0

    @property
    def claimed(self) -> bool:
        return self.status == "claimed" and isinstance(self.payload, dict)


class DialogueTaskStateRepository(Protocol):
    def turn_scope(
        self,
        thread_id: str,
    ) -> AbstractAsyncContextManager[Any]: ...

    async def refresh_current_scope(self) -> Any: ...

    async def flush_current_scope(self) -> Any: ...

    async def abort_current_scope(self) -> None: ...

    async def claim_pending_device_start(
        self,
        thread_id: str,
        *,
        expected_action_id: str,
    ) -> DeviceStartClaim: ...


_current_repository: ContextVar[DialogueTaskStateRepository | None] = ContextVar(
    "cookclaw_task_state_repository",
    default=None,
)


def current_task_state_repository() -> DialogueTaskStateRepository | None:
    return _current_repository.get()


def task_state_repository_bound() -> bool:
    return current_task_state_repository() is not None


@contextmanager
def task_state_repository_scope(repository: DialogueTaskStateRepository):
    token = _current_repository.set(repository)
    try:
        yield
    finally:
        _current_repository.reset(token)


__all__ = [
    "DeviceStartClaim",
    "DialogueTaskStateRepository",
    "TaskStateClaimStatus",
    "current_task_state_repository",
    "task_state_repository_bound",
    "task_state_repository_scope",
]
