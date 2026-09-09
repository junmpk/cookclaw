"""CookClaw 单轮执行的唯一组合根。

通道和领域模块只提交 ``TurnRequest``、运行时加载器与 handler 映射；本模块负责
选择任务态 Repository 并进入 ``TurnApplicationService``。这样文本、图片和未来消息
类型不会各自创建 UoW 或编排器。
"""

from __future__ import annotations

from collections.abc import Mapping

from app.conversation.task_state_repository import (
    RedisDialogueTaskStateRepository,
)
from app.orchestrator.turn.application_service import (
    ApplicationTurnHandler,
    RuntimeLoader,
    TurnApplicationService,
)
from app.orchestrator.turn.runtime_models import ResponseEnvelope, TurnRequest
from app.ports.task_state import DialogueTaskStateRepository


async def execute_turn(
    request: TurnRequest,
    *,
    runtime_loader: RuntimeLoader,
    handlers: Mapping[str, ApplicationTurnHandler],
    task_state_repository: DialogueTaskStateRepository | None = None,
) -> ResponseEnvelope:
    """在唯一 Repository/UoW 生命周期内执行一轮请求。"""
    repository = task_state_repository or RedisDialogueTaskStateRepository()
    return await TurnApplicationService(
        task_state_repository=repository,
    ).handle(
        request,
        runtime_loader=runtime_loader,
        handlers=handlers,
    )


__all__ = ["execute_turn"]
