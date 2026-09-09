"""CookClaw 单轮编排入口。

可选 Planner 及其评估入口归入 planning 子包。统一 Turn facade 按固定顺序先接
图片、会话重置、Memory 命令、精确命令，再接设备待确认、其余 pending state、可选 Planner、Recipe handler、
Device handler、Conversation fallback 和最终 fallback。facade 出口附加真实工具调用与状态
差异账本；画像/偏好位于首位 handler。不同消息类型可以不注册无关 handler，但业务轮次只能由第一个返回
ResponseEnvelope 的 handler 完成。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from app.orchestrator.planning.shadow_compare import (
    planner_shadow_enabled,
    record_shadow_bypass,
    schedule_shadow_evaluation,
)
from app.orchestrator.routing_models import RoutingContext
from app.orchestrator.turn.context_loader import build_turn_context
from app.orchestrator.turn.runtime_models import ResponseEnvelope, TurnRequest

logger = logging.getLogger(__name__)


TurnHandler = Callable[[TurnRequest], Awaitable[ResponseEnvelope | None]]
_HANDLER_ORDER = (
    "image_handler",
    "reset_command",
    "memory_command",
    "exact_command",
    "device_pending",
    "pending_state",
    "planner",
    "recipe_handler",
    "device_handler",
    "conversation_fallback",
    "fallback",
)


class TurnNotHandledError(RuntimeError):
    """所有 handler 都拒绝本轮时抛出；不得静默生成自由回复。"""


class TurnOrchestrator:
    """按固定顺序调度一轮对话，返回统一 ResponseEnvelope。

    Memory、精确命令、设备待确认、其余 pending 状态、Recipe 和 Device 都是显式
    handler；Conversation fallback 只负责问答、闲聊与联网搜索。出口只读采集实际
    ``ToolResult`` / ``StatePatch``。首个 handler
    命中后立即返回，因此设备命令和状态写入仍只发生一次。Planner 只处理没有被
    确定性 pending handler 接管的轮次，不能覆盖已存在的业务状态机。
    """

    def __init__(
        self,
        *,
        image_handler: TurnHandler | None = None,
        reset_command_handler: TurnHandler | None = None,
        memory_command_handler: TurnHandler | None = None,
        exact_command_handler: TurnHandler | None = None,
        device_pending_handler: TurnHandler | None = None,
        planner_handler: TurnHandler | None = None,
        pending_state_handler: TurnHandler | None = None,
        recipe_handler: TurnHandler | None = None,
        device_handler: TurnHandler | None = None,
        conversation_fallback_handler: TurnHandler | None = None,
        fallback_handler: TurnHandler | None = None,
    ) -> None:
        self._handlers: dict[str, TurnHandler | None] = {
            "image_handler": image_handler,
            "reset_command": reset_command_handler,
            "memory_command": memory_command_handler,
            "exact_command": exact_command_handler,
            "device_pending": device_pending_handler,
            "planner": planner_handler,
            "pending_state": pending_state_handler,
            "recipe_handler": recipe_handler,
            "device_handler": device_handler,
            "conversation_fallback": conversation_fallback_handler,
            "fallback": fallback_handler,
        }

    async def handle(self, turn_request: TurnRequest) -> ResponseEnvelope:
        for stage in _HANDLER_ORDER:
            handler = self._handlers[stage]
            if handler is None:
                continue
            envelope = await handler(turn_request)
            if envelope is None:
                continue
            # handler 内部 adapter 可能仍带旧领域标签；应用调度阶段才是本轮真实
            # 所有者，必须在唯一出口覆盖，避免观测和回放把 pending 误记成普通路由。
            public_stage = (
                "exact_command" if stage == "reset_command" else stage
            )
            if envelope.handled_by != public_stage:
                envelope = envelope.model_copy(
                    update={"handled_by": public_stage}
                )
            return envelope
        raise TurnNotHandledError(
            f"turn was not handled: channel={turn_request.channel}"
        )


def schedule_turn_plan_shadow(
    *,
    utterance: str,
    routing_context: RoutingContext,
    legacy_outcome: Any,
    memory_context: dict[str, Any] | None = None,
):
    """为已经完成的 Legacy 路由旁路生成计划；off/active 均 fail closed。"""
    if not planner_shadow_enabled():
        return None
    try:
        context = build_turn_context(
            utterance,
            routing_context,
            memory_context=memory_context,
        )
        return schedule_shadow_evaluation(context, legacy_outcome)
    except Exception as exc:
        logger.warning(
            "turn planner shadow scheduling failed closed: error_type=%s",
            type(exc).__name__,
        )
        return None


def record_turn_plan_bypass(
    *,
    utterance: str,
    routing_context: RoutingContext,
    bypass_reason: str,
    memory_context: dict[str, Any] | None = None,
    legacy_action: str = "early_return",
    category: str = "conversation",
    risk: str = "low",
    needs_clarification: bool = False,
) -> dict[str, Any] | None:
    """记录前置确定性分支；只写脱敏 Shadow 记录，不调用 Planner。"""
    if not planner_shadow_enabled():
        return None
    try:
        context = build_turn_context(
            utterance,
            routing_context,
            memory_context=memory_context,
        )
        return record_shadow_bypass(
            context,
            bypass_reason,
            legacy_action=legacy_action,
            category=category,
            risk=risk,
            needs_clarification=needs_clarification,
        )
    except Exception as exc:
        logger.warning(
            "turn planner bypass recording failed closed: error_type=%s",
            type(exc).__name__,
        )
        return None
