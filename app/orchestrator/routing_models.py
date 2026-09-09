"""统一路由输入与输出模型。

路由器只消费经过裁剪的结构化上下文，不直接接收整段聊天历史或完整菜谱详情。
RouteDecision 是规则、状态规则和轻量分类器共同遵守的审计契约。
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.observability.trace import current_trace_id


def _route_trace_id() -> str:
    return current_trace_id() or uuid.uuid4().hex


class RoutingContext(BaseModel):
    """一次意图路由所需的最小状态摘要。"""

    pending_action: dict[str, Any] | None = None
    pending_device_start: dict[str, Any] | None = None
    active_cooking: dict[str, Any] | None = None
    pending_clarification: dict[str, Any] | None = None
    latest_candidates: list[dict[str, Any]] = Field(default_factory=list)
    latest_focus: dict[str, Any] | None = None
    selected_recipe: dict[str, Any] | None = None
    excluded_recipe_ids: list[str] = Field(default_factory=list)
    current_task: str | None = None
    active_constraints: dict[str, list[str]] = Field(default_factory=dict)
    recent_user_turns: list[str] = Field(default_factory=list)
    stable_constraints: dict[str, list[str]] = Field(default_factory=dict)
    channel: str = "unknown"
    message_type: str = "text"

    def classifier_summary(self) -> dict[str, Any]:
        """生成低噪声、低敏感度的分类器输入，不泄露完整状态 payload。"""
        pending = self.pending_action or {}
        pending_device = self.pending_device_start or {}
        active = self.active_cooking or {}
        clarification = self.pending_clarification or {}
        focus = self.latest_focus or {}
        selected = self.selected_recipe or {}
        return {
            "pending_action": str(pending.get("kind") or "") or None,
            "pending_device_start": bool(pending_device),
            "device_choice_required": bool(pending_device.get("devices")),
            "active_cooking": bool(active),
            "pending_clarification": str(clarification.get("dimension") or "") or None,
            "candidate_count": len(self.latest_candidates),
            "candidate_names": [
                str(item.get("name") or "")[:40]
                for item in self.latest_candidates[:5]
                if str(item.get("name") or "").strip()
            ],
            "latest_focus": str(focus.get("name") or focus.get("topic") or "")[:40] or None,
            "selected_recipe": str(selected.get("name") or "")[:40] or None,
            "excluded_candidate_count": len(self.excluded_recipe_ids),
            "current_task": self.current_task,
            "active_constraints": {
                str(key): [str(item)[:40] for item in values[:8]]
                for key, values in self.active_constraints.items()
                if isinstance(values, list) and values
            },
            "recent_user_turns": [
                str(item)[:120] for item in self.recent_user_turns[-4:] if str(item).strip()
            ],
            "stable_constraints": {
                str(key): [str(item)[:40] for item in values[:8]]
                for key, values in self.stable_constraints.items()
                if isinstance(values, list) and values
            },
            "channel": self.channel,
            "message_type": self.message_type,
        }

    def context_ref(self) -> dict[str, Any]:
        """可写日志的状态引用，只保留是否存在和计数。"""
        summary = self.classifier_summary()
        return {
            "pending_action": summary["pending_action"],
            "pending_device_start": summary["pending_device_start"],
            "active_cooking": summary["active_cooking"],
            "pending_clarification": summary["pending_clarification"],
            "candidate_count": summary["candidate_count"],
            "has_focus": bool(summary["latest_focus"]),
            "has_selected_recipe": bool(summary["selected_recipe"]),
            "excluded_candidate_count": summary["excluded_candidate_count"],
            "current_task": summary["current_task"],
            "active_constraint_count": sum(
                len(values)
                for values in summary["active_constraints"].values()
            ),
            "recent_user_turn_count": len(summary["recent_user_turns"]),
            "channel": summary["channel"],
            "message_type": summary["message_type"],
        }

    def has_routing_signal(self) -> bool:
        summary = self.classifier_summary()
        return any((
            summary["pending_action"],
            summary["pending_device_start"],
            summary["active_cooking"],
            summary["pending_clarification"],
            summary["candidate_count"],
            summary["latest_focus"],
            summary["selected_recipe"],
            summary["excluded_candidate_count"],
            summary["current_task"],
            summary["active_constraints"],
            summary["recent_user_turns"],
            summary["stable_constraints"],
            self.channel != "unknown",
            self.message_type != "text",
        ))


class RouteDecision(BaseModel):
    """统一路由结论；action 驱动业务，category 仅保留粗粒度兼容性。"""

    category: str
    action: str
    source: Literal[
        "exact_rule", "state_rule", "classifier", "parse_error", "ambiguous",
    ]
    reason_code: str
    risk: Literal["low", "medium", "high"] = "low"
    slots: dict[str, Any] = Field(default_factory=dict)
    context_ref: dict[str, Any] | None = None
    needs_clarification: bool = False
    classifier_called: bool = False
    trace_id: str = Field(default_factory=_route_trace_id)
