"""统一构造 Router、回复模型和受控 Agent 使用的对话上下文。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.conversation.models import ConversationMemory, ConversationTurn


@dataclass(frozen=True)
class ConversationContext:
    """一轮对话的只读上下文；不在这里修改业务或用户资料状态。"""

    channel: str
    user_id: str | None
    current_message: str
    display_name: str | None
    preferred_name: str | None
    preferences: dict[str, list[str]]
    temporal_dietary_constraints: list[dict[str, Any]]
    account_digest: list[str]
    events: list[dict[str, Any]]
    recent_turns: list[ConversationTurn]
    recent_user_turns: list[str]
    conversation_summary: dict[str, Any]
    task_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "user_id": self.user_id,
            "current_message": self.current_message,
            "display_name": self.display_name,
            "preferred_name": self.preferred_name,
            "preferences": {
                key: list(values)
                for key, values in self.preferences.items()
            },
            "temporal_dietary_constraints": [
                dict(item) for item in self.temporal_dietary_constraints
            ],
            "account_digest": list(self.account_digest),
            "events": [dict(item) for item in self.events],
            "recent_turns": list(self.recent_turns),
            "recent_user_turns": list(self.recent_user_turns),
            "conversation_summary": dict(self.conversation_summary),
            "task_state": dict(self.task_state),
        }


def build_conversation_context(
    memory: ConversationMemory,
    account_context: dict[str, Any],
    *,
    current_message: str = "",
    include_recent: bool = True,
) -> ConversationContext:
    """合并短期会话与账号画像，并明确处理失败时的会话级资料覆盖。"""
    temporary = (
        memory.task_state.temporary_profile
        if isinstance(memory.task_state.temporary_profile, dict)
        else {}
    )
    has_temporary_name = temporary.get("preferred_name_set") is True
    preferred_name = (
        str(temporary.get("preferred_name") or "").strip() or None
        if has_temporary_name
        else str(account_context.get("preferred_name") or "").strip() or None
    )
    turns = list(memory.recent_turns[-10:]) if include_recent else []
    return ConversationContext(
        channel=memory.channel,
        user_id=memory.user_id,
        current_message=str(current_message or "").strip(),
        display_name=str(account_context.get("display_name") or "").strip() or None,
        preferred_name=preferred_name,
        preferences={
            str(key): list(values or [])
            for key, values in (account_context.get("preferences") or {}).items()
        },
        temporal_dietary_constraints=[
            dict(item)
            for item in (
                account_context.get("temporal_dietary_constraints") or []
            )
            if isinstance(item, dict)
        ],
        account_digest=[
            str(item)
            for item in (account_context.get("account_digest") or [])[-20:]
        ],
        events=[
            dict(item)
            for item in (account_context.get("events") or [])[-20:]
            if isinstance(item, dict)
        ],
        recent_turns=turns,
        recent_user_turns=[
            turn.content
            for turn in memory.recent_turns[-8:]
            if turn.role == "user" and str(turn.content).strip()
        ][-4:],
        conversation_summary=dict(memory.summary or {}) if include_recent else {},
        task_state=memory.task_state.to_dict() if include_recent else {},
    )
