"""面向生成模型的会话上下文契约。

业务记忆视图只负责提供数据；本模块继续保留最近消息的真实角色，避免把历史
用户输入或旧助手回复拼进 SystemMessage。字符串表示仅用于兼容日志、受控旧
调用和渐进迁移，不应作为新模型调用的首选输入。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

PromptRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class PromptHistoryTurn:
    role: PromptRole
    content: str


@dataclass(frozen=True, slots=True)
class ConversationPromptContext:
    """分离资料、非权威摘要和真实角色历史的只读上下文。"""

    fact_context: str = ""
    earlier_digest: tuple[str, ...] = ()
    recent_turns: tuple[PromptHistoryTurn, ...] = ()

    def __bool__(self) -> bool:
        return bool(
            self.fact_context.strip()
            or self.earlier_digest
            or self.recent_turns
        )

    def with_fact_prefix(self, value: str | None) -> ConversationPromptContext:
        """合并确定性当前任务资料，不改变已保存的历史角色。"""
        clean = str(value or "").strip()
        if not clean:
            return self
        merged = "\n\n".join(
            item for item in (clean, self.fact_context.strip()) if item
        )
        return replace(self, fact_context=merged)

    def legacy_text(self, *, lang: str = "zh") -> str:
        """为尚未迁移的纯文本消费者提供有边界的兼容表示。"""
        chunks: list[str] = []
        if self.fact_context.strip():
            chunks.append(self.fact_context.strip())
        if self.earlier_digest:
            label = (
                "Earlier conversation summary (non-authoritative):"
                if lang == "en"
                else "较早对话摘要（非权威，仅供承接）："
            )
            chunks.append(f"{label}\n" + "\n".join(self.earlier_digest))
        if self.recent_turns:
            label = "Recent conversation:" if lang == "en" else "最近对话："
            lines = []
            for turn in self.recent_turns:
                role = (
                    "User" if turn.role == "user" else "Assistant"
                ) if lang == "en" else (
                    "用户" if turn.role == "user" else "助手"
                )
                separator = ": " if lang == "en" else "："
                lines.append(f"{role}{separator}{turn.content}")
            chunks.append(f"{label}\n" + "\n".join(lines))
        return "\n\n".join(chunks)

    def __str__(self) -> str:
        # 中文兼容格式保留现有测试和旧消费者的可读性；生产模型调用会读取字段。
        return self.legacy_text(lang="zh")

    def __contains__(self, value: object) -> bool:
        return str(value) in str(self)
