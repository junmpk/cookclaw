"""澄清回复的纯解析与状态迁移语义。

本模块不访问模型、存储或通道。省略回复只能结合明确的待答维度解释，避免不同
入口各自维护正则并产生上下文漂移。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

PARTY_CONSTRAINTS = "party_constraints"
PARTY_CONSTRAINTS_DETAIL = "party_constraints_detail"


class ConstraintReply(StrEnum):
    NO_CONSTRAINTS = "no_constraints"
    HAS_CONSTRAINTS = "has_constraints"
    CONSTRAINT_DETAILS = "constraint_details"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ClarificationTransition:
    reply: ConstraintReply
    constraints_confirmed: bool | None = None
    next_dimension: str | None = None

    @property
    def recognized(self) -> bool:
        return self.reply is not ConstraintReply.UNKNOWN


_BARE_NO = {"没有", "没", "无", "none"}
_BARE_YES = {"有", "有的", "有啊", "yes", "yeah", "yep"}
_EXPLICIT_NO_RE = re.compile(
    r"(?:没(?:有)?|无)(?:什么|啥|任何)?(?:忌口|过敏|饮食限制|饮食要求|"
    r"宗教饮食要求|限制|要求)|不过敏|都能吃|什么都能吃|"
    r"\b(?:no dietary restrictions?|no restrictions?|no allergies|none)\b",
    flags=re.IGNORECASE,
)
_DETAIL_RE = re.compile(
    r"不吃|不要|忌口|过敏|素食|纯素|清真|回族|穆斯林|"
    r"\b(?:without|allergic|vegetarian|vegan|halal|muslim)\b",
    flags=re.IGNORECASE,
)


def normalize_clarification_reply(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower().strip(
        " ，,。.!！?？"
    )


def parse_constraint_reply(
    text: str,
    pending_dimension: str | None,
) -> ConstraintReply:
    """结合待答维度解释饮食安全回复。

    裸“没有”仅能回答一级安全问题；离开该上下文时保持 UNKNOWN。明确的限制细节
    只在安全澄清链中消费，避免把普通闲聊误并入当前推荐任务。
    """
    if pending_dimension not in {PARTY_CONSTRAINTS, PARTY_CONSTRAINTS_DETAIL}:
        return ConstraintReply.UNKNOWN
    value = normalize_clarification_reply(text)
    if not value:
        return ConstraintReply.UNKNOWN

    if pending_dimension == PARTY_CONSTRAINTS:
        if value in _BARE_NO or _EXPLICIT_NO_RE.search(value):
            return ConstraintReply.NO_CONSTRAINTS
        if value in _BARE_YES:
            return ConstraintReply.HAS_CONSTRAINTS

    if _DETAIL_RE.search(value):
        return ConstraintReply.CONSTRAINT_DETAILS
    return ConstraintReply.UNKNOWN


def reduce_constraint_clarification(
    text: str,
    pending_dimension: str | None,
) -> ClarificationTransition:
    """返回可测试的状态迁移决策，不直接修改 SearchRequest。"""
    reply = parse_constraint_reply(text, pending_dimension)
    if reply is ConstraintReply.NO_CONSTRAINTS:
        return ClarificationTransition(reply, constraints_confirmed=True)
    if reply is ConstraintReply.HAS_CONSTRAINTS:
        return ClarificationTransition(
            reply,
            constraints_confirmed=False,
            next_dimension=PARTY_CONSTRAINTS_DETAIL,
        )
    if reply is ConstraintReply.CONSTRAINT_DETAILS:
        return ClarificationTransition(reply, constraints_confirmed=True)
    return ClarificationTransition(reply)
