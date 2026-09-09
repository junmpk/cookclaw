"""当前菜谱任务的确定性条件追加/修改判定。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.orchestrator.search_request import SearchRequest


_CONDITION_MARKERS = (
    "清淡", "少油", "别太油", "不要太油",
    "不辣", "不要辣", "别辣", "微辣", "稍微辣", "辣一点",
    "简单", "快手", "省事", "别太复杂",
    "孩子", "儿童", "小朋友", "老人",
    "不要", "不吃", "不放", "不加", "避开", "排除",
    "再加", "加上", "还有", "也用",
    "lighter", "less oily", "not spicy", "mild", "spicier",
    "simple", "easy", "quick", "for kids", "children",
    "without", "exclude", "avoid", "add ",
)
_REVISION_MARKERS = (
    "算了", "改成", "改为", "换成", "调整", "也可以", "没关系",
    "changed my mind", "instead", "actually", "can be",
)
_EXPLICIT_RECOMMENDATION_RE = re.compile(
    r"(?:给我|帮我|请|想让你)?(?:重新)?推荐(?:一些|几个|几道|点|一份)?"
    r"|(?:给我|帮我|请)?安排(?:一顿|一桌|一下|个)?(?:饭|菜|菜单)"
    r"|(?:plan|recommend|suggest)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_PREVIOUS_TASK_REFERENCE_RE = re.compile(
    r"刚才|上一批|前面(?:那批|那些)|按(?:刚才|之前|这些|原来)(?:的)?条件"
    r"|换一批|再来一批|previous|earlier|same filters?|another batch",
    flags=re.IGNORECASE,
)
_CANDIDATE_ORDINAL = re.compile(
    r"第\s*(?:10|[1-9一二三四五六七八九十])\s*[个道]"
    r"|\b(?:first|second|third|fourth|fifth)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ConditionReduction:
    request: SearchRequest
    changed_fields: tuple[str, ...]
    operation: str


def looks_like_new_recipe_task(text: str) -> bool:
    """识别用户明确发起的下一次推荐；引用上一批时仍视为条件跟进。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not value or _EXPLICIT_PREVIOUS_TASK_REFERENCE_RE.search(value):
        return False
    return bool(_EXPLICIT_RECOMMENDATION_RE.search(value))


def looks_like_condition_update(text: str) -> bool:
    """只接管有明确条件证据的短跟进，不吞掉闲聊、设备查询或新任务。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not value or len(value) > 120:
        return False
    if looks_like_new_recipe_task(value):
        return False
    if re.sub(r"[，,。.!！?？\s]+", "", value) in {
        "不要", "不要了", "不吃了", "不用了", "先不要了",
        "no", "nomore", "nevermind",
    }:
        return False
    # “第二个不要/第一道太麻烦”属于候选排除，必须保留原始序号处理。
    if _CANDIDATE_ORDINAL.search(value):
        return False
    has_condition = any(marker in value for marker in _CONDITION_MARKERS)
    if has_condition:
        return True
    return (
        any(marker in value for marker in _REVISION_MARKERS)
        and any(marker in value for marker in ("辣", "油", "简单", "复杂", "spicy", "easy"))
    )


def reduce_recipe_conditions(
    previous: SearchRequest,
    followup_text: str,
) -> ConditionReduction | None:
    if not looks_like_condition_update(followup_text):
        return None
    before = previous.model_dump()
    refined = previous.refine(followup_text)
    after = refined.model_dump()
    changed_fields = tuple(
        key
        for key in before
        if key not in {"original_text"}
        and before.get(key) != after.get(key)
    )
    if not changed_fields:
        return None

    removed = any(
        any(
            item not in (after.get(key) or [])
            for item in (before.get(key) or [])
        )
        for key in changed_fields
        if isinstance(before.get(key), list) and isinstance(after.get(key), list)
    )
    added = any(
        any(
            item not in (before.get(key) or [])
            for item in (after.get(key) or [])
        )
        for key in changed_fields
        if isinstance(before.get(key), list) and isinstance(after.get(key), list)
    )
    operation = "replace" if removed and added else ("remove" if removed else "add")
    return ConditionReduction(
        request=refined,
        changed_fields=changed_fields,
        operation=operation,
    )
