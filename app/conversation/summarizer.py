"""本地、确定性的增量摘要兜底。

摘要只保留用户提供的历史线索；旧助手文案不能反向成为事实或下一轮的高权重
上下文。偏好仍由结构化解析器单独维护，不从自然回复推断。
"""
from __future__ import annotations

import re

from app.conversation.models import ConversationTurn
from app.conversation.preference_parser import (
    apply_preference_mutations,
    extract_preference_mutations,
)


_QUESTION_LIKE = ("什么", "为什么", "怎么", "哪", "吗", "嘛", "呢", "？", "?", "这种", "这些")
_TRANSIENT_LIKE_PREFIXES = ("点适合", "适合")
_TRANSIENT_REFERENCE_PREFIXES = ("these", "those", "them", "the options", "the recipes", "the dishes")
_LOW_INFORMATION_TURNS = {
    "嗯", "嗯嗯", "好", "好的", "可以", "行", "ok", "okay", "yes",
    "你好", "您好", "hi", "hello", "hey",
}


def _append_unique(items: list[str], value: str, limit: int = 20) -> None:
    value = value.strip()
    if value and value not in items and len(items) < limit:
        items.append(value)


def sanitize_preferences(preferences: dict[str, list[str]]) -> dict[str, list[str]]:
    """移除旧版本可能从疑问句中误抽取的伪偏好。"""
    out = {}
    for key in ("likes", "dislikes", "allergens", "dietary_constraints", "available_ingredients"):
        values = []
        for raw in (preferences or {}).get(key, []):
            value = str(raw).strip(" ，。；！？?~")
            if (
                not value
                or any(marker in value for marker in _QUESTION_LIKE)
                or (key == "likes" and value.startswith(_TRANSIENT_LIKE_PREFIXES))
                or value.lower().startswith(_TRANSIENT_REFERENCE_PREFIXES)
            ):
                continue
            _append_unique(values, value)
        out[key] = values
    return out


def update_preferences(preferences: dict[str, list[str]], user_text: str) -> dict[str, list[str]]:
    """只抽取用户明确表达的少量偏好，不根据模型回复反推用户画像。"""
    out = sanitize_preferences(preferences)
    for key in ("likes", "dislikes", "allergens", "dietary_constraints", "available_ingredients"):
        out.setdefault(key, [])
    text = re.sub(r"\s+", " ", user_text or "").strip()
    lower_text = text.lower()

    # 偏好和过敏言语行为由公共解析器统一处理。这里不再分别维护场景正则，
    # 规则，避免路由识别正确但摘要器又写入“吃西餐/推荐西餐”之类脏值。
    out, _changed = apply_preference_mutations(
        out,
        extract_preference_mutations(text),
    )
    match = re.search(r"(?:家里有|手边有|现有|只有)\s*([^。！？]{1,40})", text)
    if match:
        blob = match.group(1)
        # “家里有四口人”是用餐人数，不是冰箱食材。
        if not re.search(r"[0-9一二两三四五六七八九十]+\s*(?:口人|口|个人|人吃|人份)", blob):
            for ingredient in re.split(r"[、,，和与\s]+", blob):
                _append_unique(out["available_ingredients"], ingredient)
    english_available = re.search(r"\bi (?:have|only have)\s+([^.!?]{1,60})", lower_text)
    if english_available:
        for ingredient in re.split(r",|\band\b", english_available.group(1)):
            _append_unique(out["available_ingredients"], ingredient)
    return sanitize_preferences(out)


def summarize_incrementally(
    previous_summary: dict,
    turns: list[ConversationTurn],
    preferences: dict[str, list[str]],
) -> dict:
    """压缩较早用户消息；助手输出不作为记忆事实重新注入。"""
    digest = [
        str(item)
        for item in (previous_summary or {}).get("conversation_digest") or []
        if str(item).strip() and not str(item).lstrip().startswith("助手：")
    ]
    for turn in turns:
        if turn.role != "user":
            continue
        content = re.sub(r"\s+", " ", turn.content or "").strip()
        normalized = content.lower().strip("，。；！？?~,.! ")
        if not content or normalized in _LOW_INFORMATION_TURNS:
            continue
        line = f"用户：{content[:160]}"
        if line not in digest:
            digest.append(line)
    # 总摘要设硬上限，避免 Redis payload 和后续提示词无限增长。
    digest = digest[-16:]
    return {
        "conversation_digest": digest,
        "preferences": preferences,
        "summary_version": int((previous_summary or {}).get("summary_version", 0)) + 1,
        "summary_kind": "user_context_v2",
    }
