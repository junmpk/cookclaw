"""进入意图分类前的轻量语义标准化。

这里只修复可以确定的口语、输入法/ASR 误写和候选组指代，不生成菜名、食材、
口味或忌口。原始文本始终保留，方便日志审计和问题复现。
"""
from __future__ import annotations

from dataclasses import dataclass
import re


_EFFORT_CRITERIA = "省事|简单|容易|方便|快手|快|步骤少|操作少"
_COOKING_OR_CANDIDATE_MARKERS = (
    "菜", "食谱", "菜谱", "做", "吃", "推荐", "候选", "对比", "比较",
    "第一个", "第一道", "第二个", "第二道", "第三个", "第三道", "省事",
)


@dataclass(frozen=True)
class SemanticRewrite:
    raw_text: str
    normalized_text: str
    action_hint: str | None = None
    changes: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.raw_text != self.normalized_text


def _looks_like_cooking_or_candidate(text: str) -> bool:
    return any(marker in text for marker in _COOKING_OR_CANDIDATE_MARKERS)


def _candidate_action_hint(text: str) -> str | None:
    compact = re.sub(r"\s+", "", text)
    candidate_reference = bool(re.search(
        r"(?:这|刚才|上轮|上一轮|上一批|推荐(?:的|中)?)"
        r".{0,6}(?:[0-9一二两三四五六七八九十几]+道菜|几道|这些菜|候选)",
        compact,
    ))
    comparison = any(marker in compact for marker in (
        "对比", "比较", "区别", "差别", "哪个好", "选哪", "哪一道", "哪道更",
        "省事", "简单", "容易", "方便", "快手", "最适合", "更适合",
    ))
    return "compare_candidates" if candidate_reference and comparison else None


def rewrite_user_utterance(text: str) -> SemanticRewrite:
    """把确定无歧义的口语改成下游规则能稳定消费的标准表达。"""
    raw = str(text or "")
    normalized = raw.strip()
    changes: list[str] = []

    # 输入法高频误写；只修正语义唯一的“清除会话记忆”命令。
    rewritten = re.sub(
        r"^(?:请)?(?:帮我)?清楚(?:一下)?(?:会话)?记忆[。.!！]?$",
        "清除记忆",
        normalized,
    )
    if rewritten != normalized:
        changes.append("clear_memory_typo")
        normalized = rewritten

    # “这三个菜/这几个食谱”统一成候选追问规则使用的“这三道菜/这几道菜”。
    rewritten = re.sub(
        r"这\s*([0-9一二两三四五六七八九十几]+)\s*个\s*(?:菜|菜谱|食谱)",
        lambda match: f"这{match.group(1)}道菜",
        normalized,
    )
    if rewritten != normalized:
        changes.append("candidate_group_reference")
        normalized = rewritten

    # 常见输入法/ASR 误写：“哪知道最省事”在候选选择语境中只能是“哪一道最省事”。
    rewritten = re.sub(
        rf"哪知道(?=.{{0,8}}(?:最|更)(?:{_EFFORT_CRITERIA}))",
        "哪一道",
        normalized,
    )
    if rewritten != normalized:
        changes.append("candidate_ordinal_typo")
        normalized = rewritten

    # “这三道菜那一道最适合”通常是“哪一道”的口语/输入法误写。
    rewritten = re.sub(
        r"(这[0-9一二两三四五六七八九十几]+道菜.{0,8})那一道(?=.{0,8}(?:最|更)?适合)",
        r"\1哪一道",
        normalized,
    )
    if rewritten != normalized:
        changes.append("candidate_question_typo")
        normalized = rewritten

    # “怕麻烦”描述的是步骤偏好，不是麻味，也不是需要排除的食材。
    if _looks_like_cooking_or_candidate(normalized):
        rewritten = re.sub(
            r"(?:我)?(?:怕|嫌)(?:太)?麻烦(?!你)|"
            r"(?:我)?不想(?:太)?麻烦(?!你)|"
            r"(?:不要|别)(?:太)?麻烦(?!你)",
            "希望步骤简单",
            normalized,
        )
        if rewritten != normalized:
            changes.append("effort_preference")
            normalized = rewritten

    return SemanticRewrite(
        raw_text=raw,
        normalized_text=normalized,
        action_hint=_candidate_action_hint(normalized),
        changes=tuple(changes),
    )
