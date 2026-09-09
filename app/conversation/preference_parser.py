"""用户明确饮食偏好声明的通用解析。

这里只识别“我喜欢/不喜欢/不吃/以后别推荐”这类明确言语行为，不维护具体菜名
或食材场景分支。解析结果同时供 Redis 会话状态和异步长期画像写入使用，避免
路由、摘要器和画像存储各自用一套正则。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PreferenceOperation = Literal["add", "remove"]
PreferenceBucket = Literal[
    "likes",
    "dislikes",
    "allergens",
    "dietary_constraints",
]

_REFERENCE_TARGETS = {
    "这些", "这些菜", "这几个", "这几道", "这批", "刚才的", "刚才这些",
    "those", "these", "these dishes", "these recipes", "the options",
}
_TRANSIENT_CONTEXT_MARKERS = (
    "这顿", "这次", "今天", "今晚", "明天", "朋友", "客人", "聚餐", "大家", "我们",
    "this meal", "today", "tonight", "tomorrow", "friends", "guests", "party", "we ",
)
_TRAILING_FILLERS_RE = re.compile(
    r"(?:以后|今后)?(?:都)?(?:别|不要)(?:再)?(?:给我)?(?:推荐|安排|选).*$|"
    r"(?:以后|今后)[，,。；;！？!?].*$|"
    r"(?:明白|懂了?|知道|记住)(?:了)?(?:吗|嘛|么)?[。！？!?]*$|"
    r"(?:可以|行|好)(?:吗|嘛|么)?[。！？!?]*$",
    flags=re.IGNORECASE,
)
_TARGET_SUFFIX_RE = re.compile(
    r"(?:给我|帮我)?(?:推荐|安排|选择|选)(?:的)?$|"
    r"(?:就行|即可|为主|方向)$",
    flags=re.IGNORECASE,
)
_QUESTION_TARGET_RE = re.compile(
    r"^(?:什么|哪些|哪个|哪种|why|what|which)\b|"
    r"(?:为什么|怎么|如何|能不能|可以吗|是什么)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PreferenceMutation:
    operation: PreferenceOperation
    bucket: PreferenceBucket
    value: str
    source_span: str
    scope: Literal["session", "temporal", "long_term_candidate"] = (
        "long_term_candidate"
    )

    def to_dict(self) -> dict[str, str]:
        return {
            "operation": self.operation,
            "bucket": self.bucket,
            "value": self.value,
            "source_span": self.source_span,
            "scope": self.scope,
        }


def _clean_target(raw: str) -> str:
    value = re.sub(r"\s+", " ", str(raw or "")).strip(
        " \t\r\n，,。；;！？!?“”\"'：:"
    )
    value = _TRAILING_FILLERS_RE.sub("", value).strip(
        " \t\r\n，,。；;！？!?“”\"'：:"
    )
    value = _TARGET_SUFFIX_RE.sub("", value).strip(
        " \t\r\n，,。；;！？!?“”\"'：:"
    )
    value = re.sub(
        r"^(?:吃|食用|推荐|安排|选择|选)\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    value = re.sub(
        r"\s+(?:anymore|again|from now on)$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    value = re.sub(r"了$", "", value).strip()
    return value[:48]


def _scope_for(text: str, value: str) -> str:
    lower = str(text or "").lower()
    if any(marker in lower for marker in _TRANSIENT_CONTEXT_MARKERS):
        return "session"
    if any(marker in lower for marker in (
        "最近", "现在", "目前", "这段时间", "temporarily", "currently", "recently",
    )):
        return "temporal"
    if value.lower() in {
        "减肥", "减脂", "低脂", "控卡", "weight loss", "low-fat", "low fat",
    }:
        return "temporal"
    return "long_term_candidate"


def _valid_target(value: str) -> bool:
    normalized = value.strip().lower()
    if (
        not normalized
        or normalized in _REFERENCE_TARGETS
        or _QUESTION_TARGET_RE.search(normalized)
    ):
        return False
    if any(marker in normalized for marker in (
        "这些选项", "这些菜", "这几道", "刚才推荐", "these options",
        "these dishes", "those recipes",
    )):
        return False
    return 1 <= len(value) <= 48


def _explicit_dietary_constraint_removal_span(
    text: str,
    term: str,
) -> str | None:
    """Return the explicit statement that retires one remembered diet goal.

    Keep this narrower than the generic negation checks below: questions such as
    ``减肥可以吃吗`` or ``我已经不需要减肥了吗`` must not mutate memory.
    """
    if not re.search(r"[\u4e00-\u9fff]", term):
        return None
    escaped = re.escape(term)
    patterns = (
        (
            rf"(?:我|本人)\s*(?:现在|目前|最近)?\s*(?:已经)?\s*"
            rf"(?:不再|不)\s*(?:进行)?\s*{escaped}(?:了|啦)?"
            rf"(?=$|[，,。；;！!])"
        ),
        (
            rf"(?:我|本人)?\s*(?:已经)?\s*(?:停止|结束|取消|放弃)"
            rf"(?:了)?\s*(?:进行)?\s*{escaped}(?:了|啦)?"
            rf"(?=$|[，,。；;！!])"
        ),
        (
            rf"(?:以后|今后).{{0,24}}(?:不要|别)(?:再)?.{{0,12}}"
            rf"(?:参考|按照|考虑).{{0,12}}{escaped}(?:这个)?"
            rf"(?:标准|目标|要求)?(?=$|[，,。；;！!])"
        ),
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _append(
    out: list[PreferenceMutation],
    *,
    operation: PreferenceOperation,
    bucket: PreferenceBucket,
    raw_value: str,
    source_span: str,
    text: str,
) -> None:
    value = _clean_target(raw_value)
    if not _valid_target(value):
        return
    mutation = PreferenceMutation(
        operation=operation,
        bucket=bucket,
        value=value,
        source_span=str(source_span or "").strip(),
        scope=_scope_for(text, value),
    )
    signature = (mutation.operation, mutation.bucket, mutation.value.lower())
    if any(
        (item.operation, item.bucket, item.value.lower()) == signature
        for item in out
    ):
        return
    out.append(mutation)


def extract_preference_mutations(text: str) -> list[PreferenceMutation]:
    """提取明确偏好变更；返回空列表表示应继续走普通对话意图。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return []
    lower = value.lower()
    out: list[PreferenceMutation] = []

    # “我没有不喜欢/我不再讨厌”是撤销旧偏好，不是新增一个否定偏好。
    remove_patterns = (
        r"(?:我)?(?:没有|并不|不是)(?:真的)?(?:不喜欢|不爱吃|讨厌|不吃)\s*"
        r"(?P<target>[^，,。；;！？!?]{1,48})",
        r"(?:我)?(?:不再|已经不)(?:不喜欢|讨厌|排斥|避开)\s*"
        r"(?P<target>[^，,。；;！？!?]{1,48})",
        r"(?:取消|删除|清除|忘掉).{0,6}(?:不喜欢|不吃|忌口|饮食偏好|食物偏好)\s*"
        r"(?P<target>[^，,。；;！？!?]{1,48})",
    )
    for pattern in remove_patterns:
        for match in re.finditer(pattern, value, flags=re.IGNORECASE):
            _append(
                out,
                operation="remove",
                bucket="dislikes",
                raw_value=match.group("target"),
                source_span=match.group(0),
                text=value,
            )

    negative_patterns = (
        r"(?:^|[，,。；;！？!?但是不过但]\s*)(?:也)?(?:我是说|我的意思是)?(?:我|本人)?"
        r"(?:最近|现在|目前|这段时间|以后|今后)?(?:一直)?"
        r"(?:不喜欢(?:吃)?|不爱吃|讨厌|不吃|避开|排斥)\s*"
        r"(?P<target>[^，,。；;！？!?]{1,48})",
        r"(?:^|[，,。；;！？!?]\s*)(?:我|本人)?(?:以后|今后)?"
        r"(?:别|不要)(?:再)?(?:给我)?(?:推荐|安排|选择|选)\s*"
        r"(?P<target>[^，,。；;！？!?]{1,48})",
        r"\bi\s+(?:do not|don't|dont)\s+(?:like|eat|want)\s+"
        r"(?P<target>[^,.!?;]{1,48})",
        r"\b(?:please\s+)?(?:do not|don't|dont|never)\s+recommend\s+"
        r"(?P<target>[^,.!?;]{1,48})",
    )
    for pattern in negative_patterns:
        for match in re.finditer(pattern, value, flags=re.IGNORECASE):
            # 已被撤销表达覆盖的片段不能再次新增。
            if re.search(
                r"(?:没有|并不|不是|不再|已经不).{0,6}$",
                value[:match.start()],
                flags=re.IGNORECASE,
            ):
                continue
            _append(
                out,
                operation="add",
                bucket="dislikes",
                raw_value=match.group("target"),
                source_span=match.group(0),
                text=value,
            )

    positive_patterns = (
        r"(?:^|[，,。；;！？!?但是不过但]\s*)(?:我|本人)"
        r"(?:最近|现在|目前|这段时间)?(?:一直|比较|很|挺)?"
        r"(?:喜欢吃?|爱吃|偏爱)\s*"
        r"(?P<target>[^，,。；;！？!?]{1,48})",
        r"\bi\s+(?:like|love|prefer)\s+(?P<target>[^,.!?;]{1,48})",
    )
    for pattern in positive_patterns:
        for match in re.finditer(pattern, value, flags=re.IGNORECASE):
            _append(
                out,
                operation="add",
                bucket="likes",
                raw_value=match.group("target"),
                source_span=match.group(0),
                text=value,
            )

    # 过敏是明确的硬约束，同时保留到 dislikes，供现有检索过滤兼容消费。
    allergy_patterns = (
        r"(?:我|本人)?(?:对)?\s*(?P<target>[^，,。；;！？!?]{1,32}?)"
        r"(?:严重|重度|非常|特别)?\s*过敏",
        r"\bi(?:\s+am|'m)?\s+(?:severely\s+|seriously\s+)?allergic\s+to\s+"
        r"(?P<target>[^,.!?;]{1,48})",
    )
    for pattern in allergy_patterns:
        for match in re.finditer(pattern, value, flags=re.IGNORECASE):
            for bucket in ("allergens", "dislikes"):
                _append(
                    out,
                    operation="add",
                    bucket=bucket,
                    raw_value=match.group("target"),
                    source_span=match.group(0),
                    text=value,
                )

    diet_terms = (
        "减肥", "减脂", "低脂", "控卡", "素食", "纯素", "清真",
        "vegetarian", "vegan", "low-fat", "low fat", "weight loss", "halal",
    )
    for term in diet_terms:
        if term.lower() not in lower:
            continue
        removal_span = _explicit_dietary_constraint_removal_span(value, term)
        if removal_span:
            _append(
                out,
                operation="remove",
                bucket="dietary_constraints",
                raw_value=term,
                source_span=removal_span,
                text=value,
            )
            continue
        if re.search(
            rf"(?:没说|没有说|不是|不再|不想|取消|别记).{{0,10}}{re.escape(term)}|"
            rf"\b(?:not|never|did(?:n't| not) say).{{0,18}}{re.escape(term)}",
            lower,
            flags=re.IGNORECASE,
        ):
            continue
        if term in {"减肥", "减脂", "低脂", "控卡"}:
            explicit = bool(re.search(
                rf"(?:我|本人)?(?:最近|现在|目前|这段时间)"
                rf"(?:正在|在|开始|打算|需要|坚持)?(?:进行)?{re.escape(term)}|"
                rf"(?:我|本人)(?:正在|在|开始|打算|需要|坚持)"
                rf"(?:进行)?{re.escape(term)}",
                lower,
                flags=re.IGNORECASE,
            ))
        elif term in {"素食", "纯素", "清真"}:
            explicit = bool(re.search(
                rf"(?:我|本人)(?:是|吃|遵循|需要|一直|目前).{{0,8}}"
                rf"{re.escape(term)}|我吃素",
                lower,
                flags=re.IGNORECASE,
            ))
        else:
            explicit = bool(re.search(
                rf"\b(?:i am|i'm|i follow|i have been|i'm on|i am on)"
                rf".{{0,24}}{re.escape(term)}",
                lower,
                flags=re.IGNORECASE,
            ))
        if explicit:
            _append(
                out,
                operation="add",
                bucket="dietary_constraints",
                raw_value=term,
                source_span=term,
                text=value,
            )
    return out[:12]


def apply_preference_mutations(
    preferences: dict[str, list[str]] | None,
    mutations: list[PreferenceMutation],
) -> tuple[dict[str, list[str]], list[PreferenceMutation]]:
    """把变更应用到偏好字典，并让同值的喜欢/不喜欢互相纠正。"""
    keys = (
        "likes",
        "dislikes",
        "allergens",
        "dietary_constraints",
        "available_ingredients",
    )
    result = {
        key: [
            str(item).strip()
            for item in (preferences or {}).get(key, [])
            if str(item).strip()
        ]
        for key in keys
    }
    changed: list[PreferenceMutation] = []
    for mutation in mutations:
        bucket = mutation.bucket
        values = result.setdefault(bucket, [])
        target_lower = mutation.value.lower()
        before = list(values)
        if mutation.operation == "add":
            values[:] = [
                item for item in values if str(item).strip().lower() != target_lower
            ]
            values.append(mutation.value)
            opposite = "dislikes" if bucket == "likes" else (
                "likes" if bucket == "dislikes" else None
            )
            if opposite:
                result[opposite] = [
                    item for item in result.get(opposite, [])
                    if str(item).strip().lower() != target_lower
                ]
        else:
            values[:] = [
                item for item in values if str(item).strip().lower() != target_lower
            ]
        if values != before:
            changed.append(mutation)
    return result, changed


_PREFERENCE_QUERY_PATTERNS = (
    re.compile(r"你(?:都)?记(?:得|住)(?:了)?我(?:哪些|什么)?(?:饮食|口味|忌口)?偏好"),
    re.compile(r"你(?:都)?记(?:得|住)(?:了)?我(?:不喜欢|不吃|喜欢|爱吃)(?:什么|哪些)"),
    re.compile(r"(?:我有哪些|我的)(?:饮食|口味|忌口)?偏好(?:是什么)?"),
    re.compile(r"你保存了我哪些(?:饮食|口味|忌口)?(?:偏好|信息)"),
    re.compile(r"what (?:food |dietary )?preferences do you remember", re.IGNORECASE),
)


def is_preference_memory_query(text: str) -> bool:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return bool(value) and any(
        pattern.search(value) for pattern in _PREFERENCE_QUERY_PATTERNS
    )


def has_additional_food_task(text: str) -> bool:
    """偏好声明之外是否还要求搜索、推荐或做法；支持一轮多意图。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return False
    # “以后推荐时不要再参考这个目标”描述的是偏好的适用规则，不是要求
    # 当前立刻推荐。先去掉整段策略说明，再判断是否还存在独立食物任务。
    scrubbed = re.sub(
        r"(?:以后|今后).{0,32}?(?:不要|别)(?:再)?.{0,16}?"
        r"(?:参考|按照|考虑).{0,16}?(?:标准|目标|要求)?",
        "",
        value,
        flags=re.IGNORECASE,
    )
    scrubbed = re.sub(
        r"(?:别|不要)(?:再)?(?:给我)?(?:推荐|安排|选择|选)",
        "",
        scrubbed,
    )
    return bool(re.search(
        r"(?:帮我|给我|现在|今晚|这次|再).{0,10}(?:推荐|找|选|安排)|"
        r"(?:吃什么|做什么|怎么做|有什么菜谱|来几道|来几个)|"
        r"\b(?:recommend|suggest|find|what should i eat|how to cook)\b",
        scrubbed,
        flags=re.IGNORECASE,
    ))
