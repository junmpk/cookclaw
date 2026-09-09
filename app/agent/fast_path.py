"""菜谱检索执行、结果事实化和通道格式化工具。

意图分类与业务决策统一由 ``app.orchestrator.router`` 负责；本模块不再维护
另一套规则路由，避免同一句话在两个入口得到不同结论。
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.observability.trace import record_timeout, record_tool_call

logger = logging.getLogger(__name__)

def _parse_ingredients(ingredients, lang: str | None = None) -> list:
    """解析食材列表"""
    if isinstance(ingredients, list):
        items = []
        for item in ingredients:
            s = str(item).strip()
            if s:
                items.append(s)
    elif isinstance(ingredients, str):
        items = [ingredients]
    else:
        items = []
    if lang == "en":
        # 英文模式宁可少展示一个脏字段，也不能把中文元数据夹进英文回复。
        items = [item for item in items if _has_latin(item) and not _has_cjk(item)]
        from app.orchestrator.ingredient_display import clean_english_ingredient_tokens

        return clean_english_ingredient_tokens(items, limit=5)
    return items[:5]


def _has_cjk(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)


def _has_latin(text: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in text)


def _expand_tag(tag: str) -> list:
    """把双语标签拆成可按语言过滤的原子标签。"""
    tag = str(tag).strip()
    if not tag:
        return []

    pieces = [tag]
    if " / " in tag:
        pieces = [p.strip() for p in tag.split("/") if p.strip()]

    out = []
    for piece in pieces:
        out.extend(p.strip() for p in re.split(r"[、,，;；\|]+", piece) if p.strip())
    return out


def _parse_tags(tags, lang: str = "zh") -> list:
    """解析标签列表，并按回复语言只保留同语种标签。"""
    if isinstance(tags, dict):
        raw_tags = [v for v in tags.values() if v]
    elif isinstance(tags, list):
        raw_tags = [str(t) for t in tags if t]
    else:
        return []

    expanded = []
    for raw in raw_tags:
        expanded.extend(_expand_tag(raw))

    filtered = []
    for tag in expanded:
        has_cjk = _has_cjk(tag)
        has_latin = _has_latin(tag)
        if lang == "en":
            if has_latin and not has_cjk:
                filtered.append(tag)
        else:
            if has_cjk:
                filtered.append(tag)

    # 中文模式维持旧兜底；英文模式宁可不展示标签，也不回退到中文脏数据。
    if not filtered and lang != "en":
        filtered = expanded

    out = []
    seen = set()
    for tag in filtered:
        if tag not in seen:
            out.append(tag)
            seen.add(tag)
        if len(out) >= 4:
            break
    return out


def detect_lang(text: str) -> str:
    """判定输入主要语言（仅区分 zh/en）：中文字符占比 >0.3 视为中文，否则英文。

    用于在意图阶段基于【原始用户输入】定语言，再显式透传给检索与文案，
    避免检索内部按可能被改写的 keywords 误判语言。空/无中文默认 zh（项目主语言）。
    数据仅含 zh/en 两套，其他语种一律落到 en。
    """
    if not text:
        return "zh"
    zh = sum(1 for c in text if 0x4e00 <= ord(c) <= 0x9fff)
    total = len(text.replace(" ", ""))
    if total == 0:
        return "zh"
    return "zh" if (zh / total) > 0.3 else "en"


# ─── 多语言文案 ──────────────────────────────────────────────────────────

_MSG = {
    "zh": {
        "search_empty": "这次没在菜谱库里对上合适的结果。换个说法，或者给我一个主料、菜系或做法，我马上接着找。",
        "greeting": (
            "嗨，我是 CookClaw，你的厨房搭子 🍳\n"
            "今天想吃什么、手边有什么，直接丢给我。你负责说馋什么，我负责把菜选明白。"
        ),
    },
    "en": {
        "search_empty": "Nothing in the recipe library lined up well this time. Try another phrase, or give me one ingredient, cuisine, or cooking method and I'll search again.",
        "greeting": (
            "Hey, I'm CookClaw — your kitchen sidekick 🍳\n"
            "Tell me what you're craving or what you have on hand. You bring the appetite; I'll make the choices easier."
        ),
    },
}


def _msg(lang: str, key: str) -> str:
    """取多语言文案，未知语言回退 zh。"""
    return _MSG.get(lang, _MSG["zh"]).get(key, _MSG["zh"][key])


def normalize_exact_search_label(value: str) -> str:
    """从精确搜索原话中裁出仅用于展示的菜名，不参与召回或事实判断。"""
    original = re.sub(r"\s+", " ", str(value or "")).strip(" ，,。；;！？?~")
    if not original:
        return ""
    label = re.sub(
        r"^(?:(?:请|麻烦)(?:你)?\s*)?"
        r"(?:(?:帮我|给我|我想|想要|我要)\s*)?"
        r"(?:(?:找|搜|搜索|查|看看)\s*)?"
        r"(?:(?:一|1)\s*(?:道|个|份)\s*)?",
        "",
        original,
        flags=re.IGNORECASE,
    )
    label = re.sub(
        r"(?:怎么做|如何做|如何制作|的?(?:做法|菜谱|食谱|详细步骤|具体步骤))$",
        "",
        label,
        flags=re.IGNORECASE,
    ).strip(" ，,。；;！？?~")
    if re.search(r"[A-Za-z]", label):
        label = re.sub(
            r"^(?:please\s+)?(?:help\s+me\s+)?"
            r"(?:find|search\s+for|show\s+me)\s+(?:a|an|one)?\s*",
            "",
            label,
            flags=re.IGNORECASE,
        )
        label = re.sub(
            r"\s+(?:how\s+to\s+(?:make|cook)|recipe(?:\s+steps)?)$",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip(" ,.;!?~")
    return label or original


def _empty_search_message(query: str, search_result: dict, lang: str = "zh") -> str:
    """按真实淘汰原因解释空结果，避免所有场景复读同一段拒绝话术。"""
    request = search_result.get("_search_request") or {}
    ingredients = [
        str(item).strip() for item in request.get("ingredients") or []
        if str(item).strip()
    ]
    label = re.sub(r"\s+", " ", str(query or "")).strip(" ，。；！？?")[:40]

    if search_result.get("_exact_not_found"):
        label = normalize_exact_search_label(label)[:40]
        if lang == "en":
            return (
                f'I could not verify "{label}" in the recipe library, so I will not '
                "fill in a recipe from the web. Try another recipe name, or ask me "
                "to search the library for similar options."
            )
        return (
            f"菜谱库里暂时没有找到可核验的“{label}”，我不会从网上补一份做法。"
            "你可以换个菜名，或者让我继续从库里找相似菜。"
        )

    if search_result.get("_positive_filtered") and ingredients:
        focus = "、".join(ingredients[:2])
        if lang == "en":
            return (
                f"I found nearby recipes, but none clearly verified {focus} as an ingredient. "
                "Give me an acceptable substitute and I'll keep looking."
            )
        return (
            f"相近的菜谱有一些，但还没有一道能确认包含“{focus}”。"
            "你告诉我能替换成什么主料，我就顺着那个方向继续找。"
        )

    if search_result.get("_hard_filtered"):
        if lang == "en":
            return (
                "Once I kept your dietary exclusions in place, no suitable option remained. "
                "Keep those exclusions and change just the ingredient or cooking method."
            )
        return (
            "把你说的忌口和饮食限制保留后，这轮没有剩下合适的菜。"
            "限制不用改，换一个主料或做法就行，我再搜一轮。"
        )

    if search_result.get("_relevance_filtered_count"):
        if lang == "en":
            return (
                f'"{label}" is still fairly broad, and the nearby matches were not solid enough. '
                "Add one ingredient, cuisine, or cooking method and I can narrow it down."
            )
        return (
            f"“{label}”这个范围比较宽，刚才对上的几道还不够稳。"
            "补一个主料、菜系或做法就行，我马上缩小范围。"
        )

    return _msg(lang, "search_empty")


def build_device_cta(results: list[dict], lang: str = "zh") -> str:
    """搜索完成后的轻引导；先引导看详情，不提前把对话推向设备。"""
    names = [
        str((item.get("metadata") or {}).get("name") or "").strip()
        for item in results
        if str((item.get("metadata") or {}).get("name") or "").strip()
    ]
    if not names:
        return ""
    if lang == "en":
        if len(names) == 1:
            return f"Want to look closer at {names[0]}? Reply “details 1” and I’ll open the full recipe."
        visible = min(len(names), 6)
        choices = ", ".join(f"“details {index}”" for index in range(1, visible + 1))
        return f"Which one feels closest? Reply {choices} to open the full recipe."
    if len(names) == 1:
        return f"想仔细看看【{names[0]}】的话，回复「详情 1」，我把完整食材和步骤展开。"
    visible = min(len(names), 6)
    choices = "、".join(f"「详情 {index}」" for index in range(1, visible + 1))
    return f"你现在更偏哪一道？直接回复{choices}，我把完整食材和步骤展开。"


def _recipe_detail_reference(metadata: dict, lang: str) -> str:
    detail = metadata.get("recipe_detail")
    if not isinstance(detail, dict):
        return ""
    values = []
    seconds = detail.get("cooking_time_seconds")
    try:
        seconds_value = int(float(seconds))
    except (TypeError, ValueError):
        seconds_value = 0
    if seconds_value > 0:
        if seconds_value < 3600:
            minutes = max(1, round(seconds_value / 60))
            values.append(f"about {minutes} min" if lang == "en" else f"约{minutes}分钟")
        else:
            hours, remainder = divmod(seconds_value, 3600)
            minutes = round(remainder / 60)
            if lang == "en":
                values.append(f"about {hours} hr {minutes} min" if minutes else f"about {hours} hr")
            else:
                values.append(f"约{hours}小时{minutes}分钟" if minutes else f"约{hours}小时")
    servings = detail.get("servings")
    if isinstance(servings, (int, float)) and not isinstance(servings, bool) and servings > 0:
        value = f"{servings:g}" if isinstance(servings, float) else str(servings)
        values.append(f"{value} servings" if lang == "en" else f"{value}人份")
    steps = detail.get("steps")
    if isinstance(steps, list) and steps:
        values.append(f"{len(steps)} steps" if lang == "en" else f"{len(steps)}步")
    return " · ".join(values)


def _grounded_recipe_reason(metadata: dict, lang: str) -> str:
    """只复述结构化菜谱字段；不根据常识补风味、难度、步骤或功效。"""
    tags = _parse_tags(metadata.get("tags", []), lang=lang)[:2]
    ingredients = _parse_ingredients(metadata.get("ingredients", []), lang=lang)[:3]
    reference = _recipe_detail_reference(metadata, lang)
    if lang == "en":
        if tags and reference:
            return f"It is tagged {', '.join(tags)}, with an estimated cooking reference of {reference}."
        if tags:
            return "It leans toward " + ", ".join(tags) + "."
        if ingredients and reference:
            return f"It uses {', '.join(ingredients)}, with an estimated cooking reference of {reference}."
        if ingredients:
            return "Its main ingredients include " + ", ".join(ingredients) + "."
        return "This one stays close to the direction you asked for."
    if tags and reference:
        return f"它偏{'、'.join(tags)}，做饭参考是{reference}。"
    if tags:
        return "它更偏" + "、".join(tags) + "，可以先放进备选。"
    if ingredients and reference:
        return f"主要用到{'、'.join(ingredients)}，做饭参考是{reference}。"
    if ingredients:
        return "主要用到" + "、".join(ingredients) + "。"
    return "这道和你刚才说的方向比较接近。"


def _grounded_recommendation(results: list[dict], lang: str) -> dict:
    """推荐最终出口的事实白名单。

    自由模型文案和旧缓存 recommendation 均不进入最终回复；因此回复里的菜名
    只能来自当前 ``results``，其余陈述只能来自同一条结果的结构化字段。
    """
    if lang == "en":
        opening = (
            "Here are the current options side by side."
            if len(results) > 1 else
            "Here is the current match."
        )
    else:
        opening = (
            "我把这几道放在一起，下面只列当前菜谱里能核验到的信息。"
            if len(results) > 1 else
            "先看这道，下面只列当前菜谱里能核验到的信息。"
        )
    reasons: dict[str, str] = {}
    for result in results:
        metadata = result.get("metadata") or {}
        recipe_id = str(metadata.get("recipe_id") or result.get("id") or "").strip()
        if recipe_id:
            reasons[recipe_id] = _grounded_recipe_reason(metadata, lang)
    has_comparison_facts = sum(
        bool(
            _parse_tags((item.get("metadata") or {}).get("tags", []), lang=lang)
            or _parse_ingredients((item.get("metadata") or {}).get("ingredients", []), lang=lang)
            or _recipe_detail_reference(item.get("metadata") or {}, lang)
        )
        for item in results
    ) >= 2
    return {
        "opening": opening,
        "strategy": (
            "可以对照下面已有的食材、标签和做饭参考来选。"
            if lang != "en" and has_comparison_facts else
            "Compare only the ingredients, tags, and cooking references shown below."
            if lang == "en" and has_comparison_facts else ""
        ),
        "recipe_reasons": reasons,
        "closing": build_device_cta(results, lang),
    }


def _current_result_recommendation(search_result: dict, results: list[dict], lang: str) -> dict:
    """只接纳与本轮结果 ID/菜名对齐的已校验叙述，旧快照串味则整体降级。"""
    fallback = _grounded_recommendation(results, lang)
    if search_result.get("_recommendation_source") != "grounded_v2":
        return fallback
    raw = search_result.get("_recommendation")
    if not isinstance(raw, dict):
        return fallback
    allowed_names = {
        str((item.get("metadata") or {}).get("name") or "").strip()
        for item in results
    }
    allowed_ids = {
        str((item.get("metadata") or {}).get("recipe_id") or item.get("id") or "").strip()
        for item in results
    }
    text_fields = {
        key: re.sub(r"\s+", " ", str(raw.get(key) or "")).strip()
        for key in ("opening", "strategy", "closing")
    }
    joined = " ".join(text_fields.values())
    if any(name not in allowed_names for name in re.findall(r"【([^】]+)】", joined)):
        return fallback
    reasons = {}
    for recipe_id, reason in (raw.get("recipe_reasons") or {}).items():
        key = str(recipe_id).strip()
        text = re.sub(r"\s+", " ", str(reason or "")).strip()
        if key in allowed_ids and not any(
            name not in allowed_names for name in re.findall(r"【([^】]+)】", text)
        ):
            reasons[key] = text
    # 表达模型逐句校验后可能只剩一两条理由。缺失项按各自真实字段补齐，
    # 保证 QQ 三张卡都有可读说明，又不会借用别的菜谱事实。
    for item in results:
        metadata = item.get("metadata") or {}
        recipe_id = str(metadata.get("recipe_id") or item.get("id") or "").strip()
        if recipe_id and recipe_id not in reasons:
            reasons[recipe_id] = fallback["recipe_reasons"].get(
                recipe_id,
                _grounded_recipe_reason(metadata, lang),
            )
    # opening/strategy 独立兜底，避免模型只生成了理由时，最终卡片顶部变成
    # 光秃秃的标题。QQ 不再单独发送检索缓冲，这里必须始终有完整承接。
    if not text_fields["opening"]:
        text_fields["opening"] = fallback["opening"]
    if not text_fields["strategy"]:
        text_fields["strategy"] = fallback["strategy"]
    if not any((*text_fields.values(), *reasons.values())):
        return fallback
    return {
        **text_fields,
        "recipe_reasons": reasons,
        # 已校验的表达层可以决定自然措辞；缺失时才由确定性详情 CTA 兜底。
        "closing": text_fields["closing"] or build_device_cta(results, lang),
    }


@dataclass(frozen=True)
class SearchQueryPlan:
    original_query: str
    search_query: str
    reasoning: str = ""
    blocked_terms: tuple[str, ...] = ()


def _reasoning_max_chars() -> int:
    try:
        return max(80, min(int(os.getenv("RECIPE_REASONING_MAX_CHARS", "220")), 500))
    except (TypeError, ValueError):
        return 220


def _compact_reasoning(text: str, max_chars: int | None = None) -> str:
    """QQ 展示用短说明：保留 2-3 句，按字符预算兜底截断。"""
    max_chars = max_chars or _reasoning_max_chars()
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in ("。", "；", "，", ".", ";", ","):
        pos = cut.rfind(sep)
        if pos >= max_chars * 0.65:
            return cut[:pos + 1].rstrip()
    return cut.rstrip() + "..."


def plan_search_query(text: str, base_query: str | None = None, lang: str = "zh") -> SearchQueryPlan:
    """冻结意图层产出的查询，不在检索前按关键词重新解释用户。

    菜名、主料、口味、做法和场景的优先级已经由 ``SearchRequest`` 统一
    组织。这里若再根据某个具体场景词改写查询，就会产生案例分支，并可能
    覆盖用户明确给出的核心对象。
    """
    original = (text or "").strip()
    query = (base_query or original).strip()
    return SearchQueryPlan(original_query=query or original, search_query=query or original)


def _result_has_blocked_term(result: dict, blocked_terms: tuple[str, ...]) -> bool:
    if not blocked_terms:
        return False
    md = result.get("metadata") or {}
    fields = [
        md.get("name", ""),
        md.get("description", ""),
        " ".join(str(x) for x in _parse_tags(md.get("tags", []), lang="zh")),
        " ".join(str(x) for x in _parse_ingredients(md.get("ingredients", []))),
    ]
    haystack = " ".join(fields).lower()
    return any(term.lower() in haystack for term in blocked_terms)


def apply_search_plan(search_result: dict, plan: SearchQueryPlan, limit: int = 3) -> dict:
    """按查询规划过滤/截断结果，并把展示所需元信息挂到结果上。"""
    if not isinstance(search_result, dict):
        return search_result
    out = dict(search_result)
    results = list(search_result.get("results") or [])
    if plan.blocked_terms:
        filtered = [
            result
            for result in results
            if not _result_has_blocked_term(result, plan.blocked_terms)
        ]
        # blocked_terms 表示调用方要求不可出现的内容。过滤后为空也必须保留空集，
        # 不能为了“有答案”恢复已经命中的禁止项。
        results = filtered
    out["results"] = results[:limit]
    out["_original_query"] = plan.original_query
    out["_planned_query"] = plan.search_query
    if plan.reasoning:
        out["_search_reasoning"] = plan.reasoning
    return out


_ZH_TASTE_REASONING = (
    (("重口味", "香辣", "麻辣", "辣", "下饭"), "你想吃得有味道，我会优先保留香辣、咸鲜这类口感。但也会看具体场景，避免只按“越重越好”去推荐。"),
    (("清淡", "低脂", "减肥"), "你提到清淡或减脂，我会优先找油负担更低、食材更清爽的菜。口味可以有，但尽量不走高油、高糖、特别下饭的方向。"),
    (("快手", "简单", "省事"), "你想省时间，我会优先选步骤相对轻、出餐更快的菜。这样不用准备太多复杂配料，也更适合日常随手做。"),
    (("汤", "粥", "暖", "热乎"), "你想吃点热乎的，我会优先找汤水感更足、入口舒服的菜。整体会偏温和一点，适合晚上或胃口一般的时候吃。"),
)

_EN_TASTE_REASONING = (
    (("spicy", "bold", "strong", "heavy"), "You asked for bold flavors, so I kept the spicy and savory direction. I still avoid making the recommendation only about heavier food unless the scene fits."),
    (("light", "healthy", "low fat"), "You asked for something lighter, so I prioritized cleaner flavors and lower-burden dishes, while avoiding very oily or sugar-heavy options."),
    (("quick", "easy", "simple"), "You asked for something easy, so I prioritized recipes that should be quicker to make and need less prep."),
    (("soup", "warm", "hot"), "You asked for something warm, so I prioritized comforting dishes with a warmer, gentler profile."),
)


def _build_search_reasoning(query: str, recipes: list[dict], lang: str = "zh") -> str:
    """基于真实检索结果生成 QQ 友好的短推荐说明，不调用 LLM、不编造菜谱属性。"""
    q = (query or "").lower()
    rules = _EN_TASTE_REASONING if lang == "en" else _ZH_TASTE_REASONING
    for keywords, text in rules:
        if any(kw in q for kw in keywords):
            return _compact_reasoning(text)

    tag_counts: dict[str, int] = {}
    for recipe in recipes[:3]:
        for tag in recipe.get("tags") or []:
            tag = str(tag).strip()
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    common_tags = [tag for tag, _ in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]]

    if lang == "en":
        if common_tags:
            return _compact_reasoning(
                f"I matched your request against the recipe library and saw these options share tags like "
                f"{', '.join(common_tags)}. So I picked the closest few first; you can choose one to cook."
            )
        return _compact_reasoning("I matched your request against the recipe library and picked the closest options first.")
    if common_tags:
        return _compact_reasoning(
            f"我先按“{query}”去理解你的需求，再匹配菜谱库。"
            f"这几道共同偏向{'、'.join(common_tags)}，所以先给你放在前面；你可以从里面选一道继续做。"
        )
    return _compact_reasoning(f"我先按“{query}”去理解你的需求，再从菜谱库里挑 3 个最接近的选择。")


# ─── 直接食谱搜索 ────────────────────────────────────────────────────────

_NON_RECIPE_RECORD_TYPES = {"device_program"}


def _result_record_types(metadata: dict) -> set[str]:
    values = [metadata.get("record_type")]
    facets = metadata.get("facets")
    if isinstance(facets, dict):
        values.append(facets.get("record_type"))
    normalized: set[str] = set()
    for value in values:
        items = value if isinstance(value, (list, tuple, set)) else [value]
        normalized.update(
            str(item or "").strip().lower()
            for item in items
            if str(item or "").strip()
        )
    return normalized


def _filter_non_recipe_records(search_result: Optional[dict]) -> Optional[dict]:
    """食谱链路只保留菜谱记录，避免设备程序混入推荐候选。"""
    if not isinstance(search_result, dict):
        return search_result
    results = search_result.get("results")
    if not isinstance(results, list):
        return search_result
    kept = []
    dropped = 0
    for item in results:
        metadata = (
            item.get("metadata")
            if isinstance(item, dict) and isinstance(item.get("metadata"), dict)
            else {}
        )
        record_types = _result_record_types(metadata)
        if isinstance(item, dict) and item.get("record_type"):
            record_types.add(str(item["record_type"]).strip().lower())
        if record_types & _NON_RECIPE_RECORD_TYPES:
            dropped += 1
            continue
        kept.append(item)
    if not dropped:
        return search_result
    filtered = dict(search_result)
    filtered["results"] = kept
    filtered["count"] = len(kept)
    try:
        previous_dropped = int(
            search_result.get("_non_recipe_filtered_count") or 0
        )
    except (TypeError, ValueError):
        previous_dropped = 0
    filtered["_non_recipe_filtered_count"] = previous_dropped + dropped
    # 原 formatted 文本包含过滤前结果；公共编排器只保留新的结构化事实。
    filtered.pop("formatted", None)
    logger.warning(
        "食谱结果过滤了非菜谱记录: count=%s record_type=device_program",
        dropped,
    )
    return filtered


async def _run_search_subprocess(query: str, top_k: int = 3, lang: Optional[str] = None) -> Optional[dict]:
    """
    执行一次食谱检索（调度入口，函数名保留以兼容既有调用方）。

    默认走【进程内常驻服务】（route A）：免去逐请求子进程的启动 / 重导入 / 重连库开销。
    进程内服务不可用（依赖缺失 / 初始化失败 / 异常）时，自动回退子进程实现，保证可用性。
    设 RECIPE_SEARCH_INPROCESS=0 可强制只走子进程。
    lang：可选 zh|en，按原始用户输入显式指定检索语言（覆盖检索内部自动判定）。
    """
    started = time.monotonic()
    backend = "subprocess"
    fallback_used = False
    error_type = None
    result = None
    if os.getenv("RECIPE_SEARCH_INPROCESS", "1") != "0":
        backend = "inprocess"
        try:
            from app.agent.recipe_search_service import search as _inproc_search
            result = await _inproc_search(query, top_k=top_k, lang=lang)
            if result is None:
                fallback_used = True
                backend = "subprocess"
                logger.info("进程内检索不可用，本次回退子进程")
        except Exception as e:
            fallback_used = True
            backend = "subprocess"
            error_type = type(e).__name__
            logger.warning("进程内检索异常，回退子进程：error_type=%s", error_type)
    if result is None:
        result = await _run_search_subprocess_proc(query, top_k=top_k, lang=lang)
    result = _filter_non_recipe_records(result)
    success = bool(result and result.get("success"))
    record_tool_call(
        "recipe_search",
        duration_ms=(time.monotonic() - started) * 1000,
        success=success,
        kind="search",
        error_type=(None if success else error_type or "SEARCH_FAILED"),
        result_count=(
            len(result.get("results") or [])
            if isinstance(result, dict) else 0
        ),
        backend=backend,
        fallback=fallback_used,
    )
    return result


async def _run_search_subprocess_proc(query: str, top_k: int = 3, lang: Optional[str] = None) -> Optional[dict]:
    """
    通过子进程调用 recipe_search.py --json 获取结构化搜索结果（回退实现）。
    比通过 Agent Shell Backend 调用更快，因为跳过了 LLM 决策环节。
    """
    skill_dir = Path(__file__).resolve().parent / "skills" / "recipe-search"
    python_bin = skill_dir / ".venv" / "bin" / "python"
    args = [str(python_bin), "recipe_search.py", str(query), str(top_k), "--json"]
    if lang in ("zh", "en"):
        args.extend(["--lang", lang])
    # 子进程逐请求搜索并发下会炸；这里只是 route-A 不可用时的串行回退。
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(skill_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        elapsed = time.monotonic() - start

        if proc.returncode != 0:
            logger.warning(
                "搜索子进程失败 (%.1fs): rc=%s stderr_chars=%s",
                elapsed, proc.returncode, len(stderr),
            )
            return None

        result = json.loads(stdout.decode())
        logger.info("搜索子进程完成 (%.1fs), query_chars=%s", elapsed, len(str(query or "")))
        return result

    except asyncio.TimeoutError:
        logger.warning("搜索子进程超时 (30s), query_chars=%s", len(str(query or "")))
        record_timeout("recipe_search", error_type="TimeoutError")
        return None
    except json.JSONDecodeError as e:
        logger.warning("搜索结果 JSON 解析失败: position=%s", e.pos)
        return None
    except Exception as e:
        logger.error("搜索子进程异常: error_type=%s", type(e).__name__)
        return None


async def _format_search_response(query: str, search_result: dict, lang: str = "zh") -> str:
    """
    将搜索结果直接格式化为 QQ Bot JSON，无需 LLM 参与。
    lang：固定文案（message）按此语言输出；菜谱数据本身已在检索层按语言过滤。
    """
    display_query = str(query or "").strip()
    if lang == "en" and _has_cjk(display_query):
        display_query = "recipe search"

    def localized_query(value) -> str:
        text = str(value or "").strip()
        if lang == "en" and _has_cjk(text):
            return display_query
        return text or display_query

    results = (search_result.get("results", []) or [])[:10]
    if lang == "en":
        results = [
            result for result in results
            if _has_latin(str((result.get("metadata") or {}).get("name") or ""))
            and not _has_cjk(str((result.get("metadata") or {}).get("name") or ""))
        ]

    if not results:
        return json.dumps({
            "type": "recipe_search",
            "intent": "search",
            "lang": lang,
            "data": {"query": display_query, "total": 0, "recipes": [], "lang": lang},
            "message": _empty_search_message(display_query, search_result, lang)
        }, ensure_ascii=False)

    # 最终回复只信任当前检索结果。旧缓存或表达模型里的自由菜名、步骤和搭配
    # 一律不会进入这个出口。
    recommendation = _current_result_recommendation(search_result, results, lang)
    recipe_reasons = recommendation.get("recipe_reasons") or {}
    recipes = []
    for r in results:
        metadata = r.get("metadata", {})
        score = r.get("score", 0)
        # 最低显示 50%
        score_pct = max(int(score * 100), 50)

        recipe_id = str(metadata.get("recipe_id", r.get("id", "")))
        recipes.append({
            "id": recipe_id,
            "name": metadata.get("name", "Unknown recipe" if lang == "en" else "未知菜品"),
            "image": metadata.get("image_url", ""),
            "ingredients": _parse_ingredients(metadata.get("ingredients", []), lang=lang),
            "seasonings": _parse_ingredients(metadata.get("seasonings", []), lang=lang),
            "tags": _parse_tags(metadata.get("tags", []), lang=lang),
            "similarity": round(score, 2),
            "description": (
                "" if lang == "en" and _has_cjk(str(metadata.get("description") or ""))
                else metadata.get("description", "")
            ),
            "recipe_detail": metadata.get("recipe_detail"),
            "device_code": metadata.get("device_code"),
            "recommendation_reason": recipe_reasons.get(recipe_id, ""),
        })

    response = {
        "type": "recipe_search",
        "intent": "recommend" if search_result.get("_interaction_mode") == "recommend" else "search",
        "lang": lang,
        "data": {
            "query": display_query,
            "original_query": localized_query(search_result.get("_original_query")),
            "planned_query": localized_query(search_result.get("_planned_query")),
            "total": len(recipes),
            "recipes": recipes,
            "reasoning": "",
            "opening": " ".join(
                value for value in (
                    str(recommendation.get("opening") or "").strip(),
                    str(search_result.get("_constraint_notice") or "").strip(),
                ) if value
            ),
            "strategy": recommendation.get("strategy", ""),
            "closing": recommendation.get("closing", ""),
            "validation": dict(search_result.get("_validation") or {}),
            "lang": lang,
        },
        "message": " ".join(
            text for text in (
                str(recommendation.get("opening") or "").strip(),
                str(recommendation.get("strategy") or "").strip(),
                str(search_result.get("_constraint_notice") or "").strip(),
            ) if text
        )
    }
    return json.dumps(response, ensure_ascii=False)


def _menu_plan_summary(menu: dict, lang: str) -> str:
    requested = menu.get("requested") or {}
    fulfilled = menu.get("fulfilled") or {}
    req_dish, req_soup = int(requested.get("dish") or 0), int(requested.get("soup") or 0)
    got_dish, got_soup = int(fulfilled.get("dish") or 0), int(fulfilled.get("soup") or 0)
    if lang == "en":
        if menu.get("complete"):
            return f"This table is set with {req_dish} dishes and {req_soup} soup."
        missing_dish = max(0, req_dish - got_dish)
        missing_soup = max(0, req_soup - got_soup)
        gaps = []
        if missing_dish:
            gaps.append(f"{missing_dish} dish" + ("es" if missing_dish != 1 else ""))
        if missing_soup:
            gaps.append(f"{missing_soup} soup" + ("s" if missing_soup != 1 else ""))
        return (
            f"I can safely set out {got_dish} dishes and {got_soup} soups for now; "
            f"{' and '.join(gaps) or 'one menu slot'} is still missing, so I have not filled it "
            "with an uncertain recipe."
        )
    if menu.get("complete"):
        return f"这桌按{req_dish}菜{req_soup}汤配齐了。"
    missing_dish = max(0, req_dish - got_dish)
    missing_soup = max(0, req_soup - got_soup)
    gaps = []
    if missing_dish:
        gaps.append(f"{missing_dish}道菜")
    if missing_soup:
        gaps.append(f"{missing_soup}道汤")
    return (
        f"这轮先稳妥配到{got_dish}道菜和{got_soup}道汤，还差{'、'.join(gaps) or '一个菜位'}；"
        "没有拿不确定的菜来凑数。"
    )


def format_menu_plan_response(query: str, search_result: dict, lang: str = "zh") -> str:
    """菜单规划 → QQ/IM JSON；输出菜名严格来自分槽检索结果。"""
    results = list(search_result.get("results") or [])
    if lang == "en":
        results = [item for item in results if _has_latin(str((item.get("metadata") or {}).get("name") or ""))]
    menu = dict(search_result.get("_menu_plan") or {})
    recommendation = _current_result_recommendation(search_result, results, lang)
    narrative_summary = " ".join(
        value for value in (
            str(recommendation.get("opening") or "").strip(),
            str(recommendation.get("strategy") or "").strip(),
        ) if value
    )
    summary = (
        narrative_summary or _menu_plan_summary(menu, lang)
        if menu.get("complete")
        else " ".join(
            value for value in (_menu_plan_summary(menu, lang), narrative_summary)
            if value
        )
    )
    notice = str(search_result.get("_constraint_notice") or "").strip()
    if notice:
        summary = f"{summary} {notice}".strip()
    recipe_reasons = recommendation.get("recipe_reasons") or {}
    recipes = []
    for result in results:
        metadata = result.get("metadata") or {}
        recipe_id = str(metadata.get("recipe_id") or result.get("id") or "")
        recipes.append({
            "id": recipe_id,
            "name": metadata.get("name", ""),
            "menu_role": result.get("menu_role") or "dish",
            "image": metadata.get("image_url", ""),
            "ingredients": _parse_ingredients(metadata.get("ingredients", []), lang=lang),
            "seasonings": _parse_ingredients(metadata.get("seasonings", []), lang=lang),
            "tags": _parse_tags(metadata.get("tags", []), lang=lang),
            "description": (
                "" if lang == "en" and _has_cjk(str(metadata.get("description") or ""))
                else metadata.get("description", "")
            ),
            "recipe_detail": metadata.get("recipe_detail"),
            "device_code": metadata.get("device_code"),
            "recommendation_reason": recipe_reasons.get(recipe_id) or _grounded_recipe_reason(metadata, lang),
        })
    closing = str(recommendation.get("closing") or "").strip() or build_device_cta(results, lang)
    return json.dumps({
        "type": "menu_plan",
        "intent": "menu_plan",
        "lang": lang,
        "data": {
            "query": str(query or "").strip(),
            "requested": menu.get("requested") or {},
            "fulfilled": menu.get("fulfilled") or {},
            "missing": menu.get("missing") or {},
            "complete": bool(menu.get("complete")),
            "validation_errors": list(menu.get("errors") or []),
            "recipes": recipes,
            "closing": closing,
            "constraint_notice": notice,
            "lang": lang,
        },
        "message": summary,
    }, ensure_ascii=False)


def _markdown_recipe_section(
    result: dict,
    index: int,
    lang: str,
    *,
    reason: str = "",
    role_label: str = "",
) -> list[str]:
    """把一条真实 Milvus 结果渲染成可扫读的 Markdown 小节。"""
    metadata = result.get("metadata") or {}
    name = str(metadata.get("name") or ("Unknown recipe" if lang == "en" else "未知菜品")).strip()
    image = str(metadata.get("image_url") or "").strip()
    ingredients = _parse_ingredients(metadata.get("ingredients", []), lang=lang)
    tags = _parse_tags(metadata.get("tags", []), lang=lang)
    labels = {
        "ingredients": "Main ingredients" if lang == "en" else "主要食材",
        "tags": "Tags" if lang == "en" else "标签",
        "reason": "Why it fits" if lang == "en" else "推荐理由",
        "role": "Menu role" if lang == "en" else "菜单定位",
    }

    lines = [f"### {index}. {name}", ""]
    if image:
        # 图片紧跟菜名，符合统一卡片字段顺序：名称 → 图片 → 食材 → 标签 → 理由。
        safe_alt = re.sub(r"[\[\]()]", " ", name)
        lines.extend([f"![{safe_alt}]({image})", ""])
    if role_label:
        lines.append(f"- **🍽️ {labels['role']}**：{role_label}" if lang != "en"
                     else f"- **🍽️ {labels['role']}:** {role_label}")
    if reason:
        lines.extend([
            f"> **{labels['reason']}**：{reason}" if lang != "en"
            else f"> **{labels['reason']}:** {reason}",
            "",
        ])
    if ingredients:
        value = "、".join(ingredients[:8]) if lang != "en" else ", ".join(ingredients[:8])
        lines.append(f"- **🥘 {labels['ingredients']}**：{value}" if lang != "en"
                     else f"- **🥘 {labels['ingredients']}:** {value}")
    if tags:
        value = " · ".join(tags[:6])
        lines.append(f"- **🏷️ {labels['tags']}**：{value}" if lang != "en"
                     else f"- **🏷️ {labels['tags']}:** {value}")
    detail_reference = _recipe_detail_reference(metadata, lang)
    if detail_reference:
        lines.append(
            f"- **⏱️ Cooking reference:** {detail_reference}"
            if lang == "en" else f"- **⏱️ 做饭参考**：{detail_reference}"
        )
    lines.extend([
        "",
        (
            f'👉 Reply **"details {index}"** to open the full recipe.'
            if lang == "en"
            else f"👉 回复 **「详情 {index}」** 查看完整菜谱。"
        ),
        "",
    ])
    return lines


def format_menu_plan_markdown(query: str, search_result: dict, lang: str = "zh") -> str:
    """菜单规划 → Web Markdown；不使用自由模型叙述。"""
    results = list(search_result.get("results") or [])
    menu = dict(search_result.get("_menu_plan") or {})
    recommendation = _current_result_recommendation(search_result, results, lang)
    recipe_reasons = recommendation.get("recipe_reasons") or {}
    narrative_summary = " ".join(
        value for value in (
            str(recommendation.get("opening") or "").strip(),
            str(recommendation.get("strategy") or "").strip(),
        ) if value
    )
    summary = (
        narrative_summary or _menu_plan_summary(menu, lang)
        if menu.get("complete")
        else " ".join(
            value for value in (_menu_plan_summary(menu, lang), narrative_summary)
            if value
        )
    )
    notice = str(search_result.get("_constraint_notice") or "").strip()
    if notice:
        summary = f"{summary}\n\n> **{'Note' if lang == 'en' else '提醒'}**：{notice}"
    title = "## 🍽️ Menu plan" if lang == "en" else "## 🍽️ 推荐菜单"
    parts = [summary, "", title, ""]
    for index, result in enumerate(results, 1):
        metadata = result.get("metadata") or {}
        role = result.get("menu_role") or "dish"
        if lang == "en":
            label = "Soup" if role == "soup" else ("Guest preference dish" if role == "scoped_dish" else "Dish")
        else:
            label = "汤" if role == "soup" else ("照顾局部偏好的菜" if role == "scoped_dish" else "菜")
        recipe_id = str(metadata.get("recipe_id") or result.get("id") or "")
        reason = recipe_reasons.get(recipe_id) or _grounded_recipe_reason(metadata, lang)
        parts.extend(_markdown_recipe_section(
            result,
            index,
            lang,
            reason=reason,
            role_label=label,
        ))
    closing = str(recommendation.get("closing") or "").strip() or build_device_cta(results, lang)
    if closing:
        parts.extend([
            f"## {'How to choose' if lang == 'en' else '怎么选'}",
            "",
            closing,
        ])
    return "\n".join(parts)


def format_search_markdown(query: str, search_result: dict, lang: str = "zh") -> str:
    """检索结果 → Web Markdown；兼容入口统一委托给自然卡片 Renderer。"""
    results = list(search_result.get("results", []) or [])
    if lang == "en":
        results = [
            result for result in results
            if _has_latin(str((result.get("metadata") or {}).get("name") or ""))
            and not _has_cjk(str((result.get("metadata") or {}).get("name") or ""))
        ]
    if not results:
        title = "## No matching recipes" if lang == "en" else "## 没找到合适的菜谱"
        return f"{title}\n\n{_empty_search_message(query, search_result, lang)}"

    # 旧调用方仍直接使用本函数，但 Web 主链已经由 WebResponseRenderer 负责
    # Markdown 表达。这里仅把真实检索结果适配成同一 Envelope，避免继续维护
    # 第二套 ``##/###/字段标签`` 格式。
    from app.orchestrator.turn.response_renderer import WebResponseRenderer
    from app.orchestrator.turn.runtime_models import ResponseEnvelope

    recommendation = _current_result_recommendation(search_result, results, lang)
    recipe_reasons = recommendation.get("recipe_reasons") or {}
    notice = str(search_result.get("_constraint_notice") or "").strip()
    recipes = []
    for r in results:
        md = r.get("metadata") or {}
        recipe_id = str(md.get("recipe_id") or r.get("id") or "")
        recipes.append({
            "id": recipe_id,
            "name": md.get(
                "name",
                "Unknown recipe" if lang == "en" else "未知菜品",
            ),
            "image": md.get("image_url", ""),
            "ingredients": _parse_ingredients(md.get("ingredients", []), lang=lang),
            "tags": _parse_tags(md.get("tags", []), lang=lang),
            "device_code": md.get("device_code"),
            "recommendation_reason": str(
                recipe_reasons.get(recipe_id) or ""
            ).strip(),
        })

    message = " ".join(
        value
        for value in (
            str(recommendation.get("opening") or "").strip(),
            str(recommendation.get("strategy") or "").strip(),
            notice,
        )
        if value
    )
    envelope = ResponseEnvelope(
        response_type="recipe_search",
        intent=(
            "recommend"
            if search_result.get("_interaction_mode") == "recommend"
            else "search"
        ),
        lang=lang,
        message=message,
        data={
            "recipes": recipes,
            "closing": str(recommendation.get("closing") or "").strip(),
        },
        handled_by="recipe_handler",
    )
    return WebResponseRenderer().render(envelope)


# ─── 问候回复 ────────────────────────────────────────────────────────────
