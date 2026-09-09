"""图片识别后的文字追问处理。

本模块属于 Turn 应用层：它只在统一任务状态工作区内读取/更新图片待办，
并把用户确认后的食材交给真实菜谱检索。通道入口不再维护独立状态作用域。
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

from app.agent.fast_path import _format_search_response, _run_search_subprocess
from app.orchestrator.recommendation_response import (
    generate_recommendation_narrative,
)
from app.ports.dialogue_state import (
    clear_pending_action,
    get_pending_action,
    remember_candidates,
)


SearchProgress = Callable[[str, str], Awaitable[None]]

_IMAGE_SEARCH_CONFIRM_ZH = {
    "就这些",
    "没有了",
    "没了",
    "没有其他了",
    "没有别的了",
    "开始推荐",
    "开始搜索",
    "按这些搜",
    "可以了",
}
_IMAGE_SEARCH_CONFIRM_EN = {
    "that's all",
    "that is all",
    "nothing else",
    "search now",
    "start searching",
    "use these",
    "go ahead",
}
_IMAGE_SEARCH_CANCEL_ZH = {"取消", "算了", "先不搜了", "不搜了"}
_IMAGE_SEARCH_CANCEL_EN = {"cancel", "never mind", "stop"}


def image_ingredient_confirmation_text(
    ingredients: list[str],
    lang: str = "zh",
) -> str:
    values = [str(item).strip() for item in ingredients if str(item).strip()][:8]
    if lang == "en":
        lines = [
            "## Ingredients I can see",
            "",
            *(f"- {item}" for item in values),
            "",
            "Some ingredients may be outside the frame. Is there anything else you want to use?",
            "",
            '- To add something, reply: "add tofu and eggs"',
            '- If that is everything, reply: "that’s all"',
        ]
    else:
        lines = [
            "## 我从图片里看见的食材",
            "",
            *(f"- {item}" for item in values),
            "",
            "有些食材可能没拍进画面，还有其他食材想一起用吗？",
            "",
            "- 有的话：直接回复「再加豆腐和鸡蛋」",
            "- 没有的话：回复「就这些」",
        ]
    return "\n".join(lines)


def parse_image_ingredient_supplements(text: str, lang: str) -> list[str]:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" ，,。.!！?？")
    if not value:
        return []
    if lang == "en":
        match = re.match(r"^(?:also\s+)?add\s+(.+)$", value, flags=re.IGNORECASE)
        if not match:
            return []
        parts = re.split(
            r"\s*(?:,|;|\band\b|\+)\s*",
            match.group(1),
            flags=re.IGNORECASE,
        )
    else:
        match = re.match(
            r"^(?:再加|加上|补充|还有|另外还有|还剩)\s*[：:]?\s*(.+)$",
            value,
        )
        if not match:
            return []
        parts = re.split(
            r"[、,，；;+\s]*(?:和|跟|以及)[、,，；;+\s]*|[、,，；;+\s]+",
            match.group(1),
        )

    result: list[str] = []
    for item in parts:
        ingredient = re.sub(
            r"^(?:一点|一些|一份|一块|一盒|一个|几个|两根|两颗)\s*",
            "",
            str(item or "").strip(),
        )
        if lang == "en":
            ingredient = re.sub(
                r"^(?:some|a|an|one|a little|a bit of)\s+",
                "",
                ingredient,
                flags=re.IGNORECASE,
            ).strip()
        if ingredient and ingredient not in result:
            result.append(ingredient)
        if len(result) >= 8:
            break
    return result


def merge_confirmed_ingredients(
    recognized: list[str],
    supplements: list[str],
    limit: int = 8,
) -> list[str]:
    """显式补充项优先于视觉候选，去重后限制检索长度。"""
    values = [
        str(item).strip()
        for item in [*(supplements or []), *(recognized or [])]
        if str(item).strip()
    ]
    return list(dict.fromkeys(values))[: max(1, int(limit))]


async def handle_image_ingredient_followup(
    thread_id: str,
    text: str,
    *,
    on_search_start: SearchProgress | None = None,
) -> str | None:
    """承接图片待办；调用方必须已处于统一 Turn/UoW。"""
    action = get_pending_action(thread_id)
    if action and action.get("kind") == "image_inventory":
        # 图片处理器已经基于本批图片完成检索。用户开始文字对话后结束图片批次，
        # 让统一文本路由自然承接现有 SearchRequest/候选。
        clear_pending_action(thread_id)
        return None
    if not action or action.get("kind") != "confirm_image_ingredients":
        return None

    lang = action.get("lang") or "zh"
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    cancel_values = (
        _IMAGE_SEARCH_CANCEL_EN if lang == "en" else _IMAGE_SEARCH_CANCEL_ZH
    )
    if normalized in cancel_values:
        clear_pending_action(thread_id)
        return (
            "Okay, I’ve put that list aside. Send another photo or tell me what you want whenever you’re ready."
            if lang == "en"
            else "好，那我先把这份清单放下。想继续时再发张图，或者直接告诉我想吃什么就行。"
        )

    payload = dict(action.get("payload") or {})
    ingredients = [
        str(item).strip()
        for item in payload.get("ingredients") or []
        if str(item).strip()
    ][:8]
    supplements = parse_image_ingredient_supplements(text, lang)
    confirm_values = (
        _IMAGE_SEARCH_CONFIRM_EN if lang == "en" else _IMAGE_SEARCH_CONFIRM_ZH
    )
    if not supplements and normalized not in confirm_values:
        return image_ingredient_confirmation_text(ingredients, lang)
    if supplements:
        ingredients = merge_confirmed_ingredients(ingredients, supplements)
    if not ingredients:
        clear_pending_action(thread_id)
        return (
            "I still do not have a usable ingredient list. Please send a clearer photo."
            if lang == "en"
            else "现在还没有可用的食材清单，麻烦换一张更清楚的照片。"
        )

    clear_pending_action(thread_id)
    query = " ".join([*ingredients[:8], "recipe" if lang == "en" else "家常菜"])
    if on_search_start is not None:
        await on_search_start(query, lang)
    search_result = await _run_search_subprocess(query, top_k=9, lang=lang)
    if not (
        search_result
        and search_result.get("success")
        and search_result.get("results")
    ):
        shown = ", ".join(ingredients) if lang == "en" else "、".join(ingredients)
        return (
            f"I confirmed {shown}, but I could not find a suitable recipe this time. Try naming the one ingredient you most want to use."
            if lang == "en"
            else f"食材清单已经确认：{shown}。这一轮没搜到合适菜谱，你可以再告诉我最想优先用哪一种。"
        )

    search_result = dict(search_result)
    search_result["results"] = list(search_result.get("results") or [])[:3]
    original_question = (
        f"I have {', '.join(ingredients)}. Recommend recipes I can make."
        if lang == "en"
        else f"我有{'、'.join(ingredients)}，推荐几道能做的菜。"
    )
    narrative = await generate_recommendation_narrative(
        original_question=original_question,
        search_query=query,
        search_result=search_result,
        lang=lang,
    )
    search_result["_recommendation"] = narrative.to_dict()
    search_result["_recommendation_source"] = "grounded_v2"
    search_result["_interaction_mode"] = "recommend"
    remember_candidates(thread_id, search_result, lang=lang)

    response = json.loads(await _format_search_response(query, search_result, lang=lang))
    confirmed = (
        f"Got it — the final ingredient list is: {', '.join(ingredients)}."
        if lang == "en"
        else f"好，最终食材清单是：{'、'.join(ingredients)}。"
    )
    response["message"] = " ".join(
        value for value in (confirmed, str(response.get("message") or "").strip()) if value
    )
    return json.dumps(response, ensure_ascii=False)


__all__ = [
    "handle_image_ingredient_followup",
    "image_ingredient_confirmation_text",
    "merge_confirmed_ingredients",
    "parse_image_ingredient_supplements",
]
