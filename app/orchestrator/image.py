"""
图片入口 —— Qwen-VL 识别 -> 真实文本检索（复用与文字同一条引擎）。

orchestrator 层：语言无关、无通道耦合（通道收图/回发归通道层）。

设计要点：
  - VL 的 confidence 不可靠，不能用来自动执行；这里只做检索 query。
  - 支持多张图片一起识别，尤其是冰箱/台面食材；生食材不再硬猜成菜名。
  - search_result 只来自真实 Milvus 检索；VL 仅产出 query 候选，不编造菜谱。
"""
from __future__ import annotations

import time

from app.agent.fast_path import _run_search_subprocess
from app.orchestrator.search_request import SearchRequest


_GENERIC_HINTS = {
    "帮我看看", "看看", "这是什么", "图片", "照片", "识别", "菜谱", "做什么", "能做什么",
    "help", "image", "photo", "recipe", "what can i cook", "what is this",
}

_GENERIC_HINT_SUBSTRINGS = (
    "帮我看看", "能做什么", "可以做什么", "这些食材", "这组图片", "这几张", "有什么菜", "推荐菜谱",
    "what can i cook", "what can be cooked", "these ingredients", "this image", "these photos",
)

_MEANINGFUL_HINT_KEYWORDS = (
    "清淡", "低脂", "减肥", "低卡", "少油", "少盐", "辣", "不辣", "快手", "简单", "晚餐", "早餐",
    "夜宵", "下饭", "汤", "粥", "素", "儿童", "老人", "控糖", "高蛋白",
    "light", "healthy", "low fat", "low calorie", "spicy", "quick", "easy", "dinner", "breakfast",
    "soup", "vegetarian", "high protein",
)


def _ingredient_names(recognized: dict, limit: int = 8) -> list[str]:
    """从新/旧视觉结构中提取食材名，去重保序。"""
    out = []
    seen = set()
    for item in (recognized or {}).get("ingredients") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("ingredient") or "").strip()
        else:
            name = str(item or "").strip()
        if not name or name in seen:
            continue
        out.append(name)
        seen.add(name)
        if len(out) >= limit:
            break
    return out


def _clean_user_hint(user_hint: str | None) -> str:
    hint = " ".join(str(user_hint or "").strip().split())
    if not hint:
        return ""
    low = hint.lower()
    if low in _GENERIC_HINTS or hint in _GENERIC_HINTS:
        return ""
    if any(keyword in low or keyword in hint for keyword in _MEANINGFUL_HINT_KEYWORDS):
        return hint[:40]
    if any(marker in low or marker in hint for marker in _GENERIC_HINT_SUBSTRINGS):
        return ""
    # 很短的 caption 通常是口味/餐型约束；长句容易污染检索 query。
    return hint[:12] if len(hint) <= 12 else ""


def _build_image_search_query(recognized: dict, lang: str = "zh", user_hint: str | None = None) -> str:
    """视觉识别结果 -> 菜谱检索词。成品菜用菜名；冰箱食材用核心食材组合。"""
    rec = recognized or {}
    scene = str(rec.get("scene_type") or "").strip().lower()
    query = str(rec.get("search_query") or "").strip()
    dish = str(rec.get("dish_name") or "").strip()
    ingredients = _ingredient_names(rec, limit=5)

    if not query:
        if scene == "dish" and dish:
            query = dish
        elif ingredients:
            suffix = "家常菜" if lang != "en" else "recipe"
            query = " ".join([*ingredients, suffix])
        elif dish:
            query = dish

    hint = _clean_user_hint(user_hint)
    if query and hint and hint not in query:
        query = f"{query} {hint}"
    return query.strip()


def _recognized_food_summary(recognized: dict, lang: str = "zh") -> str:
    """给通道层展示的短识别摘要。"""
    rec = recognized or {}
    ingredients = _ingredient_names(rec, limit=8)
    if ingredients:
        joiner = ", " if lang == "en" else "、"
        return joiner.join(ingredients)
    dish = str(rec.get("dish_name") or "").strip()
    if dish:
        return dish
    return ""


def merge_image_inventory(
    recognized: dict,
    *,
    previous_action: dict | None = None,
    image_count: int = 1,
    user_hint: str | None = None,
    max_age_seconds: int = 90,
    limit: int = 12,
) -> tuple[dict, dict]:
    """把短时间内连续发来的食材图片合并成中性的“可用食材”上下文。

    只合并 ingredients/mixed 场景；成品菜图片天然是一个新识别任务。返回的
    payload 可直接写入 pending_action，在 Redis task_state 中跨 worker 恢复。
    """
    current = dict(recognized or {})
    scene = str(current.get("scene_type") or "").strip().lower()
    previous_payload: dict = {}
    previous_is_fresh = False
    if (
        isinstance(previous_action, dict)
        and previous_action.get("kind") == "image_inventory"
        and scene in {"ingredients", "mixed"}
    ):
        try:
            previous_is_fresh = (
                time.time() - float(previous_action.get("ts") or 0)
                <= max(1, int(max_age_seconds))
            )
        except (TypeError, ValueError):
            previous_is_fresh = False
        candidate = previous_action.get("payload")
        if previous_is_fresh and isinstance(candidate, dict):
            previous_scene = str(candidate.get("scene_type") or "").strip().lower()
            if previous_scene in {"ingredients", "mixed"}:
                previous_payload = dict(candidate)

    merged: list[dict] = []
    positions: dict[str, int] = {}

    def add(items) -> None:
        for raw in items or []:
            if isinstance(raw, dict):
                item = {
                    "name": str(raw.get("name") or raw.get("ingredient") or "").strip(),
                    "confidence": raw.get("confidence", 0.0),
                    "state": str(raw.get("state") or "unknown").strip() or "unknown",
                }
            else:
                item = {
                    "name": str(raw or "").strip(),
                    "confidence": 0.0,
                    "state": "unknown",
                }
            name = item["name"]
            if not name:
                continue
            if name in positions:
                index = positions[name]
                try:
                    old_confidence = float(merged[index].get("confidence") or 0.0)
                    new_confidence = float(item.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    old_confidence = new_confidence = 0.0
                if new_confidence > old_confidence:
                    merged[index] = item
                continue
            positions[name] = len(merged)
            merged.append(item)
            if len(merged) >= max(1, int(limit)):
                return

    add(previous_payload.get("ingredients"))
    add(current.get("ingredients"))
    if merged:
        current["ingredients"] = merged
        current["is_food"] = True
        if scene != "mixed" or not str(current.get("dish_name") or "").strip():
            current["scene_type"] = "ingredients"
            current["dish_name"] = ""
        # 旧 search_query 只覆盖当前一批图片，合并后必须按完整食材重建。
        current["search_query"] = ""

    hints = [
        str(item).strip()
        for item in previous_payload.get("user_hints") or []
        if str(item).strip()
    ]
    hint = str(user_hint or "").strip()
    if hint and hint not in hints:
        hints.append(hint)
    total_images = (
        int(previous_payload.get("image_count") or 0)
        + max(1, int(image_count or 1))
    )
    payload = {
        "scene_type": str(current.get("scene_type") or scene or "ingredients"),
        "dish_name": str(current.get("dish_name") or "").strip(),
        "dish_names": [
            str(item).strip()
            for item in current.get("dish_names") or []
            if str(item).strip()
        ][:3],
        "ingredients": merged,
        "image_count": total_images,
        "user_hints": hints[-4:],
    }
    return current, payload


def build_image_search_request(
    recognized: dict,
    *,
    lang: str = "zh",
    user_hint: str | None = None,
) -> SearchRequest:
    """把视觉事实和图片文字收敛为统一 SearchRequest，不再走图片专用 query 协议。"""
    rec = recognized or {}
    caption = str(user_hint or "").strip()
    request = SearchRequest.from_intent_raw(
        {"c": "recipe_recommend"},
        original_text=caption,
        keywords=[],
    )
    data = request.model_dump()
    scene = str(rec.get("scene_type") or "").strip().lower()
    dish = str(rec.get("dish_name") or "").strip()
    ingredients = _ingredient_names(rec, limit=12)
    model_query = str(rec.get("search_query") or "").strip()

    visual_request_text = (
        (
            f"图片里识别到的食材有：{'、'.join(ingredients)}，请按这些食材推荐菜谱。"
            if lang != "en" else
            f"The image recognition found: {', '.join(ingredients)}. Recommend recipes using them."
        )
        if ingredients else
        (
            f"图片里识别到的成品菜可能是：{dish}，请推荐相关菜谱。"
            if lang != "en" else
            f"The prepared dish in the image may be {dish}. Recommend related recipes."
        )
    )
    if caption:
        default_task = (
            "Please recommend dishes from the image content."
            if lang == "en" else
            "请按图片内容推荐几道菜。"
        )
        data["original_text"] = f"{caption}; {default_task}" if lang == "en" else (
            f"{caption}；{default_task}"
        )
    else:
        data["original_text"] = visual_request_text
    data["ingredients"] = ingredients
    data["dishes"] = [dish] if dish and scene == "dish" else []
    data["query"] = model_query
    data["task_operation"] = "fuzzy_recommend"
    data["result_limit"] = 3
    data["explicit_result_limit"] = False
    data["context_note"] = {
        "source": "image_recognition",
        "scene_type": scene,
    }
    return SearchRequest(**data)


def build_image_agent_context(
    recognized: dict,
    *,
    image_count: int,
    user_hint: str | None = None,
) -> dict:
    """生成可交给受控 Deep Agent 的视觉事实；不包含媒体 URL 或本地路径。"""
    rec = recognized or {}
    return {
        "source": "image_recognition",
        "scene_type": str(rec.get("scene_type") or "").strip().lower(),
        "image_count": max(1, int(image_count or 1)),
        "dish_name": str(rec.get("dish_name") or "").strip(),
        "ingredients": [
            {
                "name": str(item.get("name") or item.get("ingredient") or "").strip(),
                "confidence": item.get("confidence", 0.0),
                "state": str(item.get("state") or "unknown").strip() or "unknown",
            }
            if isinstance(item, dict)
            else {"name": str(item or "").strip(), "confidence": 0.0, "state": "unknown"}
            for item in rec.get("ingredients") or []
            if (
                str(item.get("name") or item.get("ingredient") or "").strip()
                if isinstance(item, dict)
                else str(item or "").strip()
            )
        ][:12],
        "user_caption": str(user_hint or "").strip(),
        "recognition_confidence": rec.get("confidence", 0.0),
    }


def image_recommendation_question(context: dict, lang: str = "zh") -> str:
    """把“用户发图”表达为 Agent 当前问题，同时明确视觉结果仍是待纠正假设。"""
    image_count = max(1, int((context or {}).get("image_count") or 1))
    ingredients = [
        str(item.get("name") or "").strip()
        for item in (context or {}).get("ingredients") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    dish = str((context or {}).get("dish_name") or "").strip()
    caption = str((context or {}).get("user_caption") or "").strip()
    if lang == "en":
        visual = (
            f"Across {image_count} image(s), visual recognition suggests these ingredients: "
            f"{', '.join(ingredients)}."
            if ingredients else
            f"Visual recognition suggests the prepared dish may be {dish}."
        )
        return " ".join(part for part in (
            f'The user added this caption: "{caption}".' if caption else "",
            visual,
            "Recommend relevant verified recipes and acknowledge that the visual reading may need correction.",
        ) if part)
    visual = (
        f"这{image_count}张图片里，视觉识别到的食材包括：{'、'.join(ingredients)}。"
        if ingredients else
        f"图片里的成品菜可能是【{dish}】。"
    )
    return "".join(part for part in (
        f"用户随图片说：“{caption}”。" if caption else "",
        visual,
        "请基于真实菜谱给出相关推荐，并自然说明图片识别可能需要用户纠正。",
    ) if part)


async def search_recipes_by_images(images: list[dict], lang: str = "zh", top_k: int = 3,
                                   user_hint: str | None = None) -> dict:
    """多图 -> VL 识别食材/菜名 -> 真实检索。

    images：[{"image_url": "..."}] 或 [{"image_base64": "...", "image_mime": "image/jpeg"}]
    返回 {recognized, search_query, search_result}。
    """
    from app.agent.vision import recognize_food_images

    rec = await recognize_food_images(images, lang=lang)
    query = _build_image_search_query(rec, lang=lang, user_hint=user_hint)
    if not rec.get("is_food") or not query:
        return {"recognized": rec, "search_query": query, "search_result": None}
    search_result = await _run_search_subprocess(query, top_k=top_k, lang=lang)
    return {"recognized": rec, "search_query": query, "search_result": search_result}


async def recognize_ingredients_by_images(
    images: list[dict],
    lang: str = "zh",
) -> dict:
    """只做图片理解，不提前搜索；供“识别→用户补充→检索”多轮流程使用。"""
    from app.agent.vision import recognize_food_images

    return await recognize_food_images(images, lang=lang)


async def search_recipes_by_image(image_url: str | None = None, image_base64: str | None = None,
                                  image_mime: str = "image/jpeg", lang: str = "zh", top_k: int = 3,
                                  user_hint: str | None = None) -> dict:
    """兼容旧单图接口。"""
    image = {"image_url": image_url} if image_url else {
        "image_base64": image_base64,
        "image_mime": image_mime,
    }
    return await search_recipes_by_images([image], lang=lang, top_k=top_k, user_hint=user_hint)
