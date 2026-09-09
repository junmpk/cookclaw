"""外部菜谱详情规范化与用户展示。

远端字段保持真实来源，不使用 LLM 补写缺失食材、步骤、时长或设备参数。
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


_INGREDIENT_GROUPS = (
    ("mainMaterials", "main"),
    ("recipeAccessories", "accessory"),
    ("recipeSeasoning", "seasoning"),
    ("other", "other"),
)

_PRIVATE_RESPONSE_FIELDS = {
    # 这两个字段属于外部服务账号状态，不是公共菜谱事实。
    "isCollect",
    "isPurchase",
}


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    return []


def _tag_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags = []
    for item in value:
        if isinstance(item, dict):
            name = next(
                (
                    _text(item.get(key))
                    for key in ("name", "tagName", "label")
                    if _text(item.get(key))
                ),
                "",
            )
        else:
            name = _text(item)
        if name and name not in tags:
            tags.append(name)
    return tags


def _normalize_ingredient(item: Any, group: str) -> dict | None:
    if not isinstance(item, dict):
        name = _text(item)
        return {"group": group, "name": name} if name else None
    name = _text(item.get("foodIngredientName") or item.get("name"))
    if not name:
        return None
    return {
        "group": group,
        "name": name,
        "amount": _number(item.get("amount")),
        "unit": _text(item.get("unitName") or item.get("unit")) or None,
        "remark": _text(item.get("remark")) or None,
    }


def _normalize_ingredients(recipe: dict) -> list[dict]:
    # VoList 与 ApkVo 在当前接口中内容重复：前者优先，只有缺失时才回退。
    source = recipe.get("recipeIngredientsVoList")
    if not isinstance(source, dict) or not any(source.get(key) for key, _ in _INGREDIENT_GROUPS):
        source = recipe.get("recipeIngredientsApkVo")
    if not isinstance(source, dict):
        return []

    ingredients = []
    for source_key, group in _INGREDIENT_GROUPS:
        items = source.get(source_key)
        if not isinstance(items, list):
            continue
        for item in items:
            normalized = _normalize_ingredient(item, group)
            if normalized:
                ingredients.append(normalized)
    return ingredients


def _normalize_step_parameter(item: Any) -> dict | None:
    if not isinstance(item, dict):
        return None
    return {
        "parameter_id": _text(item.get("stepId")) or None,
        "time_seconds": _number(item.get("time")),
        "temperature_c": _number(item.get("temperature")),
        "speed": _text(item.get("speed")) or None,
        "power": _number(item.get("power")),
        "turn": _number(item.get("turn")),
        "weight": _number(item.get("weight")),
        "preset_pressure": _number(item.get("presetsPressureValue")),
        "cook_pressure": _number(item.get("cookPressureValue")),
        "accessory_type": _number(item.get("accessoriesType")),
        "accessory_image_url": _text(item.get("accessoriesImage")) or None,
        "device_models": _string_list(item.get("deviceModels")),
    }


def _normalize_steps(recipe: dict) -> list[dict]:
    raw_steps = recipe.get("recipeStepVoList")
    if not isinstance(raw_steps, list):
        raw_steps = recipe.get("steps") if isinstance(recipe.get("steps"), list) else []
    steps = []
    for position, item in enumerate(raw_steps, 1):
        if not isinstance(item, dict):
            description = _text(item)
            if description:
                steps.append({
                    "number": position,
                    "type": "manual",
                    "description": description,
                    "image_url": None,
                    "video_url": None,
                    "duration_seconds": None,
                    "parameters": [],
                })
            continue
        step_type = item.get("stepType")
        parameters = [
            normalized
            for raw in (item.get("recipeStepParameterVoList") or [])
            if (normalized := _normalize_step_parameter(raw))
        ]
        description = _text(
            item.get("stepDesc")
            or item.get("description")
            or item.get("instruction")
            or item.get("content")
        )
        if not description and not parameters:
            continue
        steps.append({
            "number": int(_number(item.get("serialNumb")) or position),
            "type": "device" if step_type in (1, "1") else "manual",
            "description": description,
            "image_url": _text(item.get("introduceUrl")) or None,
            "video_url": _text(item.get("videoUrl")) or None,
            "duration_seconds": _number(item.get("stepTime")),
            "parameters": parameters,
        })
    return sorted(steps, key=lambda item: item["number"])


def _epoch_ms_iso(value: Any) -> str | None:
    timestamp = _number(value)
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(float(timestamp) / 1000, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def normalize_recipe_detail(recipe: dict, language: str = "zh") -> dict:
    """把外部服务 `data` 转成稳定的 `recipe_detail_v1`。"""
    if not isinstance(recipe, dict):
        raise ValueError("recipe detail must be an object")
    recipe_id = _text(recipe.get("id") or recipe.get("cookId"))
    if not recipe_id:
        raise ValueError("recipe detail is missing id")
    steps = _normalize_steps(recipe)
    automatic_steps = [
        step for step in steps
        if step.get("type") == "device" and step.get("parameters")
    ]
    return {
        "schema_version": "recipe_detail_v1",
        "recipe_id": recipe_id,
        "cookId": recipe_id,
        "language": "en" if language == "en" else "zh",
        "name": _text(recipe.get("name")),
        "introduction": _text(recipe.get("recipeIntroduce")) or None,
        "tips": _text(recipe.get("recipeTips")) or None,
        "media": {
            "landscape_image_url": _text(recipe.get("landscapeImageUrl")) or None,
            "portrait_image_url": _text(recipe.get("portraitImageUrl")) or None,
            "intro_video_url": _text(recipe.get("introduceVideoUrl")) or None,
        },
        "tags": _tag_names(recipe.get("tagVoList")),
        "ingredients": _normalize_ingredients(recipe),
        "steps": steps,
        "cooking_time_seconds": _number(recipe.get("cookingTime")),
        "servings": _number(recipe.get("serviceSize")),
        "challenge_level": _number(recipe.get("challengeLevel")),
        "calorie_number": _number(recipe.get("calorieNumber")),
        "category_ids": _string_list(recipe.get("categoryIds")),
        "accessory_ids": _string_list(recipe.get("accessoryIds")),
        "device_model_ids": _string_list(recipe.get("deviceModelIds")),
        "is_custom_food": bool(recipe.get("isCustomFood")),
        "executable": bool(automatic_steps),
        "source_created_at": _epoch_ms_iso(recipe.get("createTime")),
        "source_updated_at": _epoch_ms_iso(recipe.get("updateTime")),
    }


def sanitize_recipe_detail_payload(recipe: dict) -> dict:
    """保留远端回执用于追溯，但移除调用账号专属状态。"""
    if not isinstance(recipe, dict):
        return {}
    return {
        key: value
        for key, value in recipe.items()
        if key not in _PRIVATE_RESPONSE_FIELDS
    }


def _duration_text(seconds: Any, lang: str) -> str:
    value = _number(seconds)
    if value is None or value <= 0:
        return ""
    if value % 60 == 0:
        minutes = int(value // 60)
        return f"{minutes} min" if lang == "en" else f"{minutes}分钟"
    return f"{value:g} sec" if lang == "en" else f"{value:g}秒"


def _duration_summary_text(seconds: Any, lang: str) -> str:
    value = _number(seconds)
    if value is None or value <= 0:
        return ""
    seconds_text = f"{value:g} sec" if lang == "en" else f"{value:g} 秒"
    if value % 60 == 0:
        minutes = int(value // 60)
        return (
            f"{seconds_text} ({minutes} min)"
            if lang == "en" else f"{seconds_text}（{minutes} 分钟）"
        )
    return seconds_text


def _ingredient_text(item: dict) -> str:
    name = _text(item.get("name"))
    amount = item.get("amount")
    unit = _text(item.get("unit"))
    remark = _text(item.get("remark"))
    prefix = ""
    if amount is not None:
        amount_text = f"{amount:g}" if isinstance(amount, float) else str(amount)
        prefix = f"{amount_text}{unit} "
    text = f"{prefix}{name}".strip()
    if remark:
        text += f"（{remark}）"
    return text


def _step_parameter_text(step: dict, lang: str) -> str:
    parameters = step.get("parameters") or []
    if not parameters:
        return ""
    parameter = parameters[0]
    parts = []
    duration = _duration_text(
        parameter.get("time_seconds") or step.get("duration_seconds"),
        lang,
    )
    if duration:
        parts.append(duration)
    temperature = parameter.get("temperature_c")
    if temperature not in (None, 0, "0"):
        parts.append(f"{temperature:g}°C")
    speed = _text(parameter.get("speed"))
    if speed:
        parts.append(f"speed {speed}" if lang == "en" else f"转速 {speed}")
    power = parameter.get("power")
    if power is not None:
        parts.append(f"power {power:g}" if lang == "en" else f"功率 {power:g}")
    return "; ".join(parts)


def _seasoning_name_keys(values: Any) -> set[str]:
    """从菜谱真实 seasonings 列提取名称，用于详情展示分组。"""
    from app.orchestrator.ingredient_quantities import parse_grounded_ingredient

    keys: set[str] = set()
    for value in values or []:
        if isinstance(value, dict):
            name = _text(value.get("name"))
        else:
            parsed, unresolved = parse_grounded_ingredient(value)
            name = _text((parsed or {}).get("name") or unresolved)
        if name:
            keys.add(name.casefold())
    return keys


def _detail_intro_text(
    name: str,
    *,
    has_ingredients: bool,
    has_steps: bool,
    lang: str,
) -> str:
    """只根据详情中实际存在的栏目生成自然承接，不增加菜谱事实。"""
    if lang == "en":
        if has_ingredients and has_steps:
            return f"I’ve organized the saved ingredients and cooking order for [{name}] below."
        if has_ingredients:
            return f"I’ve organized the saved ingredient information for [{name}] below; no steps have been added beyond the record."
        if has_steps:
            return f"I’ve laid out the saved cooking order for [{name}] below; no ingredient details have been added beyond the record."
        return f"Here is the currently saved information for [{name}], without filling in any missing recipe details."
    if has_ingredients and has_steps:
        return f"我把【{name}】现有的用料和操作顺序整理好了，下面按菜谱记录展开。"
    if has_ingredients:
        return f"我把【{name}】现有的用料整理好了；当前没有保存的步骤，我没有自行补写。"
    if has_steps:
        return f"我把【{name}】现有的操作顺序整理好了；当前没有保存的食材明细，我没有自行补写。"
    return f"下面是【{name}】当前能够核验的信息，缺少的菜谱内容我没有自行补写。"


def _detail_closing_text(*, has_steps: bool, tips: str, lang: str) -> str:
    """结尾只引用已显示的步骤或技巧，不生成新的烹饪建议。"""
    if lang == "en":
        if has_steps and tips:
            return "Follow the saved steps in order, and keep the recorded cooking tip above in view as you go."
        if has_steps:
            return "Follow the saved steps in order. No separate cooking tip is recorded, so I haven’t added one."
        if tips:
            return "The cooking tip above is saved with this recipe, but no detailed steps are currently available."
        return "No saved steps or cooking tips are currently available, so I’ve left the missing parts unfilled."
    if has_steps and tips:
        return "照着上面已保存的步骤依次做就好，操作时也留意菜谱里这条小技巧。"
    if has_steps:
        return "按上面已保存的步骤依次做就好；当前没有单独的小技巧记录，我就不额外补写了。"
    if tips:
        return "上面这条小技巧来自当前菜谱记录；具体步骤暂未保存，我没有自行补写。"
    return "当前没有保存的步骤或小技巧，我先把能够核验的菜谱信息原样留在这里。"


def format_recipe_detail(
    detail: dict,
    lang: str = "zh",
    *,
    description: str = "",
    seasonings: Any = None,
) -> str:
    """按固定顺序渲染单条 Markdown 详情卡。"""
    lang = "en" if lang == "en" else "zh"
    name = _text(detail.get("name")) or (
        "Verified recipe" if lang == "en" else "已核验菜谱"
    )
    ingredients = [
        item for item in (detail.get("ingredients") or [])
        if isinstance(item, dict) and _ingredient_text(item)
    ]
    seasoning_keys = _seasoning_name_keys(seasonings)
    if (
        lang == "en"
        and ingredients
        and all(
            isinstance(item, dict)
            and item.get("amount") is None
            and not _text(item.get("unit"))
            for item in ingredients
        )
    ):
        from app.orchestrator.ingredient_display import clean_english_ingredient_tokens

        cleaned_names = clean_english_ingredient_tokens(
            (item.get("name") for item in ingredients),
        )
        if cleaned_names:
            source_groups = {
                _text(item.get("name")).casefold(): item.get("group") or "main"
                for item in ingredients
                if _text(item.get("name"))
            }
            ingredients = [
                {
                    "name": cleaned_name,
                    "group": (
                        "seasoning"
                        if cleaned_name.casefold() in seasoning_keys
                        else source_groups.get(cleaned_name.casefold(), "main")
                    ),
                }
                for cleaned_name in cleaned_names
            ]
    food_ingredients = []
    seasoning_ingredients = []
    for item in ingredients:
        ingredient_name = _text(item.get("name"))
        is_seasoning = (
            _text(item.get("group")).lower() == "seasoning"
            or ingredient_name.casefold() in seasoning_keys
        )
        (seasoning_ingredients if is_seasoning else food_ingredients).append(item)

    steps = detail.get("steps") or []
    intro = _detail_intro_text(
        name,
        has_ingredients=bool(ingredients),
        has_steps=bool(steps),
        lang=lang,
    )
    lines = [
        f"## 🍳 {name}",
        "",
        intro,
        "",
        "### Recipe description" if lang == "en" else "### 菜谱描述",
        _text(detail.get("introduction") or description) or (
            "No saved description is available for this recipe."
            if lang == "en" else "暂无已保存的菜谱描述。"
        ),
    ]

    media = detail.get("media") or {}
    image = _text(
        media.get("landscape_image_url")
        or media.get("portrait_image_url")
    )
    if image and re.fullmatch(r"https?://[^\s<>()]+", image, flags=re.IGNORECASE):
        lines.extend(["", f"![{name} #320px #240px]({image})"])

    lines.extend(["", "### Ingredients" if lang == "en" else "### 食材"])
    if food_ingredients:
        lines.extend(
            f"- {_ingredient_text(item)}"
            for item in food_ingredients
        )
    else:
        lines.append(
            "- No food ingredients are listed."
            if lang == "en" else "- 暂无已保存的食材。"
        )

    lines.extend(["", "### Seasonings" if lang == "en" else "### 调料"])
    if seasoning_ingredients:
        lines.extend(
            f"- {_ingredient_text(item)}"
            for item in seasoning_ingredients
        )
    else:
        lines.append(
            "- No separate seasonings are listed."
            if lang == "en" else "- 暂无单独保存的调料。"
        )

    lines.extend([
        "",
        "### Cooking steps" if lang == "en" else "### 烹饪步骤",
    ])
    if steps:
        for position, step in enumerate(steps, 1):
            number = step.get("number") or position
            description = _text(step.get("description"))
            parameters = _step_parameter_text(step, lang)
            if step.get("ai_generated") and not parameters:
                duration = _duration_text(step.get("duration_seconds"), lang)
                parameters = (
                    f"estimated {duration}" if lang == "en" else f"预计 {duration}"
                ) if duration else ""
            suffix = f" ({parameters})" if parameters else ""
            lines.append(f"{number}. {description or ('Device step' if lang == 'en' else '设备步骤')}{suffix}")
    else:
        lines.append(
            "No saved cooking steps are available."
            if lang == "en" else "暂无已保存的烹饪步骤。"
        )

    tips = _text(detail.get("tips"))
    lines.extend([
        "",
        "### Cooking tips" if lang == "en" else "### 烹饪小技巧",
        tips or (
            "No saved cooking tips are available."
            if lang == "en" else "暂无已保存的烹饪小技巧。"
        ),
        "",
        _detail_closing_text(
            has_steps=bool(steps),
            tips=tips,
            lang=lang,
        ),
    ])
    return "\n".join(lines)
