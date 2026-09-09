"""基于真实菜名与食材补全缺失的展示型菜谱详情。

模型生成内容只用于展示和检索，不生成设备控制参数，也不覆盖 Mock Device
已经提供的步骤、总时长或份量。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen

from app.core.config import settings
from app.observability.trace import observe_model_call

logger = logging.getLogger(__name__)

_COMMON_INGREDIENT_TERMS_ZH = (
    "盐", "姜", "辣椒", "花椒", "豆瓣酱", "蚝油", "醋", "淀粉", "鸡精",
    "味精", "香菜", "芝麻", "黄油", "牛奶", "鸡蛋", "面粉", "胡萝卜",
    "洋葱", "土豆", "番茄", "蘑菇", "高汤",
)
_COMMON_INGREDIENT_TERMS_EN = (
    "salt", "ginger", "chili", "chilli", "pepper", "cornstarch", "flour",
    "butter", "milk", "egg", "vinegar", "oyster sauce", "cilantro", "parsley",
    "carrot", "onion", "potato", "tomato", "mushroom", "stock", "broth",
)
_INGREDIENT_ALIASES_ZH = {
    "淀粉": ("生粉", "玉米粉", "地瓜粉", "红薯粉", "马铃薯粉", "澄粉"),
    "蘑菇": ("香菇", "菌菇", "口蘑", "平菇", "杏鲍菇", "金针菇"),
    "辣椒": ("杭椒", "青椒", "红椒", "尖椒", "朝天椒", "小米椒", "泡椒"),
    "面粉": ("小麦粉", "低筋粉", "中筋粉", "高筋粉"),
}
_QUANTITY_IN_NAME_RE = re.compile(
    r"(?<!\d)\d+(?:\.\d+)?(?:\s*[-–~至]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:kg|g|mg|ml|l|克|千克|公斤|毫升|升|个|只|颗|枚|根|片|瓣|勺|茶匙|汤匙)\b",
    flags=re.IGNORECASE,
)
_ALLOWED_QUANTITY_UNITS = {
    "g", "kg", "mg", "ml", "l", "pcs", "tsp", "tbsp", "cup", "pinch",
    "handful",
    "克", "千克", "公斤", "毫升", "升",
    "个", "只", "颗", "枚", "根", "片", "瓣", "勺", "茶匙", "汤匙",
}


_RECIPE_COMPLETION_MODEL = (
    os.getenv("RECIPE_COMPLETION_MODEL", "").strip() or settings.LLM_MODEL
)

_recipe_completion_llm = ChatQwen(
    model=_RECIPE_COMPLETION_MODEL,
    temperature=0.2,
    max_tokens=2400,
    timeout=30,
    max_retries=1,
    enable_thinking=False,
    api_key=settings.DASHSCOPE_API_KEY,
    base_url=settings.DASHSCOPE_BASE_URL,
)


def recipe_ai_completion_enabled() -> bool:
    return os.getenv("RECIPE_AI_COMPLETION_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def build_recipe_detail_seed(recipe: dict, lang: str = "zh") -> dict:
    """把 Milvus 候选的真实菜名/食材转成可补全的最小详情骨架。"""
    recipe_id = str(recipe.get("cookId") or recipe.get("id") or "").strip()
    from app.orchestrator.ingredient_quantities import build_grounded_ingredients

    source = recipe.get("ingredients_raw") or recipe.get("ingredients") or []
    ingredients, unresolved = build_grounded_ingredients(source)
    seasoning_items, seasoning_unresolved = build_grounded_ingredients(
        recipe.get("seasonings") or [],
    )
    seasoning_names = {
        str(item.get("name") or "").strip().casefold()
        for item in seasoning_items
        if str(item.get("name") or "").strip()
    } | {
        str(name or "").strip().casefold()
        for name in seasoning_unresolved
        if str(name or "").strip()
    }
    for item in ingredients:
        if str(item.get("name") or "").strip().casefold() in seasoning_names:
            item["group"] = "seasoning"
    ingredients.extend({
        "group": "seasoning" if name.casefold() in seasoning_names else "main",
        "name": name,
        "amount": None,
        "unit": None,
        "remark": None,
        "quantity_source": "llm_fallback_required",
    } for name in unresolved)
    return {
        "schema_version": "recipe_detail_v1",
        "recipe_id": recipe_id,
        "cookId": recipe_id,
        "language": "en" if lang == "en" else "zh",
        "name": re.sub(r"\s+", " ", str(recipe.get("name") or "")).strip(),
        "introduction": re.sub(
            r"\s+", " ", str(recipe.get("description") or ""),
        ).strip() or None,
        "tips": None,
        "media": {
            "landscape_image_url": str(
                recipe.get("image") or recipe.get("image_url") or ""
            ).strip() or None,
            "portrait_image_url": None,
            "intro_video_url": None,
        },
        "tags": list(recipe.get("tags") or []),
        "ingredients": ingredients,
        "steps": [],
        "cooking_time_seconds": None,
        "servings": None,
        "challenge_level": None,
        "calorie_number": None,
        "category_ids": [],
        "accessory_ids": [],
        "device_model_ids": [],
        "is_custom_food": False,
        "executable": False,
        "source_created_at": None,
        "source_updated_at": None,
    }


def _extract_json(content: Any) -> dict:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _positive_int(value: Any, *, minimum: int, maximum: int) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if minimum <= parsed <= maximum else None


def _generated_steps(value: Any) -> list[dict]:
    if not isinstance(value, list) or not 2 <= len(value) <= 12:
        return []
    steps = []
    for number, raw in enumerate(value, 1):
        if not isinstance(raw, dict):
            return []
        description = re.sub(r"\s+", " ", str(raw.get("description") or "")).strip()
        duration = _positive_int(
            raw.get("duration_seconds"),
            minimum=0,
            maximum=604800,
        )
        if not description or len(description) > 500 or duration is None:
            return []
        steps.append({
            "number": number,
            "type": "ai_generated_manual",
            "description": description,
            "image_url": None,
            "video_url": None,
            # 模型偶尔给“装盘”步骤返回 0 秒；统一记为最小展示估时 15 秒。
            "duration_seconds": max(15, duration),
            "parameters": [],
            "ai_generated": True,
        })
    return steps


def _ingredient_names(detail: dict) -> list[str]:
    names = []
    for item in detail.get("ingredients") or []:
        if not isinstance(item, dict):
            continue
        name = re.sub(r"\s+", " ", str(item.get("name") or "")).strip()
        if name and name not in names:
            names.append(name)
    return names


def _ingredient_has_quantity(item: dict) -> bool:
    if item.get("amount") not in (None, "", 0, "0"):
        return True
    name = re.sub(r"\s+", " ", str(item.get("name") or "")).strip()
    return bool(_QUANTITY_IN_NAME_RE.search(name))


def _missing_quantity_names(detail: dict, *, force: bool = False) -> list[str]:
    names = []
    for item in detail.get("ingredients") or []:
        if not isinstance(item, dict):
            continue
        name = re.sub(r"\s+", " ", str(item.get("name") or "")).strip()
        if name and (force or not _ingredient_has_quantity(item)) and name not in names:
            names.append(name)
    return names


def _generated_ingredient_quantities(
    value: Any,
    expected_names: list[str],
) -> dict[str, tuple[int | float, str]]:
    if not expected_names:
        return {}
    if not isinstance(value, list):
        return {}
    expected = set(expected_names)
    quantities: dict[str, tuple[int | float, str]] = {}
    for raw in value:
        if not isinstance(raw, dict):
            return {}
        name = re.sub(r"\s+", " ", str(raw.get("name") or "")).strip()
        unit = re.sub(r"\s+", "", str(raw.get("unit") or "")).strip()
        if name not in expected or unit.lower() not in {
            item.lower() for item in _ALLOWED_QUANTITY_UNITS
        }:
            return {}
        try:
            amount = float(raw.get("amount"))
        except (TypeError, ValueError):
            return {}
        if not 0 < amount <= 100_000:
            return {}
        quantities[name] = (
            int(amount) if amount.is_integer() else round(amount, 2),
            unit,
        )
    return quantities if set(quantities) == expected else {}


def _grounded_quantity_items(
    value: Any,
    expected_names: list[str],
) -> list[dict]:
    """校验专用补量结果，并允许模型拆开原文中的并列食材。

    例如原始字段可能是一整行 ``Cherry tomatoes, lettuce, onion``，原文没有
    各项用量。模型可以将其拆成三项，但每个返回名称都必须是原文的子串，
    不能引入任何新食材。
    """
    if not isinstance(value, list) or not expected_names:
        return []

    def normalized(text: Any) -> str:
        return re.sub(
            r"[\s,，;；.:：()（）/\\_-]+",
            "",
            str(text or "").casefold(),
        )

    allowed_units = {unit.casefold() for unit in _ALLOWED_QUANTITY_UNITS}
    expected = [(name, normalized(name)) for name in expected_names]
    result: list[dict] = []
    covered: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            return []
        name = re.sub(r"\s+", " ", str(raw.get("name") or "")).strip()
        unit = re.sub(r"\s+", "", str(raw.get("unit") or "")).strip()
        name_key = normalized(name)
        matches = [
            source_name
            for source_name, source_key in expected
            if name_key
            and (
                name_key == source_key
                or (len(name_key) >= 2 and name_key in source_key)
            )
        ]
        if len(matches) != 1 or unit.casefold() not in allowed_units:
            return []
        try:
            amount = float(raw.get("amount"))
        except (TypeError, ValueError):
            return []
        if not 0 < amount <= 100_000:
            return []
        source_name = matches[0]
        covered.add(source_name)
        result.append({
            "group": "main",
            "name": name,
            "amount": int(amount) if amount.is_integer() else round(amount, 2),
            "unit": unit,
            "remark": None,
            "quantity_source": "llm_fallback",
            "fallback_for": source_name,
        })
    return result if covered == set(expected_names) else []


async def complete_ingredient_quantities(
    *,
    recipe_id: str,
    recipe_name: str,
    ingredient_names: list[str],
    lang: str = "zh",
) -> list[dict]:
    """只为原始文本缺用量的既有食材补 amount/unit。

    不生成步骤、时长、份量或技巧；输出名称必须与输入逐字一致，调用方可安全
    丢弃模型返回的任何非预期内容。
    """
    names = []
    for raw in ingredient_names:
        name = re.sub(r"\s+", " ", str(raw or "")).strip()
        if name and name not in names:
            names.append(name)
    if not names:
        return []

    language_rule = (
        "Ingredient names are English. Keep exact names, or split a comma-separated "
        "source line only into ingredient names that occur verbatim in that line. "
        "Units must be English units: g, kg, mg, ml, l, pcs, tsp, tbsp, cup, "
        "pinch or handful."
        if lang == "en"
        else "食材名称使用中文；每个 name 必须逐字保持不变，单位使用中文或 g/ml。"
    )
    system = f"""你是 CookClaw 的食材用量补全器。菜名和食材名称均来自真实数据，
但以下少量食材的原文没有提供明确重量或单位。你只负责补 amount 和 unit。

必须遵守：
1. {language_rule}
2. ingredient_quantities 必须且只能覆盖输入的全部食材，不得新增或遗漏。
3. amount 必须是大于 0 的数字；固体优先 g，液体优先 ml，可数食材可用
   个、只、颗、枚、根、片、瓣、勺、茶匙或汤匙。
4. 只输出 JSON，不输出步骤、时长、份量、技巧、Markdown 或解释。

输出：
{{
  "ingredient_quantities": [
    {{"name": "必须与输入完全一致", "amount": 100, "unit": "g"}}
  ]
}}"""
    payload = {
        "recipe_id": str(recipe_id or ""),
        "recipe_name": str(recipe_name or ""),
        "language": "en" if lang == "en" else "zh",
        "ingredients_needing_quantities": names,
    }
    try:
        response = await observe_model_call(
            lambda: _recipe_completion_llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]),
            stage="recipe_quantity_completion",
            model=_RECIPE_COMPLETION_MODEL,
            timeout_seconds=35,
            reply_generation=True,
        )
        value = _extract_json(response.content)
        items = _grounded_quantity_items(
            value.get("ingredient_quantities"),
            names,
        )
        if not items:
            logger.warning(
                "食材用量补全返回未通过校验 recipe_id=%s names=%s response=%s",
                str(recipe_id or "")[-8:],
                names,
                str(response.content or "")[:1200],
            )
    except Exception as exc:
        logger.warning(
            "食材用量补全失败 recipe_id=%s error_type=%s",
            str(recipe_id or "")[-8:],
            type(exc).__name__,
        )
        return []
    if not items:
        return []
    return items


def _generated_tip(value: Any, ingredients: list[str], lang: str) -> str:
    tip = re.sub(r"\s+", " ", str(value or "")).strip()
    if not 8 <= len(tip) <= 180 or any(marker in tip for marker in ("#", "```")):
        return ""
    unsupported = _unsupported_ingredient_terms(
        [{"description": tip}],
        ingredients,
        lang,
    )
    return "" if unsupported else tip


def _unsupported_ingredient_terms(
    steps: list[dict],
    ingredients: list[str],
    lang: str,
) -> list[str]:
    source = " ".join(ingredients).lower()
    generated = " ".join(
        str(step.get("description") or "") for step in steps
    ).lower()
    terms = (
        _COMMON_INGREDIENT_TERMS_EN
        if lang == "en" else _COMMON_INGREDIENT_TERMS_ZH
    )
    return [
        term
        for term in terms
        if (
            term.lower() in generated
            and term.lower() not in source
            and not (
                lang != "en"
                and any(alias in source for alias in _INGREDIENT_ALIASES_ZH.get(term, ()))
            )
        )
    ]


async def complete_missing_recipe_fields(
    detail: dict,
    *,
    force: bool = False,
) -> dict:
    """用模型补齐展示详情中的缺失字段，返回带来源标记的新对象。"""
    if not isinstance(detail, dict):
        return detail
    ingredients = _ingredient_names(detail)
    name = re.sub(r"\s+", " ", str(detail.get("name") or "")).strip()
    if not name or not ingredients:
        return detail

    missing_steps = force or not bool(detail.get("steps"))
    missing_time = force or detail.get("cooking_time_seconds") in (None, 0, "0")
    missing_servings = force or detail.get("servings") in (None, 0, "0")
    missing_quantity_names = _missing_quantity_names(detail, force=force)
    missing_tips = force or not bool(str(detail.get("tips") or "").strip())
    if not any((
        missing_steps,
        missing_time,
        missing_servings,
        missing_quantity_names,
        missing_tips,
    )):
        return detail

    lang = "en" if detail.get("language") == "en" else "zh"
    language_rule = (
        "Write every step in natural English."
        if lang == "en"
        else "所有步骤使用自然、简洁的简体中文。"
    )
    system = f"""你是 CookClaw 的菜谱补全器。输入中的菜谱名称和食材来自真实数据，
但食材用量、步骤、总时长、份量或烹饪技巧可能缺失。请基于这些已知食材
生成一份合理、可在普通厨房手动完成的菜谱草稿。

必须遵守：
1. {language_rule}
2. 只能使用输入列出的食材；不得添加输入中没有的肉类、蔬菜、调料或配菜。
3. 生成 2～12 个有顺序的步骤，每步给出 description 和估算 duration_seconds
   （必须是 15～604800 的整数，装盘等短步骤也至少填写 15 秒）。
4. total_time_seconds 是整道菜从准备到完成的估算总时长，范围 60～604800 秒。
5. servings 是合理份量，范围 1～100 人；普通家常菜优先 2～4 人，
   只有食材规模明确对应聚餐或批量制作时才使用更大份量。
6. ingredient_quantities 只为 fields_to_generate 中要求补用量的原始食材生成。
   name 必须逐字使用输入名称，不得改名、合并或新增食材；固体优先用 g，
   液体优先用 ml，按数量计的食材可用 个、只、颗、根、片或瓣。
7. cooking_tip 只写一条具体、实用的火候、刀工或操作技巧，不得引入输入中
   没有的食材，不写营养或疗效，中文 20～80 字、英文 12～50 词。
8. 不生成或猜测任何设备温度、转速、功率、压力、设备型号或执行 ID。
9. 不写故事、推荐理由或 Markdown；只输出 JSON。

输出结构：
{{
  "steps": [
    {{"description": "步骤正文", "duration_seconds": 300}}
  ],
  "total_time_seconds": 1800,
  "servings": 2,
  "ingredient_quantities": [
    {{"name": "必须与输入食材名称完全一致", "amount": 250, "unit": "g"}}
  ],
  "cooking_tip": "一条具体实用的小技巧"
}}"""
    payload = {
        "recipe_id": str(detail.get("recipe_id") or ""),
        "name": name,
        "language": lang,
        "ingredients": ingredients,
        "servings": detail.get("servings"),
        "ingredients_needing_quantities": missing_quantity_names,
        "fields_to_generate": [
            field
            for field, missing in (
                ("steps", missing_steps),
                ("total_time_seconds", missing_time),
                ("servings", missing_servings),
                ("ingredient_quantities", bool(missing_quantity_names)),
                ("cooking_tip", missing_tips),
            )
            if missing
        ],
    }

    async def invoke(prompt: str) -> dict:
        response = await observe_model_call(
            lambda: _recipe_completion_llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]),
            stage="recipe_detail_completion",
            model=_RECIPE_COMPLETION_MODEL,
            timeout_seconds=35,
            reply_generation=True,
        )
        return _extract_json(response.content)

    try:
        generated = await invoke(system)
    except Exception as exc:
        logger.warning(
            "菜谱 AI 补全失败，保留原详情 recipe_id=%s error_type=%s",
            str(detail.get("recipe_id") or "")[-8:],
            type(exc).__name__,
        )
        return detail

    def validate(
        value: dict,
    ) -> tuple[
        list[dict],
        int | None,
        int | None,
        dict[str, tuple[int | float, str]],
        str,
        list[str],
    ]:
        candidate_steps = _generated_steps(value.get("steps"))
        candidate_time = _positive_int(
            value.get("total_time_seconds"),
            minimum=60,
            maximum=604800,
        )
        candidate_servings = _positive_int(
            value.get("servings"),
            minimum=1,
            maximum=100,
        )
        candidate_quantities = _generated_ingredient_quantities(
            value.get("ingredient_quantities"),
            missing_quantity_names,
        )
        candidate_tip = _generated_tip(
            value.get("cooking_tip"),
            ingredients,
            lang,
        )
        unsupported = _unsupported_ingredient_terms(
            candidate_steps,
            ingredients,
            lang,
        )
        step_time_sum = sum(
            int(step.get("duration_seconds") or 0)
            for step in candidate_steps
        )
        # 步骤时长和总时长都来自同一次生成；当模型低估总时长时，以逐步估时
        # 之和归一化，避免为了一个可机械修复的算术错误反复调用模型。
        if (
            candidate_steps
            and candidate_time is not None
            and candidate_time < step_time_sum <= 604_800
        ):
            candidate_time = step_time_sum
        return (
            candidate_steps,
            candidate_time,
            candidate_servings,
            candidate_quantities,
            candidate_tip,
            unsupported,
        )

    (
        steps,
        total_time,
        servings,
        ingredient_quantities,
        cooking_tip,
        unsupported_terms,
    ) = validate(generated)
    invalid = (
        (missing_steps and not steps)
        or (missing_time and total_time is None)
        or (missing_servings and servings is None)
        or (missing_quantity_names and not ingredient_quantities)
        or (missing_tips and not cooking_tip)
        or bool(unsupported_terms)
    )
    if invalid:
        repair_system = f"""{system}

上一次输出未通过校验。请重新生成完整 JSON，并额外遵守：
- 以下词被检测为输入食材中不存在，步骤里绝对不能再出现：{json.dumps(unsupported_terms, ensure_ascii=False)}
- 每一步 duration_seconds 的总和不能超过 total_time_seconds。
- 只能使用这份原始食材列表：{json.dumps(ingredients, ensure_ascii=False)}
- ingredient_quantities 必须完整覆盖且只能覆盖：{json.dumps(missing_quantity_names, ensure_ascii=False)}
- cooking_tip 不能引入原始食材之外的新食材。
不要解释错误，只输出修正后的 JSON。"""
        try:
            generated = await invoke(repair_system)
            (
                steps,
                total_time,
                servings,
                ingredient_quantities,
                cooking_tip,
                unsupported_terms,
            ) = validate(generated)
        except Exception as exc:
            logger.warning(
                "菜谱 AI 补全重写失败，保留原详情 recipe_id=%s error_type=%s",
                str(detail.get("recipe_id") or "")[-8:],
                type(exc).__name__,
            )
            return detail

    # 某些强关联菜名会让模型在“整份重写”后仍固执加入默认调料。此时不再
    # 随机重抽整份，而把合格草稿交给受限编辑器，只改写含禁用词的动作。
    if (
        unsupported_terms
        and steps
        and total_time is not None
        and servings is not None
    ):
        editor_system = f"""你是 CookClaw 的菜谱 JSON 受限编辑器。
下面草稿的结构、步骤数量、时长和份量已经合格，但步骤里出现了原始食材清单
没有的词。请删除含禁用食材的动作，或把该动作改写成只使用允许食材也能完成
的中性动作（例如“翻拌均匀”“继续加热至熟”）。

严格要求：
1. 禁用词绝不能出现在输出中：{json.dumps(unsupported_terms, ensure_ascii=False)}
2. 只能使用允许食材：{json.dumps(ingredients, ensure_ascii=False)}
3. 保持原语言、2～12 步、每步 description 和整数 duration_seconds。
4. 不添加任何新食材、设备参数、解释或 Markdown。
5. 保留并完整输出 ingredient_quantities 和 cooking_tip。
6. 只输出与原结构相同的 JSON：steps、total_time_seconds、servings、
   ingredient_quantities、cooking_tip。"""
        editor_payload = {
            "allowed_ingredients": ingredients,
            "forbidden_terms": unsupported_terms,
            "draft": generated,
        }
        try:
            response = await observe_model_call(
                lambda: _recipe_completion_llm.ainvoke([
                    SystemMessage(content=editor_system),
                    HumanMessage(content=json.dumps(editor_payload, ensure_ascii=False)),
                ]),
                stage="recipe_detail_repair",
                model=_RECIPE_COMPLETION_MODEL,
                timeout_seconds=35,
                reply_generation=True,
            )
            generated = _extract_json(response.content)
            (
                steps,
                total_time,
                servings,
                ingredient_quantities,
                cooking_tip,
                unsupported_terms,
            ) = validate(generated)
        except Exception as exc:
            logger.warning(
                "菜谱 AI 受限编辑失败，保留原详情 recipe_id=%s error_type=%s",
                str(detail.get("recipe_id") or "")[-8:],
                type(exc).__name__,
            )
            return detail

    if (
        (missing_steps and not steps)
        or (missing_time and total_time is None)
        or (missing_servings and servings is None)
        or (missing_quantity_names and not ingredient_quantities)
        or (missing_tips and not cooking_tip)
        or unsupported_terms
    ):
        logger.warning(
            "菜谱 AI 补全结果校验失败，保留原详情 recipe_id=%s "
            "steps_valid=%s raw_steps_type=%s raw_steps_len=%s "
            "raw_steps_missing_desc=%s raw_steps_missing_duration=%s "
            "time_valid=%s servings_valid=%s unsupported=%s",
            str(detail.get("recipe_id") or "")[-8:],
            bool(steps),
            type(generated.get("steps")).__name__,
            (
                len(generated.get("steps"))
                if isinstance(generated.get("steps"), list) else -1
            ),
            (
                sum(
                    not str(item.get("description") or "").strip()
                    for item in generated.get("steps")
                    if isinstance(item, dict)
                )
                if isinstance(generated.get("steps"), list) else -1
            ),
            (
                sum(
                    item.get("duration_seconds") is None
                    for item in generated.get("steps")
                    if isinstance(item, dict)
                )
                if isinstance(generated.get("steps"), list) else -1
            ),
            total_time is not None,
            servings is not None,
            ",".join(unsupported_terms) or "-",
        )
        return detail

    completed = deepcopy(detail)
    generated_fields = []
    if missing_steps:
        completed["steps"] = steps
        generated_fields.append("steps")
    if missing_time:
        completed["cooking_time_seconds"] = total_time
        generated_fields.append("cooking_time_seconds")
    if missing_servings:
        completed["servings"] = servings
        generated_fields.append("servings")
    if missing_quantity_names:
        completed["ingredients"] = deepcopy(detail.get("ingredients") or [])
        for item in completed["ingredients"]:
            if not isinstance(item, dict):
                continue
            name = re.sub(r"\s+", " ", str(item.get("name") or "")).strip()
            quantity = ingredient_quantities.get(name)
            if quantity and (force or not _ingredient_has_quantity(item)):
                item["amount"], item["unit"] = quantity
        generated_fields.append("ingredient_quantities")
    if missing_tips:
        completed["tips"] = cooking_tip
        generated_fields.append("tips")
    completed["ai_generated"] = True
    completed["generated_fields"] = generated_fields
    completed["completion_source"] = "llm_from_name_and_ingredients"
    completed["completion_model"] = _RECIPE_COMPLETION_MODEL
    # 模型补全的手动步骤绝不能改变设备执行能力；该值仍只由真实自动步骤决定。
    completed["executable"] = bool(detail.get("executable"))
    return completed
