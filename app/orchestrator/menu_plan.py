"""确定性菜单规划：把"四菜一汤"拆成独立真实检索组并做硬校验。

本模块不生成菜名，也不调用推荐表达模型。每个最终条目都必须来自本轮
recipe search 的真实结果；数量、类型、忌口、作用域与去重均由代码判定。
"""
from __future__ import annotations

import re
from typing import Awaitable, Callable

from app.agent.fast_path import _run_search_subprocess
from app.orchestrator.search_request import SearchRequest, filter_hard_constraint_violations


SearchCallable = Callable[[str, int, str], Awaitable[dict | None]]

_SOUP_MEALS = {"汤", "soup"}
_NON_DISH_MEALS = {
    "汤", "soup", "粥", "porridge", "饮品", "beverage", "drink",
    "甜品", "甜点", "dessert",
}

_SCOPED_CANONICAL = {
    "cuisine": {
        "湘菜": "hunan", "湖南菜": "hunan", "川菜": "sichuan", "四川菜": "sichuan",
        "粤菜": "cantonese", "广东菜": "cantonese", "鲁菜": "shandong",
        "苏菜": "jiangsu", "淮扬菜": "jiangsu", "浙菜": "zhejiang",
        "杭州菜": "zhejiang", "闽菜": "fujian", "徽菜": "anhui",
        "东北菜": "northeastern_chinese", "西北菜": "northwestern_chinese",
        "家常菜": "home_style", "日料": "japanese", "韩餐": "korean",
        "泰式": "thai", "越南菜": "vietnamese", "意大利菜": "italian",
        "法餐": "french", "地中海菜": "mediterranean", "墨西哥菜": "mexican",
        "印度菜": "indian", "中东菜": "middle_eastern", "摩洛哥菜": "moroccan",
        "美式": "american",
    },
    "flavor": {
        "辣": "spicy", "香辣": "spicy", "麻辣": "numbing_spicy",
        "咸鲜": "savory", "清淡": "light", "甜": "sweet", "甜味": "sweet",
        "酸甜": "sweet_sour", "酸辣": "sour_spicy", "酸": "sour",
        "咸香": "salty", "鲜味": "umami", "鲜香": "aromatic", "奶香": "creamy",
    },
}


def _canonical_scoped_value(dimension: str, value) -> str:
    raw = str(value).strip()
    return _SCOPED_CANONICAL[dimension].get(
        raw,
        raw.lower().replace("-", "_").replace(" ", "_"),
    )


def _as_values(value) -> list[str]:
    if isinstance(value, dict):
        value = list(value.values())
    elif isinstance(value, str):
        value = re.split(r"[、,，;/；|]", value)
    out: list[str] = []
    for item in value or []:
        text = re.sub(r"\s+", " ", str(item or "")).strip().lower()
        if text and text not in out:
            out.append(text)
    return out


def _facet_values(result: dict, dimension: str) -> list[str]:
    metadata = result.get("metadata") or {}
    facets = metadata.get("facets") or {}
    if not isinstance(facets, dict):
        return []
    return _as_values(facets.get(dimension))


def _recipe_id(result: dict) -> str:
    metadata = result.get("metadata") or {}
    return str(metadata.get("recipe_id") or result.get("id") or "").strip()


def _recipe_name(result: dict) -> str:
    return re.sub(r"\s+", " ", str((result.get("metadata") or {}).get("name") or "")).strip()


def _canonical_name(result: dict) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", _recipe_name(result).lower())


def _is_same_language(result: dict, lang: str) -> bool:
    name = _recipe_name(result)
    if not name:
        return False
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", name))
    has_latin = bool(re.search(r"[a-z]", name, flags=re.IGNORECASE))
    return (has_latin and not has_cjk) if lang == "en" else has_cjk


def _is_verified_soup(result: dict) -> bool:
    """汤槽验证：名称或类型 facet 任一命中即可，放宽以覆盖缺少 facet 的真实汤品。"""
    meals = set(_facet_values(result, "meal"))
    name = _recipe_name(result).lower()
    has_soup_facet = bool(meals.intersection(_SOUP_MEALS))
    has_soup_name = bool(re.search(r"汤|羹|\b(?:soup|broth|stew)\b", name, flags=re.IGNORECASE))
    return has_soup_facet or has_soup_name


def _soup_confidence(result: dict) -> int:
    """汤品置信度：facet + 名称双重命中得分最高，用于排序而非过滤。"""
    meals = set(_facet_values(result, "meal"))
    name = _recipe_name(result).lower()
    score = 0
    if meals.intersection(_SOUP_MEALS):
        score += 1
    if re.search(r"汤|羹|\b(?:soup|broth|stew)\b", name, flags=re.IGNORECASE):
        score += 1
    return score


def _is_verified_dish(result: dict) -> bool:
    """普通菜位拒绝已明确标注为汤粥、饮品或甜品的结果。"""
    meals = set(_facet_values(result, "meal"))
    return not meals.intersection(_NON_DISH_MEALS)


def _matches_scoped_preference(result: dict, scoped: dict) -> bool:
    requested_cuisines = {
        _canonical_scoped_value("cuisine", item)
        for item in scoped.get("cuisines") or []
    }
    requested_flavors = {
        _canonical_scoped_value("flavor", item)
        for item in scoped.get("flavors") or []
    }
    actual_cuisines = set(_facet_values(result, "cuisine"))
    actual_flavors = set(_facet_values(result, "flavor"))

    cuisine_ok = not requested_cuisines or any(
        requested in actual or actual in requested
        for requested in requested_cuisines for actual in actual_cuisines
    )
    flavor_ok = not requested_flavors or any(
        requested in actual or actual in requested
        for requested in requested_flavors for actual in actual_flavors
    )
    return cuisine_ok and flavor_ok


def _matches_required_ingredient(result: dict, required: str) -> bool:
    """必备菜位必须从名称、食材或主料 facet 中找到原始事实证据。"""
    wanted = re.sub(
        r"[^0-9a-z\u4e00-\u9fff]+",
        "",
        str(required or "").lower(),
    )
    if not wanted:
        return False
    metadata = result.get("metadata") or {}
    facets = metadata.get("facets") or {}
    evidence = [
        metadata.get("name"),
        metadata.get("description"),
        metadata.get("ingredients"),
        (facets.get("main_ingredient") if isinstance(facets, dict) else None),
    ]
    actual = [
        re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", item.lower())
        for value in evidence
        for item in _as_values(value)
    ]
    return any(wanted == item or wanted in item for item in actual if item)


def _has_any_scoped_evidence(result: dict, scoped_preferences: list[dict]) -> bool:
    for scoped in scoped_preferences:
        requested = {
            _canonical_scoped_value("cuisine" if key == "cuisines" else "flavor", item)
            for key in ("cuisines", "flavors")
            for item in scoped.get(key) or []
        }
        actual = set(_facet_values(result, "cuisine") + _facet_values(result, "flavor"))
        if any(req in value or value in req for req in requested for value in actual):
            return True
    return False


def _menu_fit_score(result: dict, request: SearchRequest, role: str) -> float:
    """基于真实 facets 做轻量整桌适配；分值只能微调，不能替代 RAG。"""
    scenes = set(_facet_values(result, "scene"))
    methods = set(_facet_values(result, "method"))
    meals = set(_facet_values(result, "meal"))
    cuisines = set(_facet_values(result, "cuisine"))
    metadata = result.get("metadata") or {}
    difficulty = str(metadata.get("difficulty") or "").strip().lower()
    requested_scenes = {
        str(item or "").strip().lower()
        for item in request.scenes
        if str(item or "").strip()
    }

    score = 0.0
    if "banquet" in scenes:
        score += 0.20
    if "rice_companion" in scenes:
        score += 0.12
    wants_home_style = any(
        str(item or "").strip().lower().replace("-", "_") in {
            "家常", "家常菜", "home_style", "homestyle",
        }
        for item in request.cuisines
    )
    if wants_home_style and "home_style" in cuisines:
        score += 0.18
    elif wants_home_style and cuisines.intersection(
        {"italian", "french", "spanish", "american", "mexican"}
    ):
        score -= 0.14
    if requested_scenes.intersection({"快手", "简单", "省事", "quick", "easy"}):
        if "quick" in scenes:
            score += 0.16
        if difficulty in {"1", "2", "简单", "容易", "easy", "simple"}:
            score += 0.08
        elif difficulty in {"4", "5", "困难", "进阶", "hard", "advanced"}:
            score -= 0.24

    # 正常聚餐菜位不优先选择泥、汁等加工态；仍保留在候选池，库内没有更
    # 合适结果时可以兜底，避免把名称写死成特例。
    if methods.intersection({"blended", "pureeing", "juiced"}):
        score -= 0.42
    if scenes.intersection({"food_preparation", "device_maintenance"}):
        score -= 0.48
    if scenes.intersection({"child_friendly", "elderly_friendly"}) and not (
        requested_scenes.intersection({"儿童", "孩子", "老人", "children", "elderly"})
    ):
        score -= 0.10
    if "staple" in meals and "main_course" not in meals:
        score -= 0.16
    return score


def _menu_overlap_penalty(result: dict, selected: list[dict]) -> float:
    """避免整桌连续使用相同主料和做法，同时不把多样性变成硬过滤。"""
    if not selected:
        return 0.0
    main = set(_facet_values(result, "main_ingredient"))
    methods = set(_facet_values(result, "method"))
    penalty = 0.0
    if main and any(main.intersection(_facet_values(item, "main_ingredient")) for item in selected):
        penalty += 0.18
    if methods and any(methods.intersection(_facet_values(item, "method")) for item in selected):
        penalty += 0.06
    return penalty


def _rank_menu_candidates(
    candidates: list[dict],
    request: SearchRequest,
    role: str,
    selected: list[dict],
) -> list[dict]:
    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda pair: (
            -(
                1.0
                - pair[0] * 0.11
                + _menu_fit_score(pair[1], request, role)
                - _menu_overlap_penalty(pair[1], selected)
            ),
            pair[0],
        )
    )
    return [item for _, item in indexed]


def _base_terms(
    request: SearchRequest,
    *,
    lang: str,
    include_soft_preferences: bool = True,
) -> list[str]:
    """只使用结构化字段，不复用含"四菜一汤/只有一个朋友"的自由 query。"""
    values = (
        request.dishes + request.ingredients + request.cuisines + request.flavors
        + request.methods + request.scenes
        + [meal for meal in request.meals if str(meal).lower() not in _NON_DISH_MEALS]
        + request.dietary_constraints
        + (request.soft_preferences[:2] if include_soft_preferences else [])
    )
    terms: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if text and not text.startswith("现有食材:") and text not in terms:
            terms.append(text)
    return terms


def _group_query(request: SearchRequest, role: str, scoped: dict | None, lang: str) -> str:
    # 局部菜位只消费对应来宾偏好；账号全局软喜好留给其它菜位。
    terms = _base_terms(
        request,
        lang=lang,
        include_soft_preferences=role != "scoped_dish",
    )
    if role == "soup":
        prefix = ["soup"] if lang == "en" else ["汤"]
    elif role == "required_dish":
        prefix = [
            str((scoped or {}).get("required_ingredient") or ""),
            "main course" if lang == "en" else "正餐",
        ]
    elif role == "scoped_dish":
        prefix = [
            str(item) for key in ("cuisines", "flavors") for item in (scoped or {}).get(key) or []
        ]
        prefix.append("main course" if lang == "en" else "正餐")
    else:
        prefix = ["main course home cooking" if lang == "en" else "家常 正餐"]
    out: list[str] = []
    for term in prefix + terms:
        if term and term not in out:
            out.append(term)
    return " ".join(out).strip()


def _pick(
    candidates: list[dict],
    *,
    count: int,
    role: str,
    request: SearchRequest,
    selected_context: list[dict],
    seen_ids: set[str],
    seen_names: set[str],
    excluded_ids: set[str],
    scoped: dict | None = None,
    all_scoped: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    eligible: list[dict] = []
    for raw in candidates:
        recipe_id = _recipe_id(raw)
        name_key = _canonical_name(raw)
        if not recipe_id or not name_key or recipe_id in excluded_ids:
            continue
        if role == "soup":
            type_ok = _is_verified_soup(raw)
        else:
            type_ok = _is_verified_dish(raw)
        if not type_ok:
            continue
        if (
            role == "required_dish"
            and scoped
            and not _matches_required_ingredient(
                raw,
                str(scoped.get("required_ingredient") or ""),
            )
        ):
            continue
        if role == "scoped_dish" and scoped and not _matches_scoped_preference(raw, scoped):
            continue
        # "只有一个朋友偏辣"只影响一个照顾菜位；其余菜位不能继续被这组
        # 局部标签占满，否则作用域又被悄悄放大成全桌。
        if role != "scoped_dish" and all_scoped and _has_any_scoped_evidence(raw, all_scoped):
            continue

        candidate = dict(raw)
        candidate["menu_role"] = role
        if role == "required_dish" and scoped:
            candidate["menu_requirement"] = str(
                scoped.get("required_ingredient") or ""
            )
        eligible.append(candidate)
    eligible = _rank_menu_candidates(
        eligible,
        request,
        role,
        selected_context,
    )
    # 汤位按置信度降序排列：facet + 名称双重命中的优先选中
    if role == "soup":
        eligible.sort(key=_soup_confidence, reverse=True)

    selected: list[dict] = []
    for candidate in eligible:
        recipe_id = _recipe_id(candidate)
        name_key = _canonical_name(candidate)
        if recipe_id in seen_ids or name_key in seen_names or len(selected) >= count:
            continue
        selected.append(candidate)
        seen_ids.add(recipe_id)
        seen_names.add(name_key)
    return selected, eligible


def validate_menu_plan(results: list[dict], request: SearchRequest) -> dict:
    """最终汇总复验；任何失败都只能返回 incomplete，不能让模型补菜。"""
    errors: list[str] = []
    dish_results = [
        item
        for item in results
        if item.get("menu_role") in {"dish", "scoped_dish", "required_dish"}
    ]
    soup_results = [item for item in results if item.get("menu_role") == "soup"]
    ids = [_recipe_id(item) for item in results]
    names = [_canonical_name(item) for item in results]

    if len(dish_results) != request.menu_dish_count:
        errors.append("dish_count_mismatch")
    if len(soup_results) != request.menu_soup_count:
        errors.append("soup_count_mismatch")
    if any(not _is_verified_dish(item) for item in dish_results):
        errors.append("dish_type_mismatch")
    if any(not _is_verified_soup(item) for item in soup_results):
        errors.append("soup_type_mismatch")
    if any(
        not any(_matches_required_ingredient(item, required) for item in dish_results)
        for required in request.required_ingredients
    ):
        errors.append("required_ingredient_missing")
    if not all(ids) or len(ids) != len(set(ids)):
        errors.append("duplicate_or_missing_recipe_id")
    if not all(names) or len(names) != len(set(names)):
        errors.append("duplicate_or_missing_recipe_name")

    checked = filter_hard_constraint_violations({"results": results}, request)
    if len(checked.get("results") or []) != len(results):
        errors.append("hard_constraint_violation")

    scoped_count = sum(item.get("menu_role") == "scoped_dish" for item in results)
    allowed_scoped = sum(
        max(0, int(scoped.get("max_slots") or 1)) for scoped in request.scoped_preferences
    )
    if scoped_count > allowed_scoped:
        errors.append("scoped_preference_over_applied")

    return {
        "complete": not errors,
        "errors": errors,
        "requested": {
            "dish": request.menu_dish_count,
            "soup": request.menu_soup_count,
        },
        "fulfilled": {"dish": len(dish_results), "soup": len(soup_results)},
        "missing": {
            "dish": max(0, request.menu_dish_count - len(dish_results)),
            "soup": max(0, request.menu_soup_count - len(soup_results)),
        },
    }


async def build_menu_plan(
    request: SearchRequest,
    *,
    lang: str = "zh",
    on_search_start: Callable[[str, str], Awaitable[None]] | None = None,
    search: SearchCallable | None = None,
    excluded_recipe_ids: set[str] | None = None,
) -> dict:
    """按局部菜位、普通菜位、汤位顺序检索并确定性分配。"""
    if not request.is_menu_plan:
        raise ValueError("menu counts are required")
    search = search or (lambda query, top_k, language: _run_search_subprocess(
        query, top_k=top_k, lang=language,
    ))
    excluded_ids = {str(item) for item in (excluded_recipe_ids or set()) if str(item)}
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    selected: list[dict] = []
    candidate_pool: list[dict] = []
    queries: list[dict] = []
    search_errors: list[str] = []

    required_ingredients = list(
        dict.fromkeys(
            str(item).strip()
            for item in request.required_ingredients
            if str(item).strip()
        )
    )[:request.menu_dish_count]
    required_slots = len(required_ingredients)
    scoped_slots = min(
        max(0, request.menu_dish_count - required_slots),
        sum(max(0, int(item.get("max_slots") or 1)) for item in request.scoped_preferences),
    )
    groups: list[tuple[str, int, dict | None]] = []
    for required in required_ingredients:
        groups.append((
            "required_dish",
            1,
            {"required_ingredient": required},
        ))
    if scoped_slots:
        # MVP 支持一个明确来宾子集；更多子集会按顺序各占自己的配额。
        remaining = scoped_slots
        for scoped in request.scoped_preferences:
            count = min(remaining, max(0, int(scoped.get("max_slots") or 1)))
            if count:
                groups.append(("scoped_dish", count, scoped))
                remaining -= count
            if remaining <= 0:
                break
    groups.append((
        "dish",
        request.menu_dish_count - required_slots - scoped_slots,
        None,
    ))
    groups.append(("soup", request.menu_soup_count, None))

    for role, count, scoped in groups:
        if count <= 0:
            continue
        query = _group_query(request, role, scoped, lang)
        queries.append({"role": role, "query": query, "requested": count})
        if on_search_start is not None:
            await on_search_start(query, lang)
        raw = await search(query, min(20, max(12, count * 5)), lang)
        if not raw or not raw.get("success"):
            search_errors.append(f"{role}_search_failed")
            continue
        raw = filter_hard_constraint_violations(raw, request)
        candidates = [item for item in (raw.get("results") or []) if _is_same_language(item, lang)]
        picked, eligible = _pick(
            candidates,
            count=count,
            role=role,
            request=request,
            selected_context=selected,
            seen_ids=seen_ids,
            seen_names=seen_names,
            excluded_ids=excluded_ids,
            scoped=scoped,
            all_scoped=request.scoped_preferences,
        )
        selected.extend(picked)
        candidate_pool.extend(eligible)

    validation = validate_menu_plan(selected, request)
    if search_errors:
        validation["errors"] = list(dict.fromkeys(validation["errors"] + search_errors))
        validation["complete"] = False
    return {
        "success": True,
        "results": selected,
        "_candidate_pool": candidate_pool,
        "_display_limit": request.result_limit,
        "_search_request": request.public_dict(),
        "_menu_plan": {
            **validation,
            "queries": queries,
            "scoped_preferences": request.scoped_preferences,
            "excluded_recipe_ids": sorted(excluded_ids),
        },
        "_grounded_output": True,
    }


async def replace_menu_slot(
    request: SearchRequest,
    current_result: dict,
    slot_index: int,
    *,
    lang: str = "zh",
    on_search_start: Callable[[str, str], Awaitable[None]] | None = None,
    search: SearchCallable | None = None,
) -> tuple[dict, bool]:
    """只替换用户点名的菜单槽位，保留其它真实菜和全部任务约束。"""
    current = list((current_result or {}).get("results") or [])
    if slot_index < 0 or slot_index >= len(current):
        return current_result, False
    target = current[slot_index]
    role = str(target.get("menu_role") or "dish")
    if role == "required_dish":
        scoped = {
            "required_ingredient": str(target.get("menu_requirement") or ""),
        }
    else:
        scoped = (
            request.scoped_preferences[0]
            if role == "scoped_dish" and request.scoped_preferences
            else None
        )
    query = _group_query(request, role, scoped, lang)
    if on_search_start is not None:
        await on_search_start(query, lang)
    search = search or (lambda value, top_k, language: _run_search_subprocess(
        value, top_k=top_k, lang=language,
    ))
    raw = await search(query, 20, lang)
    if not raw or not raw.get("success"):
        return current_result, False
    raw = filter_hard_constraint_violations(raw, request)
    candidates = [item for item in (raw.get("results") or []) if _is_same_language(item, lang)]

    others = [item for index, item in enumerate(current) if index != slot_index]
    seen_ids = {_recipe_id(item) for item in others if _recipe_id(item)}
    seen_names = {_canonical_name(item) for item in others if _canonical_name(item)}
    shown_ids = {_recipe_id(item) for item in current if _recipe_id(item)}
    shown_ids.update(
        str(item)
        for item in ((current_result or {}).get("_menu_plan") or {}).get("excluded_recipe_ids") or []
        if str(item)
    )
    picked, eligible = _pick(
        candidates,
        count=1,
        role=role,
        request=request,
        selected_context=others,
        seen_ids=seen_ids,
        seen_names=seen_names,
        excluded_ids=shown_ids,
        scoped=scoped,
        all_scoped=request.scoped_preferences,
    )
    if not picked:
        return current_result, False

    replaced = list(current)
    replaced[slot_index] = picked[0]
    validation = validate_menu_plan(replaced, request)
    if not validation.get("complete"):
        return current_result, False
    previous_menu = dict((current_result or {}).get("_menu_plan") or {})
    return {
        **dict(current_result or {}),
        "results": replaced,
        "_candidate_pool": list((current_result or {}).get("_candidate_pool") or []) + eligible,
        "_menu_plan": {
            **previous_menu,
            **validation,
            "last_replaced_slot": slot_index + 1,
        },
        "_grounded_output": True,
    }, True
