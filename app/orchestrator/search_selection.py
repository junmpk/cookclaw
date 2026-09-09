"""检索候选的最终选择：在相关性顺序上做轻量去重和多样性控制。"""
from __future__ import annotations

import re


_DRINK_MARKERS = {
    "饮品", "饮料", "果汁", "蔬果汁", "水果汁", "茶饮", "奶茶", "咖啡",
    "奶昔", "豆浆", "beverage", "drink", "drinks", "juice", "smoothie",
    "tea", "coffee",
}
_DRINK_REQUEST_MARKERS = (
    "喝什么", "喝点", "想喝", "饮品", "饮料", "果汁", "蔬果汁", "茶饮",
    "奶茶", "咖啡", "奶昔", "豆浆",
    "what to drink", "something to drink", "beverage", "drink", "juice",
    "smoothie", "tea", "coffee",
)


def _as_values(value) -> list[str]:
    if isinstance(value, dict):
        value = list(value.values())
    elif isinstance(value, str):
        value = re.split(r"[、,，;/；|]", value)
    return [
        re.sub(r"\s+", " ", str(item or "")).strip().lower()
        for item in (value or [])
        if str(item or "").strip()
    ]


def explicitly_requests_dishes(text: str) -> bool:
    """用户明确说“菜/几道菜”时，将结果类型限定为菜肴。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not value or any(marker in value for marker in _DRINK_REQUEST_MARKERS):
        return False
    return bool(
        re.search(
            r"菜(?!谱|单|系|市场|籽)|几\s*道|"
            r"\b(?:dish|dishes|savou?ry dishes|courses?)\b",
            value,
            flags=re.IGNORECASE,
        )
    )


def _is_drink_result(result: dict) -> bool:
    metadata = result.get("metadata") or {}
    facets = metadata.get("facets") or {}
    meals = _as_values(facets.get("meal")) if isinstance(facets, dict) else []
    tags = _as_values(metadata.get("tags"))
    name = str(metadata.get("name") or "").strip().lower()
    values = [*meals, *tags]
    if any(value in _DRINK_MARKERS for value in values):
        return True
    if any(
        marker in value
        for value in values
        for marker in _DRINK_MARKERS
        if re.search(r"[\u4e00-\u9fff]", marker)
    ):
        return True
    if any(marker in name for marker in _DRINK_MARKERS if re.search(r"[\u4e00-\u9fff]", marker)):
        return True
    return bool(re.search(
        r"\b(?:beverage|drink|juice|smoothie|tea|coffee)\b",
        name,
        flags=re.IGNORECASE,
    ))


def filter_requested_result_type(results: list[dict], request) -> list[dict]:
    """明确要菜时排除有真实饮品证据的结果；未明确类型时保持召回原序。"""
    original_text = str(getattr(request, "original_text", "") or "")
    if not explicitly_requests_dishes(original_text):
        return list(results or [])
    return [result for result in (results or []) if not _is_drink_result(result)]


def _canonical_dish_name(value: str) -> str:
    text = re.sub(r"[\s，。；！？、,;!?]+", "", str(value or "").lower())
    return text.replace("西红柿", "番茄").replace("鸡蛋", "蛋")


def dedupe_results(results: list[dict]) -> list[dict]:
    """按真实 ID 与规范化菜名双重去重，保留排序最靠前的一条。"""
    out: list[dict] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for result in results or []:
        metadata = result.get("metadata") or {}
        recipe_id = str(metadata.get("recipe_id") or result.get("id") or "").strip()
        name = _canonical_dish_name(metadata.get("name") or "")
        if not recipe_id or not name or recipe_id in seen_ids or name in seen_names:
            continue
        seen_ids.add(recipe_id)
        seen_names.add(name)
        out.append(result)
    return out


def prioritize_exact_matches(results: list[dict], dishes: list[str]) -> list[dict]:
    """明确菜名命中优先，避免被用户画像或多样性策略挤出前三。"""
    targets = [_canonical_dish_name(item) for item in dishes if str(item).strip()]
    if not targets:
        return list(results or [])
    exact, rest = [], []
    for result in results or []:
        name = _canonical_dish_name((result.get("metadata") or {}).get("name") or "")
        matched = any(target == name or target in name or name in target for target in targets if target)
        (exact if matched else rest).append(result)
    return exact + rest


def _soft_preference_penalty(result: dict, avoid: list[str]) -> int:
    """把“不太辣/少油/别复杂”等软偏好用于降序，不把它们当硬忌口删除。"""
    if not avoid:
        return 0
    metadata = result.get("metadata") or {}
    haystack = str({
        "name": metadata.get("name"),
        "tags": metadata.get("tags"),
        "facets": metadata.get("facets"),
        "description": metadata.get("description"),
        "recipe_detail": metadata.get("recipe_detail"),
    }).lower()
    penalty = 0
    for raw in avoid:
        term = str(raw or "").strip().lower()
        if term in {"辣", "太辣", "spicy", "too spicy"}:
            if any(marker in haystack for marker in (
                "麻辣", "香辣", "重辣", "特辣", "hot and spicy", "very spicy",
            )):
                penalty += 2
            elif "辣" in haystack or "spicy" in haystack:
                penalty += 1
        elif term in {"油", "太油", "油腻", "太油腻", "oily", "too oily"}:
            if any(marker in haystack for marker in ("油炸", "炸制", "deep-fried", "deep fried")):
                penalty += 2
            elif any(marker in haystack for marker in ("煎", "炸", "油", "fried")):
                penalty += 1
        elif term in {"甜", "太甜", "sweet", "too sweet"}:
            if "甜" in haystack or "sweet" in haystack:
                penalty += 1
        elif term in {"咸", "太咸", "salty", "too salty", "重口", "重口味"}:
            if any(marker in haystack for marker in ("咸", "重口", "salty")):
                penalty += 1
        elif term in {"复杂", "麻烦", "complex", "complicated"}:
            detail = metadata.get("recipe_detail") or {}
            steps = detail.get("steps") if isinstance(detail, dict) else None
            if isinstance(steps, list) and len(steps) >= 8:
                penalty += 1
    return penalty


def prioritize_soft_preferences(results: list[dict], request) -> list[dict]:
    """在保持真实召回顺序的基础上，把明显违背软偏好的候选向后放。"""
    avoid = list(getattr(request, "avoid", None) or [])
    indexed = list(enumerate(results or []))
    indexed.sort(key=lambda pair: (_soft_preference_penalty(pair[1], avoid), pair[0]))
    return [result for _, result in indexed]


def _canonical_food(value: object) -> str:
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())
    return (
        text.replace("西红柿", "番茄")
        .replace("鸡蛋", "蛋")
        .replace("马铃薯", "土豆")
    )


def _food_evidence(result: dict) -> list[str]:
    metadata = result.get("metadata") or {}
    facets = metadata.get("facets") or {}
    values: list[object] = [
        metadata.get("name"),
        metadata.get("description"),
        metadata.get("ingredients"),
        facets.get("main_ingredient") if isinstance(facets, dict) else None,
    ]
    out: list[str] = []
    for value in values:
        if isinstance(value, dict):
            nested = list(value.values())
        elif isinstance(value, (list, tuple, set)):
            nested = list(value)
        else:
            nested = [value]
        for item in nested:
            if isinstance(item, dict):
                item = next(
                    (
                        item.get(key)
                        for key in ("name", "ingredient_name", "ingredientName", "food")
                        if item.get(key)
                    ),
                    item,
                )
            normalized = _canonical_food(item)
            if normalized and normalized not in out:
                out.append(normalized)
    return out


def prioritize_ingredient_coverage(results: list[dict], request) -> list[dict]:
    """多食材追问先排同时命中的菜，再排只命中部分食材的真实候选。"""
    requested_pairs = [
        (str(item).strip(), _canonical_food(item))
        for item in (getattr(request, "ingredients", None) or [])
        if _canonical_food(item)
    ]
    requested_pairs = list(dict.fromkeys(requested_pairs))
    if len(requested_pairs) < 2:
        return list(results or [])

    ranked: list[tuple[int, int, dict]] = []
    for index, raw in enumerate(results or []):
        evidence = _food_evidence(raw)
        matched = [
            label
            for label, wanted in requested_pairs
            if any(
                wanted == actual or wanted in actual or actual in wanted
                for actual in evidence
            )
        ]
        requested = [label for label, _ in requested_pairs]
        candidate = dict(raw)
        candidate["_selection_priority"] = len(matched)
        candidate["_ingredient_match"] = {
            "requested": list(requested),
            "matched": matched,
            "missing": [item for item in requested if item not in matched],
            "match_type": "all" if len(matched) == len(requested) else "partial",
        }
        ranked.append((len(matched), index, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in ranked]


def _tokens(result: dict) -> set[str]:
    metadata = result.get("metadata") or {}
    values = []
    facets = metadata.get("facets") or {}
    if isinstance(facets, dict):
        for key in ("cuisine", "main_ingredient", "flavor", "method", "scene", "meal"):
            raw = facets.get(key) or []
            values.extend(raw if isinstance(raw, list) else [raw])
    tags = metadata.get("tags") or []
    if isinstance(tags, dict):
        tags = list(tags.values())
    elif isinstance(tags, str):
        tags = [tags]
    values.extend(tags[:8])
    name = str(metadata.get("name") or "")
    values.extend(re.findall(r"[一-鿿]{2,4}|[a-zA-Z]{3,}", name.lower()))
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def select_diverse_results(results: list[dict], *, limit: int = 3) -> list[dict]:
    """保持业务优先级与第一名，在同一优先级内做轻量多样性控制。"""
    ranked = dedupe_results(list(results or []))
    if len(ranked) <= limit:
        return ranked
    selected = [ranked[0]]
    selected_indexes = {0}
    token_cache = [_tokens(result) for result in ranked]
    while len(selected) < limit and len(selected_indexes) < len(ranked):
        best_index = None
        best_score = float("-inf")
        remaining_priorities = [
            int(ranked[index].get("_selection_priority") or 0)
            for index in range(len(ranked))
            if index not in selected_indexes
        ]
        current_priority = max(remaining_priorities, default=0)
        for index, _result in enumerate(ranked):
            if index in selected_indexes:
                continue
            if int(ranked[index].get("_selection_priority") or 0) != current_priority:
                continue
            # Rerank 顺序仍占主导；只对高度相似候选施加有限惩罚。
            relevance = 1.0 - index * 0.055
            similarity = max(_overlap(token_cache[index], token_cache[i]) for i in selected_indexes)
            score = relevance - similarity * 0.22
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        selected_indexes.add(best_index)
        selected.append(ranked[best_index])
    return selected
