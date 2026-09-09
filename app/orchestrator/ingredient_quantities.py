"""从菜谱原始食材文本中恢复结构化名称、用量和单位。

检索 metadata 同时保存了两套食材：
- ingredients: 为召回清洗过的纯食材名；
- ingredients_raw: 入库前的原始文本，通常包含重量。

菜谱详情必须优先使用 ingredients_raw。只有原文确实没有用量时，才允许
上层调用模型补齐；不能把召回用的纯名称误当成原始详情。
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any


_AMOUNT = (
    r"(?:\d+\s+\d+\s*/\s*\d+"
    r"|\d+\s*[⅛¼⅓⅜½⅝⅔¾⅞]"
    r"|\d+(?:[.,]\d+)?(?:\s*(?:-|–|—|~|至|to)\s*\d+(?:[.,]\d+)?)?"
    r"|\d+\s*/\s*\d+(?:st|nd|rd|th)?|[⅛¼⅓⅜½⅝⅔¾⅞])"
)
_UNITS = (
    "tablespoons?|tbsps?|teaspoons?|tsps?|millilit(?:er|re)s?|ml|"
    "kilograms?|kg|grams?|gm|g|lit(?:er|re)s?|litres?|l|cm|mm|"
    "ounces?|oz|pounds?|lbs?|cups?|pieces?|pcs?|slices?|cloves?|"
    "sprigs?|bunch(?:es)?|cans?|bottles?|pinch(?:es)?|dash(?:es)?|"
    "packets?|pcks?|inch(?:es)?|strands?|leaves|petals?|drops?|"
    "small\\s+pinch|"
    "千克|公斤|毫升|大勺|小勺|茶匙|汤匙|克|斤|两|升|个|只|条|"
    "根|块|片|张|勺|匙|杯|瓣|棵|颗|粒|朵|段|把|撮|滴|包|盒|袋|罐|碗|"
    "кг|мл|г|ч\\.\\s*л\\.|un"
)
_PREFIX = re.compile(
    rf"^\s*(?P<amount>{_AMOUNT})\s*(?P<unit>{_UNITS})\s+(?P<name>.+?)\s*$",
    re.IGNORECASE,
)
_PREFIX_COMPACT = re.compile(
    rf"^\s*(?P<amount>{_AMOUNT})\s*(?P<unit>{_UNITS})"
    rf"\s*(?P<name>[^\d].*?)\s*$",
    re.IGNORECASE,
)
_SUFFIX = re.compile(
    rf"^\s*(?P<name>.+?)\s*(?:[-–—:：]\s*)?"
    rf"(?P<amount>{_AMOUNT})\s*-?\s*(?P<unit>{_UNITS})\s*$",
    re.IGNORECASE,
)
_COUNT_PREFIX = re.compile(
    rf"^\s*(?P<amount>{_AMOUNT})\s+(?P<name>[^\d].+?)\s*$",
    re.IGNORECASE,
)
_MIDDLE = re.compile(
    rf"^\s*(?P<name>.+?)\s+(?P<amount>{_AMOUNT})\s*-?\s*(?P<unit>{_UNITS})"
    r"(?:\s*[,，]\s*|\s+)(?P<remark>.+?)\s*$",
    re.IGNORECASE,
)
_COUNT_DIMENSION = re.compile(
    rf"^\s*(?P<name>.+?[^\d\s])\s+(?P<amount>{_AMOUNT})\s*\*\s*"
    rf"(?P<size>{_AMOUNT})\s*-?\s*(?P<unit>inch(?:es)?|cm|mm)\s*$",
    re.IGNORECASE,
)
_NAME_COUNT_REMARK = re.compile(
    rf"^\s*(?P<name>.+?[^\d\s])\s+(?:up\s*to\s+)?"
    rf"(?P<amount>{_AMOUNT})(?:\s*[,，]\s*|\s+)(?P<remark>.+?)\s*$",
    re.IGNORECASE,
)
_OF_COUNT = re.compile(
    rf"^\s*(?P<prefix>.+?\bof)\s+(?P<amount>{_AMOUNT})\s+"
    r"(?P<name>[^,，]+?)(?P<remark>\s*[,，].+)?\s*$",
    re.IGNORECASE,
)
_OR_COUNT = re.compile(
    rf"^\s*or\s+(?P<amount>{_AMOUNT})\s+"
    r"(?:(?P<unit>packets?|pcks?|drops?|pieces?|pcs?)\s+(?:of\s+)?)?"
    r"(?P<name>.+?)\s*$",
    re.IGNORECASE,
)
_A_PINCH = re.compile(
    r"^\s*(?:a|one)\s+(?P<unit>pinch|dash)\s+of\s+(?P<name>.+?)\s*$",
    re.IGNORECASE,
)
_A_LOOSE_MEASURE = re.compile(
    r"^\s*(?:a|one)\s+(?P<unit>splash|handful)\s+(?:of\s+)?"
    r"(?P<name>.+?)\s*$",
    re.IGNORECASE,
)
_NAME_WITH_BARE_AMOUNT = re.compile(
    rf"^\s*(?P<name>.*?[^\d\s])\s+(?P<amount>{_AMOUNT})\s*$",
    re.IGNORECASE,
)
_NAME_AMOUNT_UNIT_REMARK = re.compile(
    rf"^\s*(?P<name>.+?[^\d\s])\s*(?P<amount>{_AMOUNT})\s*"
    rf"(?P<unit>{_UNITS})(?P<remark>\s*[,，(（].+?\s*)$",
    re.IGNORECASE,
)
_COUNT_PREFIX_COMPACT = re.compile(
    rf"^\s*(?P<amount>{_AMOUNT})(?P<name>[A-Za-z\u3400-\u9fff].+?)\s*$",
    re.IGNORECASE,
)
_PURE_NUMBER = re.compile(rf"^\s*{_AMOUNT}\s*$")
_PURE_MEASURE = re.compile(
    rf"^\s*{_AMOUNT}\s*(?:{_UNITS})\s*$",
    re.IGNORECASE,
)
_MEANINGLESS = re.compile(r"^[.\-–—…]+$")
_SECTION_HEADERS = {
    "presentation", "crust", "filling", "seasoning", "topping", "sauce",
    "salad", "the salad", "for garnish", "to garnish", "for gravy",
    "for tempering", "serves", "辅料", "调料", "主料",
}
_SECTION_LINE = re.compile(
    r"^\s*(?:serves?|for\s+(?:marinade|dry powder|garnish|gravy|tempering))"
    r"(?:\s|:|：|\(|$)",
    re.IGNORECASE,
)
_NON_INGREDIENT_NOTE = re.compile(
    r"^\s*(?:(?:skip|use)\b.*(?:fasting|recipe).*|string|.*vacuum bag.*)\s*$",
    re.IGNORECASE,
)
_UNICODE_FRACTIONS = {
    "⅛": 0.125,
    "¼": 0.25,
    "⅓": round(1 / 3, 3),
    "⅜": 0.375,
    "½": 0.5,
    "⅝": 0.625,
    "⅔": round(2 / 3, 3),
    "¾": 0.75,
    "⅞": 0.875,
}


def _amount(value: str) -> int | float | str:
    text = re.sub(r"\s+", "", str(value or "")).replace(",", ".")
    text = re.sub(r"(?<=/\d)(?:st|nd|rd|th)$", "", text, flags=re.IGNORECASE)
    mixed_unicode = re.fullmatch(r"(\d+)([⅛¼⅓⅜½⅝⅔¾⅞])", text)
    if mixed_unicode:
        return int(mixed_unicode.group(1)) + _UNICODE_FRACTIONS[
            mixed_unicode.group(2)
        ]
    if text in _UNICODE_FRACTIONS:
        return _UNICODE_FRACTIONS[text]
    if any(marker in text for marker in ("-", "–", "—", "~", "至", "to")):
        return re.sub(r"[–—~]|至|to", "-", text)
    if "/" in text:
        try:
            mixed = re.fullmatch(r"(\d+)(\d+/\d+)", text)
            number = (
                float(mixed.group(1)) + float(Fraction(mixed.group(2)))
                if mixed else float(Fraction(text))
            )
            return int(number) if number.is_integer() else round(number, 3)
        except (ValueError, ZeroDivisionError):
            return text
    number = float(text)
    return int(number) if number.is_integer() else number


def _clean_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(
        " \t\r\n,，;；:-–—"
    )


def _structured(raw: str, match: re.Match) -> dict:
    return {
        "group": "main",
        "name": _clean_name(match.group("name")),
        "amount": _amount(match.group("amount")),
        "unit": re.sub(r"\s+", " ", match.group("unit")).strip(),
        "remark": None,
        "quantity_source": "ingredients_raw",
        "source_text": raw,
    }


def parse_grounded_ingredient(raw: Any) -> tuple[dict | None, str | None]:
    """解析一条原始食材，返回 ``(有依据的结构, 待模型补量的名称)``。"""
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    # 少量原始数据把 ``100gm`` 错写成 ``10 0gm``，这里仅在数字紧邻重量
    # 单位时合并空格，避免把份量 2 和尺寸 10 等独立数字误合并。
    text = re.sub(
        r"(?<=\d)\s+(?=\d\s*(?:gm|g|kg|ml|mg)\b)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if (
        not text
        or _PURE_NUMBER.fullmatch(text)
        or _PURE_MEASURE.fullmatch(text)
        or _MEANINGLESS.fullmatch(text)
    ):
        return None, None
    if text.rstrip(":：").strip().casefold() in _SECTION_HEADERS:
        return None, None
    if text.endswith((":", "：")) and not re.search(r"\d", text):
        return None, None
    if _SECTION_LINE.match(text):
        return None, None
    if _NON_INGREDIENT_NOTE.match(text):
        return None, None

    pinch = _A_PINCH.match(text)
    if pinch:
        return {
            "group": "main",
            "name": _clean_name(pinch.group("name")),
            "amount": 1,
            "unit": pinch.group("unit").lower(),
            "remark": None,
            "quantity_source": "ingredients_raw",
            "source_text": text,
        }, None

    loose = _A_LOOSE_MEASURE.match(text)
    if loose:
        return {
            "group": "main",
            "name": _clean_name(loose.group("name")),
            "amount": 1,
            "unit": loose.group("unit").lower(),
            "remark": None,
            "quantity_source": "ingredients_raw",
            "source_text": text,
        }, None

    of_count = _OF_COUNT.match(text)
    if of_count:
        name = _clean_name(
            f"{of_count.group('prefix')} {of_count.group('name')}"
        )
        return {
            "group": "main",
            "name": name,
            "amount": _amount(of_count.group("amount")),
            "unit": "pcs",
            "remark": _clean_name(of_count.group("remark")) or None,
            "quantity_source": "ingredients_raw",
            "source_text": text,
        }, None

    or_count = _OR_COUNT.match(text)
    if or_count:
        return {
            "group": "main",
            "name": _clean_name(or_count.group("name")),
            "amount": _amount(or_count.group("amount")),
            "unit": _clean_name(or_count.group("unit")) or "pcs",
            "remark": "alternative",
            "quantity_source": "ingredients_raw",
            "source_text": text,
        }, None

    name_amount_remark = _NAME_AMOUNT_UNIT_REMARK.match(text)
    if name_amount_remark:
        return {
            "group": "main",
            "name": _clean_name(name_amount_remark.group("name")),
            "amount": _amount(name_amount_remark.group("amount")),
            "unit": re.sub(
                r"\s+", " ", name_amount_remark.group("unit")
            ).strip(),
            "remark": _clean_name(name_amount_remark.group("remark")) or None,
            "quantity_source": "ingredients_raw",
            "source_text": text,
        }, None

    dimension = _COUNT_DIMENSION.match(text)
    if dimension:
        return {
            "group": "main",
            "name": _clean_name(dimension.group("name")),
            "amount": _amount(dimension.group("amount")),
            "unit": "pcs",
            "remark": (
                f"{_amount(dimension.group('size'))} "
                f"{dimension.group('unit')} each"
            ),
            "quantity_source": "ingredients_raw",
            "source_text": text,
        }, None

    for pattern in (_PREFIX, _PREFIX_COMPACT, _SUFFIX, _MIDDLE):
        match = pattern.match(text)
        if match:
            if (
                pattern is _MIDDLE
                and str(match.group("unit") or "").casefold() in {"cm", "mm"}
                and str(match.group("name") or "").rstrip().endswith("(")
            ):
                continue
            item = _structured(text, match)
            if item.get("amount") in (0, "0"):
                return None, str(item.get("name") or "").strip() or None
            if pattern is _MIDDLE:
                remark = _clean_name(match.group("remark"))
                alternative = re.match(
                    r"^(?P<name>.+?)\s+(?P<count>\d+\s+\w+)\s*/$",
                    str(item.get("name") or ""),
                    re.IGNORECASE,
                )
                if alternative:
                    item["name"] = _clean_name(alternative.group("name"))
                    remark = (
                        f"{alternative.group('count')} alternative; {remark}"
                    )
                item["remark"] = remark
            return (item, None) if item["name"] else (None, None)

    compact_count = _COUNT_PREFIX_COMPACT.match(text)
    if compact_count:
        return {
            "group": "main",
            "name": _clean_name(compact_count.group("name")),
            "amount": _amount(compact_count.group("amount")),
            "unit": "pcs",
            "remark": None,
            "quantity_source": "ingredients_raw",
            "source_text": text,
        }, None

    name_count = _NAME_COUNT_REMARK.match(text)
    if (
        name_count
        and not re.search(r"[\u3400-\u9fff]", text)
        and not re.search(r"(?:℃|°[CF]?)", name_count.group("name"), re.IGNORECASE)
        and not re.match(
            r"(?:mins?|minutes?|secs?|seconds?|hours?|hrs?)\b",
            str(name_count.group("remark") or ""),
            re.IGNORECASE,
        )
    ):
        return {
            "group": "main",
            "name": _clean_name(name_count.group("name")),
            "amount": _amount(name_count.group("amount")),
            "unit": "pcs",
            "remark": _clean_name(name_count.group("remark")) or None,
            "quantity_source": "ingredients_raw",
            "source_text": text,
        }, None

    # “2 egg whites / 1 star anise”有明确数量但没有显式单位，按件数保存。
    count = _COUNT_PREFIX.match(text)
    if count:
        name = _clean_name(count.group("name"))
        if name:
            return {
                "group": "main",
                "name": name,
                "amount": _amount(count.group("amount")),
                "unit": "pcs",
                "remark": None,
                "quantity_source": "ingredients_raw",
                "source_text": text,
            }, None

    # “香蕉 150”只有数值没有单位，交给模型补单位，不擅自判定 g/ml。
    bare = _NAME_WITH_BARE_AMOUNT.match(text)
    if bare:
        amount = _amount(bare.group("amount"))
        if amount in (0, "0"):
            return None, _clean_name(bare.group("name"))
        if not re.search(r"[\u3400-\u9fff]", text):
            return {
                "group": "main",
                "name": _clean_name(bare.group("name")),
                "amount": amount,
                "unit": "pcs",
                "remark": None,
                "quantity_source": "ingredients_raw",
                "source_text": text,
            }, None
        return None, _clean_name(bare.group("name"))

    return None, _clean_name(text)


def build_grounded_ingredients(
    source: Any,
) -> tuple[list[dict], list[str]]:
    """批量解析并去重，保持原始顺序。"""
    values = source if isinstance(source, list) else [source]
    parsed: list[dict] = []
    unresolved: list[str] = []
    seen_structured: set[tuple[str, str, str]] = set()
    seen_unresolved: set[str] = set()

    for raw in values:
        item, missing_name = parse_grounded_ingredient(raw)
        if item:
            key = (
                str(item.get("name") or "").casefold(),
                str(item.get("amount") or ""),
                str(item.get("unit") or "").casefold(),
            )
            if key not in seen_structured:
                seen_structured.add(key)
                parsed.append(item)
        elif missing_name and missing_name.casefold() not in seen_unresolved:
            seen_unresolved.add(missing_name.casefold())
            unresolved.append(missing_name)
    return parsed, unresolved
