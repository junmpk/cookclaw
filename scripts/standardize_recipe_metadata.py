#!/usr/bin/env python3
"""标准化菜谱元数据 Excel。

规则：
- 删除明确标记的错误菜谱；
- 缺失食材优先由 Qwen 根据菜名和已有步骤补全，失败时使用本地保守提取；
- 仅补空白的描述、小贴士和调料，不覆盖已有内容；
- 删除营养成分列；
- 输出新工作簿，不覆盖源文件。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from copy import copy
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DELETE_IDS = {
    "1986343766111817729",
    "2049412961731649537",
}

REQUIRED_HEADERS = (
    "id",
    "名称",
    "食材",
    "标签",
    "图片地址",
    "lang",
    "菜谱描述",
    "预计用时",
    "难易程度",
    "小贴士",
    "调料",
    "烹饪步骤",
    "营养成分",
)

SEASONING_TERMS_EN = (
    "salt", "sugar", "syrup", "oil", "soy sauce", "vinegar", "pepper",
    "cooking wine", "wine", "brandy", "sauce", "paste", "mayonnaise",
    "mustard", "paprika", "cumin", "turmeric", "oregano", "thyme",
    "rosemary", "bay leaf", "bay leaves", "cinnamon", "clove", "cloves",
    "nutmeg", "coriander", "chilli", "chili", "curry", "fennel",
    "sesame", "stock paste", "vanilla extract", "lemon juice",
    "lime juice", "garlic powder", "onion powder", "ginger",
)
SEASONING_TERMS_ZH = (
    "盐", "糖", "冰糖", "油", "酱油", "生抽", "老抽", "醋", "胡椒",
    "料酒", "黄酒", "白酒", "红酒", "蚝油", "香油", "芝麻油", "辣椒",
    "花椒", "八角", "桂皮", "香叶", "孜然", "豆瓣酱", "甜面酱",
    "番茄酱", "沙拉酱", "芥末", "鸡精", "味精", "蔬菜精", "五香粉",
    "咖喱", "姜", "蒜", "葱", "香菜", "柠檬汁", "香草精",
)
MAIN_EXCLUDE_EN = (
    *SEASONING_TERMS_EN,
    "water", "drinking water", "boiling water",
)
MAIN_EXCLUDE_ZH = (*SEASONING_TERMS_ZH, "水", "清水", "饮用水", "开水")

METHODS_EN = (
    ("knead", "kneading"),
    ("bake", "baking"),
    ("roast", "roasting"),
    ("steam", "steaming"),
    ("stew", "slow cooking"),
    ("simmer", "simmering"),
    ("saute", "sautéing"),
    ("sauté", "sautéing"),
    ("stir-fry", "stir-frying"),
    ("boil", "boiling"),
    ("blend", "blending"),
    ("chop", "chopping and mixing"),
    ("mix", "mixing"),
)
METHODS_ZH = (
    ("揉面", "揉制"),
    ("烘烤", "烘烤"),
    ("烤", "烘烤"),
    ("蒸", "蒸制"),
    ("炖", "炖煮"),
    ("焖", "焖制"),
    ("煎", "煎制"),
    ("炸", "炸制"),
    ("炒", "炒制"),
    ("煮", "煮制"),
    ("搅拌", "搅拌"),
    ("打碎", "搅打"),
)

COUNTABLE_EN = (
    "egg", "eggs", "onion", "onions", "carrot", "carrots", "apple", "apples",
    "orange", "oranges", "lemon", "lemons", "banana", "bananas", "tomato",
    "tomatoes", "potato", "potatoes", "shallot", "shallots", "chilli",
    "chillies", "chili", "chilies", "asparagus", "oyster", "oysters",
)

LOCAL_INGREDIENT_OVERRIDES: dict[tuple[str, str], str] = {
    ("1769762236360343553", "en"): "30g brown sugar\n25g ginger slices\n20g red dates\n600ml water\n10g black tea leaves",
    ("1821010606307368961", "en"): "120g lard\n2 eggs\n150g plain flour\n300ml milk\n1/2 tsp salt\n1/8 tsp ground white pepper",
    ("1879069608701923329", "en"): "100g softened butter\n50g powdered sugar\n40g cream\n1/4 tsp salt\n160g low-gluten flour\n1 tsp matcha powder\nBiscuit sticks, as needed",
    ("1879070094188417026", "en"): "Asparagus, as needed\n1800ml water\n30ml white vinegar\n2 eggs\nBacon, as needed\n50ml Hollandaise sauce\nChives, as needed\nBlack pepper, to taste",
    ("1879071671053783041", "en"): "200g bacon\n200g chorizo\n1 onion\n3 garlic cloves\n2-4 tsp dried oregano\n3 celery stalks\n2 carrots\n80g tomato paste\n1-2 tbsp chicken stock paste\n1100ml water\n200g pearl barley\nSalt and pepper, to taste\nParsley, for garnish",
    ("1879073511061393409", "en"): "20g blue cheese\n30g sour cream\n75g mayonnaise\n10g parsley\n15g dill\n10g chives\n1 garlic clove\n20ml lemon juice\nIceberg lettuce, as needed\nRed radish, as needed\nCherry tomatoes, as needed\nSalt and white pepper, to taste",
    ("1879073965711364097", "en"): "120g flour\n80g sugar\n70g softened butter\n1 egg, beaten\n100ml milk\nGolden syrup, as needed\n500ml water",
    ("1879074026075787266", "en"): "20ml milk\n1 tsp matcha powder\n100g light cream\n2 tsp caster sugar\n1 tsp condensed milk\n30ml yogurt\nRed beans, as needed\n100g white chocolate",
    ("1879074114110033921", "en"): "200g peeled banana\n150g cream\n45g cheese\n60g yogurt",
    ("1879075701758955521", "en"): "1L water\n300g pumpkin\n250ml milk\n150g egg\n15g corn flour\n30ml condensed milk",
    ("1879077627040960513", "en"): "Kiwifruit, as needed\nApples, as needed\n100g sugar\n20ml lemon juice",
    ("1879079125300875266", "en"): "50g dark chocolate\n60g unsalted butter\n38g egg liquid\n55g caster sugar\n1/4 tsp vanilla extract\n50g high-gluten flour\n100g cream cheese\n1 egg yolk",
    ("1879079740131315714", "en"): "750ml red wine\n45g rock sugar\n4 cloves\n1 cinnamon stick\nApple, as needed\nOrange, as needed\nLemon, as needed\nRosemary, as needed",
    ("1879080829014577154", "en"): "50ml olive oil\n6 garlic cloves\n1 shallot\n50ml drinking water\n125g butter\n90g shrimp\n1 tsp garlic powder\n2 tsp paprika\n1 tsp chilli powder\n20ml lemon juice\n1/2 tsp black pepper\nParsley, for garnish",
    ("1879083283353505793", "en"): "10g dried Japanese kelp\n6 shiitake mushrooms\n300g white radish\n4 slices fresh ginger\n1 fresh red chilli\n2 tbsp sesame oil\n1/4 tsp sea salt\n2 tbsp light soy sauce\n1 tbsp blackstrap molasses\n200ml water\n1 tsp molasses powder\n1 tbsp balsamic vinegar",
    ("1879084201805418497", "en"): "1 tsp dried yeast\n140ml milk\n250g high-gluten flour\n40g caster sugar\n30g egg liquid\n11g cocoa powder\n25g softened butter\n75g chocolate chips\n75g cream\n75g chocolate",
    ("1879084376737255426", "en"): "300ml water\n60g garlic\n20g bell pepper\n1/2 tsp seasoned soy sauce\n1/8 tsp salt\nOysters, as needed\n500ml water for steaming\nChives, for garnish",
    ("1879085107573755906", "en"): "150g beef\n30g butter\n40g celery\n40g carrots\n40g onions\n20g tomatoes\n1 bay leaf\n10ml brandy\n500ml drinking water\n30g tomato sauce",
    ("1879085419156017153", "en"): "1 onion\n2 carrots\n1/2 sweet potato\n2 small celery stalks\n200-300g chicken wings and bones\nRosemary, thyme and parsley, as needed\n3 bay leaves\n5 peppercorns\n1200ml water",
    ("1879085760622694401", "en"): "500ml water\n200g wax gourd\n100g beef\n2 slices ginger\n1/2 tsp salt\nChives, for garnish",
    ("1879086003997184001", "en"): "A pinch of saffron\n20ml boiled water\n150g mayonnaise",
    ("1879086410156806146", "en"): "50g butter\n10g flour\n100ml milk\n25ml lemon juice\n1/2 tsp salt",
    ("1879086537550401538", "en"): "500ml water\n200g asparagus\n20g bell pepper\n15g olive oil\n15g light soy sauce\n1 tsp white vinegar\nCashews, as needed",
    ("1879086624494129153", "en"): "3 eggs, separated\n90g caster sugar\nA few drops of lemon juice\n35ml salad oil\n55ml milk\n1/8 tsp salt\n50g low-gluten flour",
    ("1879086694614503425", "en"): "Red bell pepper, as needed\n500g tomato\n100g sugar\n1 tsp salt\n2 tsp white vinegar\n2 tsp corn starch\n2 tsp water",
    ("1879087762303946754", "en"): "40g unsalted butter\n125ml water\n180g raw cane sugar\n110g milk powder\n1 pinch salt",
    ("1879088506327339010", "en"): "Baguette, as needed\nCherry tomatoes, diced, as needed\n15g pesto",
    ("1879088751618625538", "en"): "25ml olive oil\n10g garlic cloves\n30g brown onion\n30g celery\n350g beef chunks\n250ml red wine\n8g fresh thyme\n1 bay leaf\n20g tomato paste\n400ml beef stock\n100g carrot\n100g potato\n20g shiitake mushrooms\n1 tsp sea salt\n1/4 tsp black pepper",
    ("1902648851390074881", "en"): "Potato gnocchi, as needed\nWater, as needed\nOlive oil, as needed\nOnion, diced, as needed\n1 tbsp tomato paste\n4 sprigs thyme\nCream, as needed\nBaby spinach, as needed\n1 tsp salt\nGround black pepper and pepper flakes, to taste",
    ("1902665238611300354", "en"): "Potato gnocchi, as needed\nWater, as needed\nOlive oil, as needed\nDiced bacon, as needed\nSliced mushrooms, as needed\nPitted black olives, as needed\nCherry tomatoes, as needed\n1/2 tsp salt\nGround black pepper, to taste",
    ("1937386003629780994", "en"): "Biscuits, as needed\n500g strawberries\n150g sugar\nLabneh, as needed\nStrained yogurt, as needed\nMilk, as needed\nLemon peel, as needed",
    ("1937777479668932610", "en"): "Powdered sugar, as needed\nFlour, as needed\nButter, as needed\nYogurt, as needed\nBaking powder, as needed\nWalnuts, as needed\nGinger, as needed\nApple pieces, as needed\nCinnamon powder, as needed",
    ("1945408445107339266", "en"): "Dill, as needed\nLeeks, as needed\n70g water\nLabneh, as needed\nSalt and pepper, to taste\nSalmon, as needed\n500g water for steaming\nPotatoes, as needed",
    ("1945686041741164545", "en"): "Sugar, as needed\nMixed frozen berries, as needed\nLime juice, as needed\nEgg whites, as needed\nMixed fresh berries, for garnish",
    ("2365366770983247874", "en"): "Softened butter, as needed\nPowdered sugar, as needed\nSalt, as needed\nLow-gluten flour, as needed\nHigh-gluten flour, as needed\nStarch, as needed\nMilk powder, as needed",
    ("2369718681688391682", "en"): "1L water\n150g tagliatelle pasta\n50g butter\n50g bacon\n1/2 tsp cheese powder\n1/2 tsp salt\n30ml milk\nEgg, as needed\nGround black pepper, to taste",
    ("2369726765114638337", "en"): "1000ml water\n200g macaroni\n30g butter\n250ml milk\n120g grated cheddar cheese\n120g grated mozzarella cheese\n4g salt\nGround black pepper, to taste",
}

GENERIC_PROGRAM_INGREDIENTS_EN = {
    "Whipping": "Cream or egg whites, as required by the selected recipe",
    "Cook Eggs ( soft/waxy soft/hard)": "Eggs, as needed\nWater, as needed",
    "Ferment": "Prepared dough, batter or cultured mixture, as needed",
    "Puree": "Prepared fruit or vegetables, as needed",
    "Steam": "Ingredients to be steamed, as needed\nWater, as needed",
    "Stir": "Prepared ingredients to be stirred, as needed",
    "Yoghurt": "Milk, as needed\nYogurt starter, as needed",
    "Sous Vide": "Vacuum-sealed ingredients, as needed\nWater, as needed",
    "Reheat": "Prepared food to be reheated, as needed",
    "Fresh Recovery": "Ingredients to be refreshed, as needed\nWater, as needed",
    "Warm": "Prepared food or drink to be kept warm, as needed",
    "Smoothie": "Fruit or vegetables, as needed\nLiquid, as needed\nIce, optional",
    "Stew": "Prepared main ingredients, as needed\nWater or stock, as needed",
    "Clean": "Water, as needed",
    "Peeling": "Potatoes or other suitable root vegetables, as needed",
    "Food Processor": "Ingredients to be sliced or shredded, as needed",
    "Grinding": "Dry ingredients or spices to be ground, as needed",
    "Deep Cleaning": "Water, as needed\nSuitable cleaning agent, as directed",
    "Knead": "Flour, as needed\nWater or other recipe liquid, as needed",
    "Slow Cooking": "Prepared main ingredients, as needed\nCooking liquid, as needed",
    "Sauté": "Prepared ingredients, as needed\nCooking oil, as needed",
}

GENERIC_PROGRAM_INGREDIENTS_ZH = {
    "打发": "淡奶油或蛋清（按实际菜谱需要）",
    "煮蛋": "鸡蛋（按需）\n水（适量）",
    "发酵": "已准备好的面团、面糊或发酵基料（按需）",
    "果泥": "适合制作果泥的水果或蔬菜（按需）",
    "蒸": "待蒸食材（按需）\n水（适量）",
    "搅拌": "待搅拌食材（按需）",
    "酸奶": "牛奶（按需）\n酸奶发酵剂（按需）",
    "低温牛排": "真空包装牛排（按需）\n水（适量）",
    "加热": "待加热的熟食（按需）",
    "复鲜清洗": "待复鲜食材（按需）\n水（适量）",
    "保温": "待保温的熟食或饮品（按需）",
    "冰沙": "水果或蔬菜（按需）\n液体（按需）\n冰块（可选）",
    "炖": "已准备好的主食材（按需）\n水或高汤（适量）",
    "清洁": "水（适量）",
    "削皮": "土豆或其他适合削皮的根茎类食材（按需）",
    "切片/切丝": "待切片或切丝食材（按需）",
    "研磨": "待研磨的干性食材或香料（按需）",
    "深度清洁": "水（适量）\n设备说明允许的清洁剂（按需）",
    "和面": "面粉（按需）\n水或菜谱指定液体（按需）",
    "慢煮": "已准备好的主食材（按需）\n烹饪液体（适量）",
    "翻炒": "待炒食材（按需）\n食用油（适量）",
}


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def parse_steps(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [clean_text(item) for item in parsed if clean_text(item)]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return [part.strip() for part in re.split(r"[\r\n]+", text) if part.strip()]


def split_ingredient_lines(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    text = text.replace("\r", "\n")
    parts = re.split(r"[\n；;]+", text)
    expanded: list[str] = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip(" ,，、")
        if not part:
            continue
        # 同一行若包含多个带用量的项目，按下一项用量的起点切开。
        chunks = re.split(
            r"[,，]\s*(?=(?:\d+(?:\.\d+)?|\d+/\d+)\s*(?:"
            r"kg|g|ml|l|tbsp|tsp|cup|cups|克|千克|公斤|毫升|升|个|只|片|根|瓣))",
            part,
            flags=re.IGNORECASE,
        )
        expanded.extend(chunk.strip(" ,，、") for chunk in chunks if chunk.strip())
    return list(dict.fromkeys(expanded))


def ingredient_name(line: str, lang: str) -> str:
    text = re.sub(r"\([^)]*\)|（[^）]*）", "", clean_text(line))
    if lang == "en":
        text = re.sub(
            r"^\s*(?:\d+(?:\.\d+)?(?:\s*[-–~]\s*\d+(?:\.\d+)?)?|\d+/\d+)\s*"
            r"(?:kilograms?|grams?|millilit(?:er|re)s?|lit(?:er|re)s?|"
            r"tablespoons?|teaspoons?|cups?|pinch(?:es)?|cloves?|slices?|"
            r"pieces?|kg|ml|tbsp|tsp|pcs?|ea|g|l)?\s*(?:of\s+)?",
            "",
            text,
            flags=re.IGNORECASE,
        )
    else:
        text = re.sub(
            r"^\s*(?:\d+(?:\.\d+)?(?:\s*[-–~]\s*\d+(?:\.\d+)?)?|\d+/\d+)\s*"
            r"(?:克|千克|公斤|斤|毫升|升|个|只|颗|片|根|瓣|勺|茶匙|汤匙|撮)?\s*",
            "",
            text,
        )
    text = re.sub(
        r"\s*,?\s*(?:as needed|to taste|for garnish|optional)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(" ,，、")


def extract_local_ingredients(name: str, steps: list[str], lang: str) -> str:
    """Qwen 不可用时，从步骤中的显式用量保守提取。"""
    text = "\n".join(steps)
    results: list[str] = []
    if lang == "en":
        measured = re.compile(
            r"(?P<amount>(?:\d+(?:\.\d+)?(?:\s*[-–~]\s*\d+(?:\.\d+)?)?|\d+/\d+)\s*"
            r"(?:kg|g|ml|l|tbsp|tsp|cups?|pinch(?:es)?|cloves?|slices?|pieces?|pcs?|ea))"
            r"\s+(?:of\s+)?(?P<item>[A-Za-z][A-Za-z0-9 '/-]{1,55})",
            flags=re.IGNORECASE,
        )
        for match in measured.finditer(text):
            item = re.split(
                r"\b(?:and|then|secure|into|in the|to the|with|until|around|for)\b",
                match.group("item"),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" ,.;")
            if item and not re.search(r"\b(?:minute|minutes|hour|hours|°c|oven)\b", item, re.I):
                results.append(f"{match.group('amount')} {item}")
        countable = re.compile(
            rf"\b(\d+)\s+({'|'.join(map(re.escape, COUNTABLE_EN))})\b",
            flags=re.IGNORECASE,
        )
        for amount, item in countable.findall(text):
            results.append(f"{amount} {item}")
    else:
        measured = re.compile(
            r"(?P<amount>(?:\d+(?:\.\d+)?(?:\s*[-–~]\s*\d+(?:\.\d+)?)?|\d+/\d+)\s*"
            r"(?:克|千克|公斤|斤|毫升|升|个|只|颗|片|根|瓣|勺|茶匙|汤匙|撮))"
            r"\s*(?P<item>[\u4e00-\u9fffA-Za-z][^，。；、\n]{0,30})"
        )
        for match in measured.finditer(text):
            item = re.split(
                r"(?:然后|放入|加入|盖上|安装|烹饪|分钟|小时|温度)",
                match.group("item"),
                maxsplit=1,
            )[0].strip(" ,，。；、")
            if item and "℃" not in item:
                results.append(f"{match.group('amount')} {item}")
    results = list(dict.fromkeys(item for item in results if len(item) <= 90))
    if results:
        return "\n".join(results[:24])
    # 最保守兜底：至少保留菜名指向的主食材，不编数量。
    return name if name else ("Unknown ingredients" if lang == "en" else "食材待确认")


def extract_seasonings(ingredients: str, lang: str) -> str:
    lines = split_ingredient_lines(ingredients)
    terms = SEASONING_TERMS_EN if lang == "en" else SEASONING_TERMS_ZH
    matched = [
        line
        for line in lines
        if any(term.lower() in line.lower() for term in terms)
    ]
    if matched:
        return "\n".join(dict.fromkeys(matched))
    return (
        "No additional seasoning required."
        if lang == "en"
        else "无需额外调料"
    )


def main_ingredients(
    ingredients: str,
    lang: str,
    *,
    recipe_name: str = "",
    limit: int = 3,
) -> list[str]:
    lines = split_ingredient_lines(ingredients)
    excludes = MAIN_EXCLUDE_EN if lang == "en" else MAIN_EXCLUDE_ZH
    results = []
    for line in lines:
        lowered = line.lower()
        if any(term.lower() in lowered for term in excludes):
            continue
        name = ingredient_name(line, lang)
        if name and name.lower() not in {item.lower() for item in results}:
            results.append(name)
    compact_name = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", recipe_name.lower())

    def mentioned_in_name(item: str) -> bool:
        compact_item = re.sub(
            r"[^a-z0-9\u4e00-\u9fff]+",
            "",
            item.lower(),
        )
        if compact_item and compact_item in compact_name:
            return True
        tokens = re.findall(r"[a-z]{3,}|[\u4e00-\u9fff]{2,}", item.lower())
        return any(token in compact_name for token in tokens)

    results.sort(
        key=lambda item: 0 if mentioned_in_name(item) else 1
    )
    return results[:limit]


def cooking_method(steps: list[str], lang: str) -> str:
    source = " ".join(steps).lower()
    methods = METHODS_EN if lang == "en" else METHODS_ZH
    found = [label for marker, label in methods if marker.lower() in source]
    if not found:
        return "step-by-step preparation" if lang == "en" else "分步制作"
    unique = list(dict.fromkeys(found))
    if lang == "en":
        return " and ".join(unique[:2])
    return "与".join(unique[:2])


def generate_description(name: str, ingredients: str, steps: list[str], lang: str) -> str:
    mains = main_ingredients(ingredients, lang, recipe_name=name)
    method = cooking_method(steps, lang)
    if lang == "en":
        if mains:
            joined = ", ".join(mains)
            return (
                f"{name} is prepared mainly with {joined}, following the existing "
                f"recipe sequence for {method}. It is written for straightforward "
                "home-kitchen preparation."
            )
        return (
            f"{name} follows the existing ingredient list and cooking sequence, "
            "with clear steps suitable for home preparation."
        )
    if mains:
        joined = "、".join(mains)
        return (
            f"{name}主要使用{joined}，按照现有菜谱顺序完成{method}，"
            "步骤清晰，适合家庭厨房按流程制作。"
        )
    return f"{name}依据现有食材和烹饪步骤整理，流程清晰，适合家庭厨房按步骤完成。"


def generate_tip(steps: list[str], lang: str) -> str:
    source = " ".join(steps).lower()
    if lang == "en":
        if any(term in source for term in ("bake", "oven", "roast")):
            return (
                "Ovens vary, so start checking near the end of the stated baking time "
                "and remove the dish once the expected color and texture are reached."
            )
        if "steam" in source:
            return (
                "Start timing after steady steam is established, and avoid lifting the "
                "lid repeatedly so the cooking temperature stays consistent."
            )
        if any(term in source for term in ("saute", "sauté", "stir-fry", "fry")):
            return (
                "Measure and prepare everything before heating; add ingredients in the "
                "listed order and avoid extending the final cooking stage unnecessarily."
            )
        if any(term in source for term in ("stew", "simmer", "boil")):
            return (
                "Keep the liquid at a steady, controlled simmer and check the texture "
                "near the end before deciding whether a little more time is needed."
            )
        if any(term in source for term in ("blend", "chop", "stir", "mix")):
            return (
                "Cut larger ingredients evenly and scrape down the sides between mixing "
                "stages so the final texture remains consistent."
            )
        return (
            "Measure the ingredients in advance, follow the listed order, and use the "
            "final texture described in the steps as the cue for doneness."
        )
    if any(term in source for term in ("烤", "烘焙", "烤箱")):
        return "不同烤箱火力会有差异，接近设定时间时提前观察上色情况，达到步骤要求的颜色和质地即可取出。"
    if "蒸" in source:
        return "蒸汽稳定后再开始计时，过程中尽量不要频繁开盖，以免温度波动影响成熟度。"
    if any(term in source for term in ("炒", "煎", "炸")):
        return "下锅前先把食材称量并准备好，严格按步骤顺序加入，最后阶段不要无故延长加热时间。"
    if any(term in source for term in ("炖", "焖", "煮")):
        return "烹煮时保持稳定火力，临近结束先检查食材质地，再决定是否需要适当延长时间。"
    if any(term in source for term in ("搅拌", "打碎", "切碎", "混合")):
        return "较大的食材尽量切得均匀，分段搅拌时及时刮下容器侧壁，成品质地会更一致。"
    return "制作前先称量并备齐食材，按现有步骤顺序操作，以步骤描述的最终质地作为完成判断。"


def chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def extract_json_object(content: str) -> dict:
    text = clean_text(content)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        value = json.loads(match.group(0))
    return value if isinstance(value, dict) else {}


async def qwen_missing_ingredients(
    records: list[dict],
    *,
    cache_path: Path,
    concurrency: int,
) -> dict[str, str]:
    """仅为食材空白行调用 Qwen；缓存保证可重跑。"""
    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                cache = {
                    str(key): clean_text(value)
                    for key, value in cached.items()
                    if clean_text(value)
                }
        except (OSError, ValueError, json.JSONDecodeError):
            cache = {}

    pending = [record for record in records if record["id"] not in cache]
    if not pending:
        return cache

    api_key = clean_text(os.getenv("DASHSCOPE_API_KEY"))
    if not api_key:
        print("INGREDIENT_COMPLETION qwen=disabled reason=missing_api_key")
        return cache
    base_url = clean_text(os.getenv("DASHSCOPE_BASE_URL")) or (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = clean_text(os.getenv("RECIPE_METADATA_MODEL")) or "qwen-plus"
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.AsyncClient(trust_env=True, timeout=60),
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    prompt = """你是菜谱元数据整理器。输入是食材字段缺失的真实菜谱。
请根据菜谱名称和已有烹饪步骤，补出合理的食材清单。

规则：
1. lang=zh 使用简体中文，lang=en 使用自然英文。
2. 步骤中出现明确数量和单位时必须原样保留，不得随意改数值。
3. 只列实际食材、烹饪用水和调料，不要列刀具、锅具、附件、温度、时间或动作。
4. 每项食材单独一行；如果原步骤没有数量，可不编数量。
5. 不输出营养、说明、Markdown 或代码块。
6. 返回 JSON 对象：{"items":[{"id":"原ID","ingredients":"多行食材"}]}。
"""

    async def generate(batch: list[dict]) -> dict[str, str]:
        payload = {"recipes": batch}
        async with semaphore:
            for attempt in range(3):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": json.dumps(payload, ensure_ascii=False),
                            },
                        ],
                        temperature=0.1,
                        max_tokens=5000,
                    )
                    parsed = extract_json_object(
                        response.choices[0].message.content or ""
                    )
                    allowed = {item["id"] for item in batch}
                    result = {}
                    for item in parsed.get("items") or []:
                        recipe_id = clean_text(item.get("id"))
                        ingredients = clean_text(item.get("ingredients"))
                        if recipe_id in allowed and ingredients:
                            result[recipe_id] = ingredients
                    if result:
                        return result
                except Exception as exc:
                    if attempt == 2:
                        print(
                            "INGREDIENT_COMPLETION batch_failed "
                            f"error_type={type(exc).__name__}",
                            flush=True,
                        )
                    await asyncio.sleep(1.5 * (attempt + 1))
        return {}

    batches = chunked(pending, 8)
    completed = 0
    for task in asyncio.as_completed([
        asyncio.create_task(generate(batch)) for batch in batches
    ]):
        cache.update(await task)
        completed += 1
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"INGREDIENT_COMPLETION progress={completed}/{len(batches)} "
            f"completed_rows={len(cache)}/{len(records)}",
            flush=True,
        )
    await client.close()
    return cache


def style_sheet(ws) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    header_alignment = Alignment(horizontal="center", vertical="center")
    thin_blue = Side(style="thin", color="B4C6E7")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_alignment
        cell.border = Border(bottom=thin_blue)
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.showGridLines = False

    widths = {
        "A": 24,
        "B": 34,
        "C": 48,
        "D": 28,
        "E": 58,
        "F": 10,
        "G": 58,
        "H": 14,
        "I": 14,
        "J": 58,
        "K": 46,
        "L": 76,
    }
    for column, width in widths.items():
        ws.column_dimensions[column].width = width
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                horizontal="left",
                wrap_text=cell.column not in {1, 6, 8, 9},
            )


def validate_output(ws, generated_positions: set[tuple[int, str]]) -> dict:
    headers = [clean_text(cell.value) for cell in ws[1]]
    header_map = {name: index + 1 for index, name in enumerate(headers)}
    missing = Counter()
    lang_counts = Counter()
    deleted_present = []
    english_generated_cjk = []
    for row in range(2, ws.max_row + 1):
        recipe_id = clean_text(ws.cell(row, header_map["id"]).value)
        lang = clean_text(ws.cell(row, header_map["lang"]).value) or "zh"
        lang_counts[lang] += 1
        if recipe_id in DELETE_IDS:
            deleted_present.append(recipe_id)
        for field in ("食材", "菜谱描述", "小贴士", "调料"):
            value = clean_text(ws.cell(row, header_map[field]).value)
            if not value:
                missing[field] += 1
            if (
                lang == "en"
                and field in {"菜谱描述", "小贴士", "调料"}
                and (row, field) in generated_positions
            ):
                if re.search(r"[\u4e00-\u9fff]", value):
                    english_generated_cjk.append({
                        "row": row,
                        "id": recipe_id,
                        "field": field,
                    })
    return {
        "rows": ws.max_row - 1,
        "columns": ws.max_column,
        "headers": headers,
        "language_counts": dict(lang_counts),
        "missing_required_enrichment": dict(missing),
        "deleted_ids_still_present": deleted_present,
        "english_generated_cjk": english_generated_cjk[:50],
        "has_nutrition_column": "营养成分" in headers,
    }


async def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    wb = load_workbook(args.input)
    ws = wb[wb.sheetnames[0]]
    headers = [clean_text(cell.value) for cell in ws[1]]
    if tuple(headers) != REQUIRED_HEADERS:
        raise RuntimeError(f"字段结构不符合预期：{headers}")
    header_map = {name: index + 1 for index, name in enumerate(headers)}

    original_rows = ws.max_row - 1
    deleted = []
    for row in range(ws.max_row, 1, -1):
        recipe_id = clean_text(ws.cell(row, header_map["id"]).value)
        if recipe_id in DELETE_IDS:
            deleted.append({
                "id": recipe_id,
                "name": clean_text(ws.cell(row, header_map["名称"]).value),
                "lang": clean_text(ws.cell(row, header_map["lang"]).value),
            })
            ws.delete_rows(row, 1)
    deleted.reverse()

    missing_records = []
    for row in range(2, ws.max_row + 1):
        if clean_text(ws.cell(row, header_map["食材"]).value):
            continue
        missing_records.append({
            "id": clean_text(ws.cell(row, header_map["id"]).value),
            "name": clean_text(ws.cell(row, header_map["名称"]).value),
            "lang": (
                "en"
                if clean_text(ws.cell(row, header_map["lang"]).value) == "en"
                else "zh"
            ),
            "steps": parse_steps(ws.cell(row, header_map["烹饪步骤"]).value),
        })

    generated_ingredients = await qwen_missing_ingredients(
        missing_records,
        cache_path=args.cache,
        concurrency=args.llm_concurrency,
    )

    counts = Counter()
    generated_positions: set[tuple[int, str]] = set()
    for row in range(2, ws.max_row + 1):
        recipe_id = clean_text(ws.cell(row, header_map["id"]).value)
        name = clean_text(ws.cell(row, header_map["名称"]).value)
        lang = (
            "en"
            if clean_text(ws.cell(row, header_map["lang"]).value) == "en"
            else "zh"
        )
        steps = parse_steps(ws.cell(row, header_map["烹饪步骤"]).value)
        ingredient_cell = ws.cell(row, header_map["食材"])
        if not clean_text(ingredient_cell.value):
            program_overrides = (
                GENERIC_PROGRAM_INGREDIENTS_EN
                if lang == "en"
                else GENERIC_PROGRAM_INGREDIENTS_ZH
            )
            ingredient_cell.value = (
                LOCAL_INGREDIENT_OVERRIDES.get((recipe_id, lang))
                or program_overrides.get(name)
                or generated_ingredients.get(recipe_id)
                or extract_local_ingredients(name, steps, lang)
            )
            generated_positions.add((row, "食材"))
            counts[f"ingredients_{lang}"] += 1

        ingredients = clean_text(ingredient_cell.value)
        description_cell = ws.cell(row, header_map["菜谱描述"])
        if not clean_text(description_cell.value):
            description_cell.value = generate_description(
                name,
                ingredients,
                steps,
                lang,
            )
            generated_positions.add((row, "菜谱描述"))
            counts[f"description_{lang}"] += 1

        tips_cell = ws.cell(row, header_map["小贴士"])
        if not clean_text(tips_cell.value):
            tips_cell.value = generate_tip(steps, lang)
            generated_positions.add((row, "小贴士"))
            counts[f"tips_{lang}"] += 1

        seasoning_cell = ws.cell(row, header_map["调料"])
        if not clean_text(seasoning_cell.value):
            seasoning_cell.value = extract_seasonings(ingredients, lang)
            generated_positions.add((row, "调料"))
            counts[f"seasoning_{lang}"] += 1

    nutrition_column = header_map["营养成分"]
    ws.delete_cols(nutrition_column, 1)
    style_sheet(ws)

    validation = validate_output(ws, generated_positions)
    if (
        validation["missing_required_enrichment"]
        or validation["deleted_ids_still_present"]
        or validation["has_nutrition_column"]
        or validation["english_generated_cjk"]
    ):
        raise RuntimeError(
            "输出校验失败：" + json.dumps(validation, ensure_ascii=False)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.output)
    validation["source_file"] = str(args.input)
    validation["output_file"] = str(args.output)
    validation["original_rows"] = original_rows
    validation["deleted_rows"] = deleted
    validation["filled_counts"] = dict(counts)
    validation["elapsed_seconds"] = round(time.monotonic() - started, 2)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--llm-concurrency", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
