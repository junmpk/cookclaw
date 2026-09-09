"""
Normalize Module — 元数据治理 & query 槽位抽取（混合检索的"过滤维度"地基）

职责：
  1. clean_ingredients()  把粘连脏串 "水 200.0克韭菜 50.0克盐 5.0克" 拆成干净食材名数组
  2. derive_facets()      把当前语言 tags 归一成语言无关 canonical code
  3. build_index_text()   拼 BM25 索引文本（名称 + 食材 + 标签）
  4. extract_query_facets() 从用户 query 抽出 facet 槽位 + 素食/忌口意图

设计原则（见与产品方的约定）：
  - 硬过滤只给"违反=事故"的维度：lang / 素食(diet) / 忌口(exclude)
  - 其余 facet 一律做软偏好（可选过滤或加权），避免噪声标签硬砍误杀
  - "素食可选/素食友好" 属软标记，**不**计入严格素食硬过滤
"""
import re
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────
#  受控 facet 词表：dim -> {canonical code: [中文/英文标签及查询别名]}
#  与 scripts/enrich_recipe_tags_nutrition_qwen.py 的 DISPLAY 同源。
#  标签派生采用完整标签匹配；自然语言 query 才采用边界/片段匹配。
# ─────────────────────────────────────────────────────────────
FACET_VOCAB: Dict[str, Dict[str, List[str]]] = {
    "cuisine": {
        "hunan": ["湘菜", "湖南菜", "湖南", "Hunan"],
        "sichuan": ["川菜", "四川菜", "四川", "Sichuan"],
        "cantonese": ["粤菜", "广东菜", "广东", "Cantonese"],
        "shandong": ["鲁菜", "山东菜", "Shandong"],
        "jiangsu": ["苏菜", "淮扬菜", "Jiangsu"],
        "zhejiang": ["浙菜", "杭州菜", "浙江菜", "Zhejiang"],
        "fujian": ["闽菜", "福建菜", "Fujian"],
        "anhui": ["徽菜", "安徽菜", "Anhui"],
        "northeastern_chinese": ["东北菜", "Northeastern Chinese"],
        "northwestern_chinese": ["西北菜", "Northwestern Chinese"],
        "home_style": ["家常菜", "家常", "Home-style", "Homestyle"],
        "western": ["西餐", "Western"],
        "japanese": ["日料", "日式", "Japanese"],
        "korean": ["韩餐", "韩式", "Korean"],
        "thai": ["泰式", "泰国菜", "Thai"],
        "vietnamese": ["越南菜", "Vietnamese"],
        "italian": ["意大利菜", "Italian"],
        "french": ["法餐", "法国菜", "French"],
        "mediterranean": ["地中海菜", "Mediterranean"],
        "mexican": ["墨西哥菜", "Mexican"],
        "indian": ["印度菜", "Indian"],
        "middle_eastern": ["中东菜", "Middle Eastern"],
        "moroccan": ["摩洛哥菜", "Moroccan"],
        "american": ["美式", "美国菜", "American"],
    },
    "flavor": {
        "spicy": ["香辣", "辣", "Spicy"],
        "numbing_spicy": ["麻辣", "Numbing-spicy", "Numbing spicy"],
        "savory": ["咸鲜", "Savory"],
        "light": ["清淡", "清爽", "Light"],
        "sweet": ["甜味", "甜", "Sweet"],
        "sweet_sour": ["酸甜", "糖醋", "Sweet-and-sour", "Sweet and sour"],
        "sour_spicy": ["酸辣", "Sour-and-spicy", "Sour and spicy"],
        "sour": ["酸味", "酸", "Sour"],
        "salty": ["咸香", "Salty-savory", "Salty savory"],
        "umami": ["鲜味", "Umami"],
        "aromatic": ["鲜香", "清香", "Aromatic"],
        "creamy": ["奶香", "Creamy"],
    },
    "method": {
        "stir_fried": ["炒", "爆炒", "小炒", "Stir-fried", "Stir fried", "Stir fry"],
        "deep_fried": ["油炸", "炸", "Deep-fried", "Deep fried"],
        "fried": ["煎", "Fried", "Pan-fried", "Pan fried"],
        "steamed": ["蒸", "Steamed"],
        "stewed": ["炖", "Stewed"],
        "braised": ["红烧", "Braised"],
        "boiled": ["煮", "Boiled"],
        "simmered": ["煨", "焖", "Simmered"],
        "roasted": ["烤制", "Roasted"],
        "baked": ["烘焙", "Baked"],
        "grilled": ["烧烤", "Grilled"],
        "cold_tossed": ["凉拌", "Cold-tossed", "Cold tossed"],
        "blended": ["搅打", "Blended"],
        "juiced": ["榨汁", "Juiced"],
        "pickled": ["腌制", "Pickled"],
        "whipping": ["打发", "Whipping"],
        "fermenting": ["发酵", "Fermenting"],
        "pureeing": ["制泥", "Pureeing"],
        "stirring": ["搅拌", "Stirring"],
        "sous_vide": ["低温慢煮", "Sous vide"],
        "reheating": ["加热", "Reheating"],
        "warming": ["保温", "Keep-warm", "Keep warm"],
        "peeling": ["削皮", "Peeling"],
        "slicing": ["切片切丝", "Slicing and shredding"],
        "grinding": ["研磨", "Grinding"],
        "kneading": ["和面", "Kneading"],
        "cleaning": ["清洁", "Cleaning"],
    },
    "main_ingredient": {
        "beef": ["牛肉", "Beef"],
        "chicken": ["鸡肉", "鸡翅", "鸡腿", "Chicken"],
        "pork": ["猪肉", "五花肉", "排骨", "Pork"],
        "lamb": ["羊肉", "Lamb", "Mutton"],
        "duck": ["鸭肉", "Duck"],
        "fish": ["鱼类", "鱼", "Fish"],
        "shrimp": ["虾类", "虾", "Shrimp", "Prawn"],
        "seafood": ["海鲜", "水产", "Seafood"],
        "egg": ["蛋类", "鸡蛋", "Egg"],
        "tofu": ["豆制品", "豆腐", "Tofu"],
        "vegetable": ["蔬菜", "时蔬", "Vegetable"],
        "mushroom": ["菌菇", "蘑菇", "Mushroom"],
        "potato": ["薯类", "土豆", "Potato"],
        "rice": ["米饭", "Rice"],
        "noodle": ["面食", "面条", "Noodle"],
        "dairy": ["乳制品", "奶制品", "Dairy"],
        "fruit": ["水果", "Fruit"],
        "grain": ["谷物", "Grain"],
        "legume": ["豆类", "Legume"],
        "nuts_seeds": ["坚果种子", "坚果", "花生", "Nuts and seeds"],
        "herb_spice": ["香辛料", "香料", "Herbs and spices"],
        "water": ["水", "Water"],
        "prepared_food": ["熟食", "Prepared food"],
        "prepared_ingredient": ["预处理食材", "Prepared ingredient"],
    },
    "nutrition": {
        "high_protein": ["高蛋白", "High-protein", "High protein"],
        "low_fat": ["低脂", "Low-fat", "Low fat"],
        "low_calorie": ["低卡", "低热量", "Low-calorie", "Low calorie"],
        "high_fiber": ["高纤维", "High-fiber", "High fiber"],
        "low_carb": ["低碳水", "Low-carb", "Low carb"],
    },
    "scene": {
        "quick": ["快手菜", "快手", "省时", "Quick"],
        "rice_companion": ["下饭菜", "下饭", "Rice-companion"],
        "fat_loss": ["减脂餐", "减脂", "减肥", "Fat-loss", "Fat loss"],
        "fitness": ["健身餐", "健身", "Fitness"],
        "child_friendly": ["儿童友好", "宝宝", "Child-friendly", "Kids"],
        "elderly_friendly": ["老人友好", "老人", "Elderly-friendly"],
        "banquet": ["宴客菜", "宴客", "宴请", "Banquet"],
        "bento": ["便当", "Bento"],
        "food_preparation": ["备餐功能", "Food preparation"],
        "device_maintenance": ["设备维护", "Device maintenance"],
    },
    "meal": {
        "breakfast": ["早餐", "Breakfast"],
        "main_course": ["正餐", "主菜", "Main course"],
        "soup": ["汤品", "汤", "羹", "Soup"],
        "dessert": ["甜品", "甜点", "Dessert"],
        "beverage": ["饮品", "饮料", "Beverage", "Drink"],
        "snack": ["小吃", "Snack"],
        "staple": ["主食", "Staple"],
        "side_dish": ["配菜", "Side dish"],
        "sauce": ["酱料", "Sauce"],
        "kitchen_program": ["厨房功能", "Kitchen function"],
    },
    "diet": {
        "vegetarian": ["素食", "素菜", "Vegetarian"],
        "vegan": ["纯素", "全素", "Vegan"],
        "gluten_free": ["无麸质", "Gluten-free", "Gluten free"],
    },
}

_DIET_VEG_STRICT = ["纯素", "全素", "无肉", "vegan"]
_DIET_VEG_PLAIN = ["素食", "vegetarian", "素菜"]
# 软素食标记（**不**计入硬过滤）：可作软偏好
_DIET_VEG_SOFT = ["素食可选", "素食友好", "vegetarian-adaptable", "vegetarian-friendly"]

# 营养维度区分度太低、被 LLM 过度滥贴 → 默认不允许作为硬过滤的 facet
SOFT_ONLY_DIMS = {"nutrition", "flavor", "method", "scene", "meal"}
# 可作"可选硬过滤"的维度（区分度足够）
FILTERABLE_DIMS = {"cuisine", "main_ingredient"}

# 标签使用完整值匹配，避免 ``fried`` 误命中 ``stir-fried``。
def _is_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s) and not any("a" <= ch.lower() <= "z" for ch in s)


def _normalized_label(value: Any) -> str:
    return re.sub(r"[\s_-]+", " ", str(value or "").strip().casefold())


_EXACT_TAG2FACET: Dict[str, tuple[str, str]] = {}
_QUERY_TAG2FACET: List[tuple[str, str, str, bool]] = []
for _dim, _cans in FACET_VOCAB.items():
    for _canon, _syns in _cans.items():
        for _syn in [_canon, *_syns]:
            _normalized = _normalized_label(_syn)
            _EXACT_TAG2FACET[_normalized] = (_dim, _canon)
            _QUERY_TAG2FACET.append((_normalized, _dim, _canon, _is_cjk(_syn)))
_QUERY_TAG2FACET.sort(key=lambda x: len(x[0]), reverse=True)


# ── 1) 食材清洗 ───────────────────────────────────────────────
_UNIT = (r"克|千克|公斤|斤|两|毫升|升|ml|ML|g|G|个|只|条|根|块|片|张|"
         r"大勺|小勺|勺|茶匙|汤匙|匙|杯|瓣|棵|颗|朵|段|把|撮|滴|包|盒|袋|罐|碗|杯")
_QTY_RE = re.compile(r"[\d.／/]+\s*(?:" + _UNIT + r")?")
_AMOUNT_WORD_RE = re.compile(r"适量|少许|若干|半勺|半碗|半|数")
_LEFTOVER_RE = re.compile(r"^(?:[\d.]+|" + _UNIT + r")$")


def clean_ingredients(raw: Any) -> List[str]:
    """把粘连脏串拆成干净食材名数组（去掉数量+单位）。

    '水 200.0克韭菜 50.0克盐 5.0克' -> ['水','韭菜','盐']
    """
    if not raw:
        return []
    s = " ".join(str(x) for x in raw) if isinstance(raw, list) else str(raw)
    s = _QTY_RE.sub("|", s)
    s = _AMOUNT_WORD_RE.sub("|", s)
    out, seen = [], set()
    for p in re.split(r"[|\s、,，;；/]+", s):
        p = p.strip()
        if not p or _LEFTOVER_RE.match(p):
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ── 2) tags -> facets 归一 ────────────────────────────────────
def derive_facets(tags: Any) -> Dict[str, List[str]]:
    """把单语展示 tags 按完整标签归一成 canonical facets。"""
    if isinstance(tags, dict):
        tag_list = [str(v) for v in tags.values() if v]
    elif isinstance(tags, list):
        tag_list = [str(t) for t in tags if t]
    else:
        tag_list = [str(tags)] if tags else []

    facets: Dict[str, List[str]] = {}

    def add(dim, canon):
        facets.setdefault(dim, [])
        if canon not in facets[dim]:
            facets[dim].append(canon)

    for tag in tag_list:
        normalized = _normalized_label(tag)
        match = _EXACT_TAG2FACET.get(normalized)
        if match:
            add(*match)
        if any(_normalized_label(soft) == normalized for soft in _DIET_VEG_SOFT):
            add("diet_soft", "vegetarian_friendly")
    return facets


# ── 3) BM25 索引文本 ──────────────────────────────────────────
def build_index_text(name: str, ingredients: Any, tags: Any) -> str:
    """拼 BM25 文本：名称 + 原始食材 + 当前语言标签 + canonical code。"""
    parts: List[str] = [str(name or "")]
    if isinstance(ingredients, dict):
        parts.extend(str(v) for v in ingredients.values() if v)
    elif isinstance(ingredients, list):
        parts.extend(str(v) for v in ingredients if v)
    elif ingredients:
        parts.append(str(ingredients))
    if isinstance(tags, dict):
        parts.extend(str(v) for v in tags.values() if v)
    elif isinstance(tags, list):
        parts.extend(str(t) for t in tags if t)
    return " ".join(p for p in parts if p)


# ── 4) query 槽位抽取 ─────────────────────────────────────────
# 忌口/排除：无X / 不要X / 不吃X / 不放X / 忌X / 去X
# 捕获块允许跨 、，,分隔符（"香菜、葱和蒜"整体抓下来），再按分隔符拆成单个
_EXCLUDE_RE = re.compile(r"(?:无|不要|不吃|不放|不加|忌|去掉?|没有)\s*([一-鿿、，,]{1,12})")
_EXCLUDE_SEP_RE = re.compile(r"[和与跟、,，及]|还有")


def extract_query_facets(query: str) -> Dict[str, Any]:
    """从 query 抽取 facet 槽位 + 素食 + 忌口意图。

    Returns: {
        "facets": {dim: [canonical,...]},   # 命中的偏好维度
        "diet_veg": bool,                    # 是否要素食（硬过滤）
        "exclude": [ingredient,...],         # 忌口食材（硬过滤）
    }
    """
    q = query or ""
    normalized_query = _normalized_label(q)
    facets: Dict[str, List[str]] = {}
    occupied: Dict[str, List[tuple[int, int]]] = {}
    for syn, dim, canon, is_cjk in _QUERY_TAG2FACET:
        if not syn:
            continue
        if is_cjk:
            matches = list(re.finditer(re.escape(syn), normalized_query))
        else:
            matches = list(re.finditer(
                rf"(?<![a-z0-9]){re.escape(syn)}(?![a-z0-9])",
                normalized_query,
                flags=re.IGNORECASE,
            ))
        for match in matches:
            span = match.span()
            # 同一维度长别名优先并占位，避免 Stir-fried 同时落入 fried。
            if any(span[0] >= left and span[1] <= right for left, right in occupied.get(dim, [])):
                continue
            facets.setdefault(dim, [])
            if canon not in facets[dim]:
                facets[dim].append(canon)
            occupied.setdefault(dim, []).append(span)

    low = q.casefold()
    diet_veg = any(v.casefold() in low for v in _DIET_VEG_PLAIN + _DIET_VEG_STRICT) and \
        not any(s.casefold() in low for s in _DIET_VEG_SOFT)
    exclude = []
    for blob in _EXCLUDE_RE.findall(q):
        for part in _EXCLUDE_SEP_RE.split(blob):
            part = part.strip()
            if part and part not in exclude:
                exclude.append(part)
    return {"facets": facets, "diet_veg": diet_veg, "exclude": exclude}


if __name__ == "__main__":
    # 自测：用真实样例验证清洗 / 归一 / 槽位
    import json
    samples = [
        {"name": "酸菜煨蚕豆",
         "ingredients": ["水 200.0克韭菜 50.0克盐 5.0克猪油 50.0克蔬菜精 3.0克", "酸菜 150.0克蚕豆仁 300.0克"],
         "tags": ["川菜", "Sichuan cuisine", "家常菜", "酸咸", "煨", "braised", "蔬菜", "低脂", "素食可选", "快手菜"]},
    ]
    for s in samples:
        print("清洗食材:", clean_ingredients(s["ingredients"]))
        print("facets  :", json.dumps(derive_facets(s["tags"]), ensure_ascii=False))
        print("索引文本:", build_index_text(s["name"], s["ingredients"], s["tags"]))
    for q in ["想吃清淡的素食鸡肉菜，不要香菜", "来个川菜，无花生", "红烧排骨"]:
        print(f"query={q!r} ->", json.dumps(extract_query_facets(q), ensure_ascii=False))
