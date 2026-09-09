"""统一的食谱搜索请求协议。

LLM 只负责把用户原话整理成这个结构；检索、过滤、回复都消费同一个对象，
避免每一层重新猜一次用户需求。旧版只有 r/c/k 时仍可从原话和关键词构造。
"""
from __future__ import annotations

import re
import time
from typing import Any

from pydantic import BaseModel, Field


_LIST_FIELDS = (
    "dishes", "ingredients", "cuisines", "flavors", "methods", "scenes", "meals",
    "dietary_constraints", "exclude", "exclude_cuisines", "avoid",
    "required_ingredients", "soft_preferences",
)
_REFERENCE_WORDS = ("这几个", "这几道", "这些", "上一批", "刚才", "推荐的", "these", "those", "previous")
_GENERIC_SEARCH_WORDS = (
    "吃什么", "推荐", "随便", "没想好", "晚饭", "午饭", "早餐",
    "what should i eat", "recommend", "anything", "dinner", "lunch", "breakfast",
)
_GENERIC_QUERY_PHRASES = (
    "可以做哪些菜", "可以做什么菜", "能做哪些菜", "能做什么菜",
    "适合做哪些菜", "适合做什么菜", "做哪些菜", "做什么菜",
    "有哪些菜", "有什么菜", "哪些菜", "什么菜", "有什么推荐", "推荐一下",
    "你安排", "你来安排", "你定", "你来定", "你决定", "给我最终答案",
    "给我一个最终答案", "最终安排", "最终答案", "最终菜单", "按你说的来",
    "吃什么", "做什么", "推荐", "菜谱", "食谱", "菜",
    "what can i cook", "what can i make", "which dishes", "what dishes", "recommend",
    "recommend something", "recipes", "dishes",
)
_GENERIC_QUERY_FILLERS = (
    "帮我", "给我", "请问", "想要", "想吃", "想做", "看看", "一下", "一点",
    "今天", "今晚", "明天", "好", "呢", "呀", "啊", "吧", "嘛",
    "please", "help me", "show me", "give me", "some", "today", "tonight",
)
_SKIP_CLARIFICATION_MARKERS = (
    "随便", "你决定", "你看着办", "直接推荐", "直接给", "不用问", "别问",
    "你安排", "你来安排", "你来定", "直接定", "最终安排", "最终答案", "最终菜单",
    "按你说的来", "按你的安排", "按默认来", "按默认安排", "默认搭配",
    "你看着安排", "你看着搭配", "你看着推荐", "你做主", "随你",
    "无需澄清", "不需要澄清", "不用澄清", "不必澄清",
    "不要再问", "别再问", "无需再问", "不需要再问", "口味无所谓",
    "都可以", "什么都行", "anything", "surprise me", "you choose", "just pick",
    "final answer", "final menu", "make the final call", "do not ask", "don't ask",
    "no need to clarify", "use the default", "default is fine",
)
_BROAD_SCENE_TERMS = (
    "聚会", "聚餐", "朋友", "客人", "家庭", "家常饭", "日常", "一顿饭",
    "party", "gathering", "guests", "friends", "family meal", "everyday meal",
)
_ACTIONABLE_SCENE_MARKERS = (
    "快手", "简单", "省时", "便当", "减脂", "减肥", "低脂", "控卡",
    "高蛋白", "增肌", "儿童", "小孩", "宝宝", "老人", "长辈", "牙口",
    "天热", "天气热", "炎热", "闷热",
    "quick", "easy", "simple", "lunchbox", "weight loss", "low fat",
    "high protein", "muscle gain", "child", "kid", "baby", "elderly", "senior",
    "hot weather", "hot day",
)
_DRINKING_BACKGROUND_TERMS = (
    "白酒", "啤酒", "红酒", "黄酒", "葡萄酒", "喝酒", "喝两杯",
    "liquor", "beer", "wine", "drinks", "drinking",
)
_CUISINE_EVIDENCE_GROUPS = (
    ("湘菜", "湖南菜", "湖南", "hunan", "hunan cuisine"),
    ("赣菜", "江西菜", "江西", "jiangxi", "jiangxi cuisine"),
    ("川菜", "四川菜", "四川", "sichuan", "sichuan cuisine"),
    ("粤菜", "广东菜", "广东", "cantonese", "cantonese cuisine"),
    ("鲁菜", "山东菜", "山东", "shandong", "shandong cuisine"),
    ("苏菜", "淮扬菜", "江苏菜", "江苏", "jiangsu", "jiangsu cuisine"),
    ("浙菜", "杭州菜", "浙江菜", "浙江", "杭州", "zhejiang", "zhejiang cuisine"),
    ("闽菜", "福建菜", "福建", "fujian", "fujian cuisine"),
    ("徽菜", "安徽菜", "安徽", "anhui", "anhui cuisine"),
    ("东北菜", "东北", "northeastern chinese"),
    ("西北菜", "西北", "northwestern chinese"),
    ("家常菜", "家常", "home_style", "home-style", "homestyle"),
    ("西餐", "western", "western food"),
    ("日料", "日式", "japanese", "japanese food"),
    ("韩餐", "韩式", "korean", "korean food"),
)
_NO_DIETARY_RESTRICTION_RE = re.compile(
    r"没(?:有)?(?:什么|啥|任何)?忌口|无忌口|"
    r"没(?:有)?(?:什么|啥|任何)?过敏|不过敏|无过敏|"
    r"(?:大家|所有人|我们|我)?(?:都)?(?:什么|啥)都不忌口|"
    r"没(?:有)?(?:什么|啥|任何)?(?:饮食)?限制|"
    r"饮食没(?:有)?(?:什么|啥|任何)?限制|"
    r"没(?:有)?(?:什么|啥|任何)?要求|没要求|无要求|"
    r"都能吃|什么都能吃|"
    r"\b(?:no dietary restrictions?|no restrictions?|no allergies)\b",
    flags=re.IGNORECASE,
)
_GENERIC_CONSTRAINT_CONFIRMATION_RE = re.compile(
    r"^(?:没有|没|无|不|也|和|与|或|以及)*"
    r"(?:过敏|忌口|宗教饮食(?:要求)?|饮食(?:要求|限制)|饮食禁忌|要求|限制)"
    r"(?:也|和|与|或|以及|要求|限制)*$",
    flags=re.IGNORECASE,
)
_EFFORT_OR_EMOTION_TERMS = (
    "累", "疲惫", "加班", "不想动", "不想做", "不想折腾", "没精神", "赶时间",
    "来不及", "tired", "exhausted", "overtime", "no energy", "no time",
)
_CONTEXT_REFERENCE_WORDS = (
    "这个", "这种", "它", "用它", "这个食材", "这东西", "刚才那个",
    "this ingredient", "this one", "with it", "that ingredient",
)
_RECENT_TOPIC_IGNORE = (
    "你好", "您好", "哈喽", "hello", "hi", "thanks", "thank you", "谢谢",
    "我是谁", "你是谁", "设备", "在线", "离线", "确认", "取消", "停止",
)
_SOFT_NEGATIVE_TERMS = {
    "辣", "太辣", "油", "太油", "油腻", "太油腻", "甜", "太甜", "咸", "太咸",
    "重口", "重口味", "复杂", "麻烦",
    "spicy", "too spicy", "oily", "too oily", "sweet", "too sweet", "salty", "too salty",
}
_SOFT_POSITIVE_MODIFIERS = {
    "太辣": "微辣",
    "too spicy": "mild",
    "油": "少油",
    "太油": "少油",
    "油腻": "少油",
    "太油腻": "少油",
    "oily": "low oil",
    "too oily": "low oil",
    "甜": "少甜",
    "太甜": "少甜",
    "sweet": "less sweet",
    "too sweet": "less sweet",
    "咸": "少盐",
    "太咸": "少盐",
    "salty": "low salt",
    "too salty": "low salt",
    "重口": "清淡",
    "重口味": "清淡",
    "复杂": "简单",
    "麻烦": "简单",
}
_SPICY_CONSTRAINT_VALUES = {
    "辣", "辣的", "辣味", "香辣", "麻辣", "酸辣", "辣椒",
    "spicy", "spice", "hot", "hot and spicy",
}
_BROAD_FLAVOR_MARKERS = (
    "重口味", "重口", "口味重", "够味", "味道重", "重油重盐",
    "重油", "重盐", "bold flavor", "bold flavours", "bold flavors",
    "strong flavor", "strong flavours", "strong flavors",
    "heavy flavor", "heavy flavours", "heavy flavors",
)
_BROAD_FLAVOR_CANONICAL = {
    "重口味": ("重口味", "香辣", "咸香", "下饭菜"),
    "bold": ("bold flavor", "spicy", "savory"),
}
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
_EN_DIGITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_MENU_QUANTITY_DISSATISFACTION_RE = re.compile(
    r"(?:才|只(?:有)?|就)?"
    r"[0-9一二两三四五六七八九十]{1,3}"
    r"(?:个|道|款|种)?(?:菜谱|食谱|菜)"
    r"[\s，,。.!！?？…~·]*(?:也|还|明显|确实|实在|感觉|根本){0,3}"
    r"[\s，,。.!！?？…~·]*"
    r"(?:太少(?:了)?|少了|不够(?!辣|咸|甜|酸|香|清淡)(?:吃|分|用)?(?:了|吧|啊|呀)?|哪够)",
    flags=re.IGNORECASE,
)


def is_menu_quantity_dissatisfaction(text: str) -> bool:
    """用户是在抱怨已给菜品数量，而不是新指定返回数量。"""
    value = str(text or "").strip()
    if not value:
        return False
    if _MENU_QUANTITY_DISSATISFACTION_RE.search(value):
        return True
    return bool(re.search(
        r"\b(?:only\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|[0-9]{1,2})\s+"
        r"(?:dishes|recipes|options)\b.{0,20}\b(?:too few|not enough)\b",
        value,
        flags=re.IGNORECASE,
    ))


def _parse_count(value: str) -> int | None:
    raw = str(value or "").strip()
    if raw.isdigit():
        return int(raw)
    if raw.lower() in _EN_DIGITS:
        return _EN_DIGITS[raw.lower()]
    if raw == "十":
        return 10
    if "十" in raw:
        left, right = raw.split("十", 1)
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    return _CN_DIGITS.get(raw)


def _extract_party_size(text: str) -> int | None:
    # "我有三个朋友来吃饭"通常表示三位客人加上说话者本人。先处理这类
    # 客人数表达，避免轻量模型把它误写成"三人餐"并直接污染检索词。
    guest_match = re.search(
        r"(?:有|请了?|约了?|来了?|会来|招待)\s*([0-9一二两三四五六七八九十]{1,3})\s*个?"
        r"(?:朋友|客人|同事|同学)(?:.{0,12}(?:来|一起)?.{0,8}(?:吃饭|聚餐|吃顿饭|吃哪些|吃那些|吃什么))?",
        text or "",
        flags=re.IGNORECASE,
    )
    if guest_match:
        guests = _parse_count(guest_match.group(1))
        if guests and 1 <= guests < 100:
            return guests + 1

    patterns = (
        r"(?:一家)?\s*([0-9一二两三四五六七八九十]{1,3})\s*(?:口人|口之家|口|个人|人吃|人份)",
        r"(?:for|serves?)\s+([0-9]{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:people|persons?)",
        r"(?:there\s+(?:will|would)\s+be|we(?:'|’)ll\s+be)\s+"
        r"([0-9]{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+of\s+us",
    )
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            count = _parse_count(match.group(1))
            if count and 1 <= count <= 100:
                return count
    return None


def _has_actionable_scene(values: list[str]) -> bool:
    """季节、地点和招待背景不算选菜方向；目标/人群/效率场景才算。"""
    return any(
        marker in str(value or "").lower()
        for value in values
        for marker in _ACTIONABLE_SCENE_MARKERS
    )


def _is_grounded_scene(scene: str, source_text: str) -> bool:
    """模型 scene 只保留可用于选菜的场景，过滤地域和裸酒类背景。"""
    value = re.sub(r"\s+", " ", str(scene or "")).strip().lower()
    if not value or not _has_source_evidence(value, source_text):
        return False
    key = _compact_search_text(value)
    location_keys = {
        _compact_search_text(location)
        for location in _background_location_terms(source_text)
    }
    # “江西人”是籍贯背景；地点提取返回“江西”，因此同时处理“人/老乡”等后缀。
    if any(
        key == location
        or key in {f"{location}人", f"{location}老乡", f"{location}朋友", f"{location}客人"}
        for location in location_keys
        if location
    ):
        return False
    if key in {_compact_search_text(marker) for marker in _DRINKING_BACKGROUND_TERMS}:
        return False
    return True


def _extract_result_limit(text: str) -> int | None:
    value = str(text or "")
    # “五个菜太少了”的五是对上一批结果的评价，不是新的
    # result_limit=5。先拦截这类抱怨，由有状态路由承接上一个菜单。
    if is_menu_quantity_dissatisfaction(value):
        return None
    patterns = (
        r"(?:推荐|给我|来|想要|要)\D{0,8}?([0-9一二两三四五六七八九十]{1,3})\s*(?:个|道|款|种)[^0-9，。！？]{0,8}?(?:菜谱|食谱|菜)",
        r"([0-9一二两三四五六七八九十]{1,3})\s*(?:个|道|款|种)[^0-9，。！？]{0,8}?(?:菜谱|食谱|菜)",
        r"(?:recommend|show me|give me)\D{0,8}?([0-9]{1,2})\s+(?:dishes|recipes|options)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, value, flags=re.IGNORECASE):
            count = _parse_count(match.group(1))
            if count:
                # “一个食谱清单/一个菜谱列表”里的“一”修饰的是清单，
                # 不是用户只要一道菜；否则多人菜单会被误压成 result_limit=1。
                suffix = value[match.end():]
                if count == 1 and re.match(
                    r"\s*(?:清单|列表|单子|单(?:\s|$|给|来|吧|，|。|！|？))",
                    suffix,
                    flags=re.IGNORECASE,
                ):
                    continue
                return min(10, max(1, count))
    return None


def _extract_menu_counts(text: str) -> tuple[int, int] | None:
    """解析显式套餐数量；只有"菜 + 汤"同时出现才视为菜单规划。"""
    value = re.sub(r"\s+", "", str(text or "")).lower()
    patterns = (
        r"([0-9一二两三四五六七八九十]{1,3})(?:道)?菜(?:加|和|配|带|、|及|与)?"
        r"([0-9一二两三四五六七八九十]{1,3})(?:道)?汤",
        r"([0-9一二两三四五六七八九十]{1,3})(?:道)?汤(?:加|和|配|带|、|及|与)?"
        r"([0-9一二两三四五六七八九十]{1,3})(?:道)?菜",
    )
    for index, pattern in enumerate(patterns):
        matched = re.search(pattern, value, flags=re.IGNORECASE)
        if not matched:
            continue
        first = _parse_count(matched.group(1))
        second = _parse_count(matched.group(2))
        dish_count, soup_count = (first, second) if index == 0 else (second, first)
        if dish_count and soup_count and dish_count + soup_count <= 10:
            return dish_count, soup_count

    english_number = r"(?:[0-9]{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
    english = re.search(
        rf"({english_number})\s*(?:dishes|courses?)\s*(?:and|plus|with)?\s*"
        rf"({english_number})\s*(?:soups?)",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if english:
        dish_count = _parse_count(english.group(1))
        soup_count = _parse_count(english.group(2))
        if dish_count >= 1 and soup_count >= 1 and dish_count + soup_count <= 10:
            return dish_count, soup_count
    return None


def _extract_required_menu_ingredients(text: str) -> list[str]:
    """提取"要有一道鱼"这类整桌菜单中的必备菜位食材。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    patterns = (
        r"(?:要有|得有|必须有|至少有|一定要有)\s*(?:一|1)?\s*道\s*"
        r"([^，。！？,.!?]{1,16})",
        r"(?:其中)?\s*(?:一|1)\s*道\s*(?:要|得|必须)?\s*(?:是|用)?\s*"
        r"([^，。！？,.!?]{1,16})",
    )
    required: list[str] = []
    for pattern in patterns:
        for matched in re.finditer(pattern, value, flags=re.IGNORECASE):
            item = re.sub(
                r"(?:做的|做成的|做成|菜肴|料理|菜|就行|即可)$",
                "",
                matched.group(1).strip(),
            ).strip()
            if item and item not in {"一道", "一份", "主菜"} and item not in required:
                required.append(item)
    return required[:3]


def _extract_scoped_preferences(text: str) -> list[dict[str, Any]]:
    """把"其中一人清淡/只有一个朋友偏辣"建模为局部菜位。"""
    value = re.sub(r"\s+", "", str(text or ""))
    patterns = (
        r"(?:其中|只有|仅有|仅|就|另外)?"
        r"([0-9一二两三四五六七八九十]{1,3}|一个|一位|1个|1位)"
        r"(?:个|位)?(?:朋友|客人|人|同事|同学)"
        r"(?:喜欢吃?|想吃|口味|吃得|偏好?|要)?"
        r"(清淡|少油|不辣|微辣|偏辣|能吃辣|喜欢辣|辣口|辣|湘菜|湖南菜|川菜|四川菜)",
        r"(?:只有|仅|就)(?:其中)?(?:一个|一位|1个|1位)"
        r"(?:朋友|客人|人|同事|同学)?"
        r"(?:喜欢吃?|想吃|口味|吃得|偏好?|要)?"
        r"(清淡|少油|不辣|微辣|偏辣|能吃辣|喜欢辣|辣口|辣|湘菜|湖南菜|川菜|四川菜)",
    )
    matches: list[tuple[int, str]] = []
    for index, pattern in enumerate(patterns):
        for matched in re.finditer(pattern, value, flags=re.IGNORECASE):
            if index == 0:
                count_raw, preference = matched.group(1), matched.group(2)
                count_raw = re.sub(r"(?:个|位)$", "", count_raw)
                count = _parse_count(count_raw) or 1
            else:
                count, preference = 1, matched.group(1)
            item = (count, preference)
            if item not in matches:
                matches.append(item)

    out: list[dict[str, Any]] = []
    for count, preference in matches:
        cuisines: list[str] = []
        flavors: list[str] = []
        if preference in {"清淡", "少油", "不辣"}:
            flavors = ["清淡"]
        elif preference in {"微辣", "偏辣", "能吃辣", "喜欢辣", "辣口", "辣"}:
            flavors = ["辣"]
        elif preference in {"湘菜", "湖南菜"}:
            cuisines = ["湘菜"]
        elif preference in {"川菜", "四川菜"}:
            cuisines = ["川菜"]
        if not cuisines and not flavors:
            continue
        out.append({
            "scope": "guest_subset",
            "count": count,
            # 每个不同偏好子集先保证一个照顾菜位，避免按人数占满整桌。
            "max_slots": 1,
            "cuisines": cuisines,
            "flavors": flavors,
            "source_text": str(text or "").strip(),
        })
    return out


def _default_result_limit(party_size: int | None) -> int:
    if not party_size:
        # 明确主料/口味但未说明人数时直接给 3 个可比较选项，避免为了人数
        # 阻断最常见的"带牛肉的辣菜有哪些"演示和日常搜索。
        return 3
    if party_size <= 2:
        return 2
    if party_size <= 4:
        return 4
    if party_size <= 6:
        return 6
    if party_size <= 8:
        return 8
    return 10


def _clean_list(value: Any, *, limit: int = 12) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[、,，;；|]", value)
    out: list[str] = []
    for raw in value or []:
        item = re.sub(r"\s+", " ", str(raw or "")).strip(" ，。；！？?~")
        if item and item not in out:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _turn_value(turn: Any, field: str, default: Any = "") -> Any:
    if isinstance(turn, dict):
        return turn.get(field, default)
    return getattr(turn, field, default)


def _compact_search_text(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())


def _is_generic_constraint_confirmation(value: str) -> bool:
    """"过敏/忌口/饮食要求"是安全问题维度，不是食材或口味。"""
    return bool(_GENERIC_CONSTRAINT_CONFIRMATION_RE.fullmatch(
        _compact_search_text(value)
    ))


def delegates_recommendation_choice(text: str) -> bool:
    """用户是否明确把非安全的选菜偏好交给系统决定。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not value:
        return False
    if any(marker in value for marker in _SKIP_CLARIFICATION_MARKERS):
        return True
    compact = _compact_search_text(value)
    return bool(re.search(
        r"(?:无需|不需要|不用|不必)(?:再)?(?:澄清|询问|问)|"
        r"(?:不要|别)(?:再)?(?:询问|问)|"
        r"(?:口味|味道|偏好)(?:都)?(?:无所谓|都可以|都行|随便)|"
        r"(?:按|用)(?:系统|默认)(?:来|安排|搭配|推荐)?",
        compact,
        flags=re.IGNORECASE,
    ))


def has_explicit_no_dietary_restrictions(text: str) -> bool:
    """原话是否明确确认没有过敏、忌口或饮食限制。

    “不知道/不确定有没有”描述的是安全事实仍未知，不能因为其中包含
    “没有过敏”这一子串就被当成确认；同理，否定“没有限制”的句子也不成立。
    """
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not value or not _NO_DIETARY_RESTRICTION_RE.search(value):
        return False
    safety_term = r"(?:过敏|忌口|饮食(?:要求|限制|禁忌)|宗教饮食(?:要求)?|要求|限制)"
    uncertainty = r"(?:不确定|不知道|不清楚|不明白|待确认|未确认|没确认|没有确认|没问|没有问|还没问|未问)"
    if re.search(
        rf"(?:{uncertainty}).{{0,18}}(?:是否|有没|有没有|有无)?[^，。；！？!?]{{0,8}}{safety_term}|"
        rf"(?:是否|有没|有没有|有无)[^，。；！？!?]{{0,8}}{safety_term}.{{0,18}}{uncertainty}|"
        rf"{safety_term}.{{0,12}}(?:{uncertainty}|没(?:有)?问清楚|还没问清楚)",
        value,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        rf"(?:不是|并非|并不是|不代表).{{0,6}}(?:没(?:有)?|无).{{0,6}}{safety_term}|"
        r"(?:不是|并非|并不是|并不)\s*(?:什么|啥)?都(?:可以|行|能吃|不忌口)",
        value,
        flags=re.IGNORECASE,
    ):
        return False
    # “有没有忌口？”本身是提问，不是“没有忌口”的陈述。
    if re.search(rf"(?:有没|有没有|是否|有无)[^，。；！？!?]{{0,8}}{safety_term}", value):
        return False
    return True


def _has_source_evidence(value: Any, source_text: str) -> bool:
    """槽位值必须能由本轮用户原话直接支持，不维护领域词表或案例分支。"""
    candidate = _compact_search_text(str(value or ""))
    source = _compact_search_text(source_text)
    return bool(candidate and source and candidate in source)


def _planned_drinks_scene(text: str) -> bool:
    """只把本轮计划饮酒识别为配酒场景，排除否定和酒后不适。"""
    value = str(text or "")
    return bool(re.search(
        r"(?:要|想|准备|打算|计划|会|一起)?\s*"
        r"喝(?:点|一点|一些|几杯)?\s*(?:白酒|啤酒|红酒|黄酒|葡萄酒|酒)",
        value,
        flags=re.IGNORECASE,
    )) and not bool(re.search(
        r"(?:不|别|不要|不能|不想|避免)\s*喝|"
        r"昨天|昨晚|酒后|宿醉|喝多|喝醉|呕吐|吐了|胃.{0,4}不舒服",
        value,
        flags=re.IGNORECASE,
    ))


def _background_location_terms(text: str) -> list[str]:
    """抽取居住地/所在地点/籍贯等背景词，不把它们当选菜方向。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    out: list[str] = []
    patterns = (
        (
            r"(?:^|[，,。；;！？!?\s])(?:我|我们|本人|现在|目前|今天|今晚|夏天|冬天|人)?"
            r"(?:在|住在|位于)\s*([\u4e00-\u9fff]{2,8}?)(?="
            r"(?:招待|请|约|聚餐|吃饭|吃|做饭|想吃|想做|推荐|找|"
            r"生活|工作|出差|旅游|，|,|。|；|;|！|？|\s|$))"
        ),
        (
            r"(?:^|[，,。；;！？!?\s])(?:我|我们|本人|现在|目前|今天|今晚|人)?"
            r"(?:在|住在|位于)\s*([\u4e00-\u9fff]{2,8})(?=[，,。；;！？!?\s]|$)"
        ),
        (
            r"(?:都是|是|来自|老家(?:在|是)?|籍贯(?:在|是)?)\s*"
            r"([\u4e00-\u9fff]{2,8}?)(?:人|老乡|朋友|客人|[，,。；;！？!?\s]|$)"
        ),
        (
            r"\b(?:in|from|living in|based in)\s+([a-z][a-z .'-]{1,30})"
            r"(?=[,.!?;]|\s+(?:today|tonight|with|and|for)|$)"
        ),
    )
    for pattern in patterns:
        for matched in re.finditer(pattern, value, flags=re.IGNORECASE):
            term = str(matched.group(1) or "").strip(" ，,。；;！？!?~").lower()
            if term and term not in out:
                out.append(term)
    return out[:6]


def _has_explicit_regional_food_request(region: str, text: str) -> bool:
    """地点只有和菜系/口味诉求直接相连时，才可参与选菜。"""
    region_key = _compact_search_text(region)
    source = _compact_search_text(text)
    if not region_key or not source or region_key not in source:
        return False
    escaped = re.escape(region_key)
    return bool(
        re.search(
            rf"{escaped}(?:菜|菜系|口味|风味|料理|吃法|家乡菜|老家菜|本地菜|cuisine|food|style)",
            source,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"(?:想吃|要吃|喜欢吃|爱吃|来点|尝尝|想做|做点|"
            rf"want|crave|try|cook|make).{{0,12}}{escaped}",
            source,
            flags=re.IGNORECASE,
        )
        or re.search(
            rf"{escaped}(?:人)?.{{0,10}}(?:家乡菜|老家菜|本地菜|homecuisine|localfood)",
            source,
            flags=re.IGNORECASE,
        )
    )


def _cuisine_evidence_aliases(value: str) -> tuple[str, ...]:
    key = _compact_search_text(value)
    for group in _CUISINE_EVIDENCE_GROUPS:
        if key in {_compact_search_text(alias) for alias in group}:
            return group
    return (str(value or ""),)


def _has_explicit_cuisine_evidence(value: str, source_text: str) -> bool:
    """菜系槽位必须来自明确的吃法诉求，不能由所在地或籍贯推断。"""
    source = str(source_text or "")
    for alias in _cuisine_evidence_aliases(value):
        if not _has_source_evidence(alias, source):
            continue
        alias_key = _compact_search_text(alias)
        # “湘菜/西餐/日料”本身就是菜系；“湖南/江西/杭州”只是地点，
        # 后者还必须与“口味/菜系/想吃”等明确诉求相连。
        if (
            re.search(r"(?:菜|餐|料理|cuisine|food|style)$", alias_key)
            or alias_key in {"日料", "日式", "韩式", "家常", "western", "japanese", "korean"}
            or _has_explicit_regional_food_request(alias, source)
        ):
            return True
    return False


def _strip_background_query_token(token: str, source_text: str) -> str:
    """从模型 q 中去掉已建模的地点/饮酒背景，保留真实选菜条件。"""
    cleaned = str(token or "")
    for location in _background_location_terms(source_text):
        if not _has_explicit_regional_food_request(location, source_text):
            cleaned = re.sub(re.escape(location), " ", cleaned, flags=re.IGNORECASE)
    if _planned_drinks_scene(source_text):
        for marker in _DRINKING_BACKGROUND_TERMS:
            cleaned = re.sub(re.escape(marker), " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _explicit_flavor_facts(text: str) -> list[str]:
    """补齐轻量模型容易漏掉、但原话证据明确的常见口味事实。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    flavors: list[str] = []
    use_english = not bool(re.search(r"[\u4e00-\u9fff]", value))
    if (
        any(marker in value for marker in _BROAD_FLAVOR_MARKERS)
        and not re.search(
            r"(?:不要|别|不吃|不想吃|避免|少).{0,5}(?:重口味|重口|重油|重盐|够味)"
            r"|\b(?:not|avoid|less)\s+(?:bold|strong|heavy)\b",
            value,
            flags=re.IGNORECASE,
        )
    ):
        flavors.append("bold" if use_english else "重口味")
    if (
        any(marker in value for marker in (
            "下饭菜", "家常下饭", "更下饭", "下饭一点", "要下饭",
            "够味", "有味道一点", "savory", "savoury",
            "more flavorful", "more flavourful",
        ))
        and not re.search(r"(?:不想|不要|别|不吃).{0,4}下饭", value)
    ):
        flavors.append("savory" if use_english else "下饭")
    if any(marker in value for marker in (
        "清淡", "轻一点", "少油", "别太油", "不要太油",
        "lighter", "light", "less oily",
    )):
        flavors.append("light" if use_english else "清淡")
    if any(marker in value for marker in ("微辣", "少辣", "mildly spicy", "mild spicy")):
        flavors.append("mild" if use_english else "微辣")
    elif (
        any(marker in value for marker in (
            "香辣", "麻辣", "辣一点", "偏辣", "吃辣", "喜欢辣", "爱辣",
            "spicy",
        ))
        and not re.search(r"(?:不要|别|不吃|不想吃).{0,4}辣", value)
    ):
        flavors.append("spicy" if use_english else "辣")
    return _clean_list(flavors)


def _is_spicy_constraint(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in _SPICY_CONSTRAINT_VALUES or (
        bool(normalized)
        and any(marker in normalized for marker in ("辣", "spicy"))
    )


def _is_broad_flavor_value(value: str) -> bool:
    compact = _compact_search_text(value)
    if not compact:
        return False
    generic_suffixes = ("的菜", "菜", "口味", "flavordishes", "flavordish")
    candidates = {compact}
    for suffix in generic_suffixes:
        suffix_compact = _compact_search_text(suffix)
        if compact.endswith(suffix_compact):
            candidates.add(compact[:-len(suffix_compact)])
    return any(
        _compact_search_text(marker) in candidates
        for marker in _BROAD_FLAVOR_MARKERS
    )


def _recent_topic_from_turns(
    recent_turns: list[Any] | None,
    *,
    current_text: str,
    max_age_seconds: int = 900,
) -> str:
    """从同一 thread 的近期用户轮次中找到最近话题。

    只使用用户原话，不从助手回答猜实体，避免把 LLM 自由文本带入检索。
    handler 会在调用 Agent 前写入当前用户轮次，因此要先跳过最后一条同文本记录。
    """
    current_key = _compact_search_text(current_text)
    skipped_current = False
    now = time.time()
    checked_users = 0
    for turn in reversed(list(recent_turns or [])[-10:]):
        if str(_turn_value(turn, "role", "")).lower() != "user":
            continue
        text = re.sub(r"\s+", " ", str(_turn_value(turn, "content", ""))).strip(" ，。；！？?~")
        if not text:
            continue
        key = _compact_search_text(text)
        if not skipped_current and key == current_key:
            skipped_current = True
            continue
        checked_users += 1
        if checked_users > 3:
            break
        created_at = _turn_value(turn, "created_at", 0)
        try:
            if created_at and now - float(created_at) > max_age_seconds:
                continue
        except (TypeError, ValueError):
            pass
        lower = text.lower()
        if any(marker in lower for marker in _RECENT_TOPIC_IGNORE):
            continue
        if any(_compact_search_text(phrase) == key for phrase in _GENERIC_QUERY_PHRASES):
            continue

        # 常见短句抽取："什么是地衣" / "你知道白菜吗" / "我想吃鱼"。
        patterns = (
            r"^(?:什么是|介绍一下|说说|聊聊|了解一下)\s*([^\uff0c。；！？?]{1,24})$",
            r"(?:知道|了解|认识)\s*([^\uff0c。；！？?]{1,24})(?:吗|么)?$",
            r"(?:我)?(?:想吃|想做|想了解)\s*([^\uff0c。；！？?]{1,24})$",
        )
        for pattern in patterns:
            matched = re.search(pattern, text, flags=re.IGNORECASE)
            if matched:
                topic = matched.group(1).strip(" ，。；！？?吗么~")
                if topic:
                    return topic

        # 单词/短语是最可靠的近轮主题，例如"地衣"、"lichen"。
        if len(text) <= 24 and not re.search(r"[，。；！？?]", text):
            return text
    return ""


def _split_constraint_blob(blob: str) -> list[str]:
    """把过敏/忌口表达拆成原子食材，保留英文多词过敏原。"""
    raw = re.sub(r"\s+", " ", str(blob or "")).strip(" ，,。；;！？!?~")
    raw = re.sub(
        r"^(?:(?:还有|另外|其中|有)?"
        r"(?:一个|一位|一名|1个|1位|1名|[一二两三四五六七八九十0-9]+个|"
        r"[一二两三四五六七八九十0-9]+位)"
        r"(?:朋友|客人|同事|同学|家人|孩子|老人|人)\s*对\s*|"
        r"我|本人|自己|任何|所有|严重|重度|非常|特别|"
        r"i|me|any|all|severe(?:ly)?|serious(?:ly)?)\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.split(
        r"(?:但是|不过|但|却|然后|并且|而且|请|帮我|给我|直接(?:推荐|给)?|想吃|想做|推荐|"
        r"不要再问|别(?:再)?(?:问|询问|澄清)|无需(?:再)?(?:问|询问|澄清)|"
        r"不需要(?:再)?(?:问|询问|澄清)|你(?:来)?(?:安排|决定|定)|"
        r"你看着(?:办|安排|搭配|推荐)|按默认(?:来|安排|搭配)?|"
        r"\bbut\b|\bplease\b|\brecommend\b|\bi want\b|\bcan you\b)",
        raw,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if not raw:
        return []
    splitter = r"[、,，/]|(?:和|与|跟|及)|\band\b|\bor\b"
    out: list[str] = []
    for value in re.split(splitter, raw, flags=re.IGNORECASE):
        item = value.strip(" ，,。；;！？!?~")
        item = re.sub(r"^(?:任何|所有|any|all)\s*", "", item, flags=re.IGNORECASE)
        item = re.sub(r"\s*(?:严重|重度|非常|特别|severe|serious)$", "", item, flags=re.IGNORECASE)
        if item and item not in out and len(item) <= 32:
            out.append(item)
    return out


def _is_usable_explicit_constraint(value: str) -> bool:
    """只接受可直接用于过滤的原子约束，避免把"有人对……"当成食材。"""
    item = re.sub(r"\s+", "", str(value or "")).strip("，,。；;！？!?~").lower()
    if not item:
        return False
    # "不要再问/不要澄清"是在结束追问，不是"不要 + 食材"。
    if re.match(r"^(?:再)?(?:问|询问|澄清)", item):
        return False
    if re.search(
        r"不确定|不知道|不清楚|待确认|未确认|没(?:有)?确认|"
        r"有没|有没有|有无|是否|没(?:有)?问|还没问|未问",
        item,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(r"(?:朋友|客人|同事|同学|家人|孩子|老人|有人|一个人|一位|一名).{0,3}对", item):
        return False
    if item in {
        "食物", "某些食物", "一些食物", "有食物", "东西", "某些东西",
        "过敏原", "食物过敏", "过敏", "不清楚", "不知道", "不确定",
        "没", "没有", "无", "不", "不是", "并非",
        "food", "somefood", "something", "allergen", "allergens",
    }:
        return False
    return not bool(re.fullmatch(r"(?:我|本人|自己|有人|有些人|其他人|朋友|客人|人)", item))


def extract_explicit_excludes(text: str) -> list[str]:
    """确定性提取明确忌口和过敏原；不能依赖意图模型是否正确抽槽。"""
    out: list[str] = []
    patterns = (
        r"(?:不喜欢(?:吃)?|不爱吃|讨厌|不吃|不要|不放|不加|"
        r"避开|排斥|去掉|忌口)[：:]?\s*([^，。；！？]{1,24})",
        r"(?:^|[，,；;。！？!?])\s*(?:(?:我|本人)\s*对\s*)?"
        r"([^，,。；;！？!?]{1,30}?)(?:严重|重度|非常|特别)?\s*过敏",
        r"(?:i (?:do not|don't|dont) (?:eat|like)|without|no)\s+([^,.!?;]{1,24})",
        r"(?:i(?:\s+am|'m)?\s+)?(?:severely\s+|seriously\s+|highly\s+)?"
        r"allergic\s+to\s+([^.!?;]{1,48})",
        r"(?:i\s+have\s+(?:a|an)\s+)?([^,.!?;]{1,32})\s+allerg(?:y|ies)",
    )
    for pattern in patterns:
        for blob in re.findall(pattern, text or "", flags=re.IGNORECASE):
            for item in _split_constraint_blob(blob):
                normalized = item.lower()
                if (
                    item
                    and item not in out
                    and _is_usable_explicit_constraint(item)
                    and not any(word in normalized for word in _REFERENCE_WORDS)
                ):
                    out.append(item)
    return out[:12]


def _split_excludes(text: str) -> list[str]:
    """兼容旧内部名称。"""
    return extract_explicit_excludes(text)


_EMPTY_CONSTRAINT_VALUES = {
    "忌口", "没有忌口", "没忌口", "无忌口", "没有", "无", "都可以",
    "none", "no restrictions", "no restriction", "no allergies", "nothing",
}

_CUISINE_CONSTRAINT_ALIASES = {
    "西餐": "western", "western": "western",
    "日料": "japanese", "日式": "japanese", "japanese": "japanese",
    "韩餐": "korean", "韩式": "korean", "korean": "korean",
    "意大利菜": "italian", "italian": "italian",
    "法餐": "french", "法国菜": "french", "french": "french",
    "美式": "american", "美国菜": "american", "american": "american",
}

_INGREDIENT_EQUIVALENT_GROUPS = (
    frozenset({"香菜", "芫荽", "胡荽", "cilantro", "coriander", "coriander leaf", "coriander leaves"}),
)
_EXACT_RECIPE_SYNONYM_GROUPS = (
    ("红薯", "地瓜", "番薯"),
    ("西红柿", "番茄"),
    ("马铃薯", "土豆"),
    ("西葫芦", "角瓜"),
    ("菜花", "花椰菜"),
)
_HALAL_DIET_MARKERS = (
    "清真", "回族", "穆斯林", "伊斯兰饮食", "halal", "muslim",
)
_HALAL_FORBIDDEN_VARIANTS = frozenset({
    "猪肉", "五花肉", "猪油", "荤油", "猪骨", "猪排", "排骨", "猪蹄",
    "猪肝", "猪肚", "猪耳", "猪血", "猪皮",
    "料酒", "黄酒", "绍兴酒", "米酒", "白酒", "啤酒", "葡萄酒", "红酒",
    "朗姆酒", "威士忌", "酒酿", "味醂",
    "pork", "pork belly", "pork ribs", "pork bone", "bacon", "ham", "lard",
    "prosciutto", "pancetta", "cooking wine", "rice wine", "shaoxing wine",
    "wine", "beer", "liquor", "rum", "whisky", "whiskey", "mirin",
})


def _split_typed_excludes(values: list[str]) -> tuple[list[str], list[str]]:
    """把菜系排除与食材排除分开，避免拿"西餐"去匹配食材字符串。"""
    ingredients: list[str] = []
    cuisines: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        canonical = _CUISINE_CONSTRAINT_ALIASES.get(value.lower())
        target = cuisines if canonical else ingredients
        normalized = canonical or value
        if normalized and normalized not in target:
            target.append(normalized)
    return ingredients, cuisines


def _canonical_preference_value(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return _CUISINE_CONSTRAINT_ALIASES.get(normalized) or normalized


def _preference_value_matches(left: Any, right: Any) -> bool:
    left_value = _canonical_preference_value(left)
    right_value = _canonical_preference_value(right)
    if not left_value or not right_value:
        return False
    if _is_spicy_constraint(left_value) and _is_spicy_constraint(right_value):
        return True
    if left_value == right_value:
        return True
    left_compact = _compact_search_text(left_value)
    right_compact = _compact_search_text(right_value)
    return bool(
        left_compact
        and right_compact
        and (
            left_compact == right_compact
            or (
                min(len(left_compact), len(right_compact)) >= 2
                and (
                    left_compact in right_compact
                    or right_compact in left_compact
                )
            )
        )
    )


def remembered_preference_conflicts(
    request: "SearchRequest",
    preferences: dict[str, list[str]] | None,
) -> list[dict[str, str]]:
    """识别本轮正向要求与既有普通喜恶的冲突；过敏仍由独立安全门禁处理。"""
    positive_by_dimension = {
        "cuisine": list(request.cuisines),
        "ingredient": [
            *request.ingredients,
            *request.required_ingredients,
            *request.dishes,
        ],
        "flavor": list(request.flavors),
        "method": list(request.methods),
    }
    conflicts: list[dict[str, str]] = []
    suppressed = list(request.suppressed_inherited)
    for dislike in (preferences or {}).get("dislikes") or []:
        value = str(dislike or "").strip()
        if not value or any(
            _preference_value_matches(value, item) for item in suppressed
        ):
            continue
        canonical = _canonical_preference_value(value)
        dimensions = (
            ("cuisine",)
            if canonical in set(_CUISINE_CONSTRAINT_ALIASES.values())
            else tuple(positive_by_dimension)
        )
        for dimension in dimensions:
            if any(
                _preference_value_matches(value, item)
                for item in positive_by_dimension[dimension]
            ):
                conflicts.append({
                    "bucket": "dislikes",
                    "dimension": dimension,
                    "value": value,
                    "canonical": canonical,
                })
                break
    negative_by_dimension = {
        "cuisine": list(request.exclude_cuisines),
        "ingredient": list(request.exclude),
        "flavor": [*request.exclude, *request.avoid],
        "method": [*request.exclude, *request.avoid],
    }
    for liked in (preferences or {}).get("likes") or []:
        value = str(liked or "").strip()
        if not value or any(
            _preference_value_matches(value, item) for item in suppressed
        ):
            continue
        canonical = _canonical_preference_value(value)
        dimensions = (
            ("cuisine",)
            if canonical in set(_CUISINE_CONSTRAINT_ALIASES.values())
            else tuple(negative_by_dimension)
        )
        for dimension in dimensions:
            if any(
                _preference_value_matches(value, item)
                for item in negative_by_dimension[dimension]
            ):
                conflicts.append({
                    "bucket": "likes",
                    "dimension": dimension,
                    "value": value,
                    "canonical": canonical,
                })
                break
    return conflicts


def exact_recipe_synonym_queries(request: "SearchRequest", *, limit: int = 6) -> list[str]:
    """为明确菜名生成有限、可审计的同义词查询，不让模型自由改写菜名。"""
    if request.task_operation != "exact_search" or request.required_ingredients:
        return []
    targets = _clean_list([
        *request.dishes,
        request.canonical_question,
        request.query,
    ], limit=3)
    out: list[str] = []
    for target in targets:
        for group in _EXACT_RECIPE_SYNONYM_GROUPS:
            matched = next((term for term in group if term in target), "")
            if not matched:
                continue
            for synonym in group:
                candidate = target.replace(matched, synonym).strip()
                if candidate and candidate != target and candidate not in out:
                    out.append(candidate)
                if len(out) >= limit:
                    return out
    return out


class SearchRequest(BaseModel):
    """跨意图、检索、排序和解释层共享的稳定输入。"""

    version: str = "search_request_v1"
    original_text: str = ""
    query: str = ""
    # 顶层 intent 只判断 recipe_search / recipe_recommend；这里记录真正决定
    # 检索与澄清行为的业务操作。
    task_operation: str = "fuzzy_search"
    canonical_question: str = ""
    dishes: list[str] = Field(default_factory=list)
    ingredients: list[str] = Field(default_factory=list)
    cuisines: list[str] = Field(default_factory=list)
    flavors: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    scenes: list[str] = Field(default_factory=list)
    meals: list[str] = Field(default_factory=list)
    dietary_constraints: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    exclude_cuisines: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    # 用户已经明确回答过"是否有过敏、忌口或宗教饮食要求"。即使答案为
    # "没有"，也要保留这个事实，避免下一轮反复询问或越过安全澄清。
    constraints_confirmed: bool = False
    # "A 和 B 怎么做"要求两个主料同时出现；普通库存推荐仍可只命中其中之一。
    required_ingredients: list[str] = Field(default_factory=list)
    detail_requested: bool = False
    soft_preferences: list[str] = Field(default_factory=list)
    # 从账号真实搜索/开火历史提取的弱行为线索。当前仅用于审计和后续画像评估，
    # 不进入本轮召回或表达，避免把一次旧搜索误当成长久偏好。
    memory_terms: list[str] = Field(default_factory=list)
    memory_note: dict[str, str] = Field(default_factory=dict)
    # 同一会话近期用户轮次解决的省略主语；优先级高于账号长期饮食历史。
    context_note: dict[str, str] = Field(default_factory=dict)
    party_size: int | None = Field(default=None, ge=1, le=100)
    result_limit: int = Field(default=2, ge=1, le=10)
    # 区分"用户明确要三道"与内部默认返回数量。推荐表达默认只重点讲 1～2 道，
    # 但用户明确给出数量时必须尊重原要求。
    explicit_result_limit: bool = False
    # 显式菜单请求必须保留菜/汤结构，不能压扁成一次 top-k 搜索。
    menu_dish_count: int = Field(default=0, ge=0, le=10)
    menu_soup_count: int = Field(default=0, ge=0, le=10)
    # 仅作用于部分来宾/部分菜位的偏好，不写入账号长期画像。
    scoped_preferences: list[dict[str, Any]] = Field(default_factory=list)
    # 当前任务主动屏蔽的继承偏好；不会删除账号长期画像。
    suppressed_inherited: list[str] = Field(default_factory=list)
    # 只有确实进入本轮过滤或软排序的用户偏好才进入这里，供回复自然说明依据。
    applied_memory_constraints: list[dict[str, str]] = Field(default_factory=list)

    @property
    def is_menu_plan(self) -> bool:
        return self.menu_dish_count > 0 and self.menu_soup_count > 0

    @classmethod
    def from_intent_raw(
        cls,
        raw: dict | None,
        *,
        original_text: str,
        keywords: list[str] | None = None,
    ) -> "SearchRequest":
        structured_raw = (raw or {}).get("s") or {}
        structured = dict(structured_raw) if isinstance(structured_raw, dict) else {}
        aliases = {
            "dishes": ("dishes", "dish"),
            "ingredients": ("ingredients", "ingredient"),
            "cuisines": ("cuisines", "cuisine"),
            "flavors": ("flavors", "flavor"),
            "methods": ("methods", "method"),
            "scenes": ("scenes", "scene"),
            "meals": ("meals", "meal"),
            "dietary_constraints": ("dietary_constraints", "diet"),
            "exclude": ("exclude",),
            "avoid": ("avoid",),
            "required_ingredients": ("required_ingredients", "required_ingredient"),
            "soft_preferences": ("soft_preferences",),
        }
        values: dict[str, Any] = {
            "original_text": str(original_text or "").strip(),
            "query": str(structured.get("q") or structured.get("query") or "").strip(),
        }
        for target, candidates in aliases.items():
            raw_value = next((structured.get(key) for key in candidates if structured.get(key)), [])
            values[target] = _clean_list(raw_value)

        if not values["query"]:
            values["query"] = " ".join(_clean_list(keywords or [], limit=6)).strip()
        # 数量抱怨不是新的召回方向。丢弃分类器可能生成的“五个菜”
        # query，后续只从 pending request 恢复真实的下酒/餐次等条件。
        if is_menu_quantity_dissatisfaction(original_text):
            values["query"] = ""
        # 负向条件只信任用户原话里的明确表达，不能从场景推断新的忌口。
        extracted_excludes = [
            item for item in _clean_list(_split_excludes(original_text))
            if str(item).strip().lower() not in _EMPTY_CONSTRAINT_VALUES
        ]
        values["avoid"] = _clean_list(
            [item for item in extracted_excludes if item in _SOFT_NEGATIVE_TERMS]
        )
        hard_excludes = [
            item for item in extracted_excludes
            if item not in _SOFT_NEGATIVE_TERMS
            and not _is_generic_constraint_confirmation(item)
        ]
        values["exclude"], values["exclude_cuisines"] = _split_typed_excludes(hard_excludes)
        original_lower = original_text.lower()

        # 核心对象和场景都必须有本轮原话证据。该校验只比较文本证据，
        # 不认识任何具体菜名、食材或场景，因此换业务案例也无需改代码。
        query_tokens: list[str] = []
        for raw_token in re.split(r"\s+", str(values["query"] or "").strip()):
            if not _has_source_evidence(raw_token, original_text):
                continue
            if (
                has_explicit_no_dietary_restrictions(original_text)
                and _is_generic_constraint_confirmation(raw_token)
            ):
                continue
            token = _strip_background_query_token(raw_token, original_text)
            if token:
                query_tokens.append(token)
        values["query"] = " ".join(query_tokens)
        values["dishes"] = [
            dish for dish in values["dishes"]
            if _has_source_evidence(dish, original_text)
        ]
        values["ingredients"] = [
            ingredient for ingredient in values["ingredients"]
            if _has_source_evidence(ingredient, original_text)
        ]
        values["required_ingredients"] = _clean_list(
            [
                ingredient
                for ingredient in values["required_ingredients"]
                if _has_source_evidence(ingredient, original_text)
            ]
            + _extract_required_menu_ingredients(original_text)
        )
        # 轻量模型偶尔把"重口味"这类口味方向塞进主料/菜名。它们只能作为
        # 软推荐方向，不能进入明确食材硬校验，否则真实召回会被全部剔除。
        broad_flavor_values = [
            item for item in [*values["ingredients"], *values["dishes"]]
            if _is_broad_flavor_value(item)
        ]
        if broad_flavor_values:
            values["ingredients"] = [
                item for item in values["ingredients"] if not _is_broad_flavor_value(item)
            ]
            values["dishes"] = [
                item for item in values["dishes"] if not _is_broad_flavor_value(item)
            ]
        detail_requested = bool(re.search(
            r"怎么做|如何做|如何制作|做法|详细步骤|具体步骤|how\s+to\s+(?:make|cook)|recipe\s+steps",
            original_text,
            flags=re.IGNORECASE,
        ))
        values["detail_requested"] = detail_requested
        if (
            detail_requested
            and len(values["ingredients"]) > 1
            and re.search(r"和|与|跟|及|、|,|，|\band\b", original_text, flags=re.IGNORECASE)
            and not re.search(r"或|或者|\bor\b", original_text, flags=re.IGNORECASE)
        ):
            values["required_ingredients"] = list(values["ingredients"])
        raw_category = str((raw or {}).get("c") or "")
        values["task_operation"] = (
            "exact_search"
            if detail_requested or values["dishes"]
            else ("fuzzy_recommend" if raw_category == "recipe_recommend" else "fuzzy_search")
        )
        values["scenes"] = [
            scene for scene in values["scenes"]
            if _is_grounded_scene(scene, original_text)
        ]
        # 所在地和来宾籍贯是场景事实，不是菜系或口味偏好。菜系归一值可以
        # 与原话不同（“湖南口味”→“湘菜”），但必须能追溯到明确的吃法诉求。
        values["cuisines"] = [
            cuisine for cuisine in values["cuisines"]
            if _has_explicit_cuisine_evidence(cuisine, original_text)
        ]
        explicit_cuisine_direction = bool(values["cuisines"])
        values["flavors"] = [
            flavor for flavor in values["flavors"]
            if _has_source_evidence(flavor, original_text)
            or (
                explicit_cuisine_direction
                and bool(re.search(
                    r"口味|风味|flavou?r|taste",
                    original_text,
                    flags=re.IGNORECASE,
                ))
            )
        ]
        # 用户主动把炎热天气作为选菜条件时，可以直接据此检索并给建议。
        # 这里只保留"天热"这一客观场景，不推断"清淡、补水、低卡"等新事实。
        hot_weather = bool(re.search(
            r"(?:天气|今天|今儿|外面).{0,12}(?:有点|有些|比较|挺|很)?热|"
            r"(?:有点|有些|天气|天儿)热|炎热|闷热|"
            r"\b(?:hot weather|hot day|it(?:'s| is) hot)\b",
            original_text,
            flags=re.IGNORECASE,
        ))
        if hot_weather:
            values["scenes"] = _clean_list(
                values["scenes"] + [
                    "hot weather"
                    if not re.search(r"[\u4e00-\u9fff]", original_text)
                    else "天热"
                ]
            )
        effort_markers = (
            "简单", "快手", "省事", "别太复杂", "simple", "easy", "quick",
        )
        if (
            any(marker in original_lower for marker in effort_markers)
            and not any(
                marker in str(scene).lower()
                for scene in values["scenes"]
                for marker in effort_markers
            )
        ):
            default_effort = (
                "quick"
                if not re.search(r"[\u4e00-\u9fff]", original_text)
                else "快手"
            )
            values["scenes"] = _clean_list(
                values["scenes"] + [default_effort]
            )
        if re.search(
            r"家常(?:菜|一点|些|口味)?|\bhome[- ]?style\b|\bhomestyle\b",
            original_text,
            flags=re.IGNORECASE,
        ):
            values["cuisines"] = _clean_list(
                values["cuisines"] + [
                    "home_style"
                    if not re.search(r"[\u4e00-\u9fff]", original_text)
                    else "家常菜"
                ]
            )
        planned_drinks = _planned_drinks_scene(original_text)
        if planned_drinks or any(marker in original_lower for marker in (
            "下酒", "佐酒", "配酒", "appetizer for drinks", "with drinks",
        )):
            drinks_scene = (
                "with drinks"
                if not re.search(r"[\u4e00-\u9fff]", original_text)
                else "下酒"
            )
            values["scenes"] = _clean_list(
                values["scenes"] + [drinks_scene]
            )
        values["flavors"] = _clean_list(
            values["flavors"] + _explicit_flavor_facts(original_text)
        )

        party_size = _extract_party_size(original_text)
        requested_limit = _extract_result_limit(original_text)
        menu_counts = _extract_menu_counts(original_text)
        values["party_size"] = party_size
        if party_size and values["required_ingredients"]:
            required_keys = {
                _compact_search_text(item)
                for item in values["required_ingredients"]
                if _compact_search_text(item)
            }
            values["ingredients"] = [
                item
                for item in values["ingredients"]
                if _compact_search_text(item) not in required_keys
            ]
        if menu_counts:
            values["menu_dish_count"], values["menu_soup_count"] = menu_counts
            values["result_limit"] = sum(menu_counts)
            values["explicit_result_limit"] = True
        else:
            values["result_limit"] = requested_limit or _default_result_limit(party_size)
            values["explicit_result_limit"] = requested_limit is not None

        values["scoped_preferences"] = _extract_scoped_preferences(original_text)
        for scoped in values["scoped_preferences"]:
            scoped_cuisines = {str(item).lower() for item in scoped.get("cuisines") or []}
            scoped_flavors = {str(item).lower() for item in scoped.get("flavors") or []}
            values["cuisines"] = [
                item for item in values["cuisines"] if str(item).lower() not in scoped_cuisines
            ]
            values["flavors"] = [
                item for item in values["flavors"] if str(item).lower() not in scoped_flavors
            ]
        if party_size:
            corrected_query = re.sub(
                r"[0-9一二两三四五六七八九十]{1,3}\s*人餐",
                f"{party_size}人餐",
                values["query"],
            )
            values["query"] = corrected_query or f"{party_size}人餐"
            # “N人餐”仅用于菜单规模，不是语义召回方向；真实检索由下酒、餐次、
            # 明确主料/口味等事实驱动。没有其它方向时仍保留，避免空查询。
            if re.fullmatch(r"[0-9一二两三四五六七八九十百]+人餐", values["query"]):
                has_retrieval_direction = any((
                    values["dishes"], values["ingredients"], values["cuisines"],
                    values["flavors"], values["methods"], values["scenes"],
                    values["meals"], values["dietary_constraints"],
                ))
                if has_retrieval_direction:
                    values["query"] = ""

        # 轻量意图模型偶尔会把"家里有四口人，帮我推荐……"整段塞进食材字段。
        values["ingredients"] = [
            item for item in values["ingredients"]
            if not re.search(r"[0-9一二两三四五六七八九十]+\s*(?:口人|口|个人|人吃|人份)", item)
            and not any(marker in item for marker in ("帮我推荐", "推荐适合", "几个人", "几口人"))
        ]

        # 否定条件不能同时作为正向召回标签。例如"不要汤"不能生成 meals=["汤"]，
        # "不喜欢太甜"也不能生成 flavors=["甜"]。
        negative_terms = _clean_list(values["exclude"] + values["avoid"])
        for field in ("ingredients", "flavors", "methods", "meals"):
            values[field] = [
                item for item in values[field]
                if not any(
                    item == negative or item in negative or negative in item
                    for negative in negative_terms
                )
            ]
        values["cuisines"] = [
            item for item in values["cuisines"]
            if not any(
                _preference_value_matches(item, excluded)
                for excluded in values["exclude_cuisines"]
            )
        ]

        if any(word in original_lower for word in ("纯素", "全素", "素食", "素菜", "vegetarian", "vegan")):
            diet_value = "vegan" if "vegan" in original_lower else (
                "vegetarian" if "vegetarian" in original_lower else "素食"
            )
            values["dietary_constraints"] = _clean_list(values["dietary_constraints"] + [diet_value])
        if any(marker in original_lower for marker in _HALAL_DIET_MARKERS):
            values["dietary_constraints"] = _clean_list(
                values["dietary_constraints"] + ["halal"]
            )
        values["constraints_confirmed"] = bool(
            values["dietary_constraints"]
            or values["exclude"]
            or values["exclude_cuisines"]
            or has_explicit_no_dietary_restrictions(original_text)
        )
        preference_meta = bool(re.search(
            r"(?:没|没有|哪里|哪儿|什么时候|是否|有没|有没有).{0,12}(?:说|提|讲|记)"
            r"|\b(?:did i|when did i|where did i|have i|i did(?:n't| not)|i never)\b",
            original_lower,
        ))
        for word in ("减肥", "减脂", "低脂", "控卡", "low-fat", "low fat", "weight loss"):
            if word in original_lower and not preference_meta:
                values["scenes"] = _clean_list(values["scenes"] + [word])
        return cls(**values)

    def has_selection_direction(self) -> bool:
        """当前事实是否已有可用于选菜的正向方向。"""
        if any((
            self.dishes,
            self.ingredients,
            self.cuisines,
            self.flavors,
            self.methods,
            self.required_ingredients,
            self.scoped_preferences,
        )) or _has_actionable_scene(self.scenes):
            return True

        residual = _compact_search_text(self.query)
        # 先移除明确的负向事实，避免"不吃香菜"被误判为正向选菜方向。
        for value in sorted(
            [*self.exclude, *self.exclude_cuisines, *self.avoid],
            key=lambda item: len(_compact_search_text(str(item))),
            reverse=True,
        ):
            compact = _compact_search_text(str(value))
            if compact:
                residual = residual.replace(compact, "")
        removable = [
            *_GENERIC_SEARCH_WORDS,
            *_GENERIC_QUERY_PHRASES,
            *_GENERIC_QUERY_FILLERS,
            *_BROAD_SCENE_TERMS,
            *self.meals,
            *self.scenes,
        ]
        for value in sorted(
            removable,
            key=lambda item: len(_compact_search_text(str(item))),
            reverse=True,
        ):
            compact = _compact_search_text(str(value))
            if compact:
                residual = residual.replace(compact, "")
        number = r"0-9一二两三四五六七八九十百"
        # 先移除“11人餐”，再处理普通“11人”，避免前者被部分替换后残留“餐”。
        residual = re.sub(rf"[{number}]+人餐", "", residual)
        residual = re.sub(rf"[{number}]+个?人(?:吃|用餐)?", "", residual)
        residual = re.sub(
            rf"(?:推荐|来|要)?[{number}]+(?:道|个)(?:菜|菜谱|食谱)?",
            "",
            residual,
        )
        return bool(residual)

    def has_broad_flavor_direction(self) -> bool:
        """是否只给出了可展开的宽泛口味方向，而非明确菜名/主料。"""
        if self.dishes or self.ingredients or self.required_ingredients:
            return False
        explicit = _explicit_flavor_facts(f"{self.original_text} {self.query}")
        values = [*self.flavors, *explicit]
        return any(
            _is_broad_flavor_value(value)
            or str(value or "").strip().lower() == "bold"
            for value in values
        )

    def clarification_dimension(self) -> str | None:
        """泛推荐缺少可用方向时，只追问一个最影响结果的问题。

        具体菜名、食材、菜系、口味、做法、饮食约束等任一项已经明确，就直接
        检索；"晚饭吃什么 / 朋友来吃饭 / 推荐一下"这类只有场景没有方向的请求
        先澄清。用户明确说"随便、直接推荐、别问"时尊重其选择，不追问。
        """
        if self.is_menu_plan:
            return None
        if delegates_recommendation_choice(self.original_text):
            return None
        text = re.sub(r"\s+", " ", str(self.original_text or "")).strip().lower()

        actionable = any((
            self.dishes,
            self.ingredients,
            self.cuisines,
            self.flavors,
            self.methods,
            self.dietary_constraints,
            self.exclude,
            self.avoid,
        ))
        if actionable or self.context_note:
            return None

        # "快手早餐 / 减脂晚餐"虽然没有主料，但已经给了足够明确的筛选方向。
        if _has_actionable_scene(self.scenes):
            return None

        # 轻量模型有时只把具体对象留在 q 中。剥掉泛化措辞后仍有实词，就不要
        # 因槽位漏抽而多问一轮。
        residual = _compact_search_text(self.query)
        removable = [
            *_GENERIC_SEARCH_WORDS,
            *_GENERIC_QUERY_PHRASES,
            *_GENERIC_QUERY_FILLERS,
            *_BROAD_SCENE_TERMS,
            *self.meals,
            *self.scenes,
        ]
        for value in sorted(removable, key=lambda item: len(_compact_search_text(str(item))), reverse=True):
            compact = _compact_search_text(str(value))
            if compact:
                residual = residual.replace(compact, "")
        residual = re.sub(r"[0-9一二两三四五六七八九十百]+人餐", "", residual)
        if residual:
            return None

        if self.party_size or any(marker in text for marker in _BROAD_SCENE_TERMS):
            return "party_preferences"
        if (
            self.meals
            or any(marker in text for marker in ("晚饭", "午饭", "早餐", "晚餐", "午餐", "dinner", "lunch", "breakfast"))
            or any(marker in text for marker in _EFFORT_OR_EMOTION_TERMS)
        ):
            return "available_ingredients"
        return "ingredient_or_flavor"

    def recommendation_clarification_dimension(self) -> str | None:
        """泛推荐才追问；已明确主料/口味/做法时直接给可比较菜谱。

        人数会影响整桌菜单，但不应阻断"带牛肉的辣菜有哪些"这种单菜推荐。
        显式菜单请求仍由 ``is_menu_plan`` 的结构化数量负责。
        """
        text = re.sub(r"\s+", " ", str(self.original_text or "")).strip().lower()
        has_party_context = bool(
            self.party_size
            or any(marker in text for marker in _BROAD_SCENE_TERMS)
        )
        has_selection_preference = self.has_selection_direction()

        # 招待/多人场景中，人数、季节和地点都只是背景，不能替代安全事实。
        # 任何"直接推荐/你安排"都不能越过该门禁；确认后未给口味则使用
        # 默认多样化策略，不再把软偏好作为必填澄清项。
        if has_party_context:
            if not self.constraints_confirmed:
                return "party_constraints"
            return None

        if self.is_menu_plan or delegates_recommendation_choice(self.original_text):
            return None

        # 非多人推荐中，用户已经明确说明无忌口/无过敏，即可采用默认口味；
        # 无需为了补充软偏好再阻断一次明确推荐请求。
        if self.constraints_confirmed:
            return None

        has_direction = bool(
            has_selection_preference or self.dietary_constraints or self.exclude
        )
        if has_direction:
            return None
        if not self.party_size:
            return "recommendation_basics"
        return "flavor_preferences"

    def search_clarification_dimension(self) -> str | None:
        """模糊搜索只有负向约束或泛问法时，先补一个正向事实再检索。"""
        if self.detail_requested or self.dishes or self.required_ingredients:
            return None
        positive = any((
            self.ingredients, self.cuisines, self.flavors, self.methods,
            self.dietary_constraints,
        ))
        if positive or self.context_note:
            return None
        if delegates_recommendation_choice(self.original_text):
            return None
        # 纯忌口/排除仍然不构成选菜方向；先问主料或口味，硬约束会保留。
        if self.exclude or self.exclude_cuisines or self.avoid:
            return "ingredient_or_flavor"
        residual = _compact_search_text(self.query)
        for value in sorted(
            [*_GENERIC_QUERY_PHRASES, *_GENERIC_QUERY_FILLERS, *_GENERIC_SEARCH_WORDS],
            key=lambda item: len(_compact_search_text(str(item))),
            reverse=True,
        ):
            residual = residual.replace(_compact_search_text(str(value)), "")
        if residual:
            return None
        if self.meals:
            return "available_ingredients"
        return "ingredient_or_flavor"

    def with_task_operation(self, mode: str) -> "SearchRequest":
        """在上下文合并完成后确定业务操作，但尚不生成检索问题。"""
        data = self.model_dump()
        if (
            self.detail_requested
            or self.dishes
            or (self.required_ingredients and not self.party_size)
        ):
            operation = "exact_search"
        elif mode == "recommend":
            operation = "fuzzy_recommend"
        else:
            operation = "fuzzy_search"
        data["task_operation"] = operation
        # 旧 pending 状态可能已有标准问题；新一轮重新合并事实后必须重新冻结。
        data["canonical_question"] = ""
        return SearchRequest(**data)

    def freeze_canonical_question(self, *, lang: str = "zh") -> "SearchRequest":
        """事实充分性门禁通过后，冻结供所有下游共用的标准问题。"""
        data = self.model_dump()
        data["canonical_question"] = self.retrieval_query(lang=lang)
        return SearchRequest(**data)

    def with_recommendation_menu_defaults(self) -> "SearchRequest":
        """按人数把推荐转成确定性的菜+汤组合；显式数量/菜单保持用户原意。"""
        if self.is_menu_plan or self.explicit_result_limit:
            return self
        people = self.party_size or 1
        if people <= 1:
            dish_count, soup_count = 1, 1
        elif people == 2:
            dish_count, soup_count = 2, 1
        elif people <= 4:
            dish_count, soup_count = 3, 1
        elif people <= 6:
            dish_count, soup_count = 4, 1
        elif people <= 8:
            dish_count, soup_count = 5, 1
        else:
            dish_count, soup_count = 6, 2
        data = self.model_dump()
        data["party_size"] = people
        data["menu_dish_count"] = dish_count
        data["menu_soup_count"] = soup_count
        data["result_limit"] = dish_count + soup_count
        return SearchRequest(**data)

    # 话题切换字段：新一轮有值时替换旧值，避免红烧肉切到鸡胸肉时旧食材残留。
    _REPLACE_FIELDS = frozenset({
        "meals", "required_ingredients", "ingredients", "dishes",
    })

    def merge_clarification(self, followup: "SearchRequest") -> "SearchRequest":
        """把用户对澄清问题的回答合回原请求，下一轮直接进入真实检索。"""
        data = self.model_dump()

        for field in _LIST_FIELDS:
            previous = list(data.get(field) or [])
            current = list(getattr(followup, field) or [])
            # 话题切换字段（餐次、主食材、食材、菜名）：有值则替换，无值则保留。
            if field in self._REPLACE_FIELDS and current:
                data[field] = _clean_list(current)
            else:
                data[field] = _clean_list(previous + current)

        # 话题切换检测：新请求有独立主食材时，用新 query 替换旧 query，
        # 避免"红烧肉→鸡胸肉"时旧 query 残留影响语义检索方向。
        topic_changed = bool(
            getattr(followup, "required_ingredients", None)
            or getattr(followup, "ingredients", None)
            or getattr(followup, "dishes", None)
        )
        query_parts: list[str] = []
        queries_to_merge = [followup.query] if topic_changed else [self.query, followup.query]
        for value in queries_to_merge:
            value = re.sub(r"\s+", " ", str(value or "")).strip()
            if not value:
                continue
            cleaned = value
            for generic in sorted(
                (*_GENERIC_QUERY_PHRASES, *_GENERIC_QUERY_FILLERS),
                key=lambda item: len(_compact_search_text(item)),
                reverse=True,
            ):
                if generic:
                    cleaned = re.sub(re.escape(generic), " ", cleaned, flags=re.IGNORECASE)
            # 餐次、口味等结构化字段仍会进入 retrieval_query；这里去掉纯问法，
            # 避免"晚饭吃什么 鸡蛋"把语义检索拖向泛结果。
            value = re.sub(r"\s+", " ", cleaned).strip(" ，,。；;！？!?~")
            if not _compact_search_text(value):
                continue
            if value not in query_parts:
                query_parts.append(value)
        data["query"] = " ".join(query_parts).strip() or followup.query or self.query
        data["original_text"] = "；补充：".join(
            value for value in (self.original_text.strip(), followup.original_text.strip()) if value
        )
        data["party_size"] = followup.party_size or self.party_size
        data["constraints_confirmed"] = bool(
            self.constraints_confirmed or followup.constraints_confirmed
        )
        if followup.explicit_result_limit:
            data["result_limit"] = followup.result_limit
            data["explicit_result_limit"] = True
        else:
            data["result_limit"] = self.result_limit
            data["explicit_result_limit"] = self.explicit_result_limit
        data["context_note"] = followup.context_note or self.context_note
        if followup.is_menu_plan:
            data["menu_dish_count"] = followup.menu_dish_count
            data["menu_soup_count"] = followup.menu_soup_count
        if followup.scoped_preferences:
            data["scoped_preferences"] = list(followup.scoped_preferences)
            scoped_cuisines = {
                str(item).lower()
                for scoped in followup.scoped_preferences
                for item in (scoped.get("cuisines") or [])
            }
            scoped_flavors = {
                str(item).lower()
                for scoped in followup.scoped_preferences
                for item in (scoped.get("flavors") or [])
            }
            data["cuisines"] = [
                item for item in data["cuisines"] if str(item).lower() not in scoped_cuisines
            ]
            data["flavors"] = [
                item for item in data["flavors"] if str(item).lower() not in scoped_flavors
            ]
        # 一次旧搜索不是本轮偏好；只保留审计字段，仍不进入 retrieval_query。
        data["memory_terms"] = _clean_list(self.memory_terms + followup.memory_terms, limit=2)
        data["memory_note"] = followup.memory_note or self.memory_note
        return SearchRequest(**data)

    def merge_preferences(self, preferences: dict[str, list[str]] | None) -> "SearchRequest":
        """合并用户明确沉淀的偏好；本轮例外必须先由冲突澄清显式授权。"""
        data = self.model_dump()
        preferences = preferences or {}
        suppressed = list(data.get("suppressed_inherited") or [])
        dislikes = [
            item
            for item in (preferences.get("dislikes") or [])
            if not any(
                _preference_value_matches(item, blocked)
                for blocked in suppressed
            )
        ]
        allergens = list(preferences.get("allergens") or [])
        if any(_is_spicy_constraint(item) for item in data["flavors"]):
            spicy_dislikes = [
                item for item in dislikes if _is_spicy_constraint(item)
            ]
            if spicy_dislikes:
                dislikes = [
                    item for item in dislikes if not _is_spicy_constraint(item)
                ]
                data["suppressed_inherited"] = _clean_list([
                    *suppressed,
                    *spicy_dislikes,
                ])
        applied = [
            dict(item)
            for item in data.get("applied_memory_constraints") or []
            if isinstance(item, dict)
        ]
        original_exclude = list(data["exclude"])
        original_exclude_cuisines = list(data["exclude_cuisines"])
        original_diet = list(data["dietary_constraints"])
        # ── 改法 A：所有记忆 dislikes 都走软过滤（avoid），不再硬过滤 ──
        #
        # 旧实现把"不喜欢 X"按 _SOFT_NEGATIVE_TERMS 分成两组：
        #   软组（辣/油/甜 等口味词）→ avoid（软过滤）
        #   硬组（其他食材/菜系）→ exclude / exclude_cuisines（硬过滤）
        #
        # 问题：用户本轮说"想吃川菜"时，记忆里的"不喜欢辣"会被硬合并，
        # 把所有川菜静默过滤掉。用户体验僵硬。
        #
        # 新实现：所有 dislikes 都走 avoid（软过滤）。过敏原 (allergens) 仍然
        # 走 exclude（硬过滤，安全红线）。用户本轮显式说的"不要 X"也仍然走
        # exclude（本轮明确诉求）。
        #
        # 回退：把 `dislikes` 重新拆成 soft/hard 两组即可恢复旧行为。
        data["avoid"] = _clean_list(
            data["avoid"] + dislikes
        )
        # 过敏原仍走硬过滤（安全红线）
        data["exclude"] = _clean_list(data["exclude"] + allergens)
        for item in allergens:
            if item not in original_exclude:
                applied.append({
                    "kind": "allergen",
                    "value": item,
                    "display": item,
                    "source": "confirmed_preference",
                })
        # dislikes 现在走软过滤，审计字段也改为 avoid_kind
        for item in dislikes:
            if item not in allergens:
                applied.append({
                    "kind": "soft_dislike",
                    "value": item,
                    "display": item,
                    "source": "confirmed_preference",
                })
        remembered_diet = list(preferences.get("dietary_constraints") or [])
        # 素食属于安全边界，始终继承；减肥/低脂等目标只给泛推荐补充，
        # 不能污染"我想吃西红柿炒鸡蛋"这类明确菜名搜索。
        hard_diet = [
            item for item in remembered_diet
            if item.lower() in (
                "素食", "纯素", "全素", "vegetarian", "vegan",
                "清真", "回族", "穆斯林", "halal", "muslim",
            )
        ]
        hard_diet = [
            "halal" if item.lower() in ("清真", "回族", "穆斯林", "halal", "muslim") else item
            for item in hard_diet
        ]
        data["dietary_constraints"] = _clean_list(
            data["dietary_constraints"] + hard_diet
        )
        for item in hard_diet:
            if item not in original_diet:
                applied.append({
                    "kind": "dietary_constraint",
                    "value": item,
                    "display": item,
                    "source": "confirmed_preference",
                })
        # 只有当前轮没有明确选菜方向时，才用账号已确认的喜好和库存补充泛推荐。
        # 局部来宾偏好已经移出全局 flavors，因此其余菜位仍可使用用户本人偏好。
        explicit_direction = any((
            data["dishes"], data["ingredients"], data["cuisines"], data["flavors"],
            data["methods"],
        ))
        if not explicit_direction:
            inherited_likes = [
                item
                for item in (preferences.get("likes") or [])
                if not any(
                    _preference_value_matches(item, blocked)
                    for blocked in data.get("suppressed_inherited") or []
                )
            ]
            data["soft_preferences"] = _clean_list(
                data["soft_preferences"] + inherited_likes,
                limit=3,
            )
            for item in inherited_likes:
                if item in data["soft_preferences"]:
                    applied.append({
                        "kind": "soft_preference",
                        "value": item,
                        "display": item,
                        "source": "confirmed_preference",
                    })
            data["ingredients"] = _clean_list(
                data["ingredients"] + list(preferences.get("available_ingredients") or []),
                limit=6,
            )
        deduped_applied: list[dict[str, str]] = []
        seen_applied: set[tuple[str, str]] = set()
        for item in applied:
            signature = (
                str(item.get("kind") or ""),
                str(item.get("value") or "").lower(),
            )
            if signature in seen_applied:
                continue
            seen_applied.add(signature)
            deduped_applied.append(item)
        data["applied_memory_constraints"] = deduped_applied[:12]
        return SearchRequest(**data)

    def merge_recent_context(self, recent_turns: list[Any] | None) -> "SearchRequest":
        """为"可以做什么菜？"这类省略主语的检索承接同 thread 近期话题。"""
        if any((
            self.dishes, self.ingredients, self.cuisines, self.flavors, self.methods,
            self.dietary_constraints, self.exclude, self.exclude_cuisines, self.avoid,
        )):
            return self

        compact_query = _compact_search_text(self.query)
        residual = compact_query
        removable = [
            *_GENERIC_QUERY_PHRASES,
            *_CONTEXT_REFERENCE_WORDS,
            *self.cuisines, *self.flavors, *self.methods, *self.scenes, *self.meals,
            *self.dietary_constraints,
        ]
        for value in sorted(removable, key=lambda item: len(str(item)), reverse=True):
            residual = residual.replace(_compact_search_text(str(value)), "")
        if residual:
            return self

        topic = _recent_topic_from_turns(recent_turns, current_text=self.original_text)
        if not topic:
            return self

        data = self.model_dump()
        data["query"] = topic
        if self.detail_requested:
            # "这个怎么做/刚才那道的步骤"是精确详情请求；把已解析的近期
            # 主题冻结为目标，后续候选必须通过明确菜名校验。
            data["dishes"] = [topic]
            data["task_operation"] = "exact_search"
        data["context_note"] = {
            "kind": "recent_topic",
            "term": topic,
            "original_query": self.query,
        }
        return SearchRequest(**data)

    def merge_food_memory(self, context: dict | None) -> "SearchRequest":
        """记录弱行为线索，但不再把旧搜索直接拼进本轮检索词。

        搜过某种食材只能说明发生过一次行为，不能等同于用户长期偏好。旧实现
        把它写入 soft_preferences，导致"三个朋友来吃饭"被改成"聚会 鸡蛋"。
        现在仅保留审计信息，真实长期偏好仍由 merge_preferences 处理。
        """
        if self.context_note:
            # 近一轮明确话题比账号历史弱个性化更可靠。
            return self
        events = list((context or {}).get("events") or [])
        if not events:
            return self
        current = " ".join(
            [self.original_text, self.query, *self.dishes, *self.ingredients]
        ).lower()
        protein_terms = (
            ("鱼", "fish", ("鱼", "fish")), ("虾", "shrimp", ("虾", "shrimp", "prawn")),
            ("鸡肉", "chicken", ("鸡肉", "鸡胸", "鸡腿", "chicken")),
            ("牛肉", "beef", ("牛肉", "beef")), ("猪肉", "pork", ("猪肉", "五花肉", "排骨", "pork")),
            ("羊肉", "lamb", ("羊肉", "lamb")), ("鸭肉", "duck", ("鸭", "duck")),
            ("豆腐", "tofu", ("豆腐", "tofu")), ("鸡蛋", "egg", ("鸡蛋", "egg")),
        )
        # 地方菜、汤粥等已经是清楚的当前方向。历史主料一旦拼进检索词，容易把
        # "想喝汤"收窄成"只看牛肉汤"，或把"杭州菜"收窄成"杭州牛肉菜"。
        # 这些场景不使用行为记忆扩写，保证当前需求始终是检索主轴。
        specific_meals = {"汤", "粥", "汤羹", "甜品", "饮品", "soup", "porridge", "dessert", "drink"}
        if self.cuisines or any(str(value).lower() in specific_meals for value in self.meals):
            return self
        style_terms = (
            ("辣", "spicy", ("辣", "香辣", "麻辣", "spicy")),
            ("下饭", "savory", ("下饭", "savory")),
            ("清淡", "light", ("清淡", "少油", "light", "lighter")),
            ("快手", "quick", ("快手", "省事", "简单", "quick", "easy")),
            ("湘菜", "Hunan cuisine", ("湘菜", "湖南", "hunan")),
            ("川菜", "Sichuan cuisine", ("川菜", "四川", "sichuan")),
            ("晋菜", "Shanxi cuisine", ("晋菜", "山西", "shanxi")),
            ("土豆", "potato", ("土豆", "potato")),
        )
        # 用户本轮已经点了主要蛋白或具体菜名，历史只能闭嘴，避免"想吃牛肉"被改成鱼。
        if self.dishes or any(any(alias in current for alias in aliases) for _, _, aliases in protein_terms):
            return self

        chosen = None
        use_english = not bool(re.search(r"[\u4e00-\u9fff]", current))
        for kind in ("cooked", "searched"):
            for event in reversed(events):
                if event.get("kind") != kind:
                    continue
                recipes = [event.get("recipe") or {}] if kind == "cooked" else list(event.get("recipes") or [])
                if kind == "cooked":
                    evidence = " ".join(
                        str(value or "")
                        for recipe in recipes[:1]
                        for value in (recipe.get("name"), recipe.get("ingredients"), recipe.get("tags"))
                    ).lower()
                    candidates = protein_terms
                else:
                    search_request = event.get("search_request") or {}
                    evidence = " ".join((
                        str(event.get("query") or ""),
                        str(event.get("original_question") or ""),
                        str(search_request),
                    )).lower()
                    # 搜索行为可以说明用户主动找过某个方向；不从偶然出现的候选菜反推偏好。
                    candidates = (*protein_terms, *style_terms)
                for canonical_zh, canonical_en, aliases in candidates:
                    canonical = canonical_en if use_english else canonical_zh
                    if any(alias in evidence for alias in aliases) and canonical.lower() not in current:
                        matched_recipe = next(
                            (
                                recipe for recipe in recipes[:5]
                                if any(
                                    alias in " ".join(str(recipe.get(key) or "") for key in ("name", "ingredients", "tags")).lower()
                                    for alias in aliases
                                )
                            ),
                            recipes[0] if recipes else {},
                        )
                        chosen = {
                            "kind": kind,
                            "term": canonical,
                            "recipe_name": str((matched_recipe or {}).get("name") or ""),
                            "query": str(event.get("query") or ""),
                        }
                        break
                if chosen:
                    break
            if chosen:
                break
        if not chosen:
            return self

        data = self.model_dump()
        data["memory_terms"] = _clean_list(data["memory_terms"] + [chosen["term"]], limit=2)
        data["memory_note"] = chosen
        return SearchRequest(**data)

    def refine(self, followup_text: str) -> "SearchRequest":
        """把"换一批，清淡一点"中的新增条件叠加到上一轮请求。"""
        text = re.sub(r"\s+", " ", str(followup_text or "")).strip()
        data = self.model_dump()

        def add(field: str, *values: str) -> None:
            data[field] = _clean_list(list(data.get(field) or []) + [value for value in values if value])

        def remove_spicy_flavors() -> None:
            data["flavors"] = [
                value
                for value in data["flavors"]
                if not any(
                    marker in str(value).lower()
                    for marker in ("辣", "spicy", "hot")
                )
            ]

        def remove_query_phrases(*phrases: str) -> None:
            query = str(data.get("query") or "")
            for phrase in phrases:
                query = re.sub(re.escape(phrase), " ", query, flags=re.IGNORECASE)
            data["query"] = re.sub(r"\s+", " ", query).strip()

        lower_text = text.lower()

        # 把新出现的来宾子集偏好移出全局槽位。它只占自己的菜单配额。
        new_scoped = _extract_scoped_preferences(text)
        if new_scoped:
            data["scoped_preferences"] = new_scoped
            scoped_cuisines = {
                str(item).lower() for scoped in new_scoped for item in (scoped.get("cuisines") or [])
            }
            scoped_flavors = {
                str(item).lower() for scoped in new_scoped for item in (scoped.get("flavors") or [])
            }
            data["cuisines"] = [
                item for item in data["cuisines"] if str(item).lower() not in scoped_cuisines
            ]
            data["flavors"] = [
                item for item in data["flavors"] if str(item).lower() not in scoped_flavors
            ]

        remove_cuisine = any(phrase in lower_text for phrase in (
            "不一定湖南", "不用湖南", "不要湖南", "不必湖南", "不一定湘菜", "不用湘菜",
            "取消湘菜", "remove hunan", "not necessarily hunan", "no hunan preference",
        ))
        remove_spicy = any(phrase in lower_text for phrase in (
            "不一定辣", "不用偏辣", "取消偏辣", "不必偏辣", "那一道也不要辣", "都不要辣",
            "not necessarily spicy", "remove spicy preference", "no spicy preference",
        ))
        if remove_cuisine or remove_spicy:
            if remove_cuisine:
                data["cuisines"] = [
                    value for value in data["cuisines"]
                    if not any(marker in str(value).lower() for marker in ("湖南", "湘菜", "hunan"))
                ]
            if remove_spicy:
                data["flavors"] = [
                    value for value in data["flavors"]
                    if not any(marker in str(value).lower() for marker in ("辣", "spicy"))
                ]
            refined_scopes = []
            for scoped in data.get("scoped_preferences") or []:
                updated = dict(scoped)
                if remove_cuisine:
                    updated["cuisines"] = []
                if remove_spicy:
                    updated["flavors"] = []
                if updated.get("cuisines") or updated.get("flavors"):
                    refined_scopes.append(updated)
            data["scoped_preferences"] = refined_scopes

        if any(phrase in lower_text for phrase in (
            "这顿不用减脂", "这次不用减脂", "这顿不减肥", "这次不减肥",
            "这顿不用低脂", "不一定减脂", "not low fat for this meal", "skip weight loss this time",
        )):
            diet_markers = ("减肥", "减脂", "低脂", "控卡", "weight loss", "low fat", "low-fat")
            for field in ("scenes", "dietary_constraints", "soft_preferences"):
                data[field] = [
                    value for value in data[field]
                    if not any(marker in str(value).lower() for marker in diet_markers)
                ]
            data["suppressed_inherited"] = _clean_list(
                list(data.get("suppressed_inherited") or []) + ["减脂"]
            )
        if any(word in lower_text for word in ("清淡", "轻一点", "少油", "别太油", "不要太油", "lighter", "light", "less oily")):
            add("flavors", "light" if not re.search(r"[\u4e00-\u9fff]", text) else "清淡")
            add("avoid", "too oily" if not re.search(r"[\u4e00-\u9fff]", text) else "太油")
        no_spicy_requested = any(word in lower_text for word in (
            "不要辣", "别辣", "不辣", "都不要辣", "not spicy", "no spice",
        ))
        if no_spicy_requested:
            remove_spicy_flavors()
            data["avoid"] = [
                value
                for value in data["avoid"]
                if str(value).lower() not in {"太辣", "too spicy"}
            ]
            add("avoid", "spicy" if not re.search(r"[\u4e00-\u9fff]", text) else "辣")
            remove_query_phrases(
                "not spicy", "no spicy", "no spice",
                "香辣", "麻辣", "微辣", "spicy",
            )
        elif any(word in lower_text for word in (
            "微辣", "少辣", "稍微辣", "有一点辣", "一点点辣",
            "不要太辣", "别太辣", "mildly spicy", "a little spicy",
        )):
            remove_spicy_flavors()
            data["avoid"] = [
                value
                for value in data["avoid"]
                if str(value).lower() not in {
                    "辣", "spicy", "not spicy", "no spice",
                }
            ]
            add("flavors", "mild" if not re.search(r"[\u4e00-\u9fff]", text) else "微辣")
            add("avoid", "too spicy" if not re.search(r"[\u4e00-\u9fff]", text) else "太辣")
            remove_query_phrases("不辣", "不要辣", "not spicy", "no spice")
        elif any(word in lower_text for word in (
            "更辣", "辣一点", "香辣", "麻辣", "spicier", "more spicy",
        )):
            remove_spicy_flavors()
            data["avoid"] = [
                value
                for value in data["avoid"]
                if str(value).lower() not in {
                    "辣", "太辣", "spicy", "not spicy", "no spice", "too spicy",
                }
            ]
            add("flavors", "spicy" if not re.search(r"[\u4e00-\u9fff]", text) else "辣")
            remove_query_phrases("不辣", "不要辣", "not spicy", "no spice")

        # "一点都不下饭，换一批"是在评价上一批，并补充本轮筛选方向。
        # 它不等于长期偏好，但必须进入当前 SearchRequest；否则系统只会机械换菜，
        # 用户刚表达的不满完全没有被接住。
        if any(word in lower_text for word in (
            "不下饭", "更下饭", "下饭一点", "要下饭", "够味", "有味道一点",
            "more savory", "more flavourful", "more flavorful",
        )):
            add("flavors", "savory" if not re.search(r"[\u4e00-\u9fff]", text) else "下饭")

        if any(word in text for word in ("简单", "快手", "省事", "别太复杂")):
            add("scenes", "快手")
        if any(word in lower_text for word in (
            "孩子", "儿童", "小朋友", "小孩", "for kids", "for children",
        )):
            add("scenes", "儿童" if re.search(r"[\u4e00-\u9fff]", text) else "children")

        ingredient_match = re.search(
            r"(?:再加|加上|还有|也用|我还有)\s*([^，。！？,.!?]{1,30})"
            r"|(?:add|also have|include)\s+([^,.!?]{1,40})",
            text,
            flags=re.IGNORECASE,
        )
        if ingredient_match:
            raw_ingredients = next(
                (group for group in ingredient_match.groups() if group),
                "",
            )
            additions = [
                re.sub(
                    r"(?:也可以|可以吗|一起|吧|就行|please)$",
                    "",
                    item.strip(),
                    flags=re.IGNORECASE,
                ).strip()
                for item in re.split(r"和|与|、|以及|,|\band\b", raw_ingredients)
            ]
            additions = [
                item
                for item in additions
                if item
                and not any(
                    item.lower().startswith(marker)
                    for marker in (
                        "什么", "哪些", "别的", "其他",
                        "what else", "which", "something else", "other ",
                    )
                )
                and item.lower() not in {
                    "别的", "别的吗", "其他", "其他食材", "something else",
                    "other ingredients", "more",
                }
            ]
            add("ingredients", *additions)
        for method in ("蒸", "炒", "炖", "煮", "烤", "凉拌", "红烧"):
            if method in text:
                add("methods", method)
        for meal in ("早餐", "午餐", "晚餐", "夜宵", "汤", "粥"):
            if meal in text:
                add("meals", meal)
        for scene in ("减肥", "减脂", "低脂", "儿童", "老人", "便当"):
            if scene in text:
                add("scenes", scene)

        extracted = _split_excludes(text)
        add("avoid", *[item for item in extracted if item in _SOFT_NEGATIVE_TERMS])
        hard_items, cuisine_items = _split_typed_excludes([
            item for item in extracted if item not in _SOFT_NEGATIVE_TERMS
        ])
        if no_spicy_requested:
            hard_items = [
                item
                for item in hard_items
                if str(item).strip().lower() not in {
                    "辣", "辣的", "辣味", "spicy", "spice", "spicy food",
                }
            ]
        add("exclude", *hard_items)
        add("exclude_cuisines", *cuisine_items)
        if data != self.model_dump():
            data["original_text"] = f"{self.original_text}；本轮调整：{text}".strip("；")
        return SearchRequest(**data)

    def retrieval_query(self, *, lang: str = "zh") -> str:
        """按稳定优先级生成检索文本，同时保留负向约束。

        明确菜名/主料排在最前；用户任务短语、口味做法、场景餐次依次靠后。
        因此背景变化不会覆盖核心对象。
        """
        core = self.dishes + self.ingredients
        modifiers = (
            self.cuisines + self.flavors + self.methods
            + self.dietary_constraints + self.scenes + self.meals
            + self.soft_preferences[:3]
            + [
                _SOFT_POSITIVE_MODIFIERS[item.lower()]
                for item in self.avoid
                if item.lower() in _SOFT_POSITIVE_MODIFIERS
            ]
        )
        broad_expansion: tuple[str, ...] = ()
        if self.has_broad_flavor_direction():
            broad_expansion = _BROAD_FLAVOR_CANONICAL["bold" if lang == "en" else "重口味"]
        # q 已在构造阶段逐词做过原话证据校验。核心对象仍排在最前，随后补回
        # q 中未被槽位覆盖的真实条件，最后才是场景等修饰条件。
        source_values = core + re.split(r"\s+", self.query.strip()) + modifiers + list(broad_expansion)
        positive: list[str] = []
        for value in source_values:
            value = str(value or "").strip()
            already_covered = any(value == item or value in item for item in positive)
            if value and not already_covered and not value.startswith("现有食材:"):
                positive.append(value)
        if not positive:
            positive.append(self.original_text)
        # 负向条件只通过结构化硬过滤消费，绝不进入 embedding/BM25。
        query = " ".join(positive)
        # 清理 ASR/复制文本中的控制字符并限制长度；结构化字段仍保留在对象中，
        # 不需要把整段聊天原文塞给 embedding/rerank。
        query = re.sub(r"[\x00-\x1f\x7f]+", " ", query)
        query = re.sub(r"\s+", " ", query).strip()
        return query[:320].rstrip()

    def public_dict(self) -> dict:
        """写日志/快照用；当前不含敏感账号信息。"""
        return self.model_dump()


def requested_allergen_conflicts(text: str, allergens: list[str]) -> list[str]:
    """识别用户明确要求推荐/制作已声明过敏原；在意图模型之前阻断。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not value or not allergens:
        return []
    action_pattern = re.compile(
        r"(?:推荐|想吃|要吃|来点|来个|做|想做|直接给我|"
        r"recommend|suggest|i want|give me|show me|cook|make)\s*(.*)",
        flags=re.IGNORECASE,
    )
    action_matches = list(action_pattern.finditer(value))
    if not action_matches:
        return []
    requested_text = " ".join(match.group(1) for match in action_matches)
    conflicts = []
    for raw in allergens:
        term = str(raw or "").strip().lower()
        if not term:
            continue
        variants = {term}
        if re.fullmatch(r"[a-z][a-z -]{1,31}", term) and term.endswith("s"):
            variants.add(term[:-1])
        def contains_variant(variant: str) -> bool:
            if re.fullmatch(r"[a-z][a-z -]{0,31}", variant):
                return bool(re.search(
                    rf"(?<![a-z]){re.escape(variant)}(?![a-z])",
                    requested_text,
                    flags=re.IGNORECASE,
                ))
            return variant in requested_text

        matched_variants = [variant for variant in variants if variant and contains_variant(variant)]
        if not matched_variants:
            continue
        safe_patterns = tuple(
            pattern
            for variant in matched_variants
            for pattern in (
                rf"(?:不含|不要|不放|不加|避开|排除)\s*{re.escape(variant)}",
                rf"(?:without|no|exclude|avoid)\s+(?:any\s+)?{re.escape(variant)}",
                rf"{re.escape(variant)}\s*[- ]free",
            )
        )
        if any(re.search(pattern, requested_text, flags=re.IGNORECASE) for pattern in safe_patterns):
            continue
        conflicts.append(raw)
    return list(dict.fromkeys(conflicts))


def _canonical_food_evidence(value: Any) -> str:
    return _compact_search_text(str(value or ""))


def _flatten_constraint_values(value: Any, *, depth: int = 0) -> list[str]:
    if depth > 4 or value is None:
        return []
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_constraint_values(item, depth=depth + 1))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten_constraint_values(item, depth=depth + 1))
        return out
    return [str(value)]


def _result_constraint_haystack(metadata: dict) -> str:
    """汇总菜名、食材、调料、标签和详情中的食物事实，供最终硬过滤使用。"""
    fields = [
        metadata.get("name"),
        metadata.get("description"),
        metadata.get("ingredients"),
        metadata.get("seasonings"),
        metadata.get("condiments"),
        metadata.get("sauces"),
        metadata.get("tags"),
        metadata.get("facets"),
        metadata.get("recipe_detail"),
        metadata.get("recipeIngredientsVoList"),
    ]
    return " ".join(
        item.strip().lower()
        for field in fields
        for item in _flatten_constraint_values(field)
        if item and item.strip()
    )


def _constraint_variants(value: str) -> set[str]:
    normalized = str(value or "").strip().lower()
    variants = {normalized} if normalized else set()
    if _is_spicy_constraint(normalized):
        # "辣的"是自然语言表达，真实菜谱通常只写"香辣/辣椒/酸辣"。
        # 加入最小词根后，历史不吃辣才能真正执行硬过滤。
        variants.add("辣")
    for group in _INGREDIENT_EQUIVALENT_GROUPS:
        if normalized in group:
            variants.update(group)
    if re.fullmatch(r"[a-z][a-z -]{1,31}", normalized):
        if normalized.endswith("ies"):
            variants.add(normalized[:-3] + "y")
        elif normalized.endswith("es"):
            variants.add(normalized[:-2])
        elif normalized.endswith("s"):
            variants.add(normalized[:-1])
    return variants


def _contains_constraint_variant(haystack: str, variant: str) -> bool:
    if not variant:
        return False
    if re.fullmatch(r"[a-z][a-z -]{1,31}", variant):
        return bool(re.search(
            rf"(?<![a-z]){re.escape(variant)}(?![a-z])",
            haystack,
            flags=re.IGNORECASE,
        ))
    return variant in haystack


def _is_halal_request(request: SearchRequest) -> bool:
    return any(
        str(item or "").strip().lower() in {"halal", "清真", "回族", "穆斯林", "muslim"}
        for item in request.dietary_constraints
    )


def hard_constraint_notice(request: SearchRequest, lang: str = "zh") -> str:
    """返回必须随结果展示的能力边界；当前只处理清真认证边界。"""
    if not _is_halal_request(request):
        return ""
    if lang == "en":
        return (
            "I filtered out recipes whose available metadata mentions pork or alcohol. "
            "The recipe data is not halal-certified, so please verify ingredient labels and kitchen handling."
        )
    return (
        "我已按现有菜谱字段排除猪肉、猪油、猪骨和酒类；"
        "但菜谱库不具备清真认证，请仍以实际配料标签和厨房处理方式为准。"
    )


def _result_supports_explicit_food(result: dict, request: SearchRequest) -> bool:
    """结果至少要能证明它与用户明确菜名/主料有关。"""
    metadata = result.get("metadata") or {}
    evidence = [metadata.get("name"), metadata.get("description")]
    ingredients = metadata.get("ingredients") or []
    evidence.extend(ingredients if isinstance(ingredients, list) else [ingredients])
    facets = metadata.get("facets") or {}
    if isinstance(facets, dict):
        main_ingredient = facets.get("main_ingredient") or []
        evidence.extend(
            main_ingredient if isinstance(main_ingredient, list) else [main_ingredient]
        )
    supported = [_canonical_food_evidence(value) for value in evidence]
    dishes = [
        _canonical_food_evidence(value) for value in request.dishes
        if _canonical_food_evidence(value)
    ]
    if dishes and not any(
        wanted == actual or wanted in actual or actual in wanted
        for wanted in dishes for actual in supported if actual
    ):
        return False
    required = [] if request.is_menu_plan else [
        _canonical_food_evidence(value) for value in request.required_ingredients
        if _canonical_food_evidence(value)
    ]
    if required and not all(
        any(wanted == actual or wanted in actual for actual in supported if actual)
        for wanted in required
    ):
        return False
    if dishes or required:
        return True
    ingredients = [
        _canonical_food_evidence(value) for value in request.ingredients
        if _canonical_food_evidence(value)
    ]
    return not ingredients or any(
        wanted == actual or wanted in actual
        for wanted in ingredients for actual in supported if actual
    )


def filter_hard_constraint_violations(search_result: dict, request: SearchRequest) -> dict:
    """召回后校验明确主料/菜名和忌口，拒绝展示无事实依据的结果。"""
    positively_kept = []
    positive_removed = []
    for result in search_result.get("results") or []:
        if _result_supports_explicit_food(result, request):
            positively_kept.append(result)
        else:
            metadata = result.get("metadata") or {}
            positive_removed.append(metadata.get("recipe_id") or result.get("id"))

    kept = []
    removed = []
    for result in positively_kept:
        metadata = result.get("metadata") or {}
        haystack = _result_constraint_haystack(metadata)
        absolute_avoid = [
            item
            for item in request.avoid
            if str(item).strip().lower() in {"辣", "spicy"}
        ]
        matched = []
        for item in [*request.exclude, *absolute_avoid]:
            if any(
                _contains_constraint_variant(haystack, variant)
                for variant in _constraint_variants(item)
            ):
                matched.append(item)
        matched_diet = []
        if _is_halal_request(request):
            forbidden = sorted(
                variant for variant in _HALAL_FORBIDDEN_VARIANTS
                if _contains_constraint_variant(haystack, variant)
            )
            if forbidden:
                matched_diet.append({"constraint": "halal", "matched": forbidden})
        facets = metadata.get("facets") or {}
        cuisines = {
            str(item or "").strip().lower()
            for item in ((facets.get("cuisine") or []) if isinstance(facets, dict) else [])
        }
        tags = " ".join(str(item or "") for item in (metadata.get("tags") or [])).lower()
        matched_cuisines = [
            item for item in request.exclude_cuisines
            if str(item).lower() in cuisines
            or any(
                alias.lower() in tags
                for alias, canonical in _CUISINE_CONSTRAINT_ALIASES.items()
                if canonical == str(item).lower()
            )
        ]
        if matched or matched_cuisines or matched_diet:
            removed.append({
                "id": metadata.get("recipe_id") or result.get("id"),
                "matched": matched,
                "matched_cuisines": matched_cuisines,
                "matched_diet": matched_diet,
            })
        else:
            kept.append(result)
    out = dict(search_result)
    out["results"] = kept
    if positive_removed:
        out["_positive_filtered"] = positive_removed
    if removed:
        out["_hard_filtered"] = removed
    return out
