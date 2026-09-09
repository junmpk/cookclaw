"""基于真实检索结果生成自然推荐叙述。

意图、检索、过滤和候选选择保持确定性；Qwen 只负责把已经选出的真实菜谱
组织成自然语言。模型不可用或输出不合规时，返回空叙述，渲染层仅展示真实
菜谱卡片，不再拼接固定话术。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen

from app.core.config import settings
from app.observability.trace import (
    mark_fallback,
    observe_model_call,
    record_reply_validation,
    trace_stage,
)

logger = logging.getLogger(__name__)


# ─── 事实校验严格度分级 ─────────────────────────────────────────────────
#
# strict:   所有检查都是硬约束（旧行为，最保守）
# medium:   硬约束（内部词/健康宣称/编造菜名/编造数字）必拒；
#           软约束（口味描述/历史承接/额外搭配/否定偏好）只告警，不丢弃句子
# relaxed:  仅保留最核心的硬约束（内部词/健康宣称/编造菜名）
#
# 环境变量 GROUNDING_STRICTNESS=strict|medium|relaxed，默认 medium。

_GROUNDING_STRICTNESS = (
    os.getenv("GROUNDING_STRICTNESS", "medium").strip().lower()
    or "medium"
)
if _GROUNDING_STRICTNESS not in {"strict", "medium", "relaxed"}:
    _GROUNDING_STRICTNESS = "medium"

# ─── 用户画像注入开关 ─────────────────────────────────────────────────
#
# 默认开启完整画像注入（likes/dislikes/dietary_constraints/allergens）。
# INJECT_FULL_USER_PROFILE=0 可回退到只注入 preferred_name。

_INJECT_FULL_USER_PROFILE = (
    os.getenv("INJECT_FULL_USER_PROFILE", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)


def _build_user_profile_payload(agent_context: dict | None, lang: str) -> dict:
    """从 agent_context 提取用户画像，作为表达模型的个性化输入。

    只注入"已确认"的偏好（来自持久化存储，不是推断）。每类最多 3 条，
    避免 prompt 膨胀。INJECT_FULL_USER_PROFILE=0 时回退到只注入 preferred_name。
    """
    profile = (agent_context or {}).get("user_profile") or {}
    payload = {
        "preferred_name": str(profile.get("preferred_name") or "")[:24],
    }
    if not _INJECT_FULL_USER_PROFILE:
        return payload
    raw_prefs = profile.get("preferences") or {}
    if not isinstance(raw_prefs, dict):
        return payload
    for key, limit in (
        ("likes", 3),
        ("dislikes", 3),
        ("dietary_constraints", 3),
        ("allergens", 2),
    ):
        values = raw_prefs.get(key) or []
        if not isinstance(values, list):
            continue
        # 同语种过滤，避免英文模式出现中文偏好
        filtered = _same_language(
            [str(v).strip() for v in values if str(v).strip()],
            lang,
            limit=limit,
        )
        if filtered:
            payload[key] = filtered
    return payload

_STYLE_PROMPT = (
    Path(__file__).resolve().parent.parent / "core" / "response_style_prompt.md"
).read_text(encoding="utf-8")


@dataclass(frozen=True)
class RecommendationNarrative:
    opening: str
    strategy: str
    recipe_reasons: dict[str, str]
    closing: str

    def to_dict(self) -> dict:
        return asdict(self)


_DEMO_NATURAL_RESPONSE_ENABLED = os.getenv(
    "DEMO_NATURAL_RESPONSE_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

_recommendation_llm = ChatQwen(
    model=settings.LLM_MODEL,
    temperature=0.55 if _DEMO_NATURAL_RESPONSE_ENABLED else 0.2,
    max_tokens=1400 if _DEMO_NATURAL_RESPONSE_ENABLED else 900,
    timeout=18,
    max_retries=1,
    enable_thinking=False,
    api_key=settings.DASHSCOPE_API_KEY,
    base_url=settings.DASHSCOPE_BASE_URL,
)

_INTERNAL_WORDS = (
    "检索", "召回", "候选池", "数据库", "向量库", "菜谱库", "食谱库", "库里",
    "内部记录", "历史记录", "命中", "全匹配", "部分匹配",
    "标签为", "标签显示", "字段显示",
    "retrieval", "candidate pool", "database", "vector store", "internal memory",
    "all-match", "all match", "partial match", "match type",
)

_HISTORY_GROUNDING_TERMS_ZH = ("之前", "上次", "以前", "曾经", "记得", "记录")
_HISTORY_GROUNDING_TERMS_EN = ("previously", "last time", "earlier", "remember", "record")
_PREFERENCE_NEGATION_TERMS_ZH = (
    "避开", "不吃", "不能吃", "不太能吃", "吃不了", "不想吃", "不要吃",
    "不喜欢", "不爱吃", "怕辣", "不耐辣", "忌口", "过敏",
)
_PREFERENCE_NEGATION_TERMS_EN = (
    "avoid", "don't eat", "do not eat", "can't eat", "cannot eat",
    "don't like", "do not like", "allergic",
)
_ADDITION_GROUNDING_TERMS_ZH = (
    "再加", "加点", "加些", "放点", "丢进去", "配米饭", "搭配", "配上", "配个",
)
_ADDITION_GROUNDING_TERMS_EN = (
    "add some", "add a", "throw in", "pair with", "pair it with", "serve with",
)
_CONTROL_CONTEXT_TERMS = {
    "确认", "取消", "停止", "开始", "设备", "在线", "离线",
    "confirm", "cancel", "stop", "start", "device", "online", "offline",
}
_RELEVANT_CONTEXT_MARKERS = (
    "累", "疲惫", "饿", "馋", "烦", "糟", "开心", "难过", "委屈", "焦虑", "压力",
    "忙", "开会", "辛苦", "庆祝", "犒劳", "下班", "加班", "赶时间", "没时间",
    "冷", "热", "下雨", "胃", "不舒服", "冰箱", "食材", "主料", "吃", "饭", "菜", "口味",
    "鸡", "鸭", "鱼", "虾", "肉", "蛋", "豆", "番茄", "土豆", "面", "米", "汤",
    "清淡", "辣", "甜", "咸", "聚餐", "朋友", "客人", "家人", "孩子", "老人", "忌口", "过敏",
    "tired", "hungry", "craving", "upset", "rough day", "happy", "sad", "anxious", "stressed",
    "busy", "meeting", "celebrate", "treat myself", "after work", "overtime",
    "raining", "cold", "hot", "stomach", "fridge", "ingredient", "eat", "meal", "dinner",
    "lunch", "breakfast", "flavor", "spicy", "light", "friends", "guests", "family", "allergic",
)
_BRIDGE_HISTORY_WORDS_ZH = ("我记得", "记得你", "之前你", "上次你", "历史", "记录显示")
_BRIDGE_HISTORY_WORDS_EN = (
    "i remember", "you previously", "last time you", "your history", "records show",
)

# 这些词一旦出现在表达里，就必须能在用户当前输入或真实菜谱字段中找到依据。
# 它们覆盖最容易被模型"顺手补全"的时长、步骤、口味、功效和难度描述。
_GROUNDING_TERMS_ZH = (
    "分钟", "小时", "秒", "克", "毫升", "度", "切", "打蛋", "打个蛋", "下锅", "出锅",
    "翻炒", "焯水", "腌", "煎", "炸", "蒸", "炖", "烤", "煮", "调味",
    "酸甜", "咸鲜", "香辣", "麻辣", "清淡", "脆", "软烂", "开胃", "下饭",
    "味道很鲜", "鲜味", "香气", "浓郁", "爽口", "嫩滑", "酥脆", "多汁",
    "口味更轻", "味道更轻",
    "暖胃", "养胃", "好消化", "易消化", "减脂", "减肥", "低脂", "高蛋白",
    "疗效", "治疗", "解酒", "排毒", "增强免疫", "零难度", "不用守锅",
    "荤素", "分量", "份量", "细腻", "厚重", "不腻", "解腻",
    "简单", "容易", "省事", "复杂", "撑场面", "宴客", "适合聚餐",
    "配齐", "凑齐", "完整菜单",
    "营养", "营养价值", "维生素", "蛋白质", "番茄红素", "老少皆宜",
    "经典", "常见", "国民", "清爽", "解暑", "适合孩子", "适合老人",
    "口感", "好吃", "美味",
    "准备好", "已经配好",
)
_GROUNDING_TERMS_EN = (
    "minute", "minutes", "hour", "hours", "seconds", "grams", "degree", "degrees",
    "chop", "slice", "beat", "boil", "fry", "steam", "bake", "roast", "simmer",
    "hands-on", "cleanup", "pantry staple", "big flavor", "savory", "sweet", "spicy",
    "crispy", "tender", "comforting", "aromatic", "rich", "refreshing", "juicy",
    "creamy", "mild", "digestible", "easy to digest", "low-fat",
    "high-protein", "detox", "heal", "cure", "treat", "boost immunity",
    "simple", "easy", "effortless", "complicated", "complex", "party-friendly",
    "good for a gathering",
    "complete menu", "filled every slot",
    "nutrition", "nutritional", "vitamin", "protein", "lycopene",
    "family-friendly", "kid-friendly", "senior-friendly", "classic", "popular",
    "refreshing", "delicious", "tasty", "texture",
    "ready", "set for the table",
)
_CLOSING_ADDITION_ZH = ("家里如果有", "再加", "加点", "丢进去", "搭配", "配上", "配个")
_CLOSING_ADDITION_EN = ("pair it with", "serve it with", "add some", "add a", "if you have")
_HEALTH_CLAIM_RE_ZH = re.compile(
    r"养胃|暖胃|解酒|排毒|治疗|疗效|增强免疫|提高免疫|好消化|易消化|"
    r"高纤维|补水|补充水分|代餐|顶饱|饱腹|清理负担|"
    r"营养(?:丰富|价值)?|维生素|蛋白质|番茄红素|"
    r"(?:胃里|肚子|身体).{0,8}(?:舒服|轻松)|"
    r"(?:身体|肠胃|胃).{0,8}(?:减减?负|负担小|没负担|无负担)|"
    r"(?:有助于?|帮助|能够?|可以)\s*(?:减肥|减脂|降糖|降压)"
)
_HEALTH_CLAIM_RE_EN = re.compile(
    r"\b(?:cure|heal|detox|treats?|boosts? immunity|easy to digest|digestible|"
    r"meal replacement|keeps? you full|filling|hydrating|replenishes? fluids?|"
    r"nutritious|nutritional value|vitamins?|protein|lycopene|"
    r"helps? (?:with )?(?:weight loss|lower blood sugar|lower blood pressure))\b",
    flags=re.IGNORECASE,
)

# 推荐阶段的地域安全门禁不能只依赖 prompt。用户可以说明所在地或来宾籍贯，
# 但这些背景事实不能成为口味、菜系、酒量或“硬菜撑场面”的因果依据。
_NARRATIVE_REGIONAL_INFERENCE_ZH_RE = re.compile(
    r"(?:都是?|作为|身为|来自|老家(?:在|是)|籍贯(?:在|是)|"
    r"[\u4e00-\u9fff]{2,8}(?:人|朋友|老乡|老表|汉子|爷们|妹子))"
    r"[^。！？!?]{0,36}"
    r"(?:必须|肯定|一定|当然|天生|就得|得有|就要|都爱|爱吃|喜欢吃|"
    r"口味|辣|重口|清淡|甜|咸|够味|够劲|硬菜|撑场面|能喝|酒量|豪爽|"
    r"赣菜|川菜|湘菜|粤菜|东北菜|本地菜|家乡菜|地方菜|菜系)",
)
_NARRATIVE_REGIONAL_INFERENCE_EN_RE = re.compile(
    r"(?:\b(?:people|friends|guests|folks|men|women)\s+from\s+[a-z .'-]{2,30}|"
    r"\b[a-z .'-]{2,30}\s+(?:people|friends|guests|folks|men|women))"
    r"[^.!?]{0,40}"
    r"\b(?:must|always|naturally|surely|love|like|can\s+handle|"
    r"spicy|bold|strong|mild|sweet|salty|drink|liquor|alcohol|hearty)\b",
    flags=re.IGNORECASE,
)
_NARRATIVE_UNSUPPORTED_CLAIMS_ZH = (
    "绝对重头戏", "胶质满满", "满满胶质", "最适合配白酒", "最配白酒",
)
_NARRATIVE_UNSUPPORTED_CLAIMS_EN = (
    "absolute centerpiece", "full of collagen", "packed with collagen", "best with liquor",
    "perfect with liquor", "best with alcohol", "perfect with alcohol",
)
_NARRATIVE_ALCOHOL_POSITIONING_ZH_RE = re.compile(
    r"(?:非常适合|特别适合|最适合|完美(?:的)?|绝佳(?:的)?)[^。！？!?]{0,24}"
    r"(?:白酒|喝酒|饮酒|下酒|佐酒)|"
    r"(?:搭配|配上?|下酒|佐酒)[^。！？!?]{0,10}(?:白酒|酒)|"
    r"(?:喜欢|爱)[^。！？!?]{0,16}(?:大口吃肉)?(?:喝酒|饮酒)",
)
_NARRATIVE_ALCOHOL_POSITIONING_EN_RE = re.compile(
    r"\b(?:perfect|ideal|best|very suitable|great)\b[^.!?]{0,28}"
    r"\b(?:with|for)?\s*(?:liquor|alcohol|drinks?)\b|"
    r"\b(?:pair(?:s|ed|ing)?|serve(?:d)?|goes?)\b[^.!?]{0,16}"
    r"\b(?:liquor|alcohol|drinks?)\b|"
    r"\b(?:love|like)[^.!?]{0,18}\b(?:drinking|liquor|alcohol)\b",
    flags=re.IGNORECASE,
)


def _has_unsafe_narrative_inference(text: str, lang: str) -> bool:
    """Reject regional stereotypes and unsupported pairing/composition claims."""
    clean = _clean_text(text)
    if not clean:
        return False
    lower = clean.lower()
    if lang == "en":
        return bool(
            _NARRATIVE_REGIONAL_INFERENCE_EN_RE.search(clean)
            or any(claim in lower for claim in _NARRATIVE_UNSUPPORTED_CLAIMS_EN)
        )
    return bool(
        "老表" in clean
        or _NARRATIVE_REGIONAL_INFERENCE_ZH_RE.search(clean)
        or any(claim in clean for claim in _NARRATIVE_UNSUPPORTED_CLAIMS_ZH)
    )


def _has_unsupported_recipe_positioning(
    text: str,
    lang: str,
    ingredient_facts: list[dict] | None,
) -> bool:
    """Keep a global drinks scene separate from a per-recipe pairing claim."""
    clean = _clean_text(text)
    fact_source = json.dumps(ingredient_facts or [], ensure_ascii=False).lower()
    if lang == "en":
        # A neutral acknowledgement of the request is allowed; recipe suitability is not.
        neutral = re.sub(
            r"\b(?:the )?(?:liquor|alcohol|drinks?) (?:scene|request) is noted\b",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        if "hearty dish" in neutral.lower() and "hearty dish" not in fact_source:
            return True
        return bool(
            _NARRATIVE_ALCOHOL_POSITIONING_EN_RE.search(neutral)
            and not any(
                term in fact_source
                for term in ("pair with liquor", "pair with alcohol", "with drinks", "drink pairing")
            )
        )

    # The fallback may say only that the user-requested scene was remembered.
    neutral = re.sub(r"(?:配)?白酒场景(?:也)?(?:记下了|已记下|考虑进去了)", "", clean)
    if "硬菜" in neutral and "硬菜" not in fact_source:
        return True
    if re.search(r"(?:喜欢|爱)[^。！？!?]{0,16}(?:喝酒|饮酒)", neutral):
        return True
    if not _NARRATIVE_ALCOHOL_POSITIONING_ZH_RE.search(neutral):
        return False
    return not any(
        term in fact_source
        for term in ("搭配白酒", "配白酒", "下酒", "佐酒")
    )


def _recipe_id(result: dict) -> str:
    metadata = result.get("metadata") or {}
    return str(metadata.get("recipe_id") or result.get("id") or "").strip()


def _as_list(value, *, limit: int) -> list[str]:
    if isinstance(value, dict):
        value = list(value.values())
    elif isinstance(value, str):
        value = re.split(r"[、,，;/；]", value)
    out: list[str] = []
    for item in value or []:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _same_language(values: list[str], lang: str, *, limit: int) -> list[str]:
    out = []
    for value in values:
        has_cjk = bool(re.search(r"[一-鿿]", value))
        if (lang == "en" and not has_cjk) or (lang != "en" and has_cjk):
            out.append(value)
        if len(out) >= limit:
            break
    return out


_UNSAFE_RECOMMENDATION_TAGS = (
    "高蛋白", "低脂", "补铁", "暖身", "营养", "健康", "减脂", "减肥",
    "低卡", "低碳水", "高纤维", "补水",
    "high-protein", "low-fat", "iron-rich", "warming", "healthy",
    "nutritious", "weight-loss", "low-calorie", "low-carb", "high-fiber",
)


def _safe_recommendation_tags(values: list[str]) -> list[str]:
    """展示型推荐不使用营养、疗效或身体效果标签。"""
    return [
        value for value in values
        if not any(marker in value.lower() for marker in _UNSAFE_RECOMMENDATION_TAGS)
    ]


def _duration_label(seconds, lang: str) -> str:
    try:
        value = int(float(seconds))
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    if value < 3600:
        minutes = max(1, round(value / 60))
        return f"about {minutes} minutes" if lang == "en" else f"约{minutes}分钟"
    hours, remainder = divmod(value, 3600)
    minutes = round(remainder / 60)
    if lang == "en":
        return f"about {hours} hr {minutes} min" if minutes else f"about {hours} hr"
    return f"约{hours}小时{minutes}分钟" if minutes else f"约{hours}小时"


def _recipe_facts(search_result: dict, lang: str) -> list[dict]:
    facts = []
    for result in (search_result.get("results") or [])[:10]:
        metadata = result.get("metadata") or {}
        recipe_id = _recipe_id(result)
        name = str(metadata.get("name") or "").strip()
        if not recipe_id or not name:
            continue
        tags = _safe_recommendation_tags(
            _same_language(_as_list(metadata.get("tags"), limit=20), lang, limit=8)
        )
        ingredients = _same_language(
            _as_list(metadata.get("ingredients"), limit=16), lang, limit=12,
        )
        description = re.sub(r"\s+", " ", str(metadata.get("description") or "")).strip()
        if lang == "en" and re.search(r"[一-鿿]", description):
            description = ""
        unsafe_description_re = (
            r"营养|维生素|蛋白质|免疫|番茄红素|老少皆宜|儿童友好|老人友好|"
            r"清凉解暑|补水|低卡|低脂|减脂|减肥|健康|"
            r"\b(?:nutrition|vitamin|protein|immunity|healthy|low-fat|"
            r"low-calorie|weight loss|good for children|good for seniors)\b"
        )
        if re.search(unsafe_description_re, description, flags=re.IGNORECASE):
            description = ""
        fact = {
            "id": recipe_id,
            "name": name,
            "tags": tags,
            "ingredients": ingredients,
            "description": description[:240],
        }
        difficulty = str(metadata.get("difficulty") or "").strip()
        if difficulty:
            fact["difficulty"] = difficulty[:40]
        menu_role = str(result.get("menu_role") or "").strip()
        if menu_role:
            role_labels = {
                "dish": "菜" if lang != "en" else "dish",
                "soup": "汤" if lang != "en" else "soup",
                "required_dish": "必备主料菜" if lang != "en" else "required-ingredient dish",
                "scoped_dish": "照顾来宾偏好的菜" if lang != "en" else "guest-preference dish",
            }
            fact["menu_role"] = menu_role
            fact["menu_role_label"] = role_labels.get(menu_role, menu_role)
        menu_requirement = str(result.get("menu_requirement") or "").strip()
        if menu_requirement:
            fact["menu_requirement"] = menu_requirement[:80]
        ingredient_match = result.get("_ingredient_match")
        if isinstance(ingredient_match, dict):
            fact["ingredient_match"] = {
                key: value
                for key, value in ingredient_match.items()
                if key in {"requested", "matched", "missing", "match_type"}
            }
        detail = metadata.get("recipe_detail")
        if isinstance(detail, dict):
            steps = detail.get("steps") or []
            duration = _duration_label(detail.get("cooking_time_seconds"), lang)
            servings = detail.get("servings")
            if isinstance(steps, list) and steps:
                fact["step_count"] = len(steps)
            if duration:
                fact["estimated_time"] = duration
            if isinstance(servings, (int, float)) and not isinstance(servings, bool) and servings > 0:
                servings_value = f"{servings:g}" if isinstance(servings, float) else str(servings)
                fact["servings"] = (
                    f"{servings_value} servings"
                    if lang == "en" else f"{servings_value}人份"
                )
            if detail.get("ai_generated"):
                fact["detail_basis"] = (
                    "AI estimate based on the verified recipe name and ingredients"
                    if lang == "en" else
                    "依据真实菜名和食材生成的AI估算"
                )
        facts.append(fact)
    return facts


def _menu_plan_facts(search_result: dict) -> dict:
    menu = search_result.get("_menu_plan") or {}
    if not isinstance(menu, dict) or not menu:
        return {}
    return {
        "complete": bool(menu.get("complete")),
        "requested": dict(menu.get("requested") or {}),
        "fulfilled": dict(menu.get("fulfilled") or {}),
        "missing": dict(menu.get("missing") or {}),
    }


def _current_request(search_request) -> dict:
    if search_request is None:
        return {}
    data = search_request.model_dump() if hasattr(search_request, "model_dump") else dict(search_request)
    # 历史行为是排序参考，不交给表达模型，避免主动暴露"之前搜过什么"。
    for key in ("memory_terms", "memory_note", "soft_preferences", "context_note", "version"):
        data.pop(key, None)
    return data


def _compact_input_context(value: dict | None) -> dict:
    """只把受控的视觉事实交给表达层，不透传路径、URL 或通道原始 payload。"""
    raw = value if isinstance(value, dict) else {}
    try:
        image_count = max(1, min(int(raw.get("image_count") or 1), 20))
    except (TypeError, ValueError):
        image_count = 1
    ingredients = []
    for item in raw.get("ingredients") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            normalized = {"name": name[:80]}
            state = str(item.get("state") or "").strip()
            if state:
                normalized["state"] = state[:24]
            confidence = item.get("confidence")
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                normalized["confidence"] = round(max(0.0, min(float(confidence), 1.0)), 3)
            ingredients.append(normalized)
        else:
            name = str(item or "").strip()
            if name:
                ingredients.append({"name": name[:80]})
        if len(ingredients) >= 12:
            break
    context = {
        "source": "image_recognition",
        "scene_type": str(raw.get("scene_type") or "")[:24],
        "image_count": image_count,
        "dish_name": str(raw.get("dish_name") or "").strip()[:120],
        "ingredients": ingredients,
        "user_caption": str(raw.get("user_caption") or "").strip()[:300],
        "recognition_confidence": raw.get("recognition_confidence"),
    }
    return {
        key: item
        for key, item in context.items()
        if item not in (None, "", [], {})
    }


def _turn_value(turn, field: str, default=None):
    if isinstance(turn, dict):
        return turn.get(field, default)
    return getattr(turn, field, default)


def _recent_user_context(
    recent_turns,
    current_question: str,
    *,
    limit: int = 7,
    max_age_seconds: int = 43200,
    max_total_chars: int = 2000,
    include_assistant: bool = True,
    per_text_chars: int = 400,
) -> list[str]:
    """取同一会话近期的用户原话与助手回复，作为表达模型的连贯性上下文。

    相比旧实现（3 轮 × 180 字 × 关键词过滤）的放宽：
      - 轮数 3 → 7，时效 6h → 12h
      - 同时包含助手回复（让 LLM 知道自己说过什么，避免重复）
      - 移除关键词过滤（旧 _RELEVANT_CONTEXT_MARKERS 把"那就做这个吧"这类
        无关键词的承接话全部丢弃，是上下文断裂的元凶）
      - 单条字符上限 180 → 400
      - 总字符预算 2000（≈1000 token），超出时从最旧的开始丢
    仍保留 _CONTROL_CONTEXT_TERMS 过滤（"确认/取消/停止"等系统控制词不进表达）。
    """
    current = re.sub(r"\s+", " ", str(current_question or "")).strip().lower()
    seen: set[str] = set()
    selected: list[str] = []
    total_chars = 0
    now = time.time()
    allowed_roles = {"user"}
    if include_assistant:
        allowed_roles.add("assistant")
    for turn in reversed(list(recent_turns or [])[-20:]):
        role = str(_turn_value(turn, "role", "")).lower()
        if role not in allowed_roles:
            continue
        text = re.sub(r"\s+", " ", str(_turn_value(turn, "content", ""))).strip()
        if not text:
            continue
        key = text.lower()
        if key == current:
            continue
        if key in seen or key in _CONTROL_CONTEXT_TERMS:
            continue
        created_at = _turn_value(turn, "created_at", 0)
        try:
            if created_at and now - float(created_at) > max_age_seconds:
                continue
        except (TypeError, ValueError):
            pass
        trimmed = text[:per_text_chars]
        # 总字符预算：超出时停止收集（保持从新到旧，最旧的自然被丢）
        if total_chars + len(trimmed) > max_total_chars:
            break
        seen.add(key)
        prefix = "" if role == "user" else "[助手] "
        selected.append(f"{prefix}{trimmed}")
        total_chars += len(trimmed) + len(prefix)
        if len(selected) >= limit:
            break
    return list(reversed(selected))


def _safe_profile_preferences(profile_context: dict | None, lang: str) -> dict[str, list[str]]:
    """只给暖场模型最小化、已确认的口味偏好，不传摘要、行为历史或过敏原。"""
    preferences = (profile_context or {}).get("preferences") or {}
    allowed = ("likes", "dislikes", "dietary_constraints")
    result: dict[str, list[str]] = {}
    for key in allowed:
        values = _same_language(
            _as_list(preferences.get(key), limit=4),
            lang,
            limit=2,
        )
        if values:
            result[key] = values
    return result


def _clarification_acknowledgements(context: str, dimension: str, lang: str) -> list[str]:
    """Return a small, vetted set of scene acknowledgements.

    A pre-search model has no recipe facts.  Letting it write arbitrary prose here would
    make it impossible to prove that a dish, ingredient, health claim, or remembered
    preference was not invented.  The model may choose the most natural acknowledgement,
    but every choice is data-free and safe to render before retrieval.
    """
    lower = str(context or "").lower()
    tired = any(word in lower for word in (
        "累", "疲惫", "加班", "不想动", "不想折腾",
        "tired", "exhausted", "overtime", "low energy",
    ))
    if lang == "en":
        if dimension == "remembered_preference_conflict":
            return ["One detail conflicts with a preference you told me earlier."]
        if dimension == "remembered_dietary_constraint":
            return ["One quick check before I use that earlier goal."]
        if tired:
            return [
                "That already sounds like a long day.",
                "You have done enough thinking for one evening.",
                "This sounds like a night to keep decisions light.",
            ]
        if dimension in {"recommendation_basics", "party_size", "flavor_preferences"}:
            return [
                "Let's size this meal properly before choosing anything.",
                "A little context will make this dinner much easier to choose.",
                "Let's pin down the shape of this meal first.",
            ]
        if dimension in {
            "party_preferences",
            "party_constraints",
            "party_constraints_detail",
        }:
            return [
                "With friends coming over, one clear direction will make the table easier to plan.",
                "For a table of guests, narrowing the mood first makes the rest much easier.",
                "Let's give this get-together one clear direction before choosing anything.",
            ]
        return [
            "Let's make the choice a little easier.",
            "No need to solve the whole menu at once.",
            "We can narrow this down without turning it into homework.",
        ]
    if dimension == "remembered_preference_conflict":
        return ["这里有个条件和你之前明确说过的偏好冲突了。"]
    if dimension == "remembered_dietary_constraint":
        return ["先跟你确认一下这个阶段性目标。"]
    if tired:
        return [
            "听着就是挺费神的一天。",
            "今天已经够累了，这顿就别再费脑子。",
            "这会儿确实适合把选择变简单一点。",
        ]
    if dimension in {"recommendation_basics", "party_size", "flavor_preferences"}:
        return [
            "先把这顿饭的规模和口味定一下，后面就好搭了。",
            "这顿不用急着猜，先把最关键的两点说清楚。",
            "咱们先把这顿饭的方向收一收。",
        ]
    if dimension in {
        "party_preferences",
        "party_constraints",
        "party_constraints_detail",
    }:
        return [
            "朋友来吃饭，先把这一桌的方向定住会轻松很多。",
            "人多吃饭，先定个口味方向就好选了。",
            "这顿要招待朋友，咱们先把方向收一收。",
        ]
    return [
        "可以，咱们先把范围缩小一点。",
        "没想好很正常，不用一下子把整顿都想完。",
        "行，先定一个方向，后面就好选了。",
    ]


def _clarification_question(dimension: str, lang: str, search_request=None) -> str:
    """The only question rendered before retrieval; it contains no recipe claim."""
    request_data = (
        search_request.model_dump()
        if hasattr(search_request, "model_dump")
        else dict(search_request or {})
    )
    note = request_data.get("context_note") or {}
    constraints_confirmed = bool(request_data.get("constraints_confirmed"))
    remembered_term = str(note.get("term") or "").strip()
    remembered_bucket = str(note.get("bucket") or "dislikes")
    if lang == "en":
        if dimension == "remembered_preference_conflict":
            preference = remembered_term or "that option"
            if remembered_bucket == "likes":
                return (
                    f"You previously said you like {preference}, but this request avoids it. "
                    "Should I treat that as an exception for this meal only?"
                )
            return (
                f"You previously said you don't want {preference}, but this request points that way. "
                "Should I make an exception for this meal only?"
            )
        if dimension == "remembered_dietary_constraint":
            goal = remembered_term or "that dietary goal"
            return f"You previously mentioned {goal}. Should I still use it for this meal?"
        if dimension == "recommendation_basics":
            return "How many people are eating, and are you leaning light, hearty, or spicy? Mention any dietary restrictions too."
        if dimension == "party_size":
            return "How many people are eating? I'll size the dishes and soup around that."
        if dimension == "flavor_preferences":
            if constraints_confirmed:
                return "Would you like the table light, homestyle and savory, or on the spicy side?"
            return "Are you leaning light, hearty, or spicy—and are there any dietary restrictions?"
        if dimension == "party_preferences":
            if constraints_confirmed:
                return "Would you like the table homestyle and savory, light, or on the spicy side?"
            return "Are you leaning homestyle and hearty, light, or spicy—and are there any dietary restrictions?"
        if dimension == "party_constraints":
            return "Does anyone have allergies, dietary restrictions, or religious food requirements?"
        if dimension == "party_constraints_detail":
            return "What specific allergies, dietary restrictions, or religious food requirements should I account for?"
        if dimension == "available_ingredients":
            return "What main ingredients do you already have?"
        return "What main ingredient do you have, or what flavor are you in the mood for?"
    if dimension == "remembered_preference_conflict":
        preference = remembered_term or "这个方向"
        if remembered_bucket == "likes":
            return (
                f"你之前明确说过喜欢{preference}，但这次又想避开它。"
                "这顿要临时作为例外吗？"
            )
        return (
            f"你之前明确说过不喜欢{preference}，但这次的要求又指向它。"
            "这顿要临时作为例外吗？"
        )
    if dimension == "remembered_dietary_constraint":
        goal = remembered_term or "阶段性饮食目标"
        return f"你之前提过最近在{goal}，这顿还要按这个目标来选吗？"
    if dimension == "recommendation_basics":
        return "几个人吃？口味想清淡、家常下饭还是偏辣？有忌口也一起告诉我。"
    if dimension == "party_size":
        return "这顿几个人吃？我好按人数搭菜和汤。"
    if dimension == "flavor_preferences":
        if constraints_confirmed:
            return "口味想整体偏辣，还是以大众咸鲜、家常下饭为主？"
        return "口味想清淡、家常下饭还是偏辣？有忌口也一起告诉我。"
    if dimension == "party_preferences":
        if constraints_confirmed:
            return "你们更想吃家常下饭、清淡，还是整体偏辣？"
        return "你们更想吃家常下饭、清淡，还是偏辣？有忌口也一起告诉我。"
    if dimension == "party_constraints":
        return "你们中有人过敏、忌口，或有素食、清真等饮食要求吗？"
    if dimension == "party_constraints_detail":
        return "具体有哪些过敏、忌口，或素食、清真等饮食要求？"
    if dimension == "available_ingredients":
        return "家里现在有什么现成主料？"
    return "家里有什么主料，或者你现在更想吃清淡、下饭还是偏辣？"


def _join_clarification(acknowledgement: str, question: str, lang: str) -> str:
    separator = " " if lang == "en" else ""
    return f"{acknowledgement.strip()}{separator}{question.strip()}".strip()


def _valid_clarification_acknowledgement(text: str, allowed: list[str]) -> bool:
    """Only an exact vetted acknowledgement may cross the pre-search data boundary."""
    clean = _clean_text(text, limit=220)
    return bool(clean and clean in allowed)


def _clarification_fallback(
    context: str,
    dimension: str,
    lang: str,
    search_request=None,
) -> str:
    request_data = (
        search_request.model_dump()
        if hasattr(search_request, "model_dump")
        else dict(search_request or {})
    )
    if dimension == "flavor_preferences" and request_data.get("constraints_confirmed"):
        people = request_data.get("party_size")
        no_constraints = not any((
            request_data.get("dietary_constraints"),
            request_data.get("exclude"),
            request_data.get("exclude_cuisines"),
        ))
        has_drinks = any(
            str(scene or "").lower() in {"下酒", "with drinks"}
            for scene in request_data.get("scenes") or []
        )
        if lang == "en":
            facts = [f"I'll plan for {people} people" if people else "The table size is clear"]
            facts.append(
                "there are no dietary restrictions"
                if no_constraints else "the dietary requirements are noted"
            )
            if has_drinks:
                facts.append("and I'll keep the drinks pairing in mind")
            acknowledgement = ", ".join(facts) + "."
        else:
            facts = [
                f"我先按{people}人准备" if people else "人数已经清楚了",
                "大家没忌口" if no_constraints else "饮食要求也记下了",
            ]
            if has_drinks:
                facts.append("配酒场景也记下了")
            acknowledgement = "，".join(facts) + "。"
    else:
        acknowledgement = _clarification_acknowledgements(context, dimension, lang)[0]
    return _join_clarification(
        acknowledgement,
        _clarification_question(dimension, lang, search_request),
        lang,
    )


def _request_food_anchors(search_request, lang: str) -> list[str]:
    """提取用户本轮明确给出的主料/菜名，供安全兜底承接事实。"""
    if search_request is None:
        return []
    data = (
        search_request.model_dump()
        if hasattr(search_request, "model_dump")
        else dict(search_request)
    )
    values = [
        *_as_list(data.get("dishes"), limit=4),
        *_as_list(data.get("ingredients"), limit=6),
    ]
    values = _same_language(values, lang, limit=6)
    # 同时出现具体词和其上位词时只说更具体的一个，避免承接文案像在念槽位。
    ordered = sorted(dict.fromkeys(values), key=len, reverse=True)
    specific = [
        value for value in ordered
        if not any(value != other and value in other for other in ordered)
    ]
    return specific[:2]


def _with_core_fact_grounding(
    narrative: RecommendationNarrative,
    search_request,
    lang: str,
) -> RecommendationNarrative:
    """确保推荐导语至少提到一个明确菜名/主料，不针对任何特定场景。"""
    narrative = _with_applied_preference_disclosure(
        narrative,
        search_request,
        lang,
    )
    anchors = _request_food_anchors(search_request, lang)
    if not anchors:
        return narrative
    opening = str(narrative.opening or "").strip()
    lowered = opening.lower()
    if any(anchor.lower() in lowered for anchor in anchors):
        return narrative
    prefix = (
        f"The meal still centers on {', '.join(anchors)}."
        if lang == "en" else
        f"这顿先围绕{'、'.join(anchors)}来。"
    )
    return RecommendationNarrative(
        f"{prefix}{' ' if lang == 'en' and opening else ''}{opening}",
        narrative.strategy,
        narrative.recipe_reasons,
        narrative.closing,
    )


def _with_applied_preference_disclosure(
    narrative: RecommendationNarrative,
    search_request,
    lang: str,
) -> RecommendationNarrative:
    """兜底确保真实参与本轮推荐的偏好被自然告知用户。"""
    if search_request is None:
        return narrative
    data = (
        search_request.model_dump()
        if hasattr(search_request, "model_dump")
        else dict(search_request)
    )
    applied = [
        item
        for item in data.get("applied_memory_constraints") or []
        if isinstance(item, dict)
    ]
    if not applied:
        return narrative
    hard_avoids = list(dict.fromkeys(
        str(item.get("display") or item.get("value") or "").strip()
        for item in applied
        if item.get("kind") in {"allergen", "exclude", "exclude_cuisine"}
        and str(item.get("display") or item.get("value") or "").strip()
    ))
    # 改法 A：dislikes 走软过滤，kind=soft_dislike，告知但不强承诺"避开"
    soft_dislikes = list(dict.fromkeys(
        str(item.get("display") or item.get("value") or "").strip()
        for item in applied
        if item.get("kind") == "soft_dislike"
        and str(item.get("display") or item.get("value") or "").strip()
    ))
    dietary = list(dict.fromkeys(
        str(item.get("display") or item.get("value") or "").strip()
        for item in applied
        if item.get("kind") == "dietary_constraint"
        and str(item.get("display") or item.get("value") or "").strip()
    ))
    likes = list(dict.fromkeys(
        str(item.get("display") or item.get("value") or "").strip()
        for item in applied
        if item.get("kind") == "soft_preference"
        and str(item.get("display") or item.get("value") or "").strip()
    ))
    if not any((hard_avoids, soft_dislikes, dietary, likes)):
        return narrative
    opening = str(narrative.opening or "").strip()
    mentioned = all(
        value.lower() in opening.lower()
        for value in [*hard_avoids, *soft_dislikes, *dietary, *likes]
    )
    if mentioned and (
        ("earlier" in opening.lower() or "you said" in opening.lower())
        if lang == "en"
        else ("之前" in opening or "你说过" in opening)
    ):
        return narrative
    parts: list[str] = []
    if lang == "en":
        if hard_avoids:
            parts.append(f"I kept {', '.join(hard_avoids)} out")
        if soft_dislikes:
            parts.append(f"I kept your preference against {', '.join(soft_dislikes)} in mind")
        if dietary:
            parts.append(f"I followed {', '.join(dietary)}")
        if likes:
            parts.append(f"I also used your preference for {', '.join(likes)}")
        prefix = "Based on what you told me earlier, " + "; ".join(parts) + "."
    else:
        if hard_avoids:
            parts.append(f"这轮已经避开{'、'.join(hard_avoids)}")
        if soft_dislikes:
            parts.append(f"也参考了你之前说不喜欢{'、'.join(soft_dislikes)}")
        if dietary:
            parts.append(f"继续按{'、'.join(dietary)}筛选")
        if likes:
            parts.append(f"也参考了你喜欢{'、'.join(likes)}这一点")
        prefix = "按你之前明确说过的，" + "；".join(parts) + "。"
    return RecommendationNarrative(
        f"{prefix}{' ' if lang == 'en' and opening else ''}{opening}",
        narrative.strategy,
        narrative.recipe_reasons,
        narrative.closing,
    )


def _search_bridge_fallback(
    context: str,
    mode: str,
    lang: str,
    search_request=None,
) -> str:
    """模型不可用时的短缓冲；不包含任何菜谱事实或设备动作。"""
    lower = str(context or "").lower()
    anchors = _request_food_anchors(search_request, lang)
    tired = any(word in lower for word in (
        "累", "疲惫", "下班", "加班", "不想动", "tired", "exhausted", "after work",
    ))
    party = any(word in lower for word in (
        "朋友", "客人", "聚会", "聚餐", "friends", "guests", "party",
    ))
    if lang == "en":
        if anchors:
            return f"Got it—I'll keep {', '.join(anchors)} at the center and weigh the other conditions around it."
        if tired:
            return "That sounds like enough thinking for one day. I'll keep the choice easy and stay within what you asked for."
        if party:
            return "With people coming over, the table needs a bit of balance. I'll weigh the details you gave me before narrowing it down."
        return (
            "Got it—I'll stay close to those conditions and see what genuinely fits."
            if mode == "search" else
            "Got it. I'll weigh what matters for this meal and narrow the choice down."
        )
    if anchors:
        return f"明白，这顿先围绕{'、'.join(anchors)}来，其它条件一起考虑。"
    if tired:
        return "今天已经够费神了，这顿就别再让你做选择题。我按你说的条件慢慢收一收。"
    if party:
        return "朋友来吃饭，既要照顾口味，也得让一桌子搭得顺。我按你刚说的条件先捋一遍。"
    return "明白，我就贴着你说的这些条件找，不往外乱加。" if mode == "search" else "明白，这顿最重要的是选得合适，不是堆一串菜名。我按你说的情况仔细挑一下。"


def _valid_search_bridge(text: str, *, lang: str, grounded_source: str) -> bool:
    """检索前没有菜谱事实：只允许场景承接，不允许任何推荐结论。"""
    clean = _clean_text(text, limit=260)
    if not clean or len(clean) < 4:
        return False
    lower = clean.lower()
    if any(word in lower for word in _INTERNAL_WORDS):
        return False
    history_words = _BRIDGE_HISTORY_WORDS_EN if lang == "en" else _BRIDGE_HISTORY_WORDS_ZH
    if any(word in lower for word in history_words):
        return False
    if any(word in lower for word in (
        "启动", "开火", "开始做", "执行设备", "设备执行",
        "start the device", "start cooking", "run the device",
    )):
        return False
    if re.search(r"【[^】]+】", clean):
        return False
    if _HEALTH_CLAIM_RE_EN.search(clean) or _HEALTH_CLAIM_RE_ZH.search(clean):
        return False
    # 检索前不得声称某道菜、食材、做法或耗时适合用户。模型只能复述用户
    # 已说出的条件；所有高风险事实词必须能在当前输入/近期原话中找到。
    source = str(grounded_source or "").lower()
    negation_terms = (
        _PREFERENCE_NEGATION_TERMS_EN
        if lang == "en" else _PREFERENCE_NEGATION_TERMS_ZH
    )
    if any(term in lower and term not in source for term in negation_terms):
        return False
    for match in re.finditer(
        r"([\u4e00-\u9fff]{2,18}(?:肉|汤|羹|面|饭|粥|鸡|鸭|鱼|虾|蛋|豆腐|排骨|牛腩|土豆|茄子|丸子|饼|包|糕))"
        r"(?:也)?(?:很|挺|更|比较)?(?:合适|可以|推荐)",
        clean,
    ):
        candidate = _compact_fact_text(match.group(1))
        if candidate and candidate not in _compact_fact_text(source):
            return False
    terms = _GROUNDING_TERMS_EN if lang == "en" else _GROUNDING_TERMS_ZH
    if any(term in lower and term not in source for term in terms):
        return False
    for number in re.findall(r"\d+(?:\.\d+)?", lower):
        if number not in source:
            return False
    if lang == "en" and re.search(r"[一-鿿]", clean):
        return False
    return True


async def generate_search_bridge(
    original_question: str,
    *,
    mode: str,
    lang: str = "zh",
    search_request=None,
    recent_turns=None,
    profile_context: dict | None = None,
) -> str:
    """搜索/推荐开始前的自然承接；此时严格禁止生成任何菜谱事实。"""
    recent_context = _recent_user_context(recent_turns, original_question)
    declared_preferences = _safe_profile_preferences(profile_context, lang)
    grounded_source = " ".join([
        str(original_question or ""),
        *recent_context,
        json.dumps(declared_preferences, ensure_ascii=False),
    ])
    language = "English" if lang == "en" else "简体中文"
    task_label = "按明确条件寻找匹配菜谱" if mode == "search" else "替用户为这一顿做选择"
    system = f"""{_STYLE_PROMPT}

你正在真实菜谱查询开始之前，给用户第一段自然承接。请使用{language}。
当前任务是：{task_label}。
这时还没有任何菜谱结果，所以绝对不能出现新菜名、推荐结论、食材补充、做法、时长、营养、疗效、图片或设备操作。
按固定顺序组织这一段：
1. 用户明确有累、饿、烦、开心、纠结等情绪时，第一句先接住情绪；没有明确情绪时就回应场景，不能硬猜情绪。
2. 第二句可自然承接 current_question、recent_user_context 或 declared_preferences 中与本轮直接相关的已知条件，让用户感觉你听懂了，而不是复述字段。
3. 用户同时给出明确主料/菜名和人数、时间、情绪、身体状态等场景条件时，必须把主料/菜名作为本轮核心事实；场景只能辅助选择，不能把任务改成另一类食物。
declared_preferences 只包含用户主动确认过的稳定偏好。可以自然照顾，但不能说"我记得、上次、历史记录"，也不能把无关旧偏好强塞进当前话题。
只能说"会按这些条件来选"，不能解释某种口味或食材对身体有什么好处。尤其禁止"给肠胃减负、暖胃、养胃、好消化、补能量、治愈"等健康效果。
不能提系统实现。
不要问问题（需要澄清的请求已在更早步骤处理），不要说"我先挑菜、我去检索、为你推荐以下"。
写成一个自然段，共 1～2 句，像熟人微信聊天，短一点、有语气但不过度表演。
只输出 JSON：{{"message":"..."}}。"""
    payload = {
        "current_question": str(original_question or "").strip(),
        "recent_user_context": recent_context,
        "declared_preferences": declared_preferences,
        "known_request": _current_request(search_request),
    }
    try:
        response = await observe_model_call(
            lambda: _recommendation_llm.ainvoke([
                SystemMessage(content=system),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]),
            stage="recommendation_bridge",
            model=settings.LLM_MODEL,
            timeout_seconds=15,
            reply_generation=True,
        )
        message = _clean_text(_extract_json(response.content).get("message"), limit=260)
        if _valid_search_bridge(message, lang=lang, grounded_source=grounded_source):
            return message
        logger.warning("检索前缓冲文案未通过事实校验，使用安全兜底")
    except Exception as exc:
        logger.warning("检索前缓冲模型调用失败，使用安全兜底：error_type=%s", type(exc).__name__)
    return _search_bridge_fallback(
        grounded_source,
        mode,
        lang,
        search_request,
    )


# 澄清话术 LLM 化开关，默认开启。CLARIFICATION_LLM=0 可回退到旧的固定话术
# （测试 / 灰度期排查用）。
_CLARIFICATION_LLM_ENABLED = (
    os.getenv("CLARIFICATION_LLM", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)

_CLARIFICATION_INTERNAL_WORDS = (
    "检索", "召回", "候选", "数据库", "提示词", "模型", "意图",
    "retrieval", "candidate", "database", "prompt", "model", "intent",
)
_CLARIFICATION_SAFETY_TERMS_ZH = (
    "过敏", "忌口", "不吃", "饮食限制", "饮食要求", "素食", "清真", "宗教",
)
_CLARIFICATION_SAFETY_TERMS_EN = (
    "allerg", "dietary restriction", "food restriction", "do not eat",
    "don't eat", "vegetarian", "vegan", "halal", "religious",
)
_CLARIFICATION_FLAVOR_TERMS_ZH = (
    "口味", "清淡", "家常", "下饭", "偏辣", "辣", "咸鲜", "酸甜", "重口",
)
_CLARIFICATION_FLAVOR_TERMS_EN = (
    "flavor", "flavour", "light", "homestyle", "hearty", "spicy", "savory", "savoury",
)

# 地域身份可以作为用户明确给出的场景事实，但不能被扩写成群体口味或性格。
# 这里拦截的是“身份 + 必然/通常 + 口味或酒量”的推断句，不拦截“在杭州请客”
# 或“用户明确说想吃江西菜”这类事实承接。
_CLARIFICATION_REGIONAL_STEREOTYPE_ZH_RE = re.compile(
    r"(?:都是?|作为|身为|来自|老家(?:在|是)|籍贯(?:在|是)|"
    r"[\u4e00-\u9fff]{2,8}(?:人|老乡|汉子|爷们|妹子))"
    r"[^。！？!?]{0,28}"
    r"(?:肯定|一定|必须|当然|天生|就|都|一般|通常|自然|爱|能)"
    r"[^。！？!?]{0,20}"
    r"(?:口味|辣|重口|清淡|甜|咸|够味|够劲|硬菜|喝酒|白酒|酒量|豪爽|"
    r"赣菜|川菜|湘菜|粤菜|东北菜|本地菜|家乡菜|地方菜|菜系)",
)
_CLARIFICATION_REGIONAL_PROFILE_ZH_RE = re.compile(
    r"(?:作为|身为|来自|老家(?:在|是)|籍贯(?:在|是)|"
    r"[\u4e00-\u9fff]{2,8}(?:人|老乡|汉子|爷们|妹子))"
    r"[^。！？!?]{0,20}"
    r"(?:口味(?:偏|重|轻)|偏爱|爱吃|喜欢吃|能喝|酒量|豪爽|硬菜|够味|够劲|"
    r"香辣|麻辣|重口|清淡|赣菜|川菜|湘菜|粤菜|东北菜|本地菜|家乡菜|地方菜|菜系)",
)
_CLARIFICATION_REGIONAL_STEREOTYPE_EN_RE = re.compile(
    r"(?:\b(?:people|folks|men|women)\s+from\s+[a-z .'-]{2,30}|"
    r"\b[a-z .'-]{2,30}\s+(?:people|folks|men|women))"
    r"[^.!?]{0,32}"
    r"\b(?:all|always|must|naturally|usually|surely|love|like|can\s+handle)\b"
    r"[^.!?]{0,24}"
    r"\b(?:spicy|bold|strong|mild|sweet|salty|drink|liquor|alcohol)\b",
    re.IGNORECASE,
)


def _valid_llm_clarification(
    text: str,
    *,
    dimension: str,
    lang: str,
    search_request=None,
) -> bool:
    """Validate pre-search prose before it reaches the user.

    The model may make the wording natural, but it may not infer preferences from
    regional identity or drift away from the deterministic clarification target.
    """
    clean = str(text or "").strip()
    if not clean or len(clean) > 200:
        return False
    lower = clean.lower()
    if any(word in lower for word in _CLARIFICATION_INTERNAL_WORDS):
        return False
    if clean.count("？") + clean.count("?") != 1:
        return False

    if lang == "en":
        if _CLARIFICATION_REGIONAL_STEREOTYPE_EN_RE.search(clean):
            return False
        safety_terms = _CLARIFICATION_SAFETY_TERMS_EN
        flavor_terms = _CLARIFICATION_FLAVOR_TERMS_EN
    else:
        # “老表”即使没有继续补口味，也是在替用户生成地域化群体称呼；澄清场景
        # 不需要它，直接使用安全兜底更稳妥。
        if (
            "老表" in clean
            or _CLARIFICATION_REGIONAL_STEREOTYPE_ZH_RE.search(clean)
            or _CLARIFICATION_REGIONAL_PROFILE_ZH_RE.search(clean)
        ):
            return False
        safety_terms = _CLARIFICATION_SAFETY_TERMS_ZH
        flavor_terms = _CLARIFICATION_FLAVOR_TERMS_ZH

    if dimension in {"party_constraints", "party_constraints_detail"}:
        return any(term in lower for term in safety_terms)
    if dimension in {"flavor_preferences", "party_preferences"}:
        request_data = (
            search_request.model_dump()
            if hasattr(search_request, "model_dump")
            else dict(search_request or {})
        )
        if request_data.get("constraints_confirmed") and any(
            term in lower for term in safety_terms
        ):
            return False
        return any(term in lower for term in flavor_terms)
    return True


async def generate_recommendation_clarification(
    original_question: str,
    dimension: str,
    lang: str = "zh",
    search_request=None,
    recent_turns=None,
) -> str:
    """检索前的单轮澄清：用 LLM 生成自然话术，承接用户原话，只问一个问题。

    旧实现是直接拼固定话术（"咱们先把范围缩小一点。家里有什么主料..."），
    每次都一样、问题太多、客服腔重。新实现：
      1. 把 dimension + 用户原话 + 对话历史 交给 LLM
      2. LLM 生成一句自然的承接 + 一个最相关的问题
      3. LLM 失败/超时时回退到旧的固定话术
      4. 强制只问一个问题（不要连环问）
    CLARIFICATION_LLM=0 可回退到旧实现（测试 / 灰度期排查用）。
    """
    # 开关关闭时直接走旧实现
    if not _CLARIFICATION_LLM_ENABLED:
        recent_context = _recent_user_context(recent_turns, original_question)
        context_text = " ".join([str(original_question or ""), *recent_context])
        return _clarification_fallback(
            context_text,
            dimension,
            lang,
            search_request,
        )

    # 维度对应的"澄清目标"自然语言描述（给 LLM 看，不直接给用户）
    dimension_hints = {
        "recommendation_basics": "需要知道这顿饭的大致方向（人数或口味倾向），但不能一次问多个",
        "party_size": "需要知道几个人吃",
        "flavor_preferences": "只需要知道口味倾向（清淡/家常/偏辣等）；已明确无忌口时绝不能重问忌口",
        "available_ingredients": "需要知道用户手头有什么主料",
        "ingredient_or_flavor": "需要知道主料或口味方向",
        "party_preferences": "需要知道聚餐的口味方向",
        "party_constraints": "需要确认有没有人过敏/忌口/宗教饮食要求",
        "party_constraints_detail": "需要知道具体的过敏原或饮食限制",
        "remembered_preference_conflict": "用户本轮要求和之前记录的偏好冲突，需要确认",
        "remembered_dietary_constraint": "用户之前提过的阶段性饮食目标，需要确认这顿是否还用",
    }
    hint = dimension_hints.get(dimension, "需要向用户确认一个关键信息才能继续")

    # 对话历史（承接上下文）
    recent_context = _recent_user_context(recent_turns, original_question)

    language = "English" if lang == "en" else "简体中文"
    system = (
        "你是 CookClaw，一个懂做饭、会聊天的厨房搭子。\n"
        f"请使用{language}。\n"
        "\n"
        "你的任务：用户正在找你帮忙选菜，但你还需要一个关键信息才能开始找。\n"
        "你需要写一句话：\n"
        "  1. 先自然地接住用户刚才说的话（不要重复、不要说\"好的\"\"可以\"这类空话）\n"
        "  2. 然后只问一个最相关的问题（不要一次问多个，不要列选项清单）\n"
        "  3. 像朋友在微信里说话，不要客服腔，不要\"咱们先把范围缩小一点\"这类套话\n"
        "  4. 整段控制在 1～2 句、30～80 字\n"
        "  5. 最多 1 个 Emoji，不滥用感叹号\n"
        "  6. 籍贯、民族、地域身份和居住地只作场景背景，不得据此推断任何人的口味、菜系偏好、酒量或性格\n"
        "  7. 不得使用‘老表’等地域化群体称呼，也不得写‘某地人就爱辣’‘够辣够硬’‘能喝’等刻板话术\n"
        "  8. 只有用户明确说出的口味和饮食要求才可承接；本轮要确认过敏或忌口时，不得擅自改问辣不辣\n"
        "\n"
        "直接输出澄清话术本身，不要加引号、前缀或解释。"
    )

    context_lines = [f"用户刚说：{str(original_question or '').strip()}"]
    if recent_context:
        context_lines.append("之前对话（最新在后）：")
        for line in recent_context[-3:]:
            context_lines.append(f"  - {line}")
    context_lines.append(f"这一轮需要澄清的方向：{hint}")

    user_prompt = "\n".join(context_lines)

    try:
        response = await _recommendation_llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user_prompt),
        ])
        text = str(response.content or "").strip().strip("\"'")
        if _valid_llm_clarification(
            text,
            dimension=dimension,
            lang=lang,
            search_request=search_request,
        ):
            return text
        logger.warning(
            "clarification LLM 文案未通过地域/维度安全校验，回退固定话术：dimension=%s",
            dimension,
        )
    except asyncio.TimeoutError:
        logger.warning("clarification LLM 超时，回退固定话术")
    except Exception as e:
        logger.warning("clarification LLM 失败，回退固定话术：%s", e)

    # 回退：用旧的固定话术（至少保证功能可用）
    recent_context_text = " ".join([str(original_question or ""), *recent_context])
    return _clarification_fallback(
        recent_context_text,
        dimension,
        lang,
        search_request,
    )


def _empty_narrative() -> RecommendationNarrative:
    return RecommendationNarrative("", "", {}, "")


def build_recommendation_narrative(
    original_question: str,
    search_query: str,
    search_result: dict,
    lang: str = "zh",
    search_request=None,
) -> RecommendationNarrative:
    """无模型兜底：不生成对话话术，只保留真实卡片数据。"""
    return _empty_narrative()


def _extract_json(content: str) -> dict:
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
        value = json.loads(match.group(0))
    return value if isinstance(value, dict) else {}


def _clean_text(value, *, limit: int = 360) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


_INGREDIENT_CLAIM_PATTERNS_ZH = (
    re.compile(r"(?:用了|用到|含有|包含|加入|加了|放了|主料(?:是|有)|食材(?:是|有|包括)|配料(?:是|有|包括))\s*([^，。；！？]{1,60})"),
    re.compile(
        r"(?:^|[，。；：:]\s*)(?:【[^】]{1,80}】\s*)?(?:它|这道菜?|这款)?"
        r"(?<!没)有(?!点|些|空|人|什么|没有)\s*([^，。；！？]{1,60})"
    ),
)
_INGREDIENT_CLAIM_PATTERNS_EN = (
    re.compile(
        r"\b(?:contains?|includes?|uses?|made with|features?|ingredients? (?:include|are)|has)\s+"
        r"([^,.!?;]{1,80})",
        flags=re.IGNORECASE,
    ),
)
_CLAIM_GENERIC_WORDS_ZH = (
    "主要", "主要的", "新鲜", "新鲜的", "少许", "一点", "一些", "适量",
    "作为主料", "做主料", "等食材", "等配料", "等", "食材", "配料", "主料", "标签",
)
_CLAIM_GENERIC_WORDS_EN = (
    "a", "an", "the", "some", "fresh", "main", "ingredient", "ingredients", "tag", "tags",
)


def _compact_fact_text(value: str) -> str:
    return re.sub(r"[^0-9a-z一-鿿]+", "", str(value or "").lower())


def _ingredient_claim_items(claim: str, lang: str) -> list[str]:
    if lang == "en":
        values = re.split(r"\s*(?:,|\band\b|\bor\b|/)\s*", claim, flags=re.IGNORECASE)
        generic = _CLAIM_GENERIC_WORDS_EN
    else:
        values = re.split(r"[、/]|(?:以及|还有|和|与|及)", claim)
        generic = _CLAIM_GENERIC_WORDS_ZH
    out = []
    for raw in values:
        value = str(raw or "").strip(" ，,。；;:：！!？?")
        for word in sorted(generic, key=len, reverse=True):
            if lang == "en":
                value = re.sub(rf"\b{re.escape(word)}\b", " ", value, flags=re.IGNORECASE)
            else:
                value = value.replace(word, "")
        value = re.sub(r"\s+", " ", value).strip()
        if value:
            out.append(value)
    return out


def _claim_item_supported(item: str, fact: dict) -> bool:
    claim = _compact_fact_text(item)
    if not claim:
        return True
    source_values = [
        fact.get("name"),
        *(fact.get("ingredients") or []),
        *(fact.get("tags") or []),
        fact.get("description"),
    ]
    compact_values = [
        _compact_fact_text(value) for value in source_values if _compact_fact_text(value)
    ]
    if any(claim == value or claim in value for value in compact_values):
        return True

    # A short claim may name two adjacent real fields without a conjunction.  Remove
    # only complete fact values; any residue means the model introduced a new token.
    residue = claim
    for value in sorted(compact_values, key=len, reverse=True):
        if len(value) >= 2:
            residue = residue.replace(value, "")
    return not residue


def _ingredient_claims_are_grounded(text: str, facts: list[dict], lang: str) -> bool:
    """Reject ingredient/tag assertions not supported by the referenced recipe facts."""
    patterns = _INGREDIENT_CLAIM_PATTERNS_EN if lang == "en" else _INGREDIENT_CLAIM_PATTERNS_ZH
    claims = [match.group(1) for pattern in patterns for match in pattern.finditer(text)]
    if not claims:
        return True

    mentioned_names = {
        name for name in re.findall(r"【([^】]{1,80})】", text)
    } if lang != "en" else {
        str(fact.get("name") or "")
        for fact in facts
        if str(fact.get("name") or "").lower() in text.lower()
    }
    scoped = [fact for fact in facts if fact.get("name") in mentioned_names] or list(facts)
    if not scoped:
        return False

    for claim in claims:
        items = _ingredient_claim_items(claim, lang)
        if not items:
            return False
        # A reason is scoped to one fact.  In a multi-recipe sentence, accept a shared
        # assertion only when every named recipe supports it; ambiguous attribution is
        # safer to drop than to assign one recipe's ingredient to another.
        if any(not all(_claim_item_supported(item, fact) for fact in scoped) for item in items):
            return False
    return True


def _references_foreign_recipe_fact(text: str, own_fact: dict, facts: list[dict]) -> bool:
    """Prevent a reason from borrowing another result's name, ingredient, or tag."""
    text_compact = _compact_fact_text(text)
    own_values = [
        own_fact.get("name"),
        *(own_fact.get("ingredients") or []),
        *(own_fact.get("tags") or []),
    ]
    own_compact = " ".join(_compact_fact_text(value) for value in own_values)
    for fact in facts:
        if str(fact.get("id")) == str(own_fact.get("id")):
            continue
        foreign_values = [
            fact.get("name"),
            *(fact.get("ingredients") or []),
            *(fact.get("tags") or []),
        ]
        for value in foreign_values:
            term = _compact_fact_text(value)
            if len(term) >= 2 and term in text_compact and term not in own_compact:
                return True
    return False


def _text_is_grounded(
    text: str,
    *,
    allowed_names: set[str],
    lang: str,
    grounded_source: str,
    closing: bool = False,
    ingredient_facts: list[dict] | None = None,
) -> bool:
    """分级校验：硬约束必拒，软约束按 GROUNDING_STRICTNESS 决定。

    硬约束（任何严格度都拒）：
      - 内部词（检索/候选/数据库/提示词/模型）
      - 健康宣称（养胃/减肥/解酒 等）
      - 英文输出中出现中文
      - 菜名不在 allowed_names（编造菜名）

    软约束（仅 strict 拒；medium/relaxed 只告警）：
      - 口味/做法描述词不在 source（_GROUNDING_TERMS）
      - 历史承接词不在 source（_HISTORY_GROUNDING_TERMS）
      - 额外搭配词不在 source（_ADDITION_GROUNDING_TERMS）
      - 否定偏好词不在 source（_PREFERENCE_NEGATION_TERMS）
      - closing 字段额外搭配（_CLOSING_ADDITION）
      - 数字不在 source（medium 严格度下也必拒，relaxed 下告警）
      - 跨菜谱事实借用（ingredient_facts 校验）

    任一软约束触发都会 log debug，便于灰度期观察。
    """
    clean = _clean_text(text)
    if not clean:
        return False
    lower = clean.lower()

    # 地域身份推断和无菜谱事实支持的成分/配酒最高级结论始终是硬拒绝，
    # 不受 GROUNDING_STRICTNESS 灰度等级影响。
    if _has_unsafe_narrative_inference(clean, lang):
        return False
    if _has_unsupported_recipe_positioning(clean, lang, ingredient_facts):
        return False

    strictness = _GROUNDING_STRICTNESS

    # ─── 硬约束（任何严格度都拒）─────────────────────────────
    if any(word in lower for word in _INTERNAL_WORDS):
        return False
    if lang == "en" and re.search(r"[一-鿿]", lower):
        return False
    health_claim_re = _HEALTH_CLAIM_RE_EN if lang == "en" else _HEALTH_CLAIM_RE_ZH
    if health_claim_re.search(lower):
        return False
    for name in re.findall(r"【([^】]{1,80})】", clean):
        if name not in allowed_names:
            return False
    # 中文菜名必须放在【】里；拦住绕过括号的编造菜名
    if lang != "en":
        for candidate in re.findall(
            r"(?<!【)([一-鿿]{1,16}?(?:肉|汤|羹|面|饭|粥|鸡|鸭|鱼|虾|蛋|菜|豆腐|排骨|牛腩|土豆|茄子|丸子|饼|包|糕))"
            r"(?:也)?(?:更|很|挺|正)?(?:可以|合适|推荐)",
            clean,
        ):
            candidate = candidate.strip("，。；：、")
            if candidate and len(candidate) >= 2 and not any(
                candidate in name or name in candidate for name in allowed_names
            ):
                return False

    source_lower = str(grounded_source or "").lower()

    # ─── 数字校验（strict/medium 必拒；relaxed 告警）────────
    numbers_ungrounded = [
        n for n in re.findall(
            r"\d+(?:\.\d+)?|[一二两三四五六七八九十百]+(?=分钟|小时|秒|克|毫升|度|人)",
            lower,
        )
        if n not in source_lower
    ]
    if numbers_ungrounded:
        if strictness != "relaxed":
            return False
        logger.debug("grounding(relaxed): 数字未 grounding: %s", numbers_ungrounded)

    # ─── 软约束：口味/做法描述词 ─────────────────────────────
    grounding_terms = _GROUNDING_TERMS_EN if lang == "en" else _GROUNDING_TERMS_ZH
    unmatched_grounding = [
        t for t in grounding_terms if t in lower and t not in source_lower
    ]
    if unmatched_grounding:
        if strictness == "strict":
            return False
        logger.debug("grounding(%s): 口味词未 grounding: %s", strictness, unmatched_grounding[:3])

    # ─── 软约束：历史承接词 ─────────────────────────────────
    history_terms = _HISTORY_GROUNDING_TERMS_EN if lang == "en" else _HISTORY_GROUNDING_TERMS_ZH
    unmatched_history = [
        t for t in history_terms if t in lower and t not in source_lower
    ]
    if unmatched_history:
        if strictness == "strict":
            return False
        logger.debug("grounding(%s): 历史词未 grounding: %s", strictness, unmatched_history)

    # ─── 软约束：额外搭配词 ─────────────────────────────────
    addition_terms = _ADDITION_GROUNDING_TERMS_EN if lang == "en" else _ADDITION_GROUNDING_TERMS_ZH
    unmatched_addition = [
        t for t in addition_terms if t in lower and t not in source_lower
    ]
    if unmatched_addition:
        if strictness == "strict":
            return False
        logger.debug("grounding(%s): 搭配词未 grounding: %s", strictness, unmatched_addition)

    # ─── 软约束：否定偏好词 ─────────────────────────────────
    negation_terms = (
        _PREFERENCE_NEGATION_TERMS_EN
        if lang == "en" else _PREFERENCE_NEGATION_TERMS_ZH
    )
    unmatched_negation = [
        t for t in negation_terms if t in lower and t not in source_lower
    ]
    if unmatched_negation:
        if strictness == "strict":
            return False
        logger.debug("grounding(%s): 否定偏好未 grounding: %s", strictness, unmatched_negation)

    # ─── 软约束：closing 字段额外搭配 ───────────────────────
    if closing:
        closing_addition = _CLOSING_ADDITION_EN if lang == "en" else _CLOSING_ADDITION_ZH
        hit_closing = [t for t in closing_addition if t in lower]
        if hit_closing:
            if strictness == "strict":
                return False
            logger.debug("grounding(%s): closing 额外搭配: %s", strictness, hit_closing)

    # ─── 软约束：跨菜谱事实借用 ─────────────────────────────
    if ingredient_facts is not None and not _ingredient_claims_are_grounded(
        clean, ingredient_facts, lang,
    ):
        if strictness == "strict":
            return False
        logger.debug("grounding(%s): 跨菜谱事实借用", strictness)

    return True



def _validate_narrative(
    payload: dict,
    facts: list[dict],
    lang: str,
    *,
    grounded_source: str,
    context_source: str = "",
) -> RecommendationNarrative:
    allowed_ids = {item["id"] for item in facts}
    allowed_names = {item["name"] for item in facts}
    checked_count = 0
    kept_count = 0

    def grounded_sentences(value, *, closing_field: bool = False) -> str:
        """逐句保留可核验内容，避免一句越界导致整段自然表达被清空。"""
        nonlocal checked_count, kept_count
        clean = _clean_text(value)
        sentences = re.findall(r"[^。！？.!?；;]+[。！？.!?；;]?", clean)
        kept = []
        for sentence in sentences:
            sentence = sentence.strip()
            checked_count += int(bool(sentence))
            if sentence and _text_is_grounded(
                sentence,
                allowed_names=allowed_names,
                lang=lang,
                grounded_source=grounded_source,
                closing=closing_field,
                ingredient_facts=facts,
            ):
                kept.append(sentence)
                kept_count += 1
        return " ".join(kept)

    opening = grounded_sentences(payload.get("opening"))
    strategy = grounded_sentences(payload.get("strategy"))
    closing = grounded_sentences(payload.get("closing"), closing_field=True)

    raw_reasons = payload.get("recipe_reasons") or {}
    reasons = {}
    facts_by_id = {str(item["id"]): item for item in facts}
    if isinstance(raw_reasons, dict):
        for recipe_id, reason in raw_reasons.items():
            key = str(recipe_id).strip()
            text = _clean_text(reason, limit=260)
            fact = facts_by_id.get(key)
            # 单菜理由只能由这道菜自己的事实支撑。用户说"想简单点"只能
            # 说明需求，不能反向证明任意候选做法简单。
            reason_source = json.dumps(fact, ensure_ascii=False) if fact else ""
            if fact and key in allowed_ids and not _references_foreign_recipe_fact(text, fact, facts):
                reason_sentences = re.findall(r"[^。！？.!?；;]+[。！？.!?；;]?", text)
                kept = [
                    sentence.strip()
                    for sentence in reason_sentences
                    if sentence.strip() and _text_is_grounded(
                        sentence.strip(),
                        allowed_names={str(fact.get("name") or "")},
                        lang=lang,
                        grounded_source=reason_source,
                        ingredient_facts=[fact],
                    )
                ]
                checked_count += sum(
                    1 for sentence in reason_sentences if sentence.strip()
                )
                kept_count += len(kept)
                if kept:
                    reasons[key] = " ".join(kept)
    record_reply_validation(checked=checked_count, kept=kept_count)
    return RecommendationNarrative(opening, strategy, reasons, closing)


def _safe_fallback_narrative(
    original_question: str,
    facts: list[dict],
    lang: str,
    search_request=None,
) -> RecommendationNarrative:
    """模型不可用时只复述本轮明确条件和真实菜名，不补齐缺失事实。"""
    names = [item["name"] for item in facts[:3] if item.get("name")]
    if not names:
        return _empty_narrative()
    anchors = _request_food_anchors(search_request, lang)
    request_data = _current_request(search_request)
    menu_facts = [
        item for item in facts
        if item.get("menu_role") in {"dish", "soup", "required_dish", "scoped_dish"}
    ]
    ingredient_matches = [
        item for item in facts
        if isinstance(item.get("ingredient_match"), dict)
    ]

    if menu_facts:
        requested_dishes = int(request_data.get("menu_dish_count") or 0)
        requested_soups = int(request_data.get("menu_soup_count") or 0)
        dishes = [item for item in menu_facts if item.get("menu_role") != "soup"]
        soups = [item for item in menu_facts if item.get("menu_role") == "soup"]
        complete = (
            len(dishes) == requested_dishes
            and len(soups) == requested_soups
        )
        party_size = int(request_data.get("party_size") or 0)
        exclusions = _same_language(
            _as_list(request_data.get("exclude"), limit=3)
            + _as_list(request_data.get("avoid"), limit=2)
            + _as_list(request_data.get("exclude_cuisines"), limit=2),
            lang,
            limit=4,
        )
        required = next(
            (item for item in dishes if item.get("menu_role") == "required_dish"),
            None,
        )
        if lang == "en":
            size = f"For {party_size} people, " if party_size else ""
            shape = f"{requested_dishes} dishes and {requested_soups} soup"
            constraint_text = (
                f", keeping out {', '.join(exclusions)}"
                if exclusions else ""
            )
            opening = (
                f"{size}this table is set as {shape}{constraint_text}."
                if complete else
                f"{size}I have {len(dishes)} dishes and {len(soups)} soups in place"
                f"{constraint_text}; the remaining slots are still open."
            )
            named = []
            if required:
                named.append(
                    f"{required['name']} fills the requested {required.get('menu_requirement') or 'main ingredient'} slot"
                )
            if soups:
                named.append(f"the soup is {soups[0]['name']}")
            strategy = "; ".join(named) + "." if named else (
                "The listed dishes are the verified table for this round."
            )
            closing = "Would you like to keep this table or replace one dish?"
        else:
            size = f"{party_size}个人这桌" if party_size else "这桌"
            shape = f"{requested_dishes}菜{requested_soups}汤"
            constraint_text = (
                f"，{'、'.join(exclusions)}都避开"
                if exclusions else ""
            )
            no_constraints = bool(request_data.get("constraints_confirmed")) and not any((
                request_data.get("dietary_constraints"),
                request_data.get("exclude"),
                request_data.get("exclude_cuisines"),
            ))
            has_drinks = any(
                str(scene or "").lower() in {"下酒", "with drinks"}
                for scene in request_data.get("scenes") or []
            )
            confirmed_context = ""
            if complete:
                context_parts = []
                if no_constraints:
                    context_parts.append("大家没忌口")
                if has_drinks:
                    context_parts.append("配白酒场景也记下了")
                if context_parts:
                    confirmed_context = "，" + "，".join(context_parts)
            opening = (
                f"{size}就按{shape}来配{constraint_text}{confirmed_context}，"
                "口味按默认多样化搭配。"
                if complete else
                f"{size}先配到{len(dishes)}道菜和{len(soups)}道汤{constraint_text}，"
                "没凑齐的菜位先留着。"
            )
            named = []
            if required:
                named.append(
                    f"你点名的{required.get('menu_requirement') or '主料'}菜用【{required['name']}】"
                )
            if soups:
                named.append(f"汤用【{soups[0]['name']}】")
            strategy = "，".join(named) + "。" if named else (
                "这几道就按当前菜单位置一起上桌。"
            )
            closing = "这桌方向可以吗，还是想先换掉其中一道？"
        return RecommendationNarrative(opening, strategy, {}, closing)

    if ingredient_matches:
        all_matches = [
            item
            for item in ingredient_matches
            if (item.get("ingredient_match") or {}).get("match_type") == "all"
        ]
        requested = list(
            (ingredient_matches[0].get("ingredient_match") or {}).get("requested") or []
        )
        if lang == "en":
            opening = (
                f"{' and '.join(requested)} can keep going in a few directions; "
                f"{len(all_matches)} of these recipes use both."
            )
            compared = " and ".join(item["name"] for item in (all_matches or ingredient_matches)[:2])
            strategy = f"Start by comparing {compared}; both keep the requested ingredients together."
            closing = "Which one should I open first?"
        else:
            opening = (
                f"{'和'.join(requested)}还能继续往下做，"
                f"这组里有{len(all_matches)}道同时用到这两样食材。"
            )
            compared = "和".join(
                f"【{item['name']}】"
                for item in (all_matches or ingredient_matches)[:2]
            )
            strategy = f"可以先比较{compared}，两道都没有把你刚说的食材拆开。"
            closing = "你想先看哪一道的做法？"
        return RecommendationNarrative(opening, strategy, {}, closing)

    directions = [
        *_as_list(request_data.get("flavors"), limit=2),
        *_as_list(request_data.get("methods"), limit=2),
        *_as_list(request_data.get("cuisines"), limit=2),
    ]
    directions = _same_language(directions, lang, limit=3)
    known = [*anchors, *directions]
    if lang == "en":
        opening = (
            f"I kept this round centered on {', '.join(known)}."
            if known else
            "Here are the recipe options I can verify."
        )
        chosen = " and ".join(names[:2])
        strategy = f"Start by comparing {chosen}; the other result can stay as a backup." if len(names) > 2 else (
            f"Start by comparing {chosen}." if len(names) > 1 else f"The current option is {chosen}."
        )
        closing = (
            "Does this one fit the mood, or should I change direction?"
            if len(names) == 1 else
            "Which one is closer to what you want right now?"
        )
    else:
        opening = (
            f"这轮就按你明确说的{'、'.join(known)}来选。"
            if known else
            "下面是当前能核验到的菜谱选项。"
        )
        chosen = "和".join(f"【{name}】" for name in names[:2])
        strategy = f"可以先比较{chosen}，其余的留作备选。" if len(names) > 2 else (
            f"可以先比较{chosen}。" if len(names) > 1 else f"目前更贴近的是{chosen}。"
        )
        closing = (
            "这道合胃口吗？不对我就换个方向。"
            if len(names) == 1 else
            "这两道里，你现在更偏哪一个？"
        )
    return RecommendationNarrative(opening, strategy, {}, closing)


def _is_default_diverse_party_menu(
    current_request: dict,
    menu_plan: dict,
) -> bool:
    """Whether the table direction was delegated to deterministic defaults.

    Location, origin, meal time and a drinks scene are context, not positive food
    directions.  Without an explicit dish, ingredient, cuisine, flavor or method,
    prose generation must not invent a positioning for the whole table.
    """
    return bool(
        menu_plan
        and current_request.get("constraints_confirmed")
        and not any(
            current_request.get(key)
            for key in (
                "dishes",
                "ingredients",
                "required_ingredients",
                "cuisines",
                "flavors",
                "methods",
            )
        )
    )


async def generate_recommendation_narrative(
    original_question: str,
    search_query: str,
    search_result: dict,
    lang: str = "zh",
    search_request=None,
    recent_turns=None,
    *,
    deep_agent_enabled: bool = False,
    agent_context: dict | None = None,
    input_context: dict | None = None,
) -> RecommendationNarrative:
    """依据真实候选组织推荐文案；受控 Agent 和旧单次模型共用事实校验。"""
    facts = _recipe_facts(search_result, lang)
    if not facts:
        return _empty_narrative()

    language = "English" if lang == "en" else "简体中文"
    system = f"""{_STYLE_PROMPT}

你现在只负责把已经选出的真实菜谱组织成一段自然推荐。请使用{language}。

必须遵守：
1. 只能推荐输入中的菜谱，不能创造新菜名、食材、时长、营养、功效或难度。
2. 不得提及检索、候选、数据库、历史记录、内部记忆、模型或系统。
3. 中文菜名必须原样放在【】中；英文菜名必须保持原名。
4. 只有 real_recipes 明确提供 step_count、estimated_time 或 servings 时才能引用；estimated_time 必须表达为估算，不能改写成精确承诺；不能从食材自行推断风味、健康效果或难度。
5. recipe_reasons 中每个菜谱 ID 只能引用该 ID 对应对象自己的字段，不能借用另一道菜的事实。
6. real_recipes 中的 detail_basis 只表示步骤、时长和份量属于展示估算，绝不能据此声称可以驱动设备。
7. current_request.image_context 来自视觉识别。可以自然说"图片里识别到/看起来有"，但必须表达为可纠正的视觉判断；用户没有明确写"冰箱"时，不能说成冰箱。
8. response_scenario=party_menu 时，按 menu_plan 和 menu_role 解释这桌怎么组成；menu_plan.complete=false 时必须明确说还缺哪些菜位。
9. response_scenario=ingredient_explore 时，优先解释 match_type=all 的菜如何同时回应多个食材；partial 只作为围绕一种食材的扩展选择。
10. user_profile 中若有 likes/dislikes/dietary_constraints/allergens，自然融入推荐语（如"按你之前说的，这轮避开辣"），不要生硬罗列；字段为空时不要假装使用了旧偏好。
11. 籍贯、民族、地域身份和居住地只能作为背景，不得据此推断口味、辣度、菜系、酒量、性格，也不得推断需要“硬菜”“够味”或“撑场面”；只有用户明确点名菜系或口味时才能采用。
12. 不得使用“老表”等地域化群体称呼。current_question 提到白酒只允许承接“配酒场景”，不能声称某道菜“最适合配白酒”；real_recipes 没有明确字段时不得声称“胶质满满”、口感、成分或风味。

只输出 JSON：
{{
  "opening": "2至3句，中文约70～140字；接住用户场景、复述最关键的明确条件并给出有烟火气的判断",
  "strategy": "1至2句，基于真实做法、估算时间、步骤数或份量告诉用户怎么选",
  "recipe_reasons": {{"菜谱ID": "朋友式的一句具体理由，不要系统腔"}},
  "closing": "一句自然问题，引导选择或查看详情，可为空"
}}"""
    current_request = _current_request(search_request)
    menu_plan = _menu_plan_facts(search_result)
    ingredient_explore = len(current_request.get("ingredients") or []) >= 2
    response_scenario = (
        "party_menu"
        if menu_plan
        else ("ingredient_explore" if ingredient_explore else "recipe_recommendation")
    )
    current_request["_response_scenario"] = response_scenario
    if menu_plan:
        current_request["_menu_plan"] = menu_plan
    image_context = _compact_input_context(input_context)
    if image_context:
        current_request["image_context"] = image_context
    payload = {
        "current_question": str(original_question or "").strip(),
        "retrieval_query": str(search_query or "").strip(),
        "response_scenario": response_scenario,
        "menu_plan": menu_plan,
        "current_request": current_request,
        "recent_user_context": _recent_user_context(recent_turns, original_question),
        "user_profile": _build_user_profile_payload(agent_context, lang),
        "real_recipes": facts,
    }
    context_source = json.dumps(
        {key: value for key, value in payload.items() if key != "real_recipes"},
        ensure_ascii=False,
    )
    grounding_suffix: list[str] = []
    if any(
        current_request.get(key)
        for key in ("exclude", "exclude_cuisines", "avoid", "dietary_constraints")
    ):
        grounding_suffix.extend(
            _PREFERENCE_NEGATION_TERMS_EN
            if lang == "en" else _PREFERENCE_NEGATION_TERMS_ZH
        )
    if menu_plan.get("complete"):
        grounding_suffix.extend(
            ("complete menu", "filled every slot", "ready", "set for the table")
            if lang == "en" else
            ("配齐", "凑齐", "完整菜单", "准备好", "已经配好")
        )
    validation_source = " ".join((
        json.dumps(payload, ensure_ascii=False),
        " ".join(grounding_suffix),
    )).strip()

    # 用户把整桌软偏好交给默认策略时，表达也必须保持确定性。场景中的籍贯、
    # 地点或白酒不能授权模型重新定义整桌口味，更不能生成酒效或人群画像。
    if _is_default_diverse_party_menu(current_request, menu_plan):
        mark_fallback("DEFAULT_DIVERSE_PARTY_MENU_DETERMINISTIC")
        fallback = _safe_fallback_narrative(
            original_question,
            facts,
            lang,
            search_request,
        )
        return _with_core_fact_grounding(fallback, search_request, lang)

    if deep_agent_enabled:
        try:
            from app.agent.controlled_deep_agent import (
                compose_recommendation_with_deep_agent,
            )

            agent_payload = await compose_recommendation_with_deep_agent(
                model=_recommendation_llm,
                current_question=payload["current_question"],
                recent_user_context=payload["recent_user_context"],
                current_request=payload["current_request"],
                recipes=facts,
                lang=lang,
                conversation_state=agent_context,
                timeout_seconds=settings.CONVERSATION_DEEP_AGENT_TIMEOUT_SECONDS,
            )
            with trace_stage("reply_validation"):
                narrative = _validate_narrative(
                    agent_payload or {},
                    facts,
                    lang,
                    grounded_source=validation_source,
                    context_source=context_source,
                )
            if not (narrative.opening or narrative.strategy):
                mark_fallback("DEEP_AGENT_REPLY_VALIDATION_FALLBACK")
                fallback = _safe_fallback_narrative(
                    original_question,
                    facts,
                    lang,
                    search_request,
                )
                narrative = RecommendationNarrative(
                    fallback.opening,
                    fallback.strategy,
                    narrative.recipe_reasons,
                    narrative.closing or fallback.closing,
                )
                logger.warning(
                    "受控 Deep Agent 推荐表达未通过完整事实校验，已使用安全兜底"
                )
            else:
                # Agent 的部分字段可能因越界被逐句过滤；常见推荐场景用同一批
                # 结构化事实补齐缺口，不让用户看到半截上下文或内部排序术语。
                fallback = _safe_fallback_narrative(
                    original_question,
                    facts,
                    lang,
                    search_request,
                )
                opening = narrative.opening
                if response_scenario == "party_menu":
                    requested_dishes = int(current_request.get("menu_dish_count") or 0)
                    requested_soups = int(current_request.get("menu_soup_count") or 0)
                    party_size = int(current_request.get("party_size") or 0)
                    zh_numbers = "零一二三四五六七八九十"
                    size_terms = {str(party_size)}
                    dish_terms = {str(requested_dishes)}
                    soup_terms = {str(requested_soups)}
                    if lang != "en":
                        for number, target in (
                            (party_size, size_terms),
                            (requested_dishes, dish_terms),
                            (requested_soups, soup_terms),
                        ):
                            if 0 <= number <= 10:
                                target.add(zh_numbers[number])
                    mentions_size = not party_size or any(
                        f"{term}{' people' if lang == 'en' else '人'}" in opening
                        or (lang == "en" and f"for {term}" in opening.lower())
                        for term in size_terms
                    )
                    mentions_shape = (
                        any(term in opening for term in dish_terms)
                        and any(term in opening for term in soup_terms)
                        and (
                            ("dish" in opening.lower() and "soup" in opening.lower())
                            if lang == "en" else
                            ("菜" in opening and "汤" in opening)
                        )
                    )
                    if not (mentions_size and mentions_shape):
                        opening = fallback.opening
                strategy = narrative.strategy
                if response_scenario == "party_menu" and not any(
                    str(fact.get("name") or "") in strategy
                    for fact in facts
                ):
                    strategy = fallback.strategy
                narrative = RecommendationNarrative(
                    opening or fallback.opening,
                    strategy or fallback.strategy,
                    narrative.recipe_reasons,
                    narrative.closing or fallback.closing,
                )
                logger.info(
                    "受控 Deep Agent 推荐表达完成: recipe_count=%s valid_reason_count=%s",
                    len(facts),
                    len(narrative.recipe_reasons),
                )
            return _with_core_fact_grounding(
                narrative,
                search_request,
                lang,
            )
        except Exception as exc:
            mark_fallback(f"DEEP_AGENT_REPLY_{type(exc).__name__.upper()}")
            logger.warning(
                "受控 Deep Agent 推荐表达失败，使用事实安全兜底: error_type=%s",
                type(exc).__name__,
            )
            fallback = _safe_fallback_narrative(
                original_question,
                facts,
                lang,
                search_request,
            )
            return _with_core_fact_grounding(
                fallback,
                search_request,
                lang,
            )

    async def invoke(prompt: str, timeout: int) -> RecommendationNarrative:
        response = await observe_model_call(
            lambda: _recommendation_llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]),
            stage="recommendation_reply_legacy",
            model=settings.LLM_MODEL,
            timeout_seconds=timeout,
            reply_generation=True,
        )
        with trace_stage("reply_validation"):
            return _validate_narrative(
                _extract_json(response.content),
                facts,
                lang,
                grounded_source=validation_source,
                context_source=context_source,
            )

    try:
        narrative = await invoke(system, 20)
        if not any((narrative.opening, narrative.strategy, narrative.recipe_reasons, narrative.closing)):
            # 第一次常见问题是模型顺手补了"十分钟、配米饭、暖胃"等常识。
            # 再给一次极窄的表达任务；仍不合规则安全降级，不把补写内容交给用户。
            repair_system = f"""{system}

上一次表达未通过事实校验。现在请严格重写：
- opening 只能回应用户当前状态，不描述菜的做法、味道、营养、功效或时长。
- strategy 和 recipe_reasons 只能原样引用 real_recipes 中的菜名、tags、ingredients；没有的字段一句也不补。
- recipe_reasons 必须按 ID 严格对齐，不能把其它菜的食材或标签写到这道菜上。
- 不得从食材推断风味，不得从 tags 推断分钟数或操作步骤。
- 不得从籍贯、地域、民族或所在地推断口味、菜系、酒量、硬菜或性格；不得使用地域化群体称呼。
- 不得写“胶质满满”“最适合配白酒”等 real_recipes 未明确支持的成分或最高级搭配结论。
- closing 最多问一个问题，只能问用户想看哪道菜的已存详情，不能建议任何搭配或新增食材。
- 如果某道菜除了 name 没有可用事实，只能说它是当前给出的选项，不能解释它"省事、好吃、营养或适合某类人"。
仍然只输出约定的 JSON；句式要自然，不要复制固定模板。"""
            narrative = await invoke(repair_system, 16)
        if (
            not (narrative.opening or narrative.strategy)
            or (response_scenario == "party_menu" and not narrative.opening)
        ):
            mark_fallback("LEGACY_REPLY_VALIDATION_FALLBACK")
            fallback = _safe_fallback_narrative(original_question, facts, lang, search_request)
            narrative = RecommendationNarrative(
                narrative.opening or fallback.opening,
                narrative.strategy or fallback.strategy,
                narrative.recipe_reasons,
                narrative.closing or fallback.closing,
            )
            logger.warning("推荐表达缺少有效导语，已使用事实安全的自然兜底")
        return _with_core_fact_grounding(
            narrative,
            search_request,
            lang,
        )
    except Exception as exc:
        mark_fallback(f"LEGACY_REPLY_{type(exc).__name__.upper()}")
        logger.warning("推荐表达模型调用失败，使用事实安全的自然兜底：error_type=%s", type(exc).__name__)
        fallback = _safe_fallback_narrative(original_question, facts, lang, search_request)
        return _with_core_fact_grounding(
            fallback,
            search_request,
            lang,
        )
