"""
确定性路由 —— orchestrator 层。

把 Web(chat_stream) 与 IM(qqbot_chat) 共享的「意图分类 → detect_lang →
尝试 recipe_search / greeting」前段收敛到一处，产出结构化 FastPathOutcome。
渲染（JSON / Markdown / 纯文本）、是否流式、以及 Agent 回退仍留在各通道边缘，
以保持各自的返回结构与流式能力不变。

不反向依赖 participle_agent（仅依赖 intent + fast_path），避免循环 import。
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from app.orchestrator.intent import (
    BusinessOperation,
    IntentCategory,
    IntentResult,
    classify_intent,
)
from app.orchestrator.routing_models import RouteDecision, RoutingContext
from app.orchestrator.routing_policy import (
    decide_exact_route,
    infer_classifier_action,
    outcome_action,
)
from app.orchestrator.clarification_copy import (
    affirmative_without_pending_message,
)
from app.agent.fast_path import (
    _run_search_subprocess,
    apply_search_plan,
    detect_lang,
    normalize_exact_search_label,
    plan_search_query,
)
from app.orchestrator.recommendation_response import (
    generate_recommendation_clarification,
    generate_search_bridge,
    generate_recommendation_narrative,
)
from app.orchestrator.search_request import (
    SearchRequest,
    delegates_recommendation_choice,
    exact_recipe_synonym_queries,
    extract_explicit_excludes,
    filter_hard_constraint_violations,
    has_explicit_no_dietary_restrictions,
    hard_constraint_notice,
    is_menu_quantity_dissatisfaction,
    remembered_preference_conflicts,
    requested_allergen_conflicts,
)
from app.orchestrator.search_selection import (
    filter_requested_result_type,
    prioritize_exact_matches,
    prioritize_ingredient_coverage,
    prioritize_soft_preferences,
    select_diverse_results,
)
from app.orchestrator.menu_plan import build_menu_plan
from app.orchestrator.web_search import (
    WebSearchResult,
    search_web,
    should_search_web,
)
from app.observability.trace import record_route_decision
from app.orchestrator.smalltalk import is_greeting_message
from app.orchestrator.device_knowledge import (
    device_product_answer,
    is_device_product_question,
)

logger = logging.getLogger(__name__)

_NO_PREFERENCE_REPLIES = {
    "都行", "都可以", "什么都行", "没有", "没忌口", "没有忌口", "你定吧",
    "没什么要求", "没有什么要求", "没要求", "没有要求",
    "你决定", "你看着办", "你安排", "你来安排", "随便", "看着办",
    "给我最终答案", "给我一个最终答案", "anything", "anything is fine",
    "anything works", "none", "no preference", "no restrictions", "you choose",
    "surprise me",
}
_ABANDON_PENDING_MARKERS = (
    "算了", "不说这个", "换个话题", "重新来", "重来",
    "never mind", "forget that", "new topic", "start over", "switch topic",
)
_TASTE_OR_CONSTRAINT_RE = re.compile(
    r"清淡|偏辣|辣|甜|咸|酸|下饭|家常|重口|不吃|不要|忌口|过敏|素食|纯素|"
    r"\b(?:light|spicy|sweet|savou?ry|homestyle|without|allergic|vegetarian|vegan)\b",
    flags=re.IGNORECASE,
)
_CONSTRAINT_REPLY_RE = re.compile(
    r"不吃|不要|忌口|过敏|素食|纯素|清真|回族|穆斯林|都能吃|什么都能吃|"
    r"\b(?:without|allergic|vegetarian|vegan|halal|muslim|"
    r"no restrictions?|no allergies)\b",
    flags=re.IGNORECASE,
)
_CONTEXTUAL_NO_CONSTRAINT_REPLIES = {
    "没有", "没", "无", "none", "都行", "都可以", "什么都行",
}
_CONTEXTUAL_HAS_CONSTRAINT_REPLIES = {
    "有", "有的", "有啊", "yes", "yeah", "yep",
}
_HAS_CONSTRAINTS_REPLY_RE = re.compile(
    r"(?<!没)有(?:一些|些|一点|点|具体)?"
    r"(?:忌口|过敏|饮食(?:要求|限制)|宗教饮食)",
    flags=re.IGNORECASE,
)
_REMEMBERED_DIET_AFFIRM_RE = re.compile(
    r"^(?:是|是的|对|对的|嗯|还在|继续|要|需要|按这个来|yes|yeah|yep|still|continue)$",
    flags=re.IGNORECASE,
)
_REMEMBERED_DIET_NEGATIVE_RE = re.compile(
    r"^(?:不|不是|不了|不用(?:了)?|不需要(?:了)?|没有|没|不在了|不减了|"
    r"这(?:顿|次)(?:不按这个|不用|不减肥|不减脂)|"
    r"no|nope|not anymore|not now|skip it)$",
    flags=re.IGNORECASE,
)
_PREFERENCE_CONFLICT_AFFIRM_RE = re.compile(
    r"^(?:是|是的|对|对的|嗯|可以|行|好|就这次|这次可以|临时例外|"
    r"yes|yeah|yep|okay|ok|just this time)$",
    flags=re.IGNORECASE,
)
_PREFERENCE_CONFLICT_NEGATIVE_RE = re.compile(
    r"^(?:不|不是|不了|不用|不要|不例外|还是避开|继续避开|"
    r"no|nope|do not|don't|keep avoiding it)$",
    flags=re.IGNORECASE,
)
_TRANSIENT_DIET_MARKERS = (
    "减肥", "减脂", "低脂", "控卡",
    "weight loss", "low-fat", "low fat",
)
_EFFORT_REPLY_RE = re.compile(
    r"简单|快手|省事|少油|少盐|\d+\s*(?:分钟|小时)|"
    r"\b(?:easy|quick|simple|low[- ]oil|low[- ]salt|\d+\s*(?:minutes?|hours?))\b",
    flags=re.IGNORECASE,
)
_ZH_INGREDIENT_REPLY_RE = re.compile(
    r"鸡蛋|番茄|西红柿|土豆|萝卜|蘑菇|香菇|豆腐|白菜|青菜|菠菜|豆角|茄子|"
    r"黄瓜|洋葱|牛肉|猪肉|鸡肉|鸭肉|羊肉|排骨|牛腩|鱼肉|鱼|虾|米饭|大米|"
    r"面条|面粉|馒头|玉米|南瓜|红薯|花生|牛奶|奶酪"
)
_EN_INGREDIENT_REPLY_RE = re.compile(
    r"\b(?:eggs?|tomato(?:es)?|potato(?:es)?|carrots?|mushrooms?|tofu|cabbage|"
    r"spinach|onions?|beef|pork|chicken|duck|lamb|ribs?|fish|shrimp|rice|noodles?|"
    r"flour|corn|pumpkin|peanuts?|milk|cheese)\b",
    flags=re.IGNORECASE,
)


def _relevance_threshold() -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv("RECIPE_RELEVANCE_THRESHOLD", "0.20"))))
    except (TypeError, ValueError):
        return 0.20


def _broad_relevance_threshold() -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv("RECIPE_BROAD_RELEVANCE_THRESHOLD", "0.17"))))
    except (TypeError, ValueError):
        return 0.17


def _score_at_least(item: dict, threshold: float) -> bool:
    try:
        return float(item.get("score") or 0.0) >= threshold
    except (TypeError, ValueError):
        return False


async def _notify_search_start(callback, query: str, lang: str, bridge: str) -> None:
    """兼容旧的二参数进度回调；新通道可接收第三个自然缓冲字段。"""
    if callback is None:
        return
    try:
        params = inspect.signature(callback).parameters.values()
        accepts_bridge = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) or sum(
            p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for p in params
        ) >= 3
    except (TypeError, ValueError):
        accepts_bridge = False
    if accepts_bridge:
        await callback(query, lang, bridge)
    else:
        await callback(query, lang)


async def _run_with_search_bridge(
    search_awaitable,
    *,
    question: str,
    query: str,
    mode: str,
    lang: str,
    search_request: SearchRequest,
    recent_turns: Optional[list],
    profile_context: Optional[dict],
    on_search_start: Optional[Callable[..., Awaitable[None]]],
) -> tuple[str, dict]:
    """执行真实检索；进度回调不再生成第二份面向用户的回复。

    推荐结果由唯一回复层组织最终表达。搜索开始阶段只传递空的进度信号，
    通道可以据此显示“正在输入”，但不能先发一段由另一模型生成的话。
    """
    search_task = asyncio.create_task(search_awaitable)
    try:
        bridge = ""
        await _notify_search_start(on_search_start, query, lang, bridge)
        return bridge, await search_task
    except BaseException:
        if not search_task.done():
            search_task.cancel()
        await asyncio.gather(search_task, return_exceptions=True)
        raise


def _abandons_pending_request(text: str) -> bool:
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return any(marker in value for marker in _ABANDON_PENDING_MARKERS)


def is_finalize_recommendation_request(text: str) -> bool:
    """用户把剩余选择权交给助手，要求基于已收集事实直接给最终推荐。"""
    return delegates_recommendation_choice(text)


def _repeats_complete_pending_request(
    text: str,
    pending_search_request: dict | None,
) -> bool:
    """重复发送同一完整请求，表示用户要求系统继续执行。

    只比较当前原话与 pending 的原始任务，不用菜系或口味等推断字段。
    后续安全维度门禁仍独立生效，因此重复请求不能替用户确认忌口。
    """
    if not pending_search_request:
        return False
    try:
        pending = SearchRequest(**pending_search_request)
    except (TypeError, ValueError):
        return False

    def compact(value: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())

    current = compact(text)
    original = compact(str(pending.original_text or "").split("；补充：", 1)[0])
    return bool(current and original and current == original)


def _continues_pending_party_menu_by_quantity(
    text: str,
    pending_search_request: dict | None,
    pending_clarification_dimension: str | None,
) -> bool:
    """“五个菜太少”承接当前多人菜单，不开新搜索任务。"""
    if not pending_search_request or not is_menu_quantity_dissatisfaction(text):
        return False
    # 数量反馈不能替用户回答“是否仍按旧饮食目标/偏好”。
    if pending_clarification_dimension in {
        "remembered_dietary_constraint",
        "remembered_preference_conflict",
    }:
        return False
    try:
        pending = SearchRequest(**pending_search_request)
    except (TypeError, ValueError):
        return False
    return bool(pending.party_size or pending.is_menu_plan)


def _normalized_short_reply(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower().strip(
        " ，,。.!！?？"
    )


def _explicit_transient_diet_term(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return next(
        (marker for marker in _TRANSIENT_DIET_MARKERS if marker in value),
        "",
    )


def _active_temporal_dietary_constraint(items: list[dict] | None) -> dict:
    now = time.time()
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        value = str(raw.get("value") or "").strip()
        if not value:
            continue
        try:
            expires_at = float(raw.get("expires_at") or 0)
        except (TypeError, ValueError):
            expires_at = 0
        if expires_at and expires_at <= now:
            continue
        return {
            "value": value,
            "expires_at": expires_at,
        }
    return {}


def _is_contextual_no_constraints_reply(
    text: str,
    dimension: str | None,
    intent=None,
) -> bool:
    """裸否定只有在正在回答饮食安全问题时，才表示“没有限制”。

    ``SearchRequest.from_intent_raw`` 仍不会把单独的“没有”全局解释成无忌口；
    这里依赖待澄清维度补上分轮对话里省略的宾语。
    """
    if dimension != "party_constraints":
        return False
    value = _normalized_short_reply(text)
    # 安全问题下的“没有/都可以”是确定性短答，文本证据优先。
    if value in _CONTEXTUAL_NO_CONSTRAINT_REPLIES:
        return True
    # 保留明确 false 的调用契约；真实路由在此函数之前会先校验
    # 公共的文本事实 helper，因此分类器 false 不会覆盖明确原话。
    if intent is not None and getattr(intent, 'is_no_constraints', None) is False:
        return False
    return has_explicit_no_dietary_restrictions(value)


def _is_explicit_no_constraints_reply(text: str, intent=None) -> bool:
    """本轮原话是否明确说明无忌口/过敏/饮食限制。"""
    del intent
    return has_explicit_no_dietary_restrictions(text)


def _is_contextual_has_constraints_reply(
    text: str,
    dimension: str | None,
) -> bool:
    """饮食安全问题的裸肯定需要继续收集具体限制，不能直接视为已确认。"""
    if dimension not in {"party_constraints", "party_constraints_detail"}:
        return False
    value = _normalized_short_reply(text)
    return bool(
        value in _CONTEXTUAL_HAS_CONSTRAINT_REPLIES
        or _HAS_CONSTRAINTS_REPLY_RE.search(value)
    )


def _looks_like_clarification_reply(text: str, dimension: str | None = None) -> bool:
    """识别对上一问的短补充，避免把普通闲聊误吞成菜谱条件。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not value or len(value) > 100:
        return False
    if _abandons_pending_request(value) or any(marker in value for marker in (
        "不聊了", "先不吃", "改天", "cancel", "not now",
    )):
        return False
    # “你来安排/给最终答案”是在明确放弃继续补充软偏好。它可以越过口味、
    # 食材等非安全问题，但不能替用户确认过敏、忌口或宗教饮食要求。
    if is_finalize_recommendation_request(value):
        return dimension not in {
            "party_constraints",
            "party_constraints_detail",
        }
    if any(marker in value for marker in (
        "为什么", "怎么", "是什么", "能不能", "多少", "几步",
        "why", "how", "what is", "can i", "how much",
    )):
        return False
    if _is_contextual_has_constraints_reply(value, dimension):
        return True
    if dimension == "remembered_dietary_constraint":
        normalized = _normalized_short_reply(value)
        return bool(
            _REMEMBERED_DIET_AFFIRM_RE.fullmatch(normalized)
            or _REMEMBERED_DIET_NEGATIVE_RE.fullmatch(normalized)
            or any(marker in normalized for marker in _TRANSIENT_DIET_MARKERS)
        )
    if dimension == "remembered_preference_conflict":
        normalized = _normalized_short_reply(value)
        return bool(
            _PREFERENCE_CONFLICT_AFFIRM_RE.fullmatch(normalized)
            or _PREFERENCE_CONFLICT_NEGATIVE_RE.fullmatch(normalized)
        )
    if value.strip(" ，,。.!！?") in _NO_PREFERENCE_REPLIES:
        return True

    taste_or_constraint = bool(_TASTE_OR_CONSTRAINT_RE.search(value))
    effort = bool(_EFFORT_REPLY_RE.search(value))
    party_size = bool(re.search(
        r"(?:^|\D)[一二两三四五六七八九十\d]{1,3}\s*(?:个人|人|位|people|guests?)(?:\D|$)",
        value,
        flags=re.IGNORECASE,
    ))
    inventory = bool(
        _ZH_INGREDIENT_REPLY_RE.search(value) or _EN_INGREDIENT_REPLY_RE.search(value)
    ) and bool(re.search(
        r"家里|冰箱|现成|剩下|还有|只有|有|没(?:有)?|"
        r"\b(?:fridge|pantry|ingredient|ingredients|have|has|only|left|no)\b",
        value,
        flags=re.IGNORECASE,
    ))
    # 用户也可能只回一个主料；限制为很短的食材短语，不能因为一句话里偶然
    # 出现“人/have/with”就吞掉“人生好难”或 “I have a headache”。
    compact_ingredient_reply = value.strip(" ，,。.!！?")
    zh_remainder = _ZH_INGREDIENT_REPLY_RE.sub("", compact_ingredient_reply)
    zh_remainder = re.sub(r"[和与、,，\s]+", "", zh_remainder)
    en_remainder = _EN_INGREDIENT_REPLY_RE.sub("", compact_ingredient_reply)
    en_remainder = re.sub(r"\b(?:and|or)\b|[,\s]+", "", en_remainder)
    ingredient_only = len(value) <= 24 and bool(compact_ingredient_reply) and (
        not zh_remainder or not en_remainder
    )

    if dimension in {"party_constraints", "party_constraints_detail"}:
        return bool(_CONSTRAINT_REPLY_RE.search(value))
    if dimension in {"party_preferences", "recommendation_basics"}:
        return taste_or_constraint or effort or party_size
    if dimension == "party_size":
        return party_size
    if dimension == "flavor_preferences":
        return taste_or_constraint or effort
    if dimension == "available_ingredients":
        return inventory or ingredient_only or taste_or_constraint or effort
    return inventory or ingredient_only or taste_or_constraint or effort or party_size


def _sanitize_no_constraints_followup(
    request: SearchRequest,
    question: str,
    dimension: str | None,
    intent: "IntentResult | None" = None,
) -> SearchRequest:
    """安全问题的“没有”只确认事实，不把维度名送进菜谱检索。"""
    normalized = _normalized_short_reply(question)
    if (
        dimension == "remembered_dietary_constraint"
        and (
            _REMEMBERED_DIET_NEGATIVE_RE.fullmatch(normalized)
            or any(
                phrase in normalized
                for phrase in (
                    "不减肥", "不减脂", "不用减肥", "不用减脂",
                    "not losing weight", "not on a diet",
                )
            )
        )
    ):
        data = request.model_dump()
        data["dietary_constraints"] = [
            item for item in data.get("dietary_constraints") or []
            if not any(
                marker in str(item).lower()
                for marker in _TRANSIENT_DIET_MARKERS
            )
        ]
        data["scenes"] = [
            item for item in data.get("scenes") or []
            if not any(
                marker in str(item).lower()
                for marker in _TRANSIENT_DIET_MARKERS
            )
        ]
        # 当前句是在回答“旧阶段目标是否仍适用”，不是新的召回方向。
        # 合并前彻底丢弃分类器从纠正话术生成的 query；合并后的 request
        # 带有“；补充：”标记，此时保留从 pending 恢复的原始 query，仅去掉
        # 可能残留的独立阶段目标 token。
        if "；补充：" not in str(data.get("original_text") or ""):
            data["query"] = ""
        else:
            data["query"] = " ".join(
                part
                for part in str(data.get("query") or "").split()
                if not any(
                    marker == part.lower()
                    for marker in _TRANSIENT_DIET_MARKERS
                )
            )
        # “不要参考这个标准”描述的是记忆策略，不是食材/忌口。分类器或
        # 规则解析出的这类代词目标不得进入真实菜谱的排除过滤。
        for field in ("exclude", "exclude_cuisines"):
            data[field] = [
                item
                for item in data.get(field) or []
                if not re.search(
                    r"(?:参考|按照|考虑).*(?:标准|目标|要求)|"
                    r"(?:这个|该)(?:标准|目标|要求)|"
                    r"(?:that|this)\s+(?:standard|goal|requirement)",
                    str(item),
                    flags=re.IGNORECASE,
                )
            ]
        data["context_note"] = {}
        return SearchRequest(**data)
    # 分类器的 no_constraints 只能辅助路由，不能单独授权安全事实。
    # 完整陈述由公共 helper 确定性校验；裸“没有”仍必须依赖
    # party_constraints 上下文补全被省略的宾语。
    has_explicit_no_constraints = _is_explicit_no_constraints_reply(question)
    if not (
        has_explicit_no_constraints
        or _is_contextual_no_constraints_reply(question, dimension, intent)
    ):
        return request
    data = request.model_dump()
    data["constraints_confirmed"] = True
    data["exclude"] = [
        item for item in data.get("exclude") or []
        if not re.search(r"过敏|忌口|宗教饮食|饮食要求|饮食限制|要求|限制", str(item))
    ]
    data["exclude_cuisines"] = [
        item for item in data.get("exclude_cuisines") or []
        if not re.search(r"过敏|忌口|宗教饮食|饮食要求|饮食限制|要求|限制", str(item))
    ]
    if not re.search(
        r"(?:我是|吃|需要|只吃|要)(?:纯素|全素|素食|清真|halal|vegetarian|vegan)",
        question,
        flags=re.IGNORECASE,
    ):
        data["dietary_constraints"] = []
    positive = [
        *(data.get("dishes") or []),
        *(data.get("ingredients") or []),
        *(data.get("cuisines") or []),
        *(data.get("flavors") or []),
        *(data.get("methods") or []),
        *(data.get("scenes") or []),
        *(data.get("meals") or []),
    ]
    data["query"] = " ".join(dict.fromkeys(str(item).strip() for item in positive if str(item).strip()))
    return SearchRequest(**data)


def _sanitize_has_constraints_followup(
    request: SearchRequest,
    question: str,
    dimension: str | None,
) -> SearchRequest:
    """裸“有”只表示存在限制，丢弃分类模型猜测的具体限制。"""
    if not _is_contextual_has_constraints_reply(question, dimension):
        return request
    data = request.model_dump()
    data["constraints_confirmed"] = False
    data["dietary_constraints"] = []
    data["exclude"] = []
    data["exclude_cuisines"] = []
    data["query"] = ""
    return SearchRequest(**data)


def _filter_result_language(search_result: dict, lang: str) -> dict:
    """英文结果禁止混入中文菜名；其余脏字段由渲染层继续按语言清理。"""
    if lang != "en":
        return search_result
    out = dict(search_result)
    kept = []
    for result in search_result.get("results") or []:
        name = str((result.get("metadata") or {}).get("name") or "")
        if re.search(r"[A-Za-z]", name) and not re.search(r"[一-鿿]", name):
            kept.append(result)
    out["results"] = kept
    return out


@dataclass
class FastPathOutcome:
    """确定性前段的结论。

    kind:
      - "search"   命中确定性检索，附 search_query/search_result，调用方按通道渲染；
      - "detail"   明确做法请求命中一条已核验候选，调用方直接读取真实详情；
      - "menu_plan" 显式多菜多汤菜单，分组真实检索并完成硬校验；
      - "clarify"  推荐条件不足，本轮只自然追问一个问题，不调用菜谱检索；
      - "web_search" 命中公共联网搜索，附 web_search_result；
      - 明确菜名在本地无核验结果时返回 grounded 空搜索，不生成网络菜谱；
      - "device_knowledge" 设备产品、型号和用法知识，禁止查询用户设备实例；
      - "ambiguous" 状态不足或分类器解析失败，本轮只追问一个关键问题；
      - "greeting" 问候，调用方按通道生成自然动态回复；
      - "agent"    兼容名称：交给上层确定性 Device Handler 或单次无工具
                   QA/smalltalk；搜索失败必须 fail closed，不能补造菜谱。
    intent 始终带上，供边缘做细分判断（如 recipe_execute 的日志）。
    """

    kind: str
    lang: str
    intent: IntentResult
    search_query: Optional[str] = None
    search_result: Optional[dict] = None
    search_request: Optional[SearchRequest] = None
    web_search_result: Optional[WebSearchResult] = None
    safety_terms: Optional[list[str]] = None
    clarification_message: Optional[str] = None
    clarification_dimension: Optional[str] = None
    bridge_message: Optional[str] = None
    direct_message: Optional[str] = None
    decision: Optional[RouteDecision] = None

    def __post_init__(self) -> None:
        """保证每个出口都有统一决策，并发出一条可聚合的结构化日志。"""
        if self.decision is None:
            fallback_action, fallback_reason, risk = infer_classifier_action(
                self.intent.category.value,
                self.intent.action,
                None,
            )
            action = outcome_action(self.kind, fallback_action)
            reason = self.intent.reason_code or fallback_reason
            if self.kind == "clarify":
                reason = f"{reason}_NEEDS_CLARIFICATION"
            self.decision = RouteDecision(
                category=self.intent.category.value,
                action=action,
                source=(
                    self.intent.source
                    if self.intent.source in {
                        "exact_rule", "state_rule", "classifier", "parse_error", "ambiguous",
                    }
                    else "classifier"
                ),
                reason_code=reason,
                risk=risk,
                slots=dict(self.intent.slots or {}),
                context_ref=self.intent.context_ref,
                needs_clarification=(
                    self.intent.needs_clarification or self.kind == "clarify"
                ),
                classifier_called=self.intent.source in {"classifier", "parse_error"},
            )
        logger.info(
            "route_decision trace_id=%s source=%s reason_code=%s category=%s "
            "action=%s risk=%s outcome=%s needs_clarification=%s "
            "classifier_called=%s context_ref=%s",
            self.decision.trace_id,
            self.decision.source,
            self.decision.reason_code,
            self.decision.category,
            self.decision.action,
            self.decision.risk,
            self.kind,
            self.decision.needs_clarification,
            self.decision.classifier_called,
            self.decision.context_ref or {},
        )
        record_route_decision(self.decision, outcome=self.kind)


def _attach_decision_id(search_result: dict) -> dict:
    """给同一轮检索的进程内状态和持久快照一个共同审计标识。"""
    if isinstance(search_result, dict) and not search_result.get("_decision_id"):
        search_result["_decision_id"] = uuid.uuid4().hex
    return search_result


def _routing_clarification_message(reason_code: str, lang: str) -> str:
    """拒判时只追问一个关键问题，不把模糊输入交给自由工具链。"""
    if reason_code == "ORDINAL_WITHOUT_CANDIDATES":
        return (
            "Which list does that number refer to? Search for recipes first, then choose one of the returned options."
            if lang == "en" else
            "这个序号指的是哪一组候选？请先搜索菜谱，我拿到真实候选后你再选。"
        )
    if reason_code == "STOP_WITHOUT_ACTIVE_TASK":
        return (
            "I do not have an active cooking task to stop. Which task or device do you mean?"
            if lang == "en" else
            "我这里没有可停止的活动烹饪任务。你指的是哪个任务或设备？"
        )
    if reason_code == "PROGRESS_WITHOUT_ACTIVE_TASK":
        return (
            "I do not have an active cooking task to check. Which task do you mean?"
            if lang == "en" else
            "我这里没有可查询进度的活动烹饪任务。你指的是哪个任务？"
        )
    if reason_code == "AFFIRMATIVE_WITHOUT_PENDING_ACTION":
        return affirmative_without_pending_message(lang)
    if reason_code == "CONTINUE_WITHOUT_CONTEXT":
        return (
            "There is no unfinished step in this conversation yet. What would you like to continue with?"
            if lang == "en" else
            "这个会话里还没有可继续的步骤。你想从哪件事开始？"
        )
    return (
        "I could not safely determine the requested action. What would you like me to do next?"
        if lang == "en" else
        "我没能安全确定你要执行的动作。你希望我下一步做什么？"
    )


def _is_clarification_reply(
    intent: "IntentResult",
    question: str,
    dimension: str | None,
) -> bool:
    """语义信号优先判断是否为澄清回复；未判断时回退正则。"""
    # 信号明确：分类器已判断
    if intent.is_constraint_reply is True:
        return True
    if intent.is_finalize_request is True:
        return False
    if intent.is_topic_change is True:
        return False
    if intent.is_preference_affirm is True:
        return True
    if intent.is_preference_negate is True:
        return True
    if intent.is_no_constraints is True:
        return True
    # 信号缺失：回退正则
    return _looks_like_clarification_reply(question, dimension)


async def route_fast_path(
    question: str,
    on_search_start: Optional[Callable[..., Awaitable[None]]] = None,
    preferences: Optional[dict[str, list[str]]] = None,
    food_memory: Optional[dict] = None,
    recent_turns: Optional[list] = None,
    memory_context_loader: Optional[Callable[[str], Awaitable[dict]]] = None,
    pending_search_request: Optional[dict] = None,
    pending_clarification_dimension: Optional[str] = None,
    pending_asked_dimensions: Optional[list[str]] = None,
    routing_context: RoutingContext | None = None,
    recommendation_deep_agent_enabled: bool = False,
    recommendation_agent_context: Optional[dict] = None,
    intent_override: IntentResult | None = None,
) -> FastPathOutcome:
    """意图分类 → 语言判定 → 确定性检索 / 问候 的共享前段。

    行为与原 chat_stream / qqbot_chat 的内联逻辑一致：
      - recipe_search 且 related 且有 keywords → 进程内优先检索（失败回退子进程）；
        仅在检索 success 时算命中，否则返回兼容 kind=agent；上层必须按搜索失败
        fail closed，不能用模型补造菜谱。
      - 纯 greeting → kind=greeting，由调用方使用无工具 LLM 动态回复。
      - 其余 → kind=agent。
    """
    lang = detect_lang(question)
    routing_context = routing_context or RoutingContext()

    allergy_declared = bool(re.search(
        r"过敏|\ballergic\s+to\b|\ballerg(?:y|ies)\b",
        question,
        flags=re.IGNORECASE,
    ))
    explicit_allergens = extract_explicit_excludes(question) if allergy_declared else []

    # 纯问候无需先调用意图分类模型；带其它内容的句子不会命中这里，仍由完整路由判断。
    if is_greeting_message(question):
        decision = RouteDecision(
            category=IntentCategory.greeting.value,
            action="casual",
            source="exact_rule",
            reason_code="PURE_GREETING",
            context_ref=routing_context.context_ref(),
        )
        return FastPathOutcome(
            kind="greeting",
            lang=lang,
            intent=IntentResult(
                related=True,
                category=IntentCategory.greeting,
                keywords=[],
                action=decision.action,
                source=decision.source,
                reason_code=decision.reason_code,
                context_ref=decision.context_ref,
            ),
            decision=decision,
        )

    # 产品型号、选购和用法属于知识问答，不得查询用户账号下的设备实例。
    if is_device_product_question(question):
        decision = RouteDecision(
            category=IntentCategory.cooking_qa.value,
            action="cooking_qa",
            source="exact_rule",
            reason_code="DEVICE_PRODUCT_KNOWLEDGE",
            context_ref=routing_context.context_ref(),
        )
        return FastPathOutcome(
            kind="device_knowledge",
            lang=lang,
            intent=IntentResult(
                related=True,
                category=IntentCategory.cooking_qa,
                keywords=["device knowledge" if lang == "en" else "设备知识"],
                action=decision.action,
                source=decision.source,
                reason_code=decision.reason_code,
                context_ref=decision.context_ref,
            ),
            direct_message=device_product_answer(question, lang),
            decision=decision,
        )

    # 过敏原是业务安全门禁，不能依赖意图模型先分类正确。只读取稳定偏好，
    # 在任何意图分类或菜谱检索前阻断用户明确要求制作已声明过敏原的请求。
    preloaded_preferences: dict = {}
    long_term_memory_status = str(
        (food_memory or {}).get("long_term_memory_status") or ""
    )
    temporal_dietary_constraints = list(
        (food_memory or {}).get("temporal_dietary_constraints") or []
    )
    if preferences is None and memory_context_loader is not None:
        preload_scope = "route" if routing_context.has_routing_signal() else "preferences"
        preloaded_preferences = await memory_context_loader(preload_scope)
        long_term_memory_status = str(
            preloaded_preferences.get("long_term_memory_status") or ""
        )
        preferences = preloaded_preferences.get("preferences") or {}
        temporal_dietary_constraints = list(
            preloaded_preferences.get("temporal_dietary_constraints") or []
        )
        if preload_scope == "route":
            routing_context = routing_context.model_copy(update={
                "recent_user_turns": list(
                    preloaded_preferences.get("recent_user_turns") or []
                )[-4:],
                "stable_constraints": {
                    str(key): list(values)
                    for key, values in (preferences or {}).items()
                    if isinstance(values, list) and values
                },
            })
    remembered_allergens = list((preferences or {}).get("allergens") or [])
    conflicts = requested_allergen_conflicts(
        question,
        [*remembered_allergens, *explicit_allergens],
    )
    if conflicts:
        decision = RouteDecision(
            category=IntentCategory.unknown.value,
            action="safety_block",
            source="exact_rule",
            reason_code="REMEMBERED_ALLERGEN_CONFLICT",
            risk="high",
            slots={"allergen_count": len(conflicts)},
            context_ref=routing_context.context_ref(),
        )
        return FastPathOutcome(
            kind="safety_block",
            lang=lang,
            intent=IntentResult(
                related=True,
                category=IntentCategory.unknown,
                keywords=[],
                action=decision.action,
                source=decision.source,
                reason_code=decision.reason_code,
                slots=decision.slots,
                context_ref=decision.context_ref,
            ),
            safety_terms=conflicts,
            decision=decision,
        )

    policy_decision = None
    if intent_override is not None:
        # Active Planner 只能覆盖低风险粗意图；上面的过敏和产品知识门禁仍先执行。
        # 高风险/状态精确动作在进入 Planner 前已交给确定性 handler。
        intent = intent_override
    else:
        policy_decision = decide_exact_route(question, routing_context)
    if intent_override is None and policy_decision is not None:
        intent = IntentResult(
            related=policy_decision.category != IntentCategory.off_topic.value,
            category=IntentCategory.from_str(policy_decision.category),
            keywords=[],
            action=policy_decision.action,
            source=policy_decision.source,
            reason_code=policy_decision.reason_code,
            slots=policy_decision.slots,
            context_ref=policy_decision.context_ref,
            needs_clarification=policy_decision.needs_clarification,
        )
    elif intent_override is None and routing_context.has_routing_signal():
        intent = await classify_intent(question, routing_context)
    elif intent_override is None:
        # 兼容现有测试桩和无状态调用者；新签名仍允许只传当前一句。
        intent = await classify_intent(question)

    explicitly_abandons_pending = _abandons_pending_request(question)
    repeats_pending_request = _repeats_complete_pending_request(
        question,
        pending_search_request,
    )
    continues_party_menu_by_quantity = _continues_pending_party_menu_by_quantity(
        question,
        pending_search_request,
        pending_clarification_dimension,
    )
    finalizes_request = bool(
        not explicitly_abandons_pending
        and (
            repeats_pending_request
            or (
                intent.is_topic_change is not True
                and (
                    intent.is_finalize_request is True
                    or is_finalize_recommendation_request(question)
                )
            )
        )
    )

    if (
        (
            (policy_decision and policy_decision.action == "ambiguous")
            or intent.source == "parse_error"
            or (intent.needs_clarification and intent.action in {"unknown", "ambiguous"})
        )
        # 已有推荐任务时，“你安排/按默认来”的字面证据足以承接旧任务。
        # 即使这一轮分类解析失败，也不应把 pending 丢掉或转成泛化拒判。
        and not (
            pending_search_request
            and (finalizes_request or continues_party_menu_by_quantity)
        )
    ):
        reason_code = (
            policy_decision.reason_code
            if policy_decision is not None
            else intent.reason_code
        )
        return FastPathOutcome(
            kind="ambiguous",
            lang=lang,
            intent=intent,
            direct_message=_routing_clarification_message(reason_code, lang),
            decision=policy_decision,
        )

    # “四菜一汤”是产品结构，不依赖轻量模型是否正确分类。确定性解析命中时
    # 直接提升为 recipe_recommend，再由独立 menu_plan 路由处理。
    menu_probe = SearchRequest.from_intent_raw(
        intent.raw,
        original_text=question,
        keywords=intent.keywords,
    )
    if menu_probe.is_menu_plan and (
        not intent.related or intent.category not in {
            IntentCategory.recipe_search, IntentCategory.recipe_recommend,
        }
    ):
        intent = IntentResult(
            related=True,
            category=IntentCategory.recipe_recommend,
            keywords=intent.keywords,
            action="new_search",
            source=intent.source,
            reason_code=f"{intent.reason_code}_MENU_PLAN_OVERRIDE",
            slots=intent.slots,
            context_ref=intent.context_ref,
            search_request=menu_probe,
            raw=intent.raw,
        )

    # 用户往往只回“清淡一点”或“冰箱有鸡蛋”。即使轻量模型把这种孤立短句
    # 分到 cooking_qa / unknown，只要本 thread 正在等推荐条件，就把它视为
    # 对上一问的补充；真正的问答、新话题和取消不会被吞掉。
    # finalize 是“继续当前任务并把软选择交给系统”，不是放弃任务。
    # 只有明确取消/重来/话题切换才能阻止 pending request 合并。
    abandons_pending = bool(
        pending_search_request and (
            explicitly_abandons_pending
            or (
                intent.is_topic_change is True
                and not repeats_pending_request
                and not continues_party_menu_by_quantity
            )
        )
    )
    if (
        pending_search_request
        and pending_clarification_dimension in {
            "remembered_dietary_constraint",
            "remembered_preference_conflict",
        }
        and not _looks_like_clarification_reply(
            question,
            pending_clarification_dimension,
        )
    ):
        # 用户没有回答“是否仍按旧目标/是否临时例外”，而是
        # 直接给了新方向；新事实优先，不再把旧冲突请求合进来。
        abandons_pending = True
    if (
        pending_search_request
        and (finalizes_request or continues_party_menu_by_quantity)
        and not abandons_pending
        and intent.category != IntentCategory.recipe_recommend
    ):
        # 分类器可能把“你安排”单独分到 off_topic/unknown。
        # 这里只提升为当前推荐的承接轮，后续仍会合并旧人数、
        # 数量和已确认约束，并重新经过安全维度门禁。
        followup_request = SearchRequest.from_intent_raw(
            intent.raw,
            original_text=question,
            keywords=intent.keywords,
        )
        intent = intent.model_copy(update={
            "related": True,
            "category": IntentCategory.recipe_recommend,
            "action": "refine_search",
            "source": "state_rule",
            "reason_code": (
                "PENDING_MENU_QUANTITY_REVISION"
                if continues_party_menu_by_quantity
                else "PENDING_FINALIZE_REQUEST"
            ),
            "context_ref": routing_context.context_ref(),
            "search_request": followup_request,
        })
    if (
        pending_search_request
        and not abandons_pending
        and intent.category in {
            IntentCategory.recipe_search,
            IntentCategory.cooking_qa,
            IntentCategory.off_topic,
            IntentCategory.unknown,
        }
        and _is_clarification_reply(intent, question, pending_clarification_dimension)
    ):
        followup_request = SearchRequest.from_intent_raw(
            intent.raw,
            original_text=question,
            keywords=intent.keywords,
        )
        intent = IntentResult(
            related=True,
            category=IntentCategory.recipe_recommend,
            keywords=intent.keywords,
            action="refine_search",
            source="state_rule",
            reason_code="PENDING_CLARIFICATION_REPLY",
            slots=intent.slots,
            context_ref=routing_context.context_ref(),
            search_request=followup_request,
            raw=intent.raw,
        )

    # 意图明确后才决定记忆读取范围。精确菜名/做法只需要账号稳定偏好；
    # 泛推荐和澄清承接才加载短期轮次、摘要和饮食事件。
    if memory_context_loader is not None and intent.category in {
        IntentCategory.recipe_search,
        IntentCategory.recipe_recommend,
        IntentCategory.recipe_execute,
        IntentCategory.cooking_qa,
    }:
        scope = "preferences"
        if intent.category in {
            IntentCategory.recipe_search,
            IntentCategory.recipe_recommend,
        }:
            request_probe = intent.search_request or SearchRequest.from_intent_raw(
                intent.raw,
                original_text=question,
                keywords=intent.keywords,
            )
            if (
                intent.category == IntentCategory.recipe_recommend
                or request_probe.task_operation != "exact_search"
                or pending_search_request
            ):
                scope = "full"
        loaded_context = (
            preloaded_preferences
            if scope == "preferences" and preloaded_preferences
            else await memory_context_loader(scope)
        )
        loaded_memory_status = str(
            loaded_context.get("long_term_memory_status") or ""
        )
        if loaded_memory_status:
            long_term_memory_status = loaded_memory_status
        if preferences is None:
            preferences = loaded_context.get("preferences") or {}
        temporal_dietary_constraints = list(
            loaded_context.get("temporal_dietary_constraints")
            or temporal_dietary_constraints
        )
        if food_memory is None and scope == "full":
            food_memory = loaded_context
        if recent_turns is None and scope == "full":
            recent_turns = list(loaded_context.get("recent_turns") or [])

    if intent.related and intent.category in {
        IntentCategory.recipe_search, IntentCategory.recipe_recommend,
    }:
        mode = "recommend" if intent.category == IntentCategory.recipe_recommend else "search"
        request = intent.search_request or SearchRequest.from_intent_raw(
            intent.raw,
            original_text=question,
            keywords=intent.keywords,
        )
        request = _sanitize_no_constraints_followup(
            request,
            question,
            pending_clarification_dimension,
            intent=intent,
        )
        request = _sanitize_has_constraints_followup(
            request,
            question,
            pending_clarification_dimension,
        )
        resolved_preference_conflict = False
        if (
            pending_search_request
            and pending_clarification_dimension == "remembered_preference_conflict"
            and not abandons_pending
        ):
            normalized_reply = _normalized_short_reply(question)
            try:
                previous_conflict_request = SearchRequest(
                    **pending_search_request
                )
            except (TypeError, ValueError):
                previous_conflict_request = None
            if (
                previous_conflict_request is not None
                and _PREFERENCE_CONFLICT_AFFIRM_RE.fullmatch(normalized_reply)
            ):
                note = previous_conflict_request.context_note or {}
                term = str(note.get("term") or "").strip()
                data = previous_conflict_request.model_dump()
                data["suppressed_inherited"] = list(dict.fromkeys([
                    *(data.get("suppressed_inherited") or []),
                    term,
                ]))
                data["context_note"] = {
                    "kind": "temporary_preference_override",
                    "term": term,
                }
                request = SearchRequest(**data)
                resolved_preference_conflict = True
            elif (
                previous_conflict_request is not None
                and _PREFERENCE_CONFLICT_NEGATIVE_RE.fullmatch(normalized_reply)
            ):
                term = str(
                    (previous_conflict_request.context_note or {}).get("term")
                    or ""
                ).strip()
                remembered_bucket = str(
                    (previous_conflict_request.context_note or {}).get("bucket")
                    or "dislikes"
                )
                message = (
                    (
                        f"Okay, I'll keep your earlier preference for {term}. Tell me the new direction you want."
                        if remembered_bucket == "likes"
                        else f"Okay, I'll keep avoiding {term}. Tell me the new direction you want."
                    )
                    if lang == "en" else (
                        f"好，那就继续按你之前喜欢{term}来。你换个想吃的方向告诉我就行。"
                        if remembered_bucket == "likes"
                        else f"好，那就继续避开{term}。你换个想吃的方向告诉我就行。"
                    )
                )
                return FastPathOutcome(
                    kind="ambiguous",
                    lang=lang,
                    intent=intent,
                    direct_message=message,
                    clarification_dimension="remembered_preference_conflict_declined",
                )
        current_conflicts = remembered_preference_conflicts(
            request,
            preferences,
        )
        if current_conflicts and not resolved_preference_conflict:
            conflict = current_conflicts[0]
            data = request.model_dump()
            data["context_note"] = {
                "kind": "remembered_preference_conflict",
                "term": str(conflict.get("value") or ""),
                "dimension": str(conflict.get("dimension") or ""),
                "bucket": str(conflict.get("bucket") or ""),
            }
            request = SearchRequest(**data)
            message = await generate_recommendation_clarification(
                original_question=question,
                dimension="remembered_preference_conflict",
                lang=lang,
                search_request=request,
                recent_turns=recent_turns,
            )
            return FastPathOutcome(
                kind="clarify",
                lang=lang,
                intent=intent,
                search_request=request,
                clarification_message=message,
                clarification_dimension="remembered_preference_conflict",
            )
        request = request.merge_preferences(preferences)
        remembered_diet_dimension = None
        remembered_diet = _active_temporal_dietary_constraint(
            temporal_dietary_constraints
        )
        if (
            mode == "recommend"
            and not pending_search_request
            and remembered_diet
            and not _explicit_transient_diet_term(question)
        ):
            # 阶段性目标不会作为永久硬偏好静默生效。只有本轮确实准备拿它
            # 参与选菜时才挂到 pending request，并先向用户确认一次。
            data = request.model_dump()
            term = str(remembered_diet["value"])
            data["scenes"] = list(dict.fromkeys([
                *(data.get("scenes") or []),
                term,
            ]))
            data["context_note"] = {
                "kind": "remembered_dietary_constraint",
                "term": term,
                "expires_at": str(remembered_diet.get("expires_at") or ""),
            }
            request = SearchRequest(**data)
            remembered_diet_dimension = "remembered_dietary_constraint"
        # “晚饭吃什么”是一个新的选择任务，不能自动继承上一轮牛肉或更早的
        # 湖南口味；只有条件搜索中的明确省略指代才承接近期主题。
        if mode == "search":
            request = request.merge_recent_context(recent_turns)
        if request.context_note:
            print(
                "  🧩 近期上下文承接: "
                f"kind={request.context_note.get('kind') or '-'} "
                f"term_chars={len(str(request.context_note.get('term') or ''))}"
            )
        # 语义信号：偏好确认/拒绝（decision point 5）
        if (
            intent.is_preference_affirm is True
            and pending_clarification_dimension == "remembered_preference_conflict"
        ):
            resolved_preference_conflict = True
        if (
            intent.is_preference_negate is True
            and pending_clarification_dimension == "remembered_preference_conflict"
        ):
            abandons_pending = True

        # 语义信号：话题切换（decision point 4）。同一完整请求的确定性
        # 文本证据优先于分类器的误判，这种情况仍要 merge 已确认的安全事实。
        if (
            intent.is_topic_change is True
            and not repeats_pending_request
            and not continues_party_menu_by_quantity
        ):
            pass  # 不 merge，保留新请求
        elif (
            pending_search_request
            and not abandons_pending
            and not resolved_preference_conflict
        ):
            try:
                previous_request = SearchRequest(**pending_search_request)
                request = previous_request.merge_clarification(request)
                request = _sanitize_no_constraints_followup(
                    request,
                    question,
                    pending_clarification_dimension,
                    intent=intent,
                )
                if continues_party_menu_by_quantity:
                    # 旧菜单数量既然已被明确否定，就不再把它当成
                    # 显式上限。保留旧任务的人数和已确认安全条件，
                    # 后续统一按当前人数生成默认菜+汤结构。
                    data = request.model_dump()
                    data["explicit_result_limit"] = False
                    data["menu_dish_count"] = 0
                    data["menu_soup_count"] = 0
                    request = SearchRequest(**data)
            except (TypeError, ValueError):
                # 过期版本或损坏的进程内状态不能阻断当前请求。
                pass

        if (
            long_term_memory_status in {"unavailable", "invalid"}
            and not request.constraints_confirmed
        ):
            has_constraints = _is_contextual_has_constraints_reply(
                question,
                pending_clarification_dimension,
            )
            dimension = (
                "party_constraints_detail"
                if has_constraints
                else "party_constraints"
            )
            message = (
                (
                    "I can't read your saved allergy and dietary-safety preferences right now. "
                    "Please list the allergies, foods to avoid, or religious dietary requirements for this meal; "
                    "if there are none, say 'none'. I'll search the verified recipe library after you confirm."
                    if not has_constraints
                    else "Please list the specific allergies, foods to avoid, or religious dietary requirements for this meal."
                )
                if lang == "en"
                else (
                    "我现在读不到你已保存的过敏和饮食安全偏好。请明确告诉我这次有哪些过敏、忌口或宗教饮食要求；没有也请直接说“没有”。确认后我再查真实菜谱库。"
                    if not has_constraints
                    else "请具体告诉我这次有哪些过敏、忌口或宗教饮食要求；事实确认后我再查真实菜谱库。"
                )
            )
            return FastPathOutcome(
                kind="clarify",
                lang=lang,
                intent=intent,
                search_request=request,
                clarification_message=message,
                clarification_dimension=dimension,
            )

        request = request.merge_food_memory(food_memory)
        request = request.with_task_operation(mode)
        intent = intent.model_copy(update={
            "business_operation": BusinessOperation(request.task_operation),
            "search_request": request,
        })
        if request.memory_note:
            print(
                "  🧠 账号记忆关联: "
                f"kind={request.memory_note.get('kind')} "
                f"term_chars={len(str(request.memory_note.get('term') or ''))}"
            )
        missing_dimension = remembered_diet_dimension or (
            request.search_clarification_dimension()
            if mode == "search"
            else request.recommendation_clarification_dimension()
        )
        # 已经确认“有限制”并进入细节收集后，“你安排/随便”不能让状态
        # 倒退成泛化的 party_constraints。只有本轮明确表示无限制时才算安全事实已完成。
        if (
            pending_clarification_dimension == "party_constraints_detail"
            and not request.constraints_confirmed
            and not _is_explicit_no_constraints_reply(question, intent)
        ):
            missing_dimension = "party_constraints_detail"
        # “有”只确认存在限制，不能当成安全事实已经采集完整。依赖上一问的
        # 维度把它推进到二级澄清，下一轮再收集具体过敏原或饮食要求。
        if (
            missing_dimension == "party_constraints"
            and _is_contextual_has_constraints_reply(
                question,
                pending_clarification_dimension,
            )
        ):
            missing_dimension = "party_constraints_detail"
        # 不再依赖“同一问题问两次”才退出；只有用户明确把软选择权
        # 交给系统时才跳过非安全维度。过敏/忌口/宗教饮食仍不可越过。
        # 直接进入检索；只有过敏/忌口/宗教饮食的安全确认不可越过。
        if (
            finalizes_request
            and missing_dimension
            and missing_dimension not in {
                "party_constraints",
                "party_constraints_detail",
            }
        ):
            missing_dimension = None
        if missing_dimension:
            message = await generate_recommendation_clarification(
                original_question=question,
                dimension=missing_dimension,
                lang=lang,
                search_request=request,
                recent_turns=recent_turns,
            )
            return FastPathOutcome(
                kind="clarify",
                lang=lang,
                intent=intent,
                search_request=request,
                clarification_message=message,
                clarification_dimension=missing_dimension,
            )
        request = request.freeze_canonical_question(lang=lang)
        intent = intent.model_copy(update={"search_request": request})
        # 只有明确人数时才把普通推荐扩成整桌菜单；“牛肉 + 辣”这类条件已经
        # 足够直接返回 3 道真实菜谱，不能再机械追问人数或强塞一道汤。
        if mode == "recommend" and request.party_size:
            request = request.with_recommendation_menu_defaults()
        if request.is_menu_plan:
            menu_query = request.retrieval_query(lang=lang)
            bridge, search_result = await _run_with_search_bridge(
                build_menu_plan(
                    request,
                    lang=lang,
                    on_search_start=None,
                ),
                question=question,
                query=menu_query,
                mode="recommend",
                lang=lang,
                search_request=request,
                recent_turns=recent_turns,
                profile_context=food_memory,
                on_search_start=on_search_start,
            )
            notice = hard_constraint_notice(request, lang)
            if notice:
                search_result["_constraint_notice"] = notice
            query_summary = " | ".join(
                str(item.get("query") or "")
                for item in (search_result.get("_menu_plan") or {}).get("queries") or []
                if item.get("query")
            )
            narrative = await generate_recommendation_narrative(
                original_question=question,
                search_query=query_summary or request.retrieval_query(lang=lang),
                search_result=search_result,
                lang=lang,
                search_request=request,
                recent_turns=recent_turns,
                deep_agent_enabled=recommendation_deep_agent_enabled,
                agent_context=recommendation_agent_context,
            )
            search_result["_recommendation"] = narrative.to_dict()
            search_result["_recommendation_source"] = "grounded_v2"
            search_result["_interaction_mode"] = "recommend"
            _attach_decision_id(search_result)
            return FastPathOutcome(
                kind="menu_plan",
                lang=lang,
                intent=intent,
                search_query=query_summary or request.retrieval_query(lang=lang),
                search_result=search_result,
                search_request=request,
                bridge_message=bridge,
            )
        raw_query = request.retrieval_query(lang=lang)
        if not raw_query:
            return FastPathOutcome(kind="agent", lang=lang, intent=intent, search_request=request)
        plan = plan_search_query(question, raw_query, lang=lang)
        # 条件搜索默认展示最多 5 条达标结果；推荐数量由菜单规格或用户显式要求决定。
        # 候选池适当放大，给硬过滤、多样性选择和“换一批”留出空间。
        display_limit = (
            min(10, max(1, request.result_limit))
            if request.explicit_result_limit or mode == "recommend"
            else 5
        )
        # 用户显式要求 N 道菜时，放大候选池以给过滤和去重留足空间，
        # 避免"10 道菜"只返回 3 道。非显式场景维持默认比例。
        if request.explicit_result_limit and display_limit >= 6:
            top_k = min(30, max(12, display_limit * 3))
        else:
            top_k = min(20, max(9, display_limit * 2))
        if request.task_operation == "exact_search":
            # 精确搜索直接查本地库；开始阶段只触发通道进度状态，不额外发送
            # 一条“正在查询”的对话消息，避免用户收到两个回复者的口吻。
            bridge = ""
            await _notify_search_start(on_search_start, plan.search_query, lang, bridge)
            search_result = await _run_search_subprocess(
                plan.search_query,
                top_k=top_k,
                lang=lang,
            )
        else:
            bridge, search_result = await _run_with_search_bridge(
                _run_search_subprocess(plan.search_query, top_k=top_k, lang=lang),
                question=question,
                query=plan.search_query,
                mode=mode,
                lang=lang,
                search_request=request,
                recent_turns=recent_turns,
                profile_context=food_memory,
                on_search_start=on_search_start,
            )
        if search_result and search_result.get("success"):
            search_result = filter_hard_constraint_violations(search_result, request)
            notice = hard_constraint_notice(request, lang)
            if notice:
                search_result["_constraint_notice"] = notice
            search_result = _filter_result_language(search_result, lang)
            search_result = apply_search_plan(search_result, plan, limit=top_k)
            candidate_pool = filter_requested_result_type(
                list(search_result.get("results") or []),
                request,
            )
            candidate_pool = prioritize_exact_matches(
                candidate_pool, request.dishes
            )
            if request.task_operation != "exact_search":
                candidate_pool = prioritize_soft_preferences(candidate_pool, request)
                candidate_pool = prioritize_ingredient_coverage(candidate_pool, request)
            threshold = _relevance_threshold()
            if mode == "search":
                effective_threshold = (
                    min(threshold, _broad_relevance_threshold())
                    if request.has_broad_flavor_direction()
                    else threshold
                )
                before_threshold = len(candidate_pool)
                candidate_pool = [
                    item for item in candidate_pool
                    if _score_at_least(item, effective_threshold)
                ]
                search_result["_relevance_threshold"] = effective_threshold
                search_result["_relevance_filtered_count"] = before_threshold - len(candidate_pool)
            if request.task_operation == "exact_search" and not candidate_pool:
                # 只尝试有限、可审计的菜名同义词；重试结果仍经过同一套事实
                # 校验、硬约束和相关性门槛，全部失败后才进入网络参考。
                for alias_query in exact_recipe_synonym_queries(request, limit=2):
                    alias_result = await _run_search_subprocess(
                        alias_query,
                        top_k=top_k,
                        lang=lang,
                    )
                    if not alias_result or not alias_result.get("success"):
                        continue
                    alias_request = request.model_copy(update={
                        "query": alias_query,
                        "canonical_question": alias_query,
                        "dishes": [alias_query],
                    })
                    alias_result = filter_hard_constraint_violations(
                        alias_result, alias_request,
                    )
                    notice = hard_constraint_notice(request, lang)
                    if notice:
                        alias_result["_constraint_notice"] = notice
                    alias_result = _filter_result_language(alias_result, lang)
                    alias_plan = plan_search_query(
                        alias_query, alias_query, lang=lang,
                    )
                    alias_result = apply_search_plan(
                        alias_result, alias_plan, limit=top_k,
                    )
                    alias_pool = prioritize_exact_matches(
                        list(alias_result.get("results") or []),
                        [alias_query],
                    )
                    alias_pool = [
                        item for item in alias_pool
                        if _score_at_least(item, threshold)
                    ]
                    if alias_pool:
                        alias_result["_exact_alias_query"] = alias_query
                        alias_result["_exact_alias_of"] = (
                            request.canonical_question or request.query
                        )
                        alias_result["_relevance_threshold"] = threshold
                        search_result = alias_result
                        candidate_pool = alias_pool
                        break
            if request.task_operation == "exact_search" and not candidate_pool:
                # 菜谱事实只能来自真实 Milvus 结果。精确菜名没有通过硬约束和
                # 相关性门槛时明确返回空结果，不用联网模型补造一份做法。
                search_result["results"] = []
                search_result["_candidate_pool"] = []
                search_result["_display_limit"] = 0
                search_result["_search_request"] = request.public_dict()
                search_result["_interaction_mode"] = "exact_not_found"
                search_result["_exact_not_found"] = True
                search_result["_grounded_output"] = True
                _attach_decision_id(search_result)
                exact_display_query = normalize_exact_search_label(
                    (request.dishes[0] if request.dishes else "")
                    or "、".join(request.required_ingredients[:3])
                    or request.canonical_question
                    or request.query
                    or plan.search_query
                )
                return FastPathOutcome(
                    kind="search",
                    lang=lang,
                    intent=intent,
                    search_query=exact_display_query,
                    search_request=request,
                    search_result=search_result,
                )
            if request.task_operation == "exact_search" and candidate_pool:
                # 进入这里前已经通过明确菜名/必需食材硬校验。精确菜谱搜索只
                # 绑定排名最高的真实候选并返回详情，不再退化成相似菜列表。
                search_result["results"] = [candidate_pool[0]]
                search_result["_candidate_pool"] = candidate_pool
                search_result["_display_limit"] = 1
                search_result["_search_request"] = request.public_dict()
                search_result["_interaction_mode"] = "detail"
                search_result["_grounded_output"] = True
                _attach_decision_id(search_result)
                return FastPathOutcome(
                    kind="detail",
                    lang=lang,
                    intent=intent,
                    search_query=plan.search_query,
                    search_result=search_result,
                    search_request=request,
                    bridge_message=bridge,
                )
            search_result["results"] = select_diverse_results(candidate_pool, limit=display_limit)
            search_result["_candidate_pool"] = candidate_pool
            search_result["_display_limit"] = display_limit
            search_result["_search_request"] = request.public_dict()
            fulfilled = len(search_result["results"])
            search_result["_validation"] = {
                "requested": display_limit if request.explicit_result_limit else None,
                "fulfilled": fulfilled,
                "count_complete": (fulfilled == display_limit) if request.explicit_result_limit else True,
                "unique": len({
                    str((item.get("metadata") or {}).get("recipe_id") or item.get("id") or "")
                    for item in search_result["results"]
                }) == fulfilled,
                # hard filter 已在选择前执行；被移除的候选不会进入最终 results。
                "hard_constraints_passed": True,
            }
            narrative = await generate_recommendation_narrative(
                original_question=question,
                search_query=plan.search_query,
                search_result=search_result,
                lang=lang,
                search_request=request,
                recent_turns=recent_turns,
                deep_agent_enabled=recommendation_deep_agent_enabled,
                agent_context=recommendation_agent_context,
            )
            search_result["_recommendation"] = narrative.to_dict()
            search_result["_recommendation_source"] = "grounded_v2"
            search_result["_interaction_mode"] = mode
            search_result["_grounded_output"] = True
            _attach_decision_id(search_result)
            return FastPathOutcome(
                kind="search",
                lang=lang,
                intent=intent,
                search_query=plan.search_query,
                search_result=search_result,
                search_request=request,
                bridge_message=bridge,
            )

    # 普通/未知问题中的实时事实走公共联网服务。菜谱与设备意图在此之前保持各自
    # 的确定性数据源，避免网页结果混入真实菜谱库或设备状态。
    if intent.category in {IntentCategory.off_topic, IntentCategory.unknown} and should_search_web(question):
        web_result = await search_web(question, lang=lang)
        return FastPathOutcome(
            kind="web_search",
            lang=lang,
            intent=intent,
            web_search_result=web_result,
        )

    if intent.category == IntentCategory.greeting:
        return FastPathOutcome(kind="greeting", lang=lang, intent=intent)

    return FastPathOutcome(kind="agent", lang=lang, intent=intent)
