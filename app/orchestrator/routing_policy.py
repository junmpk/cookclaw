"""高精度路由规则与分类器结论归一化。

这里不执行搜索或设备命令，只产出 RouteDecision。高风险动作仍必须经过既有
设备状态机的权限、状态、确认和启动验证。
"""
from __future__ import annotations

import re

from app.orchestrator.routing_models import RouteDecision, RoutingContext


_CONFIRM_EXACT = {
    "确认", "确认开始", "确认开火", "开始烹饪", "可以", "好", "好的", "行",
    "confirm", "confirm start", "confirm cooking", "start cooking", "yes", "ok", "okay",
}
_CANCEL_EXACT = {
    "取消", "算了", "不", "不要", "不用", "别", "先不",
    "cancel", "no", "never mind",
}
_STOP_EXACT = {
    "停", "停止", "停一下", "别做了", "不做了", "暂停", "取消烹饪", "停止烹饪",
    "stop", "pause", "stop cooking", "cancel cooking",
}
_PROGRESS_MARKERS = (
    "做到哪里", "做到哪", "哪一步", "进行到", "进度", "还要多久", "多久做好",
    "cooking progress", "how long", "which step", "how far",
)
_CONTINUE_EXACT = {
    "继续", "接着", "往下", "continue", "go on", "keep going",
}
_REFINE_MARKERS = (
    "换一批", "换几个", "再来一批", "再来几道", "多来几道",
    "再推荐几道", "上一批", "next batch", "more options",
)


def _normalized(text: str) -> str:
    return re.sub(r"[，,。.!！?？；;\s]+", " ", str(text or "").strip().lower()).strip()


def _looks_like_ordinal(text: str) -> bool:
    value = _normalized(text)
    return bool(re.fullmatch(
        r"(?:就 |做 |选 |选择 |用设备做 )?"
        r"(?:第 ?)?(?:10|[1-9]|[一二三四五六七八九十])(?: ?[个道号])?(?: 吧)?|"
        r"(?:the )?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
        r"(?: one)?",
        value,
        flags=re.IGNORECASE,
    ))


def decide_exact_route(
    utterance: str,
    context: RoutingContext,
) -> RouteDecision | None:
    """按固定优先级处理状态相关、无歧义的动作；未命中才允许调用分类器。"""
    value = _normalized(utterance)
    ref = context.context_ref()
    pending_action = str((context.pending_action or {}).get("kind") or "")

    if context.pending_device_start:
        if value in _CANCEL_EXACT or value in _STOP_EXACT:
            return RouteDecision(
                category="recipe_execute",
                action="cancel_pending",
                source="state_rule",
                reason_code="PENDING_DEVICE_CANCEL_EXACT",
                risk="medium",
                context_ref=ref,
            )
        if value in _CONFIRM_EXACT:
            action = "select_device" if (context.pending_device_start or {}).get("devices") else "confirm_start"
            return RouteDecision(
                category="recipe_execute",
                action=action,
                source="state_rule",
                reason_code=(
                    "PENDING_DEVICE_SELECTION_REQUIRED"
                    if action == "select_device"
                    else "PENDING_DEVICE_CONFIRM_EXACT"
                ),
                risk="high" if action == "confirm_start" else "medium",
                context_ref=ref,
                needs_clarification=action == "select_device",
            )

    if context.active_cooking and value in _STOP_EXACT:
        return RouteDecision(
            category="recipe_execute",
            action="stop",
            source="state_rule",
            reason_code="ACTIVE_TASK_STOP_EXACT",
            risk="high",
            context_ref=ref,
        )
    if context.active_cooking and any(marker in value for marker in _PROGRESS_MARKERS):
        return RouteDecision(
            category="recipe_execute",
            action="progress",
            source="state_rule",
            reason_code="ACTIVE_TASK_PROGRESS_QUERY",
            risk="low",
            context_ref=ref,
        )
    if value in _STOP_EXACT:
        return RouteDecision(
            category="unknown",
            action="ambiguous",
            source="ambiguous",
            reason_code="STOP_WITHOUT_ACTIVE_TASK",
            context_ref=ref,
            needs_clarification=True,
        )
    if any(marker in value for marker in _PROGRESS_MARKERS):
        return RouteDecision(
            category="unknown",
            action="ambiguous",
            source="ambiguous",
            reason_code="PROGRESS_WITHOUT_ACTIVE_TASK",
            context_ref=ref,
            needs_clarification=True,
        )

    if _looks_like_ordinal(value):
        if context.latest_candidates:
            return RouteDecision(
                category="recipe_execute",
                action="choose_candidate",
                source="state_rule",
                reason_code="LATEST_CANDIDATE_ORDINAL",
                risk="medium",
                context_ref=ref,
            )
        return RouteDecision(
            category="unknown",
            action="ambiguous",
            source="ambiguous",
            reason_code="ORDINAL_WITHOUT_CANDIDATES",
            context_ref=ref,
            needs_clarification=True,
        )

    if context.latest_candidates and any(marker in value for marker in _REFINE_MARKERS):
        return RouteDecision(
            category="recipe_search",
            action="refine_search",
            source="state_rule",
            reason_code="LATEST_CANDIDATES_REFINE_REQUEST",
            context_ref=ref,
        )

    if pending_action and value in _CANCEL_EXACT:
        return RouteDecision(
            category="unknown",
            action="cancel_pending",
            source="state_rule",
            reason_code="PENDING_ACTION_CANCEL_EXACT",
            risk="medium",
            context_ref=ref,
        )

    if value in _CONFIRM_EXACT and not pending_action:
        return RouteDecision(
            category="unknown",
            action="ambiguous",
            source="ambiguous",
            reason_code="AFFIRMATIVE_WITHOUT_PENDING_ACTION",
            context_ref=ref,
            needs_clarification=True,
        )
    if value in _CONTINUE_EXACT and not any((
        pending_action,
        context.pending_device_start,
        context.active_cooking,
        context.pending_clarification,
        context.latest_candidates,
        context.latest_focus,
        context.selected_recipe,
        context.current_task,
    )):
        return RouteDecision(
            category="unknown",
            action="ambiguous",
            source="ambiguous",
            reason_code="CONTINUE_WITHOUT_CONTEXT",
            context_ref=ref,
            needs_clarification=True,
        )
    return None


def infer_classifier_action(
    category: str,
    raw_action: str | None,
    context: RoutingContext | None,
) -> tuple[str, str, str]:
    """把模型 action 归一化，并给出 reason_code/risk；不信任自报置信度。"""
    allowed = {
        "new_search", "refine_search", "compare_candidates", "choose_candidate",
        "device_status", "select_device", "confirm_start", "prepare_device",
        "progress", "stop", "memory_correct", "cooking_qa", "casual", "unknown",
    }
    requested = str(raw_action or "").strip()
    if requested not in allowed:
        requested = ""
    if requested == "unknown" and category != "unknown":
        requested = ""

    default_by_category = {
        "recipe_search": "new_search",
        "recipe_recommend": "new_search",
        "recipe_execute": "prepare_device",
        "device_manage": "device_status",
        "cooking_qa": "cooking_qa",
        "greeting": "casual",
        "off_topic": "casual",
        "unknown": "unknown",
    }
    action = requested or default_by_category.get(category, "unknown")
    if action == "refine_search" and not (
        context and (context.latest_candidates or context.pending_clarification)
    ):
        action = "new_search"
    risk = "high" if action in {"confirm_start", "stop"} else (
        "medium" if action in {"choose_candidate", "select_device", "prepare_device"} else "low"
    )
    return action, f"CLASSIFIER_{category.upper()}_{action.upper()}", risk


def outcome_action(kind: str, fallback: str) -> str:
    """最终 handler 类型覆盖粗分类动作，使日志记录实际下沉结果。"""
    return {
        "safety_block": "safety_block",
        "device_knowledge": "cooking_qa",
        "detail": "show_recipe_detail",
        "menu_plan": "new_search",
        "web_search": "web_search",
        "recipe_web_reference": "web_reference",
        "greeting": "casual",
        "clarify": "clarify",
        "ambiguous": "ambiguous",
    }.get(kind, fallback)
