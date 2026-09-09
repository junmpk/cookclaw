"""受控 Deep Agent：只读取真实会话事实，参与低风险推荐与候选选择。"""
from __future__ import annotations

import asyncio
import json
import time
from contextvars import ContextVar
from typing import Any

from deepagents import FilesystemPermission, create_deep_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.messages import HumanMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.observability.trace import (
    current_trace,
    observe_model_call,
    record_tool_call,
    trace_stage,
)


class CandidateChoiceDecision(BaseModel):
    """Agent 的唯一可执行输出；外层仍须校验 ID 是否属于当前候选。"""

    selected_recipe_id: str = Field(default="", max_length=128)
    reason: str = Field(default="", max_length=240)


class RecommendationReplyDecision(BaseModel):
    """推荐表达草稿；返回后仍由事实校验器逐句过滤。"""

    opening: str = Field(default="", max_length=500)
    strategy: str = Field(default="", max_length=500)
    recipe_reasons: dict[str, str] = Field(default_factory=dict)
    closing: str = Field(default="", max_length=300)


_candidate_context: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "cookclaw_candidate_context",
    default=(),
)
_conversation_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "cookclaw_conversation_context",
    default=None,
)
_allowed_tools: ContextVar[frozenset[str]] = ContextVar(
    "cookclaw_allowed_agent_tools",
    default=frozenset({"get_current_recipe_candidates"}),
)
_tool_reads: ContextVar[dict[str, int] | None] = ContextVar(
    "cookclaw_agent_tool_reads",
    default=None,
)
_agent_operation: ContextVar[str] = ContextVar(
    "cookclaw_controlled_agent_operation",
    default="unknown",
)
_candidate_agent = None
_candidate_agent_model = None
_recommendation_agent = None
_recommendation_agent_model = None

_TASK_REQUEST_FIELDS = {
    "original_text",
    "query",
    "task_operation",
    "canonical_question",
    "dishes",
    "ingredients",
    "cuisines",
    "flavors",
    "methods",
    "scenes",
    "meals",
    "dietary_constraints",
    "exclude",
    "exclude_cuisines",
    "avoid",
    "required_ingredients",
    "detail_requested",
    "party_size",
    "result_limit",
    "explicit_result_limit",
    "menu_dish_count",
    "menu_soup_count",
    "scoped_preferences",
    "suppressed_inherited",
    "applied_memory_constraints",
    "image_context",
    "_response_scenario",
    "_menu_plan",
}


def _as_list(value: Any, *, limit: int = 12) -> list[str]:
    if isinstance(value, dict):
        value = list(value.values())
    elif isinstance(value, str):
        value = [value]
    return [
        str(item).strip()[:120]
        for item in (value or [])
        if str(item).strip()
    ][:limit]


def compact_candidate_facts(recipes: list[dict], *, limit: int = 10) -> list[dict]:
    """只给 Agent 真实候选中的必要事实，避免把完整详情和内部字段塞进 Prompt。"""
    facts = []
    for position, recipe in enumerate((recipes or [])[:limit], 1):
        recipe_id = str(
            recipe.get("cookId") or recipe.get("recipe_id") or recipe.get("id") or ""
        ).strip()
        name = str(recipe.get("name") or "").strip()
        if not recipe_id or not name:
            continue
        detail = (
            recipe.get("recipe_detail")
            if isinstance(recipe.get("recipe_detail"), dict)
            else {}
        )
        steps = detail.get("steps") or detail.get("recipeStepVoList") or []
        fact = {
            "position": position,
            "recipe_id": recipe_id,
            "name": name[:120],
            "tags": _as_list(recipe.get("tags")),
            "ingredients": _as_list(recipe.get("ingredients"), limit=8),
            "seasonings": _as_list(recipe.get("seasonings"), limit=8),
            "description": str(recipe.get("description") or "").strip()[:180],
            "step_count": len(steps) if isinstance(steps, list) else None,
        }
        if recipe.get("step_count") is not None:
            fact["step_count"] = recipe.get("step_count")
        if recipe.get("estimated_time"):
            fact["estimated_time"] = str(recipe.get("estimated_time"))[:80]
        if recipe.get("servings"):
            fact["servings"] = str(recipe.get("servings"))[:80]
        for key in (
            "difficulty",
            "menu_role",
            "menu_role_label",
            "menu_requirement",
            "ingredient_match",
        ):
            if recipe.get(key) not in (None, "", [], {}):
                fact[key] = recipe.get(key)
        facts.append(fact)
    return facts


def _compact_search_request(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return {}

    def compact_value(item: Any, *, depth: int = 0) -> Any:
        if isinstance(item, str):
            return item[:500]
        if isinstance(item, (int, float, bool)) or item is None:
            return item
        if isinstance(item, list):
            return [
                compact_value(entry, depth=depth + 1)
                for entry in item[:12]
            ]
        if isinstance(item, dict) and depth < 3:
            return {
                str(child_key)[:40]: compact_value(child_value, depth=depth + 1)
                for child_key, child_value in list(item.items())[:16]
                if child_value not in (None, "", [], {})
            }
        return str(item)[:500]

    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _TASK_REQUEST_FIELDS or item in (None, "", [], {}):
            continue
        compact[key] = compact_value(item)
    return compact


def compact_conversation_state(
    state: Any,
    *,
    recent_conversation: str = "",
) -> dict[str, Any]:
    """生成 Agent 可读但不可执行的任务状态，不暴露设备 ID 或 pending payload。"""
    if hasattr(state, "to_dict"):
        state = state.to_dict()
    raw = dict(state) if isinstance(state, dict) else {}
    if raw.get("schema_version") == "controlled_context_v1":
        raw["recent_conversation"] = (
            str(recent_conversation or "")[-4000:]
            if recent_conversation
            else str(raw.get("recent_conversation") or "")[-4000:]
        )
        return raw
    user_profile = (
        raw.get("user_profile")
        if isinstance(raw.get("user_profile"), dict)
        else {}
    )
    selected_raw = raw.get("selected_recipe")
    selected = compact_candidate_facts(
        [selected_raw] if isinstance(selected_raw, dict) else [],
        limit=1,
    )
    focus = raw.get("focus") if isinstance(raw.get("focus"), dict) else {}
    pending = (
        raw.get("pending_action")
        if isinstance(raw.get("pending_action"), dict)
        else {}
    )
    pending_device = (
        raw.get("pending_device_start")
        if isinstance(raw.get("pending_device_start"), dict)
        else {}
    )
    active = (
        raw.get("active_cooking")
        if isinstance(raw.get("active_cooking"), dict)
        else {}
    )
    return {
        "schema_version": "controlled_context_v1",
        "current_task": str(raw.get("current_task") or "") or None,
        "active_search_request": _compact_search_request(
            raw.get("active_search_request")
        ),
        "selected_recipe": selected[0] if selected else None,
        "excluded_recipe_ids": [
            str(item)[:128]
            for item in (raw.get("excluded_recipe_ids") or [])[:20]
            if str(item).strip()
        ],
        "focus": (
            {
                "name": str(focus.get("name") or focus.get("topic") or "")[:120],
                "verified_recipe": bool(focus.get("verified_recipe")),
            }
            if focus else None
        ),
        "pending_action": (
            {"kind": str(pending.get("kind") or "")[:80]}
            if pending.get("kind") else None
        ),
        "pending_device_start": (
            {
                "waiting_confirmation": True,
                "recipe_id": str(
                    pending_device.get("cookId")
                    or pending_device.get("recipe_id")
                    or ""
                )[:128],
                "recipe_name": str(pending_device.get("name") or "")[:120],
                "device_choice_required": bool(pending_device.get("devices")),
                "device_already_selected": bool(pending_device.get("device_id")),
            }
            if pending_device else None
        ),
        "active_cooking": (
            {
                "active": True,
                "recipe_id": str(
                    active.get("cookId") or active.get("recipe_id") or ""
                )[:128],
                "recipe_name": str(active.get("name") or "")[:120],
            }
            if active else None
        ),
        "language": (
            str(raw.get("language"))
            if raw.get("language") in {"zh", "en"}
            else None
        ),
        "user_profile": {
            "preferred_name": (
                str(user_profile.get("preferred_name") or "").strip()[:24]
                or None
            ),
        },
        "recent_conversation": str(recent_conversation or "")[-4000:],
    }


@tool
def get_current_recipe_candidates() -> dict:
    """读取当前会话已由真实检索核验的菜谱候选；不得用于设备执行。"""
    return {"candidates": [dict(item) for item in _candidate_context.get()]}


@tool
def get_current_recipe_search_results() -> dict:
    """读取本轮已完成检索并经过外层过滤的结构化菜谱结果。"""
    return {
        "ok": True,
        "data": {"recipes": [dict(item) for item in _candidate_context.get()]},
    }


@tool
def get_current_conversation_state() -> dict:
    """读取裁剪后的当前任务状态；设备信息只包含流程阶段，不含设备标识。"""
    return {"ok": True, "data": dict(_conversation_context.get() or {})}


@tool
def get_verified_recipe_detail(recipe_id: str) -> dict:
    """按真实候选 ID 读取当前可核验详情；不能访问候选之外的菜谱。"""
    target = str(recipe_id or "").strip()
    recipe = next(
        (
            dict(item)
            for item in _candidate_context.get()
            if str(item.get("recipe_id") or "") == target
        ),
        None,
    )
    if recipe is None:
        return {
            "ok": False,
            "error": {
                "code": "RECIPE_NOT_IN_CURRENT_RESULTS",
                "message": "Recipe is not part of the current verified results.",
            },
        }
    return {"ok": True, "data": {"recipe": recipe}}


@wrap_tool_call
async def _allow_candidate_tool_only(request, handler):
    """阻断所有非白名单工具，并限制同一决策内的重复读取。"""
    tool_call = request.tool_call or {}
    tool_name = str(tool_call.get("name") or "")
    structured_tools = {
        CandidateChoiceDecision.__name__,
        RecommendationReplyDecision.__name__,
    }
    if tool_name in _allowed_tools.get():
        started = time.monotonic()
        reads = dict(_tool_reads.get() or {})
        max_reads = 3 if tool_name == "get_verified_recipe_detail" else 1
        if reads.get(tool_name, 0) >= max_reads:
            record_tool_call(
                tool_name,
                duration_ms=(time.monotonic() - started) * 1000,
                success=False,
                error_type="TOOL_READ_LIMIT",
            )
            return ToolMessage(
                content=f"Tool {tool_name} exceeded its read limit for this decision.",
                tool_call_id=str(tool_call.get("id") or ""),
                status="error",
            )
        reads[tool_name] = reads.get(tool_name, 0) + 1
        _tool_reads.set(reads)
        try:
            result = await handler(request)
        except Exception as exc:
            record_tool_call(
                tool_name,
                duration_ms=(time.monotonic() - started) * 1000,
                success=False,
                error_type=type(exc).__name__,
            )
            raise
        success = getattr(result, "status", None) != "error"
        record_tool_call(
            tool_name,
            duration_ms=(time.monotonic() - started) * 1000,
            success=success,
            error_type=None if success else "TOOL_RESULT_ERROR",
        )
        return result
    if tool_name in structured_tools:
        return await handler(request)
    record_tool_call(
        tool_name or "unknown",
        duration_ms=0,
        success=False,
        error_type="TOOL_NOT_ALLOWED",
    )
    return ToolMessage(
        content=f"Tool {tool_name or 'unknown'} is not allowed in this agent.",
        tool_call_id=str(tool_call.get("id") or ""),
        status="error",
    )


@wrap_model_call
async def _observe_agent_model_call(request, handler):
    operation = _agent_operation.get()
    model = getattr(request, "model", None)
    model_name = str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or ""
    )
    return await observe_model_call(
        lambda: handler(request),
        stage=f"deep_agent_model.{operation}",
        model=model_name,
        reply_generation=True,
    )


_CANDIDATE_AGENT_PROMPT = """
You are CookClaw's constrained candidate-choice agent.

Your only job is to choose at most one recipe from the current verified candidate list.
You must call get_current_conversation_state exactly once and
get_current_recipe_candidates exactly once before deciding.
Use only the tool results and the user's current message.
Return selected_recipe_id exactly as supplied by the tool. If the facts are insufficient,
return an empty selected_recipe_id and briefly explain what is missing in reason.

Never invent a recipe, ingredient, nutrition claim, cooking time, device state, or user history.
Never start, stop, confirm, or prepare a device action.
Do not use filesystem, todo, execute, or subagent tools.
Do not produce the final user-facing reply; the deterministic outer layer will validate the ID
and generate a grounded response.
""".strip()


_RECOMMENDATION_AGENT_PROMPT = """
You are CookClaw's constrained recommendation-response agent.

Your job is to organize already verified recipe search results into a short, natural reply.
You must call get_current_conversation_state exactly once and
get_current_recipe_search_results exactly once. Those results already contain all facts needed
for this reply; do not request full recipe details.

Use the active request, current user message, recent user context, and verified recipe facts.
If user_profile contains a confirmed preferred_name, you may use it naturally at most once;
do not force it into every reply and never treat it as a recipe or device fact.
When active_search_request.image_context is present, naturally acknowledge the visual scene,
recognized dish, and recognized ingredients. They are visual hypotheses, not user-confirmed
facts: say that you recognized or can see them, and do not claim the user explicitly listed
them. Never call the source a fridge unless the current caption explicitly says so.
When active_search_request.applied_memory_constraints is present, say naturally and explicitly
that this round used the user's earlier stated preference, for example that a disliked cuisine
was avoided. Mention only entries present in that field and never call them a database record
or internal memory.
When active_search_request._response_scenario is party_menu, use _menu_plan plus each recipe's
menu_role and menu_requirement to explain the table naturally. A false complete flag means the
menu is incomplete: state the missing dish or soup count and never describe it as complete.
When active_search_request._response_scenario is ingredient_explore, use ingredient_match:
all-match recipes answer the combined-ingredient request first; partial matches are broader
ideas around only the matched ingredient and must not be described as using every requested one.
The current user message overrides older context. Recipe-specific IDs, names, ingredients,
tags, time, serving, and step-count facts must come from the read-only recipe tools. Visual
ingredients may be mentioned only as recognition hypotheses from image_context. Never infer
health effects, hidden ingredients, difficulty, compatibility, device state, or successful execution.

Do not start, stop, confirm, or prepare a device action. A pending device state is context only.
Do not use filesystem, todo, execute, or subagent tools.
Return only the structured recommendation fields. The outer layer will validate every sentence
and render the final channel response.
""".strip()


def _deny_all_filesystem() -> list[FilesystemPermission]:
    return [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ]


def _get_candidate_agent(model):
    """按模型实例惰性构造，避免未启用时初始化旧式 Agent 工具栈。"""
    global _candidate_agent, _candidate_agent_model
    if _candidate_agent is not None and _candidate_agent_model is model:
        return _candidate_agent
    _candidate_agent = create_deep_agent(
        model=model,
        tools=[
            get_current_conversation_state,
            get_current_recipe_candidates,
        ],
        system_prompt=_CANDIDATE_AGENT_PROMPT,
        middleware=[_observe_agent_model_call, _allow_candidate_tool_only],
        response_format=CandidateChoiceDecision,
        permissions=_deny_all_filesystem(),
        name="cookclaw_candidate_choice",
    )
    _candidate_agent_model = model
    return _candidate_agent


def _get_recommendation_agent(model):
    """构造只读推荐表达 Agent；不注册检索执行或设备工具。"""
    global _recommendation_agent, _recommendation_agent_model
    if _recommendation_agent is not None and _recommendation_agent_model is model:
        return _recommendation_agent
    _recommendation_agent = create_deep_agent(
        model=model,
        tools=[
            get_current_conversation_state,
            get_current_recipe_search_results,
        ],
        system_prompt=_RECOMMENDATION_AGENT_PROMPT,
        middleware=[_observe_agent_model_call, _allow_candidate_tool_only],
        response_format=RecommendationReplyDecision,
        permissions=_deny_all_filesystem(),
        name="cookclaw_recommendation_reply",
    )
    _recommendation_agent_model = model
    return _recommendation_agent


async def choose_candidate_with_deep_agent(
    *,
    model,
    question: str,
    conversation_context: str,
    recipes: list[dict],
    lang: str,
    conversation_state: Any = None,
    timeout_seconds: float = 15,
) -> str | None:
    """返回经过候选集合校验的 recipe_id；任何异常或越界输出都拒绝采用。"""
    candidate_facts = compact_candidate_facts(recipes)
    allowed_ids = {
        str(item.get("recipe_id") or "")
        for item in candidate_facts
        if str(item.get("recipe_id") or "")
    }
    if not allowed_ids:
        return None

    candidate_token = _candidate_context.set(tuple(candidate_facts))
    conversation_token = _conversation_context.set(
        compact_conversation_state(
            conversation_state,
            recent_conversation=conversation_context,
        )
    )
    allowed_token = _allowed_tools.set(frozenset({
        "get_current_conversation_state",
        "get_current_recipe_candidates",
    }))
    reads_token = _tool_reads.set({})
    operation_token = _agent_operation.set("candidate_choice")
    trace = current_trace()
    if trace is not None:
        trace.deep_agent_enabled = True
        trace.deep_agent_used = True
    payload = {
        "language": lang if lang in {"zh", "en"} else "zh",
        "current_user_message": str(question or "")[:500],
        "instruction": (
            "Select one verified candidate only when the supplied facts support the choice."
        ),
    }
    try:
        with trace_stage("deep_agent.candidate_choice"):
            result = await asyncio.wait_for(
                _get_candidate_agent(model).ainvoke(
                    {"messages": [HumanMessage(content=json.dumps(payload, ensure_ascii=False))]},
                    config={"recursion_limit": 8},
                ),
                timeout=max(1.0, float(timeout_seconds)),
            )
    finally:
        _agent_operation.reset(operation_token)
        _tool_reads.reset(reads_token)
        _allowed_tools.reset(allowed_token)
        _conversation_context.reset(conversation_token)
        _candidate_context.reset(candidate_token)

    decision = result.get("structured_response") if isinstance(result, dict) else None
    if isinstance(decision, CandidateChoiceDecision):
        selected_id = decision.selected_recipe_id.strip()
    elif isinstance(decision, dict):
        selected_id = str(decision.get("selected_recipe_id") or "").strip()
    else:
        return None
    return selected_id if selected_id in allowed_ids else None


async def compose_recommendation_with_deep_agent(
    *,
    model,
    current_question: str,
    recent_user_context: list[str],
    current_request: dict[str, Any],
    recipes: list[dict],
    lang: str,
    conversation_state: Any = None,
    timeout_seconds: float = 15,
) -> dict[str, Any] | None:
    """生成结构化推荐草稿；最终是否可展示由调用方事实校验决定。"""
    candidate_facts = compact_candidate_facts(recipes)
    allowed_ids = {
        str(item.get("recipe_id") or "")
        for item in candidate_facts
        if str(item.get("recipe_id") or "")
    }
    if not allowed_ids:
        return None

    state = compact_conversation_state(
        conversation_state,
        recent_conversation="\n".join(
            str(item)[:500] for item in (recent_user_context or [])[-4:]
        ),
    )
    state["active_search_request"] = _compact_search_request(current_request)
    candidate_token = _candidate_context.set(tuple(candidate_facts))
    conversation_token = _conversation_context.set(state)
    allowed_token = _allowed_tools.set(frozenset({
        "get_current_conversation_state",
        "get_current_recipe_search_results",
    }))
    reads_token = _tool_reads.set({})
    operation_token = _agent_operation.set("recommendation_reply")
    trace = current_trace()
    if trace is not None:
        trace.deep_agent_enabled = True
        trace.deep_agent_used = True
    payload = {
        "language": lang if lang in {"zh", "en"} else "zh",
        "current_user_message": str(current_question or "")[:500],
        "instruction": (
            "Write a natural reply using only the read-only facts. Never mention search, "
            "retrieval, candidates, databases, models, or internal state. Recipe reasons "
            "may use only that recipe's name, ingredients, tags, difficulty, estimated time, "
            "step count, serving count, menu role, menu requirement, or ingredient_match; "
            "do not infer taste, nutrition, health effects, difficulty, popularity, or suitability. "
            "Write those facts as natural reasons; never say '标签为', '标签显示', '字段显示', "
            "'tagged as', or expose field names. "
            "For party_menu, acknowledge the current guest count, menu shape, applied exclusions, "
            "required slots, and whether the menu is complete, then explain the roles naturally. "
            "For ingredient_explore, answer as a follow-up: explain all-match recipes first and "
            "describe partial matches only as broader alternatives around the matched ingredient. "
            "Use Chinese recipe names inside 【】 and ask only one easy closing question."
        ),
    }
    try:
        with trace_stage("deep_agent.recommendation_reply"):
            result = await asyncio.wait_for(
                _get_recommendation_agent(model).ainvoke(
                    {"messages": [HumanMessage(content=json.dumps(payload, ensure_ascii=False))]},
                    config={"recursion_limit": 10},
                ),
                timeout=max(1.0, float(timeout_seconds)),
            )
    finally:
        _agent_operation.reset(operation_token)
        _tool_reads.reset(reads_token)
        _allowed_tools.reset(allowed_token)
        _conversation_context.reset(conversation_token)
        _candidate_context.reset(candidate_token)

    decision = result.get("structured_response") if isinstance(result, dict) else None
    if isinstance(decision, RecommendationReplyDecision):
        data = decision.model_dump()
    elif isinstance(decision, dict):
        try:
            data = RecommendationReplyDecision(**decision).model_dump()
        except (TypeError, ValueError):
            return None
    else:
        return None
    data["recipe_reasons"] = {
        str(recipe_id): str(reason)
        for recipe_id, reason in (data.get("recipe_reasons") or {}).items()
        if str(recipe_id) in allowed_ids
    }
    return data
