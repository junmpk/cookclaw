"""图片轮次的统一领域处理器。

通道层只负责把媒体转成视觉模型可读输入以及最终发送。视觉识别、派生事实、
Bounded Planner 门禁、真实菜谱检索、候选状态和 ResponseEnvelope 都在这里收口。
图片轮次永远不注册设备动作。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.agent import fast_path as agent_fast_path
from app.conversation.runtime_memory import (
    PlannerMemoryView,
    RuntimeMemorySnapshot,
    ShortTermRuntimeSnapshot,
)
from app.conversation.service import supports_session_memory
from app.core.config import settings
from app.observability.trace import (
    current_trace,
    current_trace_id,
    mark_fallback,
    record_domain_result,
)
from app.orchestrator import image as image_orchestrator
from app.orchestrator import recommendation_response
from app.orchestrator.planning.active_planner import evaluate_active_plan
from app.orchestrator.planning.plan_executor import BoundedPlanExecutor
from app.orchestrator.planning.planner_rollout import (
    planner_active_actions,
    planner_rollout_decision,
)
from app.orchestrator.routing_models import RoutingContext
from app.orchestrator.search_request import (
    filter_hard_constraint_violations,
    hard_constraint_notice,
)
from app.orchestrator.search_selection import (
    filter_requested_result_type,
    prioritize_exact_matches,
    prioritize_ingredient_coverage,
    prioritize_soft_preferences,
    select_diverse_results,
)
from app.orchestrator.turn.context_loader import build_turn_context
from app.orchestrator.turn.response_renderer import response_to_envelope
from app.orchestrator.turn.runtime_models import (
    DerivedTurnContext,
    ResponseEnvelope,
    TurnRequest,
)
from app.ports.dialogue_state import (
    clear_pending_action,
    get_pending_action,
    remember_candidates,
    routing_state_snapshot,
    set_pending_action,
)

logger = logging.getLogger(__name__)

ImageInputConverter = Callable[[str], dict[str, Any] | None]
ImageLanguageResolver = Callable[[str, str], Awaitable[str]]
ImageMemoryLoader = Callable[
    [str, str],
    Awaitable[tuple[list[Any], dict[str, Any]]],
]


@dataclass(frozen=True, slots=True)
class ImageHandlerResult:
    """图片领域结果；内部执行事实与公共通道协议明确分开。"""

    envelope: ResponseEnvelope
    derived_context: DerivedTurnContext | None = None
    image_context: dict[str, Any] | None = None
    result_code: str = "IMAGE_RESULT"
    success: bool = False
    result_count: int | None = None
    duration_ms: float = 0.0

    def with_envelope(self, envelope: ResponseEnvelope) -> ImageHandlerResult:
        """绑定应用服务唯一出口完成后的 Envelope。"""
        return ImageHandlerResult(
            envelope=envelope,
            derived_context=self.derived_context,
            image_context=self.image_context,
            result_code=self.result_code,
            success=self.success,
            result_count=self.result_count,
            duration_ms=self.duration_ms,
        )

    def channel_value(self) -> str | dict[str, Any]:
        """保持 QQ/微信/WhatsApp 当前发送契约，便于逐通道灰度。"""
        payload = self.envelope.public_payload()
        if (
            self.envelope.response_type in {"recipe_search", "menu_plan"}
            and isinstance(payload, dict)
        ):
            result: dict[str, Any] = {
                "kind": "recipe_search",
                "response": payload,
            }
            if self.image_context is not None:
                result["image_context"] = dict(self.image_context)
            return result
        return str(self.envelope.message or "")


def record_image_turn_observability(
    result: ImageHandlerResult,
    *,
    request: TurnRequest,
) -> None:
    """在应用服务 finalize 后记录图片领域结果和统一编排出口。"""
    envelope = result.envelope
    record_domain_result(
        "image_flow",
        duration_ms=result.duration_ms,
        success=result.success,
        result_code=result.result_code,
        result_count=result.result_count,
        operation="recognize_and_recommend",
        state_patch_count=len(envelope.state_patches),
        risk="low",
    )
    trace = current_trace()
    if trace is None:
        return
    derived_ref = (
        result.derived_context.context_ref()
        if result.derived_context is not None
        else {}
    )
    trace.add_event(
        "turn_orchestrator",
        mode="image",
        channel=request.channel,
        handled_by=envelope.handled_by,
        input_chars=len(request.utterance),
        derived_source=derived_ref.get("source"),
        media_count=derived_ref.get("media_count"),
        ingredient_count=derived_ref.get("ingredient_count"),
        dish_count=derived_ref.get("dish_count"),
        tool_result_count=len(envelope.tool_results),
        state_patch_count=len(envelope.state_patches),
        tool_failure_count=sum(
            1 for tool_result in envelope.tool_results if not tool_result.success
        ),
    )


def _clean_values(values: Any, *, limit: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        value = (
            str(raw.get("name") or raw.get("ingredient") or "").strip()
            if isinstance(raw, dict)
            else str(raw or "").strip()
        )
        value = value[:80]
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
        if len(cleaned) >= limit:
            break
    return cleaned


def build_vision_derived_context(
    recognized: dict[str, Any],
    *,
    image_count: int,
) -> DerivedTurnContext:
    """把视觉输出裁剪为 Planner 标准派生事实，不携带媒体地址。"""
    dishes = _clean_values(recognized.get("dish_names"), limit=4)
    dish_name = str(recognized.get("dish_name") or "").strip()[:80]
    if dish_name and dish_name not in dishes:
        dishes.insert(0, dish_name)
        dishes = dishes[:4]
    return DerivedTurnContext(
        scene_type=str(recognized.get("scene_type") or "unknown").strip()[:40],
        media_count=max(1, min(6, int(image_count or 1))),
        ingredients=_clean_values(recognized.get("ingredients"), limit=12),
        dishes=dishes,
        # 视觉模型自报 confidence 不能作为安全事实；默认始终可由用户纠正。
        recognition_uncertain=True,
    )


async def _planner_memory_context(
    thread_id: str,
    utterance: str,
    *,
    runtime_memory: RuntimeMemorySnapshot | None = None,
) -> PlannerMemoryView | dict[str, Any] | None:
    if not supports_session_memory(thread_id):
        return None
    try:
        from app.conversation.service import get_conversation_service

        return await get_conversation_service().planner_memory_view(
            thread_id,
            current_message=utterance,
            runtime_memory=runtime_memory,
        )
    except Exception as exc:
        logger.warning(
            "图片 Planner 上下文读取失败，本轮只用视觉与任务态: error_type=%s",
            type(exc).__name__,
        )
        return {"long_term_memory_status": "unavailable"}


def _routing_context(request: TurnRequest) -> RoutingContext:
    return RoutingContext(
        **routing_state_snapshot(request.thread_id),
        channel=request.channel,
        message_type=request.message_type,
    )


def _text_envelope(
    message: str,
    *,
    trace_id: str | None,
    handled_by: str = "image_handler",
) -> ResponseEnvelope:
    return ResponseEnvelope(
        response_type="text",
        message=message,
        handled_by=handled_by,
        trace_id=trace_id,
    )


async def handle_image_turn(
    media_urls: list[str],
    *,
    request: TurnRequest,
    image_input_converter: ImageInputConverter,
    language_resolver: ImageLanguageResolver,
    memory_loader: ImageMemoryLoader,
    short_term_snapshot: ShortTermRuntimeSnapshot | None = None,
) -> ImageHandlerResult:
    """执行一轮图片识别与 grounded 推荐，并返回统一 Envelope。

    图片先经视觉工具生成带来源的 ``DerivedTurnContext``。命中 active 灰度时，
    与文本使用同一个 TurnPlanner、Validator 和单步 Executor；图片只注册
    ``recipe.search`` / ``recipe.recommend``，其它计划均不执行并回退到同一个
    deterministic grounded handler，绝不开放设备能力。
    """
    started = time.monotonic()
    if request.message_type != "image":
        raise ValueError("image handler requires message_type=image")
    thread_id = request.thread_id
    trace_id = request.trace_id or current_trace_id()
    image_context: dict[str, Any] | None = None
    derived_context: DerivedTurnContext | None = None

    def finish(
        envelope: ResponseEnvelope,
        *,
        result_code: str,
        success: bool,
        result_count: int | None = None,
    ) -> ImageHandlerResult:
        return ImageHandlerResult(
            envelope=envelope,
            derived_context=derived_context,
            image_context=image_context,
            result_code=result_code,
            success=success,
            result_count=result_count,
            duration_ms=(time.monotonic() - started) * 1_000,
        )

    lang = await language_resolver(thread_id, request.utterance)
    images = [
        image
        for media in media_urls or []
        if (image := image_input_converter(media)) is not None
    ]
    if not images:
        message = (
            "I could not access those images. Please send them again, or type the ingredients."
            if lang == "en" else
            "没拿到这组图片（可能已过期），麻烦重新发一次，或直接打字告诉我有哪些食材。"
        )
        return finish(
            _text_envelope(message, trace_id=trace_id),
            result_code="IMAGE_INPUT_UNAVAILABLE",
            success=False,
        )

    recognized = await image_orchestrator.recognize_ingredients_by_images(
        images,
        lang=lang,
    )
    if recognized.get("error"):
        logger.warning(
            "[图片识别] 视觉服务未完成识别：error_type=%s image_count=%s",
            str(recognized.get("error") or "unknown")[:64],
            len(images),
        )
        message = (
            "I received the image, but visual recognition took a brief kitchen break 😅\n\n"
            "Please send it once more. If you are in a hurry, type what you have—such as "
            '"eggs, tomatoes, and tofu"—and I’ll search the real recipe collection right away.'
            if lang == "en" else
            "图片我已经收到啦，不过视觉识别刚才开了个小差 😅\n\n"
            "麻烦重新发一次；如果赶时间，也可以直接告诉我“有鸡蛋、番茄和豆腐”，"
            "我马上按真实菜谱库给你推荐。"
        )
        return finish(
            _text_envelope(message, trace_id=trace_id),
            result_code="VISION_FAILED",
            success=False,
        )

    if not recognized.get("is_food"):
        recognized_items = (
            recognized.get("ingredients")
            or recognized.get("items")
            or []
        )
        logger.info(
            "[图片识别] 未识别到可用食材/菜品：is_food=%s scene=%s items=%s",
            bool(recognized.get("is_food")),
            str(recognized.get("scene_type") or "unknown")[:24],
            len(recognized_items) if isinstance(recognized_items, list) else 0,
        )
        message = (
            "This “ingredient” is staying out of the pan today 😄\n\n"
            "I could not identify anything cookable in the image. Try another photo of your groceries "
            "or ingredients on the counter—or simply type what you have."
            if lang == "en" else
            "这位“食材”今天暂时不下锅 😄\n\n"
            "我在图片里没认出能做菜的东西。你可以重新拍一张菜篮、购物袋或案板上的食材，"
            "也可以直接告诉我“有鸡蛋、番茄和豆腐”，我再认真给你安排。"
        )
        return finish(
            _text_envelope(message, trace_id=trace_id),
            result_code="VISION_NOT_FOOD",
            success=True,
        )

    existing_action = get_pending_action(thread_id)
    recognized, inventory_payload = image_orchestrator.merge_image_inventory(
        recognized,
        previous_action=existing_action,
        image_count=int(recognized.get("image_count") or len(images)),
        user_hint=request.utterance,
    )
    scene = str(recognized.get("scene_type") or "").strip().lower()
    if inventory_payload.get("ingredients") and scene in {"ingredients", "mixed"}:
        set_pending_action(
            thread_id,
            "image_inventory",
            payload=inventory_payload,
            lang=lang,
        )
    elif existing_action and existing_action.get("kind") in {
        "confirm_image_ingredients",
        "image_inventory",
    }:
        clear_pending_action(thread_id)

    effective_user_hint = "；".join(
        str(item).strip()
        for item in inventory_payload.get("user_hints") or []
        if str(item).strip()
    ) or request.utterance.strip()
    image_count = int(inventory_payload.get("image_count") or len(images))
    derived_context = build_vision_derived_context(
        recognized,
        image_count=image_count,
    )
    request = request.model_copy(update={"derived_context": derived_context})
    search_request = image_orchestrator.build_image_search_request(
        recognized,
        lang=lang,
        user_hint=effective_user_hint,
    )
    runtime_memory: RuntimeMemorySnapshot | None = None
    memory_context: dict[str, Any] = {}
    if supports_session_memory(thread_id):
        try:
            from app.conversation.service import get_conversation_service

            service = get_conversation_service()
            runtime_memory = await service.load_runtime_memory(
                thread_id,
                current_message=effective_user_hint,
                short_term_snapshot=short_term_snapshot,
            )
            memory_context = await service.routing_context(
                thread_id,
                scope="preferences",
                current_message=effective_user_hint,
                runtime_memory=runtime_memory,
            )
            trace = current_trace()
            if trace is not None:
                trace.add_event(
                    "runtime_memory",
                    **runtime_memory.context_ref(),
                )
        except Exception as exc:
            logger.warning(
                "图片推荐记忆水合失败，本轮按长期资料不可读处理: error_type=%s",
                type(exc).__name__,
            )
            memory_context = {"long_term_memory_status": "unavailable"}

        long_term_status = str(
            memory_context.get("long_term_memory_status") or "invalid"
        )
        if (
            long_term_status in {"unavailable", "invalid"}
            and not search_request.constraints_confirmed
        ):
            message = (
                "I can see food in the image, but I can't read your saved allergy and dietary-safety preferences right now. "
                "Please type the allergies, foods to avoid, or religious dietary requirements for this meal; if there are none, say 'none'."
                if lang == "en" else
                "图片里的食材我看到了，但现在读不到你已保存的过敏和饮食安全偏好。"
                "请先用文字明确这次有哪些过敏、忌口或宗教饮食要求；没有也请直接说“没有”。"
            )
            return finish(
                _text_envelope(message, trace_id=trace_id),
                result_code="IMAGE_MEMORY_SAFETY_UNAVAILABLE",
                success=False,
            )
        search_request = search_request.merge_preferences(
            memory_context.get("preferences") or {}
        )

    query = search_request.retrieval_query(lang=lang)
    summary = image_orchestrator._recognized_food_summary(recognized, lang=lang)
    if not query:
        message = (
            "Those ingredients are surprisingly camera-shy 😄 "
            "Try another angle or type what you have."
            if lang == "en" else
            "这些食材躲镜头的本事有点强，我暂时没看清 😄 "
            "换个角度再拍一张，或者直接打字告诉我有哪些食材吧。"
        )
        return finish(
            _text_envelope(message, trace_id=trace_id),
            result_code="VISION_QUERY_EMPTY",
            success=False,
        )

    image_context = image_orchestrator.build_image_agent_context(
        recognized,
        image_count=image_count,
        user_hint=effective_user_hint,
    )

    async def grounded_recipe_handler(_step=None, _plan=None, _context=None):
        search_result = await agent_fast_path._run_search_subprocess(
            query,
            top_k=9,
            lang=lang,
        )
        if search_result and search_result.get("success"):
            search_result = filter_hard_constraint_violations(
                search_result,
                search_request,
            )
            notice = hard_constraint_notice(search_request, lang)
            if notice:
                search_result["_constraint_notice"] = notice
            candidates = filter_requested_result_type(
                list(search_result.get("results") or []),
                search_request,
            )
            candidates = prioritize_exact_matches(candidates, search_request.dishes)
            candidates = prioritize_soft_preferences(candidates, search_request)
            candidates = prioritize_ingredient_coverage(candidates, search_request)
            search_result["results"] = select_diverse_results(candidates, limit=3)
        if not (
            search_result
            and search_result.get("success")
            and search_result.get("results")
        ):
            message = (
                (
                    f"I recognized {summary}, but did not find a solid recipe match. "
                    "Tell me which ingredient matters most and I’ll narrow it down."
                )
                if lang == "en" and summary else
                (
                    f"我从图片里识别到：{summary}，但这轮没找到足够合适的菜谱。"
                    "你告诉我最想优先用哪一种，我再缩小范围。"
                )
                if summary else
                (
                    "I found ingredient clues in the image, but no suitable recipe match. "
                    "Try a clearer image or type the ingredients."
                    if lang == "en" else
                    "图片里有食材线索，但这轮没搜到合适菜谱。"
                    "换张更清楚的图，或者直接打字告诉我食材试试。"
                )
            )
            return _text_envelope(
                message,
                trace_id=trace_id,
                handled_by="image_recipe_handler",
            )

        search_result = dict(search_result)
        search_result["_search_request"] = search_request.public_dict()
        search_result["_interaction_mode"] = "recommend"
        search_result["_grounded_output"] = True
        current_question = image_orchestrator.image_recommendation_question(
            image_context or {},
            lang=lang,
        )
        recent_turns, agent_context = await memory_loader(
            thread_id,
            current_question,
        )
        agent_context["image_context"] = image_context
        narrative = (
            await recommendation_response.generate_recommendation_narrative(
                original_question=current_question,
                search_query=query,
                search_result=search_result,
                lang=lang,
                search_request=search_request,
                recent_turns=recent_turns,
                deep_agent_enabled=settings.IMAGE_DEEP_AGENT_ENABLED,
                agent_context=agent_context,
                input_context=image_context,
            )
        )
        search_result["_recommendation"] = narrative.to_dict()
        search_result["_recommendation_source"] = "grounded_v2"
        remember_candidates(thread_id, search_result, lang=lang)
        return response_to_envelope(
            await agent_fast_path._format_search_response(
                query,
                search_result,
                lang=lang,
            ),
            handled_by="image_recipe_handler",
            trace_id=trace_id,
        )

    envelope: ResponseEnvelope | None = None
    final_result_code: str | None = None
    rollout = planner_rollout_decision(thread_id, message_type="image")
    if rollout.enabled:
        planner_context = build_turn_context(
            request.utterance,
            _routing_context(request),
            memory_context=await _planner_memory_context(
                thread_id,
                request.utterance,
                runtime_memory=runtime_memory,
            ),
            trace_id=trace_id,
            derived_context=derived_context,
        )
        decision = await evaluate_active_plan(planner_context)
        if (
            decision.execution_allowed
            and decision.plan is not None
            and decision.action in {"recipe.search", "recipe.recommend"}
        ):
            execution = await BoundedPlanExecutor(
                {
                    "recipe.search": grounded_recipe_handler,
                    "recipe.recommend": grounded_recipe_handler,
                },
                allowed_actions=planner_active_actions(),
            ).execute(decision.plan, planner_context)
            if execution.status == "executed":
                envelope = execution.envelope
                trace = current_trace()
                if trace is not None:
                    trace.add_event(
                        "planner_execution",
                        status="executed",
                        action=execution.action,
                        result_code="IMAGE_GROUNDED_RECIPE_EXECUTED",
                    )
            else:
                # 搜索 handler 一旦开始就可能已有只读外部调用；不得重新检索。
                code = execution.error_code or "IMAGE_PLANNER_EXECUTION_FAILED"
                mark_fallback(code)
                final_result_code = code
                trace = current_trace()
                if trace is not None:
                    trace.add_event(
                        "planner_execution",
                        status="failed_closed",
                        action=execution.action,
                        result_code=code,
                    )
                envelope = _text_envelope(
                    "I could not safely finish the image search. "
                    "Please send the image again or type the ingredients."
                    if lang == "en" else
                    "这次图片搜索没有安全完成，我不会重复检索。"
                    "请重新发图，或直接打字告诉我食材。",
                    trace_id=trace_id,
                    handled_by="image_planner",
                )
        else:
            # 图片输入只允许 grounded 食谱动作。设备、联网、记忆和自由对话计划
            # 均不执行；沿用同一个 deterministic handler，不重跑视觉识别。
            mark_fallback(
                decision.fallback_code
                or "IMAGE_PLANNER_ACTION_NOT_GROUNDED_RECIPE"
            )
            trace = current_trace()
            if trace is not None:
                trace.add_event(
                    "planner_execution",
                    status="rejected",
                    action=decision.action,
                    result_code=(
                        decision.fallback_code
                        or "IMAGE_PLANNER_ACTION_NOT_GROUNDED_RECIPE"
                    ),
                )

    if envelope is None:
        envelope = await grounded_recipe_handler()

    result_count = (
        len(envelope.data.get("recipes") or [])
        if envelope.response_type in {"recipe_search", "menu_plan"}
        else 0
    )
    result_code = final_result_code
    if result_code is None:
        result_code = (
            "IMAGE_RECIPE_READY"
            if envelope.response_type in {"recipe_search", "menu_plan"}
            else "IMAGE_RECIPE_NO_MATCH"
        )
    return finish(
        envelope,
        result_code=result_code,
        success=envelope.response_type in {"recipe_search", "menu_plan"},
        result_count=result_count,
    )
