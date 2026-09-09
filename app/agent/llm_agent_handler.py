"""LLM Agent 集成处理器

将 LLM-Centric Agent 集成到现有的 handler 架构中。
采用渐进式迁移策略：先处理 conversation_fallback，再逐步扩展。
"""

import os
import logging
import json
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from app.orchestrator.turn.runtime_models import TurnRequest, ResponseEnvelope
from app.orchestrator.turn.response_renderer import response_to_envelope

if TYPE_CHECKING:
    from app.agent.llm_agent import LLMAgent

logger = logging.getLogger(__name__)


def _extract_structured_data(tool_calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """从工具调用中提取结构化数据

    遍历所有工具调用，识别搜索类工具，提取返回的结构化数据（如菜谱列表），
    组合成一个统一的数据结构供前端渲染。

    Args:
        tool_calls: 工具调用列表

    Returns:
        Optional[Dict]: 结构化数据，如果没有搜索类工具调用则返回 None
    """
    if not tool_calls:
        return None

    # 搜索类工具名称
    search_tools = {
        "search_recipes",
        "search_by_ingredients",
        "search_by_cuisine",
        "search_by_cooking_method",
        "search_by_taste",
        "search_by_dietary",
        "search_by_cooking_time",
        "search_by_difficulty",
        "search_by_occasion",
    }

    # 菜谱详情类工具
    detail_tools = {
        "get_recipe_details",
        "get_cooking_steps",
        "get_ingredient_list",
    }

    # 收集所有搜索结果
    all_recipes = []
    search_query = None
    recipe_details = []

    for tool_call in tool_calls:
        tool_name = tool_call.get("tool")
        tool_result = tool_call.get("result")

        if not tool_result or tool_result.status.value != "success":
            continue

        data = tool_result.data
        if not data:
            continue

        # 处理搜索类工具
        if tool_name in search_tools:
            recipes = data.get("results", [])
            if recipes:
                all_recipes.extend(recipes)
                # 记录搜索查询（取第一个）
                if search_query is None:
                    search_query = (
                        data.get("query")
                        or data.get("ingredients")
                        or data.get("cuisine")
                        or data.get("method")
                        or data.get("taste")
                        or data.get("dietary")
                        or data.get("occasion")
                        or ""
                    )

        # 处理菜谱详情类工具
        elif tool_name in detail_tools:
            recipe_details.append(data)

    # 如果有搜索结果，构建结构化数据
    if all_recipes:
        return {
            "type": "recipe_search",
            "data": {
                "query": search_query,
                "total": len(all_recipes),
                "recipes": all_recipes
            }
        }

    # 如果有菜谱详情，构建结构化数据
    if recipe_details:
        return {
            "type": "recipe_details",
            "data": {
                "details": recipe_details
            }
        }

    return None


# 功能开关：是否启用 LLM Agent
LLM_AGENT_ENABLED = os.getenv("LLM_AGENT_ENABLED", "false").lower() in {"true", "1", "yes"}

# 灰度比例：0-100，表示使用 LLM Agent 的百分比
LLM_AGENT_ROLLOUT_PERCENT = int(os.getenv("LLM_AGENT_ROLLOUT_PERCENT", "0"))


def should_use_llm_agent(thread_id: str) -> bool:
    """判断是否应该使用 LLM Agent

    Args:
        thread_id: 会话 ID

    Returns:
        bool: 是否使用 LLM Agent
    """
    if not LLM_AGENT_ENABLED:
        return False

    if LLM_AGENT_ROLLOUT_PERCENT <= 0:
        return False

    if LLM_AGENT_ROLLOUT_PERCENT >= 100:
        return True

    # 基于 thread_id 的哈希值进行灰度
    hash_value = hash(thread_id) % 100
    return hash_value < LLM_AGENT_ROLLOUT_PERCENT


async def llm_agent_conversation_handler(
    request: TurnRequest,
    agent: Optional["LLMAgent"] = None
) -> ResponseEnvelope:
    """使用 LLM Agent 处理对话

    这是 LLM-Centric 架构的核心处理器，替代原来的 conversation_fallback_handler。

    Args:
        request: 轮次请求
        agent: LLM Agent 实例（可选，默认使用全局实例）

    Returns:
        ResponseEnvelope: 响应信封
    """
    if agent is None:
        # 该实验链路默认关闭；延迟加载可避免未启用的工具依赖阻断主应用启动。
        from app.agent.agent_factory import get_global_agent

        agent = get_global_agent()

    try:
        # 调用 LLM Agent
        response = await agent.chat(
            thread_id=request.thread_id,
            user_message=request.utterance
        )

        # 从工具调用中提取结构化数据
        structured_data = _extract_structured_data(response.tool_calls)

        # 构建 JSON 响应（包含自然语言回复 + 结构化数据）
        if structured_data:
            json_response = {
                "type": structured_data.get("type", "chat"),
                "message": response.content,
                "data": structured_data.get("data", {})
            }
            raw_response = json.dumps(json_response, ensure_ascii=False)
        else:
            raw_response = response.content

        envelope = response_to_envelope(
            raw_response,
            handled_by="llm_agent",
            trace_id=request.trace_id,
        )

        # 记录工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_result = tool_call.get("result")
                if tool_result:
                    envelope.add_tool_result(
                        tool_name=tool_call.get("tool", "unknown"),
                        success=tool_result.status.value == "success",
                        result_code="LLM_AGENT_TOOL_CALL",
                        data=tool_result.data
                    )

        return envelope

    except Exception as e:
        logger.error(f"LLM Agent 处理失败: {e}", exc_info=True)

        # 降级到错误响应
        error_response = "抱歉，我现在遇到了一些问题，请稍后再试。"
        return response_to_envelope(
            error_response,
            handled_by="llm_agent_error",
            trace_id=request.trace_id,
        )


async def llm_agent_with_confirmation_handler(
    request: TurnRequest,
    confirmed: bool,
    confirmation_data: dict,
    agent: Optional["LLMAgent"] = None
) -> ResponseEnvelope:
    """处理需要确认的操作

    当 LLM Agent 返回需要确认的操作时，用户确认后调用此处理器。

    Args:
        request: 轮次请求
        confirmed: 用户是否确认
        confirmation_data: 确认数据
        agent: LLM Agent 实例

    Returns:
        ResponseEnvelope: 响应信封
    """
    if agent is None:
        from app.agent.agent_factory import get_global_agent

        agent = get_global_agent()

    try:
        # 调用确认处理
        response = await agent.confirm_action(
            thread_id=request.thread_id,
            confirmed=confirmed,
            confirmation_data=confirmation_data
        )

        # 从工具调用中提取结构化数据
        structured_data = _extract_structured_data(response.tool_calls)

        # 构建 JSON 响应（包含自然语言回复 + 结构化数据）
        if structured_data:
            json_response = {
                "type": structured_data.get("type", "chat"),
                "message": response.content,
                "data": structured_data.get("data", {})
            }
            raw_response = json.dumps(json_response, ensure_ascii=False)
        else:
            raw_response = response.content

        envelope = response_to_envelope(
            raw_response,
            handled_by="llm_agent_confirmation",
            trace_id=request.trace_id,
        )

        return envelope

    except Exception as e:
        logger.error(f"LLM Agent 确认处理失败: {e}", exc_info=True)

        # 降级到错误响应
        error_response = "抱歉，操作处理失败，请稍后再试。"
        return response_to_envelope(
            error_response,
            handled_by="llm_agent_confirmation_error",
            trace_id=request.trace_id,
        )


def create_llm_agent_handler():
    """创建 LLM Agent 处理器函数

    返回一个可以用于 execute_turn 的处理器函数。

    Returns:
        Callable: 处理器函数
    """
    async def handler(context) -> Optional[ResponseEnvelope]:
        """LLM Agent 处理器

        Args:
            context: 执行上下文

        Returns:
            ResponseEnvelope: 响应信封
        """
        # 检查是否应该使用 LLM Agent
        if not should_use_llm_agent(context.request.thread_id):
            return None

        # 实验 Agent 仍处于渐进接入阶段，工具工厂或其可选依赖不可用时不能
        # 阻断现有稳定对话链。返回 None 让 Turn facade 继续走确定性 fallback；
        # 真实设备和菜谱事实仍由原有受控 handler 负责。
        try:
            return await llm_agent_conversation_handler(context.request)
        except Exception as exc:
            logger.warning(
                "LLM Agent 初始化失败，回退现有对话链: error_type=%s",
                type(exc).__name__,
                exc_info=True,
            )
            return None

    return handler
