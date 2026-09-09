"""LLM Agent 工厂

创建和初始化 LLM Agent 实例，连接所有组件。
"""

from typing import List, Optional
from app.agent.llm_agent import LLMAgent, create_default_system_prompt
from app.agent.tools import Tool
from app.agent.tools.search_recipes import (
    SearchRecipesTool,
    SearchByIngredientsTool,
    SearchByCuisineTool
)
from app.agent.tools.search_advanced import (
    SearchByCookingMethodTool,
    SearchByTasteTool,
    SearchByDietaryTool,
    SearchByCookingTimeTool,
    SearchByDifficultyTool,
    SearchByOccasionTool
)
from app.agent.tools.control_device import (
    ControlDeviceTool,
    ConfirmDeviceControlTool,
    GetDeviceStatusTool
)
from app.agent.tools.get_recipe_details import (
    GetRecipeDetailsTool,
    GetCookingStepsTool,
    GetIngredientListTool,
    GetRecipeMarkdownTool
)
from app.core.llm_client import LLMClient
from app.conversation.manager import ConversationManager
from app.conversation.service import ConversationService
from app.agent.recipe_search_service import RecipeSearchService
from app.agent.skills.recipe_operation.device_control import DeviceController


def create_default_tools(
    search_service: RecipeSearchService,
    device_controller: DeviceController
) -> List[Tool]:
    """创建默认工具列表

    Args:
        search_service: 搜索服务
        device_controller: 设备控制器

    Returns:
        List[Tool]: 工具列表（15 个工具）
    """
    return [
        # 基础搜索工具（3 个）
        SearchRecipesTool(search_service),
        SearchByIngredientsTool(search_service),
        SearchByCuisineTool(search_service),

        # 高级搜索工具（6 个）
        SearchByCookingMethodTool(search_service),
        SearchByTasteTool(search_service),
        SearchByDietaryTool(search_service),
        SearchByCookingTimeTool(search_service),
        SearchByDifficultyTool(search_service),
        SearchByOccasionTool(search_service),

        # 菜谱详情工具（4 个，从 PostgreSQL 查询，不需要 search_service）
        GetRecipeDetailsTool(),
        GetCookingStepsTool(),
        GetIngredientListTool(),
        GetRecipeMarkdownTool(),

        # 设备控制工具（3 个）
        ControlDeviceTool(device_controller),
        ConfirmDeviceControlTool(device_controller),
        GetDeviceStatusTool(device_controller),
    ]


def create_llm_agent(
    conversation_service: Optional[ConversationService] = None,
    search_service: Optional[RecipeSearchService] = None,
    device_controller: Optional[DeviceController] = None,
    llm_client: Optional[LLMClient] = None,
    conversation_manager: Optional[ConversationManager] = None,
    system_prompt: Optional[str] = None
) -> LLMAgent:
    """创建 LLM Agent 实例

    这是主入口函数，用于创建完整的 LLM Agent。

    Args:
        conversation_service: 会话服务（可选，会自动创建）
        search_service: 搜索服务（可选，会自动创建）
        device_controller: 设备控制器（可选，会自动创建）
        llm_client: LLM 客户端（可选，会自动创建）
        conversation_manager: 会话管理器（可选，会自动创建）
        system_prompt: 系统提示词（可选，会使用默认）

    Returns:
        LLMAgent: 配置好的 LLM Agent 实例
    """
    # 创建 LLM 客户端
    if llm_client is None:
        llm_client = LLMClient()

    # 创建会话管理器
    if conversation_manager is None:
        if conversation_service is None:
            conversation_service = ConversationService()
        conversation_manager = ConversationManager(conversation_service)

    # 创建搜索服务
    if search_service is None:
        search_service = RecipeSearchService()

    # 创建设备控制器
    if device_controller is None:
        device_controller = DeviceController()

    # 创建工具
    tools = create_default_tools(search_service, device_controller)

    # 创建系统提示词
    if system_prompt is None:
        system_prompt = create_default_system_prompt()

    # 创建并返回 LLM Agent
    return LLMAgent(
        llm_client=llm_client,
        conversation_manager=conversation_manager,
        tools=tools,
        system_prompt=system_prompt
    )


# 全局 Agent 实例（单例）
_global_agent: Optional[LLMAgent] = None


def get_global_agent() -> LLMAgent:
    """获取全局 LLM Agent 实例

    使用单例模式，确保整个应用只有一个 Agent 实例。

    Returns:
        LLMAgent: 全局 Agent 实例
    """
    global _global_agent

    if _global_agent is None:
        _global_agent = create_llm_agent()

    return _global_agent


def reset_global_agent():
    """重置全局 Agent 实例

    用于测试或重新初始化。
    """
    global _global_agent
    _global_agent = None
