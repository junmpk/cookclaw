"""LLM-Centric Agent 核心框架

这是新的 LLM 自主决策架构的核心组件。
LLM 直接理解用户意图，决定调用哪些工具，生成自然回复。
"""

import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from app.agent.tools import Tool, ToolResult, ToolStatus
from app.core.llm_client import LLMClient
from app.conversation.manager import ConversationManager
from app.agent.guardrails import (
    check_recipe_grounding,
    check_device_safety,
    check_content,
    log_audit,
    GuardrailResult
)


@dataclass
class AgentResponse:
    """Agent 响应

    Attributes:
        content: 自然语言回复
        tool_calls: 工具调用列表
        requires_confirmation: 是否需要用户确认
        confirmation_prompt: 确认提示语
        confirmation_data: 确认数据
    """
    content: str
    tool_calls: List[Dict[str, Any]]
    requires_confirmation: bool = False
    confirmation_prompt: str = ""
    confirmation_data: Optional[Dict[str, Any]] = None


class LLMAgent:
    """LLM-Centric Agent

    核心理念：
    - LLM 直接理解用户意图（不再显式分类）
    - LLM 决定调用哪些工具（不再 if-else 路由）
    - LLM 生成自然回复（不再模板化）
    - 关键操作需要人工确认（护栏）
    """

    def __init__(
        self,
        llm_client: LLMClient,
        conversation_manager: ConversationManager,
        tools: List[Tool],
        system_prompt: str
    ):
        """初始化 LLM Agent

        Args:
            llm_client: LLM 客户端
            conversation_manager: 会话管理器
            tools: 可用工具列表
            system_prompt: 系统提示词
        """
        self.llm_client = llm_client
        self.conversation_manager = conversation_manager
        self.tools = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt

    async def chat(
        self,
        thread_id: str,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """处理用户消息

        这是 Agent 的主循环：
        1. 加载会话历史
        2. 调用 LLM（带工具）
        3. 如果有工具调用，执行工具
        4. 如果工具需要确认，返回确认提示
        5. 继续对话直到 LLM 生成最终回复

        Args:
            thread_id: 会话 ID
            user_message: 用户消息
            context: 额外上下文

        Returns:
            AgentResponse: Agent 响应
        """
        # 加载会话历史
        messages = await self.conversation_manager.load_messages(thread_id)

        # 添加系统提示词
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {
                "role": "system",
                "content": self.system_prompt
            })

        # 添加用户消息
        messages.append({
            "role": "user",
            "content": user_message
        })

        # 添加上下文信息
        if context:
            context_msg = self._format_context(context)
            messages.append({
                "role": "system",
                "content": f"当前上下文：{context_msg}"
            })

        # LLM 循环（可能多次工具调用）
        tool_calls_log = []
        max_iterations = 5  # 防止无限循环

        for iteration in range(max_iterations):
            # 调用 LLM
            response = await self.llm_client.chat(
                messages=messages,
                tools=[tool.get_schema() for tool in self.tools.values()]
            )

            # 检查是否有工具调用
            if response.tool_calls:
                # 执行工具调用
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])

                    # 执行工具
                    if tool_name in self.tools:
                        tool = self.tools[tool_name]
                        result = await tool.execute(**tool_args)

                        # 记录工具调用
                        tool_calls_log.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result": result
                        })

                        # 检查是否需要确认
                        if result.requires_confirmation:
                            # 保存会话状态
                            await self.conversation_manager.save_messages(
                                thread_id,
                                messages
                            )

                            # 返回确认提示
                            return AgentResponse(
                                content=result.confirmation_prompt,
                                tool_calls=tool_calls_log,
                                requires_confirmation=True,
                                confirmation_prompt=result.confirmation_prompt,
                                confirmation_data={
                                    "tool_name": tool_name,
                                    "tool_args": tool_args,
                                    "result_data": result.data
                                }
                            )

                        # 添加工具结果到消息
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(
                                {"status": result.status.value, "data": result.data, "message": result.message},
                                ensure_ascii=False
                            )
                        })
                    else:
                        # 工具不存在
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps({
                                "status": "error",
                                "message": f"未知工具: {tool_name}"
                            })
                        })

                # 继续循环，让 LLM 处理工具结果
                continue

            else:
                # LLM 生成了最终回复
                break

        # === Phase 4: 应用护栏检查 ===
        guardrail_results = []
        final_content = response.content

        # 1. 内容安全检查
        content_check = check_content(final_content)
        guardrail_results.append(content_check)

        if not content_check.passed:
            # 内容安全检查失败，返回安全回复
            logger.warning(f"内容安全检查失败: {content_check.reason}")
            final_content = "抱歉，我的回复可能包含不当内容，请重新描述您的问题。"

        # 2. 菜谱真实性检查（如果有搜索工具调用）
        search_results = []
        for tool_call in tool_calls_log:
            if tool_call["tool"].startswith("search_"):
                result = tool_call["result"]
                if result.status == ToolStatus.SUCCESS and result.data:
                    recipes = result.data.get("results", [])
                    search_results.extend(recipes)

        if search_results:
            grounding_check = check_recipe_grounding(final_content, search_results)
            guardrail_results.append(grounding_check)

            if not grounding_check.passed:
                # 菜谱真实性检查失败，添加警告
                logger.warning(f"菜谱真实性检查失败: {grounding_check.reason}")
                # 不阻断回复，但记录警告
                final_content += "\n\n⚠️ 注意：以上部分菜谱信息可能不准确，请以实际搜索结果为准。"

        # 3. 记录审计日志
        log_audit(
            operation="chat",
            user_id=thread_id,
            details={
                "user_message": user_message,
                "response_length": len(final_content),
                "tool_calls_count": len(tool_calls_log),
                "tools_used": [tc["tool"] for tc in tool_calls_log]
            },
            guardrail_results=guardrail_results
        )

        # 保存会话历史
        messages.append({
            "role": "assistant",
            "content": final_content
        })
        await self.conversation_manager.save_messages(thread_id, messages)

        return AgentResponse(
            content=final_content,
            tool_calls=tool_calls_log,
            requires_confirmation=False
        )

    async def confirm_action(
        self,
        thread_id: str,
        confirmed: bool,
        confirmation_data: Dict[str, Any]
    ) -> AgentResponse:
        """确认或取消操作

        Args:
            thread_id: 会话 ID
            confirmed: 用户是否确认
            confirmation_data: 确认数据

        Returns:
            AgentResponse: Agent 响应
        """
        # 加载会话历史
        messages = await self.conversation_manager.load_messages(thread_id)

        # 添加用户确认消息
        if confirmed:
            messages.append({
                "role": "user",
                "content": "是的，确认执行"
            })
        else:
            messages.append({
                "role": "user",
                "content": "不，取消操作"
            })

        # 重新执行工具（带确认标志）
        tool_name = confirmation_data.get("tool_name")
        tool_args = confirmation_data.get("tool_args", {})

        # === Phase 4: 设备安全检查 ===
        guardrail_results = []

        if tool_name and "device" in tool_name.lower():
            # 这是设备控制操作，进行安全检查
            device_id = tool_args.get("device_id", "unknown")
            action = tool_args.get("action", "unknown")

            device_check = check_device_safety(
                action=action,
                device_id=device_id,
                confirmed=confirmed,
                device_status=None  # 可以传入实际设备状态
            )
            guardrail_results.append(device_check)

            if not device_check.passed:
                logger.warning(f"设备安全检查失败: {device_check.reason}")
                return AgentResponse(
                    content=f"⚠️ 安全错误：{device_check.reason}",
                    tool_calls=[],
                    requires_confirmation=False
                )

        if tool_name in self.tools:
            tool = self.tools[tool_name]

            # 添加 confirmed 参数
            tool_args["confirmed"] = confirmed

            result = await tool.execute(**tool_args)

            # 添加工具结果到消息
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "status": result.status.value,
                    "data": result.data,
                    "message": result.message
                }, ensure_ascii=False)
            })

            # 让 LLM 生成最终回复
            response = await self.llm_client.chat(messages=messages)

            # 内容安全检查
            content_check = check_content(response.content)
            guardrail_results.append(content_check)

            final_content = response.content
            if not content_check.passed:
                logger.warning(f"内容安全检查失败: {content_check.reason}")
                final_content = "抱歉，我的回复可能包含不当内容，请重新描述您的问题。"

            # 记录审计日志
            log_audit(
                operation="confirm_action",
                user_id=thread_id,
                details={
                    "tool_name": tool_name,
                    "confirmed": confirmed,
                    "tool_args": tool_args,
                    "result_status": result.status.value,
                    "response_length": len(final_content)
                },
                guardrail_results=guardrail_results
            )

            messages.append({
                "role": "assistant",
                "content": final_content
            })
            await self.conversation_manager.save_messages(thread_id, messages)

            return AgentResponse(
                content=final_content,
                tool_calls=[],
                requires_confirmation=False
            )

        # 记录取消操作的审计日志
        log_audit(
            operation="confirm_action_cancelled",
            user_id=thread_id,
            details={
                "tool_name": tool_name,
                "confirmed": confirmed,
                "tool_args": tool_args
            },
            guardrail_results=guardrail_results
        )

        return AgentResponse(
            content="操作已取消",
            tool_calls=[],
            requires_confirmation=False
        )

    def _format_context(self, context: Dict[str, Any]) -> str:
        """格式化上下文信息

        Args:
            context: 上下文字典

        Returns:
            str: 格式化后的上下文字符串
        """
        parts = []

        if "user_profile" in context:
            profile = context["user_profile"]
            if profile.get("name"):
                parts.append(f"用户姓名: {profile['name']}")
            if profile.get("preferences"):
                parts.append(f"用户偏好: {', '.join(profile['preferences'])}")

        if "device_status" in context:
            devices = context["device_status"]
            if devices:
                device_list = [f"{d['name']}({d['status']})" for d in devices]
                parts.append(f"设备状态: {', '.join(device_list)}")

        if "recent_searches" in context:
            searches = context["recent_searches"]
            if searches:
                parts.append(f"最近搜索: {', '.join(searches[:3])}")

        return "; ".join(parts) if parts else "无"


def create_default_system_prompt() -> str:
    """创建默认系统提示词

    Returns:
        str: 系统提示词
    """
    return """你是一个智能厨房助手，帮助用户找到菜谱、控制智能厨房设备。

## 你的核心原则

1. **理解优先**：理解用户的真实意图，不要机械匹配关键词
2. **自主决策**：根据用户需求自主选择工具，不要依赖固定规则
3. **自然对话**：用友好、自然的方式回复，像朋友一样
4. **诚实透明**：找不到就诚实告知，不编造信息

## 你拥有的工具（16 个）

### 搜索工具（9 个）

**基础搜索（3 个）**：
- `search_recipes`：通用搜索，适合模糊查询（如"今晚吃什么"、"推荐几个菜"）
- `search_by_ingredients`：按食材搜索，当用户提到具体食材时使用（如"冰箱里有番茄和鸡蛋"）
- `search_by_cuisine`：按菜系搜索，当用户提到菜系时使用（如"想吃川菜"、"来点粤菜"）

**高级搜索（6 个）**：
- `search_by_cooking_method`：按烹饪方法搜索（如"想炒个菜"、"蒸点东西"、"烤个鸡翅"）
- `search_by_taste`：按口味搜索（如"想吃甜的"、"来点清淡的"、"要辣的"）
- `search_by_dietary`：按饮食限制搜索（如"素食"、"低脂"、"减肥餐"）
- `search_by_cooking_time`：按烹饪时间搜索（如"快手菜"、"15分钟内能做好的"）
- `search_by_difficulty`：按难度搜索（如"简单的菜"、"新手能做的"）
- `search_by_occasion`：按场合搜索（如"聚餐"、"约会"、"工作餐"）

### 菜谱详情工具（4 个）

- `get_recipe_details`：获取菜谱完整信息（结构化数据）
- `get_cooking_steps`：获取烹饪步骤
- `get_ingredient_list`：获取食材清单
- `get_recipe_markdown`：获取格式化的 Markdown 详情（适合直接展示给用户）

### 设备控制工具（3 个）

- `control_device`：控制设备（需要用户确认）
- `confirm_device_control`：确认设备操作
- `get_device_status`：查询设备状态

## 工具选择指南

**当用户说...** → **你应该...**

### 搜索菜谱

- "今晚吃什么"、"推荐几个菜" → `search_recipes`（通用搜索）
- "冰箱里有番茄、鸡蛋、土豆" → `search_by_ingredients`（按食材）
- "想吃川菜"、"来点粤菜" → `search_by_cuisine`（按菜系）
- "想炒个菜"、"蒸点东西" → `search_by_cooking_method`（按烹饪方法）
- "想吃甜的"、"来点清淡的" → `search_by_taste`（按口味）
- "素食"、"低脂餐" → `search_by_dietary`（按饮食限制）
- "快手菜"、"15分钟内" → `search_by_cooking_time`（按时间）
- "简单的菜"、"新手能做的" → `search_by_difficulty`（按难度）
- "聚餐要准备的菜"、"约会吃什么" → `search_by_occasion`（按场合）

**组合使用**：
- "想吃清淡的、30分钟内能做的菜" → 可以先用 `search_by_taste`，再用 `search_by_cooking_time` 过滤
- "冰箱有鸡肉，想做简单的" → 可以先用 `search_by_ingredients`，再用 `search_by_difficulty` 过滤

### 查看菜谱详情

- "这道菜怎么做"、"告诉我步骤" → `get_cooking_steps`
- "需要什么食材"、"食材清单" → `get_ingredient_list`
- "这个菜谱的详细信息" → `get_recipe_details`（结构化数据）
- "给我看看这个菜谱"、"展示菜谱" → `get_recipe_markdown`（格式化 Markdown，适合直接展示）

### 控制设备

- "启动电饭煲"、"开始煮饭" → `control_device`（会自动要求用户确认）
- "电饭煲现在什么状态" → `get_device_status`

## 回复风格

- **简洁明了**：不啰嗦，直奔主题
- **友好自然**：像朋友一样对话，不要机械
- **主动引导**：搜索后主动问用户是否需要详情
- **诚实透明**：找不到就诚实告知，不编造

## 示例对话

### 示例 1：通用搜索
用户："今晚吃什么"
你：[调用 search_recipes]
    给你推荐几道菜：
    1. 番茄炒蛋 - 经典家常，10分钟搞定
    2. 清炒时蔬 - 清淡健康，15分钟完成
    3. 红烧肉 - 下饭硬菜，需要40分钟

    你对哪道菜感兴趣？我可以告诉你详细做法。

### 示例 2：按食材搜索
用户："冰箱里有番茄和鸡蛋，能做什么"
你：[调用 search_by_ingredients]
    番茄和鸡蛋可以做很多菜！推荐这几道：
    1. 番茄炒蛋 - 最经典的搭配
    2. 番茄鸡蛋汤 - 清淡开胃
    3. 番茄鸡蛋面 - 简单快手

    想做哪一道？我告诉你步骤。

### 示例 3：组合搜索
用户："想吃清淡的、快点能做的"
你：[先调用 search_by_taste，再调用 search_by_cooking_time]
    清淡又快速的菜，推荐这几道：
    1. 清炒黄瓜 - 5分钟搞定
    2. 凉拌西红柿 - 10分钟完成
    3. 蒸蛋羹 - 15分钟出锅

    需要看哪道菜的详细做法吗？

### 示例 4：设备控制
用户："启动电饭煲煮饭"
你：[调用 control_device]
    确定要启动电饭煲煮饭吗？（预计需要40分钟）

用户："是的"
你：[调用 confirm_device_control]
    好的，电饭煲已启动！预计40分钟后饭就煮好了。我会提前5分钟提醒你。

### 示例 5：查看菜谱详情
用户："番茄炒蛋怎么做"
你：[调用 get_cooking_steps]
    番茄炒蛋的步骤：
    1. 番茄切块，鸡蛋打散
    2. 热锅凉油，先炒鸡蛋
    3. 鸡蛋盛出，再炒番茄
    4. 番茄出汁后，倒回鸡蛋
    5. 加盐调味，翻炒均匀

    需要看食材清单吗？

## 重要提醒

1. **不要编造菜谱**：所有菜谱信息必须来自工具返回的结果
2. **设备控制必须确认**：任何设备操作都要先让用户确认
3. **主动引导对话**：搜索后主动问用户是否需要更多信息
4. **灵活组合工具**：可以连续调用多个工具来满足用户需求
5. **理解上下文**：记住之前的对话，不要重复询问
"""
