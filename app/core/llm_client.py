"""LLM 客户端封装

封装现有的 Qwen/DashScope 集成，提供统一的 LLM 调用接口。
支持工具调用（Function Calling）。
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import httpx
from app.core.config import settings


@dataclass
class LLMResponse:
    """LLM 响应

    Attributes:
        content: 文本回复
        tool_calls: 工具调用列表（如果有）
        usage: Token 使用统计
    """
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None


class LLMClient:
    """LLM 客户端

    封装 Qwen/DashScope API，支持工具调用。
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None
    ):
        """初始化 LLM 客户端

        Args:
            model: 模型名称，默认使用配置中的主模型
            api_key: API Key，默认使用配置
            base_url: API Base URL，默认使用配置
        """
        self.model = model or settings.LLM_MODEL
        self.api_key = api_key or settings.DASHSCOPE_API_KEY
        self.base_url = base_url or settings.DASHSCOPE_BASE_URL

        # 创建 HTTP 客户端
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """调用 LLM

        Args:
            messages: 消息列表
            tools: 工具定义列表（OpenAI Function Calling 格式）
            temperature: 温度参数
            max_tokens: 最大 Token 数

        Returns:
            LLMResponse: LLM 响应
        """
        # 构建请求
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # 如果有工具，添加工具定义
        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = "auto"

        # 调用 API
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()

            # 解析响应
            choice = result["choices"][0]
            message = choice["message"]

            content = message.get("content", "")
            tool_calls = message.get("tool_calls")
            usage = result.get("usage")

            return LLMResponse(
                content=content or "",
                tool_calls=tool_calls,
                usage=usage
            )

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"LLM API 调用失败: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"LLM API 调用失败: {str(e)}")

    async def close(self):
        """关闭 HTTP 客户端"""
        await self.client.aclose()


class MockLLMClient:
    """Mock LLM 客户端（用于测试）

    模拟 LLM 响应，用于单元测试。
    """

    def __init__(self, responses: List[Dict[str, Any]] = None):
        """初始化 Mock 客户端

        Args:
            responses: 预设的响应列表
        """
        self.responses = responses or []
        self.call_count = 0

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """模拟 LLM 调用

        Args:
            messages: 消息列表
            tools: 工具定义列表
            temperature: 温度参数
            max_tokens: 最大 Token 数

        Returns:
            LLMResponse: 预设的响应
        """
        if self.call_count < len(self.responses):
            response_data = self.responses[self.call_count]
            self.call_count += 1

            return LLMResponse(
                content=response_data.get("content", ""),
                tool_calls=response_data.get("tool_calls"),
                usage={"prompt_tokens": 100, "completion_tokens": 50}
            )
        else:
            # 默认响应
            return LLMResponse(
                content="抱歉，我现在无法回答这个问题。",
                tool_calls=None,
                usage={"prompt_tokens": 100, "completion_tokens": 20}
            )

    async def close(self):
        """关闭客户端（Mock 实现）"""
        pass
