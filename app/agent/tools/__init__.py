"""工具接口定义

定义 LLM Agent 可调用的工具接口和数据结构。
"""

from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum


class ToolStatus(str, Enum):
    """工具执行状态"""
    SUCCESS = "success"
    ERROR = "error"
    CONFIRMATION_REQUIRED = "confirmation_required"


@dataclass
class ToolResult:
    """工具执行结果

    Attributes:
        status: 执行状态
        data: 返回数据（JSON 可序列化）
        message: 人类可读的消息
        requires_confirmation: 是否需要用户确认
        confirmation_prompt: 确认提示语
    """
    status: ToolStatus
    data: Optional[Any] = None
    message: str = ""
    requires_confirmation: bool = False
    confirmation_prompt: str = ""

    @classmethod
    def success(cls, data: Any = None, message: str = "") -> "ToolResult":
        """创建成功结果"""
        return cls(status=ToolStatus.SUCCESS, data=data, message=message)

    @classmethod
    def error(cls, message: str, data: Any = None) -> "ToolResult":
        """创建错误结果"""
        return cls(status=ToolStatus.ERROR, data=data, message=message)

    @classmethod
    def requires_confirmation(
        cls,
        confirmation_prompt: str,
        data: Any = None
    ) -> "ToolResult":
        """创建需要确认的结果"""
        return cls(
            status=ToolStatus.CONFIRMATION_REQUIRED,
            data=data,
            message=confirmation_prompt,
            requires_confirmation=True,
            confirmation_prompt=confirmation_prompt
        )


class Tool:
    """工具基类

    所有工具必须继承此类并实现 execute 方法。
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    async def execute(self, **kwargs) -> ToolResult:
        """执行工具

        Args:
            **kwargs: 工具参数（由 LLM 生成）

        Returns:
            ToolResult: 执行结果
        """
        raise NotImplementedError

    def get_schema(self) -> dict:
        """获取工具 schema（OpenAI Function Calling 格式）

        Returns:
            dict: 工具 schema
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
