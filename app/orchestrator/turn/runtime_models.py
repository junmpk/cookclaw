"""阶段 2 的单轮运行协议。

这些模型只描述一次业务轮次、工具结果、状态变更和最终响应。它们不授权设备
副作用；启动、停止、记忆写入等动作仍必须由既有确定性处理器完成。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TurnChannel = Literal["qq", "weixin", "whatsapp", "web", "unknown"]
StateScope = Literal["session", "profile", "device", "none"]
StateOperation = Literal["set", "clear", "append", "none"]


class DerivedTurnContext(BaseModel):
    """由受控前处理工具产生的单轮事实。

    当前只承载视觉识别的结构化线索。这里不保存图片 URL、本地路径或二进制内容；
    识别结果是可纠正的工具派生事实，不等同于用户陈述，也不能授权设备副作用。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["derived_turn_context_v1"] = (
        "derived_turn_context_v1"
    )
    source: Literal["vision"] = "vision"
    scene_type: str = Field(default="unknown", max_length=40)
    media_count: int = Field(default=1, ge=1, le=6)
    ingredients: list[str] = Field(
        default_factory=list,
        max_length=12,
        repr=False,
    )
    dishes: list[str] = Field(
        default_factory=list,
        max_length=4,
        repr=False,
    )
    recognition_uncertain: bool = True

    def planner_payload(self) -> dict[str, Any]:
        """返回 Planner 可见值；媒体地址和鉴权信息从未进入本模型。"""
        return self.model_dump()

    def context_ref(self) -> dict[str, Any]:
        """返回可记录的形状，不暴露识别值。"""
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "scene_type": self.scene_type,
            "media_count": self.media_count,
            "ingredient_count": len(self.ingredients),
            "dish_count": len(self.dishes),
            "recognition_uncertain": self.recognition_uncertain,
        }


class TurnRequest(BaseModel):
    """通道进入统一编排器时的稳定输入。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["turn_request_v1"] = "turn_request_v1"
    utterance: str = Field(max_length=10_000, repr=False)
    thread_id: str = Field(min_length=1, max_length=512, repr=False)
    channel: TurnChannel = "unknown"
    message_type: str = Field(default="text", min_length=1, max_length=24)
    trace_id: str | None = Field(default=None, max_length=64)
    derived_context: DerivedTurnContext | None = Field(
        default=None,
        repr=False,
    )

    def context_ref(self) -> dict[str, Any]:
        """只返回可记录的形状，不暴露原话或 thread_id。"""
        result = {
            "schema_version": self.schema_version,
            "channel": self.channel,
            "message_type": self.message_type,
            "input_chars": len(self.utterance),
            "has_trace": bool(self.trace_id),
        }
        if self.derived_context is not None:
            result["derived_context"] = self.derived_context.context_ref()
        return result


class ToolResult(BaseModel):
    """领域工具的统一结果；阶段 2.3 从真实 Trace 映射，不授权工具执行。"""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1, max_length=80)
    success: bool
    code: str = Field(default="", max_length=80)
    facts: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = Field(default=None, max_length=80)
    duration_ms: float = Field(default=0.0, ge=0)


class StatePatch(BaseModel):
    """实际状态前后差异；只记录字段名，不自行写入存储。"""

    model_config = ConfigDict(extra="forbid")

    scope: StateScope = "none"
    operation: StateOperation = "none"
    keys: list[str] = Field(default_factory=list, max_length=32)
    reason_code: str = Field(default="", max_length=80)


class ResponseEnvelope(BaseModel):
    """核心层返回给通道 Renderer 的统一响应。

    ``channel_payload`` 表示现有 IM JSON 协议的原始公开载荷。内部的工具结果、状态
    patch、handler 和 trace 不会被 Renderer 暴露给终端用户。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["response_envelope_v1"] = "response_envelope_v1"
    response_type: str = Field(default="text", min_length=1, max_length=80)
    intent: str | None = Field(default=None, max_length=80)
    lang: str | None = Field(default=None, max_length=16)
    message: str = Field(default="", max_length=200_000)
    data: dict[str, Any] = Field(default_factory=dict)
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    channel_payload: dict[str, Any] | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    tool_results: list[ToolResult] = Field(default_factory=list, max_length=16)
    state_patches: list[StatePatch] = Field(default_factory=list, max_length=16)
    handled_by: str = Field(default="", max_length=80)
    trace_id: str | None = Field(default=None, max_length=64)

    def public_payload(self) -> dict[str, Any] | None:
        """生成通道可见 payload；不包含内部编排字段。"""
        if self.channel_payload is not None:
            return deepcopy(self.channel_payload)
        if self.response_type == "text" and not any(
            (self.intent, self.lang, self.data, self.extra_fields)
        ):
            return None
        payload: dict[str, Any] = {
            "type": self.response_type,
            "intent": self.intent or "chat",
            "lang": self.lang or "zh",
            "data": deepcopy(self.data),
            "message": self.message,
        }
        for key, value in self.extra_fields.items():
            if key not in payload:
                payload[key] = deepcopy(value)
        return payload
