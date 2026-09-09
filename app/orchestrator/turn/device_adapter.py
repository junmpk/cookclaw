"""将确定性 Device Handler 的兼容响应适配为统一编排结果。

本模块只描述已经完成的设备领域决策。设备启动、停止和状态查询仍由既有
确定性函数执行；Adapter 不根据回复文案猜测指令是否发送或设备是否已运行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.observability.trace import record_domain_result
from app.orchestrator.turn.response_renderer import response_to_envelope
from app.orchestrator.turn.runtime_models import ResponseEnvelope, ToolResult


def _label(value: object, *, limit: int = 80) -> str:
    return str(value or "").strip()[:limit]


def _result_code(value: object) -> str:
    normalized = re.sub(
        r"[^A-Z0-9]+",
        "_",
        str(value or "DEVICE_HANDLED").upper(),
    ).strip("_")
    return (normalized or "DEVICE_HANDLED")[:80]


@dataclass(frozen=True, slots=True)
class DeviceHandlerResult:
    """显式设备领域结果；不包含菜谱、设备或用户输入正文。"""

    envelope: ResponseEnvelope
    operation: str
    success: bool
    result_code: str
    risk: str = "low"
    command_sent: bool | None = None
    state_unknown: bool = False
    duration_ms: float = 0.0

    def observe(self) -> None:
        record_domain_result(
            "device_flow",
            duration_ms=self.duration_ms,
            success=self.success,
            result_code=self.result_code,
            operation=self.operation,
            command_sent=self.command_sent,
            state_unknown=self.state_unknown,
            risk=self.risk,
        )


def adapt_device_response(
    raw_response: str,
    *,
    trace_id: str | None,
    operation: str,
    success: bool,
    result_code: str,
    risk: str = "low",
    observed_tool: ToolResult | None = None,
    duration_ms: float = 0.0,
) -> DeviceHandlerResult:
    """适配旧响应；工具事实只取自结构化 ToolResult，不解析用户可见文案。"""
    command_sent: bool | None = None
    state_unknown = False
    if observed_tool is not None:
        value = observed_tool.facts.get("command_sent")
        if isinstance(value, bool):
            command_sent = value
        state_unknown = bool(observed_tool.facts.get("state_unknown"))

    result = DeviceHandlerResult(
        envelope=response_to_envelope(
            raw_response,
            handled_by="device_handler",
            trace_id=trace_id,
        ),
        operation=_label(operation, limit=64) or "respond",
        success=bool(success),
        result_code=_result_code(result_code),
        risk=_label(risk, limit=16) or "low",
        command_sent=command_sent,
        state_unknown=state_unknown,
        duration_ms=max(0.0, float(duration_ms)),
    )
    result.observe()
    return result
