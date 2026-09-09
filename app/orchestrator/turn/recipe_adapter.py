"""将 Recipe Handler 的兼容响应适配为统一编排结果。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.observability.trace import record_domain_result
from app.orchestrator.turn.response_renderer import response_to_envelope
from app.orchestrator.turn.runtime_models import ResponseEnvelope


def _label(value: object, *, limit: int = 80) -> str:
    return str(value or "").strip()[:limit]


def _result_code(value: object) -> str:
    normalized = re.sub(
        r"[^A-Z0-9]+",
        "_",
        str(value or "RECIPE_HANDLED").upper(),
    ).strip("_")
    return (normalized or "RECIPE_HANDLED")[:80]


@dataclass(frozen=True, slots=True)
class RecipeHandlerResult:
    """显式 Recipe 结果；审计字段不包含菜谱正文或用户输入。"""

    envelope: ResponseEnvelope
    operation: str
    success: bool
    result_code: str
    result_count: int = 0
    duration_ms: float = 0.0

    def observe(self) -> None:
        record_domain_result(
            "recipe_flow",
            duration_ms=self.duration_ms,
            success=self.success,
            result_code=self.result_code,
            operation=self.operation,
            result_count=self.result_count,
        )


def adapt_recipe_response(
    raw_response: str,
    *,
    trace_id: str | None,
    operation: str,
    success: bool,
    result_code: str,
    result_count: int = 0,
    duration_ms: float = 0.0,
) -> RecipeHandlerResult:
    """只在 Recipe Adapter 边界解析一次旧响应，保留公开 payload。"""
    result = RecipeHandlerResult(
        envelope=response_to_envelope(
            raw_response,
            handled_by="recipe_handler",
            trace_id=trace_id,
        ),
        operation=_label(operation, limit=64) or "respond",
        success=bool(success),
        result_code=_result_code(result_code),
        result_count=max(0, int(result_count)),
        duration_ms=max(0.0, float(duration_ms)),
    )
    result.observe()
    return result
