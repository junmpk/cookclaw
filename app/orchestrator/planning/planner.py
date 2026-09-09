"""一次模型调用的结构化 Turn Planner。

本模块不注册任何工具，也不读取存储。Shadow 与 Active 共用同一个单轮 Planner；
Active 结果还必须经过确定性校验和白名单执行器。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from pydantic import ValidationError

from app.core.config import settings
from app.observability.trace import extract_token_usage, observe_model_call
from app.orchestrator.planning.models import PlannerCallResult, TurnContext, TurnPlan


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PLANNER_PROMPT = (
    _PROJECT_ROOT / "app" / "core" / "turn_planner_prompt.md"
).read_text(encoding="utf-8")
_planner_llm = None
_planner_llm_signature: tuple[Any, ...] | None = None


def _planner_model_signature() -> tuple[Any, ...]:
    return (
        settings.TURN_PLANNER_MODEL,
        settings.TURN_PLANNER_TIMEOUT_SECONDS,
        settings.DASHSCOPE_API_KEY,
        settings.DASHSCOPE_BASE_URL,
        settings.INTENT_ENABLE_THINKING,
    )


def get_planner_llm():
    """按当前配置惰性构造模型，off 模式不会初始化 Planner 客户端。"""
    global _planner_llm, _planner_llm_signature
    signature = _planner_model_signature()
    if _planner_llm is not None and _planner_llm_signature == signature:
        return _planner_llm
    model = ChatQwen(
        model=settings.TURN_PLANNER_MODEL,
        # Planner 是协议生成器，不是创意写作模型。固定低温度减少同一上下文
        # 在 schema 字段、动作和槽位上的随机漂移。
        temperature=0,
        max_tokens=1_000,
        timeout=settings.TURN_PLANNER_TIMEOUT_SECONDS,
        max_retries=0,
        enable_thinking=settings.INTENT_ENABLE_THINKING,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
    )
    # DashScope 兼容接口支持 JSON mode。这里仍保留下游 Pydantic 与
    # PlanValidator 双重校验；JSON mode 只约束传输格式，不授予任何动作权限。
    _planner_llm = model.bind(response_format={"type": "json_object"})
    _planner_llm_signature = signature
    return _planner_llm


def _message_content(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        return str(text or "").strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _json_object(text: str) -> dict[str, Any]:
    """接受纯 JSON 或单个 Markdown JSON fence；其它前后缀均视为解析失败。"""
    value = str(text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise ValueError("invalid_json_fence")
        value = "\n".join(lines[1:-1]).strip()
        if value.lower().startswith("json"):
            value = value[4:].lstrip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("planner_output_must_be_object")
    return parsed


class TurnPlanner:
    """受约束单轮 Planner；模型可注入以便离线测试。"""

    def __init__(
        self,
        model: Any | None = None,
        *,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._model = model
        self.model_name = str(
            model_name or settings.TURN_PLANNER_MODEL
        ).strip()
        self.timeout_seconds = max(
            0.1,
            float(
                timeout_seconds
                if timeout_seconds is not None
                else settings.TURN_PLANNER_TIMEOUT_SECONDS
            ),
        )

    def _resolve_model(self):
        return self._model if self._model is not None else get_planner_llm()

    async def plan(self, context: TurnContext) -> PlannerCallResult:
        started = time.monotonic()
        messages = [
            SystemMessage(content=_PLANNER_PROMPT),
            HumanMessage(
                content=json.dumps(
                    context.planner_payload(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        ]
        try:
            response = await observe_model_call(
                lambda: self._resolve_model().ainvoke(messages),
                stage="turn_planner",
                model=self.model_name,
                timeout_seconds=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return PlannerCallResult(
                status="timeout",
                error_code="PLANNER_TIMEOUT",
                model=self.model_name,
                duration_ms=(time.monotonic() - started) * 1_000,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return PlannerCallResult(
                status="model_error",
                error_code="PLANNER_MODEL_ERROR",
                model=self.model_name,
                duration_ms=(time.monotonic() - started) * 1_000,
            )

        usage = extract_token_usage(response)
        duration_ms = (time.monotonic() - started) * 1_000
        try:
            payload = _json_object(_message_content(response))
            plan = TurnPlan.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            return PlannerCallResult(
                status="parse_error",
                error_code="PLANNER_SCHEMA_ERROR",
                model=self.model_name,
                duration_ms=duration_ms,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )
        return PlannerCallResult(
            status="success",
            plan=plan,
            model=self.model_name,
            duration_ms=duration_ms,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )
