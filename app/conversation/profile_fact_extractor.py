"""一次、无工具、结构化的通用用户画像事实提取器。"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.conversation.profile_facts import (
    ProfileFactCandidate,
    ProfileFactExtraction,
    contains_sensitive_profile_data,
    filter_grounded_profile_facts,
    may_contain_profile_fact,
)
from app.core.config import settings
from app.observability.trace import observe_model_call


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_EXTRACTOR_PROMPT = (
    _PROJECT_ROOT / "app" / "core" / "profile_memory_extractor_prompt.md"
).read_text(encoding="utf-8")
_extractor_llm = None
_extractor_signature: tuple[Any, ...] | None = None


def _model_signature() -> tuple[Any, ...]:
    return (
        settings.PROFILE_FACT_EXTRACTOR_MODEL,
        settings.PROFILE_FACT_EXTRACTOR_TIMEOUT_SECONDS,
        settings.DASHSCOPE_API_KEY,
        settings.DASHSCOPE_BASE_URL,
    )


def get_profile_fact_extractor_llm():
    """按配置惰性构造客户端；功能关闭且没有候选消息时不会初始化。"""
    global _extractor_llm, _extractor_signature
    signature = _model_signature()
    if _extractor_llm is not None and _extractor_signature == signature:
        return _extractor_llm
    model = ChatQwen(
        model=settings.PROFILE_FACT_EXTRACTOR_MODEL,
        temperature=0,
        max_tokens=600,
        timeout=settings.PROFILE_FACT_EXTRACTOR_TIMEOUT_SECONDS,
        max_retries=0,
        enable_thinking=False,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
    )
    _extractor_llm = model.bind(response_format={"type": "json_object"})
    _extractor_signature = signature
    return _extractor_llm


def _message_content(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "").strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        ).strip()
    return str(content or "").strip()


def _json_object(text: str) -> dict[str, Any]:
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
        raise ValueError("profile_fact_output_must_be_object")
    return parsed


class ProfileFactExtractorResult(BaseModel):
    """提取结果不携带被拒候选正文，便于安全观测。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["skipped", "success", "timeout", "model_error", "parse_error"]
    facts: list[ProfileFactCandidate] = Field(default_factory=list, max_length=3)
    candidate_count: int = Field(default=0, ge=0, le=3)
    rejected_count: int = Field(default=0, ge=0, le=3)
    model: str = ""
    duration_ms: float = Field(default=0, ge=0)
    error_code: str | None = None


class ProfileFactExtractor:
    """模型可注入，生产和离线测试共用同一解析及安全校验。"""

    def __init__(
        self,
        model: Any | None = None,
        *,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._model = model
        self.model_name = str(
            model_name or settings.PROFILE_FACT_EXTRACTOR_MODEL
        ).strip()
        self.timeout_seconds = max(
            0.01,
            float(
                timeout_seconds
                if timeout_seconds is not None
                else settings.PROFILE_FACT_EXTRACTOR_TIMEOUT_SECONDS
            ),
        )

    def _resolve_model(self):
        return self._model if self._model is not None else get_profile_fact_extractor_llm()

    async def extract(
        self,
        user_message: str,
        *,
        force: bool = False,
    ) -> ProfileFactExtractorResult:
        text = " ".join(str(user_message or "").split()).strip()
        if not text or (not force and not may_contain_profile_fact(text)):
            return ProfileFactExtractorResult(status="skipped", model=self.model_name)
        # 密钥、联系方式、精确地址等不仅不能落库，也不应为了判断是否记忆而
        # 再发送给提取模型。
        if contains_sensitive_profile_data(text):
            return ProfileFactExtractorResult(
                status="skipped",
                model=self.model_name,
                error_code="PROFILE_FACT_SENSITIVE_INPUT_REJECTED",
            )
        started = time.monotonic()
        messages = [
            SystemMessage(content=_EXTRACTOR_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {"user_message": text[:2_000]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        ]
        try:
            response = await observe_model_call(
                lambda: self._resolve_model().ainvoke(messages),
                stage="profile_fact_extractor",
                model=self.model_name,
                timeout_seconds=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return ProfileFactExtractorResult(
                status="timeout",
                model=self.model_name,
                duration_ms=(time.monotonic() - started) * 1_000,
                error_code="PROFILE_FACT_EXTRACTOR_TIMEOUT",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ProfileFactExtractorResult(
                status="model_error",
                model=self.model_name,
                duration_ms=(time.monotonic() - started) * 1_000,
                error_code="PROFILE_FACT_EXTRACTOR_MODEL_ERROR",
            )

        try:
            payload = _json_object(_message_content(response))
            extraction = ProfileFactExtraction.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            return ProfileFactExtractorResult(
                status="parse_error",
                model=self.model_name,
                duration_ms=(time.monotonic() - started) * 1_000,
                error_code="PROFILE_FACT_EXTRACTOR_SCHEMA_ERROR",
            )
        accepted = filter_grounded_profile_facts(text, extraction.facts)
        return ProfileFactExtractorResult(
            status="success",
            facts=accepted,
            candidate_count=len(extraction.facts),
            rejected_count=len(extraction.facts) - len(accepted),
            model=self.model_name,
            duration_ms=(time.monotonic() - started) * 1_000,
        )
