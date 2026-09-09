"""
意图分类 —— orchestrator 层（从 app/agent/participle_agent.py 抽出）。

提供强类型意图模型（IntentCategory / IntentResult）+ 轻量 LLM 分类器 classify_intent()。
兼容原 participle_agent._intent_check() 的 r/c/k，并扩展结构化 action/SearchRequest：
  - 同一个 qwen-plus 轻量模型 + 同一份 app/core/intent_check_prompt.md；
  - 解析 r/c/a/k/slots/s；
  - 异常/解析失败回退 related=True, category=unknown, keywords=[]（与旧行为一致）。
仅把返回值从 dict 升级为 IntentResult（旧 r/c/k 仍可解析，见 from_raw）。
"""
from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field, model_validator

from app.core.config import settings
from app.observability.trace import observe_model_call
from app.orchestrator.routing_models import RoutingContext
from app.orchestrator.routing_policy import infer_classifier_action
from app.orchestrator.search_request import SearchRequest

# 复用主应用一致的 .env（uvicorn reload 下也能找到）；与 participle_agent 解析同一个根
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class IntentCategory(str, Enum):
    """意图类别。枚举值与 intent_check_prompt.md 输出的 c 字段字符串一致。"""

    recipe_search = "recipe_search"
    recipe_recommend = "recipe_recommend"
    recipe_execute = "recipe_execute"
    device_manage = "device_manage"
    cooking_qa = "cooking_qa"
    greeting = "greeting"
    off_topic = "off_topic"
    unknown = "unknown"

    @classmethod
    def from_str(cls, value: Any) -> "IntentCategory":
        """任意 c 值 → 枚举；未知/异常值统一落 unknown。

        与旧分类兼容：未知 c 不命中 search/greeting/execute，最终交给上层单次
        无工具 QA/smalltalk 或确定性拒答。
        """
        try:
            return cls(value)
        except (ValueError, TypeError):
            return cls.unknown


class BusinessOperation(str, Enum):
    """跨分类模型和业务路由共享的强业务操作。"""

    exact_search = "exact_search"
    fuzzy_search = "fuzzy_search"
    fuzzy_recommend = "fuzzy_recommend"
    device_control = "device_control"
    device_query = "device_query"
    general = "general"


class IntentResult(BaseModel):
    """意图分类结果（强类型）。保留 raw 便于排查与未来扩展 domain/action/slots。"""

    related: bool = True
    category: IntentCategory = IntentCategory.unknown
    keywords: list[str] = Field(default_factory=list)
    confidence: Optional[float] = None  # 当前 prompt 不输出，保持 None，勿编造数值
    action: str = "unknown"
    source: str = "classifier"
    reason_code: str = "CLASSIFIER_UNKNOWN_UNKNOWN"
    slots: dict[str, Any] = Field(default_factory=dict)
    context_ref: Optional[dict[str, Any]] = None
    needs_clarification: bool = False
    search_request: Optional[SearchRequest] = None
    business_operation: BusinessOperation = BusinessOperation.general
    raw: Optional[dict] = None

    # ── 语义信号（分类器输出；None=未判断，router 走正则兜底）──
    is_finalize_request: Optional[bool] = None
    is_no_constraints: Optional[bool] = None
    is_constraint_reply: Optional[bool] = None
    is_topic_change: Optional[bool] = None
    is_preference_affirm: Optional[bool] = None
    is_preference_negate: Optional[bool] = None

    @model_validator(mode="after")
    def infer_business_operation(self) -> "IntentResult":
        """让测试桩和内部构造也遵守与模型 JSON 相同的业务操作映射。"""
        if self.business_operation != BusinessOperation.general:
            return self
        if self.search_request is not None:
            self.business_operation = BusinessOperation(self.search_request.task_operation)
        elif self.category == IntentCategory.recipe_execute:
            self.business_operation = BusinessOperation.device_control
        elif self.category == IntentCategory.device_manage:
            self.business_operation = BusinessOperation.device_query
        return self

    @classmethod
    def from_raw(
        cls,
        data: dict,
        *,
        original_text: str = "",
        routing_context: RoutingContext | None = None,
    ) -> "IntentResult":
        """从旧 JSON（字段 r/c/k）解析，保持与 _intent_check 完全一致的默认值。

        keywords 的 None → []（旧代码里 None 为假值、永不进入 ' '.join，等价无害）。
        """
        keywords = data.get("k", []) or []
        if isinstance(keywords, str):
            keywords = [keywords]
        elif not isinstance(keywords, list):
            keywords = []
        category = IntentCategory.from_str(data.get("c", "unknown"))
        action, reason_code, _risk = infer_classifier_action(
            category.value,
            data.get("a"),
            routing_context,
        )
        search_request = None
        if category in {IntentCategory.recipe_search, IntentCategory.recipe_recommend}:
            search_request = SearchRequest.from_intent_raw(
                data,
                original_text=original_text,
                keywords=[str(k) for k in keywords],
            )
        if search_request is not None:
            business_operation = BusinessOperation(search_request.task_operation)
        elif category == IntentCategory.recipe_execute:
            business_operation = BusinessOperation.device_control
        elif category == IntentCategory.device_manage:
            business_operation = BusinessOperation.device_query
        else:
            business_operation = BusinessOperation.general
        signals = data.get("signals") or {}
        if not isinstance(signals, dict):
            signals = {}
        return cls(
            related=bool(data.get("r", True)),
            category=category,
            keywords=[str(k) for k in keywords],
            action=action,
            source="classifier",
            reason_code=reason_code,
            slots=dict(data.get("slots") or {}) if isinstance(data.get("slots"), dict) else {},
            context_ref=(
                routing_context.context_ref()
                if routing_context and routing_context.has_routing_signal()
                else None
            ),
            needs_clarification=bool(data.get("needs_clarification", False)),
            search_request=search_request,
            business_operation=business_operation,
            raw=data,
            is_finalize_request=signals.get("finalize"),
            is_no_constraints=signals.get("no_constraints"),
            is_constraint_reply=signals.get("constraint_reply"),
            is_topic_change=signals.get("topic_change"),
            is_preference_affirm=signals.get("preference_affirm"),
            is_preference_negate=signals.get("preference_negate"),
        )


# 意图分类提示词 + 轻量模型；默认固定 qwen-plus，也可用 INTENT_MODEL 做全链路对比。
_INTENT_PROMPT = (_PROJECT_ROOT / "app" / "core" / "intent_check_prompt.md").read_text(encoding="utf-8")

_intent_llm = ChatQwen(
    model=settings.INTENT_MODEL,
    max_tokens=320,
    timeout=10,
    max_retries=1,
    enable_thinking=settings.INTENT_ENABLE_THINKING,
    api_key=settings.DASHSCOPE_API_KEY,
    base_url=settings.DASHSCOPE_BASE_URL,
)


def _log_token_usage(label: str, resp_or_msg) -> None:
    """从 LLM 响应或 AIMessage 提取并打印 token 用量（原 participle_agent 同名函数平移）。"""
    usage = None
    if hasattr(resp_or_msg, "usage_metadata") and resp_or_msg.usage_metadata:
        um = resp_or_msg.usage_metadata
        usage = {"input": um.get("input_tokens", 0), "output": um.get("output_tokens", 0)}
    elif hasattr(resp_or_msg, "response_metadata") and resp_or_msg.response_metadata:
        rm = resp_or_msg.response_metadata
        tu = rm.get("token_usage") or rm.get("usage") or {}
        usage = {
            "input": tu.get("input_tokens", tu.get("prompt_tokens", 0)),
            "output": tu.get("output_tokens", tu.get("completion_tokens", 0)),
        }
    if usage:
        total = usage["input"] + usage["output"]
        print(f"  📊 [{label}] tokens — input: {usage['input']}, output: {usage['output']}, total: {total}")
    else:
        print(f"  📊 [{label}] tokens — (无用量数据)")


async def classify_intent(
    question: str,
    routing_context: RoutingContext | None = None,
) -> IntentResult:
    """轻量级意图分类；有状态时只附带紧凑 RoutingContext 摘要。"""
    _t0 = time.monotonic()
    try:
        classifier_input: str = question
        if routing_context and routing_context.has_routing_signal():
            classifier_input = json.dumps(
                {
                    "utterance": question,
                    "context": routing_context.classifier_summary(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        messages = [
            SystemMessage(content=_INTENT_PROMPT),
            HumanMessage(content=classifier_input),
        ]
        resp = await observe_model_call(
            lambda: _intent_llm.ainvoke(messages),
            stage="intent_classifier",
            model=settings.INTENT_MODEL,
            intent=True,
        )
        raw = resp.content.strip()

        _log_token_usage("意图分类", resp)

        # 清理可能的 markdown 代码块包裹
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rstrip("`").strip()

        data = json.loads(raw)
        _elapsed = time.monotonic() - _t0
        print(
            f"  🔍 意图分类 ({_elapsed:.2f}s): r={data.get('r')}, "
            f"c={data.get('c')}, a={data.get('a') or '-'}, "
            f"keyword_count={len(data.get('k') or [])}, "
            f"context={bool(routing_context and routing_context.has_routing_signal())}"
        )
        return IntentResult.from_raw(
            data,
            original_text=question,
            routing_context=routing_context,
        )
    except (json.JSONDecodeError, Exception) as e:
        _elapsed = time.monotonic() - _t0
        print(f"  ⚠️ 意图分类异常 ({_elapsed:.2f}s): error_type={type(e).__name__}")
        # 分类失败显式标记 parse_error，由统一路由拒判并只追问一个关键问题。
        return IntentResult(
            related=True,
            category=IntentCategory.unknown,
            keywords=[],
            action="unknown",
            source="parse_error",
            reason_code="CLASSIFIER_PARSE_ERROR",
            context_ref=(
                routing_context.context_ref()
                if routing_context and routing_context.has_routing_signal()
                else None
            ),
            needs_clarification=True,
        )
