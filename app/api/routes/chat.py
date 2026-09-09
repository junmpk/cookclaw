"""
核心 HTTP API 路由 —— 根 / 健康检查 / 流式问答（从 app/main.py 抽出）。

只承载 RAG/Agent 自有的请求链路（语言无关的服务接口），不依赖任何通道全局状态。
路由本身不带前缀，由 main.py 挂到 `settings.API_PREFIX`（/api/v1）下，路径与重构前一致：
  GET  /api/v1/        根（存活确认）
  GET  /api/v1/health  健康检查
  POST /api/v1/chat    流式问答（SSE）→ participle_agent.chat_stream
"""
import json
import re
import uuid
from typing import Annotated

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, StringConstraints, field_validator

from app.agent.participle_agent import chat_stream
from app.core.config import settings

router = APIRouter()

_THREAD_ID_HEADER = "X-CookClaw-Thread-Id"
_MAX_QUESTION_CHARS = 8_000
_MAX_THREAD_ID_CHARS = 128
_WEB_THREAD_ID_PATTERN = re.compile(r"(?:web:)?[0-9a-fA-F]{32}")

QuestionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=_MAX_QUESTION_CHARS),
]
ThreadId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=_MAX_THREAD_ID_CHARS),
]


class ChatRequest(BaseModel):
    """问答请求体"""

    question: QuestionText
    # 旧客户端只传 question 时继续可用；新客户端应保存响应头里的 ID，
    # 并在后续请求中原样传回，才能承接待澄清状态和候选上下文。
    thread_id: ThreadId | None = None

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id_for_header(cls, value: str | None) -> str | None:
        if value is not None and not _WEB_THREAD_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "thread_id must be a server-issued 128-bit Web session token"
            )
        return value


def _resolve_web_thread_id(value: str | None) -> str:
    """返回可安全写入响应头的 Web 会话 ID。"""
    if value is None:
        return f"web:{uuid.uuid4().hex}"
    if not _WEB_THREAD_ID_PATTERN.fullmatch(value):
        raise ValueError("invalid Web session token")
    # Web session token 具有 128 bit 熵并作为匿名会话 bearer 使用；不能接受
    # 低熵自定义 ID，也不能借用 IM 命名空间。
    return value if value.startswith("web:") else f"web:{value}"


@router.get("/")
async def root():
    # 根路径，用来快速确认服务在不在。
    return {"message": f"{settings.PROJECT_NAME} is running"}


@router.get("/health")
async def health_check():
    # 健康检查接口，部署探活和启动确认都会用到。
    return {
        "status": "ok",
        "version": settings.VERSION,
        "models": {
            "profile": settings.LLM_MODEL_PROFILE,
            "main": settings.LLM_MODEL,
            "qa": settings.QA_MODEL,
            "intent": settings.INTENT_MODEL,
        },
        "web_search": {
            "enabled": settings.WEB_SEARCH_ENABLED,
            "model": settings.WEB_SEARCH_MODEL,
            "strategy": settings.WEB_SEARCH_STRATEGY,
        },
        "orchestration": {
            "turn_mode": "unified",
            "planner_mode": settings.TURN_PLANNER_MODE,
            "planner_model": settings.TURN_PLANNER_MODEL,
            "planner_rollout_percent": (
                settings.TURN_PLANNER_ACTIVE_ROLLOUT_PERCENT
            ),
        },
    }


@router.post("/chat")
async def chat_question(req: ChatRequest):
    """流式问答接口（SSE）"""
    thread_id = _resolve_web_thread_id(req.thread_id)

    # 这个接口返回 SSE，前端能一边收到一边显示。
    async def event_stream():
        async for chunk in chat_stream(req.question, thread_id=thread_id):
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            _THREAD_ID_HEADER: thread_id,
            "Cache-Control": "no-store",
        },
    )
