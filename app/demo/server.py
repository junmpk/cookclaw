"""uvicorn app.demo.server:app --host 127.0.0.1 --port 8010 (single worker)."""
import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field, field_validator

from .agents import Agents
from .chat_intent import (
    brief_from_task_state,
    decision_from_chat,
    requests_graph,
    requests_unspecified_adjustment,
    slot_adjustment_from_chat,
)
from .engine import Engine
from .knowledge import RULES
from .models import Brief
from .retrieval import RecipeStore
from .storage import DemoStorage

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def _local_demo_values() -> dict[str, str]:
    """只读取当前公开 checkout 的配置，不向上搜索，也不在导入时改环境。"""
    values: dict[str, str] = {}
    for path in (ROOT / ".env.demo", ROOT / ".env"):
        for key, value in dotenv_values(path).items():
            if value is not None and key not in values:
                values[str(key)] = str(value)
    return values


def _apply_demo_environment(values: dict[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}

    def assign(key: str, value: str, *, force: bool = False) -> None:
        if not force and key in os.environ:
            return
        if key not in previous:
            previous[key] = os.environ.get(key)
        os.environ[key] = value

    for key, value in values.items():
        assign(key, value)
    assign("RECIPE_SEARCH_BACKEND", "public_demo", force=True)
    assign("CONVERSATION_STORE", "memory", force=True)
    assign("PROFILE_STORE", "same", force=True)
    model = os.getenv("DEMO_MODEL", "qwen3.7-flash-2026-07-15")
    for setting in ("LLM_MODEL", "QA_MODEL", "INTENT_MODEL"):
        assign(setting, model, force=True)
    return previous


def _restore_environment(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class Decision(BaseModel):
    version: int = Field(ge=1)
    approved: bool


class Revision(BaseModel):
    version: int = Field(ge=1)
    brief: Brief


class ChatMessage(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    # shared 是普通聊天；rehearsal 仅供自动化测试或高级演示使用。
    mode: Literal["shared", "live", "rehearsal"] = "shared"
    thread_id: str | None = Field(default=None, pattern=r"^web:[0-9a-f]{32}$")

    @field_validator("message")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("消息不能为空")
        return value.strip()


def _graph_response(run: dict, *, thread_id: str, reply: str) -> dict:
    return {
        "thread_id": thread_id,
        "reply": reply,
        "graph_run_id": run["id"],
        "graph_status": run["status"],
        "response": {
            "schema_version": "response_envelope_v1",
            "response_type": "workflow",
            "intent": "menu_plan",
            "lang": "zh",
            "message": reply,
            "data": {"graph_run_id": run["id"], "status": run["status"]},
            "extra_fields": {},
            "tool_results": [],
            "state_patches": [],
            "handled_by": "langgraph_bridge",
            "trace_id": None,
        },
        "state": run.get("state") or {},
        "traces": [],
        "mode": "langgraph_bridge",
    }


def create_app(data_dir: Path | None = None, recipe_store=None, runner_factory=Agents):
    local_values = _local_demo_values()
    directory = data_dir or Path(
        os.getenv("DEMO_DATA_DIR")
        or local_values.get("DEMO_DATA_DIR")
        or str(ROOT / ".demo")
    )

    @asynccontextmanager
    async def lifespan(app):
        previous = _apply_demo_environment(local_values)
        recipes = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            storage = DemoStorage(directory / "runs.sqlite")
            storage.mark_interrupted()
            recipes = recipe_store or RecipeStore(directory / "recipes.db")
            from .shared_chat import bind_recipe_store

            bind_recipe_store(recipes)
            async with AsyncSqliteSaver.from_conn_string(
                str(directory / "checkpoints.sqlite")
            ) as saver:
                app.state.engine = Engine(storage, saver, recipes, runner_factory)
                yield
                await app.state.engine.close()
        finally:
            if recipes is not None:
                recipes.close()
            _restore_environment(previous)

    app = FastAPI(title="CookClaw · Three-agent demo", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=Path(__file__).parent / "web"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "web" / "index.html")

    @app.get("/api/config")
    async def config():
        key = os.getenv("DASHSCOPE_API_KEY", "")
        return {"live_available": bool(key and key != "replace_me"),
                "recipes_ready": (directory / "recipes.db").exists(),
                "model": os.getenv("DEMO_MODEL", "qwen3.7-flash-2026-07-15"), "rules": RULES}

    @app.post("/api/chat")
    async def chat(body: ChatMessage):
        if not os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY") == "replace_me":
            raise HTTPException(409, "自然语言聊天需要配置 DASHSCOPE_API_KEY。高级面板仍可运行规则演练。")
        from app.agent.participle_agent import handle_web_turn
        from app.conversation.service import get_conversation_service
        from app.observability.trace import collect_turn_traces
        thread_id = body.thread_id or "web:" + uuid.uuid4().hex
        engine = app.state.engine

        # 图运行与聊天线程绑定：后续“确认/取消”不再需要用户复制 run_id。
        linked = engine.chat_run(thread_id)
        if linked and linked["status"] == "running":
            return _graph_response(
                linked,
                thread_id=thread_id,
                reply="三位 Agent 正在协作中，我会先完成检索、饮食分析和菜单校验。",
            )
        if linked and linked["status"] in {"awaiting_confirmation", "completed", "cancelled", "blocked"}:
            adjustment = slot_adjustment_from_chat(body.message)
            if adjustment is not None:
                revised = engine.revise_chat_slot(
                    thread_id,
                    adjustment.slot_index,
                    adjustment.request,
                )
                if revised is None:
                    return _graph_response(
                        linked,
                        thread_id=thread_id,
                        reply="我没有找到这道菜对应的菜单槽位，请回复例如“第二道换掉”。",
                    )
                return _graph_response(
                    revised,
                    thread_id=thread_id,
                    reply=f"已提交第 {adjustment.slot_index + 1} 道菜的局部修订，其余菜品和汤保持不变；完成后请重新确认。",
                )
            if requests_unspecified_adjustment(body.message):
                return _graph_response(
                    linked,
                    thread_id=thread_id,
                    reply="可以调整，请告诉我是第几道，例如“第二道换清淡一点”。",
                )
        if linked and linked["status"] == "awaiting_confirmation":
            decision = decision_from_chat(body.message)
            if decision is None:
                return _graph_response(
                    linked,
                    thread_id=thread_id,
                    reply="方案已经准备好了。请回复“确认”执行 Mock 流程，或回复“取消”结束本轮。",
                )
            await engine.decide(linked["id"], linked["version"], decision)
            updated = engine.get(linked["id"])
            return _graph_response(
                updated,
                thread_id=thread_id,
                reply=("已收到确认，正在执行 Mock 流程。" if decision
                        else "已取消本次 Mock 执行，方案保留在当前任务中。"),
            )
        try:
            with collect_turn_traces() as traces:
                async with asyncio.timeout(150):
                    envelope = await handle_web_turn(body.message, thread_id)
                    state = await get_conversation_service().load_task_state(thread_id)
                    # 只有用户明确请求多 Agent 时才进入图；普通推荐仍留在共享核心。
                    if requests_graph(body.message):
                        brief = brief_from_task_state(state, mode=(
                            "rehearsal" if body.mode == "rehearsal" else "live"
                        ))
                        if brief is not None:
                            run_id = engine.start_chat_graph(thread_id, brief)
                            run = engine.get(run_id)
                            response = _graph_response(
                                run,
                                thread_id=thread_id,
                                reply="已把当前需求交给三位 Agent：食谱研究、饮食分析和菜单规划。完成后我会让你确认是否执行。",
                            )
                            response["state"] = state.to_dict()
                            response["traces"] = traces
                            return response
            return {"thread_id": thread_id, "reply": envelope.message,
                    "response": envelope.model_dump(), "state": state.to_dict(),
                    "traces": traces, "mode": "shared_runtime"}
        except Exception as exc:
            logger.warning("Shared chat failed: %s", type(exc).__name__)
            raise HTTPException(503, "本轮对话未完成，请检查模型及本地会话配置后重试。") from None

    @app.post("/api/runs", status_code=202)
    async def start(brief: Brief):
        engine = app.state.engine
        engine.check_mode(brief)
        if len(engine.jobs) >= 6:
            raise HTTPException(429, "演示任务较多，请稍后再试")
        run_id = uuid.uuid4().hex
        engine.storage.create(run_id, brief.model_dump())
        engine.launch(run_id, {"brief": brief.model_dump(), "version": 1, "revision_count": 0, "research_count": 0})
        return {"id": run_id}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        return app.state.engine.get(run_id)

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, after: int = Query(default=0, ge=0)):
        engine = app.state.engine
        engine.get(run_id)
        async def stream():
            cursor = after
            while True:
                for event in engine.storage.events(run_id, cursor):
                    cursor = event["id"]
                    yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                if engine.get(run_id)["status"] != "running":
                    # Drain any event committed between the previous read and status check.
                    for event in engine.storage.events(run_id, cursor):
                        yield f"id: {event['id']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    yield "event: done\ndata: {}\n\n"
                    return
                yield ": keepalive\n\n"
                await asyncio.sleep(0.3)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    @app.post("/api/runs/{run_id}/decision", status_code=202)
    async def decision(run_id: str, body: Decision):
        engine = app.state.engine
        return await engine.decide(run_id, body.version, body.approved)

    @app.post("/api/runs/{run_id}/revise", status_code=202)
    async def revise(run_id: str, body: Revision):
        engine = app.state.engine
        async with engine.locks.setdefault(run_id, asyncio.Lock()):
            row = engine.get(run_id)
            if row["version"] != body.version or row["status"] == "running" or run_id in engine.jobs:
                raise HTTPException(409, "任务正在运行或版本已变化")
            engine.check_mode(body.brief)
            engine.storage.revise(run_id, body.brief.model_dump())
            engine.launch(run_id, {"brief": body.brief.model_dump(), "version": row["version"] + 1,
                                  "revision_count": 0, "research_count": 0})
            return {"id": run_id}

    @app.post("/api/runs/{run_id}/retry", status_code=202)
    async def retry(run_id: str):
        engine = app.state.engine
        async with engine.locks.setdefault(run_id, asyncio.Lock()):
            row = engine.get(run_id)
            if row["status"] not in {"error", "interrupted"} or run_id in engine.jobs:
                raise HTTPException(409, "当前任务不需要重试")
            engine.check_mode(Brief(**row["brief"]))
            engine.storage.update(run_id, "running")
            engine.launch(run_id, None)
            return {"id": run_id}

    return app


app = create_app()
