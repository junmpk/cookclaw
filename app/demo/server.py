"""uvicorn app.demo.server:app --host 127.0.0.1 --port 8010 (single worker)."""
import asyncio
import base64
import json
import logging
import os
import re
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
    brief_from_search_request,
    decision_from_chat,
    graph_routing_decision,
    requests_unspecified_adjustment,
    slot_adjustment_from_chat,
)
from .engine import Engine
from .knowledge import RULES
from .models import Brief
from .retrieval import build_recipe_store, recipes_from_search_result
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


class ImageChatMessage(BaseModel):
    data_url: str = Field(min_length=32, max_length=12_000_000, repr=False)
    caption: str = Field(default="", max_length=500)
    thread_id: str | None = Field(default=None, pattern=r"^web:[0-9a-f]{32}$")

    @field_validator("data_url")
    @classmethod
    def valid_image_data_url(cls, value: str) -> str:
        matched = re.fullmatch(
            r"data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=\r\n]+)",
            value,
            flags=re.IGNORECASE,
        )
        if not matched:
            raise ValueError("只支持 JPEG、PNG 或 WebP 图片")
        try:
            payload = base64.b64decode(
                re.sub(r"\s+", "", matched.group(2)),
                validate=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("图片内容无法解码") from exc
        if not payload or len(payload) > 8 * 1024 * 1024:
            raise ValueError("图片必须小于 8MB")
        return value


def _web_thread_id(value: str | None) -> str:
    return value or "web:" + uuid.uuid4().hex


def _graph_response(
    run: dict,
    *,
    thread_id: str,
    reply: str,
    routing: dict | None = None,
) -> dict:
    owner_thread_id = str(run.get("thread_id") or thread_id)
    profile = dict((run.get("state") or {}).get("complexity_profile") or {})
    routing_data = routing or (
        {
            "use_graph": True,
            "score": profile.get("total_score", 0),
            "trigger": "complexity",
            "reasons": profile.get("reasons", []),
            "profile": profile,
        }
        if profile
        else None
    )
    return {
        "thread_id": owner_thread_id,
        "reply": reply,
        "graph_run_id": run["id"],
        "graph_status": run["status"],
        "response": {
            "schema_version": "response_envelope_v1",
            "response_type": "workflow",
            "intent": "menu_plan",
            "lang": "zh",
            "message": reply,
            "data": {
                "graph_run_id": run["id"],
                "thread_id": owner_thread_id,
                "version": run.get("version", 1),
                "status": run["status"],
                "workflow_identity": run.get("workflow_identity") or {},
                "metrics": run.get("metrics") or {},
                **({"routing": routing_data} if routing_data else {}),
            },
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
        previous_search_backend = os.environ.get("RECIPE_SEARCH_BACKEND")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            storage = DemoStorage(directory / "runs.sqlite")
            storage.mark_interrupted()
            recipes = recipe_store or build_recipe_store(
                directory / "recipes.db",
                backend=os.getenv("DEMO_RECIPE_BACKEND", "auto"),
            )
            backend_name = str(getattr(recipes, "backend_name", "injected"))
            if backend_name != "injected":
                os.environ["RECIPE_SEARCH_BACKEND"] = (
                    "public_demo"
                    if backend_name == "public_rehearsal"
                    else "hybrid"
                )
            from .shared_chat import bind_recipe_store

            bind_recipe_store(recipes)
            async with AsyncSqliteSaver.from_conn_string(
                str(directory / "checkpoints.sqlite")
            ) as saver:
                app.state.engine = Engine(storage, saver, recipes, runner_factory)
                app.state.recipe_backend = backend_name
                yield
                await app.state.engine.close()
        finally:
            if recipes is not None:
                recipes.close()
            if previous_search_backend is None:
                os.environ.pop("RECIPE_SEARCH_BACKEND", None)
            else:
                os.environ["RECIPE_SEARCH_BACKEND"] = previous_search_backend
            _restore_environment(previous)

    app = FastAPI(title="CookClaw · Multi-agent demo", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=Path(__file__).parent / "web"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "web" / "index.html")

    @app.get("/api/config")
    async def config():
        key = os.getenv("DASHSCOPE_API_KEY", "")
        backend = str(getattr(app.state, "recipe_backend", "unknown"))
        return {"live_available": bool(key and key != "replace_me"),
                "recipes_ready": backend == "hybrid_rag" or (directory / "recipes.db").exists(),
                "recipe_backend": backend,
                "workflow_contract": {
                    "status_owner": "graph_run",
                    "checkpoint_owner": "langgraph",
                    "execution_provider": "mock",
                    "recovery": "same_run_same_version_same_checkpoint",
                },
                "model": os.getenv("DEMO_MODEL", "qwen3.7-flash-2026-07-15"), "rules": RULES}

    @app.post("/api/chat")
    async def chat(body: ChatMessage):
        if not os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY") == "replace_me":
            raise HTTPException(409, "自然语言聊天需要配置 DASHSCOPE_API_KEY。高级面板仍可运行规则演练。")
        from app.agent.participle_agent import handle_web_turn
        from app.conversation.service import get_conversation_service
        from app.observability.trace import collect_turn_traces
        from app.orchestrator.turn.runtime_models import ResponseEnvelope
        thread_id = _web_thread_id(body.thread_id)
        engine = app.state.engine

        # 图运行与聊天线程绑定：后续“确认/取消”不再需要用户复制 run_id。
        linked = engine.chat_run(thread_id)
        if linked and linked["status"] == "running":
            selected = list(
                (linked.get("state") or {})
                .get("complexity_profile", {})
                .get("selected_agents", [])
            )
            return _graph_response(
                linked,
                thread_id=thread_id,
                reply=f"多 Agent 团队正在协作中，本轮已动态启用 {len(selected) or 3} 个专业角色。",
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
                handoff: dict = {}

                async def workflow_handoff(_runtime, outcome):
                    """由统一 Turn Runtime 在 Recipe handler 前交接复杂菜单。"""
                    if outcome.kind != "menu_plan" or outcome.search_request is None:
                        return None
                    brief = brief_from_search_request(
                        outcome.search_request,
                        mode=(
                            "rehearsal"
                            if body.mode == "rehearsal"
                            else "live"
                        ),
                    )
                    routing = graph_routing_decision(body.message, brief)
                    if brief is None or not routing.use_graph:
                        return None
                    candidates = recipes_from_search_result(outcome.search_result)
                    if not candidates:
                        # 无真实检索证据时拒绝接管，后续 Recipe handler 按原有
                        # fail-closed 规则返回，Graph 不制造候选。
                        return None
                    run_id = engine.start_chat_graph(
                        thread_id,
                        brief,
                        seed_candidates=candidates,
                    )
                    selected = list(
                        routing.profile.selected_agents
                        if routing.profile
                        else []
                    )
                    reply = (
                        f"已识别为复杂配餐任务，自动组建 {len(selected)} 个 Agent 的协作团队。"
                        "右侧会展示选角依据、并行时间线和共享状态；完成后请确认是否执行。"
                        if routing.trigger == "complexity"
                        else
                        f"已按你的要求组建 {len(selected)} 个 Agent 的协作团队。"
                        "完成后我会让你确认是否执行。"
                    )
                    handoff.update({
                        "run_id": run_id,
                        "routing": routing.public_data(),
                    })
                    return ResponseEnvelope(
                        response_type="workflow",
                        intent="menu_plan",
                        lang="zh",
                        message=reply,
                        data={
                            "graph_run_id": run_id,
                            "thread_id": thread_id,
                            "routing": routing.public_data(),
                        },
                        handled_by="workflow_handoff",
                    )

                async with asyncio.timeout(150):
                    envelope = await handle_web_turn(
                        body.message,
                        thread_id,
                        workflow_handoff=workflow_handoff,
                    )
                    if envelope.handled_by == "workflow_handoff":
                        run = engine.get(str(handoff["run_id"]))
                        response = _graph_response(
                            run,
                            thread_id=thread_id,
                            reply=envelope.message,
                            routing=handoff.get("routing"),
                        )
                        response["response"] = envelope.model_dump()
                        response["traces"] = traces
                        return response
                    state = await get_conversation_service().load_task_state(
                        thread_id
                    )
            return {"thread_id": thread_id, "reply": envelope.message,
                    "response": envelope.model_dump(), "state": state.to_dict(),
                    "traces": traces, "mode": "shared_runtime"}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Shared chat failed: %s", type(exc).__name__)
            raise HTTPException(503, "本轮对话未完成，请检查模型及本地会话配置后重试。") from None

    @app.post("/api/chat/image")
    async def chat_image(body: ImageChatMessage):
        if not os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY") == "replace_me":
            raise HTTPException(409, "图片识别需要配置 DASHSCOPE_API_KEY。")
        from app.conversation.service import get_conversation_service
        from app.main import _handle_image_media
        from app.observability.trace import collect_turn_traces

        thread_id = _web_thread_id(body.thread_id)
        try:
            with collect_turn_traces() as traces:
                async with asyncio.timeout(150):
                    result = await _handle_image_media(
                        [body.data_url],
                        chat_id=thread_id,
                        user_text=body.caption,
                    )
                    state = await get_conversation_service().load_task_state(thread_id)
            public = (
                result.get("response")
                if isinstance(result, dict) and isinstance(result.get("response"), dict)
                else {}
            )
            reply = str(public.get("message") or result or "图片已经处理。")
            response = {
                "schema_version": "response_envelope_v1",
                "response_type": str(public.get("type") or "text"),
                "intent": public.get("intent") or "image_ingredients",
                "lang": public.get("lang") or "zh",
                "message": reply,
                "data": public.get("data") or {},
                "extra_fields": {},
                "tool_results": [],
                "state_patches": [],
                "handled_by": "image_handler",
                "trace_id": traces[-1].get("trace_id") if traces else None,
            }
            return {
                "thread_id": thread_id,
                "reply": reply,
                "response": response,
                "state": state.to_dict(),
                "traces": traces,
                "image_context": result.get("image_context") if isinstance(result, dict) else None,
                "mode": "shared_runtime",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Shared image chat failed: %s", type(exc).__name__)
            raise HTTPException(
                503,
                "图片识别未完成，请检查视觉模型、食谱库和网络后重试。",
            ) from None

    @app.post("/api/runs", status_code=202)
    async def start(brief: Brief):
        engine = app.state.engine
        engine.check_mode(brief)
        if len(engine.jobs) >= 6:
            raise HTTPException(429, "演示任务较多，请稍后再试")
        run_id = uuid.uuid4().hex
        thread_id = _web_thread_id(None)
        engine.storage.create(
            run_id,
            brief.model_dump(),
            thread_id=thread_id,
        )
        row = engine.storage.get(run_id)
        graph_input = engine.initial_input(
            brief,
            identity=engine.workflow_identity(row or {}),
        )
        engine.storage.update(run_id, "running", graph_input)
        engine.launch(run_id, graph_input)
        return {"id": run_id, "thread_id": thread_id}

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
            revised = engine.storage.get(run_id)
            graph_input = engine.initial_input(
                body.brief,
                identity=engine.workflow_identity(revised or {}),
                version=row["version"] + 1,
            )
            engine.storage.update(run_id, "running", graph_input)
            engine.launch(run_id, graph_input)
            return {"id": run_id}

    @app.post("/api/runs/{run_id}/retry", status_code=202)
    async def retry(run_id: str):
        engine = app.state.engine
        async with engine.locks.setdefault(run_id, asyncio.Lock()):
            row = engine.get(run_id)
            if row["status"] not in {"error", "interrupted"} or run_id in engine.jobs:
                raise HTTPException(409, "当前任务不需要重试")
            engine.check_mode(Brief(**row["brief"]))
            attempt = engine.retry(run_id)
            return {"id": run_id, "attempt": attempt}

    return app


app = create_app()
