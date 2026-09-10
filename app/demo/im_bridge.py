"""把 QQ、微信、WhatsApp 的共享文本入口桥接到演示 LangGraph。"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.orchestrator.turn.runtime_models import ResponseEnvelope

from .agents import Agents
from .chat_intent import (
    brief_from_task_state,
    decision_from_chat,
    requests_graph,
    requests_unspecified_adjustment,
    slot_adjustment_from_chat,
)
from .engine import Engine
from .retrieval import RecipeStore
from .storage import DemoStorage

Fallback = Callable[[], Awaitable[ResponseEnvelope]]
StateLoader = Callable[[str], Awaitable[object]]


class IMGraphBridge:
    """通道无关的 Graph 会话协调器；设备动作始终停留在 Mock。"""

    def __init__(self, engine: Engine, connection, recipes):
        self.engine = engine
        self.connection = connection
        self.recipes = recipes

    @classmethod
    async def create(
        cls,
        data_dir: Path,
        *,
        recipe_store=None,
        runner_factory=Agents,
    ) -> IMGraphBridge:
        data_dir.mkdir(parents=True, exist_ok=True)
        storage = DemoStorage(data_dir / "runs.sqlite")
        storage.mark_interrupted()
        recipes = recipe_store or RecipeStore(data_dir / "recipes.db")
        connection = await aiosqlite.connect(data_dir / "checkpoints.sqlite")
        saver = AsyncSqliteSaver(connection)
        return cls(
            Engine(storage, saver, recipes, runner_factory),
            connection,
            recipes,
        )

    async def close(self) -> None:
        await self.engine.close()
        self.recipes.close()
        await self.connection.close()

    async def handle(
        self,
        question: str,
        thread_id: str,
        *,
        fallback: Fallback,
        state_loader: StateLoader,
    ) -> ResponseEnvelope:
        channel = str(thread_id or "").split(":", 1)[0].lower()
        if channel not in {"qq", "weixin", "whatsapp"}:
            return await fallback()

        linked = self.engine.chat_run(thread_id)
        if linked and linked["status"] == "running":
            linked = await self.engine.wait(linked["id"])

        if linked and linked["status"] in {
            "awaiting_confirmation",
            "completed",
            "cancelled",
            "blocked",
        }:
            adjustment = slot_adjustment_from_chat(question)
            if adjustment is not None:
                revised = self.engine.revise_chat_slot(
                    thread_id,
                    adjustment.slot_index,
                    adjustment.request,
                )
                if revised is None:
                    return self._text(
                        "没有找到对应的菜单位置，请回复例如“第二道换清淡一点”。"
                    )
                revised = await self.engine.wait(revised["id"])
                return self._run_response(revised, adjusted=True)
            if requests_unspecified_adjustment(question):
                return self._text(
                    "可以调整。请告诉我是第几道，例如“第二道换掉”或“第一道换清淡一点”。"
                )

        if linked and linked["status"] == "awaiting_confirmation":
            decision = decision_from_chat(question)
            if decision is None:
                return self._run_response(linked)
            await self.engine.decide(linked["id"], linked["version"], decision)
            updated = await self.engine.wait(linked["id"])
            return self._run_response(updated)

        envelope = await fallback()
        if not requests_graph(question):
            return envelope

        state = await state_loader(thread_id)
        mode = os.getenv("MULTI_AGENT_BRIDGE_MODE", "live").strip().lower()
        brief = brief_from_task_state(
            state,
            mode="rehearsal" if mode == "rehearsal" else "live",
        )
        if brief is None:
            return envelope
        self.engine.check_mode(brief)
        run_id = self.engine.start_chat_graph(thread_id, brief)
        return self._run_response(await self.engine.wait(run_id))

    @staticmethod
    def _text(message: str) -> ResponseEnvelope:
        return ResponseEnvelope(
            response_type="workflow",
            intent="menu_plan",
            lang="zh",
            message=message,
            handled_by="langgraph_bridge",
        )

    def _run_response(self, run: dict, *, adjusted: bool = False) -> ResponseEnvelope:
        status = str(run.get("status") or "error")
        if status == "running":
            return self._text("三位 Agent 正在协作，请稍后再发送一条消息查看结果。")
        if status == "error":
            return self._text("本轮协作没有完成，请检查食谱库、模型配置与网络后重试。")
        if status == "blocked":
            return self._text("当前真实候选无法满足约束，请调整食材、忌口或菜品数量后重试。")
        if status == "cancelled":
            return self._text("已取消本次 Mock 执行，菜单记录仍然保留。")
        if status == "completed":
            execution = dict((run.get("state") or {}).get("execution") or {})
            return self._text(
                str(execution.get("message") or "Mock 执行已经完成，未控制真实设备。")
            )

        state = dict(run.get("state") or {})
        menu = dict(state.get("menu") or {})
        catalog = {
            str(item.get("id")): item
            for item in state.get("candidates") or []
            if isinstance(item, dict) and item.get("id")
        }
        recipes = []
        for recipe_id in menu.get("recipe_ids") or []:
            recipe = dict(catalog.get(str(recipe_id)) or {})
            if not recipe:
                continue
            kind = str(recipe.get("kind") or "dish")
            recipes.append(
                {
                    "id": str(recipe.get("id") or ""),
                    "name": str(recipe.get("name") or "未知菜品"),
                    "image": str(recipe.get("image_url") or ""),
                    "ingredients": list(recipe.get("ingredients") or [])[:8],
                    "tags": [kind],
                    "menu_role": kind,
                    "source": str(recipe.get("source") or ""),
                    "detail_url": str(recipe.get("source") or ""),
                }
            )
        opening = (
            "已按你的要求调整指定菜品，其他菜单项保持不变。"
            if adjusted
            else "三位 Agent 已完成食谱研究、饮食分析和菜单校验。"
        )
        return ResponseEnvelope(
            response_type="menu_plan",
            intent="menu_plan",
            lang="zh",
            message=opening,
            data={
                "opening": opening,
                "strategy": str(menu.get("explanation") or ""),
                "recipes": recipes,
                "closing": (
                    "回复“确认”启动 Mock 执行，回复“取消”结束；"
                    "也可以说“第二道换掉”或“第一道换清淡一点”。"
                ),
                "graph_run_id": str(run.get("id") or ""),
                "graph_status": status,
            },
            handled_by="langgraph_bridge",
        )
