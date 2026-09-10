"""LangGraph 演示运行引擎。

Web 演示和 IM bridge 共用这一层；模块本身不加载 .env，也不修改进程环境。
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import HTTPException
from langgraph.types import Command

from .agents import Agents
from .graph import build_graph
from .models import Brief

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, storage, saver, recipes, runner_factory=Agents):
        self.storage, self.saver, self.recipes = storage, saver, recipes
        self.runner_factory = runner_factory
        self.jobs: dict[str, asyncio.Task] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.capacity = asyncio.Semaphore(3)

    def get(self, run_id):
        row = self.storage.get(run_id)
        if row is None:
            raise HTTPException(404, "任务不存在")
        return row

    def chat_run(self, thread_id: str):
        run_id = self.storage.chat_run_id(thread_id)
        return self.storage.get(run_id) if run_id else None

    def start_chat_graph(
        self,
        thread_id: str,
        brief: Brief,
        replacement: dict | None = None,
    ) -> str:
        current = self.chat_run(thread_id)
        if (
            current
            and current["status"] in {"running", "awaiting_confirmation"}
            and not replacement
        ):
            return current["id"]
        run_id = uuid.uuid4().hex
        self.storage.create(run_id, brief.model_dump())
        self.storage.link_chat(thread_id, run_id)
        graph_input = {
            "brief": brief.model_dump(),
            "version": 1,
            "revision_count": 0,
            "research_count": 0,
        }
        if replacement:
            graph_input["replacement"] = replacement
        self.launch(run_id, graph_input)
        return run_id

    def revise_chat_slot(
        self,
        thread_id: str,
        slot_index: int,
        revision_request: str = "",
    ) -> dict | None:
        current = self.chat_run(thread_id)
        if not current:
            return None
        state = current.get("state") or {}
        menu = state.get("menu") or {}
        previous_ids = [str(item) for item in menu.get("recipe_ids") or []]
        if slot_index < 0 or slot_index >= len(previous_ids):
            return None
        brief = Brief(**dict(current.get("brief") or {}))
        replacement = {
            "slot_index": slot_index,
            "previous_ids": previous_ids,
            "exclusions": list(brief.exclusions),
            "request": str(revision_request or "").strip()[:300],
        }
        run_id = self.start_chat_graph(thread_id, brief, replacement=replacement)
        return self.get(run_id)

    async def decide(self, run_id: str, version: int, approved: bool):
        async with self.locks.setdefault(run_id, asyncio.Lock()):
            row = self.get(run_id)
            if row["version"] != version:
                raise HTTPException(409, "方案已修改，请确认最新版本")
            if row["status"] == "completed":
                return row
            if row["status"] != "awaiting_confirmation" or run_id in self.jobs:
                raise HTTPException(409, "当前没有可确认的方案")
            self.storage.update(run_id, "running")
            self.launch(
                run_id,
                Command(resume={"approved": approved, "version": version}),
            )
            return {"id": run_id}

    def check_mode(self, brief):
        import os

        key = os.getenv("DASHSCOPE_API_KEY")
        if brief.mode == "live" and (not key or key == "replace_me"):
            raise HTTPException(
                409,
                "请在本地 .env.demo 配置 DASHSCOPE_API_KEY，或选择规则演练。",
            )

    def launch(self, run_id, graph_input):
        task = asyncio.create_task(self.run(run_id, graph_input))
        self.jobs[run_id] = task

        def cleanup(finished):
            if self.jobs.get(run_id) is finished:
                self.jobs.pop(run_id, None)

        task.add_done_callback(cleanup)

    async def wait(self, run_id: str, timeout: float = 150.0) -> dict:
        """等待当前后台图到达稳定状态，供同步 IM 回调返回最终卡片。"""
        task = self.jobs.get(run_id)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except TimeoutError:
                return self.get(run_id)
        return self.get(run_id)

    async def run(self, run_id, graph_input):
        row = self.get(run_id)

        async def emit(role, kind, data):
            self.storage.event(run_id, row["version"], role, kind, data)

        graph = build_graph(
            self.runner_factory(self.recipes, emit, row["brief"]["mode"]),
            self.saver,
            self.storage,
            run_id,
            emit,
        )
        config = {
            "configurable": {"thread_id": f"{run_id}:v{row['version']}"},
            "recursion_limit": 40,
        }
        try:
            async with self.capacity, asyncio.timeout(360):
                await emit(
                    "system",
                    "run_start",
                    {"mode": row["brief"]["mode"], "version": row["version"]},
                )
                await graph.ainvoke(graph_input, config=config)
                snapshot = await graph.aget_state(config)
                state = dict(snapshot.values)
                status = (
                    "awaiting_confirmation"
                    if snapshot.next
                    else state.get("status", "blocked")
                )
                self.storage.update(run_id, status, state)
                await emit("system", "run_end", {"status": status})
        except asyncio.CancelledError:
            self.storage.update(run_id, "interrupted")
            raise
        # Provider、检索和图执行错误统一转成安全状态，不能让后台任务静默退出。
        except Exception as exc:  # noqa: BLE001
            logger.warning("Demo run failed: %s", type(exc).__name__)
            self.storage.update(run_id, "error")
            await emit(
                "system",
                "error",
                {
                    "type": type(exc).__name__,
                    "message": "本次流程未完成。请检查食谱库、模型配置与网络；可从检查点重试。",
                },
            )

    async def close(self):
        jobs = list(self.jobs.values())
        for task in jobs:
            task.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
