#!/usr/bin/env python3
"""初始化并检查 CookClaw 的 PostgreSQL / Redis 数据地基。

脚本只执行幂等 DDL 和连接健康检查，不打印密码、用户标识或聊天正文。
生产替换部署会在应用启动前调用；任何失败都会触发发布脚本自动回滚。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.conversation.postgres_profile_store import PostgresChannelUserProfileStore
from app.conversation.milvus_memory_index import MilvusProfileMemoryIndex
from app.conversation.redis_store import RedisConversationStore
from app.recipe_detail_store import PostgresRecipeDetailStore


async def initialize() -> dict[str, object]:
    profile_store = PostgresChannelUserProfileStore.from_env()
    recipe_store = PostgresRecipeDetailStore.from_env()
    redis_store = RedisConversationStore.from_env()
    general_memory_enabled = (
        os.getenv("CONVERSATION_GENERAL_MEMORY_ENABLED", "false").lower()
        == "true"
    )
    memory_index = (
        MilvusProfileMemoryIndex.from_env(require_server=True)
        if general_memory_enabled
        else None
    )
    applied: list[str] = []
    try:
        pool = await profile_store._get_pool()
        async with pool.acquire() as connection:
            for path in sorted((PROJECT_ROOT / "db" / "migrations").glob("*.sql")):
                # macOS 归档在 Linux 解压时可能生成 AppleDouble 伴生文件
                # `._001_*.sql`；它不是 SQL，也不应进入迁移序列。
                if path.name.startswith("._"):
                    continue
                await connection.execute(path.read_text(encoding="utf-8"))
                applied.append(path.name)

        postgres_profile_ready = await profile_store.healthcheck()
        postgres_recipe_ready = await recipe_store.healthcheck()
        redis_ready = await redis_store.healthcheck()
        if not all((postgres_profile_ready, postgres_recipe_ready, redis_ready)):
            raise RuntimeError("one or more data-store health checks returned false")
        memory_index_ready: bool | str = "disabled"
        memory_permissions_ready: bool | str = "disabled"
        if memory_index is not None:
            # initialize 会在集合不存在时创建独立集合；探针不含用户数据，成功
            # upsert 后立即 delete，用于在服务重启前验证实际写删权限。
            await memory_index.initialize()
            memory_index_ready = await memory_index.healthcheck()
            if not memory_index_ready:
                raise RuntimeError("Milvus user-memory collection healthcheck returned false")
            await memory_index.permission_probe()
            memory_permissions_ready = True
        return {
            "ok": True,
            "migrations": applied,
            "postgres_channel_users": postgres_profile_ready,
            "postgres_recipe_details": postgres_recipe_ready,
            "redis": redis_ready,
            "milvus_user_memory": memory_index_ready,
            "milvus_user_memory_permissions": memory_permissions_ready,
        }
    finally:
        if memory_index is not None:
            await memory_index.close()
        await redis_store.close()
        await recipe_store.close()
        await profile_store.close()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(initialize()), ensure_ascii=False, sort_keys=True))
