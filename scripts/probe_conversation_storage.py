#!/usr/bin/env python3
"""验证 ConversationService 的 Redis 短期会话 + PG 跨会话画像链路。

仅写入随机探针账号，并在 finally 中物理清理探针数据。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.conversation.postgres_profile_store import PostgresChannelUserProfileStore
from app.conversation.redis_store import RedisConversationStore
from app.conversation.service import build_qq_thread_id, get_conversation_service


async def probe() -> dict[str, bool]:
    probe_id = f"storage-probe-{uuid.uuid4().hex}"
    first_thread = build_qq_thread_id("group", "storage-probe", probe_id)
    second_thread = build_qq_thread_id("dm", probe_id, probe_id)
    service = get_conversation_service()
    if not isinstance(service.store, RedisConversationStore):
        raise RuntimeError("CONVERSATION_STORE did not select Redis")
    if not isinstance(service.profile_store, PostgresChannelUserProfileStore):
        raise RuntimeError("PROFILE_STORE did not select PostgreSQL")
    redis_store = service.store
    profile_store = service.profile_store
    short_roundtrip = False
    cross_thread_profile = False
    fact_trace = False
    try:
        health = await asyncio.wait_for(
            service.healthcheck(),
            timeout=float(
                os.getenv("CONVERSATION_STORAGE_HEALTH_TIMEOUT_SECONDS", "10")
            ),
        )
        if not all(health.values()):
            raise RuntimeError(
                f"conversation storage healthcheck returned false: {health}"
            )
        await service.append_turn(
            first_thread,
            "user",
            "我喜欢蒜香，也不吃香菜",
            channel="qq",
            user_id=probe_id,
        )
        await service.append_turn(
            first_thread,
            "assistant",
            "记住了。",
            channel="qq",
            user_id=probe_id,
        )
        loaded = await redis_store.load(first_thread)
        short_roundtrip = bool(
            loaded
            and [turn.role for turn in loaded.recent_turns] == ["user", "assistant"]
        )

        # 创建第二个短会话，再读取同一通道账号的 PG 长期画像。
        await service.append_turn(
            second_thread,
            "user",
            "今天吃什么",
            channel="qq",
            user_id=probe_id,
        )
        context = await service.account_memory_context(second_thread)
        cross_thread_profile = (
            "蒜香" in context["preferences"]["likes"]
            and "香菜" in context["preferences"]["dislikes"]
        )
        profile = await profile_store.load(f"profile:qq:{probe_id}")
        active_facts = {
            (item.get("type"), item.get("value"))
            for item in (profile.long_term_facts if profile else [])
            if item.get("status") == "active"
        }
        fact_trace = (
            ("preference_like", "蒜香") in active_facts
            and ("preference_dislike", "香菜") in active_facts
        )
        if not all((short_roundtrip, cross_thread_profile, fact_trace)):
            raise RuntimeError("conversation storage probe assertion failed")
        return {
            "short_roundtrip": short_roundtrip,
            "cross_thread_profile": cross_thread_profile,
            "fact_trace": fact_trace,
            "cleaned": True,
        }
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors: list[Exception] = []
        for thread_id in (first_thread, second_thread):
            try:
                await redis_store.delete(thread_id)
            except Exception as exc:
                cleanup_errors.append(exc)
        try:
            pool = await profile_store._get_pool()
            await pool.execute(
                """
                DELETE FROM public.channel_users
                WHERE channel = 'qq'
                  AND channel_account_id = 'default'
                  AND channel_user_id = $1
                """,
                probe_id,
            )
        except Exception as exc:
            cleanup_errors.append(exc)
        try:
            await service.close()
        except Exception as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            message = (
                "conversation storage probe cleanup failed: "
                + ", ".join(type(exc).__name__ for exc in cleanup_errors)
            )
            if primary_error is None:
                raise RuntimeError(message) from cleanup_errors[0]
            print(message, file=sys.stderr)


if __name__ == "__main__":
    print(json.dumps(asyncio.run(probe()), ensure_ascii=False, sort_keys=True))
