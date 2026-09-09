#!/usr/bin/env python3
"""从 PostgreSQL 事实源重建 Milvus 用户记忆派生索引。

输出只包含画像数和事实数，不打印 profile key、用户 ID 或事实正文。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.conversation.milvus_memory_index import MilvusProfileMemoryIndex
from app.conversation.postgres_profile_store import (
    PostgresChannelUserProfileStore,
    parse_profile_key,
)
from app.conversation.profile_facts import active_general_profile_facts


def _long_term_memory(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        decoded = json.loads(value)
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def _row_profile_key(row) -> str:
    channel = str(row["channel"] or "")
    user_id = str(row["channel_user_id"] or "")
    if channel == "weixin":
        account_id = str(row["channel_account_id"] or "default")
        return f"profile:weixin:{account_id}:{user_id}"
    return f"profile:{channel}:{user_id}"


async def _rows(store: PostgresChannelUserProfileStore, profile_key: str | None):
    pool = await store._get_pool()
    if profile_key:
        channel, account_id, user_id = parse_profile_key(profile_key)
        return await pool.fetch(
            """
            SELECT channel, channel_account_id, channel_user_id, long_term_memory
            FROM public.channel_users
            WHERE channel = $1
              AND channel_account_id = $2
              AND channel_user_id = $3
              AND status = 'active'
            """,
            channel,
            account_id,
            user_id,
        )
    return await pool.fetch(
        """
        SELECT channel, channel_account_id, channel_user_id, long_term_memory
        FROM public.channel_users
        WHERE status = 'active'
        ORDER BY id
        """
    )


async def reindex(*, profile_key: str | None, dry_run: bool) -> dict[str, object]:
    store = PostgresChannelUserProfileStore.from_env()
    index = MilvusProfileMemoryIndex.from_env(require_server=True)
    profile_count = 0
    fact_count = 0
    try:
        rows = await _rows(store, profile_key)
        if not dry_run:
            await index.initialize()
            await index.permission_probe()
        for row in rows:
            current_profile_key = _row_profile_key(row)
            long_term = _long_term_memory(row["long_term_memory"])
            facts = active_general_profile_facts(
                list(long_term.get("facts") or []),
                limit=50,
            )
            profile_count += 1
            fact_count += len(facts)
            if dry_run:
                continue
            # PostgreSQL 是事实源；先删除该匿名画像在 Milvus 的全部旧副本，
            # 再只写回当前有效事实，从而清掉过期或历史删除残留。
            await index.delete_profile(current_profile_key)
            await index.upsert(current_profile_key, facts)
        return {
            "ok": True,
            "dry_run": dry_run,
            "profiles": profile_count,
            "active_facts": fact_count,
        }
    finally:
        await index.close()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the derived Milvus user-memory index from PostgreSQL",
    )
    parser.add_argument(
        "--profile-key",
        help="仅重建一个内部 profile key；输出不会回显该值",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计，不连接或修改 Milvus",
    )
    args = parser.parse_args()
    result = asyncio.run(
        reindex(
            profile_key=str(args.profile_key or "").strip() or None,
            dry_run=bool(args.dry_run),
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
