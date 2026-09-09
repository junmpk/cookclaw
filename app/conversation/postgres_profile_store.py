"""PostgreSQL 分通道长期画像存储。"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

import asyncpg

from app.conversation.models import ConversationMemory
from app.conversation.store import ConversationStoreConflict
from app.core.env import getenv_resolved


def parse_profile_key(profile_key: str) -> tuple[str, str, str]:
    """把 profile key 转成 channel、channel_account_id、channel_user_id。"""
    parts = str(profile_key or "").split(":")
    if len(parts) < 3 or parts[0] != "profile":
        raise ValueError("profile key must start with profile:<channel>:")
    channel = parts[1]
    if channel == "weixin":
        # 兼容升级前的 profile:weixin:<user_id>；旧数据没有 bot account，
        # 统一归到 default，新产生的数据始终使用四段 key。
        if len(parts) == 3:
            return channel, "default", parts[2]
        if len(parts) < 4:
            raise ValueError("weixin profile key requires user")
        return channel, parts[2] or "default", ":".join(parts[3:])
    return channel, "default", ":".join(parts[2:])


def _json_object(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        decoded = json.loads(value)
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


class PostgresChannelUserProfileStore:
    """把 profile:* ConversationMemory 映射到 public.channel_users。"""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        sslmode: str = "prefer",
        min_pool_size: int = 1,
        max_pool_size: int = 5,
    ) -> None:
        self._connect_kwargs = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "ssl": sslmode,
            "command_timeout": float(os.getenv("POSTGRES_COMMAND_TIMEOUT_SECONDS", "10")),
            "server_settings": {"application_name": "cookclaw-profile-store"},
        }
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "PostgresChannelUserProfileStore":
        profile = (
            getenv_resolved("DATA_SERVICE_PROFILE", "local") or "local"
        ).strip().lower()
        prefix = "POSTGRES_SERVER_" if profile == "server" else "POSTGRES_"
        host = (getenv_resolved(f"{prefix}HOST", "") or "").strip()
        if not host:
            raise ValueError(f"{prefix}HOST is required for PostgreSQL profile storage")
        return cls(
            host=host,
            port=int(getenv_resolved(f"{prefix}PORT", "5432") or "5432"),
            user=getenv_resolved(f"{prefix}USER", "") or "",
            password=getenv_resolved(f"{prefix}PASSWORD", "") or "",
            database=getenv_resolved(f"{prefix}DB", "") or "",
            sslmode=getenv_resolved(f"{prefix}SSLMODE", "prefer") or "prefer",
            min_pool_size=max(1, int(os.getenv("POSTGRES_POOL_MIN_SIZE", "1"))),
            max_pool_size=max(1, int(os.getenv("POSTGRES_POOL_MAX_SIZE", "5"))),
        )

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(
                    **self._connect_kwargs,
                    min_size=self._min_pool_size,
                    max_size=max(self._min_pool_size, self._max_pool_size),
                )
        return self._pool

    async def load(self, profile_key: str) -> ConversationMemory | None:
        channel, account_id, user_id = parse_profile_key(profile_key)
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT
                display_name,
                preferred_name,
                profile,
                long_term_memory,
                memory_version,
                EXTRACT(EPOCH FROM created_at) AS created_epoch,
                EXTRACT(EPOCH FROM updated_at) AS updated_epoch
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
        if row is None:
            return None
        profile = _json_object(row["profile"])
        long_term = _json_object(row["long_term_memory"])
        summary_state = long_term.get("summary_state")
        return ConversationMemory(
            thread_id=profile_key,
            channel=channel,
            user_id=user_id,
            display_name=str(row["display_name"] or "").strip() or None,
            preferred_name=str(row["preferred_name"] or "").strip() or None,
            preferences=profile,
            summary=dict(summary_state) if isinstance(summary_state, dict) else {},
            food_history=list(long_term.get("food_history") or []),
            long_term_facts=list(long_term.get("facts") or []),
            version=int(row["memory_version"] or 0),
            created_at=float(row["created_epoch"] or 0),
            updated_at=float(row["updated_epoch"] or 0),
        )

    async def save(self, memory: ConversationMemory, expires_at: float) -> None:
        values = self._write_values(memory)
        expected_version = max(0, int(memory.version) - 1)
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO public.channel_users (
                channel,
                channel_account_id,
                channel_user_id,
                display_name,
                preferred_name,
                profile,
                long_term_memory,
                memory_version,
                last_seen_at,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9, $9
            )
            ON CONFLICT (channel, channel_account_id, channel_user_id)
            DO UPDATE SET
                display_name = COALESCE(EXCLUDED.display_name, public.channel_users.display_name),
                preferred_name = EXCLUDED.preferred_name,
                profile = EXCLUDED.profile,
                long_term_memory = EXCLUDED.long_term_memory,
                memory_version = EXCLUDED.memory_version,
                last_seen_at = EXCLUDED.last_seen_at,
                updated_at = EXCLUDED.updated_at,
                status = 'active'
            WHERE public.channel_users.memory_version = $10
            RETURNING id
            """,
            *values,
            expected_version,
        )
        if row is None:
            raise ConversationStoreConflict(
                f"profile version conflict for {values[0]}:{values[1]}"
            )

    def _write_values(self, memory: ConversationMemory) -> tuple:
        channel, account_id, user_id = parse_profile_key(memory.thread_id)
        now = datetime.fromtimestamp(memory.updated_at, tz=timezone.utc)
        digest = list((memory.summary or {}).get("account_digest") or [])
        profile = dict(memory.preferences or {})
        # 可用食材是会话级条件，不能把它升级成跨会话长期画像。
        profile["available_ingredients"] = []
        long_term = {
            "facts": list(memory.long_term_facts or [])[-100:],
            "summary": "\n".join(str(item) for item in digest[-10:])[:2000],
            "summary_state": dict(memory.summary or {}),
            "food_history": list(memory.food_history or [])[-40:],
        }
        return (
            channel,
            account_id,
            user_id,
            str(memory.display_name or "").strip() or None,
            str(memory.preferred_name or "").strip() or None,
            json.dumps(profile, ensure_ascii=False),
            json.dumps(long_term, ensure_ascii=False),
            int(memory.version),
            now,
        )

    async def import_if_newer(self, memory: ConversationMemory) -> bool:
        """导入专用：只写入更高版本，不覆盖已有线上新数据。"""
        values = self._write_values(memory)
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO public.channel_users (
                channel,
                channel_account_id,
                channel_user_id,
                display_name,
                preferred_name,
                profile,
                long_term_memory,
                memory_version,
                last_seen_at,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9, $9
            )
            ON CONFLICT (channel, channel_account_id, channel_user_id)
            DO UPDATE SET
                display_name = COALESCE(EXCLUDED.display_name, public.channel_users.display_name),
                preferred_name = COALESCE(EXCLUDED.preferred_name, public.channel_users.preferred_name),
                profile = EXCLUDED.profile,
                long_term_memory = EXCLUDED.long_term_memory,
                memory_version = EXCLUDED.memory_version,
                last_seen_at = EXCLUDED.last_seen_at,
                updated_at = EXCLUDED.updated_at,
                status = 'active'
            WHERE public.channel_users.memory_version < EXCLUDED.memory_version
            RETURNING id
            """,
            *values,
        )
        return row is not None

    async def delete(self, profile_key: str) -> None:
        channel, account_id, user_id = parse_profile_key(profile_key)
        pool = await self._get_pool()
        await pool.execute(
            """
            UPDATE public.channel_users
            SET status = 'deleted', updated_at = NOW()
            WHERE channel = $1
              AND channel_account_id = $2
              AND channel_user_id = $3
            """,
            channel,
            account_id,
            user_id,
        )

    async def cleanup_expired(self, now: float | None = None) -> int:
        # 长期画像不使用短期 TTL 自动删除。
        return 0

    async def healthcheck(self) -> bool:
        pool = await self._get_pool()
        return bool(await pool.fetchval("SELECT TRUE"))

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
