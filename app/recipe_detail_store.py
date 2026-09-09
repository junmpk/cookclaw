"""PostgreSQL 菜谱详情缓存：按 recipe_id + language 保存真实接口详情。"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime

import asyncpg

from app.core.env import getenv_resolved


class PostgresRecipeDetailStore:
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
            "command_timeout": float(
                os.getenv("POSTGRES_COMMAND_TIMEOUT_SECONDS", "10")
            ),
            "server_settings": {"application_name": "cookclaw-recipe-detail-store"},
        }
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "PostgresRecipeDetailStore":
        profile = (
            getenv_resolved("DATA_SERVICE_PROFILE", "local") or "local"
        ).strip().lower()
        prefix = "POSTGRES_SERVER_" if profile == "server" else "POSTGRES_"
        host = (getenv_resolved(f"{prefix}HOST", "") or "").strip()
        if not host:
            raise ValueError(f"{prefix}HOST is required for recipe detail storage")
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

    async def load(
        self,
        recipe_id: str,
        language: str,
        *,
        max_age_seconds: int | None = None,
    ) -> dict | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT detail, raw_payload, source_updated_at, fetched_at
            FROM public.recipe_details
            WHERE recipe_id = $1
              AND language = $2
              AND (
                    $3::INTEGER IS NULL
                    OR fetched_at >= NOW() - make_interval(secs => $3::INTEGER)
              )
            """,
            str(recipe_id),
            "en" if language == "en" else "zh",
            max_age_seconds,
        )
        if row is None:
            return None
        return {
            "detail": dict(row["detail"]),
            "raw_payload": dict(row["raw_payload"]),
            "source_updated_at": row["source_updated_at"],
            "fetched_at": row["fetched_at"],
        }

    async def save(self, detail: dict, raw_payload: dict) -> None:
        recipe_id = str(detail.get("recipe_id") or "").strip()
        language = "en" if detail.get("language") == "en" else "zh"
        if not recipe_id:
            raise ValueError("recipe detail is missing recipe_id")
        source_updated_at = detail.get("source_updated_at")
        parsed_source_updated_at = (
            datetime.fromisoformat(source_updated_at)
            if isinstance(source_updated_at, str) and source_updated_at
            else None
        )
        pool = await self._get_pool()
        await pool.execute(
            """
            INSERT INTO public.recipe_details (
                recipe_id,
                language,
                name,
                detail,
                raw_payload,
                source_updated_at,
                fetched_at,
                updated_at
            ) VALUES (
                $1, $2, $3, $4::jsonb, $5::jsonb, $6, NOW(), NOW()
            )
            ON CONFLICT (recipe_id, language)
            DO UPDATE SET
                name = EXCLUDED.name,
                detail = EXCLUDED.detail,
                raw_payload = EXCLUDED.raw_payload,
                source_updated_at = EXCLUDED.source_updated_at,
                fetched_at = NOW(),
                updated_at = NOW()
            """,
            recipe_id,
            language,
            str(detail.get("name") or ""),
            json.dumps(detail, ensure_ascii=False),
            json.dumps(raw_payload, ensure_ascii=False),
            parsed_source_updated_at,
        )

    async def healthcheck(self) -> bool:
        pool = await self._get_pool()
        return bool(await pool.fetchval("SELECT to_regclass('public.recipe_details') IS NOT NULL"))

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


_store: PostgresRecipeDetailStore | None = None


def recipe_detail_storage_enabled() -> bool:
    return os.getenv("RECIPE_DETAIL_STORE", "postgres").strip().lower() == "postgres"


def get_recipe_detail_store() -> PostgresRecipeDetailStore:
    global _store
    if _store is None:
        _store = PostgresRecipeDetailStore.from_env()
    return _store


async def load_recipe_detail(
    recipe_id: str,
    language: str,
    *,
    max_age_seconds: int | None = None,
) -> dict | None:
    if not recipe_detail_storage_enabled():
        return None
    return await get_recipe_detail_store().load(
        recipe_id,
        language,
        max_age_seconds=max_age_seconds,
    )


async def save_recipe_detail(detail: dict, raw_payload: dict) -> None:
    if recipe_detail_storage_enabled():
        await get_recipe_detail_store().save(detail, raw_payload)


async def close_recipe_detail_store() -> None:
    global _store
    if _store is not None:
        await _store.close()
        _store = None
