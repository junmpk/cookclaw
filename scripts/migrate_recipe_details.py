#!/usr/bin/env python3
"""创建并验证 PostgreSQL 菜谱详情表，不输出连接凭据或详情正文。"""
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

from app.recipe_detail_store import PostgresRecipeDetailStore


async def migrate(recipe_id: str | None = None, language: str = "en") -> dict:
    store = PostgresRecipeDetailStore.from_env()
    try:
        pool = await store._get_pool()
        await pool.execute(
            (PROJECT_ROOT / "db/migrations/002_create_recipe_details.sql").read_text()
        )
        fetched = False
        if recipe_id:
            from app.orchestrator.cook import fetch_recipe_details

            result = await fetch_recipe_details(recipe_id, lang=language)
            fetched = bool(result.get("ok"))
            if not fetched:
                raise RuntimeError(
                    f"recipe fetch failed with code={result.get('code') or 'UNKNOWN'}"
                )
        verification = await pool.fetchrow(
            """
            SELECT
                to_regclass('public.recipe_details') IS NOT NULL AS table_exists,
                COUNT(*) AS detail_count,
                COUNT(*) FILTER (
                    WHERE detail ->> 'schema_version' <> 'recipe_detail_v1'
                       OR detail ->> 'recipe_id' <> recipe_id
                ) AS invalid_count,
                COUNT(*) FILTER (
                    WHERE raw_payload ? 'isCollect'
                       OR raw_payload ? 'isPurchase'
                ) AS private_state_count
            FROM public.recipe_details
            """
        )
        cached = await store.load(
            recipe_id,
            language,
            max_age_seconds=86400,
        ) if recipe_id else None
        return {
            "table_exists": bool(verification["table_exists"]),
            "detail_count": int(verification["detail_count"]),
            "invalid_count": int(verification["invalid_count"]),
            "private_state_count": int(verification["private_state_count"]),
            "requested_recipe_fetched": fetched,
            "requested_recipe_cached": cached is not None,
            "cached_ingredient_count": len(
                ((cached or {}).get("detail") or {}).get("ingredients") or []
            ),
            "cached_step_count": len(
                ((cached or {}).get("detail") or {}).get("steps") or []
            ),
        }
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-id")
    parser.add_argument("--language", choices=("zh", "en"), default="en")
    args = parser.parse_args()
    print(json.dumps(
        asyncio.run(migrate(args.recipe_id, args.language)),
        ensure_ascii=False,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
