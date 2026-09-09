#!/usr/bin/env python3
"""用真实菜名和食材批量补全菜谱详情，并重建 Milvus hybrid 向量集合。

流程：
  generate  Qwen 生成步骤/总时长/份量，逐条写 JSONL，可断点续跑
  build     把详情加入 metadata 与检索文本，重新生成 dense embedding，写 staging
  validate  校验 staging 的数量、详情覆盖率、业务键和 dense/BM25 检索
  promote   将生产集合改名为带时间戳的备份，再把 staging 原子改名为生产集合

默认 ``--phase all`` 只执行 generate/build/validate。生产切换必须显式传
``--promote``，且不能与 ``--limit`` 同时使用。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PROJECT_ROOT / "app/agent/skills/recipe-search"
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SKILL_ROOT))

# 当前远端 Milvus Server 使用 standard analyzer；必须在导入建表模块前设置。
os.environ.setdefault("MILVUS_ANALYZER_TYPE", "standard")

from app.orchestrator.recipe_completion import (  # noqa: E402
    build_recipe_detail_seed,
    complete_missing_recipe_fields,
)
from dataset.milvus import (  # noqa: E402
    build_milvus_client,
    create_hybrid_collection,
    safe_milvus_uri,
)
from module.embedding import get_embedding_model  # noqa: E402

QUERY_LIMIT = 16_384
TEXT_LIMIT = 4_096
DEFAULT_CACHE = PROJECT_ROOT / "outputs/milvus_recipe_detail_completion.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/milvus_recipe_detail_enrichment_report.json"


def _metadata(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _business_key(metadata: dict) -> str:
    recipe_id = str(metadata.get("recipe_id") or "").strip()
    lang = "en" if metadata.get("lang") == "en" else "zh"
    return f"{recipe_id}:{lang}" if recipe_id else ""


def _valid_detail(detail: Any, metadata: dict | None = None) -> bool:
    if not isinstance(detail, dict):
        return False
    steps = detail.get("steps")
    if (
        detail.get("schema_version") != "recipe_detail_v1"
        or not detail.get("ai_generated")
        or not isinstance(steps, list)
        or not 2 <= len(steps) <= 12
        or not isinstance(detail.get("cooking_time_seconds"), (int, float))
        or not 60 <= detail["cooking_time_seconds"] <= 604_800
        or not isinstance(detail.get("servings"), (int, float))
        or not 1 <= detail["servings"] <= 100
        or detail.get("executable") is not False
    ):
        return False
    if any(
        step.get("type") != "ai_generated_manual"
        or step.get("parameters")
        or not str(step.get("description") or "").strip()
        for step in steps
        if isinstance(step, dict)
    ):
        return False
    if any(not isinstance(step, dict) for step in steps):
        return False
    if metadata is not None:
        return (
            str(detail.get("recipe_id") or "").strip()
            == str(metadata.get("recipe_id") or "").strip()
            and detail.get("language")
            == ("en" if metadata.get("lang") == "en" else "zh")
        )
    return True


def _read_cache(path: Path) -> tuple[dict[str, dict], int]:
    records: dict[str, dict] = {}
    malformed = 0
    if not path.exists():
        return records, malformed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            key = str(item.get("business_key") or "")
            detail = item.get("detail")
            if key and _valid_detail(detail):
                records[key] = detail
    return records, malformed


def _fetch_rows(client, collection: str, *, include_dense: bool = False) -> list[dict]:
    if collection not in client.list_collections():
        raise RuntimeError(f"集合不存在：{collection}")
    client.load_collection(collection)
    output_fields = ["text", "metadata"]
    if include_dense:
        output_fields.append("dense")
    rows = client.query(
        collection_name=collection,
        filter="id >= 0",
        output_fields=output_fields,
        limit=QUERY_LIMIT,
    )
    if len(rows) >= QUERY_LIMIT:
        raise RuntimeError(f"集合达到查询上限 {QUERY_LIMIT}，请实现分页后再执行")
    keys = [_business_key(_metadata(row.get("metadata"))) for row in rows]
    if "" in keys:
        raise RuntimeError("源集合存在缺少 recipe_id 的记录")
    if len(keys) != len(set(keys)):
        raise RuntimeError("源集合存在重复的 (recipe_id, lang) 业务键")
    return sorted(rows, key=lambda row: _business_key(_metadata(row.get("metadata"))))


def _selected(rows: list[dict], limit: int | None) -> list[dict]:
    return rows[:limit] if limit else rows


async def _generate(
    rows: list[dict],
    cache_path: Path,
    *,
    concurrency: int,
    max_attempts: int,
) -> dict[str, dict]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached, malformed = _read_cache(cache_path)
    selected_keys = {_business_key(_metadata(row.get("metadata"))) for row in rows}
    reusable = {key: value for key, value in cached.items() if key in selected_keys}
    pending = [
        row for row in rows
        if _business_key(_metadata(row.get("metadata"))) not in reusable
    ]
    print(
        f"[generate] target={len(rows)} cached={len(reusable)} "
        f"pending={len(pending)} malformed_cache_lines={malformed}",
        flush=True,
    )
    if not pending:
        return reusable

    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    for row in pending:
        queue.put_nowait(row)
    for _ in range(concurrency):
        queue.put_nowait(None)

    write_lock = asyncio.Lock()
    state = {"done": 0, "failed": []}
    started = time.monotonic()

    async def worker(worker_id: int) -> None:
        while True:
            row = await queue.get()
            if row is None:
                queue.task_done()
                return
            md = _metadata(row.get("metadata"))
            key = _business_key(md)
            recipe = {
                "id": str(md.get("recipe_id") or ""),
                "name": md.get("name") or "",
                "ingredients": md.get("ingredients") or [],
                "ingredients_raw": md.get("ingredients_raw") or [],
                "image_url": md.get("image_url") or "",
                "tags": md.get("tags") or [],
            }
            detail: dict = {}
            for attempt in range(1, max_attempts + 1):
                detail = await complete_missing_recipe_fields(
                    build_recipe_detail_seed(recipe, str(md.get("lang") or "zh")),
                )
                if _valid_detail(detail, md):
                    break
                if attempt < max_attempts:
                    await asyncio.sleep(min(2**attempt, 5))
            if _valid_detail(detail, md):
                item = {
                    "business_key": key,
                    "recipe_id": str(md.get("recipe_id") or ""),
                    "lang": "en" if md.get("lang") == "en" else "zh",
                    "detail": detail,
                }
                encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                async with write_lock:
                    with cache_path.open("a", encoding="utf-8") as handle:
                        handle.write(encoded + "\n")
                        handle.flush()
                    reusable[key] = detail
                    state["done"] += 1
                    done = state["done"]
                    if done == 1 or done % 20 == 0 or done == len(pending):
                        elapsed = max(time.monotonic() - started, 0.001)
                        rate = done / elapsed
                        remaining = (len(pending) - done) / rate if rate else 0
                        print(
                            f"[generate] {done}/{len(pending)} new "
                            f"rate={rate:.2f}/s eta={remaining/60:.1f}m",
                            flush=True,
                        )
            else:
                async with write_lock:
                    state["failed"].append(key)
                    print(
                        f"[generate] FAILED worker={worker_id} key={key}",
                        flush=True,
                    )
            queue.task_done()

    workers = [
        asyncio.create_task(worker(index + 1))
        for index in range(max(1, concurrency))
    ]
    await queue.join()
    await asyncio.gather(*workers)
    if state["failed"]:
        failed_preview = ", ".join(state["failed"][:10])
        raise RuntimeError(
            f"{len(state['failed'])} 条生成失败；可重跑断点续传。示例：{failed_preview}"
        )
    return reusable


def _detail_index_text(base_text: str, detail: dict) -> str:
    step_text = " ".join(
        str(step.get("description") or "").strip()
        for step in detail.get("steps") or []
    )
    fields = [
        str(base_text or "").strip(),
        str(detail.get("name") or "").strip(),
        step_text,
        f"total_time_seconds {int(detail.get('cooking_time_seconds') or 0)}",
        f"servings {int(detail.get('servings') or 0)}",
    ]
    return " ".join(value for value in fields if value)[:TEXT_LIMIT]


def _build(
    client,
    rows: list[dict],
    details: dict[str, dict],
    staging: str,
    *,
    recreate: bool,
    insert_batch: int,
) -> None:
    missing = [
        _business_key(_metadata(row.get("metadata")))
        for row in rows
        if _business_key(_metadata(row.get("metadata"))) not in details
    ]
    if missing:
        raise RuntimeError(f"缓存缺少 {len(missing)} 条详情，不能建 staging")
    if staging in client.list_collections() and not recreate:
        raise RuntimeError(f"staging 已存在：{staging}；确认后使用 --recreate-staging")

    texts = [
        _detail_index_text(
            str(row.get("text") or ""),
            details[_business_key(_metadata(row.get("metadata")))],
        )
        for row in rows
    ]
    print(f"[build] encoding {len(texts)} enriched texts ...", flush=True)
    embeddings = get_embedding_model().encode(texts, batch_size=10)
    if len(embeddings) != len(rows) or embeddings.shape[1] != 1024:
        raise RuntimeError(f"embedding shape 异常：{embeddings.shape}")

    create_hybrid_collection(client, staging, 1024, recreate=recreate)
    inserted = 0
    for offset in range(0, len(rows), insert_batch):
        data = []
        for index, row in enumerate(rows[offset:offset + insert_batch], offset):
            md = deepcopy(_metadata(row.get("metadata")))
            key = _business_key(md)
            md["recipe_detail"] = details[key]
            encoded_size = len(
                json.dumps(md, ensure_ascii=False, separators=(",", ":")).encode()
            )
            if encoded_size > 60_000:
                raise RuntimeError(f"metadata 过大 ({encoded_size} bytes)：{key}")
            data.append({
                "text": texts[index],
                "dense": embeddings[index].tolist(),
                "metadata": md,
            })
        client.insert(collection_name=staging, data=data)
        inserted += len(data)
        print(f"[build] inserted={inserted}/{len(rows)}", flush=True)

    client.flush(staging)
    client.load_collection(staging)
    print(f"[build] staging flushed and loaded: {staging}", flush=True)


def _validate(client, collection: str, expected: int) -> dict:
    rows = _fetch_rows(client, collection)
    invalid = []
    keys = []
    for row in rows:
        md = _metadata(row.get("metadata"))
        keys.append(_business_key(md))
        if not _valid_detail(md.get("recipe_detail"), md):
            invalid.append(_business_key(md))
    dense = client.search(
        collection_name=collection,
        data=[[0.0] * 1024],
        anns_field="dense",
        limit=1,
        output_fields=["metadata"],
        search_params={"metric_type": "COSINE"},
    )
    sparse_probe = str(
        _metadata(rows[0].get("metadata")).get("name") or "recipe"
    ).strip()
    sparse = client.search(
        collection_name=collection,
        data=[sparse_probe],
        anns_field="sparse",
        limit=1,
        output_fields=["metadata"],
        search_params={"metric_type": "BM25"},
    )
    result = {
        "collection": collection,
        "expected_rows": expected,
        "actual_rows": len(rows),
        "unique_business_keys": len(set(keys)),
        "valid_recipe_details": len(rows) - len(invalid),
        "invalid_recipe_detail_keys": invalid[:20],
        "dense_probe_hits": len(dense[0]) if dense else 0,
        "sparse_probe_hits": len(sparse[0]) if sparse else 0,
        "sparse_probe": sparse_probe,
    }
    if (
        len(rows) != expected
        or len(set(keys)) != expected
        or invalid
        or result["dense_probe_hits"] < 1
        or result["sparse_probe_hits"] < 1
    ):
        raise RuntimeError(f"staging 校验失败：{json.dumps(result, ensure_ascii=False)}")
    print(f"[validate] {json.dumps(result, ensure_ascii=False)}", flush=True)
    return result


def _promote(client, source: str, staging: str, expected: int) -> str:
    _validate(client, staging, expected)
    backup = f"{source}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if backup in client.list_collections():
        raise RuntimeError(f"备份集合已存在：{backup}")
    client.release_collection(source)
    client.release_collection(staging)
    client.rename_collection(old_name=source, new_name=backup)
    try:
        client.rename_collection(old_name=staging, new_name=source)
    except Exception:
        client.rename_collection(old_name=backup, new_name=source)
        raise
    client.load_collection(source)
    print(f"[promote] production={source} backup={backup}", flush=True)
    return backup


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("generate", "build", "validate", "promote", "all"),
        default="all",
    )
    parser.add_argument("--source", default="recipe_hybrid")
    parser.add_argument("--staging", default="recipe_hybrid_enriched")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--insert-batch", type=int, default=250)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--recreate-staging", action="store_true")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="all 完成并校验后切换生产集合",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0")
    if not 1 <= args.concurrency <= 32:
        parser.error("--concurrency 必须在 1..32")
    if (args.phase == "promote" or args.promote) and args.limit:
        parser.error("生产切换不能与 --limit 一起使用")
    return args


async def main() -> int:
    args = _parse_args()
    uri = os.getenv("RECIPE_MILVUS_URI", "").strip()
    if not uri:
        raise RuntimeError("RECIPE_MILVUS_URI 未配置")
    client = build_milvus_client(uri)
    print(f"[preflight] milvus={safe_milvus_uri(uri)} source={args.source}", flush=True)
    source_rows = _fetch_rows(client, args.source)
    rows = _selected(source_rows, args.limit)
    empty_inputs = [
        _business_key(_metadata(row.get("metadata")))
        for row in rows
        if not str(_metadata(row.get("metadata")).get("name") or "").strip()
        or not (_metadata(row.get("metadata")).get("ingredients") or [])
    ]
    if empty_inputs:
        raise RuntimeError(
            f"{len(empty_inputs)} 条缺少菜名或食材，不能有依据地生成：{empty_inputs[:10]}"
        )
    print(
        f"[preflight] source_rows={len(source_rows)} selected={len(rows)} "
        f"model={os.getenv('RECIPE_COMPLETION_MODEL') or 'default'}",
        flush=True,
    )

    details, _ = _read_cache(args.cache)
    if args.phase in ("generate", "all"):
        details = await _generate(
            rows,
            args.cache,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
        )
    if args.phase in ("build", "all"):
        _build(
            client,
            rows,
            details,
            args.staging,
            recreate=args.recreate_staging,
            insert_batch=args.insert_batch,
        )

    validation = None
    if args.phase in ("validate", "all"):
        validation = _validate(client, args.staging, len(rows))
    backup = None
    if args.phase == "promote" or args.promote:
        if len(rows) != len(source_rows):
            raise RuntimeError("只允许完整数据集切换生产")
        backup = _promote(client, args.source, args.staging, len(source_rows))
        validation = _validate(client, args.source, len(source_rows))

    report = {
        "finished_at": datetime.now().astimezone().isoformat(),
        "source": args.source,
        "staging": args.staging,
        "source_rows": len(source_rows),
        "processed_rows": len(rows),
        "cache": str(args.cache),
        "cache_valid_records": len(details),
        "model": os.getenv("RECIPE_COMPLETION_MODEL") or "default",
        "validation": validation,
        "backup": backup,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    client.close()
    print(f"[done] report={args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
