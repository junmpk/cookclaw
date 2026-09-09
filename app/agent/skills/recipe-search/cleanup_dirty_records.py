#!/usr/bin/env python3
"""
清理 Milvus 脏菜谱数据：缺少食材或图片地址的记录默认只统计，显式 --apply 才删除。

用法：
  # 先 dry-run 查看将删除哪些记录
  .venv/bin/python cleanup_dirty_records.py

  # 确认后删除线上 hybrid 集合中的脏数据
  .venv/bin/python cleanup_dirty_records.py --apply

  # 同时清理基线集合
  .venv/bin/python cleanup_dirty_records.py --collections recipe_hybrid,recipe_collection --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(_PROJECT_ROOT / ".env")

# 本地 SSH 隧道 / 内网 Milvus 不应走系统代理；gRPC 也可能读取这些变量。
_NO_PROXY = "127.0.0.1,localhost,::1"
os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY", ""), _NO_PROXY]))
os.environ["no_proxy"] = ",".join(filter(None, [os.environ.get("no_proxy", ""), _NO_PROXY]))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MILVUS_CONFIG, SPLIT_LANG_ENABLED, hybrid_collection_for  # noqa: E402
from dataset.milvus import build_milvus_client, safe_milvus_uri  # noqa: E402


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        return not s or s.lower() in {"nan", "none", "null", "未知", "未知食材"}
    if isinstance(value, (list, tuple, set)):
        return not any(not _is_blank(v) for v in value)
    if isinstance(value, dict):
        return not any(not _is_blank(v) for v in value.values())
    return False


def _metadata(row: dict) -> dict:
    md = row.get("metadata", {})
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except Exception:
            return {}
    return md if isinstance(md, dict) else {}


def _dirty_reasons(md: dict, required_fields: Iterable[str]) -> list[str]:
    if not md:
        return ["metadata"]
    reasons = []
    for field in required_fields:
        if _is_blank(md.get(field)):
            reasons.append(field)
    return reasons


def _default_collections() -> list[str]:
    if SPLIT_LANG_ENABLED:
        return [hybrid_collection_for("zh"), hybrid_collection_for("en")]
    return [MILVUS_CONFIG["hybrid_collection"]]


def _iter_rows(client, collection: str, batch_size: int):
    """优先用 query_iterator；老版本 pymilvus 回退 offset 分页。"""
    fields = ["id", "metadata"]
    if hasattr(client, "query_iterator"):
        iterator = client.query_iterator(
            collection_name=collection,
            filter="",
            output_fields=fields,
            batch_size=batch_size,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                yield from batch
        finally:
            try:
                iterator.close()
            except Exception:
                pass
        return

    offset = 0
    while True:
        batch = client.query(
            collection_name=collection,
            filter="",
            output_fields=fields,
            limit=batch_size,
            offset=offset,
        )
        if not batch:
            break
        yield from batch
        if len(batch) < batch_size:
            break
        offset += len(batch)


def _delete_ids(client, collection: str, ids: list[int], batch_size: int) -> int:
    deleted = 0
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        expr = f"id in {chunk}"
        client.delete(collection_name=collection, filter=expr)
        deleted += len(chunk)
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description="删除 Milvus 中缺少食材或图片地址的菜谱记录")
    parser.add_argument("--uri", default=MILVUS_CONFIG["uri"], help="Milvus URI，默认读取 .env/RECIPE_MILVUS_URI")
    parser.add_argument("--collections", default=",".join(_default_collections()),
                        help="逗号分隔集合名；默认按当前检索配置清理 hybrid 集合")
    parser.add_argument("--required", default="ingredients,image_url",
                        help="逗号分隔的 metadata 必填字段，默认 ingredients,image_url")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--sample", type=int, default=20, help="每个集合最多打印多少条脏数据样例")
    parser.add_argument("--apply", action="store_true", help="真正执行删除；不加则只 dry-run")
    args = parser.parse_args()

    collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    required_fields = [f.strip() for f in args.required.split(",") if f.strip()]
    if not collections:
        print("❌ 没有指定集合")
        return 2
    if not required_fields:
        print("❌ 没有指定必填字段")
        return 2

    print(f"Milvus URI: {safe_milvus_uri(args.uri)}")
    print(f"Mode: {'APPLY DELETE' if args.apply else 'DRY RUN'}")
    print(f"Collections: {', '.join(collections)}")
    print(f"Required metadata fields: {', '.join(required_fields)}")
    print()

    client = build_milvus_client(args.uri)
    existing = set(client.list_collections())
    total_dirty = 0
    total_deleted = 0

    try:
        for collection in collections:
            if collection not in existing:
                print(f"⚠️  跳过不存在集合：{collection}")
                continue

            client.load_collection(collection)
            scanned = 0
            dirty_ids: list[int] = []
            reason_counts: dict[str, int] = {}
            samples = []

            for row in _iter_rows(client, collection, args.batch_size):
                scanned += 1
                md = _metadata(row)
                reasons = _dirty_reasons(md, required_fields)
                if not reasons:
                    continue
                row_id = row.get("id")
                if row_id is None:
                    continue
                dirty_ids.append(int(row_id))
                for reason in reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if len(samples) < args.sample:
                    samples.append({
                        "id": row_id,
                        "recipe_id": md.get("recipe_id"),
                        "name": md.get("name"),
                        "lang": md.get("lang"),
                        "missing": reasons,
                        "ingredients": md.get("ingredients"),
                        "image_url": md.get("image_url"),
                    })

            total_dirty += len(dirty_ids)
            print(f"[{collection}] scanned={scanned}, dirty={len(dirty_ids)}, reasons={reason_counts}")
            for s in samples:
                print("  sample:", json.dumps(s, ensure_ascii=False))

            if args.apply and dirty_ids:
                deleted = _delete_ids(client, collection, dirty_ids, args.batch_size)
                total_deleted += deleted
                print(f"  deleted={deleted}")
            print()
    finally:
        try:
            client.close()
        except Exception:
            pass

    if args.apply:
        print(f"✅ 清理完成：dirty={total_dirty}, deleted={total_deleted}")
    else:
        print(f"DRY RUN 完成：dirty={total_dirty}。确认无误后加 --apply 执行删除。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
