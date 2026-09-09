"""
迁移脚本：旧的纯语义集合 → 新的 hybrid 集合（复用 dense 向量，零重新 embedding）。

做了三件事：
  1. 从旧集合读出每条的 dense 向量 + metadata
  2. 清洗 metadata：拆干净食材(ingredients) + 归一结构化 facets
  3. 写入新 hybrid 集合（text+BM25稀疏 + dense + metadata）；BM25 稀疏向量入库时自动生成

旧集合原样保留，作为评测基线 / 回滚。

用法（技能目录下）：
  .venv/bin/python migrate_hybrid.py                 # recipe_collection -> recipe_hybrid
  .venv/bin/python migrate_hybrid.py --recreate      # 目标已存在则先清空重建
  .venv/bin/python migrate_hybrid.py --src X --dst Y
"""
import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[4] / ".env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MILVUS_CONFIG  # noqa: E402
from module.normalize import clean_ingredients, derive_facets, build_index_text  # noqa: E402
from dataset.milvus import create_hybrid_collection, build_milvus_client, safe_milvus_uri  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402
from loguru import logger  # noqa: E402

BATCH = 500


def _transform(md):
    """纯语义 metadata -> hybrid metadata，并产出 BM25 文本。"""
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except Exception:
            md = {}
    md = md if isinstance(md, dict) else {}

    name = md.get("name", "未知菜品")
    raw_ing = md.get("ingredients", [])
    tags = md.get("tags", [])

    explicit_facets = md.get("facets")
    facets = explicit_facets if isinstance(explicit_facets, dict) else derive_facets(tags)
    canonical = [
        str(value)
        for values in facets.values()
        if isinstance(values, list)
        for value in values
        if str(value).strip()
    ]
    text = build_index_text(name, raw_ing, [*tags, *canonical])

    new_md = {
        "recipe_id": md.get("recipe_id", ""),
        "name": name,
        "ingredients": raw_ing,
        "ingredients_raw": md.get("ingredients_raw", raw_ing),
        "seasonings": md.get("seasonings", []),
        "tags": tags,
        "facets": facets,
        "nutrition": md.get("nutrition", {}),
        "image_url": md.get("image_url", ""),
        "description": md.get("description", ""),
        "lang": md.get("lang", ""),
        "estimated_time": md.get("estimated_time"),
        "difficulty": md.get("difficulty", ""),
        "tips": md.get("tips", ""),
        "steps": md.get("steps", []),
        "recipe_detail": md.get("recipe_detail"),
    }
    return new_md, text


def main():
    args = sys.argv[1:]
    recreate = "--recreate" in args
    src = _arg(args, "--src", os.getenv("MILVUS_COLLECTION", "recipe_collection"))
    dst = _arg(args, "--dst", os.getenv("MILVUS_HYBRID_COLLECTION", "recipe_hybrid"))

    uri = MILVUS_CONFIG["uri"]
    client = build_milvus_client(uri)

    if src not in client.list_collections():
        print(f"❌ 源集合不存在：{src} @ {safe_milvus_uri(uri)}")
        sys.exit(1)
    client.load_collection(src)

    # 1) 读旧集合（dense 向量 + metadata）
    logger.info(f"读取源集合 {src} ...")
    # Milvus Server 默认要求 offset + limit <= 16384。当前菜谱规模约 3000 条，
    # 使用服务端允许的最大窗口即可完整读取，也兼容本地 Milvus Lite。
    rows = client.query(src, filter="id >= 0", output_fields=["embedding", "metadata"], limit=16384)
    logger.info(f"读到 {len(rows)} 条")
    if not rows:
        print("❌ 源集合为空")
        sys.exit(1)

    dim = len(rows[0]["embedding"])

    # 2) 建新 hybrid 集合
    create_hybrid_collection(client, dst, dim, recreate=recreate)

    # 3) 转换 + 一次性写入（复用 dense 向量）
    # ⚠️ 必须单次 insert（单 segment）：milvus-lite 对 BM25 稀疏列的跨 segment
    #    后台 compaction 有 bug（"vector column must be FixedSizeList, got binary"），
    #    分批写会产生多 segment、合并时坏掉集合，导致之后 load_collection 失败。
    data = []
    for r in rows:
        new_md, text = _transform(r.get("metadata", {}))
        data.append({"text": text, "dense": r["embedding"], "metadata": new_md})
    client.insert(dst, data=data)
    n = len(data)
    logger.info(f"  单次写入 {n} 条（单 segment，规避 lite 多段 compaction bug）")

    client.flush(dst)

    # ⚠️ 强制在本进程内把 BM25 稀疏索引构建并持久化：
    #    milvus-lite 的稀疏索引是后台异步建的，若进程在它完成前退出，
    #    下个进程 load_collection 会报 "FixedSizeList, got binary"。
    #    这里 load + 跑一次 sparse 检索，逼它落盘。
    client.load_collection(dst)
    client.search(dst, data=["占位查询"], anns_field="sparse", limit=1,
                  output_fields=["metadata"], search_params={"metric_type": "BM25"})
    logger.info("已强制构建并持久化 BM25 稀疏索引")

    stats = client.get_collection_stats(dst)
    client.close()

    print(f"\n✅ 迁移完成：{src} → {dst}，写入 {n} 条（dim={dim}），复用向量、零重新 embedding。")
    print(f"   目标集合统计：{stats}")
    print(f'   测试：.venv/bin/python recipe_search.py "红烧肉" 3')


def _arg(args, key, default):
    if key in args:
        i = args.index(key)
        if i + 1 < len(args):
            return args[i + 1]
    return default


if __name__ == "__main__":
    main()
