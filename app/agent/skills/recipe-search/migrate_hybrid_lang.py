"""
迁移脚本（多语言分集合，route ②）：纯语义基线集合 → 按 lang 分流到两个 hybrid 集合。

  recipe_collection  ──按 metadata.lang 分流──>  recipe_hybrid_zh / recipe_hybrid_en
  （复用 dense 向量，零重新 embedding；text + BM25 稀疏入库时自动生成）

语言由集合隔离后，检索按 detect_lang 路由到对应集合（见 config.hybrid_collection_for），
不再需要 metadata["lang"] 过滤，且各语言 BM25 / 向量空间互不干扰。

旧集合 recipe_collection / recipe_hybrid 原样保留（基线 / 回滚）。

用法（技能目录下）：
  .venv/bin/python migrate_hybrid_lang.py                # recipe_collection -> recipe_hybrid_{zh,en}
  .venv/bin/python migrate_hybrid_lang.py --recreate     # 目标已存在则先清空重建
  .venv/bin/python migrate_hybrid_lang.py --src X        # 自定义源集合
然后开启路由：.env 设 MILVUS_SPLIT_LANG=1
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
from loguru import logger  # noqa: E402

LANGS = ("zh", "en")


def _as_dict(md):
    if isinstance(md, str):
        try:
            return json.loads(md)
        except Exception:
            return {}
    return md if isinstance(md, dict) else {}


def _transform(md):
    """旧 metadata -> 新 metadata（清洗食材 + 派生 facets），并产出 BM25 文本。"""
    name = md.get("name", "未知菜品")
    raw_ing = md.get("ingredients", [])
    tags = md.get("tags", [])
    new_md = {
        "recipe_id": md.get("recipe_id", ""),
        "name": name,
        "ingredients": clean_ingredients(raw_ing),   # 清洗后的食材名数组
        "ingredients_raw": raw_ing,                  # 原始串留痕
        "tags": tags,
        "facets": derive_facets(tags),               # 结构化过滤维度
        "image_url": md.get("image_url", ""),
        "description": md.get("description", ""),
        "lang": md.get("lang", ""),
    }
    return new_md, build_index_text(name, raw_ing, tags)


def _arg(args, key, default):
    if key in args:
        i = args.index(key)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def main():
    args = sys.argv[1:]
    recreate = "--recreate" in args
    src = _arg(args, "--src", os.getenv("MILVUS_COLLECTION", "recipe_collection"))
    base = os.getenv("MILVUS_HYBRID_COLLECTION", "recipe_hybrid")

    uri = MILVUS_CONFIG["uri"]
    client = build_milvus_client(uri)

    if src not in client.list_collections():
        print(f"❌ 源集合不存在：{src} @ {safe_milvus_uri(uri)}")
        sys.exit(1)
    client.load_collection(src)

    logger.info(f"读取源集合 {src} ...")
    # Server 默认 offset+limit<=16384；当前 ~3000 条，单窗口可完整读取，也兼容 Lite。
    rows = client.query(src, filter="id >= 0", output_fields=["embedding", "metadata"], limit=16384)
    logger.info(f"读到 {len(rows)} 条")
    if not rows:
        print("❌ 源集合为空")
        sys.exit(1)
    dim = len(rows[0]["embedding"])

    # 按 lang 分流（数据仅 zh/en；非法 lang 兜底归 zh 并告警）
    buckets = {lang: [] for lang in LANGS}
    other = 0
    for r in rows:
        md = _as_dict(r.get("metadata", {}))
        raw_lang = md.get("lang", "")
        lang = raw_lang if raw_lang in LANGS else "zh"
        if raw_lang not in LANGS:
            other += 1
        new_md, text = _transform(md)
        buckets[lang].append({"text": text, "dense": r["embedding"], "metadata": new_md})
    if other:
        logger.warning(f"{other} 条 lang 非 zh/en，已兜底归 zh")

    for lang in LANGS:
        dst = f"{base}_{lang}"
        data = buckets[lang]
        logger.info(f"[{lang}] 目标集合 {dst}：{len(data)} 条")
        create_hybrid_collection(client, dst, dim, recreate=recreate)
        if not data:
            logger.warning(f"[{lang}] 无数据，跳过 insert")
            continue
        # ⚠️ 单次 insert（单 segment）：规避 milvus-lite BM25 跨段 compaction bug
        client.insert(dst, data=data)
        client.flush(dst)
        # 强制构建并持久化 BM25 稀疏索引：load + 跑一次 sparse 检索逼它落盘
        # （否则下个进程 load 可能报 "FixedSizeList, got binary"）
        client.load_collection(dst)
        client.search(dst, data=["占位查询"], anns_field="sparse", limit=1,
                      output_fields=["metadata"], search_params={"metric_type": "BM25"})
        logger.info(f"[{lang}] 写入 {len(data)} 条并已持久化 BM25 稀疏索引")

    client.close()
    print(f"\n✅ 分语言迁移完成：{src} → {base}_zh / {base}_en（复用向量，零重新 embedding）")
    print(f"   zh={len(buckets['zh'])} 条，en={len(buckets['en'])} 条")
    print(f"   开启路由：.env 设 MILVUS_SPLIT_LANG=1")
    print(f'   测试：MILVUS_SPLIT_LANG=1 .venv/bin/python recipe_search.py "红烧肉" 3')


if __name__ == "__main__":
    main()
