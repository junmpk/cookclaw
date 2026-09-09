"""
translate_recipes.py — 翻译 Agent：把中文菜谱译成【指定目标语言】，补充 recipe_hybrid_<lang> 集合。

支持任意目标语言（--target en/es/fr/ja/...）。每门语言独立缓存 translate_cache_<lang>.jsonl、
独立集合 recipe_hybrid_<lang>。facets 是语言无关过滤维度，复用不译；目标语言文本重新 embedding。
红线（AGENTS.md「禁止编造食谱数据」）：翻译只改写文本，不新增/编造食材或步骤。

前提：先跑 migrate_hybrid_lang.py（需 recipe_hybrid_zh 作翻译源；recipe_hybrid_<lang> 不存在会新建）。
复用 enrich_tags 范式：异步并发 + 断点缓存（key = zh recipe_id）。

用法（技能目录下，一门语言走两步）：
  .venv/bin/python translate_recipes.py --target es --limit 5 --dry   # 抽样预览，不写缓存
  .venv/bin/python translate_recipes.py --target es                  # 全量翻译写缓存
  .venv/bin/python translate_recipes.py --target es --build          # 重建 recipe_hybrid_es
"""
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

import httpx
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(_PROJECT_ROOT / ".env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MILVUS_CONFIG, EMBEDDING_CONFIG  # noqa: E402
from module.normalize import build_index_text  # noqa: E402
from dataset.milvus import build_milvus_client, create_hybrid_collection  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from loguru import logger  # noqa: E402

MODEL = os.getenv("TRANSLATE_MODEL", "qwen-plus")
CONCURRENCY = 8
_SKILL_DIR = Path(__file__).resolve().parent

# 目标语言代码 → 语言名（给翻译 prompt 用）。要支持新语言在此登记即可。
LANG_NAMES = {
    "en": "English（英文）", "es": "Spanish（西班牙语）", "fr": "French（法语）",
    "de": "German（德语）", "pt": "Portuguese（葡萄牙语）", "it": "Italian（意大利语）",
    "ru": "Russian（俄语）", "ja": "Japanese（日语）", "ko": "Korean（韩语）",
    "ar": "Arabic（阿拉伯语）", "th": "Thai（泰语）", "vi": "Vietnamese（越南语）",
    "id": "Indonesian（印尼语）",
}

_client = AsyncOpenAI(
    api_key=EMBEDDING_CONFIG["api_key"],
    base_url=EMBEDDING_CONFIG["base_url"],
    http_client=httpx.AsyncClient(trust_env=False),
)


def _cache_path(target: str) -> Path:
    return _SKILL_DIR / f"translate_cache_{target}.jsonl"


def _build_prompt(target: str) -> str:
    name = LANG_NAMES.get(target, target)
    return f"""你是中文菜谱翻译专家。把给定中文菜谱信息译成自然地道的 {name}。
只输出一个 JSON 对象：{{"name": "...", "ingredients": ["...", ...], "description": "..."}}
要求：
- 菜名地道，符合 {name} 的菜谱表达习惯，不要逐字直译；
- ingredients 逐项翻译食材名（输入已是干净食材名，无需保留数量）；
- 烹饪术语用 {name} 的标准译法；
- description 输入为空则给空字符串；
- 只输出 JSON，不要解释、不要代码块。"""


def _as_dict(md):
    if isinstance(md, str):
        try:
            return json.loads(md)
        except Exception:
            return {}
    return md if isinstance(md, dict) else {}


async def translate_one(sem, prompt, name, ingredients, description):
    """调 Qwen 翻译一条；失败重试 3 次，最终失败返回 None。"""
    ing_str = "、".join(ingredients) if isinstance(ingredients, list) else str(ingredients or "")
    user = f"菜名：{name}\n食材：{ing_str}\n描述：{description or ''}"
    async with sem:
        for attempt in range(3):
            try:
                resp = await _client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": prompt},
                              {"role": "user", "content": user}],
                    temperature=0.3, max_tokens=600,
                )
                raw = (resp.choices[0].message.content or "").strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                obj = json.loads(raw)
                if isinstance(obj, dict) and obj.get("name"):
                    obj.setdefault("ingredients", [])
                    obj.setdefault("description", "")
                    return obj
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"翻译失败：error_type={type(e).__name__}")
                await asyncio.sleep(1.5 * (attempt + 1))
        return None


def _read_zh(client, src):
    """从分语言 zh 集合读 zh 菜谱 metadata（翻译源永远是中文）。"""
    if src not in client.list_collections():
        print(f"❌ 源集合不存在：{src}。请先跑 migrate_hybrid_lang.py 建分语言集合。")
        sys.exit(1)
    client.load_collection(src)
    rows = client.query(src, filter="id >= 0", output_fields=["metadata"], limit=16384)
    return [_as_dict(r.get("metadata", {})) for r in rows]


def _load_cache(target):
    cache = {}
    p = _cache_path(target)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                o = json.loads(line)
                cache[o["id"]] = o
            except Exception:
                pass
    return cache


def _rid(md, i):
    return str(md.get("recipe_id") or f"_idx{i}")


async def run_translate(args):
    if not EMBEDDING_CONFIG.get("api_key"):
        print("❌ 未读到 DASHSCOPE_API_KEY，请先在项目根 .env 配置。")
        sys.exit(1)
    target = args.target
    prompt = _build_prompt(target)
    logger.info(f"目标语言：{target}（{LANG_NAMES.get(target, target)}），模型 {MODEL}")

    uri = MILVUS_CONFIG["uri"]
    client = build_milvus_client(uri)
    base = os.getenv("MILVUS_HYBRID_COLLECTION", "recipe_hybrid")
    src = args.src or f"{base}_zh"

    zh = _read_zh(client, src)
    client.close()
    rows = zh[:args.limit] if args.limit else zh
    logger.info(f"源 {src}：{len(zh)} 条 zh，本次处理 {len(rows)} 条")

    cache = {} if args.dry else _load_cache(target)
    sem = asyncio.Semaphore(CONCURRENCY)
    todo = [(i, md) for i, md in enumerate(rows) if _rid(md, i) not in cache]
    logger.info(f"待翻译 {len(todo)} 条（缓存命中 {len(rows) - len(todo)}）")

    async def worker(i, md):
        tr = await translate_one(sem, prompt, md.get("name", ""), md.get("ingredients", []), md.get("description", ""))
        if tr and not args.dry:
            rec = {"id": _rid(md, i), "zh_md": md, "translation": tr}
            with open(_cache_path(target), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return md.get("name", ""), tr

    done = 0
    tasks = [asyncio.create_task(worker(i, md)) for i, md in todo]
    for fut in asyncio.as_completed(tasks):
        name, tr = await fut
        done += 1
        if args.dry or done <= 5 or done % 100 == 0:
            en = (tr or {}).get("name", "✗失败")
            ings = (tr or {}).get("ingredients", [])
            print(f"  [{done}/{len(todo)}] {name} → {en} | {ings[:4]}")

    if args.dry:
        print(f"\n（dry：未写缓存。确认 {target} 译文 OK 后去 --dry 跑，再 --target {target} --build。）")
    else:
        print(f"\n✅ {target} 翻译完成，缓存 → {_cache_path(target).name}（累计 {len(_load_cache(target))} 条）。下一步：--target {target} --build")


def run_build(args):
    """单次重建 recipe_hybrid_<target> = 原有该语言原生数据 + 翻译缓存（目标语言文本重新 embedding）。"""
    from module.embedding import get_embedding_model

    target = args.target
    uri = MILVUS_CONFIG["uri"]
    client = build_milvus_client(uri)
    base = os.getenv("MILVUS_HYBRID_COLLECTION", "recipe_hybrid")
    col = f"{base}_{target}"

    # 1) 读原有该语言集合，保留非翻译的原生数据（如 en 的 559 原生英文菜）
    existing, dim = [], MILVUS_CONFIG["vector_dim"]
    if col in client.list_collections():
        client.load_collection(col)
        rows = client.query(col, filter="id >= 0", output_fields=["metadata", "dense"], limit=16384)
        for r in rows:
            md = _as_dict(r.get("metadata", {}))
            if md.get("translated"):   # 跳过上轮翻译产物，避免重复累积（幂等）
                continue
            text = build_index_text(md.get("name", ""), md.get("ingredients", []), md.get("tags", []))
            existing.append({"text": text, "dense": r["dense"], "metadata": md})
            dim = len(r["dense"])
    logger.info(f"[{target}] 原生（非翻译）保留 {len(existing)} 条")

    # 2) 翻译缓存 → 目标语言菜 + 重新 embedding
    cache = _load_cache(target)
    if not cache:
        print(f"❌ {_cache_path(target).name} 为空，请先跑 --target {target} 翻译。")
        sys.exit(1)
    embed = get_embedding_model(api_key=EMBEDDING_CONFIG["api_key"], base_url=EMBEDDING_CONFIG["base_url"],
                                model=EMBEDDING_CONFIG["model"], timeout=EMBEDDING_CONFIG.get("timeout", 30))
    metas, texts = [], []
    for zid, o in cache.items():
        tr, zmd = o["translation"], o["zh_md"]
        tgt_md = {
            "recipe_id": zmd.get("recipe_id", ""),
            "name": tr.get("name", ""),
            "ingredients": tr.get("ingredients", []),
            "ingredients_raw": tr.get("ingredients", []),
            "tags": zmd.get("tags", []),
            "facets": zmd.get("facets", {}),     # 语言无关过滤维度，复用
            "image_url": zmd.get("image_url", ""),
            "description": tr.get("description", ""),
            "lang": target,
            "translated": True,
            "translated_from": zmd.get("recipe_id", "") or zid,
        }
        metas.append(tgt_md)
        texts.append(build_index_text(tgt_md["name"], tgt_md["ingredients"], tgt_md["tags"]))
    logger.info(f"[{target}] 翻译 {len(texts)} 条，向量化中（text-embedding-v4）...")
    vecs = embed.encode(texts)
    translated = [{"text": t, "dense": v.tolist(), "metadata": m}
                  for t, v, m in zip(texts, vecs, metas)]

    # 3) 单次重建（recreate + 单 segment insert，规避 Lite BM25 多段 compaction bug）
    all_data = existing + translated
    create_hybrid_collection(client, col, dim, recreate=True)
    client.insert(col, data=all_data)
    client.flush(col)
    client.load_collection(col)
    client.search(col, data=["placeholder"], anns_field="sparse", limit=1,
                  output_fields=["metadata"], search_params={"metric_type": "BM25"})
    stats = client.get_collection_stats(col)
    client.close()
    print(f"\n✅ 重建 {col}：原生 {len(existing)} + 翻译 {len(translated)} = {len(all_data)} 条")
    print(f"   集合统计：{stats}")
    print(f'   测试：MILVUS_SPLIT_LANG=1 .venv/bin/python -c "...search lang={target}..."')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="目标语言代码，如 en/es/fr/ja（见 LANG_NAMES）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（抽样）")
    ap.add_argument("--dry", action="store_true", help="只预览翻译，不写缓存")
    ap.add_argument("--src", type=str, default="", help="zh 源集合（默认 recipe_hybrid_zh）")
    ap.add_argument("--build", action="store_true", help="从缓存重建 recipe_hybrid_<target>")
    args = ap.parse_args()
    if args.target == "zh":
        print("❌ 目标语言不能是 zh（zh 是翻译源）。")
        sys.exit(1)
    if args.build:
        run_build(args)
    else:
        asyncio.run(run_translate(args))


if __name__ == "__main__":
    main()
