#!/usr/bin/env python3
"""Generate test tags without sending real recipe names or ingredients.

By default this script reuses existing strong tag cache first. With
--ignore-strong-cache, every row uses only anonymous language + broad ingredient
categories, which is safer for stale-cache test datasets.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import httpx
import openpyxl
from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "app" / "agent" / "skills" / "recipe-search"
load_dotenv(ROOT / ".env")


def _load_skill_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


enrich_tags = _load_skill_module("recipe_search_enrich_tags", SKILL_DIR / "enrich_tags.py")
config = _load_skill_module("recipe_search_config", SKILL_DIR / "config.py")
EMBEDDING_CONFIG = config.EMBEDDING_CONFIG

STRONG_CACHE = SKILL_DIR / "tag_cache.jsonl"
ANON_CACHE = SKILL_DIR / "tag_cache_anonymized.jsonl"
MODEL = os.getenv("TAG_MODEL", "qwen-plus")
CONCURRENCY = 6

SAFE_SYSTEM_PROMPT = """你是菜谱标签专家。你只能看到匿名菜谱语言和宽泛食材类别，
不能假设具体菜名或具体食材。请生成适合测试检索链路的通用标签。

要求：
- 输出 JSON 数组，元素为标签字符串。
- 标签要中英双语成对出现。
- 覆盖主食材类别、口味倾向、烹饪方式、营养特点、餐次/类型、场景人群。
- 不要输出具体菜名，不要编造具体食材。
- 共 12-20 个标签。
"""

CATEGORY_RULES = [
    ("pork", ["猪", "pork", "ham", "bacon", "sausage"]),
    ("beef", ["牛", "beef", "steak"]),
    ("lamb", ["羊", "lamb", "mutton"]),
    ("chicken", ["鸡", "chicken", "wing", "drumstick"]),
    ("duck", ["鸭", "duck"]),
    ("seafood", ["鱼", "虾", "蟹", "贝", "蛤", "鲜", "fish", "shrimp", "crab", "clam", "seafood"]),
    ("egg", ["蛋", "egg"]),
    ("tofu-or-soy", ["豆腐", "豆皮", "腐竹", "豆干", "tofu", "soy"]),
    ("beans", ["豆", "bean", "pea", "lentil"]),
    ("mushroom", ["菇", "菌", "mushroom"]),
    ("vegetable", ["菜", "瓜", "笋", "椒", "葱", "姜", "蒜", "萝卜", "茄", "番茄", "土豆", "vegetable", "pepper", "tomato", "potato"]),
    ("fruit", ["果", "莓", "苹果", "香蕉", "柠檬", "fruit", "apple", "banana", "lemon", "berry", "avocado"]),
    ("rice-or-grain", ["米", "饭", "粥", "麦", "oat", "rice", "grain"]),
    ("noodle-or-flour", ["面", "粉", "饼", "noodle", "flour", "pasta", "bread"]),
    ("dairy", ["奶", "芝士", "黄油", "cheese", "milk", "butter", "cream", "yogurt", "yakult"]),
    ("nuts", ["坚果", "花生", "芝麻", "nut", "peanut", "sesame"]),
    ("sauce-or-spice", ["酱", "油", "盐", "糖", "醋", "辣", "spice", "sauce", "salt", "sugar", "vinegar", "chili"]),
    ("soup-or-liquid", ["汤", "水", "高汤", "broth", "stock", "soup", "water"]),
    ("beverage", ["饮", "汁", "茶", "咖啡", "drink", "juice", "tea", "coffee"]),
    ("dessert", ["甜", "糖", "巧克力", "dessert", "sweet", "chocolate"]),
]


def load_cache(path: Path, key_name: str = "uid") -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            tags = item.get("tags", [])
            if isinstance(tags, list):
                cache[str(item[key_name])] = [str(t).strip() for t in tags if str(t).strip()]
        except Exception:
            continue
    return cache


def row_uid(row: dict) -> str:
    return f'{row["lang"]}:{row["id"]}'


def infer_categories(ingredients: str) -> list[str]:
    text = str(ingredients or "").lower()
    categories: list[str] = []
    for category, keywords in CATEGORY_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            categories.append(category)
    if not categories:
        categories.append("mixed-ingredients")
    return categories[:8]


def anon_key(row: dict) -> str:
    categories = ",".join(infer_categories(row.get("ing", "")))
    return f'{row["lang"]}:{categories}'


def build_user_prompt(key: str) -> str:
    lang, categories = key.split(":", 1)
    return f"language: {lang}\ningredient_categories: {categories}"


def parse_tags(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    tags = json.loads(raw)
    if not isinstance(tags, list):
        return []
    return [str(t).strip() for t in tags if str(t).strip()]


async def generate_tags(client: AsyncOpenAI, sem: asyncio.Semaphore, key: str) -> tuple[str, list[str]]:
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SAFE_SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(key)},
                    ],
                    temperature=0.2,
                    max_tokens=360,
                )
                return key, parse_tags(resp.choices[0].message.content or "")
            except Exception as exc:
                if attempt == 2:
                    print(f"warn: anonymized tag generation failed for {key}: {exc}")
                await asyncio.sleep(1.2 * (attempt + 1))
    return key, []


async def main_async(args: argparse.Namespace) -> None:
    if not EMBEDDING_CONFIG.get("api_key") or "填入" in EMBEDDING_CONFIG["api_key"]:
        raise SystemExit("missing DASHSCOPE_API_KEY in project .env")

    rows = enrich_tags._read_all(Path(args.src))
    strong_cache = {} if args.ignore_strong_cache else load_cache(STRONG_CACHE)
    anon_cache = load_cache(ANON_CACHE, key_name="anon_key")

    missing = [row for row in rows if not strong_cache.get(row_uid(row))]
    keys = sorted({anon_key(row) for row in missing if not anon_cache.get(anon_key(row))})

    print(f"rows={len(rows)}")
    print(f"strong_cache_hits={len(rows) - len(missing)}")
    print(f"missing_rows={len(missing)}")
    print(f"anonymous_model_requests={len(keys)}")

    if args.dry:
        for key in keys[:20]:
            print(f"sample_anonymous_prompt={build_user_prompt(key)}")
        return

    client = AsyncOpenAI(
        api_key=EMBEDDING_CONFIG["api_key"],
        base_url=EMBEDDING_CONFIG["base_url"],
        http_client=httpx.AsyncClient(trust_env=False),
    )
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [asyncio.create_task(generate_tags(client, sem, key)) for key in keys]
    generated: dict[str, list[str]] = {}
    done = 0
    for fut in asyncio.as_completed(tasks):
        key, tags = await fut
        generated[key] = tags
        done += 1
        if done <= 5 or done % 20 == 0 or done == len(keys):
            print(f"generated={done}/{len(keys)} key={key} tags={len(tags)}")
        with ANON_CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"anon_key": key, "tags": tags}, ensure_ascii=False) + "\n")

    anon_cache.update(generated)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "recipes"
    ws.append(["id", "名称", "图片地址", "食材", "lang", "标签"])

    tagged = 0
    anon_tagged = 0
    for row in rows:
        uid = row_uid(row)
        tags = strong_cache.get(uid)
        if tags:
            tagged += 1
        else:
            tags = anon_cache.get(anon_key(row), [])
            if tags:
                tagged += 1
                anon_tagged += 1
        ws.append([row["id"], row["name"], row["image"], row["ing"], row["lang"], "、".join(tags)])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"out={out}")
    print(f"tagged_rows={tagged}")
    print(f"anonymized_tagged_rows={anon_tagged}")
    print(f"missing_after={len(rows) - tagged}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(ROOT / "tmp" / "recipes_merged_for_ingest.xlsx"))
    parser.add_argument("--out", default=str(ROOT / "tmp" / "recipes_merged_anonymized_enriched_for_ingest.xlsx"))
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--ignore-strong-cache", action="store_true", help="全部使用匿名大类标签，避免旧缓存误命中")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
