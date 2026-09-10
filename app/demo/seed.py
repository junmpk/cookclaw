"""下载公开来源的原始食谱字段并灌入 Milvus；不生成食谱。

python -m app.demo.seed
数据来自独立 MyPlate.food 的 USDA 食谱存档，原始来源和存档 URL 均保留。
只导入带 Public Domain Mark 的记录；网络失败时不创建占位食谱。
"""
import asyncio
import html
import json
import os
import re
from pathlib import Path

import httpx

from .models import Recipe
from .retrieval import ingest

SOURCES = [
    ("chicken-and-broccoli-bake", "dish"),
    ("broccoli-omelet", "dish"),
    ("bean-and-vegetable-salad", "dish"),
    ("broccoli-baked-potatoes", "dish"),
    ("apple-banana-salad-peanuts", "dish"),
    ("baked-eggs-cheese", "dish"),
    ("quick-chicken-vegetable-soup", "soup"),
    ("vegetable-soup-chicken", "soup"),
]


def parse_recipe(page: str, slug: str, kind: str) -> Recipe:
    blocks = re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', page, re.S)
    for block in blocks:
        data = json.loads(block)
        if data.get("@type") != "Recipe":
            continue
        if "publicdomain/mark/1.0" not in str(data.get("license", "")):
            raise ValueError(f"{slug}: 未找到公开领域许可，停止导入")
        ingredients = data.get("recipeIngredient", [])
        if not ingredients or not data.get("isBasedOn"):
            raise ValueError(f"{slug}: 缺少食材或原始来源")
        image = data.get("image")
        if isinstance(image, list):
            image = image[0] if image else None
        if isinstance(image, dict):
            image = image.get("url")
        return Recipe(
            id=slug, name=html.unescape(data["name"]), kind=kind,
            source=f"https://myplate.food/recipes/{slug}",
            image_url=image if isinstance(image, str) else None,
            ingredients=[html.unescape(x) for x in ingredients],
            steps=[html.unescape(x["text"]) for x in data.get("recipeInstructions", [])],
            nutrition={k: str(v) for k, v in data.get("nutrition", {}).items() if not k.startswith("@")},
            ingredients_complete=True,
        )
    raise ValueError(f"{slug}: 未找到 Recipe JSON-LD")


async def main():
    recipes = []
    provenance = []
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for slug, kind in SOURCES:
            url = f"https://myplate.food/recipes/{slug}"
            response = await client.get(url)
            response.raise_for_status()
            recipes.append(parse_recipe(response.text, slug, kind))
            provenance.append({"archive": url, "original": f"https://www.myplate.gov/recipes/{slug}",
                               "license": "https://creativecommons.org/publicdomain/mark/1.0/"})
            print(f"已核实：{recipes[-1].name}")
    directory = Path(os.getenv("DEMO_DATA_DIR", ".demo"))
    count = await asyncio.to_thread(ingest, directory / "recipes.db", recipes)
    # Runtime artifact, ignored by Git; no credentials or internal data are copied.
    (directory / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(f"已导入 {count} 条真实来源食谱 → {directory / 'recipes.db'}")


if __name__ == "__main__":
    asyncio.run(main())
