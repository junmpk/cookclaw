"""Demo recipe retrieval adapters.

``HybridRecipeStore`` delegates to CookClaw's production-style retrieval service
(dense + BM25 + RRF + rerank). ``RecipeStore`` remains an explicitly labelled,
offline rehearsal fallback so a fresh public clone can still be demonstrated
without distributing the private recipe dataset.
"""
import asyncio
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any

from .models import Recipe

DIMENSION = 256
COLLECTION = "cookclaw_public_demo_v1"
TRANSLATIONS = {
    "鸡": "chicken", "西兰花": "broccoli", "鸡蛋": "egg", "汤": "soup",
    "蔬菜": "vegetable", "豆": "bean", "土豆": "potato", "胡萝卜": "carrot",
}


def vector(text: str) -> list[float]:
    for chinese, english in TRANSLATIONS.items():
        text = text.replace(chinese, " " + english + " ")
    result = [0.0] * DIMENSION
    for token in re.findall(r"[a-z]+|[\u4e00-\u9fff]", text.lower()):
        slot = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % DIMENSION
        result[slot] += 1
    norm = math.sqrt(sum(n * n for n in result)) or 1
    return [n / norm for n in result]


class RecipeStore:
    backend_name = "public_rehearsal"

    def __init__(self, path: Path):
        self.path = path
        self._client = None
        self._gate = asyncio.Lock()

    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient
            self._client = MilvusClient(uri=str(self.path))
        return self._client

    def _search(self, query: str, limit: int) -> list[Recipe]:
        if not self.path.exists():
            raise RuntimeError("食谱库未准备，请先运行 python -m app.demo.seed")
        client = self.client()
        if not client.has_collection(COLLECTION):
            raise RuntimeError("公开演示集合不存在，请先运行 python -m app.demo.seed")
        client.load_collection(COLLECTION)
        groups = client.search(
            collection_name=COLLECTION, data=[vector(query)], limit=limit,
            output_fields=["recipe"], search_params={"metric_type": "COSINE"},
        )
        return [Recipe.model_validate(hit["entity"]["recipe"]) for hit in groups[0]]

    async def search(self, query: str, limit: int = 12) -> list[Recipe]:
        async with self._gate:
            return await asyncio.to_thread(self._search, query, min(max(limit, 1), 16))

    def close(self):
        if self._client:
            self._client.close()


def _text_values(values: Any) -> list[str]:
    result: list[str] = []
    for item in values or []:
        value = (
            str(
                item.get("name")
                or item.get("ingredient")
                or item.get("description")
                or item.get("content")
                or ""
            ).strip()
            if isinstance(item, dict)
            else str(item or "").strip()
        )
        if value and value not in result:
            result.append(value)
    return result


def _recipe_kind(name: str, metadata: dict[str, Any]) -> str:
    facets = metadata.get("facets") if isinstance(metadata.get("facets"), dict) else {}
    labels = [
        name,
        *[str(item) for item in metadata.get("tags") or []],
        *[str(item) for item in facets.get("meal") or []],
        *[str(item) for item in facets.get("dish_type") or []],
    ]
    value = " ".join(labels).casefold()
    return "soup" if "soup" in value or "汤" in value or "羹" in value else "dish"


def recipe_from_search_hit(hit: dict[str, Any]) -> Recipe:
    """Hydrate a graph-safe recipe from a grounded core-search hit."""
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else hit
    detail = metadata.get("recipe_detail") if isinstance(metadata.get("recipe_detail"), dict) else {}
    recipe_id = str(hit.get("id") or metadata.get("recipe_id") or detail.get("recipe_id") or "").strip()
    name = str(metadata.get("name") or detail.get("name") or "").strip()
    if not recipe_id or not name:
        raise ValueError("hybrid search hit is missing grounded recipe identity")
    ingredients = _text_values(metadata.get("ingredients") or detail.get("ingredients"))
    steps = _text_values(detail.get("steps"))
    source = str(detail.get("source") or metadata.get("source") or "CookClaw recipe collection").strip()
    nutrition = detail.get("nutrition") if isinstance(detail.get("nutrition"), dict) else {}
    return Recipe(
        id=recipe_id,
        name=name,
        source=source,
        image_url=str(metadata.get("image_url") or detail.get("image_url") or "").strip() or None,
        ingredients=ingredients,
        steps=steps,
        kind=_recipe_kind(name, metadata),
        nutrition={str(key): str(value) for key, value in nutrition.items()},
        ingredients_complete=bool(detail.get("ingredients_complete")),
        detail_basis="source" if steps else "missing",
    )


def recipes_from_search_result(search_result: dict[str, Any] | None) -> list[Recipe]:
    """把共享 Runtime 已取得的 grounded RAG 结果交给 Graph，避免重复检索。"""
    recipes: list[Recipe] = []
    for hit in (search_result or {}).get("results") or []:
        try:
            recipe = recipe_from_search_hit(hit)
        except (TypeError, ValueError):
            continue
        if recipe.id not in {item.id for item in recipes}:
            recipes.append(recipe)
    return recipes[:16]


class HybridRecipeStore:
    """Adapter shared by chat and dynamically selected agents; never invents fallback hits."""

    backend_name = "hybrid_rag"

    async def search(self, query: str, limit: int = 12) -> list[Recipe]:
        from app.agent.recipe_search_service import search

        result = await search(query, top_k=min(max(1, int(limit)), 16))
        if not result or not result.get("success"):
            raise RuntimeError("CookClaw hybrid retrieval is unavailable")
        recipes: list[Recipe] = []
        for hit in result.get("results") or []:
            try:
                recipe = recipe_from_search_hit(hit)
            except (TypeError, ValueError):
                continue
            if recipe.id not in {item.id for item in recipes}:
                recipes.append(recipe)
        if not recipes:
            raise RuntimeError("CookClaw hybrid retrieval returned no grounded recipes")
        return recipes

    def close(self):
        return None


def _configured_milvus_exists() -> bool:
    uri = str(os.getenv("RECIPE_MILVUS_URI") or "").strip()
    if not uri:
        return False
    if uri.startswith(("http://", "https://", "tcp:", "unix:")):
        return True
    path = Path(uri).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / "agent" / "skills" / "recipe-search" / path
    return path.exists()


def build_recipe_store(path: Path, backend: str | None = None):
    """Select one explicit backend. ``auto`` prefers real hybrid RAG when ready."""
    selected = str(backend or os.getenv("DEMO_RECIPE_BACKEND") or "auto").strip().lower()
    if selected == "auto":
        selected = "hybrid" if _configured_milvus_exists() else "public_rehearsal"
    if selected in {"hybrid", "hybrid_rag"}:
        return HybridRecipeStore()
    if selected in {"public", "public_demo", "public_rehearsal"}:
        return RecipeStore(path)
    raise ValueError(f"Unsupported DEMO_RECIPE_BACKEND: {selected}")


def ingest(path: Path, recipes: list[Recipe]) -> int:
    from pymilvus import MilvusClient
    path.parent.mkdir(parents=True, exist_ok=True)
    client = MilvusClient(uri=str(path))
    try:
        if not client.has_collection(COLLECTION):
            client.create_collection(
                collection_name=COLLECTION, dimension=DIMENSION, id_type="string",
                max_length=256, metric_type="COSINE", auto_id=False,
            )
        client.upsert(collection_name=COLLECTION, data=[{
            "id": r.id, "vector": vector(r.name + " " + " ".join(r.ingredients)),
            "recipe": r.model_dump(),
        } for r in recipes])
        return len(recipes)
    finally:
        client.close()
