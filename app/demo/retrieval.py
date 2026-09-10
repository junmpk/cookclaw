"""可复现的本地 Milvus 检索。哈希向量是演示基线，不冒充语义 Embedding。"""
import asyncio
import hashlib
import math
import re
from pathlib import Path

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
