"""通用用户画像事实的 Milvus 派生索引。

PostgreSQL ``channel_users.long_term_memory.facts`` 始终是事实源；本模块只保存
匿名画像哈希、事实 ID 和规范化文本，用于相关性召回。索引丢失或暂时不可用时，
不得影响 PostgreSQL 中事实的写入、纠错和删除。
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Iterable, Protocol

import httpx
from openai import OpenAI
from pymilvus import DataType, MilvusClient

from app.conversation.profile_facts import canonical_stored_profile_fact_text


_REMOTE_MILVUS_SCHEMES = ("http://", "https://", "tcp://", "unix:")
_COLLECTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")
_FACT_ID_RE = re.compile(r"^mem_[a-f0-9]{24}$")


def profile_memory_hash(profile_key: str) -> str:
    """生成 Milvus 分区键；索引中不保存通道用户 ID 或原始 profile key。"""
    return hashlib.sha256(str(profile_key or "").encode("utf-8")).hexdigest()


def is_server_milvus_uri(uri: str) -> bool:
    return str(uri or "").strip().lower().startswith(_REMOTE_MILVUS_SCHEMES)


@dataclass(frozen=True)
class MemoryIndexHit:
    fact_id: str
    score: float


class MemoryEmbedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class DashScopeMemoryEmbedder:
    """延迟初始化的 DashScope embedding 客户端。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "").strip()
        self.model = str(model or "").strip()
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self._client: OpenAI | None = None
        self._lock = threading.Lock()

    def _get_client(self) -> OpenAI:
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for memory embeddings")
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    http_client=httpx.Client(trust_env=False),
                )
        return self._client

    def encode(self, texts: list[str]) -> list[list[float]]:
        values = [str(item or "").strip() for item in texts if str(item or "").strip()]
        if not values:
            return []
        output: list[list[float]] = []
        client = self._get_client()
        # DashScope text-embedding-v4 单批最多 10 条。
        for offset in range(0, len(values), 10):
            response = client.embeddings.create(
                model=self.model,
                input=values[offset:offset + 10],
            )
            output.extend([list(item.embedding) for item in response.data])
        return output

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class MilvusProfileMemoryIndex:
    """独立集合上的异步 facade；阻塞 Milvus/Embedding 调用在线程池执行。"""

    def __init__(
        self,
        *,
        uri: str,
        collection_name: str = "cookclaw_user_memory_v1",
        vector_dim: int = 1024,
        token: str = "",
        db_name: str = "",
        timeout_seconds: float = 10.0,
        concurrency: int = 4,
        embedder: MemoryEmbedder | None = None,
        client=None,
        require_server: bool = True,
    ) -> None:
        self.uri = str(uri or "").strip()
        self.collection_name = str(collection_name or "").strip()
        self.vector_dim = max(1, int(vector_dim))
        self.token = str(token or "").strip()
        self.db_name = str(db_name or "").strip()
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        if not _COLLECTION_RE.fullmatch(self.collection_name):
            raise ValueError("invalid MEMORY_MILVUS_COLLECTION")
        if require_server and not is_server_milvus_uri(self.uri):
            raise ValueError(
                "general memory requires RECIPE_MILVUS_URI/MEMORY_MILVUS_URI "
                "to point to Milvus Server"
            )
        recipe_collections = {
            os.getenv("MILVUS_COLLECTION", "recipe_collection").strip(),
            os.getenv("MILVUS_HYBRID_COLLECTION", "recipe_hybrid").strip(),
        }
        if self.collection_name in recipe_collections:
            raise ValueError("memory index must use an independent Milvus collection")
        self.embedder = embedder or DashScopeMemoryEmbedder(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
            timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT", "30")),
        )
        self._client = client
        self._initialized = False
        self._target_created_during_initialize = False
        self._sync_init_lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    @classmethod
    def from_env(cls, *, require_server: bool = True) -> "MilvusProfileMemoryIndex":
        recipe_uri = os.getenv("RECIPE_MILVUS_URI", "").strip()
        # 即使配置了专用覆盖，测试/生产启用通用记忆时也必须先确认食谱库
        # 本身已切到 Server，避免两个 Milvus 部署目标悄然漂移。
        if require_server and not is_server_milvus_uri(recipe_uri):
            raise ValueError("RECIPE_MILVUS_URI must point to Milvus Server")
        uri = os.getenv("MEMORY_MILVUS_URI", "").strip() or recipe_uri
        return cls(
            uri=uri,
            collection_name=os.getenv(
                "MEMORY_MILVUS_COLLECTION",
                "cookclaw_user_memory_v1",
            ),
            vector_dim=int(os.getenv("MEMORY_MILVUS_VECTOR_DIM", "1024")),
            token=os.getenv("MILVUS_TOKEN", ""),
            db_name=os.getenv("MILVUS_DB_NAME", ""),
            timeout_seconds=float(os.getenv("MEMORY_MILVUS_TIMEOUT_SECONDS", "10")),
            concurrency=int(os.getenv("MEMORY_MILVUS_CONCURRENCY", "4")),
            require_server=require_server,
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        kwargs: dict[str, object] = {
            "uri": self.uri,
            "timeout": self.timeout_seconds,
        }
        if self.token:
            kwargs["token"] = self.token
        if self.db_name:
            kwargs["db_name"] = self.db_name
        self._client = MilvusClient(**kwargs)
        return self._client

    def _validate_existing_schema(self, client) -> None:
        description = client.describe_collection(self.collection_name)
        fields = {
            str(field.get("name") or ""): field
            for field in (description.get("fields") or [])
            if isinstance(field, dict)
        }
        required = {"fact_id", "profile_hash", "category", "key", "text", "updated_at", "dense"}
        missing = sorted(required.difference(fields))
        if missing:
            raise RuntimeError(
                "memory collection schema is incompatible: missing=" + ",".join(missing)
            )
        dense = fields["dense"]
        params = dense.get("params") if isinstance(dense.get("params"), dict) else {}
        dimension = int(params.get("dim") or dense.get("dim") or 0)
        if dimension and dimension != self.vector_dim:
            raise RuntimeError(
                f"memory collection vector dimension mismatch: {dimension} != {self.vector_dim}"
            )

    def _create_collection_sync(self, client, collection_name: str) -> None:
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            "fact_id",
            DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field("profile_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("category", DataType.VARCHAR, max_length=32)
        schema.add_field("key", DataType.VARCHAR, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=512)
        schema.add_field("updated_at", DataType.DOUBLE)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.vector_dim)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            timeout=self.timeout_seconds,
        )

    def _initialize_sync(self) -> None:
        if self._initialized:
            return
        with self._sync_init_lock:
            if self._initialized:
                return
            client = self._get_client()
            if self.collection_name not in client.list_collections():
                self._create_collection_sync(client, self.collection_name)
                self._target_created_during_initialize = True
            self._validate_existing_schema(client)
            client.load_collection(self.collection_name)
            self._initialized = True

    async def initialize(self) -> None:
        async with self._semaphore:
            await asyncio.to_thread(self._initialize_sync)

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        raw = self.embedder.encode(texts)
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        vectors = [list(vector) for vector in raw]
        if len(vectors) != len(texts):
            raise RuntimeError("embedding result count mismatch")
        if any(len(vector) != self.vector_dim for vector in vectors):
            raise RuntimeError("embedding vector dimension mismatch")
        return vectors

    def _upsert_sync(self, profile_key: str, facts: list[dict]) -> int:
        self._initialize_sync()
        records: list[dict] = []
        safe_facts = [item for item in facts if isinstance(item, dict)]
        texts = [canonical_stored_profile_fact_text(item) for item in safe_facts]
        vectors = self._vectors(texts) if texts else []
        partition = profile_memory_hash(profile_key)
        for fact, text, vector in zip(safe_facts, texts, vectors):
            fact_id = str(fact.get("id") or "")
            if not _FACT_ID_RE.fullmatch(fact_id) or not text.strip():
                continue
            records.append({
                "fact_id": fact_id,
                "profile_hash": partition,
                "category": str(fact.get("category") or "")[:32],
                "key": str(fact.get("key") or "")[:64],
                "text": text[:512],
                "updated_at": float(fact.get("updated_at") or 0),
                "dense": vector,
            })
        if not records:
            return 0
        self._get_client().upsert(
            collection_name=self.collection_name,
            data=records,
            timeout=self.timeout_seconds,
        )
        return len(records)

    async def upsert(self, profile_key: str, facts: Iterable[dict]) -> int:
        values = [dict(item) for item in facts if isinstance(item, dict)]
        if not values:
            return 0
        async with self._semaphore:
            return await asyncio.to_thread(self._upsert_sync, profile_key, values)

    @staticmethod
    def _id_filter(profile_key: str, fact_ids: Iterable[str]) -> str:
        ids = sorted({str(item) for item in fact_ids if _FACT_ID_RE.fullmatch(str(item))})
        if not ids:
            return ""
        quoted = ", ".join(f'"{item}"' for item in ids)
        return (
            f'profile_hash == "{profile_memory_hash(profile_key)}" '
            f"and fact_id in [{quoted}]"
        )

    def _delete_sync(self, profile_key: str, fact_ids: list[str]) -> int:
        self._initialize_sync()
        expression = self._id_filter(profile_key, fact_ids)
        if not expression:
            return 0
        self._get_client().delete(
            collection_name=self.collection_name,
            filter=expression,
            timeout=self.timeout_seconds,
        )
        return len(set(fact_ids))

    async def delete(self, profile_key: str, fact_ids: Iterable[str]) -> int:
        values = [str(item) for item in fact_ids]
        if not values:
            return 0
        async with self._semaphore:
            return await asyncio.to_thread(self._delete_sync, profile_key, values)

    def _delete_profile_sync(self, profile_key: str) -> None:
        self._initialize_sync()
        self._get_client().delete(
            collection_name=self.collection_name,
            filter=f'profile_hash == "{profile_memory_hash(profile_key)}"',
            timeout=self.timeout_seconds,
        )

    async def delete_profile(self, profile_key: str) -> None:
        async with self._semaphore:
            await asyncio.to_thread(self._delete_profile_sync, profile_key)

    def _search_sync(self, profile_key: str, query: str, limit: int) -> list[MemoryIndexHit]:
        self._initialize_sync()
        vector = self._vectors([query])[0]
        results = self._get_client().search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="dense",
            filter=f'profile_hash == "{profile_memory_hash(profile_key)}"',
            limit=max(1, int(limit)),
            output_fields=["fact_id"],
            search_params={"metric_type": "COSINE"},
            timeout=self.timeout_seconds,
        )
        hits: list[MemoryIndexHit] = []
        for group in results or []:
            for hit in group or []:
                if not isinstance(hit, dict):
                    continue
                entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else hit
                fact_id = str(entity.get("fact_id") or hit.get("id") or "")
                if not _FACT_ID_RE.fullmatch(fact_id):
                    continue
                hits.append(MemoryIndexHit(
                    fact_id=fact_id,
                    score=float(hit.get("distance") or 0),
                ))
        return hits

    async def search(
        self,
        profile_key: str,
        query: str,
        *,
        limit: int = 8,
    ) -> list[MemoryIndexHit]:
        value = " ".join(str(query or "").split()).strip()
        if not value:
            return []
        async with self._semaphore:
            return await asyncio.to_thread(
                self._search_sync,
                profile_key,
                value,
                max(1, int(limit)),
            )

    def _permission_probe_sync(self) -> None:
        """验证独立集合 create 及目标集合 upsert/delete 权限。"""
        self._initialize_sync()
        client = self._get_client()
        if not self._target_created_during_initialize:
            # 目标集合已存在时，不能把“能看见集合”误当成当前账号仍有建集合权限。
            # 创建随机空集合后立即 drop；失败会中止初始化，集合名不含用户数据。
            suffix = f"_rbac_probe_{uuid.uuid4().hex[:12]}"
            probe_collection = f"{self.collection_name[:255 - len(suffix)]}{suffix}"
            created = False
            try:
                self._create_collection_sync(client, probe_collection)
                created = True
            finally:
                if created:
                    client.drop_collection(
                        probe_collection,
                        timeout=self.timeout_seconds,
                    )
        probe_id = f"mem_{uuid.uuid4().hex[:24]}"
        probe_partition = "0" * 64
        record = {
            "fact_id": probe_id,
            "profile_hash": probe_partition,
            "category": "system",
            "key": "permission_probe",
            "text": "CookClaw memory permission probe",
            "updated_at": 0.0,
            "dense": [0.0] * self.vector_dim,
        }
        record["dense"][0] = 1.0
        # 先验证 delete 并清掉此前权限异常可能留下的匿名探针，再验证 upsert，
        # 最后删除本轮探针。整个过程不读取或写入任何用户事实。
        client.delete(
            collection_name=self.collection_name,
            filter=f'profile_hash == "{probe_partition}"',
            timeout=self.timeout_seconds,
        )
        client.upsert(
            collection_name=self.collection_name,
            data=[record],
            timeout=self.timeout_seconds,
        )
        client.delete(
            collection_name=self.collection_name,
            filter=f'fact_id == "{probe_id}" and profile_hash == "{probe_partition}"',
            timeout=self.timeout_seconds,
        )

    async def permission_probe(self) -> None:
        async with self._semaphore:
            await asyncio.to_thread(self._permission_probe_sync)

    async def healthcheck(self) -> bool:
        try:
            await self.initialize()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        client, self._client = self._client, None
        self._initialized = False
        self._target_created_during_initialize = False
        if client is not None:
            close = getattr(client, "close", None)
            if close:
                await asyncio.to_thread(close)
        close_embedder = getattr(self.embedder, "close", None)
        if close_embedder:
            await asyncio.to_thread(close_embedder)
