import asyncio

import pytest

from app.conversation.milvus_memory_index import (
    MemoryIndexHit,
    MilvusProfileMemoryIndex,
    profile_memory_hash,
)
from app.conversation.profile_facts import ProfileFactCandidate
from app.conversation.service import ConversationService, build_qq_thread_id
from app.conversation.store import InMemoryConversationStore


class FakeEmbedder:
    def __init__(self, dim=4):
        self.dim = dim
        self.texts = []

    def encode(self, texts):
        self.texts.extend(texts)
        return [[float(index + 1)] + [0.0] * (self.dim - 1) for index, _ in enumerate(texts)]


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeMilvusClient:
    def __init__(self, *, collection="cookclaw_user_memory_v1", dim=4, exists=True):
        self.collection = collection
        self.dim = dim
        self.collections = {collection} if exists else set()
        self.created = []
        self.dropped = []
        self.loaded = []
        self.upserts = []
        self.deletes = []
        self.searches = []
        self.search_result = []
        self.closed = False

    def list_collections(self):
        return list(self.collections)

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self.collections.add(kwargs["collection_name"])

    def describe_collection(self, _name):
        return {
            "fields": [
                {"name": "fact_id"},
                {"name": "profile_hash"},
                {"name": "category"},
                {"name": "key"},
                {"name": "text"},
                {"name": "updated_at"},
                {"name": "dense", "params": {"dim": self.dim}},
            ]
        }

    def drop_collection(self, name, **_kwargs):
        self.dropped.append(name)
        self.collections.discard(name)

    def load_collection(self, name):
        self.loaded.append(name)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        return {"upsert_count": len(kwargs["data"])}

    def delete(self, **kwargs):
        self.deletes.append(kwargs)
        return {"delete_count": 1}

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return self.search_result

    def close(self):
        self.closed = True


def _stored_fact(fact_id="mem_0123456789abcdef01234567"):
    return {
        "id": fact_id,
        "type": "profile_fact",
        "category": "work",
        "key": "occupation",
        "value": "AI应用开发工程师",
        "subject": "self",
        "updated_at": 123.0,
        "evidence": "这段原文不得进入 Milvus",
        "source_thread_hash": "thread-hash",
    }


def _candidate():
    return ProfileFactCandidate(
        operation="upsert",
        category="work",
        key="occupation",
        value="AI应用开发工程师",
        subject="self",
        scope="stable",
        sensitivity="normal",
        evidence="我是AI应用开发工程师",
    )


def test_from_env_requires_recipe_milvus_server_and_independent_collection(monkeypatch):
    monkeypatch.setenv("RECIPE_MILVUS_URI", "recipe_milvus.db")
    with pytest.raises(ValueError, match="RECIPE_MILVUS_URI"):
        MilvusProfileMemoryIndex.from_env()

    monkeypatch.setenv("RECIPE_MILVUS_URI", "http://milvus.internal:19530")
    monkeypatch.setenv("MEMORY_MILVUS_COLLECTION", "recipe_hybrid")
    with pytest.raises(ValueError, match="independent"):
        MilvusProfileMemoryIndex.from_env()


def test_initialize_creates_separate_collection_and_permission_probe():
    async def run():
        client = FakeMilvusClient(exists=False)
        index = MilvusProfileMemoryIndex(
            uri="http://milvus.internal:19530",
            vector_dim=4,
            embedder=FakeEmbedder(),
            client=client,
        )
        await index.initialize()
        await index.permission_probe()

        assert len(client.created) == 1
        assert client.created[0]["collection_name"] == "cookclaw_user_memory_v1"
        assert client.dropped == []
        assert client.loaded == ["cookclaw_user_memory_v1"]
        probe = client.upserts[0]["data"][0]
        assert probe["profile_hash"] == "0" * 64
        assert set(probe) == {
            "fact_id", "profile_hash", "category", "key", "text", "updated_at", "dense"
        }
        assert len(client.deletes) == 2
        assert client.deletes[0]["filter"] == f'profile_hash == "{"0" * 64}"'
        assert "fact_id ==" in client.deletes[1]["filter"]

    asyncio.run(run())


def test_permission_probe_rechecks_create_when_target_collection_exists():
    async def run():
        client = FakeMilvusClient(exists=True)
        index = MilvusProfileMemoryIndex(
            uri="http://milvus.internal:19530",
            vector_dim=4,
            embedder=FakeEmbedder(),
            client=client,
        )
        await index.initialize()
        await index.permission_probe()

        assert len(client.created) == 1
        probe_name = client.created[0]["collection_name"]
        assert probe_name.startswith("cookclaw_user_memory_v1_rbac_probe_")
        assert client.dropped == [probe_name]
        assert probe_name not in client.collections
        assert client.upserts[0]["collection_name"] == "cookclaw_user_memory_v1"
        assert all(
            item["collection_name"] == "cookclaw_user_memory_v1"
            for item in client.deletes
        )

    asyncio.run(run())


def test_upsert_only_contains_canonical_fact_and_anonymous_profile_hash():
    async def run():
        client = FakeMilvusClient()
        embedder = FakeEmbedder()
        index = MilvusProfileMemoryIndex(
            uri="http://milvus.internal:19530",
            vector_dim=4,
            embedder=embedder,
            client=client,
        )
        profile_key = "profile:qq:user-secret"
        assert await index.upsert(profile_key, [_stored_fact()]) == 1

        record = client.upserts[0]["data"][0]
        assert record["profile_hash"] == profile_memory_hash(profile_key)
        assert profile_key not in str(record)
        assert "user-secret" not in str(record)
        assert "原文" not in str(record)
        assert record["text"] == "用户职业：AI应用开发工程师"
        assert set(record) == {
            "fact_id", "profile_hash", "category", "key", "text", "updated_at", "dense"
        }

    asyncio.run(run())


def test_search_always_filters_by_anonymous_profile_hash():
    async def run():
        client = FakeMilvusClient()
        client.search_result = [[
            {
                "id": "mem_0123456789abcdef01234567",
                "distance": 0.91,
                "entity": {"fact_id": "mem_0123456789abcdef01234567"},
            },
            {"id": "not-a-memory-id", "distance": 1.0},
        ]]
        index = MilvusProfileMemoryIndex(
            uri="http://milvus.internal:19530",
            vector_dim=4,
            embedder=FakeEmbedder(),
            client=client,
        )
        profile_key = "profile:qq:user-a"
        hits = await index.search(profile_key, "我的工作", limit=3)

        assert hits == [MemoryIndexHit("mem_0123456789abcdef01234567", 0.91)]
        assert client.searches[0]["filter"] == (
            f'profile_hash == "{profile_memory_hash(profile_key)}"'
        )
        assert profile_key not in client.searches[0]["filter"]

    asyncio.run(run())


def test_milvus_lite_smoke_validates_client_schema_and_crud_contract(tmp_path):
    """Server 权限由部署探针验证；Lite 只用于本地校验 pymilvus 调用契约。"""
    async def run():
        uri = str(tmp_path / "memory-index.db")
        index = MilvusProfileMemoryIndex(
            uri=uri,
            collection_name="test_user_memory",
            vector_dim=4,
            embedder=FakeEmbedder(),
            require_server=False,
        )
        profile_key = "profile:qq:user-a"
        try:
            await index.initialize()
            assert await index.upsert(profile_key, [_stored_fact()]) == 1
            hits = await index.search(profile_key, "我的工作", limit=3)
            assert [item.fact_id for item in hits] == [
                "mem_0123456789abcdef01234567"
            ]
            assert await index.delete(
                profile_key,
                ["mem_0123456789abcdef01234567"],
            ) == 1
            assert await index.search(profile_key, "我的工作", limit=3) == []
        finally:
            await index.close()

        # 第二个实例面对已存在目标集合，实际走临时集合 create/drop 分支；
        # 这里只校验 pymilvus API 契约，不能替代 Server RBAC 证据。
        recheck = MilvusProfileMemoryIndex(
            uri=uri,
            collection_name="test_user_memory",
            vector_dim=4,
            embedder=FakeEmbedder(),
            require_server=False,
        )
        try:
            await recheck.permission_probe()
        finally:
            await recheck.close()

    asyncio.run(run())


class FakeProfileMemoryIndex:
    def __init__(self):
        self.upsert_calls = []
        self.delete_calls = []
        self.hits = []
        self.upsert_error = None
        self.search_error = None

    async def upsert(self, profile_key, facts):
        self.upsert_calls.append((profile_key, [dict(item) for item in facts]))
        if self.upsert_error:
            raise self.upsert_error
        return len(facts)

    async def delete(self, profile_key, fact_ids):
        self.delete_calls.append((profile_key, list(fact_ids)))
        return len(fact_ids)

    async def search(self, _profile_key, _query, *, limit):
        if self.search_error:
            raise self.search_error
        return self.hits[:limit]

    async def close(self):
        return None


def test_service_writes_pg_first_and_post_filters_stale_milvus_hits():
    async def run():
        store = InMemoryConversationStore()
        index = FakeProfileMemoryIndex()
        service = ConversationService(
            store,
            profile_store=store,
            profile_memory_index=index,
            memory_recall_min_score=0.35,
        )
        thread_id = build_qq_thread_id("dm", "user-a", "user-a")
        result = await service.apply_general_profile_facts(thread_id, [_candidate()])
        fact_id = result.upserted[0]["id"]
        index.hits = [
            MemoryIndexHit("mem_aaaaaaaaaaaaaaaaaaaaaaaa", 0.99),
            MemoryIndexHit(fact_id, 0.88),
        ]

        recalled = await service.recall_general_profile_facts(thread_id, "我的工作")
        assert [item["id"] for item in recalled] == [fact_id]
        assert index.upsert_calls[0][0] == "profile:qq:user-a"
        assert (await store.load("profile:qq:user-a")).long_term_facts

    asyncio.run(run())


def test_index_failure_does_not_rollback_pg_and_recall_falls_back_to_pg():
    async def run():
        store = InMemoryConversationStore()
        index = FakeProfileMemoryIndex()
        index.upsert_error = OSError("milvus unavailable")
        service = ConversationService(
            store,
            profile_store=store,
            profile_memory_index=index,
        )
        thread_id = build_qq_thread_id("dm", "user-a", "user-a")
        await service.apply_general_profile_facts(thread_id, [_candidate()])
        assert len(await service.general_profile_facts(thread_id)) == 1

        index.search_error = OSError("milvus unavailable")
        recalled = await service.recall_general_profile_facts(thread_id, "我的工作")
        assert [item["value"] for item in recalled] == ["AI应用开发工程师"]

    asyncio.run(run())


def test_clear_removes_pg_facts_before_best_effort_index_cleanup():
    async def run():
        store = InMemoryConversationStore()
        index = FakeProfileMemoryIndex()
        service = ConversationService(
            store,
            profile_store=store,
            profile_memory_index=index,
        )
        thread_id = build_qq_thread_id("dm", "user-a", "user-a")
        result = await service.apply_general_profile_facts(thread_id, [_candidate()])
        fact_id = result.upserted[0]["id"]

        removed = await service.clear_general_profile_facts(thread_id)
        assert removed == [fact_id]
        assert await service.general_profile_facts(thread_id) == []
        assert index.delete_calls[-1] == ("profile:qq:user-a", [fact_id])

    asyncio.run(run())
