"""
Milvus Dataset Module - 向量数据库管理（MilvusClient 统一接口）

同一套代码同时支持：
  - 本地 Milvus Lite：uri 指向本地文件，如 "recipe_milvus.db"
  - 远程 Milvus 服务：uri 形如 "http://host:19530"

检索能力：
  - search()        纯语义（dense / COSINE）—— 保留作为评测基线
  - hybrid_search() 混合召回：dense(COSINE) + 关键词(BM25 稀疏)，RRF 融合，
                    支持 filter 表达式（元数据过滤）。

hybrid 集合 schema（由 create_hybrid_collection 建）：
  id(自增) + text(VARCHAR, jieba 分词) + sparse(BM25 自动生成) +
  dense(FLOAT_VECTOR) + metadata(JSON)
"""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from pymilvus import (
    MilvusClient, DataType, Function, FunctionType,
    AnnSearchRequest, RRFRanker, WeightedRanker,
)
from loguru import logger

# Milvus Lite 支持 jieba，但部分 Milvus Server 构建没有内置该 analyzer。
# 生产环境可通过 MILVUS_ANALYZER_TYPE=standard 使用兼容性更好的标准 analyzer。
JIEBA_ANALYZER = {"type": os.getenv("MILVUS_ANALYZER_TYPE", "jieba")}


def _lookup_device_code(recipe_id) -> int | None:
    """从 JSON 映射查找 recipe_id 对应的设备 code。"""
    try:
        from module.id_mapping import get_device_code
        return get_device_code(recipe_id) if recipe_id is not None else None
    except Exception:
        return None


def safe_milvus_uri(uri: str) -> str:
    """用于日志的 Milvus 地址；去除 userinfo、路径、query 和本地绝对路径。"""
    raw = str(uri or "").strip()
    if raw.startswith(("http://", "https://", "tcp://")):
        try:
            parsed = urlsplit(raw)
            host = parsed.hostname or "-"
            port = f":{parsed.port}" if parsed.port else ""
            return f"{parsed.scheme}://{host}{port}"
        except ValueError:
            return "remote://invalid"
    if raw.startswith("unix:"):
        return "unix:<redacted>"
    return f"lite:{Path(raw).name or '<default>'}"


def create_hybrid_collection(client, name, dim, recreate=False):
    """建混合检索集合：text(分词) + sparse(BM25) + dense(COSINE) + metadata。

    BM25 Function 在入库时自动把 text 转成 sparse 稀疏向量，查询侧无需手动算。
    """
    exists = name in client.list_collections()
    if exists and recreate:
        client.drop_collection(name)
        logger.info(f"已删除旧集合：{name}")
        exists = False
    if exists:
        logger.info(f"集合 {name} 已存在（需重建请传 recreate=True）")
        return

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    # 关键词路：原文文本字段，开启 jieba 分词
    schema.add_field("text", DataType.VARCHAR, max_length=4096,
                     enable_analyzer=True, analyzer_params=JIEBA_ANALYZER)
    # BM25 产出的稀疏向量
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    # 语义路：稠密向量（复用已有 embedding）
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("metadata", DataType.JSON)

    # text --BM25--> sparse
    schema.add_function(Function(
        name="text_bm25",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    ))

    index_params = client.prepare_index_params()
    # ⚠️ sparse 必须用 SPARSE_INVERTED_INDEX，不能用 AUTOINDEX：
    #    milvus-lite 下 AUTOINDEX 的稀疏索引无法跨进程持久化，
    #    新进程 load 会报 "vector column must be FixedSizeList, got binary"。
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    logger.info(f"已创建 hybrid 集合：{name} (dense dim={dim}, COSINE + BM25)")


def build_milvus_client(uri: str) -> MilvusClient:
    """统一构造 MilvusClient：远程 Server 自动带 MILVUS_TOKEN / MILVUS_DB_NAME（若设），
    本地 Lite（文件路径）忽略鉴权。查询(MilvusManager)与灌库脚本共用，保证鉴权口径一致。"""
    kwargs = {"uri": uri}
    if uri.startswith(("http://", "https://", "tcp://", "unix:")):
        token = os.getenv("MILVUS_TOKEN", "").strip()
        db_name = os.getenv("MILVUS_DB_NAME", "").strip()
        if token:
            kwargs["token"] = token
        if db_name:
            kwargs["db_name"] = db_name
    return MilvusClient(**kwargs)


class MilvusManager:
    """Milvus 数据库管理器（基于 MilvusClient，兼容 Lite / Server）"""

    def __init__(self, uri=None, host="localhost", port=19530,
                 collection_name="recipe_collection", vector_dim=1024):
        if not uri:
            uri = f"http://{host}:{port}"
        self.uri = uri
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = None
        self._connect()

    def _connect(self):
        """建立 Milvus 连接（Lite 打开本地文件；远程 Server 自动带鉴权，见 build_milvus_client）"""
        try:
            self.client = build_milvus_client(self.uri)
            logger.info(f"✅ 已连接 Milvus: {safe_milvus_uri(self.uri)}")
            if self.collection_name in self.client.list_collections():
                self.client.load_collection(self.collection_name)
                logger.info(f"✅ 集合就绪（已 load）：{self.collection_name}")
            else:
                logger.warning(f"⚠️ 集合 {self.collection_name} 不存在（需先灌库）")
        except Exception as e:
            logger.error(f"❌ 连接 Milvus 失败：error_type={type(e).__name__}")
            raise

    # ── 结果格式化（search / hybrid_search 共用，保持上层读取契约不变）──
    @staticmethod
    def _format_hits(results):
        formatted = []
        for hits in results:
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                entity = hit.get("entity", hit)
                md = entity.get("metadata", {})
                if isinstance(md, str):
                    try:
                        md = json.loads(md)
                    except Exception:
                        md = {}
                if not isinstance(md, dict):
                    md = {}
                facets = md.get("facets", {})
                if not isinstance(facets, dict):
                    facets = {}
                facet_record_types = facets.get("record_type") or []
                if not isinstance(facet_record_types, list):
                    facet_record_types = [facet_record_types]
                record_type = (
                    str(md.get("record_type") or "").strip()
                    or next(
                        (
                            str(value).strip()
                            for value in facet_record_types
                            if str(value).strip()
                        ),
                        "",
                    )
                )
                hit_id = hit.get("id")
                formatted.append({
                    "metadata": {
                        "name": md.get("name", "未知菜品"),
                        "ingredients": md.get("ingredients", "未知"),
                        "ingredients_raw": md.get("ingredients_raw", md.get("ingredients", [])),
                        "seasonings": md.get("seasonings", []),
                        "tags": md.get("tags", []),
                        "facets": facets,
                        "nutrition": md.get("nutrition", {}),
                        "image_url": md.get("image_url", ""),
                        "description": md.get("description", ""),
                        "estimated_time": md.get("estimated_time"),
                        "difficulty": md.get("difficulty", ""),
                        "tips": md.get("tips", ""),
                        "steps": md.get("steps", []),
                        "recipe_detail": md.get("recipe_detail"),
                        "recipe_id": md.get("recipe_id", hit_id),
                        "device_code": md.get("device_code") or _lookup_device_code(md.get("recipe_id")),
                        "lang": md.get("lang", ""),
                        "record_type": record_type,
                    },
                    "score": float(hit.get("distance", 0.0)),
                    "id": hit_id,
                })
        return formatted

    def search(self, query_embedding, top_k=5, filter_expr=""):
        """纯语义检索（dense / COSINE）—— 评测基线。

        注意：老集合稠密字段名为 `embedding`，hybrid 集合为 `dense`，
        这里自动探测当前集合用哪个字段。
        """
        if self.client is None:
            raise RuntimeError("Milvus 客户端未初始化")
        if self.collection_name not in self.client.list_collections():
            raise RuntimeError(f"集合 {self.collection_name} 不存在，请先灌库")

        anns_field = self._dense_field()
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field=anns_field,
            limit=top_k,
            output_fields=["metadata"],
            search_params={"metric_type": "COSINE"},
            filter=filter_expr or "",
        )
        out = self._format_hits(results)
        # COSINE：milvus 返回的是距离，转成相似度(1=最像)以兼容老的展示约定
        for r in out:
            r["score"] = 1.0 - r["score"]
        return out

    def hybrid_search(self, query_embedding, query_text, top_k=3,
                      filter_expr="", recall_k=30, rrf_k=60, weights=None):
        """混合召回：dense(语义) + sparse(BM25 关键词)，RRF/加权 融合。

        Args:
            query_embedding: 查询稠密向量（语义路）
            query_text:      原始查询字符串（关键词路，服务端 jieba 分词 + BM25）
            top_k:           最终返回条数
            filter_expr:     元数据过滤表达式（两路都生效）
            recall_k:        每路召回深度（融合前），一般 >> top_k
            rrf_k:           RRF 常数（默认 60）
            weights:         (dense_w, sparse_w) 给定则用 WeightedRanker，否则 RRF

        Returns: [{"metadata":{...}, "score": 融合分(越大越相关), "id":...}, ...]
        """
        if self.client is None:
            raise RuntimeError("Milvus 客户端未初始化")
        if self.collection_name not in self.client.list_collections():
            raise RuntimeError(f"集合 {self.collection_name} 不存在，请先灌库")

        dense_req = AnnSearchRequest(
            data=[query_embedding], anns_field="dense",
            param={"metric_type": "COSINE"}, limit=recall_k, expr=filter_expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text], anns_field="sparse",
            param={"metric_type": "BM25"}, limit=recall_k, expr=filter_expr or None,
        )
        ranker = WeightedRanker(*weights) if weights else RRFRanker(rrf_k)

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=top_k,
            output_fields=["metadata"],
        )
        return self._format_hits(results)

    def _dense_field(self):
        """探测当前集合的稠密向量字段名（老库=embedding，新库=dense）。"""
        try:
            fields = self.client.describe_collection(self.collection_name).get("fields", [])
            names = {f.get("name") for f in fields}
            return "dense" if "dense" in names else "embedding"
        except Exception:
            return "dense"


_milvus_managers = {}


def get_milvus_manager(uri=None, host=None, port=None,
                       collection_name="recipe_collection", vector_dim=1024,
                       force_fresh=None):
    """获取 Milvus 管理器实例。

    force_fresh=True：每次新建连接（子进程逐请求模式）。
    force_fresh=False：按 uri:collection 复用缓存连接（进程内常驻服务：避免每查询重连/重 load，
                      迁到 Milvus Server 后尤其省下逐查询的 grpc 握手）。
    未显式指定时由环境变量 MILVUS_REUSE_CONN 决定（=1 复用，默认新建以兼容子进程模式）。
    """
    if force_fresh is None:
        force_fresh = os.getenv("MILVUS_REUSE_CONN", "0") != "1"
    if force_fresh:
        logger.debug("🔄 创建新的数据库连接（实时查询模式）")
        return MilvusManager(uri=uri, host=host or "localhost", port=port or 19530,
                             collection_name=collection_name, vector_dim=vector_dim)
    key = f"{uri}:{collection_name}"
    if key not in _milvus_managers:
        _milvus_managers[key] = MilvusManager(
            uri=uri, host=host or "localhost", port=port or 19530,
            collection_name=collection_name, vector_dim=vector_dim)
    return _milvus_managers[key]
