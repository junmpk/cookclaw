"""
Recipe Search 配置文件

敏感信息（Milvus 地址、Embedding API Key）从环境变量读取，
请在项目根目录 .env 文件中配置：
    RECIPE_MILVUS_URI=recipe_milvus.db   # 本地 Milvus Lite 文件；或 http://host:19530 连远程
    MILVUS_HOST=...                      # （可选，旧式）填了会拼成 http://host:port
    MILVUS_PORT=19530
    DASHSCOPE_API_KEY=sk-...
    DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
"""

import os
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent


def _get_env(key: str, default: str = "") -> str:
    """读取环境变量"""
    return os.getenv(key, default)


def _resolve_milvus_uri() -> str:
    """解析 Milvus 连接 URI。

    优先级：MILVUS_URI > MILVUS_HOST(:PORT) > 本地 Lite 文件(默认)。
    本地相对路径基于本技能目录解析为绝对路径，避免受运行时 CWD 影响。
    """
    # 注意：变量名不能用 MILVUS_URI —— 那是 pymilvus 的保留环境变量，
    # 它在 import 时按 http 地址校验，填本地文件路径会导致 pymilvus 导入即崩。
    uri = _get_env("RECIPE_MILVUS_URI", "").strip()
    if uri:
        if uri.startswith(("http://", "https://", "unix:", "tcp:")):
            return uri
        p = Path(uri)
        return str(p if p.is_absolute() else _SKILL_DIR / p)
    host = _get_env("MILVUS_HOST", "").strip()
    if host:
        return f"http://{host}:{_get_env('MILVUS_PORT', '19530')}"
    # 默认：本技能目录下的本地 Milvus Lite 文件
    return str(_SKILL_DIR / "recipe_milvus.db")


MILVUS_CONFIG = {
    "uri": _resolve_milvus_uri(),
    "host": _get_env("MILVUS_HOST", ""),
    "port": int(_get_env("MILVUS_PORT", "19530")),
    # 纯语义基线集合（评测用）
    "collection_name": _get_env("MILVUS_COLLECTION", "recipe_collection"),
    # 混合检索集合（线上查询用）：dense + BM25 稀疏 + 结构化 facets
    "hybrid_collection": _get_env("MILVUS_HYBRID_COLLECTION", "recipe_hybrid"),
    "vector_dim": int(_get_env("MILVUS_VECTOR_DIM", "1024")),
}

# 多语言分集合路由（route ②）：开启后按 query 语言路由到 recipe_hybrid_{lang} 独立集合，
# 语言由集合隔离（建库期分离），检索不再拼 metadata["lang"] 过滤；BM25/向量各语言纯净。
# 默认关 → 走现状单集合 recipe_hybrid + lang 元数据过滤（不破坏现网，可灰度切换）。
SPLIT_LANG_ENABLED = _get_env("MILVUS_SPLIT_LANG", "0") != "0"


def hybrid_collection_for(lang: str) -> str:
    """按语言返回 hybrid 集合名。
    分集合模式：recipe_hybrid_<lang>（任意语言，如 _zh/_en/_es/_ja）；
    单集合模式（默认）：recipe_hybrid（语言走 metadata 过滤，兼容现状）。
    """
    base = MILVUS_CONFIG["hybrid_collection"]
    if SPLIT_LANG_ENABLED:
        return f"{base}_{lang or 'zh'}"   # 任意语言集合；空兜底 zh
    return base

# 混合召回参数
HYBRID_CONFIG = {
    "enabled": _get_env("HYBRID_ENABLED", "1") != "0",
    "recall_k": int(_get_env("HYBRID_RECALL_K", "30")),  # 每路融合前召回深度
    "rrf_k": int(_get_env("HYBRID_RRF_K", "60")),         # RRF 常数
}

# 元数据过滤策略（详见 normalize.py 的硬/软口径）
#   硬过滤(违反=事故)：lang(按 query 语言) / 素食 / 忌口
#   软维度(cuisine/main_ingredient/...)默认不硬过滤，交给 hybrid 排序
FILTER_CONFIG = {
    "lang_auto": _get_env("FILTER_LANG_AUTO", "1") != "0",   # 按 query 语言过滤 zh/en，去跨语言重复
    "diet_hard": True,                                        # "要素食" → 硬过滤 facets.diet
    "exclude_hard": True,                                     # "无花生" → Python 子串硬过滤食材
    "hard_facet_dims": [],                                    # 需硬过滤的软维度（默认空=全软）
}

# 重排（精排）：混合召回候选池 → DashScope gte-rerank-v2 → top_k
RERANK_CONFIG = {
    "enabled": _get_env("RERANK_ENABLED", "1") != "0",
    "model": _get_env("RERANK_MODEL", "gte-rerank-v2"),      # gte-rerank(v1) 该账号未开通
    "endpoint": _get_env(
        "RERANK_ENDPOINT",
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"),
    "candidate_k": int(_get_env("RERANK_CANDIDATE_K", "20")),  # 送进重排的召回池大小
    "timeout": int(_get_env("RERANK_TIMEOUT", "10")),
}

# 查询理解：口语化 query → 结构化检索意图（qwen-plus）。短 query 走规则省延迟。
# 默认关：实测对召回零增量（hybrid+rerank 已够鲁棒），多词忌口已用规则修好；
# 需要口语鲁棒性/隐含约束理解时 QU_ENABLED=1 开。
QU_CONFIG = {
    "enabled": _get_env("QU_ENABLED", "0") != "0",
    "model": _get_env("QU_MODEL", "qwen-plus"),
    "min_chars": int(_get_env("QU_MIN_CHARS", "9")),   # query 长度 ≥ 此值才调 LLM
    "timeout": int(_get_env("QU_TIMEOUT", "8")),
}

EMBEDDING_CONFIG = {
    "api_key": _get_env("DASHSCOPE_API_KEY", ""),
    "base_url": _get_env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "model": _get_env("EMBEDDING_MODEL", "text-embedding-v4"),
    "timeout": int(_get_env("EMBEDDING_TIMEOUT", "30")),
}

SEARCH_CONFIG = {
    "default_top_k": 3,
    "max_top_k": 20,
    "min_similarity_score": 0.5,
}

DISPLAY_CONFIG = {
    "max_display_results": 3,
    "show_score": True,
    "show_ingredients": True,
    "show_tags": True,
    "format_template": {
        "header": "🍳 为您找到以下食谱：\n\n",
        "item_template": "{index}. **{name}**（相似度：{score}%）\n   🥘 食材：{ingredients}\n   🏷️ 标签：{tags}\n\n",
        "footer": ""
    }
}

LOG_CONFIG = {
    "level": "INFO",
    "format": "{time:HH:mm:ss} | {level} | {message}",
}
