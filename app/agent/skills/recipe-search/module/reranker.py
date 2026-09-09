"""
Reranker Module — DashScope gte-rerank-v2 重排（精排）

混合召回(hybrid top-N) → 交叉编码重排 → top-k。复用 DASHSCOPE_API_KEY。
注意：rerank 走 DashScope 自有端点（不是 OpenAI 兼容的 /compatible-mode/v1）。

设计原则：**重排绝不阻断检索**。任何失败（无 key / 网络 / 非 200 / 配额）
都返回 None，由调用方回退到召回原序。
"""
import os
from typing import List, Optional, Tuple

import httpx
from loguru import logger

DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class Reranker:
    """DashScope text-rerank 封装（默认 gte-rerank-v2）。"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gte-rerank-v2",
                 endpoint: str = DEFAULT_ENDPOINT, timeout: int = 10):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def rerank(self, query: str, documents: List[str], top_n: int) -> Optional[List[Tuple[int, float]]]:
        """对 documents 按与 query 的相关性重排。

        Returns:
            [(原始下标, relevance_score), ...] 按相关性降序；失败返回 None。
        """
        if not self.api_key or not documents:
            return None
        body = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": min(top_n, len(documents)), "return_documents": False},
        }
        try:
            r = httpx.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=body, timeout=self.timeout,
                # trust_env=False：忽略本机代理，直连 DashScope（与 embedding 一致）
                trust_env=False,
            )
            if r.status_code != 200:
                logger.warning(
                    f"[Rerank] HTTP {r.status_code}，回退召回原序：response_chars={len(r.text)}"
                )
                return None
            results = r.json().get("output", {}).get("results", [])
            return [(int(it["index"]), float(it["relevance_score"])) for it in results]
        except Exception as e:
            logger.warning(f"[Rerank] 调用异常，回退召回原序：error_type={type(e).__name__}")
            return None


_reranker: Optional[Reranker] = None


def get_reranker(api_key: Optional[str] = None, model: str = "gte-rerank-v2",
                 endpoint: str = DEFAULT_ENDPOINT, timeout: int = 10) -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker(api_key=api_key, model=model, endpoint=endpoint, timeout=timeout)
    return _reranker


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[5] / ".env")  # module/ 比技能根深一层
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import EMBEDDING_CONFIG, RERANK_CONFIG

    rr = get_reranker(api_key=EMBEDDING_CONFIG["api_key"], model=RERANK_CONFIG["model"],
                      endpoint=RERANK_CONFIG["endpoint"], timeout=RERANK_CONFIG["timeout"])
    q = "酸辣土豆丝"
    docs = ["红烧肉。食材：五花肉、酱油", "炒土豆丝。食材：土豆、青椒",
            "宫保鸡丁。食材：鸡肉、花生", "醋溜土豆丝。食材：土豆、醋、干辣椒"]
    out = rr.rerank(q, docs, top_n=3)
    print("rerank:", out)
    if out:
        for idx, sc in out:
            print(f"  {sc:.4f}  {docs[idx]}")
