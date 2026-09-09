"""
Embedding Module - 向量化模型
通过阿里云百炼 text-embedding-v4 进行文本向量化
"""
import os
from typing import List, Union, Optional
import numpy as np
import httpx
from openai import OpenAI
from loguru import logger


class EmbeddingModel:
    """阿里云百炼 text-embedding-v4 向量化模型封装"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
        timeout: int = 30
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置，请设置环境变量或在 config.py 中配置")

        # trust_env=False：忽略环境里的 SOCKS/HTTP 代理，直连域内的 DashScope
        # （否则本机配了代理但 venv 没装 socksio 时会报错）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=httpx.Client(trust_env=False),
        )
        self.model = model

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 10,   # DashScope text-embedding-v4 单批上限为 10
        max_length: int = 8192,
        **kwargs
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        logger.debug(f"正在编码 {len(texts)} 条文本...")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def compute_similarity(
        self,
        embeddings_1: np.ndarray,
        embeddings_2: np.ndarray
    ) -> np.ndarray:
        norm1 = embeddings_1 / np.linalg.norm(embeddings_1, axis=1, keepdims=True)
        norm2 = embeddings_2 / np.linalg.norm(embeddings_2, axis=1, keepdims=True)
        return norm1 @ norm2.T

    def get_embedding_dimension(self) -> int:
        return 1024


_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    api_key: Optional[str] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: str = "text-embedding-v4",
    timeout: int = 30,
    force_reload: bool = False
) -> EmbeddingModel:
    global _embedding_model

    if _embedding_model is None or force_reload:
        _embedding_model = EmbeddingModel(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout
        )

    return _embedding_model


def encode_texts(
    texts: Union[str, List[str]],
    batch_size: int = 12,
    max_length: int = 8192
) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(texts, batch_size=batch_size, max_length=max_length)
