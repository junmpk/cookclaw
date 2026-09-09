"""
食谱检索 —— 进程内常驻服务（route A）。

替代「每请求 fork 一个子进程跑 recipe_search.py」的旧实现：
  - Milvus 连接 / 集合 load / Embedding client 在首次使用（或启动 warmup）时初始化一次，常驻复用；
  - 之后每个请求只做必要的网络调用（embedding + rerank），不再付进程启动 / 重导入 / 重连库的固定开销；
  - 用信号量限制并发：Milvus Lite 是单文件嵌入式库，并发访问不安全，故默认串行(1)；
    迁到 Milvus Server 后，把环境变量 RECIPE_SEARCH_CONCURRENCY 调大即可放开并发。

注意：本模块把检索技能目录加入 sys.path 并直接 import recipe_search_tool，
因此运行主服务的 Python 环境必须装好该技能的运行期依赖
（pymilvus / milvus-lite / jieba / numpy / loguru，见根 pyproject.toml）。
若依赖缺失或初始化失败，warmup()/search() 返回 False/None，
调用方（fast_path._run_search_subprocess）会自动回退到子进程实现，不影响可用性。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# 检索技能目录：把它加入 sys.path 才能 import recipe_search_tool 及其相对依赖
# （recipe_search.py 用裸 import：from config import .../from normalize import .../import preload）
_SKILL_DIR = Path(__file__).resolve().parent / "skills" / "recipe-search"

# 并发上限：Milvus Lite 单文件嵌入式库，多线程并发访问不安全 → 默认串行(1)。
# 切到 Milvus Server 后可调大（如 8）以放开并发。
_CONCURRENCY = max(1, int(os.getenv("RECIPE_SEARCH_CONCURRENCY", "1")))

# 预热查询：启动时跑一次真实检索，提前 load Milvus 集合 + 建好 embedding/rerank client。
_WARMUP_QUERY = os.getenv("RECIPE_SEARCH_WARMUP_QUERY", "鸡蛋")

_recipe_search_tool: Optional[Callable] = None   # 懒加载的技能函数
_sem: Optional[asyncio.Semaphore] = None         # 并发闸门（首次在事件循环内建立）
_load_failed = False                             # import 失败标记，避免每次重试


def _gate() -> asyncio.Semaphore:
    """惰性创建并发信号量（须在运行的事件循环内创建）。"""
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(_CONCURRENCY)
    return _sem


def _load_tool() -> bool:
    """把技能目录加入 sys.path 并 import recipe_search_tool（仅一次）。"""
    global _recipe_search_tool, _load_failed
    if _recipe_search_tool is not None:
        return True
    if _load_failed:
        return False
    try:
        # 确保 .env 已加载到 os.environ（DASHSCOPE_API_KEY / RECIPE_MILVUS_URI 等）。
        # 主服务在 app.core.config 导入时已 load_dotenv；这里再兜一次，便于独立测试。
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        # 常驻模式复用 Milvus 连接（避免每查询重连/重 load）；用户显式设 0 可关。
        os.environ.setdefault("MILVUS_REUSE_CONN", "1")
        if str(_SKILL_DIR) not in sys.path:
            sys.path.insert(0, str(_SKILL_DIR))
        from recipe_search import recipe_search_tool  # type: ignore
        _recipe_search_tool = recipe_search_tool
        logger.info("食谱检索进程内服务已加载（并发上限=%d）", _CONCURRENCY)
        return True
    except Exception as e:
        _load_failed = True
        logger.warning("食谱检索进程内服务加载失败，运行时将回退子进程：%s", e)
        return False


def is_available() -> bool:
    """进程内服务是否可用（已成功加载技能函数）。"""
    return _recipe_search_tool is not None


async def search(query: str, top_k: int = 3, lang: Optional[str] = None) -> Optional[dict]:
    """
    进程内执行一次食谱检索，返回与旧子进程一致的结构 dict
    （{"success": True, "query", "count", "results", "formatted"} 或 {"success": False, ...}）。
    服务不可用或异常时返回 None，由调用方回退子进程。
    lang：可选 zh|en，按原始用户输入显式指定检索语言（覆盖检索内部自动判定）。
    """
    if not _load_tool():
        return None
    try:
        # to_thread：检索内部是同步阻塞调用（Milvus + 两次 HTTP），丢线程池避免阻塞事件循环。
        # 信号量：限制并发（Lite 默认串行），防止突刺打穿。
        async with _gate():
            return await asyncio.to_thread(_recipe_search_tool, query, top_k, lang)
    except Exception as e:
        logger.error("食谱检索进程内执行失败：%s", e, exc_info=True)
        return None


async def warmup() -> bool:
    """
    启动预热：跑一次真实检索，提前建立 Milvus 连接 / load 集合 / 初始化 embedding·rerank client，
    使首个真实请求即命中热实例。非致命——失败仅告警，运行时自动回退子进程。
    """
    if os.getenv("RECIPE_SEARCH_INPROCESS", "1") == "0":
        logger.info("食谱检索进程内服务已禁用（RECIPE_SEARCH_INPROCESS=0），跳过预热")
        return False
    if not _load_tool():
        logger.warning("食谱检索预热跳过：进程内服务不可用（缺依赖？），运行时回退子进程")
        return False
    try:
        res = await search(_WARMUP_QUERY, top_k=1)
        ok = bool(res and res.get("success"))
        if ok:
            logger.info("食谱检索进程内服务预热完成（常驻热实例就绪）")
        else:
            logger.warning("食谱检索预热返回非成功：%s", (res or {}).get("error"))
        return ok
    except Exception as e:
        logger.warning("食谱检索预热失败（运行时回退子进程）：%s", e)
        return False
