"""
Chat Context Module - 管理用户对话上下文
解析目标用户 ID（多用户安全）
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Optional
from loguru import logger


CHAT_ID_CACHE_DIR = Path.home() / ".openclaw" / "workspace" / "state"


def _fingerprint(value: object) -> str:
    raw = str(value or "")
    return "-" if not raw else hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]


def ensure_chat_id(explicit_target: Optional[str] = None) -> str:
    """
    获取目标用户 ID（按优先级）

    优先级:
    1. explicit_target 显式传入
    2. OPENCLAW_CHAT_ID 环境变量
    3. OPENCLAW_TARGET 环境变量
    4. 缓存文件 ~/.openclaw/workspace/state/last_chat_id.txt

    Args:
        explicit_target: 显式传入的目标用户 ID

    Returns:
        目标用户 ID 字符串

    Raises:
        RuntimeError: 无法获取任何目标用户 ID
    """
    if explicit_target:
        logger.debug(f"Using explicit target: {_fingerprint(explicit_target)}")
        return explicit_target

    target = os.environ.get("OPENCLAW_CHAT_ID")
    if target:
        logger.debug(f"Using OPENCLAW_CHAT_ID: {_fingerprint(target)}")
        return target

    target = os.environ.get("OPENCLAW_TARGET")
    if target:
        logger.debug(f"Using OPENCLAW_TARGET: {_fingerprint(target)}")
        return target

    cache_file = CHAT_ID_CACHE_DIR / "last_chat_id.txt"
    if cache_file.exists():
        try:
            target = cache_file.read_text(encoding="utf-8").strip()
            if target:
                logger.debug(f"Using cached chat ID: {_fingerprint(target)}")
                return target
        except Exception as e:
            logger.debug(f"Failed to read chat ID cache: error_type={type(e).__name__}")

    raise RuntimeError(
        "No target user ID found. "
        "Set OPENCLAW_CHAT_ID or OPENCLAW_TARGET env var, "
        "or pass explicit target."
    )


def get_current_chat_id(explicit_target: Optional[str] = None) -> Optional[str]:
    """
    获取当前对话 ID（不抛异常，返回 None）

    Args:
        explicit_target: 显式传入的目标用户 ID

    Returns:
        目标用户 ID 字符串，或 None
    """
    try:
        return ensure_chat_id(explicit_target)
    except RuntimeError:
        return None


def write_chat_id_to_cache(chat_id: str) -> None:
    """
    将 chat ID 写入缓存文件

    Args:
        chat_id: 要缓存的用户 ID
    """
    try:
        CHAT_ID_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CHAT_ID_CACHE_DIR / "last_chat_id.txt"
        cache_file.write_text(chat_id, encoding="utf-8")
        logger.debug(f"Cached chat ID: {_fingerprint(chat_id)}")
    except Exception as e:
        logger.debug(f"Failed to write chat ID cache: error_type={type(e).__name__}")
