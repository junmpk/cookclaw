"""
User-Agent 构建、配置辅助函数
"""

import os
import platform
from typing import Optional

from app.qqbot.constants import QQBOT_VERSION, PORTAL_HOST


def build_user_agent() -> str:
    """
    构建 QQ Bot API 请求的 User-Agent 字符串。

    Returns:
        str: User-Agent 字符串，格式: QQBot/{version} (hermes-agent; {os}/{arch})
    """
    os_name = platform.system() or "Unknown"
    os_arch = platform.machine() or "Unknown"
    return f"QQBot/{QQBOT_VERSION} (hermes-agent; {os_name}/{os_arch})"


def get_app_id(extra: Optional[dict] = None) -> Optional[str]:
    """
    获取 Bot App ID，优先从 extra 配置读取，其次从环境变量。

    Args:
        extra: 配置中的 extra 字典

    Returns:
        Optional[str]: App ID，未配置时返回 None
    """
    if extra and extra.get("app_id"):
        return extra["app_id"]
    return os.getenv("QQ_APP_ID")


def get_client_secret(extra: Optional[dict] = None) -> Optional[str]:
    """
    获取 Bot Client Secret，优先从 extra 配置读取，其次从环境变量。

    Args:
        extra: 配置中的 extra 字典

    Returns:
        Optional[str]: Client Secret，未配置时返回 None
    """
    if extra and extra.get("client_secret"):
        return extra["client_secret"]
    return os.getenv("QQ_CLIENT_SECRET")


def get_stt_api_key(extra: Optional[dict] = None) -> Optional[str]:
    """
    获取语音转文字 API Key，优先从 extra 配置读取，其次从环境变量。

    Args:
        extra: 配置中的 extra 字典

    Returns:
        Optional[str]: STT API Key，未配置时返回 None
    """
    if extra and extra.get("stt") and extra["stt"].get("apiKey"):
        return extra["stt"]["apiKey"]
    return os.getenv("QQ_STT_API_KEY")


def get_portal_host(extra: Optional[dict] = None) -> str:
    """
    获取 Portal 域名。

    Args:
        extra: 配置中的 extra 字典

    Returns:
        str: Portal 域名，默认 q.qq.com
    """
    if extra and extra.get("portal_host"):
        return extra["portal_host"]
    return os.getenv("QQ_PORTAL_HOST", PORTAL_HOST)


def strip_markdown(text: str) -> str:
    """
    简易 Markdown 清理，将 Markdown 格式转换为纯文本。

    Args:
        text: Markdown 文本

    Returns:
        str: 纯文本
    """
    import re

    # 移除粗体标记
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # 移除斜体标记
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # 移除删除线
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # 移除行内代码
    text = re.sub(r"`(.+?)`", r"\1", text)
    # 移除代码块
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip("`").strip(), text)
    # 移除链接，保留文字
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    # 移除标题标记
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    return text.strip()


def truncate_message(text: str, max_length: int = 4000) -> list[str]:
    """
    将长消息分块，每块不超过 max_length 字符。

    Args:
        text: 消息文本
        max_length: 每块最大长度

    Returns:
        list[str]: 消息块列表
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # 尝试在换行符处分割
        split_pos = text.rfind("\n", 0, max_length)
        if split_pos == -1:
            split_pos = max_length

        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")

    return chunks


def strip_at_mention(content: str) -> str:
    """
    去除群消息中的 @bot 前缀。

    Args:
        content: 原始消息内容

    Returns:
        str: 去除 @mention 后的内容
    """
    import re
    # 匹配 @用户 或 @!用户 格式
    return re.sub(r"^@\S+\s*", "", content).strip()
