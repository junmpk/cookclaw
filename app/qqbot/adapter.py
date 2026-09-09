"""
QQ Bot 核心适配器 — WebSocket 连接 + REST API

实现 QQ Bot 完整生命周期:
  连接 → 认证 → 心跳 → 消息收发 → 断线重连
"""

import asyncio
import hashlib
import io
import logging
import os
import platform
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from app.qqbot.constants import (
    API_BASE,
    CONNECT_TIMEOUT_SECONDS,
    DEFAULT_API_TIMEOUT,
    DEFAULT_INTENTS,
    DEDUP_MAX_SIZE,
    DEDUP_WINDOW_SECONDS,
    FILE_UPLOAD_TIMEOUT,
    MAX_MESSAGE_LENGTH,
    MAX_QUICK_DISCONNECT_COUNT,
    MAX_RECONNECT_ATTEMPTS,
    MEDIA_TYPE_FILE,
    MEDIA_TYPE_IMAGE,
    MEDIA_TYPE_VIDEO,
    MEDIA_TYPE_VOICE,
    MSG_TYPE_MARKDOWN,
    MSG_TYPE_MEDIA,
    MSG_TYPE_TEXT,
    MSG_TYPE_INPUT_NOTIFY,
    QQBOT_VERSION,
    QUICK_DISCONNECT_THRESHOLD,
    RATE_LIMIT_DELAY,
    RECONNECT_BACKOFF,
    TOKEN_URL,
    WS_CLOSE_BOT_BANNED,
    WS_CLOSE_BOT_OFFLINE,
    WS_CLOSE_INVALID_TOKEN,
    WS_CLOSE_RATE_LIMITED,
    WS_CLOSE_SESSION_ERROR_END,
    WS_CLOSE_SESSION_ERROR_START,
    WS_CLOSE_SESSION_INVALID_SEQ,
    WS_CLOSE_SESSION_INVALID_START,
    WS_CLOSE_SESSION_TIMEOUT,
    WS_OP_DISPATCH,
    WS_OP_HEARTBEAT,
    WS_OP_HEARTBEAT_ACK,
    WS_OP_HELLO,
    WS_OP_IDENTIFY,
    WS_OP_RECONNECT,
    WS_OP_RESUME,
)
from app.qqbot.utils import (
    build_user_agent,
    get_app_id,
    get_client_secret,
    get_stt_api_key,
    strip_at_mention,
    strip_markdown,
    truncate_message,
)

logger = logging.getLogger(__name__)


def _plaintext_message_logs_enabled() -> bool:
    return os.getenv("LOG_PLAINTEXT_MESSAGES", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _log_fingerprint(value: object) -> str:
    raw = str(value or "")
    if _plaintext_message_logs_enabled():
        return raw or "-"
    return "-" if not raw else "sha256:" + hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _redact_endpoint(endpoint: str) -> str:
    if _plaintext_message_logs_enabled():
        return str(endpoint or "")
    return re.sub(
        r"/(?:users|groups)/([^/]+)",
        lambda match: match.group(0).replace(match.group(1), _log_fingerprint(match.group(1))),
        str(endpoint or ""),
    )


def _redact_url(value: object) -> str:
    """日志只保留协议与主机，丢弃可能携带签名/token 的路径和 query。"""
    try:
        parsed = urlparse(str(value or ""))
        host = parsed.hostname or "-"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme or '-'}://{host}{port}"
    except (TypeError, ValueError):
        return "invalid-url"

# ─── 打字指示器常量 ────────────────────────────────────────────────────
_TYPING_DEBOUNCE_SECONDS: float = 50.0
_TYPING_INPUT_SECONDS: int = 60
_RECONNECT_WAIT_SECONDS: float = 15.0


# ─── 数据模型 ──────────────────────────────────────────────────────────

class MessageType(str, Enum):
    """消息类型"""
    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    VIDEO = "video"
    FILE = "file"
    MIXED = "mixed"


@dataclass
class MessageSource:
    """消息来源信息"""
    chat_id: str       # 聊天 ID (openid / group_openid / channel_id)
    user_id: str       # 发送者 ID
    chat_type: str     # "dm" | "group" | "guild"


@dataclass
class MessageEvent:
    """入站消息事件"""
    source: MessageSource
    text: str
    message_type: MessageType = MessageType.TEXT
    message_id: str = ""
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    raw_data: Optional[dict] = None


def _incoming_message_type(content: str, attachment_result: dict) -> MessageType:
    """根据正文和附件确定入站类型，语音类型通过元数据传递而非污染用户文本。"""
    image_urls = attachment_result.get("image_urls") or []
    voice_transcripts = attachment_result.get("voice_transcripts") or []
    has_other_text = bool(
        str(content or "").strip()
        or str(attachment_result.get("attachment_info") or "").strip()
    )
    if image_urls:
        return MessageType.MIXED if has_other_text or voice_transcripts else MessageType.IMAGE
    if voice_transcripts:
        return MessageType.MIXED if has_other_text else MessageType.VOICE
    return MessageType.TEXT


@dataclass
class SendResult:
    """出站消息发送结果"""
    success: bool
    message_id: str = ""
    error: str = ""
    retryable: bool = True
    raw_response: Optional[dict] = None


# ─── 消息去重 ──────────────────────────────────────────────────────────

class DedupCache:
    """基于 OrderedDict 的 LRU 消息去重缓存"""

    def __init__(self, max_size: int = DEDUP_MAX_SIZE, window: int = DEDUP_WINDOW_SECONDS):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._max_size = max_size
        self._window = window

    def is_duplicate(self, msg_id: str) -> bool:
        """检查消息是否重复"""
        now = time.time()
        # 清理过期条目
        expired = [k for k, v in self._cache.items() if now - v > self._window]
        for k in expired:
            del self._cache[k]

        if msg_id in self._cache:
            # 更新访问顺序
            self._cache.move_to_end(msg_id)
            return True

        # 添加新条目
        self._cache[msg_id] = now
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return False


# ─── SSRF 防护 ─────────────────────────────────────────────────────────

_PRIVATE_NETWORKS = [
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "0.0.0.0/8",
    "169.254.0.0/16",
]


def is_safe_url(url: str) -> bool:
    """
    检查 URL 是否安全（非内网地址）。

    Args:
        url: 待检查的 URL

    Returns:
        bool: URL 是否安全
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        # 简易检查：拒绝常见内网域名和 IP
        if hostname in ("localhost", "localhost.localdomain"):
            return False
        if hostname.endswith(".local") or hostname.endswith(".internal"):
            return False
        # 检查 IP 地址
        import ipaddress
        try:
            ip = ipaddress.ip_address(hostname)
            for network in _PRIVATE_NETWORKS:
                if ip in ipaddress.ip_network(network):
                    return False
        except ValueError:
            pass  # 不是 IP 地址，是域名，继续
        return True
    except Exception:
        return False


# ─── QQAdapter 核心适配器 ──────────────────────────────────────────────

class QQAdapter:
    """
    QQ Bot 核心适配器

    负责 WebSocket 连接管理、消息收发、媒体处理、断线重连等。
    """

    def __init__(self, extra: Optional[dict] = None):
        """
        初始化 QQ Bot 适配器。

        Args:
            extra: 配置中的 extra 字典，包含 app_id、client_secret 等
        """
        extra = extra or {}

        # ─── 凭证 ──────────────────────────────────────────────
        self._app_id: str = get_app_id(extra) or ""
        self._client_secret: str = get_client_secret(extra) or ""

        # ─── 消息格式 ──────────────────────────────────────────
        self._markdown_support: bool = extra.get("markdown_support", True)

        # ─── ACL 策略 ──────────────────────────────────────────
        self._dm_policy: str = extra.get("dm_policy", "open")
        self._allow_from: list = extra.get("allow_from", [])
        self._group_policy: str = extra.get("group_policy", "open")
        self._group_allow_from: list = extra.get("group_allow_from", [])

        # ─── STT 配置 ──────────────────────────────────────────
        self._stt_config: dict = extra.get("stt", {})
        self._stt_api_key: str = get_stt_api_key(extra) or ""

        # ─── HTTP / WebSocket ──────────────────────────────────
        self._http_client: Optional[httpx.AsyncClient] = None
        self._ws = None  # aiohttp WebSocket
        self._session = None  # aiohttp session

        # ─── Token 管理 ────────────────────────────────────────
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

        # ─── WebSocket 状态 ────────────────────────────────────
        self._heartbeat_interval: float = 30.0
        self._session_id: Optional[str] = None
        self._last_seq: Optional[int] = None
        self._connected: bool = False
        self._ws_url: Optional[str] = None

        # ─── 任务管理 ──────────────────────────────────────────
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # ─── 打字指示器 ────────────────────────────────────────
        self._last_msg_id: Dict[str, str] = {}
        self._typing_sent_at: Dict[str, float] = {}

        # ─── 媒体上传缓存 ──────────────────────────────────────
        self._upload_cache: Dict[str, dict] = {}

        # ─── 聊天类型缓存 ──────────────────────────────────────
        self._chat_type_map: Dict[str, str] = {}

        # ─── 消息去重 ──────────────────────────────────────────
        self._dedup = DedupCache()

        # ─── 消息回调 ──────────────────────────────────────────
        self._message_handler: Optional[Callable] = None

        # ─── 重连状态 ──────────────────────────────────────────
        self._quick_disconnect_count: int = 0
        self._last_connect_time: float = 0.0
        self._reconnect_event = asyncio.Event()
        self._shutting_down: bool = False

        # ─── 消息序列号 ────────────────────────────────────────
        self._msg_seq: int = 0

        # ─── 锁 ────────────────────────────────────────────────
        self._platform_lock_acquired: bool = False

        # ─── User-Agent ────────────────────────────────────────
        self._user_agent: str = build_user_agent()

    def set_message_handler(self, handler: Callable):
        """
        设置消息回调处理器。

        Args:
            handler: 异步回调函数，签名为 async handler(event: MessageEvent)
        """
        self._message_handler = handler

    # ─── 连接管理 ──────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """
        建立 QQ Bot 连接。

        Returns:
            bool: 连接是否成功
        """
        try:
            # 1. 前置检查
            if not self._app_id or not self._client_secret:
                logger.error("缺少必要配置: app_id 或 client_secret")
                return False

            # 2. 初始化 HTTP 客户端
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(
                    timeout=DEFAULT_API_TIMEOUT,
                    headers={"User-Agent": self._user_agent},
                    trust_env=False,  # 忽略系统代理(ALL_PROXY/HTTP(S)_PROXY)，直连境内 QQ 服务器；与 embedding.py 一致
                )

            # 3. 获取 access_token
            token = await self._ensure_token()
            if not token:
                logger.error("获取 access_token 失败")
                return False

            # 4. 获取 WebSocket Gateway URL
            gateway_url = await self._get_gateway_url()
            if not gateway_url:
                logger.error("获取 Gateway URL 失败")
                return False

            self._ws_url = gateway_url

            # 5. 建立 WebSocket 连接
            connected = await self._open_ws(gateway_url)
            if not connected:
                logger.error("WebSocket 连接失败")
                return False

            self._connected = True
            self._last_connect_time = time.time()
            logger.info(f"QQ Bot 连接成功 (app_id={self._app_id})")

            return True

        except Exception as e:
            logger.error("连接异常: error_type=%s", type(e).__name__)
            return False

    async def disconnect(self):
        """断开连接并清理资源"""
        self._shutting_down = True
        self._connected = False

        # 取消任务
        for task in (self._listen_task, self._heartbeat_task):
            if task and not task.done():
                task.cancel()

        # 关闭 WebSocket
        if self._ws and not self._ws.closed:
            await self._ws.close()

        # 关闭 HTTP 客户端
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

        # 关闭 aiohttp session
        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("QQ Bot 已断开连接")

    async def _ensure_token(self) -> Optional[str]:
        """
        获取 access_token，带缓存和单飞模式。

        Returns:
            Optional[str]: access_token
        """
        async with self._token_lock:
            now = time.time()
            if self._access_token and now < self._token_expires_at - 60:
                return self._access_token

            try:
                resp = await self._http_client.post(
                    TOKEN_URL,
                    json={
                        "appId": self._app_id,
                        "clientSecret": self._client_secret,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                self._access_token = data.get("access_token")
                expires_in = int(data.get("expires_in", 7200))
                self._token_expires_at = now + expires_in

                logger.debug(f"Token 已刷新, 过期时间: {expires_in}s")
                return self._access_token

            except Exception as e:
                logger.error("获取 Token 失败: error_type=%s", type(e).__name__)
                return None

    async def _get_gateway_url(self) -> Optional[str]:
        """
        获取 WebSocket Gateway URL。

        Returns:
            Optional[str]: Gateway URL
        """
        try:
            resp = await self._http_client.get(
                f"{API_BASE}/gateway",
                headers={"Authorization": f"QQBot {self._access_token}"},
            )
            resp.raise_for_status()
            data = resp.json()
            url = data.get("url")
            logger.debug("Gateway URL received: %s", _redact_url(url))
            return url
        except Exception as e:
            logger.error("获取 Gateway URL 失败: error_type=%s", type(e).__name__)
            return None

    async def _open_ws(self, url: str) -> bool:
        """
        建立 WebSocket 连接。

        Args:
            url: WebSocket Gateway URL

        Returns:
            bool: 是否成功建立
        """
        try:
            import aiohttp
        except ImportError:
            logger.error("缺少 aiohttp 依赖，请安装: pip install aiohttp")
            return False

        try:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            self._ws = await self._session.ws_connect(
                url,
                timeout=aiohttp.WSMsgType.CLOSE,
                heartbeat=None,
            )

            # 等待 Hello 消息
            msg = await asyncio.wait_for(
                self._ws.receive(),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )

            if msg.type != aiohttp.WSMsgType.TEXT:
                logger.error(f"WebSocket 握手失败: 期望 TEXT, 收到 {msg.type}")
                return False

            import json
            data = json.loads(msg.data)

            if data.get("op") != WS_OP_HELLO:
                logger.error(f"WebSocket 握手失败: 期望 op={WS_OP_HELLO}, 收到 op={data.get('op')}")
                return False

            # 保存心跳间隔
            heartbeat_interval = data.get("d", {}).get("heartbeat_interval", 41250)
            self._heartbeat_interval = heartbeat_interval / 1000.0  # ms → s

            # 发送 Identify 或 Resume
            if self._session_id and self._last_seq is not None:
                await self._send_resume()
            else:
                await self._send_identify()

            # 启动监听和心跳任务
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            return True

        except asyncio.TimeoutError:
            logger.error("WebSocket 连接超时")
            return False
        except Exception as e:
            logger.error("WebSocket 连接异常: error_type=%s", type(e).__name__)
            return False

    # ─── WebSocket 认证 ────────────────────────────────────────────────

    async def _send_identify(self):
        """发送 Identify (op=2) 进行首次认证"""
        import json

        payload = {
            "op": WS_OP_IDENTIFY,
            "d": {
                "token": f"QQBot {self._access_token}",
                "intents": DEFAULT_INTENTS,
                "shard": [0, 1],
                "properties": {
                    "$os": platform.system(),
                    "$browser": "hermes-agent",
                    "$device": "hermes-agent",
                },
            },
        }

        await self._ws.send_str(json.dumps(payload))
        logger.debug("已发送 Identify (op=2)")

    async def _send_resume(self):
        """发送 Resume (op=6) 进行断线恢复"""
        import json

        payload = {
            "op": WS_OP_RESUME,
            "d": {
                "token": f"QQBot {self._access_token}",
                "session_id": self._session_id,
                "seq": self._last_seq,
            },
        }

        await self._ws.send_str(json.dumps(payload))
        logger.debug(f"已发送 Resume (op=6), session_id={self._session_id}, seq={self._last_seq}")

    # ─── 心跳保活 ──────────────────────────────────────────────────────

    async def _heartbeat_loop(self):
        """心跳循环"""
        import json

        try:
            while self._connected and not self._shutting_down:
                # 实际发送间隔 = 服务器间隔 × 0.8
                interval = self._heartbeat_interval * 0.8
                await asyncio.sleep(interval)

                if not self._connected or self._ws is None or self._ws.closed:
                    break

                heartbeat_payload = {
                    "op": WS_OP_HEARTBEAT,
                    "d": self._last_seq,
                }
                await self._ws.send_str(json.dumps(heartbeat_payload))
                logger.debug(f"心跳已发送: op={WS_OP_HEARTBEAT}, seq={self._last_seq}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("心跳异常: error_type=%s", type(e).__name__)
            if self._connected:
                await self._handle_disconnect()

    # ─── 消息监听 ──────────────────────────────────────────────────────

    async def _listen_loop(self):
        """WebSocket 消息监听循环"""
        import aiohttp
        import json

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._on_ws_message(data)
                    except json.JSONDecodeError:
                        logger.warning(f"无效的 JSON 消息: {msg.data[:200]}")
                    except Exception as e:
                        logger.error("处理消息异常: error_type=%s", type(e).__name__)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket 错误: {self._ws.exception()}")
                    break

                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    logger.info(f"WebSocket 关闭: code={msg.data}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("监听循环异常: error_type=%s", type(e).__name__)
        finally:
            if self._connected:
                await self._handle_disconnect()

    async def _on_ws_message(self, data: dict):
        """
        处理 WebSocket 消息。

        Args:
            data: 解析后的 JSON 数据
        """
        op = data.get("op")
        d = data.get("d")
        s = data.get("s")
        t = data.get("t")

        # 更新序列号
        if s is not None:
            self._last_seq = s

        if op == WS_OP_DISPATCH:
            # 服务端推送消息
            if t == "READY":
                self._session_id = d.get("session_id") if d else None
                logger.info(f"READY: session_id={self._session_id}")
            elif t == "RESUMED":
                logger.info("RESUMED: 断线恢复成功")
            else:
                await self._on_message(t, d)

        elif op == WS_OP_HEARTBEAT_ACK:
            logger.debug("心跳 ACK 收到")

        elif op == WS_OP_RECONNECT:
            logger.warning("服务端要求重连 (op=7)")
            await self._handle_disconnect()

        elif op == WS_OP_HELLO:
            # 重连时可能再次收到 Hello
            if d and isinstance(d, dict):
                heartbeat_interval = d.get("heartbeat_interval", 41250)
                self._heartbeat_interval = heartbeat_interval / 1000.0

        else:
            logger.debug(f"未处理的 OP: {op}, data={data}")

    # ─── 入站消息处理 ──────────────────────────────────────────────────

    async def _on_message(self, event_type: str, d: Any):
        """
        消息路由分发。

        Args:
            event_type: 事件类型
            d: 事件数据
        """
        if not d or not isinstance(d, dict):
            return

        msg_id = d.get("id", "")
        if not msg_id:
            return

        # 消息去重
        if self._dedup.is_duplicate(msg_id):
            logger.debug(f"重复消息已忽略: {msg_id}")
            return

        # 路由到对应处理器
        handler_map = {
            "C2C_MESSAGE_CREATE": self._handle_c2c_message,
            "GROUP_AT_MESSAGE_CREATE": self._handle_group_message,
            "GUILD_MESSAGE_CREATE": self._handle_guild_message,
            "GUILD_AT_MESSAGE_CREATE": self._handle_guild_message,
            "DIRECT_MESSAGE_CREATE": self._handle_dm_message,
        }

        handler = handler_map.get(event_type)
        if handler:
            try:
                await handler(d)
            except Exception as e:
                logger.error("处理 %s 异常: error_type=%s", event_type, type(e).__name__)
        else:
            logger.debug(f"未处理的事件类型: {event_type}")

    async def _handle_c2c_message(self, d: dict):
        """
        处理 C2C 私聊消息。

        Args:
            d: 事件数据
        """
        msg_id = d.get("id", "")
        content = d.get("content", "")
        author = d.get("author", {})
        timestamp = d.get("timestamp", "")

        user_openid = author.get("user_openid", "")

        # ACL 检查

        if not self._is_dm_allowed(user_openid):
            logger.debug("C2C 消息被 ACL 拒绝: user=%s", _log_fingerprint(user_openid))
            return

        # 处理附件
        attachments = d.get("attachments", [])
        attachment_result = await self._process_attachments(attachments)

        # 构建完整文本
        full_text = content
        if attachment_result.get("voice_transcripts"):
            full_text += "\n" + "\n".join(attachment_result["voice_transcripts"])
        if attachment_result.get("attachment_info"):
            full_text += "\n" + attachment_result["attachment_info"]

        # 确定消息类型
        image_urls = attachment_result.get("image_urls", [])
        image_types = attachment_result.get("image_media_types", [])
        msg_type = _incoming_message_type(content, attachment_result)

        # 保存最后消息 ID (用于 typing)
        self._last_msg_id[user_openid] = msg_id

        # 构建事件
        event = MessageEvent(
            source=MessageSource(
                chat_id=user_openid,
                user_id=user_openid,
                chat_type="dm",
            ),
            text=full_text.strip(),
            message_type=msg_type,
            message_id=msg_id,
            media_urls=image_urls,
            media_types=image_types,
            timestamp=self._parse_timestamp(timestamp),
            raw_data=d,
        )

        # 缓存聊天类型
        self._chat_type_map[user_openid] = "c2c"

        await self._dispatch_message(event)

    async def _handle_group_message(self, d: dict):
        """
        处理群@消息。

        Args:
            d: 事件数据
        """
        msg_id = d.get("id", "")
        content = d.get("content", "")
        author = d.get("author", {})
        timestamp = d.get("timestamp", "")

        group_openid = d.get("group_openid", "")
        member_openid = author.get("member_openid", "")

        # ACL 检查
        if not self._is_group_allowed(group_openid, member_openid):
            logger.debug(
                "群消息被 ACL 拒绝: group=%s member=%s",
                _log_fingerprint(group_openid), _log_fingerprint(member_openid),
            )
            return

        # 去除 @bot 前缀
        content = strip_at_mention(content)

        # 处理附件
        attachments = d.get("attachments", [])
        attachment_result = await self._process_attachments(attachments)

        # 构建完整文本
        full_text = content
        if attachment_result.get("voice_transcripts"):
            full_text += "\n" + "\n".join(attachment_result["voice_transcripts"])
        if attachment_result.get("attachment_info"):
            full_text += "\n" + attachment_result["attachment_info"]

        # 确定消息类型
        image_urls = attachment_result.get("image_urls", [])
        image_types = attachment_result.get("image_media_types", [])
        msg_type = _incoming_message_type(content, attachment_result)

        # 保存最后消息 ID
        self._last_msg_id[group_openid] = msg_id

        event = MessageEvent(
            source=MessageSource(
                chat_id=group_openid,
                user_id=member_openid,
                chat_type="group",
            ),
            text=full_text.strip(),
            message_type=msg_type,
            message_id=msg_id,
            media_urls=image_urls,
            media_types=image_types,
            timestamp=self._parse_timestamp(timestamp),
            raw_data=d,
        )

        # 缓存聊天类型
        self._chat_type_map[group_openid] = "group"

        await self._dispatch_message(event)

    async def _handle_guild_message(self, d: dict):
        """
        处理频道消息。

        Args:
            d: 事件数据
        """
        msg_id = d.get("id", "")
        content = d.get("content", "")
        author = d.get("author", {})
        timestamp = d.get("timestamp", "")

        channel_id = d.get("channel_id", "")
        guild_id = d.get("guild_id", "")
        user_id = author.get("user_openid", "") or str(author.get("id", ""))

        # 去除 @bot 前缀
        content = strip_at_mention(content)

        # 处理附件
        attachments = d.get("attachments", [])
        attachment_result = await self._process_attachments(attachments)

        full_text = content
        if attachment_result.get("voice_transcripts"):
            full_text += "\n" + "\n".join(attachment_result["voice_transcripts"])
        if attachment_result.get("attachment_info"):
            full_text += "\n" + attachment_result["attachment_info"]

        image_urls = attachment_result.get("image_urls", [])
        image_types = attachment_result.get("image_media_types", [])

        self._last_msg_id[channel_id] = msg_id

        event = MessageEvent(
            source=MessageSource(
                chat_id=channel_id,
                user_id=user_id,
                chat_type="guild",
            ),
            text=full_text.strip(),
            message_type=_incoming_message_type(content, attachment_result),
            message_id=msg_id,
            media_urls=image_urls,
            media_types=image_types,
            timestamp=self._parse_timestamp(timestamp),
            raw_data=d,
        )

        self._chat_type_map[channel_id] = "guild"

        await self._dispatch_message(event)

    async def _handle_dm_message(self, d: dict):
        """
        处理私信消息 (频道私信)。

        Args:
            d: 事件数据
        """
        # 频道私信与 C2C 处理类似
        msg_id = d.get("id", "")
        content = d.get("content", "")
        author = d.get("author", {})
        timestamp = d.get("timestamp", "")

        user_openid = author.get("user_openid", "") or str(author.get("id", ""))

        if not self._is_dm_allowed(user_openid):
            return

        attachments = d.get("attachments", [])
        attachment_result = await self._process_attachments(attachments)

        full_text = content
        if attachment_result.get("voice_transcripts"):
            full_text += "\n" + "\n".join(attachment_result["voice_transcripts"])
        if attachment_result.get("attachment_info"):
            full_text += "\n" + attachment_result["attachment_info"]

        image_urls = attachment_result.get("image_urls", [])
        image_types = attachment_result.get("image_media_types", [])

        self._last_msg_id[user_openid] = msg_id

        event = MessageEvent(
            source=MessageSource(
                chat_id=user_openid,
                user_id=user_openid,
                chat_type="dm",
            ),
            text=full_text.strip(),
            message_type=_incoming_message_type(content, attachment_result),
            message_id=msg_id,
            media_urls=image_urls,
            media_types=image_types,
            timestamp=self._parse_timestamp(timestamp),
            raw_data=d,
        )

        await self._dispatch_message(event)

    async def _dispatch_message(self, event: MessageEvent):
        """
        分发消息到回调处理器。

        Args:
            event: 消息事件
        """
        if self._message_handler:
            try:
                await self._message_handler(event)
            except Exception as e:
                logger.error("消息回调异常: error_type=%s", type(e).__name__)
        else:
            logger.debug("无消息处理器，消息已忽略: message=%s", _log_fingerprint(event.message_id))

    # ─── 附件处理 ──────────────────────────────────────────────────────

    async def _process_attachments(self, attachments: list) -> dict:
        """
        处理消息附件列表。

        Args:
            attachments: QQ 消息附件列表

        Returns:
            dict: {
                "image_urls": [...],
                "image_media_types": [...],
                "voice_transcripts": [...],  # 纯转写文本，不含展示前缀
                "attachment_info": "..."
            }
        """
        result = {
            "image_urls": [],
            "image_media_types": [],
            "voice_transcripts": [],
            "attachment_info": "",
        }

        if not attachments:
            return result

        for att in attachments:
            content_type = att.get("content_type", "")
            url = att.get("url", "")
            filename = att.get("filename", "")

            if content_type.startswith("image/"):
                # 图片: 下载并缓存
                cached_path = await self._download_and_cache(url, filename)
                if cached_path:
                    result["image_urls"].append(cached_path)
                    result["image_media_types"].append(content_type)

            elif content_type == "voice" or content_type.startswith("audio/") or self._is_audio_file(filename):
                # 语音: 优先 QQ ASR → WAV URL STT → 原始 URL STT
                transcript = await self._transcribe_voice(att)
                if transcript:
                    result["voice_transcripts"].append(transcript.strip())

            else:
                # 其他附件
                if filename:
                    result["attachment_info"] += f"[Attachment: {filename}] "

        return result

    def _is_audio_file(self, filename: str) -> bool:
        """检查文件名是否为音频文件"""
        audio_exts = {".silk", ".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".amr"}
        return Path(filename).suffix.lower() in audio_exts if filename else False

    async def _transcribe_voice(self, attachment: dict) -> str:
        """
        语音转文字。

        优先级:
        1. QQ 内置 asr_refer_text
        2. voice_wav_url 上的 STT
        3. 原始附件 URL 上的 STT

        Args:
            attachment: 附件数据

        Returns:
            str: 转写文本 (空字符串表示失败)
        """
        # 1. QQ 内置 ASR
        asr_text = attachment.get("asr_refer_text", "")
        if asr_text:
            return asr_text

        # 2. WAV URL STT
        wav_url = attachment.get("voice_wav_url", "")
        if wav_url:
            text = await self._do_stt(wav_url)
            if text:
                return text

        # 3. 原始 URL STT
        original_url = attachment.get("url", "")
        if original_url:
            text = await self._do_stt(original_url)
            if text:
                return text

        return ""

    async def _do_stt(self, audio_url: str) -> str:
        """
        调用 STT API 进行语音转文字。

        Args:
            audio_url: 音频文件 URL

        Returns:
            str: 转写文本
        """
        if not self._stt_api_key:
            return ""

        provider = self._stt_config.get("provider", "zai")
        base_url = self._stt_config.get("baseUrl", "")
        model = self._stt_config.get("model", "glm-asr")

        try:
            if provider == "zai":
                # GLM-ASR (智谱)
                resp = await self._http_client.post(
                    f"{base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self._stt_api_key}"},
                    data={"model": model, "url": audio_url},
                )
                resp.raise_for_status()
                return resp.json().get("text", "")

            elif provider == "openai":
                # Whisper (OpenAI 兼容)
                resp = await self._http_client.post(
                    f"{base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self._stt_api_key}"},
                    data={"model": model, "url": audio_url},
                )
                resp.raise_for_status()
                return resp.json().get("text", "")

        except Exception as e:
            logger.error("STT 转写失败 (%s): error_type=%s", provider, type(e).__name__)

        return ""

    async def _download_and_cache(self, url: str, filename: str = "") -> Optional[str]:
        """
        下载文件并缓存到本地。

        Args:
            url: 文件 URL
            filename: 文件名

        Returns:
            Optional[str]: 本地缓存路径
        """
        if not url or not is_safe_url(url):
            return None

        try:
            resp = await self._http_client.get(url, timeout=FILE_UPLOAD_TIMEOUT)
            resp.raise_for_status()

            # 生成缓存路径
            cache_dir = Path("cache/qqbot/media")
            cache_dir.mkdir(parents=True, exist_ok=True)

            ext = Path(filename).suffix if filename else ".bin"
            cache_path = cache_dir / f"{uuid.uuid4().hex}{ext}"

            cache_path.write_bytes(resp.content)
            return str(cache_path)

        except Exception as e:
            logger.error(
                "下载缓存失败: source=%s error_type=%s",
                _redact_url(url), type(e).__name__,
            )
            return None

    # ─── 出站消息发送 ──────────────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SendResult:
        """
        发送文本消息。

        Args:
            chat_id: 目标聊天 ID
            content: 消息内容
            reply_to: 回复的消息 ID
            metadata: 保留参数

        Returns:
            SendResult: 发送结果
        """
        # 连接检查
        if not self._connected:
            waited = await self._wait_for_reconnection()
            if not waited:
                return SendResult(success=False, error="未连接", retryable=True)

        # 格式化
        if self._markdown_support:
            formatted = content
        else:
            formatted = strip_markdown(content)

        # 分块
        chunks = truncate_message(formatted, MAX_MESSAGE_LENGTH)

        results = []
        for chunk in chunks:
            result = await self._send_chunk(chat_id, chunk, reply_to)
            results.append(result)
            if not result.success:
                return result

        # 返回最后一个结果
        return results[-1] if results else SendResult(success=False, error="无内容")

    async def _send_chunk(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """
        发送单个消息块，带重试。

        Args:
            chat_id: 目标聊天 ID
            content: 消息内容
            reply_to: 回复的消息 ID

        Returns:
            SendResult: 发送结果
        """
        chat_type = self._guess_chat_type(chat_id)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if chat_type == "c2c":
                    result = await self._send_c2c_text(chat_id, content, reply_to)
                elif chat_type == "group":
                    result = await self._send_group_text(chat_id, content, reply_to)
                elif chat_type == "guild":
                    result = await self._send_guild_text(chat_id, content, reply_to)
                else:
                    return SendResult(success=False, error=f"未知聊天类型: {chat_type}")

                if result.success:
                    return result

                # 永久错误检查
                error_lower = result.error.lower()
                if any(kw in error_lower for kw in ("invalid", "forbidden", "not found", "unauthorized")):
                    result.retryable = False
                    return result

            except Exception as e:
                logger.warning(
                    "发送失败 (attempt=%s/%s): error_type=%s",
                    attempt + 1, max_retries, type(e).__name__,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 退避: 1s, 2s

        return SendResult(success=False, error=f"发送失败: 已重试 {max_retries} 次", retryable=True)

    async def _send_c2c_text(
        self,
        openid: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """发送 C2C 私聊文本消息"""
        self._msg_seq += 1

        if self._markdown_support:
            body: dict = {
                "markdown": {"content": content},
                "msg_type": MSG_TYPE_MARKDOWN,
                "msg_seq": self._msg_seq,
            }
        else:
            body = {
                "content": content,
                "msg_type": MSG_TYPE_TEXT,
                "msg_seq": self._msg_seq,
            }
            if reply_to:
                body["message_reference"] = {"message_id": reply_to}

        if reply_to:
            body["msg_id"] = reply_to

        return await self._api_post(
            f"/v2/users/{openid}/messages",
            body,
        )

    async def _send_group_text(
        self,
        group_openid: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """发送群文本消息"""
        self._msg_seq += 1

        if self._markdown_support:
            body: dict = {
                "markdown": {"content": content},
                "msg_type": MSG_TYPE_MARKDOWN,
                "msg_seq": self._msg_seq,
            }
        else:
            body = {
                "content": content,
                "msg_type": MSG_TYPE_TEXT,
                "msg_seq": self._msg_seq,
            }
            if reply_to:
                body["message_reference"] = {"message_id": reply_to}

        if reply_to:
            body["msg_id"] = reply_to

        return await self._api_post(
            f"/v2/groups/{group_openid}/messages",
            body,
        )

    async def _send_guild_text(
        self,
        channel_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """发送频道文本消息"""
        body: dict = {"content": content}
        if reply_to:
            body["msg_id"] = reply_to

        return await self._api_post(
            f"/channels/{channel_id}/messages",
            body,
        )

    # ─── 媒体发送 ──────────────────────────────────────────────────────

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: str = "",
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """发送图片"""
        return await self._send_media(chat_id, image_url, MEDIA_TYPE_IMAGE, "image", caption, reply_to)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: str = "",
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """发送语音"""
        return await self._send_media(chat_id, audio_path, MEDIA_TYPE_VOICE, "voice", caption, reply_to)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: str = "",
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """发送视频"""
        return await self._send_media(chat_id, video_path, MEDIA_TYPE_VIDEO, "video", caption, reply_to)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: str = "",
        file_name: str = "",
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """发送文件"""
        return await self._send_media(
            chat_id, file_path, MEDIA_TYPE_FILE, "file", caption, reply_to, file_name
        )

    async def _send_media(
        self,
        chat_id: str,
        media_source: str,
        file_type: int,
        kind: str,
        caption: str = "",
        reply_to: Optional[str] = None,
        file_name: str = "",
    ) -> SendResult:
        """
        通用媒体发送流程:
        1. 加载媒体
        2. 上传获取 file_info
        3. 发送媒体消息

        Args:
            chat_id: 目标聊天 ID
            media_source: 媒体来源 (URL 或本地路径)
            file_type: 媒体类型常量
            kind: 媒体种类 ("image"/"voice"/"video"/"file")
            caption: 说明文字
            reply_to: 回复的消息 ID
            file_name: 自定义文件名
        """
        # 连接检查
        if not self._connected:
            waited = await self._wait_for_reconnection()
            if not waited:
                return SendResult(success=False, error="未连接", retryable=True)

        chat_type = self._guess_chat_type(chat_id)

        # Guild 不支持原生媒体上传
        if chat_type == "guild":
            return SendResult(
                success=False,
                error="Guild media send not supported via this path",
                retryable=False,
            )

        # 加载媒体
        media_data = await self._load_media(media_source, file_name)
        if (
            kind == "image"
            and media_source.startswith(("http://", "https://"))
            and "file_data" not in media_data
        ):
            # 菜谱图片必须由 CookClaw 下载后再上传。若下载失败，交给上层降级为
            # 无图文本，不能再次让 QQ 服务端抓外链，否则会重新引入破图问题。
            return SendResult(
                success=False,
                error="图片下载失败，已禁止外链回退",
                retryable=True,
            )

        # 上传媒体
        file_info = await self._upload_media(
            chat_type, chat_id, file_type, media_data, file_name
        )
        if not file_info:
            return SendResult(success=False, error="媒体上传失败", retryable=True)

        # 发送媒体消息
        self._msg_seq += 1
        body: dict = {
            "msg_type": MSG_TYPE_MEDIA,
            "media": {"file_info": file_info},
            "msg_seq": self._msg_seq,
        }
        if caption:
            body["content"] = caption
        if reply_to:
            body["msg_id"] = reply_to

        endpoint = (
            f"/v2/users/{chat_id}/messages"
            if chat_type == "c2c"
            else f"/v2/groups/{chat_id}/messages"
        )

        return await self._api_post(endpoint, body)

    async def _load_media(self, source: str, file_name: str = "") -> dict:
        """
        加载媒体数据。

        URL 会先下载到本地再 base64 编码（避免 QQ 服务器无法访问外部 URL）。
        本地文件直接 base64 编码。

        Args:
            source: URL 或本地路径
            file_name: 自定义文件名

        Returns:
            dict: {"file_data": "base64..."} 或 {"url": ...}（下载失败时回退）
        """
        import base64

        if source.startswith(("http://", "https://")):
            # URL → 先下载 → base64 编码
            try:
                resp = await self._http_client.get(source, timeout=FILE_UPLOAD_TIMEOUT)
                resp.raise_for_status()
                encoded = base64.b64encode(resp.content).decode("utf-8")
                logger.debug(
                    "已下载并编码媒体: source=%s bytes=%s",
                    _redact_url(source), len(resp.content),
                )
                return {"file_data": encoded}
            except Exception as e:
                logger.warning(
                    "下载媒体失败，回退为 URL 模式: source=%s error_type=%s",
                    _redact_url(source), type(e).__name__,
                )
                return {"url": source}

        # 本地文件 → base64
        try:
            import base64
            path = Path(source)
            if path.exists():
                data = path.read_bytes()
                encoded = base64.b64encode(data).decode("utf-8")
                return {"file_data": encoded}
        except Exception as e:
            logger.error("加载本地媒体失败: error_type=%s", type(e).__name__)

        return {"url": source}

    async def _upload_media(
        self,
        chat_type: str,
        chat_id: str,
        file_type: int,
        media_data: dict,
        file_name: str = "",
    ) -> Optional[str]:
        """
        上传媒体文件，获取 file_info。

        Args:
            chat_type: 聊天类型 ("c2c" / "group")
            chat_id: 聊天 ID
            file_type: 媒体类型常量
            media_data: 媒体数据
            file_name: 自定义文件名

        Returns:
            Optional[str]: file_info 字符串
        """
        endpoint = (
            f"/v2/users/{chat_id}/files"
            if chat_type == "c2c"
            else f"/v2/groups/{chat_id}/files"
        )

        body: dict = {
            "file_type": file_type,
            "srv_send_msg": False,
        }

        if "url" in media_data:
            body["url"] = media_data["url"]
        elif "file_data" in media_data:
            body["file_data"] = media_data["file_data"]

        if file_name and file_type == MEDIA_TYPE_FILE:
            body["file_name"] = file_name

        # 带重试的上传
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await self._api_post(endpoint, body, timeout=FILE_UPLOAD_TIMEOUT)
                if result.success and result.raw_response:
                    file_info = result.raw_response.get("file_info") or result.raw_response.get("file_uuid")
                    if file_info:
                        return file_info
                    # 尝试从 raw_response 中获取
                    file_info = result.raw_response.get("file_info")
                    if file_info:
                        return file_info

                raw = result.raw_response
                keys = sorted(raw.keys()) if isinstance(raw, dict) else []
                logger.warning(
                    "上传响应无 file_info: response_type=%s keys=%s",
                    type(raw).__name__, keys,
                )

            except Exception as e:
                logger.warning(
                    "媒体上传失败 (attempt=%s/%s): error_type=%s",
                    attempt + 1, max_retries, type(e).__name__,
                )

            if attempt < max_retries - 1:
                await asyncio.sleep(1.5 * (attempt + 1))

        return None

    # ─── 打字指示器 ────────────────────────────────────────────────────

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        """
        发送打字指示器 (仅 C2C 有效)。

        Args:
            chat_id: 目标聊天 ID
            metadata: 未使用
        """
        chat_type = self._guess_chat_type(chat_id)
        if chat_type != "c2c":
            return

        # 防抖
        now = time.time()
        last_sent = self._typing_sent_at.get(chat_id, 0)
        if now - last_sent < _TYPING_DEBOUNCE_SECONDS:
            return

        msg_id = self._last_msg_id.get(chat_id)
        if not msg_id:
            return

        self._msg_seq += 1
        body = {
            "msg_type": MSG_TYPE_INPUT_NOTIFY,
            "msg_id": msg_id,
            "input_notify": {
                "input_type": 1,
                "input_second": _TYPING_INPUT_SECONDS,
            },
            "msg_seq": self._msg_seq,
        }

        try:
            await self._api_post(f"/v2/users/{chat_id}/messages", body)
            self._typing_sent_at[chat_id] = now
        except Exception as e:
            logger.debug("发送打字指示器失败: error_type=%s", type(e).__name__)

    # ─── ACL 检查 ──────────────────────────────────────────────────────

    def _is_dm_allowed(self, user_openid: str) -> bool:
        """
        检查私聊是否被允许。

        Args:
            user_openid: 用户 openid

        Returns:
            bool: 是否允许
        """
        if self._dm_policy == "open":
            return True
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return "*" in self._allow_from or user_openid in self._allow_from
        return False

    def _is_group_allowed(self, group_openid: str, member_openid: str = "") -> bool:
        """
        检查群聊是否被允许。

        Args:
            group_openid: 群 openid
            member_openid: 群成员 openid

        Returns:
            bool: 是否允许
        """
        if self._group_policy == "open":
            return True
        if self._group_policy == "disabled":
            return False
        if self._group_policy == "allowlist":
            return (
                "*" in self._group_allow_from
                or group_openid in self._group_allow_from
            )
        return False

    # ─── 断线重连 ──────────────────────────────────────────────────────

    async def _handle_disconnect(self):
        """处理 WebSocket 断开"""
        self._connected = False

        if self._shutting_down:
            return

        # 检查快速断连
        now = time.time()
        if self._last_connect_time and (now - self._last_connect_time) < QUICK_DISCONNECT_THRESHOLD:
            self._quick_disconnect_count += 1
            if self._quick_disconnect_count >= MAX_QUICK_DISCONNECT_COUNT:
                logger.error(
                    f"连续 {MAX_QUICK_DISCONNECT_COUNT} 次快速断连，停止重连。请检查 Bot 权限。"
                )
                return
        else:
            self._quick_disconnect_count = 0

        # 取消现有任务
        for task in (self._listen_task, self._heartbeat_task):
            if task and not task.done():
                task.cancel()

        # 开始重连
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            if self._shutting_down:
                return

            logger.info(f"开始重连 (attempt={attempt + 1}/{MAX_RECONNECT_ATTEMPTS})...")
            success = await self._reconnect(attempt)

            if success:
                self._reconnect_event.set()
                return

            # 退避等待
            backoff_idx = min(attempt, len(RECONNECT_BACKOFF) - 1)
            wait_time = RECONNECT_BACKOFF[backoff_idx]
            logger.info(f"等待 {wait_time}s 后重试...")
            await asyncio.sleep(wait_time)

        logger.error(f"重连失败: 已尝试 {MAX_RECONNECT_ATTEMPTS} 次")

    async def _reconnect(self, backoff_idx: int) -> bool:
        """
        执行一次重连。

        Args:
            backoff_idx: 当前重连索引

        Returns:
            bool: 是否成功
        """
        try:
            # 刷新 Token
            token = await self._ensure_token()
            if not token:
                logger.error("重连失败: 无法获取 Token")
                return False

            # 获取 Gateway URL
            gateway_url = await self._get_gateway_url()
            if not gateway_url:
                logger.error("重连失败: 无法获取 Gateway URL")
                return False

            # 关闭旧连接
            if self._ws and not self._ws.closed:
                await self._ws.close()

            # 重新连接
            connected = await self._open_ws(gateway_url)
            if connected:
                self._connected = True
                self._last_connect_time = time.time()
                logger.info("重连成功")
                return True

            return False

        except Exception as e:
            logger.error("重连异常: error_type=%s", type(e).__name__)
            return False

    def _handle_ws_close_code(self, code: int) -> Tuple[bool, bool]:
        """
        处理 WebSocket 关闭码。

        Args:
            code: 关闭码

        Returns:
            Tuple[should_clear_session, should_stop]:
                - should_clear_session: 是否清除 session
                - should_stop: 是否停止重连
        """
        if code == WS_CLOSE_INVALID_TOKEN:
            # Token 无效 → 清除缓存 Token
            self._access_token = None
            self._token_expires_at = 0
            return False, False

        if code in (WS_CLOSE_SESSION_INVALID_START, WS_CLOSE_SESSION_INVALID_SEQ, WS_CLOSE_SESSION_TIMEOUT):
            # 会话无效 → 清除 session_id
            self._session_id = None
            return True, False

        if code == WS_CLOSE_RATE_LIMITED:
            # 限速 → 等待
            logger.warning(f"限速 (code={code})，等待 {RATE_LIMIT_DELAY}s")
            return False, False

        if WS_CLOSE_SESSION_ERROR_START <= code <= WS_CLOSE_SESSION_ERROR_END:
            # 会话错误 → 清除 session_id
            self._session_id = None
            return True, False

        if code == WS_CLOSE_BOT_OFFLINE:
            # Bot 离线 → 停止重连
            logger.error(f"Bot 离线/沙盒 (code={code})，停止重连")
            return False, True

        if code == WS_CLOSE_BOT_BANNED:
            # Bot 被封禁 → 停止重连
            logger.error(f"Bot 被封禁 (code={code})，停止重连")
            return False, True

        return False, False

    async def _wait_for_reconnection(self, timeout: float = _RECONNECT_WAIT_SECONDS) -> bool:
        """
        等待重连完成。

        Args:
            timeout: 等待超时时间

        Returns:
            bool: 是否在超时前重连成功
        """
        self._reconnect_event.clear()
        try:
            await asyncio.wait_for(self._reconnect_event.wait(), timeout=timeout)
            return self._connected
        except asyncio.TimeoutError:
            return False

    # ─── 辅助方法 ──────────────────────────────────────────────────────

    def _guess_chat_type(self, chat_id: str) -> str:
        """
        猜测聊天类型。

        Args:
            chat_id: 聊天 ID

        Returns:
            str: "c2c" | "group" | "guild"
        """
        return self._chat_type_map.get(chat_id, "c2c")

    def _next_msg_seq(self) -> int:
        """获取下一个消息序列号"""
        self._msg_seq += 1
        return self._msg_seq

    @staticmethod
    def _parse_timestamp(ts) -> Optional[datetime]:
        """
        解析时间戳。

        Args:
            ts: ISO 8601 字符串或毫秒时间戳

        Returns:
            Optional[datetime]: 解析后的时间
        """
        if not ts:
            return None
        try:
            if isinstance(ts, (int, float)):
                return datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts)
            from datetime import timezone
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None

    async def _api_post(
        self,
        endpoint: str,
        body: dict,
        timeout: Optional[float] = None,
    ) -> SendResult:
        """
        发送 REST API POST 请求。

        Args:
            endpoint: API 端点 (相对路径)
            body: 请求体
            timeout: 超时时间

        Returns:
            SendResult: 发送结果
        """
        token = await self._ensure_token()
        if not token:
            return SendResult(success=False, error="无 access_token", retryable=True)

        url = f"{API_BASE}{endpoint}"
        headers = {
            "Authorization": f"QQBot {token}",
            "Content-Type": "application/json",
            "User-Agent": self._user_agent,
        }

        try:
            resp = await self._http_client.post(
                url,
                json=body,
                headers=headers,
                timeout=timeout or DEFAULT_API_TIMEOUT,
            )

            if resp.status_code in (200, 201, 204):
                data = resp.json() if resp.text else {}
                msg_id = data.get("id", "")
                if body.get("msg_type") == MSG_TYPE_INPUT_NOTIFY:
                    print(
                        f"\n  ⌨️ [QQBot] 输入状态已发送 → "
                        f"{_redact_endpoint(endpoint)}\n"
                    )
                    return SendResult(
                        success=True,
                        message_id=msg_id,
                        raw_response=data,
                    )
                # 本地测试可输出完整正文；生产仍默认脱敏。
                if "content" in body:
                    message_content = str(body.get("content") or "")
                elif "markdown" in body:
                    message_content = str((body.get("markdown") or {}).get("content") or "")
                else:
                    message_content = str(body.get("content") or "")
                logged_content = (
                    message_content
                    if _plaintext_message_logs_enabled()
                    else f"[redacted chars={len(message_content)}]"
                )
                print(
                    f"\n  ✅ [QQBot] 发送成功 → {_redact_endpoint(endpoint)}\n"
                    f"     消息ID: {_log_fingerprint(msg_id)}\n"
                    f"     内容: {logged_content}\n"
                )
                return SendResult(
                    success=True,
                    message_id=msg_id,
                    raw_response=data,
                )

            error_text = resp.text
            logger.error("QQ API 错误: status=%s response_chars=%s", resp.status_code, len(error_text))
            print(
                f"\n  ❌ [QQBot] 发送失败 → {_redact_endpoint(endpoint)}\n"
                f"     HTTP {resp.status_code}: [response redacted chars={len(error_text)}]\n"
            )

            retryable = resp.status_code not in (401, 403, 404)
            return SendResult(
                success=False,
                error=f"HTTP {resp.status_code}: {error_text}",
                retryable=retryable,
                raw_response=resp.json() if resp.text else None,
            )

        except Exception as e:
            logger.error("API 请求异常: error_type=%s", type(e).__name__)
            return SendResult(success=False, error=str(e), retryable=True)

    # ─── 运行入口 ──────────────────────────────────────────────────────

    async def run(self):
        """
        运行 QQ Bot 适配器 (阻塞)。

        连接 → 消息循环 → 断线重连
        """
        logger.info("QQ Bot 适配器启动...")

        while not self._shutting_down:
            try:
                success = await self.connect()
                if not success:
                    if self._shutting_down:
                        break
                    logger.warning("连接失败，5 秒后重试...")
                    await asyncio.sleep(5)
                    continue

                # 等待断连
                while self._connected and not self._shutting_down:
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("运行异常: error_type=%s", type(e).__name__)
                if not self._shutting_down:
                    await asyncio.sleep(5)

        await self.disconnect()
        logger.info("QQ Bot 适配器已停止")

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected
