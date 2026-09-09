"""
WhatsApp HTTP 适配器

通过 HTTP API 与 app/whatsapp/service（基于 Baileys 的 Node.js 微服务）通信。
负责：
  - 发送消息（文本、媒体）
  - 发送已读回执
  - 发送 emoji 反应
  - 查询连接状态
  - 获取 QR 码
"""

import logging
import os
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


def _endpoint_summary(value: object) -> str:
    try:
        parsed = urlparse(str(value or ""))
        host = parsed.hostname or "-"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme or '-'}://{host}{port}"
    except (TypeError, ValueError):
        return "invalid-url"


# ─── 数据模型 ──────────────────────────────────────────────────────────


@dataclass
class WhatsAppMessageEvent:
    """WhatsApp 入站消息事件（由 Webhook 推送）"""

    event: str  # "message" | "message.reaction" | "message.read" | "connection.update"
    timestamp: int
    message_id: str = ""
    from_jid: str = ""
    from_number: str = ""
    chat_type: str = "dm"  # "dm" | "group"
    text: str = ""
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    quoted_message_id: str = ""
    group_jid: str = ""
    participant: str = ""
    reaction: str = ""
    connection_state: str = ""

    @classmethod
    def from_webhook(cls, data: dict) -> "WhatsAppMessageEvent":
        """从 Webhook payload 解析"""
        d = data.get("data", {})
        return cls(
            event=data.get("event", ""),
            timestamp=data.get("timestamp", 0),
            message_id=d.get("messageId", ""),
            from_jid=d.get("from", ""),
            from_number=d.get("fromNumber", ""),
            chat_type=d.get("chatType", "dm"),
            text=d.get("text", ""),
            media_urls=d.get("mediaUrls", []),
            media_types=d.get("mediaTypes", []),
            quoted_message_id=d.get("quotedMessageId", ""),
            group_jid=d.get("groupJid", ""),
            participant=d.get("participant", ""),
            reaction=d.get("reaction", ""),
            connection_state=d.get("connectionState", ""),
        )


@dataclass
class WhatsAppStatus:
    """WhatsApp 服务状态"""

    running: bool = False
    connected: bool = False
    connection_state: str = "disconnected"
    phone_number: str = ""
    last_connected_at: str = ""
    uptime: int = 0
    webhook_url: str = ""


# ─── 消息处理器类型 ────────────────────────────────────────────────────

MessageHandler = Callable[[WhatsAppMessageEvent], Coroutine[Any, Any, None]]


# ─── WhatsApp 适配器 ──────────────────────────────────────────────────


class WhatsAppAdapter:
    """
    WhatsApp HTTP 适配器

    通过 HTTP API 与 WhatsApp 微服务通信。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self._base_url = (base_url or os.getenv("WHATSAPP_SERVICE_URL", "http://localhost:3001")).rstrip("/")
        self._api_token = api_token or os.getenv("WHATSAPP_API_TOKEN", "")
        self._timeout = timeout
        self._http_client: Optional[httpx.AsyncClient] = None
        self._message_handler: Optional[MessageHandler] = None

    # ─── 生命周期 ──────────────────────────────────────────────────

    async def start(self) -> None:
        """初始化 HTTP 客户端"""
        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            headers=self._auth_headers,
        )
        logger.info("WhatsApp 适配器已初始化 → %s", _endpoint_summary(self._base_url))

        # 检查服务状态
        try:
            status = await self.get_status()
            if status.connected:
                logger.info("WhatsApp 已连接（账号标识不写入日志）")
            else:
                logger.info(f"WhatsApp 未连接 (状态: {status.connection_state})")
        except Exception as e:
            logger.warning("无法连接 WhatsApp 服务: error_type=%s", type(e).__name__)

    async def stop(self) -> None:
        """关闭 HTTP 客户端"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("WhatsApp 适配器已关闭")

    # ─── 消息处理 ──────────────────────────────────────────────────

    def set_message_handler(self, handler: MessageHandler) -> None:
        """设置入站消息处理器"""
        self._message_handler = handler

    async def handle_webhook(self, payload: dict) -> None:
        """处理来自 WhatsApp 微服务的 Webhook 回调"""
        event = WhatsAppMessageEvent.from_webhook(payload)

        if event.event == "connection.update":
            logger.info(f"WhatsApp 连接状态更新: {event.connection_state}")
            return

        if event.event == "message" and self._message_handler:
            try:
                await self._message_handler(event)
            except Exception as e:
                logger.error("处理 WhatsApp 消息失败: error_type=%s", type(e).__name__)

    # ─── 发送消息 ──────────────────────────────────────────────────

    async def send_message(
        self,
        to: str,
        text: str,
        media_url: Optional[str] = None,
        quoted_message_id: Optional[str] = None,
    ) -> dict:
        """
        发送 WhatsApp 消息

        Args:
            to: 目标号码（E.164 格式）或群组 JID
            text: 消息文本
            media_url: 媒体 URL（可选）
            quoted_message_id: 引用回复的消息 ID（可选）

        Returns:
            dict: {"success": bool, "messageId": str, "error": str}
        """
        body: dict = {"to": to, "text": text}
        if media_url:
            body["mediaUrl"] = media_url
        if quoted_message_id:
            body["quotedMessageId"] = quoted_message_id

        return await self._post("/api/send", body)

    async def send_read_receipt(self, jid: str, message_ids: List[str]) -> dict:
        """发送已读回执"""
        return await self._post("/api/read", {"jid": jid, "messageIds": message_ids})

    async def send_reaction(self, jid: str, message_id: str, emoji: str) -> dict:
        """发送 emoji 反应"""
        return await self._post("/api/react", {"jid": jid, "messageId": message_id, "emoji": emoji})

    # ─── 状态查询 ──────────────────────────────────────────────────

    async def get_status(self) -> WhatsAppStatus:
        """获取 WhatsApp 服务状态"""
        data = await self._get("/api/status")
        return WhatsAppStatus(
            running=data.get("running", False),
            connected=data.get("connected", False),
            connection_state=data.get("connectionState", "disconnected"),
            phone_number=data.get("phoneNumber", ""),
            last_connected_at=data.get("lastConnectedAt", ""),
            uptime=data.get("uptime", 0),
            webhook_url=data.get("webhookUrl", ""),
        )

    async def get_qr_code(self) -> Optional[str]:
        """获取 QR 码数据"""
        data = await self._get("/api/qr")
        return data.get("qr")

    async def start_qr_login(self) -> dict:
        """退出当前账号、清理旧凭据并生成新的扫码登录二维码。"""
        return await self._post("/api/login/qr/start", {})

    async def configure_webhook(self, url: str, secret: Optional[str] = None) -> dict:
        """配置 WhatsApp 微服务的 Webhook URL"""
        body: dict = {"url": url}
        if secret:
            body["secret"] = secret
        return await self._post("/api/webhook", body)

    # ─── 私有方法 ──────────────────────────────────────────────────

    @property
    def _auth_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        return headers

    async def _get(self, path: str) -> dict:
        if not self._http_client:
            raise RuntimeError("适配器未启动")
        resp = await self._http_client.get(path)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, body: dict) -> dict:
        if not self._http_client:
            raise RuntimeError("适配器未启动")
        resp = await self._http_client.post(path, json=body)
        resp.raise_for_status()
        return resp.json()
