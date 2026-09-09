
"""
微信机器人 HTTP 适配器

通过 HTTP API 与 weixinbot 微服务（Node.js）通信。
负责：
  - 发送消息（文本、媒体）
  - QR 码登录
  - 查询连接状态
  - 处理 Webhook 回调
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class WeixinAuthExpiredError(RuntimeError):
    """微信 iLink bot_token 已过期，需要重新扫码。"""


# ─── 数据模型 ──────────────────────────────────────────────────────────


@dataclass
class WeixinMessageEvent:
    """微信入站消息事件（由 Webhook 推送）"""

    event: str = "message"
    timestamp: int = 0
    from_user_id: str = ""
    to_user_id: str = ""
    text: str = ""
    context_token: str = ""
    account_id: str = ""
    message_id: Optional[int] = None
    session_id: str = ""
    image_path: str = ""
    voice_path: str = ""
    file_path: str = ""
    video_path: str = ""
    voice_media_type: str = ""
    file_media_type: str = ""

    @classmethod
    def from_webhook(cls, payload: dict) -> "WeixinMessageEvent":
        """从 Webhook payload 解析"""
        data = payload.get("data", {})
        return cls(
            event=payload.get("event", ""),
            timestamp=payload.get("timestamp", 0),
            from_user_id=data.get("fromUserId", ""),
            to_user_id=data.get("toUserId", ""),
            text=data.get("text", ""),
            context_token=data.get("contextToken", ""),
            account_id=data.get("accountId", ""),
            message_id=data.get("messageId"),
            session_id=data.get("sessionId", ""),
            image_path=data.get("imagePath", ""),
            voice_path=data.get("voicePath", ""),
            file_path=data.get("filePath", ""),
            video_path=data.get("videoPath", ""),
            voice_media_type=data.get("voiceMediaType", ""),
            file_media_type=data.get("fileMediaType", ""),
        )


@dataclass
class WeixinStatus:
    """微信服务状态"""

    running: bool = False
    started: bool = False
    connected: bool = False
    auth_expired: bool = False
    last_error: str = ""
    token: str = ""
    account_id: str = ""
    api_host: str = ""
    account_count: int = 0
    started_count: int = 0
    connected_count: int = 0
    auth_expired_count: int = 0
    max_accounts: int = 10
    accounts: List[dict] = field(default_factory=list)


# ─── 消息处理器类型 ────────────────────────────────────────────────────

MessageHandler = Callable[[WeixinMessageEvent], Coroutine[Any, Any, None]]


# ─── 微信适配器 ──────────────────────────────────────────────────────


class WeixinAdapter:
    """
    微信机器人 HTTP 适配器

    通过 HTTP API 与 weixinbot Node.js 微服务通信。
    """

    # QR 登录最长等待时间（5 分钟）
    LOGIN_TIMEOUT = 300.0

    def __init__(
        self,
        base_url: str = "http://localhost:3003",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._message_handler: Optional[MessageHandler] = None
        self._client = httpx.AsyncClient(timeout=timeout)
        # 长超时客户端，用于 QR 登录等耗时操作
        self._long_client = httpx.AsyncClient(timeout=self.LOGIN_TIMEOUT)

    def set_message_handler(self, handler: MessageHandler) -> None:
        """设置消息回调处理器"""
        self._message_handler = handler

    async def start(self, account_id: str = "") -> None:
        """启动一个微信账号；未指定时启动全部已注册账号。"""
        payload = {"accountId": account_id} if account_id else {}
        resp = await self._client.post(f"{self.base_url}/start", json=payload)
        if resp.status_code == 401:
            try:
                data = resp.json()
            except Exception:
                data = {}
            if data.get("error") == "auth_expired" or data.get("code") == -14:
                raise WeixinAuthExpiredError("微信登录已过期，需要重新扫码")
        resp.raise_for_status()
        logger.info("WeixinBot monitor started")

    async def stop(self, account_id: str = "") -> None:
        """停止一个微信账号；未指定时停止全部账号。"""
        payload = {"accountId": account_id} if account_id else {}
        resp = await self._client.post(f"{self.base_url}/stop", json=payload)
        resp.raise_for_status()
        logger.info("WeixinBot monitor stopped")

    async def get_status(self) -> WeixinStatus:
        """获取服务状态"""
        resp = await self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        data = resp.json()
        return WeixinStatus(
            running=data.get("status") == "ok",
            started=data.get("started", False),
            connected=data.get("connected", False),
            auth_expired=data.get("authExpired", False),
            last_error=data.get("lastError", ""),
            token=data.get("token", ""),
            account_id=data.get("accountId", ""),
            api_host=data.get("apiHost", ""),
            account_count=int(data.get("accountCount", 0) or 0),
            started_count=int(data.get("startedCount", 0) or 0),
            connected_count=int(data.get("connectedCount", 0) or 0),
            auth_expired_count=int(data.get("authExpiredCount", 0) or 0),
            max_accounts=int(data.get("maxAccounts", 10) or 10),
            accounts=list(data.get("accounts") or []),
        )

    async def login_with_qr(self) -> dict:
        """QR 码登录（一步完成：获取二维码 + 等待确认，最长 5 分钟）"""
        resp = await self._long_client.post(f"{self.base_url}/login/qr")
        resp.raise_for_status()
        return resp.json()

    async def start_qr_login(self) -> dict:
        """获取 QR 码（不等待确认，快速返回）"""
        resp = await self._client.post(f"{self.base_url}/login/qr/start")
        resp.raise_for_status()
        return resp.json()

    async def wait_qr_login(self, session_key: str) -> dict:
        """等待 QR 码确认（长轮询，最长 5 分钟）"""
        resp = await self._long_client.post(
            f"{self.base_url}/login/qr/wait",
            json={"sessionKey": session_key},
        )
        resp.raise_for_status()
        return resp.json()

    async def set_token(
        self,
        token: str,
        base_url: str = "",
        account_id: str = "",
        user_id: str = "",
    ) -> None:
        """设置完整扫码会话；base_url 必须和 token 一起切换。"""
        resp = await self._client.post(
            f"{self.base_url}/token",
            json={
                "token": token,
                "baseUrl": base_url,
                "accountId": account_id,
                "userId": user_id,
            },
        )
        resp.raise_for_status()

    async def send_message(
        self,
        to: str,
        text: str,
        context_token: str = "",
        account_id: str = "",
    ) -> dict:
        """发送文本消息"""
        payload: Dict[str, Any] = {"to": to, "text": text}
        if context_token:
            payload["contextToken"] = context_token
        if account_id:
            payload["accountId"] = account_id
        resp = await self._client.post(f"{self.base_url}/send/text", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def send_media(
        self,
        to: str,
        file_path: str,
        text: str = "",
        context_token: str = "",
        account_id: str = "",
    ) -> dict:
        """发送媒体消息"""
        payload: Dict[str, Any] = {"to": to, "filePath": file_path}
        if text:
            payload["text"] = text
        if context_token:
            payload["contextToken"] = context_token
        if account_id:
            payload["accountId"] = account_id
        resp = await self._client.post(f"{self.base_url}/send/media", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def remove_account(self, account_id: str) -> dict:
        """停止并删除一个微信账号的本地登录态。"""
        resp = await self._client.post(
            f"{self.base_url}/accounts/remove",
            json={"accountId": account_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def handle_webhook(self, payload: dict) -> None:
        """处理 Webhook 回调（由 FastAPI 路由调用）"""
        event = payload.get("event", "")

        if event == "message" and self._message_handler:
            msg_event = WeixinMessageEvent.from_webhook(payload)
            try:
                await self._message_handler(msg_event)
            except Exception as e:
                logger.error("Weixin message handler error: error_type=%s", type(e).__name__)
        else:
            logger.debug(f"Weixin webhook event: {event}")

    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        await self._client.aclose()
        await self._long_client.aclose()
