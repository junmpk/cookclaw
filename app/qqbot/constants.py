"""
QQ Bot 共享常量

源码位置: gateway/platforms/qqbot/constants.py
版本: QQBOT_VERSION = "1.1.0"
"""

# ─── 版本 ──────────────────────────────────────────────────────────────
QQBOT_VERSION: str = "1.1.0"

# ─── API 端点 ──────────────────────────────────────────────────────────
API_BASE: str = "https://api.sgroup.qq.com"
TOKEN_URL: str = "https://bots.qq.com/app/getAppAccessToken"
PORTAL_HOST: str = "q.qq.com"

# ─── 超时 ──────────────────────────────────────────────────────────────
DEFAULT_API_TIMEOUT: float = 30.0       # REST API 默认超时 (秒)
FILE_UPLOAD_TIMEOUT: float = 120.0      # 文件上传超时 (秒)
CONNECT_TIMEOUT_SECONDS: float = 20.0   # WebSocket 连接超时 (秒)

# ─── 消息 ──────────────────────────────────────────────────────────────
MAX_MESSAGE_LENGTH: int = 4000          # 单条消息最大字符数

# ─── 消息类型 ──────────────────────────────────────────────────────────
MSG_TYPE_TEXT: int = 0          # 纯文本消息
MSG_TYPE_MARKDOWN: int = 2     # Markdown 消息
MSG_TYPE_INPUT_NOTIFY: int = 6 # 输入状态通知
MSG_TYPE_MEDIA: int = 7        # 媒体消息

# ─── 媒体类型 ──────────────────────────────────────────────────────────
MEDIA_TYPE_IMAGE: int = 1  # 图片
MEDIA_TYPE_VIDEO: int = 2  # 视频
MEDIA_TYPE_VOICE: int = 3  # 语音
MEDIA_TYPE_FILE: int = 4   # 文件

# ─── 重连 ──────────────────────────────────────────────────────────────
RECONNECT_BACKOFF: list = [2, 5, 10, 30, 60]  # 退避时间表 (秒)
MAX_RECONNECT_ATTEMPTS: int = 100              # 最大重连次数
RATE_LIMIT_DELAY: int = 60                     # 限速等待时间 (秒)

# ─── 快速断连检测 ──────────────────────────────────────────────────────
QUICK_DISCONNECT_THRESHOLD: float = 5.0   # 快速断连判定阈值 (秒)
MAX_QUICK_DISCONNECT_COUNT: int = 3       # 连续快速断连上限

# ─── 扫码配置 ──────────────────────────────────────────────────────────
ONBOARD_POLL_INTERVAL: float = 2.0  # 轮询扫码结果间隔 (秒)

# ─── 消息去重 ──────────────────────────────────────────────────────────
DEDUP_WINDOW_SECONDS: int = 300  # 去重时间窗口 (5 分钟)
DEDUP_MAX_SIZE: int = 1000       # 最大缓存条目数

# ─── Intents 位掩码 ────────────────────────────────────────────────────
INTENT_DIRECT_MESSAGE: int = 1 << 12          # 4096   — 私信事件
INTENT_C2C_GROUP_AT_MESSAGES: int = 1 << 25   # 33554432 — C2C + 群@消息
INTENT_PUBLIC_GUILD_MESSAGES: int = 1 << 30   # 1073741824 — 频道消息

# 默认 Intents 组合
DEFAULT_INTENTS: int = (
    INTENT_C2C_GROUP_AT_MESSAGES
    | INTENT_PUBLIC_GUILD_MESSAGES
    | INTENT_DIRECT_MESSAGE
)

# ─── WebSocket OP Codes ────────────────────────────────────────────────
WS_OP_DISPATCH: int = 0          # 服务端推送消息
WS_OP_HEARTBEAT: int = 1         # 心跳
WS_OP_IDENTIFY: int = 2          # 首次连接 Identify
WS_OP_RESUME: int = 6            # 断线恢复 Resume
WS_OP_RECONNECT: int = 7         # 服务端要求重连
WS_OP_HELLO: int = 10            # 握手 Hello
WS_OP_HEARTBEAT_ACK: int = 11    # 心跳 ACK

# ─── WebSocket 关闭码 ──────────────────────────────────────────────────
WS_CLOSE_INVALID_TOKEN: int = 4004          # Token 无效
WS_CLOSE_SESSION_INVALID_START: int = 4006  # 会话无效 (开始)
WS_CLOSE_SESSION_INVALID_SEQ: int = 4007    # 序列号无效
WS_CLOSE_RATE_LIMITED: int = 4008           # 限速
WS_CLOSE_SESSION_TIMEOUT: int = 4009        # 会话超时
WS_CLOSE_SESSION_ERROR_START: int = 4900    # 会话错误范围起始
WS_CLOSE_SESSION_ERROR_END: int = 4913      # 会话错误范围结束
WS_CLOSE_BOT_OFFLINE: int = 4914            # Bot 离线/沙盒
WS_CLOSE_BOT_BANNED: int = 4915             # Bot 被封禁
