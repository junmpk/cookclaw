# QQ 通道服务架构文档

> **更新日期**：2026-08-07  
> **相关文件**：`app/qqbot/adapter.py`, `app/qqbot/constants.py`, `app/main.py`

---

## 一、整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                     QQ 官方服务器                              │
│                  (WebSocket Gateway)                         │
└────────────────┬─────────────────────────────────────────────┘
                 │ WebSocket 长连接
                 │ (access_token 认证)
                 ▼
┌──────────────────────────────────────────────────────────────┐
│              QQ Bot Adapter (app/qqbot/adapter.py)           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ WebSocket 连接│  │ 消息解析     │  │ 消息路由     │      │
│  │ _open_ws()   │  │ _on_message()│  │ _handle_*()  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────┬─────────────────────────────────────────────┘
                 │ 回调 _qqbot_message_handler
                 ▼
┌──────────────────────────────────────────────────────────────┐
│          主服务 (app/agent/participle_agent.py)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ qqbot_chat() │  │ 意图分类     │  │ 业务处理     │      │
│  │ 入口函数     │  │ → 路由决策   │  │ → AI 生成    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────┬─────────────────────────────────────────────┘
                 │ 返回格式化回复
                 ▼
┌──────────────────────────────────────────────────────────────┐
│              QQ Bot Adapter (发送回复)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 格式转换     │  │ 长消息分块   │  │ send() 发送  │      │
│  │ JSON → MD    │  │ 图片/文本    │  │ → QQ 服务器  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────┬─────────────────────────────────────────────┘
                 │ WebSocket
                 ▼
┌──────────────────────────────────────────────────────────────┐
│                     QQ 用户                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 二、连接方式：WebSocket 长连接

QQ Bot 使用 **WebSocket 长连接**（非 HTTP 回调 webhook）连接到 QQ 官方 API。

**关键代码**（`app/qqbot/adapter.py`）：

```python
async def connect(self) -> bool:
    """建立与 QQ 服务器的 WebSocket 长连接"""
    # 1. 获取 WebSocket Gateway URL
    gateway_url = await self._get_gateway_url()
    
    # 2. 建立 WebSocket 连接
    self._ws = await self._open_ws(gateway_url)
    
    # 3. 发送 IDENTIFY 认证
    await self._send_identify()
    
    # 4. 启动心跳和消息监听
    await asyncio.gather(
        self._heartbeat_loop(),
        self._listen_loop(),
    )
```

**WebSocket 协议操作码**（`app/qqbot/constants.py`）：

```python
WS_OP_DISPATCH: int = 0          # 服务端推送消息
WS_OP_HEARTBEAT: int = 1         # 心跳
WS_OP_IDENTIFY: int = 2          # 首次连接 Identify
WS_OP_RESUME: int = 6            # 断线恢复 Resume
WS_OP_RECONNECT: int = 7         # 服务端要求重连
WS_OP_HELLO: int = 10            # 握手 Hello
WS_OP_HEARTBEAT_ACK: int = 11    # 心跳 ACK
```

---

## 三、认证流程

### 3.1 三段式认证

```
1. app_id + client_secret → access_token
   ↓
2. access_token → gateway URL
   ↓
3. 连接到 gateway URL，建立 WebSocket 长连接
```

### 3.2 Step 1: 获取 Access Token

**API 端点**：`https://bots.qq.com/app/getAppAccessToken`

**请求**：
```http
POST https://bots.qq.com/app/getAppAccessToken
Content-Type: application/json

{
  "appId": "your_app_id",
  "clientSecret": "your_client_secret"
}
```

**响应**：
```json
{
  "access_token": "xxx",
  "expires_in": 7200  // 2 小时
}
```

**代码实现**（`app/qqbot/adapter.py`）：

```python
async def _ensure_token(self) -> Optional[str]:
    """获取并缓存 access_token"""
    # 检查缓存
    if self._access_token and time.time() < self._token_expires_at:
        return self._access_token
    
    # 用 app_id + client_secret 换取 access_token
    resp = await self._http_client.post(
        TOKEN_URL,  # https://bots.qq.com/app/getAppAccessToken
        json={
            "appId": self._app_id,
            "clientSecret": self._client_secret,
        },
    )
    
    data = await resp.json()
    self._access_token = data.get("access_token")
    expires_in = int(data.get("expires_in", 7200))
    self._token_expires_at = time.time() + expires_in
    
    return self._access_token
```

### 3.3 Step 2: 获取 Gateway URL

**API 端点**：`https://api.sgroup.qq.com/gateway`

**请求**：
```http
GET https://api.sgroup.qq.com/gateway
Authorization: QQBot <your_access_token>
```

**响应**：
```json
{
  "url": "wss://api.sgroup.qq.com/websocket"
}
```

**代码实现**（`app/qqbot/adapter.py:461-480`）：

```python
async def _get_gateway_url(self) -> Optional[str]:
    """获取 WebSocket Gateway URL"""
    try:
        resp = await self._http_client.get(
            f"{API_BASE}/gateway",  # https://api.sgroup.qq.com/gateway
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
```

### 3.4 Step 3: 建立 WebSocket 连接

**代码实现**（`app/qqbot/adapter.py:482-520`）：

```python
async def _open_ws(self, url: str) -> bool:
    """建立 WebSocket 连接"""
    import aiohttp
    
    if self._session is None or self._session.closed:
        self._session = aiohttp.ClientSession()
    
    try:
        self._ws = await self._session.ws_connect(
            url,
            timeout=CONNECT_TIMEOUT_SECONDS,
        )
        return True
    except Exception as e:
        logger.error("WebSocket 连接失败: %s", e)
        return False
```

### 3.5 Token 缓存机制

代码中有 token 缓存，避免频繁请求：

```python
async def _ensure_token(self) -> Optional[str]:
    # 检查缓存（过期前 5 分钟刷新）
    if self._access_token and time.time() < self._token_expires_at - 300:
        return self._access_token
    
    # 缓存失效，重新获取
    resp = await self._http_client.post(TOKEN_URL, json={...})
    self._access_token = data.get("access_token")
    expires_in = int(data.get("expires_in", 7200))
    self._token_expires_at = time.time() + expires_in
    
    return self._access_token
```

---

## 四、消息接收

### 4.1 消息监听循环

```python
async def _listen_loop(self):
    """持续监听 WebSocket 消息"""
    while self._ws and not self._ws.closed:
        try:
            msg = await self._ws.receive()
            
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                op = data.get("op")
                
                if op == WS_OP_DISPATCH:
                    event_type = data.get("t")
                    event_data = data.get("d")
                    await self._on_message(event_type, event_data)
                
                elif op == WS_OP_HELLO:
                    # 握手成功
                    pass
                
                elif op == WS_OP_HEARTBEAT_ACK:
                    # 心跳确认
                    pass
            
        except Exception as e:
            logger.error("消息监听异常: %s", e)
            break
```

### 4.2 消息路由

```python
async def _on_message(self, event_type: str, d: Any):
    """处理从 QQ 服务器接收到的消息"""
    handler_map = {
        "C2C_MESSAGE_CREATE": self._handle_c2c_message,      # 私聊
        "GROUP_AT_MESSAGE_CREATE": self._handle_group_message, # 群聊@
        "GUILD_MESSAGE_CREATE": self._handle_guild_message,   # 频道
        "GUILD_AT_MESSAGE_CREATE": self._handle_guild_message,
        "DIRECT_MESSAGE_CREATE": self._handle_dm_message,     # 私信
    }
    
    handler = handler_map.get(event_type)
    if handler:
        await handler(d)
```

### 4.3 消息解析

**消息结构**：
```json
{
  "op": 0,
  "t": "C2C_MESSAGE_CREATE",
  "s": 12345,
  "d": {
    "id": "message_id",
    "content": "用户消息文本",
    "timestamp": "2026-08-07T10:00:00+08:00",
    "author": {
      "id": "user_id",
      "username": "用户名"
    },
    "attachments": [
      {
        "content_type": "image",
        "url": "https://..."
      }
    ]
  }
}
```

---

## 五、消息转发

### 5.1 回调机制

在 `app/main.py` 中设置回调：

```python
# 创建 QQ Adapter
qq_adapter = await _create_qqbot_adapter()

# 设置消息回调
qq_adapter.set_message_handler(_qqbot_message_handler)

# 启动后台任务
asyncio.create_task(qq_adapter.run())
```

### 5.2 回调处理函数

```python
async def _qqbot_message_handler(event):
    """接收 QQ 消息并转发到主服务"""
    
    # 1. 构建会话 ID
    thread_id = build_qq_thread_id(
        event.source.chat_type,    # 私聊/群聊/频道
        event.source.chat_id,      # 聊天 ID
        event.source.user_id,      # 用户 ID
    )
    
    # 2. 调用主服务（participle_agent.qqbot_chat）
    raw_response = await qqbot_chat(
        event.text,                            # 用户消息
        thread_id=thread_id,                   # 会话 ID
        on_search_progress=_qq_search_progress, # 进度回调
        source_message_type=source_message_type,
    )
    
    # 3. 处理响应
    # raw_response 是 JSON 字符串，包含 AI 生成的回复
```

### 5.3 关键文件

| 文件 | 函数 | 职责 |
|---|---|---|
| `app/main.py:632-665` | `_qqbot_message_handler` | 设置回调 |
| `app/agent/participle_agent.py:5119-5174` | `qqbot_chat()` | 主服务入口 |

---

## 六、消息回复

### 6.1 回复流程

```python
# 在 _qqbot_message_handler 中
if reply_text.strip() and _qq_adapter:
    # 长消息分块发送
    for message_index, message_text in enumerate(reply_messages):
        if not str(message_text or "").strip():
            continue
        
        # 调用 QQ Adapter 发送
        await _qq_adapter.send(
            event.source.chat_id,           # 目标聊天 ID
            message_text,                    # 消息内容（Markdown 格式）
            reply_to=event.message_id if message_index == 0 else None,  # 引用原消息
        )
```

### 6.2 QQ Adapter 发送方法

```python
async def send(self, chat_id: str, content: str, reply_to: str = None):
    """发送消息到 QQ"""
    # 根据 chat_type 选择不同的 API
    if chat_type == "c2c":
        # 私聊 API
        await self._send_c2c_message(chat_id, content, reply_to)
    elif chat_type == "group":
        # 群聊 API
        await self._send_group_message(chat_id, content, reply_to)
    elif chat_type == "guild":
        # 频道 API
        await self._send_guild_message(chat_id, content, reply_to)
```

---

## 七、消息流转时序图

```
用户发QQ消息
    │
    ▼
┌─────────────────┐
│ QQ 官方服务器    │
└────────┬────────┘
         │ WebSocket 推送
         │ { "op": 0, "t": "C2C_MESSAGE_CREATE", "d": {...} }
         ▼
┌─────────────────────────────────────┐
│ QQ Bot Adapter                       │
│ 1. _listen_loop() 接收消息          │
│ 2. _on_message() 解析事件类型        │
│ 3. _handle_c2c_message() 处理私聊   │
│ 4. 构建 MessageEvent 对象            │
│ 5. 调用 message_handler(event)      │
└────────┬────────────────────────────┘
         │ 回调 _qqbot_message_handler
         ▼
┌─────────────────────────────────────┐
│ app/main.py                          │
│ 1. 构建 thread_id                   │
│ 2. 调用 qqbot_chat(text, thread_id) │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ participle_agent.qqbot_chat()        │
│ 1. 加载会话状态                      │
│ 2. 意图分类 (qwen-plus)             │
│ 3. 路由决策                          │
│ 4. 业务处理（检索/设备/闲聊）        │
│ 5. 生成回复                          │
│ 6. 返回 JSON 字符串                  │
└────────┬────────────────────────────┘
         │ raw_response (JSON)
         ▼
┌─────────────────────────────────────┐
│ app/main.py                          │
│ 1. 解析 JSON 响应                    │
│ 2. 格式化为 Markdown                │
│ 3. 长消息分块                        │
│ 4. 调用 _qq_adapter.send()          │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ QQ Bot Adapter.send()                │
│ 1. 根据 chat_type 选择 API          │
│ 2. POST 到 QQ 官方 API              │
│ 3. 发送消息给用户                    │
└────────┬────────────────────────────┘
         │ HTTP API
         ▼
┌─────────────────┐
│ QQ 官方服务器    │
└────────┬────────┘
         │ 推送给用户
         ▼
┌─────────────────┐
│ QQ 用户          │
└─────────────────┘
```

---

## 八、关键配置

### 8.1 环境变量

在 `.env` 文件中配置：

```bash
QQ_BOT_ENABLED=true              # 启用 QQ bot
QQ_APP_ID=your_app_id            # QQ Bot App ID
QQ_CLIENT_SECRET=your_secret     # QQ Bot Client Secret
```

### 8.2 API 端点常量

**文件**：`app/qqbot/constants.py`

```python
API_BASE: str = "https://api.sgroup.qq.com"
TOKEN_URL: str = "https://bots.qq.com/app/getAppAccessToken"
PORTAL_HOST: str = "q.qq.com"

# 超时配置
DEFAULT_API_TIMEOUT: float = 30.0       # REST API 默认超时 (秒)
FILE_UPLOAD_TIMEOUT: float = 120.0      # 文件上传超时 (秒)
CONNECT_TIMEOUT_SECONDS: float = 20.0   # WebSocket 连接超时 (秒)

# 重连配置
RECONNECT_BACKOFF: list = [2, 5, 10, 30, 60]  # 退避时间表 (秒)
MAX_RECONNECT_ATTEMPTS: int = 100              # 最大重连次数
```

### 8.3 启动流程

**文件**：`app/main.py`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ...
    if settings.QQ_BOT_ENABLED:
        # 创建 QQ Adapter
        qq_adapter = await _create_qqbot_adapter()
        
        # 设置消息回调
        qq_adapter.set_message_handler(_qqbot_message_handler)
        
        # 启动后台任务
        asyncio.create_task(qq_adapter.run())
    
    yield
```

---

## 九、总结

| 问题 | 答案 |
|---|---|
| **连接方式** | WebSocket 长连接（非 HTTP 回调） |
| **认证方式** | app ID + secret → access_token → WebSocket 认证 |
| **Gateway URL 获取** | `GET https://api.sgroup.qq.com/gateway` |
| **消息接收** | WebSocket 推送，按事件类型路由 |
| **消息转发** | 回调 `_qqbot_message_handler` → `qqbot_chat()` |
| **消息回复** | 格式化 → `_qq_adapter.send()` → QQ API |

### 核心文件

| 文件 | 职责 |
|---|---|
| `app/qqbot/adapter.py` | QQ 适配器（WebSocket + 消息处理） |
| `app/qqbot/constants.py` | API 端点、超时、重连等常量 |
| `app/main.py` | 启动 + 回调设置 |
| `app/agent/participle_agent.py` | 主服务入口（`qqbot_chat()`） |

### 为什么需要两步认证

1. **安全**：access_token 有有效期（2 小时），需要定期刷新
2. **灵活**：QQ 可以动态分配不同的 WebSocket 服务器（负载均衡）
3. **标准化**：这是 QQ Bot API 的标准流程，和微信、Discord 等类似

---

## 十、故障排查

### 10.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|---|---|---|
| 无法连接 QQ | access_token 无效 | 检查 app_id 和 client_secret 是否正确 |
| Gateway URL 获取失败 | token 过期 | 检查 token 缓存逻辑 |
| 消息收不到 | WebSocket 断连 | 检查心跳机制和重连逻辑 |
| 回复发送失败 | API 权限不足 | 检查 QQ Bot 后台配置 |

### 10.2 日志查看

```bash
# 查看 QQ Bot 日志
tail -f logs/qqbot.log

# 查看主服务日志
tail -f logs/app.log
```

### 10.3 调试技巧

```python
# 启用调试日志
import logging
logging.getLogger("app.qqbot").setLevel(logging.DEBUG)

# 查看 Gateway URL
logger.debug("Gateway URL received: %s", gateway_url)
```

---

**文档结束**

如有问题，请查看：
- QQ Bot 官方文档：https://bot.q.qq.com/wiki/
- CookClaw 架构文档：`docs/architecture.md`
