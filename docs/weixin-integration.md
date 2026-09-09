# 微信机器人集成指南 — CookClaw

> 本文档介绍如何通过 weixinbot 微服务（Node.js）将微信 iLink 机器人集成到 CookClaw 项目中。
> 基于 `@tencent-weixin/openclaw-weixin` v2.4.3 提取的独立 SDK。

---

## 架构概览

```
多个微信账号 / 用户
    ↕ (微信 iLink 协议)
weixinbot (Node.js / iLink HTTP JSON API)
    ↕ (HTTP API + Webhook)
CookClaw (Python / FastAPI)
    ↕
AI Agent (食谱搜索、设备控制)
```

### 组件说明

| 组件 | 技术栈 | 职责 |
|------|--------|------|
| **weixinbot** | Node.js + TypeScript | 微信 iLink 协议处理、QR 码登录、消息收发、CDN 媒体处理 |
| **CookClaw** | Python + FastAPI | AI Agent、食谱搜索、设备控制、Webhook 接收 |
| **WeixinAdapter** | Python + httpx | HTTP 客户端，调用 weixinbot API |

### 架构模式

与 WhatsApp 集成一致：Python 主进程在启动时通过 `subprocess` 拉起 Node.js 微服务，两者通过 HTTP API + Webhook 通信。

```
┌──────────────────────────────────────────────────────────────┐
│ CookClaw (Python FastAPI :8000)                               │
│                                                                │
│  main.py lifespan()                                            │
│    ├─ 启动 weixinbot 子进程: node dist/server.js               │
│    ├─ 等待 /health 就绪                                        │
│    ├─ 恢复全部已保存账号 → POST /token                         │
│    ├─ POST /start 启动全部账号的独立监听任务                    │
│    └─ 关闭时 → POST /stop + terminate()                       │
│                                                                │
│  API 路由:                                                     │
│    GET  /api/v1/weixin/status        → 查询状态                │
│    POST /api/v1/weixin/login/qr      → 一步式 QR 登录          │
│    POST /api/v1/weixin/login/qr/start → 获取 QR 码             │
│    POST /api/v1/weixin/login/qr/wait  → 等待扫码确认           │
│    POST /api/v1/weixin/webhook       → 接收消息回调            │
│                                                                │
│  消息处理:                                                     │
│    Webhook → WeixinAdapter.handle_webhook()                    │
│           → _weixin_message_handler()                          │
│           → qqbot_chat(text, thread_id=account_id+from_user_id) │
│           → WeixinAdapter.send_message(account_id, to, reply)   │
└──────────────────────────────────────────────────────────────┘
         ↕ HTTP API (localhost:3003)
┌──────────────────────────────────────────────────────────────┐
│ weixinbot (Node.js :3003)                                     │
│                                                                │
│  server.ts — HTTP 路由:                                        │
│    GET  /health           → 健康检查                           │
│    POST /login/qr         → 一步式 QR 登录                     │
│    POST /login/qr/start   → 获取 QR 码                        │
│    POST /login/qr/wait    → 等待扫码确认                       │
│    POST /token            → 设置 bot_token                    │
│    GET  /accounts         → 列出全部账号状态                   │
│    POST /accounts/remove  → 移除指定账号                       │
│    POST /start            → 启动消息监听                       │
│    POST /stop             → 停止消息监听                       │
│    POST /send/text        → 发送文本消息                       │
│    POST /send/media       → 发送媒体消息                       │
│                                                                │
│  SDK 核心:                                                     │
│    auth.ts    → QR 码登录流程                                  │
│    api.ts     → iLink HTTP API 通信层                          │
│    monitor.ts → 长轮询消息监听                                 │
│    messaging  → 消息发送（文本/图片/视频/文件）                  │
│    cdn.ts     → CDN 上传下载（AES-128-ECB 加密）               │
│    media.ts   → 媒体下载与 SILK→WAV 转码                       │
│    state.ts   → 状态持久化（~/.weixinbot/）                    │
│    markdown-filter.ts → Markdown 出站过滤                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 前置条件

| 条件 | 要求 |
|------|------|
| **Node.js** | v20+ |
| **Python** | v3.12+ |
| **微信** | 可用的微信账号（用于扫码登录） |
| **手机** | 已安装微信，用于扫码确认 |

### 第一步：编译 weixinbot 微服务

```bash
cd app/weixinbot

# 安装依赖
npm install

# 编译 TypeScript
npm run build
```

编译输出在 `app/weixinbot/dist/` 目录。

### 第二步：配置 CookClaw

在 CookClaw 的 `.env` 文件中添加：

```bash
# ─── 微信机器人 ──────────────────────────────────────
WEIXIN_ENABLED=true
WEIXIN_SERVICE_URL=http://localhost:3003
WEIXIN_TOKEN=                    # 可选，首次通过 QR 码登录后自动获取
WEIXIN_WEBHOOK_SECRET=           # 可选，Webhook 密钥
WEIXIN_MAX_ACCOUNTS=10           # 默认最多 10 个，范围 1～50
```

### 第三步：启动 CookClaw

```bash
# 在项目根目录
uv run python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

CookClaw 启动时会自动：
1. 检查 `WEIXIN_ENABLED=true`
2. 通过 `subprocess` 启动 `node dist/server.js`（端口 3003）
3. 等待健康检查通过
4. 从 Python 与 Node 状态目录恢复所有已保存账号，并分别启动消息监听

### 第四步：QR 码登录

**方式一：一步式登录**（阻塞等待，最长 5 分钟）

```bash
curl -X POST http://localhost:8000/api/v1/weixin/login/qr
```

返回：
```json
{
  "success": true,
  "botToken": "xxx@im.bot:060000xxx",
  "accountId": "xxx@im.bot",
  "baseUrl": "https://ilinkai.weixin.qq.com",
  "userId": "xxx@im.wechat",
  "message": "登录成功"
}
```

**方式二：两步式登录**（推荐，前端可先展示二维码）

```bash
# 第 1 步：获取 QR 码（立即返回）
curl -X POST http://localhost:8000/api/v1/weixin/login/qr/start

# 返回：
# {
#   "qrcodeUrl": "https://liteapp.weixin.qq.com/q/xxx?qrcode=xxx&bot_type=3",
#   "sessionKey": "94fb3072-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
#   "message": "请用手机微信扫描二维码"
# }

# 第 2 步：等待扫码确认（长轮询，最长 5 分钟）
curl -X POST http://localhost:8000/api/v1/weixin/login/qr/wait \
  -H "Content-Type: application/json" \
  -d '{"session_key": "94fb3072-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}'

# 返回：
# {
#   "success": true,
#   "botToken": "xxx@im.bot:060000xxx",
#   "message": "登录成功"
# }
```

重复第四步即可继续添加账号，登录页会展示每个账号的在线、过期和未连接状态。达到
`WEIXIN_MAX_ACCOUNTS` 后需先移除不用的账号。

### 第五步：启动消息监听

登录成功后，启动消息监听：

```bash
curl -X POST http://localhost:3003/start \
  -H "Content-Type: application/json" \
  -d '{"accountId":"xxx@im.bot"}'
```

省略 `accountId` 时启动全部已注册账号。CookClaw 启动时会自动执行此步骤。

---

## weixinbot 微服务 API

### 基础 URL

```
http://localhost:3003
```

### 端点列表

#### 健康检查

```bash
GET /health
```

```json
{
  "status": "ok",
  "started": true,
  "connected": true,
  "accountCount": 2,
  "connectedCount": 2,
  "maxAccounts": 10,
  "accounts": [{"accountId":"a@im.bot","connected":true}]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | `string` | 服务状态，`"ok"` 表示正常 |
| `started` | `bool` | 是否至少有一个消息监听已启动 |
| `accountCount` | `int` | 已注册账号总数 |
| `connectedCount` | `int` | 当前在线账号数 |
| `maxAccounts` | `int` | 允许的账号上限 |
| `accounts` | `array` | 每个账号的连接、过期和错误状态 |
| `token` | `string` | `"configured"` / `"not set"` / `"not initialized"` |

#### QR 码登录（一步式）

```bash
POST /login/qr
```

阻塞等待扫码确认，最长 5 分钟。返回 `QRLoginResult`。

#### 获取 QR 码

```bash
POST /login/qr/start
```

```json
{
  "qrcodeUrl": "https://liteapp.weixin.qq.com/q/xxx",
  "sessionKey": "uuid-string",
  "message": "请用手机微信扫描二维码"
}
```

#### 等待扫码确认

```bash
POST /login/qr/wait
Content-Type: application/json

{"sessionKey": "uuid-string"}
```

```json
{
  "success": true,
  "botToken": "xxx@im.bot:060000xxx",
  "accountId": "xxx@im.bot",
  "baseUrl": "https://ilinkai.weixin.qq.com",
  "userId": "xxx@im.wechat",
  "message": "登录成功"
}
```

#### 设置 Token

```bash
POST /token
Content-Type: application/json

{"token":"bot_token_string","accountId":"xxx@im.bot","userId":"xxx@im.wechat"}
```

#### 启动消息监听

```bash
POST /start
Content-Type: application/json

{"accountId":"xxx@im.bot"}
```

```json
{"ok": true, "message": "bot starting"}
```

#### 停止消息监听

```bash
POST /stop
Content-Type: application/json

{"accountId":"xxx@im.bot"}
```

```json
{"ok": true}
```

#### 发送文本消息

```bash
POST /send/text
Content-Type: application/json

{
  "accountId": "xxx@im.bot",
  "to": "用户ID",
  "text": "消息内容",
  "contextToken": "可选，上下文令牌"
}
```

```json
{"messageId": "msg_id_string"}
```

#### 发送媒体消息

```bash
POST /send/media
Content-Type: application/json

{
  "accountId": "xxx@im.bot",
  "to": "用户ID",
  "filePath": "/path/to/file",
  "text": "可选，媒体说明",
  "contextToken": "可选，上下文令牌"
}
```

自动根据文件 MIME 类型路由：图片 → `sendImageMessage`，视频 → `sendVideoMessage`，其他 → `sendFileMessage`。

---

## CookClaw API

### 微信状态

```bash
GET /api/v1/weixin/status
```

```json
{
  "enabled": true,
  "started": false,
  "connected": false,
  "auth_expired": true,
  "last_error": "session timeout",
  "token": "configured",
  "account_count": 2,
  "connected_count": 1,
  "max_accounts": 10,
  "accounts": [{"accountId":"a@im.bot","connected":true}]
}
```

`started` 只表示监听任务是否在运行，`connected` 才表示当前会话通过服务端预检。
当 `auth_expired=true` 时，iLink 已返回 `errcode=-14`，需要重新扫码；不能仅根据
`token=configured` 判断在线。

### 微信 QR 码登录（一步式）

```bash
POST /api/v1/weixin/login/qr
```

### 微信 QR 码登录（两步式）

```bash
# 获取 QR 码
POST /api/v1/weixin/login/qr/start

# 等待确认
POST /api/v1/weixin/login/qr/wait
Content-Type: application/json
{"session_key": "uuid-string"}
```

### 微信 Webhook（内部端点）

```bash
POST /api/v1/weixin/webhook
```

此端点由 weixinbot 微服务调用，推送入站消息事件。

---

## Webhook 事件格式

### 消息事件

```json
{
  "event": "message",
  "timestamp": 1705312200000,
  "data": {
    "fromUserId": "xxx@im.wechat",
    "toUserId": "xxx@im.bot",
    "text": "你好，帮我搜索红烧肉食谱",
    "contextToken": "ctx_token_string",
    "accountId": "xxx@im.bot",
    "messageId": 12345,
    "sessionId": "session_id",
    "imagePath": "",
    "voicePath": "",
    "filePath": "",
    "videoPath": "",
    "voiceMediaType": "",
    "fileMediaType": ""
  }
}
```

### 消息处理流程

```
1. weixinbot 长轮询 getUpdates() 收到微信消息
2. 解析消息内容（文本/图片/语音/视频/文件）
3. 如有媒体，下载到本地 mediaDir
4. 构建 WeixinMessage + MessageContext
5. 通过 Webhook POST 到 CookClaw /api/v1/weixin/webhook
6. CookClaw WeixinAdapter.handle_webhook() 解析事件
7. 调用 _weixin_message_handler(event)
8. qqbot_chat(event.text, thread_id=`weixin:dm:<accountId>:<fromUserId>`) 获取 AI 回复
9. WeixinAdapter.send_message(from_user_id, response, account_id=event.account_id) 从原账号回复
10. weixinbot 调用 sendMessage() 发送到微信
```

---

## SDK 核心模块

### 文件结构

```
app/weixinbot/src/
├── index.ts           # WeixinBot 主类（门面模式）
├── types.ts           # 全部 TypeScript 类型定义
├── api.ts             # iLink HTTP API 通信层
├── auth.ts            # QR 码登录认证
├── cdn.ts             # CDN 上传/下载（AES-128-ECB 加密）
├── media.ts           # 媒体下载与 SILK→WAV 转码
├── messaging.ts       # 消息发送（文本/图片/视频/文件）
├── monitor.ts         # 长轮询消息监听循环
├── state.ts           # 状态持久化（accounts/syncBuffers/contextTokens）
├── markdown-filter.ts # StreamingMarkdownFilter（字符级状态机）
├── logger.ts          # 日志模块（console + file）
├── utils.ts           # 工具函数（randomId, MIME 映射等）
├── vendor.d.ts        # 第三方类型声明
└── server.ts          # HTTP 微服务入口
```

### 模块职责

| 模块 | 行为 |
|------|------|
| **api.ts** | `apiGetFetch()` / `apiPostFetch()` — 所有 iLink API 调用的基础层。自动附加 `base_info`、`bot_token`、`bot_agent` 参数。支持自定义超时（长轮询 35s，普通 15s）。 |
| **auth.ts** | `startQRLogin()` → 获取 QR 码；`waitForQRLogin()` → 长轮询等待扫码确认；`loginWithQRCode()` → 一步式登录。支持 `scaned_but_redirect` 重定向。 |
| **cdn.ts** | AES-128-ECB 加密/解密（微信 CDN 协议要求）。`uploadFileToWeixin()` / `uploadVideoToWeixin()` / `uploadFileAttachmentToWeixin()` — 先获取上传 URL，再 PUT 加密数据。 |
| **media.ts** | `downloadMediaFromItem()` — 从消息中下载媒体文件。自动检测 SILK 格式并转码为 WAV。 |
| **messaging.ts** | `sendTextMessage()` — 发送纯文本，自动通过 `StreamingMarkdownFilter` 过滤 Markdown。`sendImageMessage()` / `sendVideoMessage()` / `sendFileMessage()` — 上传 CDN 后发送。 |
| **monitor.ts** | `startMonitor()` — 长轮询 `getUpdates()` 循环。处理新消息、通知上下文 token 变化、同步 buffer 更新。支持 `AbortController` 优雅停止。 |
| **state.ts** | 持久化到 `~/.weixinbot/` 目录。存储：accounts（token/baseUrl/userId）、syncBuffers（长轮询游标）、contextTokens（会话上下文）。 |
| **markdown-filter.ts** | 字符级状态机，逐字符过滤 Markdown 格式（`**bold**` → `bold`），保留可读性。用于出站消息清洗。 |

---

## iLink 协议关键参数

### API 基础

| 参数 | 值 |
|------|---|
| Base URL | `https://ilinkai.weixin.qq.com` |
| 认证方式 | `bot_token` 查询参数 |
| 协议格式 | HTTP JSON |

### 关键接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `ilink/bot/get_bot_qrcode` | POST | 获取登录 QR 码 |
| `ilink/bot/get_qrcode_status` | GET | 轮询扫码状态（长轮询 35s） |
| `ilink/bot/getUpdates` | POST | 长轮询获取新消息 |
| `ilink/bot/sendMessage` | POST | 发送消息 |
| `ilink/bot/getUploadUrl` | POST | 获取 CDN 上传地址 |
| `ilink/bot/getConfig` | GET | 获取配置（含 CDN AES 密钥） |
| `ilink/bot/sendTyping` | POST | 发送打字指示 |
| `ilink/bot/notifyStart` | POST | 通知上线 |
| `ilink/bot/notifyStop` | POST | 通知下线 |

### CDN 加密

- 算法：AES-128-ECB
- 密钥来源：`getConfig()` 返回的 `cdn_aes_key`（Base64 编码）
- 上传流程：获取 URL → AES 加密文件数据 → PUT 到 CDN
- 下载流程：GET CDN URL → AES 解密响应体

### QR 登录状态码

| 状态 | 说明 |
|------|------|
| `wait` | 等待扫码 |
| `scaned` | 已扫码，等待确认 |
| `confirmed` | 已确认，登录成功 |
| `expired` | 二维码已过期 |
| `scaned_but_redirect` | 已扫码但需要重定向到其他服务器 |
| `binded_redirect` | 已绑定过，无需重复登录 |
| `need_verifycode` | 需要验证码 |
| `verify_code_blocked` | 验证码被阻止 |

---

## 环境变量参考

### weixinbot 微服务（通过 CookClaw 传递）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `3003` | HTTP API 端口 |
| `WEBHOOK_URL` | — | CookClaw Webhook 地址 |
| `WEBHOOK_SECRET` | — | Webhook 认证密钥 |
| `WEIXIN_TOKEN` | — | 微信 bot_token（也可通过 QR 登录获取） |
| `WEIXIN_STATE_DIR` | `~/.weixinbot/` | 状态存储目录 |
| `WEIXIN_MAX_ACCOUNTS` | `10` | 最大并存账号数，范围 1～50 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### CookClaw (.env)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WEIXIN_ENABLED` | `false` | 是否启用微信机器人 |
| `WEIXIN_SERVICE_URL` | `http://localhost:3003` | 微服务地址 |
| `WEIXIN_TOKEN` | — | 微信 bot_token（首次登录后自动保存） |
| `WEIXIN_WEBHOOK_SECRET` | — | Webhook 密钥 |
| `WEIXIN_MAX_ACCOUNTS` | `10` | 最大并存账号数，范围 1～50 |

---

## 状态持久化

CookClaw 和 Node 微服务分别保存多账号状态：

```
app/weixinbot/.wx_session.json       # Python 侧全部账号会话（v2）
~/.weixinbot/state/accounts/
├── <accountId>.json                 # 单账号 token/baseUrl/userId
├── <accountId>.sync.json            # 单账号长轮询游标
└── <accountId>.context-tokens.json  # 单账号会话上下文令牌
```

### `.wx_session.json` 格式

```json
{
  "version": 2,
  "accounts": {
    "xxx@im.bot": {
      "token": "xxx@im.bot:060000xxx",
      "base_url": "https://ilinkai.weixin.qq.com",
      "account_id": "xxx@im.bot",
      "user_id": "xxx@im.wechat",
      "saved_at": 1784646000.0
    }
  }
}
```

---

## 超时配置

| 操作 | 超时 | 说明 |
|------|------|------|
| 普通 API 调用 | 15s | `sendMessage`、`getConfig` 等 |
| 长轮询 API | 35s | `getUpdates`、`get_qrcode_status` |
| 配置获取 | 10s | `getConfig` |
| QR 登录（Python 端） | 300s（5 分钟） | `login_with_qr()`、`wait_qr_login()` |
| 普通操作（Python 端） | 30s | `send_message()`、`get_status()` 等 |

---

## 故障排除

### weixinbot 微服务无法启动

```bash
# 检查 Node.js 版本
node --version  # 需要 v20+

# 重新编译
cd app/weixinbot
npm run build

# 手动启动测试
node dist/server.js
```

### QR 码登录超时

QR 登录已配置 5 分钟超时。如果仍然超时：

```bash
# 使用两步式登录
curl -X POST http://localhost:8000/api/v1/weixin/login/qr/start
# 在浏览器打开返回的 qrcodeUrl 扫码
curl -X POST http://localhost:8000/api/v1/weixin/login/qr/wait \
  -H "Content-Type: application/json" \
  -d '{"session_key": "返回的sessionKey"}'
```

### CookClaw 无法连接微服务

```bash
# 检查微服务是否运行
curl http://localhost:3003/health

# 检查端口是否被占用
lsof -i :3003
```

### 消息发送失败

```bash
# 检查服务状态
curl http://localhost:3003/health
# started 应为 true，token 应为 "configured"

# 如果 token 未设置，重新登录
curl -X POST http://localhost:3003/login/qr

# 如果未启动监听
curl -X POST http://localhost:3003/start
```

### Token 丢失

CookClaw 扫码得到的全部账号 token 保存在 `app/weixinbot/.wx_session.json`；Node 侧每个账号的
凭证、同步游标和上下文状态保存在 `~/.weixinbot/state/accounts/`。部署替换时需要保留这两处状态；
任何一个账号的 token 文件丢失后，该账号需要重新扫码，不影响其他账号。

### Token 已过期

iLink 当前未公开固定 token 有效期，也没有 refresh token 接口，因此不要在代码中写死过期天数。
启动时以服务端返回为准：`errcode=-14` / `session timeout` 表示会话已过期。服务会立即停止监听、
将健康状态标记为 `auth_expired=true`，并要求通过 `/weixin/login` 重新扫码；不会继续把过期 token
误报为在线或静默等待一小时。

---

## 安全注意事项

| 项目 | 建议 |
|------|------|
| **Webhook Secret** | 使用强随机字符串，防止伪造消息推送 |
| **网络暴露** | 微服务默认绑定 localhost，不要直接暴露到公网 |
| **Token 存储** | `~/.weixinbot/` 目录权限应设为 700 |
| **HTTPS** | 生产环境使用反向代理加 TLS |
| **CDN 密钥** | AES 密钥由微信服务器动态下发，不本地存储 |
