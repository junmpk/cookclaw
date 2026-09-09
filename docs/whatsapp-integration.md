# WhatsApp 集成指南 — CookClaw

> 本文档介绍如何通过 Baileys 微服务将 WhatsApp 集成到 CookClaw 项目中。

---

## 架构概览

```
WhatsApp 用户
    ↕ (WhatsApp Web 协议)
app/whatsapp/service (Node.js / Baileys)
    ↕ (HTTP API + Webhook)
CookClaw (Python / FastAPI)
    ↕
AI Agent (食谱搜索、设备控制)
```

### 组件说明

| 组件 | 技术栈 | 职责 |
|------|--------|------|
| **app/whatsapp/service** | Node.js + Baileys + Express | WhatsApp 协议处理、QR 码登录、消息收发、媒体处理 |
| **CookClaw** | Python + FastAPI | AI Agent、食谱搜索、设备控制、Webhook 接收 |
| **WhatsAppAdapter** | Python + httpx | HTTP 客户端，调用 WhatsApp 微服务 API |

---

## 快速开始

### 前置条件

| 条件 | 要求 |
|------|------|
| **Node.js** | v20+ |
| **Python** | v3.12+ |
| **WhatsApp** | 可用的 WhatsApp 账号 |
| **手机** | 已安装 WhatsApp，用于扫码配对 |

### 第一步：启动 WhatsApp 微服务

```bash
cd app/whatsapp/service

# 安装依赖
npm install

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置 WEBHOOK_URL 和 API_TOKEN

# 启动服务（开发模式）
npm run dev

# 或构建后启动（生产模式）
npm run build
npm start
```

启动后会显示 QR 码：

```
████████████████████████████████
████████████████████████████████
████████████████████████████████

📱 请打开 WhatsApp → 设置 → 关联设备 → 扫描此 QR 码
```

### 第二步：配置 CookClaw

在 CookClaw 的 `.env` 文件中添加：

```bash
# WhatsApp 集成
WHATSAPP_ENABLED=true
WHATSAPP_SERVICE_URL=http://localhost:3001
WHATSAPP_API_TOKEN=your-api-token-here
WHATSAPP_WEBHOOK_SECRET=your-webhook-secret-here
```

### 第三步：启动 CookClaw

```bash
# 在项目根目录
uv run python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 短期会话记忆

WhatsApp 已接入与 QQ、微信共用的 Redis 短期会话服务。私聊按 WhatsApp JID 隔离；群聊按
“群组 JID + 成员 JID”隔离，避免不同成员共享偏好、搜索候选或摘要。默认保留最近
20 轮，超过轮数或 token 预算后压缩为摘要，默认 24 小时未活动自动过期。账号长期
画像按通道身份写入 PostgreSQL。

```bash
DATA_SERVICE_PROFILE=local
CONVERSATION_STORE=redis
PROFILE_STORE=postgres
CONVERSATION_TTL_SECONDS=86400
CONVERSATION_MAX_TURNS=20
CONVERSATION_KEEP_RECENT_TURNS=6
CONVERSATION_MAX_TOKENS=6000
```

本地开发需要先建立 SSH 隧道；服务器部署把 `DATA_SERVICE_PROFILE` 改为 `server`。
连接参数见 `.env.example`，切换与回滚记录见
`docs/database/CookClaw-P0用户与记忆存储方案.md`。

### 浏览器扫码登录

主服务启动后，直接在本机浏览器访问以下页面。页面会自动刷新二维码，并在扫码成功后
显示连接状态：

```text
http://127.0.0.1:8000/api/v1/whatsapp/login
```

二维码 PNG 接口为 `GET /api/v1/whatsapp/qr/image`，原始二维码数据接口仍保留为
`GET /api/v1/whatsapp/qr`，供程序调用。

---

## WhatsApp 微服务 API

### 基础 URL

```
http://localhost:3001
```

### 认证

所有 `/api/*` 端点需要 Bearer Token 认证：

```
Authorization: Bearer your-api-token-here
```

### 端点列表

#### 健康检查

```bash
GET /health
```

```json
{ "status": "ok", "service": "cookclaw-whatsapp" }
```

#### 服务状态

```bash
GET /api/status
```

```json
{
  "running": true,
  "connected": true,
  "connectionState": "connected",
  "phoneNumber": "+15551234567",
  "lastConnectedAt": "2025-01-15T10:30:00.000Z",
  "uptime": 3600,
  "webhookUrl": "http://localhost:8000/api/v1/whatsapp/webhook"
}
```

#### 获取 QR 码

```bash
GET /api/qr
```

```json
{
  "qr": "2@abc123...",
  "message": "请使用 WhatsApp 扫描此 QR 码"
}
```

#### 发送消息

```bash
POST /api/send
Content-Type: application/json

{
  "to": "+15551234567",
  "text": "Hello from CookClaw!"
}
```

发送媒体：

```json
{
  "to": "+15551234567",
  "text": "这是一张图片",
  "mediaUrl": "https://example.com/image.jpg"
}
```

发送到群组：

```json
{
  "to": "123456789-1234567890@g.us",
  "text": "群组消息"
}
```

#### 发送已读回执

```bash
POST /api/read

{
  "jid": "15551234567@s.whatsapp.net",
  "messageIds": ["message-id-1", "message-id-2"]
}
```

#### 发送 emoji 反应

```bash
POST /api/react

{
  "jid": "15551234567@s.whatsapp.net",
  "messageId": "message-id-1",
  "emoji": "👍"
}
```

#### 配置 Webhook

```bash
POST /api/webhook

{
  "url": "http://localhost:8000/api/v1/whatsapp/webhook",
  "secret": "your-webhook-secret"
}
```

---

## CookClaw API

### WhatsApp 状态

```bash
GET /api/v1/whatsapp/status
```

### WhatsApp QR 码

```bash
GET /api/v1/whatsapp/qr
```

### WhatsApp Webhook（内部端点）

```bash
POST /api/v1/whatsapp/webhook
```

此端点由 WhatsApp 微服务调用，推送入站消息事件。

---

## Webhook 事件格式

### 消息事件

```json
{
  "event": "message",
  "timestamp": 1705312200000,
  "data": {
    "messageId": "3EB0xxxxx",
    "from": "15551234567@s.whatsapp.net",
    "fromNumber": "+15551234567",
    "chatType": "dm",
    "text": "你好，帮我搜索红烧肉食谱",
    "mediaUrls": [],
    "mediaTypes": [],
    "quotedMessageId": null,
    "groupJid": null,
    "participant": null
  }
}
```

### 群组消息事件

```json
{
  "event": "message",
  "timestamp": 1705312200000,
  "data": {
    "messageId": "3EB0xxxxx",
    "from": "123456789-1234567890@g.us",
    "fromNumber": "+15551234567",
    "chatType": "group",
    "text": "@Bot 搜索番茄炒蛋",
    "groupJid": "123456789-1234567890@g.us",
    "participant": "15551234567@s.whatsapp.net"
  }
}
```

### 连接状态事件

```json
{
  "event": "connection.update",
  "timestamp": 1705312200000,
  "data": {
    "connectionState": "connected"
  }
}
```

---

## 目标地址格式

| 格式 | 说明 | 示例 |
|------|------|------|
| E.164 手机号 | 私聊 | `+15551234567` |
| WhatsApp JID | 私聊（内部） | `15551234567@s.whatsapp.net` |
| 群组 JID | 群聊 | `123456789-1234567890@g.us` |

---

## 环境变量参考

### WhatsApp 微服务 (.env)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `3001` | HTTP API 端口 |
| `WEBHOOK_URL` | — | CookClaw Webhook 地址 |
| `WEBHOOK_SECRET` | — | Webhook 认证密钥 |
| `API_TOKEN` | — | HTTP API 认证令牌 |
| `LOG_LEVEL` | `info` | 日志级别 |
| `AUTH_FOLDER` | `./auth` | 凭据存储路径 |

### CookClaw (.env)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WHATSAPP_ENABLED` | `false` | 是否启用 WhatsApp |
| `WHATSAPP_SERVICE_URL` | `http://localhost:3001` | 微服务地址 |
| `WHATSAPP_API_TOKEN` | — | 微服务 API 令牌 |
| `WHATSAPP_WEBHOOK_SECRET` | — | Webhook 密钥 |

---

## 故障排除

### WhatsApp 微服务无法启动

```bash
# 检查 Node.js 版本
node --version  # 需要 v20+

# 重新安装依赖
cd app/whatsapp/service
rm -rf node_modules package-lock.json
npm install
```

### QR 码不显示

```bash
# 检查网络连接（中国大陆需代理）
export HTTPS_PROXY=http://your-proxy:8080

# 重启服务
npm run dev
```

### CookClaw 无法连接微服务

```bash
# 检查微服务是否运行
curl http://localhost:3001/health

# 检查 CookClaw 配置
echo $WHATSAPP_ENABLED
echo $WHATSAPP_SERVICE_URL
```

### 消息发送失败

```bash
# 检查连接状态
curl http://localhost:3001/api/status -H "Authorization: Bearer your-token"

# 如果状态为 logged_out，需要重新扫码
curl http://localhost:3001/api/qr -H "Authorization: Bearer your-token"
```

### 凭据丢失

凭据保存在 `app/whatsapp/service/auth/` 目录中。如果凭据丢失，需要重新扫码：

```bash
# 删除旧凭据
rm -rf app/whatsapp/service/auth/

# 重启服务获取新 QR 码
npm run dev
```

---

## 安全注意事项

| 项目 | 建议 |
|------|------|
| **API Token** | 使用强随机字符串 |
| **Webhook Secret** | 使用独立的密钥 |
| **网络暴露** | 微服务默认绑定 localhost，不要直接暴露到公网 |
| **凭据备份** | 定期备份 `app/whatsapp/service/auth/` 目录 |
| **HTTPS** | 生产环境使用反向代理加 TLS |
