/**
 * 微信机器人 HTTP 服务
 *
 * 作为独立微服务运行，暴露 REST API 供 Python 后端调用。
 * 收到微信消息时，通过 Webhook 回调 Python 后端。
 *
 * 端口：默认 3003
 * 环境变量：
 *   PORT         - 服务端口（默认 3003）
 *   WEBHOOK_URL  - Python 后端 Webhook 地址
 *   WEBHOOK_SECRET - Webhook 密钥
 *   WEIXIN_TOKEN - 微信 bot_token（可选，也可通过 API 登录获取）
 *   WEIXIN_STATE_DIR - 状态存储目录
 *   WEIXIN_MAX_ACCOUNTS - 最大并存账号数（默认 10，范围 1-50）
 *   LOG_LEVEL    - 日志级别
 */
import { createServer } from "node:http";

import { WeixinBot } from "./index.js";
import { setLogLevel, logger } from "./logger.js";
import { deleteAccount, listAccountIds, loadAccount, setStateDir } from "./state.js";
import { fingerprint, redactUrl, safeError } from "./utils.js";
import type { WeixinMessage, MessageContext } from "./types.js";

// ---------------------------------------------------------------------------
// 配置
// ---------------------------------------------------------------------------

const PORT = parseInt(process.env.PORT || "3003", 10);
const HOST = process.env.HOST || "127.0.0.1";
const WEBHOOK_URL = process.env.WEBHOOK_URL || "";
const WEBHOOK_SECRET = process.env.WEBHOOK_SECRET || "";
const WEIXIN_STATE_DIR = process.env.WEIXIN_STATE_DIR || undefined;
const MAX_ACCOUNTS = Math.min(50, Math.max(1, parseInt(process.env.WEIXIN_MAX_ACCOUNTS || "10", 10) || 10));

// ---------------------------------------------------------------------------
// Bot 多账号运行时
// ---------------------------------------------------------------------------

type BotRuntime = {
  accountId: string;
  userId: string;
  bot: WeixinBot;
  started: boolean;
  connected: boolean;
  authExpired: boolean;
  lastError: string;
  monitorTask: Promise<void> | null;
};

type SessionInput = {
  token: string;
  baseUrl?: string;
  accountId?: string;
  userId?: string;
};

const bots = new Map<string, BotRuntime>();

if (WEIXIN_STATE_DIR) setStateDir(WEIXIN_STATE_DIR);

function isAuthExpiredError(err: unknown): boolean {
  if (err && typeof err === "object" && "code" in err && (err as { code?: number }).code === -14) {
    return true;
  }
  return String(err).includes("WeixinAuthExpiredError") || String(err).includes("session timeout");
}

function markBotError(runtime: BotRuntime, err: unknown): void {
  runtime.started = false;
  runtime.connected = false;
  runtime.authExpired = isAuthExpiredError(err);
  runtime.lastError = runtime.authExpired ? "session timeout" : safeError(err);
}

function clearBotError(runtime: BotRuntime): void {
  runtime.authExpired = false;
  runtime.lastError = "";
}

function createRuntime(session: SessionInput): BotRuntime {
  const accountId = session.accountId?.trim() || "default";
  const runtime = {} as BotRuntime;
  const bot = new WeixinBot({
    baseUrl: session.baseUrl || process.env.WEIXIN_BASE_URL || undefined,
    token: session.token,
    stateDir: WEIXIN_STATE_DIR,
    logLevel: process.env.LOG_LEVEL || "INFO",
    onMessage: async (message: WeixinMessage, ctx: MessageContext) => {
      await forwardToWebhook(message, ctx);
    },
  });
  Object.assign(runtime, {
    accountId,
    userId: session.userId || "",
    bot,
    started: false,
    connected: false,
    authExpired: false,
    lastError: "",
    monitorTask: null,
  });
  bot.setSession(session.token, {
    accountId,
    baseUrl: session.baseUrl,
    userId: session.userId,
  });
  return runtime;
}

async function registerSession(session: SessionInput): Promise<BotRuntime> {
  const accountId = session.accountId?.trim() || "default";
  const existing = bots.get(accountId);
  if (!existing && bots.size >= MAX_ACCOUNTS) {
    throw new Error(`account_limit_reached:${MAX_ACCOUNTS}`);
  }
  if (existing) await stopRuntime(existing);
  const runtime = createRuntime({ ...session, accountId });
  bots.set(accountId, runtime);
  return runtime;
}

function runtimeStatus(runtime: BotRuntime): Record<string, unknown> {
  let apiHost = "";
  try {
    apiHost = new URL(runtime.bot.getBaseUrl()).host;
  } catch {
    apiHost = runtime.bot.getBaseUrl();
  }
  return {
    accountId: runtime.accountId,
    started: runtime.started,
    connected: runtime.connected && !runtime.authExpired,
    authExpired: runtime.authExpired,
    lastError: runtime.lastError,
    token: runtime.bot.getToken() ? "configured" : "not set",
    apiHost,
  };
}

function healthSnapshot(): Record<string, unknown> {
  const accounts = [...bots.values()].map(runtimeStatus);
  const connectedCount = accounts.filter((item) => item.connected).length;
  const startedCount = accounts.filter((item) => item.started).length;
  const authExpiredCount = accounts.filter((item) => item.authExpired).length;
  return {
    status: "ok",
    started: startedCount > 0,
    connected: connectedCount > 0,
    authExpired: accounts.length > 0 && authExpiredCount === accounts.length,
    lastError: accounts.length === 1 ? accounts[0].lastError : "",
    token: accounts.length > 0 ? "configured" : "not set",
    accountId: accounts.length === 1 ? accounts[0].accountId : "",
    apiHost: accounts.length === 1 ? accounts[0].apiHost : "",
    accountCount: accounts.length,
    startedCount,
    connectedCount,
    authExpiredCount,
    maxAccounts: MAX_ACCOUNTS,
    accounts,
  };
}

function resolveRuntime(accountId?: string): BotRuntime | null {
  if (accountId?.trim()) return bots.get(accountId.trim()) || null;
  if (bots.size === 1) return bots.values().next().value || null;
  return null;
}

async function startRuntime(runtime: BotRuntime): Promise<void> {
  if (runtime.started) return;
  try {
    await runtime.bot.verifySession();
    clearBotError(runtime);
  } catch (err) {
    markBotError(runtime, err);
    logger.error(`[${fingerprint(runtime.accountId)}] Bot session check failed: ${safeError(err)}`);
    throw err;
  }

  runtime.connected = false;
  runtime.started = true;
  runtime.monitorTask = runtime.bot.start(runtime.accountId, {
    notify: false,
    onConnected: () => {
      runtime.connected = true;
      clearBotError(runtime);
      logger.info(
        `Bot connected: account=${fingerprint(runtime.accountId)} apiHost=${new URL(runtime.bot.getBaseUrl()).host}`
      );
    },
  }).catch((err: Error) => {
    logger.error(`[${fingerprint(runtime.accountId)}] Bot start error: ${safeError(err)}`);
    markBotError(runtime, err);
  });
}

async function stopRuntime(runtime: BotRuntime): Promise<void> {
  if (!runtime.started && !runtime.monitorTask) return;
  await runtime.bot.stop();
  runtime.started = false;
  runtime.connected = false;
  runtime.monitorTask = null;
}

function restoreSavedAccounts(): void {
  const candidates: SessionInput[] = [];
  const envToken = process.env.WEIXIN_TOKEN || "";
  if (envToken) {
    candidates.push({
      token: envToken,
      baseUrl: process.env.WEIXIN_BASE_URL,
      accountId: process.env.WEIXIN_ACCOUNT_ID || "default",
      userId: process.env.WEIXIN_USER_ID,
    });
  }
  for (const accountId of listAccountIds()) {
    const saved = loadAccount(accountId);
    if (saved?.token) {
      candidates.push({ token: saved.token, baseUrl: saved.baseUrl, accountId, userId: saved.userId });
    }
  }

  for (const candidate of candidates) {
    const accountId = candidate.accountId || "default";
    if (bots.has(accountId)) continue;
    if (bots.size >= MAX_ACCOUNTS) {
      logger.warn(`微信已保存账号超过上限 ${MAX_ACCOUNTS}，忽略账号 ${fingerprint(accountId)}`);
      continue;
    }
    bots.set(accountId, createRuntime(candidate));
  }
  logger.info(`Weixin accounts restored: ${bots.size}/${MAX_ACCOUNTS}`);
}

// ---------------------------------------------------------------------------
// Webhook 转发
// ---------------------------------------------------------------------------

async function forwardToWebhook(message: WeixinMessage, ctx: MessageContext): Promise<void> {
  if (!WEBHOOK_URL) {
    logger.warn("WEBHOOK_URL not configured, dropping message");
    return;
  }

  const payload = {
    event: "message",
    timestamp: Date.now(),
    data: {
      fromUserId: ctx.fromUserId,
      toUserId: ctx.toUserId,
      text: ctx.text,
      contextToken: ctx.contextToken,
      accountId: ctx.accountId,
      imagePath: ctx.imagePath || undefined,
      voicePath: ctx.voicePath || undefined,
      filePath: ctx.filePath || undefined,
      videoPath: ctx.videoPath || undefined,
      voiceMediaType: ctx.voiceMediaType || undefined,
      fileMediaType: ctx.fileMediaType || undefined,
      messageId: message.message_id,
      sessionId: message.session_id,
    },
  };

  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (WEBHOOK_SECRET) {
      headers["X-Webhook-Secret"] = WEBHOOK_SECRET;
    }

    const res = await fetch(WEBHOOK_URL, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      logger.error(`Webhook forward failed: ${res.status}`);
    }
  } catch (err) {
    logger.error(`Webhook forward error: ${safeError(err)}`);
  }
}

// ---------------------------------------------------------------------------
// HTTP 路由处理
// ---------------------------------------------------------------------------

async function handleRequest(req: any, res: any): Promise<void> {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const method = req.method;

  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (method === "OPTIONS") { res.writeHead(204); res.end(); return; }

  try {
    // ─── 健康检查 ───
    if (url.pathname === "/health" && method === "GET") {
      return jsonRes(res, 200, healthSnapshot());
    }

    if (url.pathname === "/accounts" && method === "GET") {
      return jsonRes(res, 200, healthSnapshot());
    }

    // ─── QR 码登录 ───
    if (url.pathname === "/login/qr" && method === "POST") {
      if (bots.size >= MAX_ACCOUNTS) {
        return jsonRes(res, 409, { error: "account_limit_reached", maxAccounts: MAX_ACCOUNTS });
      }
      const b = new WeixinBot({
        stateDir: WEIXIN_STATE_DIR,
        logLevel: process.env.LOG_LEVEL || "INFO",
        onMessage: forwardToWebhook,
      });
      const result = await b.loginWithQRCode();
      if (result.success && result.botToken) {
        const runtime = await registerSession({
          token: result.botToken,
          accountId: result.accountId,
          baseUrl: result.baseUrl,
          userId: result.userId,
        });
        await startRuntime(runtime);
      }
      return jsonRes(res, 200, result);
    }

    // ─── 获取 QR 码（不等待确认） ───
    if (url.pathname === "/login/qr/start" && method === "POST") {
      if (bots.size >= MAX_ACCOUNTS) {
        return jsonRes(res, 409, { error: "account_limit_reached", maxAccounts: MAX_ACCOUNTS });
      }
      const { startQRLogin } = await import("./auth.js");
      const result = await startQRLogin();
      return jsonRes(res, 200, result);
    }

    // ─── 等待 QR 码确认 ───
    if (url.pathname === "/login/qr/wait" && method === "POST") {
      const body = await readBody(req);
      const { sessionKey } = JSON.parse(body || "{}");
      if (!sessionKey) return jsonRes(res, 400, { error: "sessionKey required" });

      const { waitForQRLogin } = await import("./auth.js");
      const result = await waitForQRLogin(sessionKey);
      return jsonRes(res, 200, result);
    }

    // ─── 设置 Token ───
    if (url.pathname === "/token" && method === "POST") {
      const body = await readBody(req);
      const { token, baseUrl, accountId, userId } = JSON.parse(body || "{}");
      if (!token) return jsonRes(res, 400, { error: "token required" });
      if (!accountId) return jsonRes(res, 400, { error: "accountId required" });

      try {
        const runtime = await registerSession({ token, baseUrl, accountId, userId });
        return jsonRes(res, 200, { ok: true, account: runtimeStatus(runtime) });
      } catch (err) {
        if (String(err).includes("account_limit_reached")) {
          return jsonRes(res, 409, { error: "account_limit_reached", maxAccounts: MAX_ACCOUNTS });
        }
        throw err;
      }
    }

    // ─── 启动监听 ───
    if (url.pathname === "/start" && method === "POST") {
      const body = await readBody(req);
      const { accountId } = JSON.parse(body || "{}");
      const targets = accountId ? [bots.get(accountId)].filter(Boolean) as BotRuntime[] : [...bots.values()];
      if (!targets.length) return jsonRes(res, 404, { error: accountId ? "account_not_found" : "no_accounts" });

      const results = await Promise.allSettled(targets.map(startRuntime));
      const failed = results
        .map((result, index) => ({ result, runtime: targets[index] }))
        .filter((item) => item.result.status === "rejected");
      if (accountId && failed.length) {
        const runtime = failed[0].runtime;
        return jsonRes(res, runtime.authExpired ? 401 : 502, {
          error: runtime.authExpired ? "auth_expired" : "start_failed",
          code: runtime.authExpired ? -14 : undefined,
          accountId: runtime.accountId,
          message: runtime.lastError,
        });
      }
      return jsonRes(res, 200, {
        ok: failed.length === 0,
        message: failed.length ? "some accounts failed to start" : "bot accounts starting",
        ...healthSnapshot(),
      });
    }

    // ─── 停止监听 ───
    if (url.pathname === "/stop" && method === "POST") {
      const body = await readBody(req);
      const { accountId } = JSON.parse(body || "{}");
      const targets = accountId ? [bots.get(accountId)].filter(Boolean) as BotRuntime[] : [...bots.values()];
      if (!targets.length) return jsonRes(res, 404, { error: accountId ? "account_not_found" : "no_accounts" });
      await Promise.allSettled(targets.map(stopRuntime));
      return jsonRes(res, 200, { ok: true, ...healthSnapshot() });
    }

    if (url.pathname === "/accounts/remove" && method === "POST") {
      const body = await readBody(req);
      const { accountId } = JSON.parse(body || "{}");
      if (!accountId) return jsonRes(res, 400, { error: "accountId required" });
      const runtime = bots.get(accountId);
      if (!runtime) return jsonRes(res, 404, { error: "account_not_found" });
      await stopRuntime(runtime);
      bots.delete(accountId);
      deleteAccount(accountId);
      return jsonRes(res, 200, { ok: true, ...healthSnapshot() });
    }

    // ─── 发送文本消息 ───
    if (url.pathname === "/send/text" && method === "POST") {
      const body = await readBody(req);
      const { accountId, to, text, contextToken } = JSON.parse(body || "{}");
      if (!to || !text) return jsonRes(res, 400, { error: "to and text required" });
      const runtime = resolveRuntime(accountId);
      if (!runtime) {
        return jsonRes(res, accountId ? 404 : 400, {
          error: accountId ? "account_not_found" : "accountId required when multiple accounts are registered",
        });
      }

      const result = await runtime.bot.sendText(to, text, contextToken);
      return jsonRes(res, 200, result);
    }

    // ─── 发送媒体消息 ───
    if (url.pathname === "/send/media" && method === "POST") {
      const body = await readBody(req);
      const { accountId, to, filePath, text, contextToken } = JSON.parse(body || "{}");
      if (!to || !filePath) return jsonRes(res, 400, { error: "to and filePath required" });
      const runtime = resolveRuntime(accountId);
      if (!runtime) {
        return jsonRes(res, accountId ? 404 : 400, {
          error: accountId ? "account_not_found" : "accountId required when multiple accounts are registered",
        });
      }

      const result = await runtime.bot.sendMedia(filePath, to, text, contextToken);
      return jsonRes(res, 200, result);
    }

    // ─── 404 ───
    return jsonRes(res, 404, { error: "not found" });
  } catch (err) {
    logger.error(`Request error: ${safeError(err)}`);
    return jsonRes(res, 500, { error: "internal_error" });
  }
}

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

function jsonRes(res: any, status: number, body: any): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

function readBody(req: any): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

// ---------------------------------------------------------------------------
// 启动服务
// ---------------------------------------------------------------------------

restoreSavedAccounts();

const server = createServer(handleRequest);

server.listen(PORT, HOST, () => {
  logger.info(`WeixinBot HTTP service listening on ${HOST}:${PORT}`);
  logger.info(`WeixinBot account capacity: ${bots.size}/${MAX_ACCOUNTS}`);
  if (WEBHOOK_URL) {
    logger.info(`Webhook URL: ${redactUrl(WEBHOOK_URL)}`);
  }
});
