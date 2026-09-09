/**
 * WeixinBot SDK — 独立微信机器人 SDK
 *
 * 从 @tencent-weixin/openclaw-weixin 提取核心逻辑，
 * 去除 OpenClaw 依赖，可直接集成到业务系统中。
 *
 * @example
 * ```typescript
 * import { WeixinBot } from "./src/index.js";
 *
 * const bot = new WeixinBot({
 *   token: "your-bot-token",
 *   onMessage: async (msg, ctx) => {
 *     console.log(`收到消息: ${ctx.text}`);
 *     await ctx.replyText("收到！");
 *   },
 * });
 *
 * // 启动监听
 * await bot.start();
 *
 * // 或先登录
 * const result = await bot.loginWithQRCode();
 * ```
 */
import path from "node:path";
import os from "node:os";

import { startMonitor } from "./monitor.js";
import { loginWithQRCode as doLoginWithQRCode } from "./auth.js";
import { notifyStart, notifyStop } from "./api.js";
import { setStateDir, getStateDir, saveAccount, loadAccount, restoreContextTokens } from "./state.js";
import { setLogLevel, createLogger } from "./logger.js";
import { sendTextMessage, sendMediaFile } from "./messaging.js";
import { getContextToken } from "./state.js";

import type { WeixinBotConfig, OnMessageCallback, WeixinMessage, MessageContext } from "./types.js";

export type { WeixinBotConfig, OnMessageCallback, WeixinMessage, MessageContext } from "./types.js";

const DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com";
const DEFAULT_CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c";
const SESSION_EXPIRED_ERRCODE = -14;

type StartOptions = {
  notify?: boolean;
  onConnected?: () => void;
};

function sessionExpiredError(message = "Weixin session expired"): Error & { code: number } {
  const error = new Error(message) as Error & { code: number };
  error.name = "WeixinAuthExpiredError";
  error.code = SESSION_EXPIRED_ERRCODE;
  return error;
}

function assertSessionActive(resp: { ret?: number; errcode?: number; errmsg?: string }): void {
  const code = resp.errcode ?? resp.ret ?? 0;
  if (code === SESSION_EXPIRED_ERRCODE) {
    throw sessionExpiredError(resp.errmsg || "session timeout");
  }
  if (code !== 0) {
    throw new Error(`Weixin session check failed: code=${code} ${resp.errmsg || ""}`.trim());
  }
}

/**
 * 微信机器人 SDK 主类
 *
 * 封装了 QR 码登录、消息监听、消息发送等完整功能。
 */
export class WeixinBot {
  private config: {
    baseUrl: string;
    cdnBaseUrl: string;
    token: string;
    botAgent: string;
    longPollTimeoutMs: number;
    stateDir: string;
    mediaDir: string;
    logLevel: string;
    onMessage: OnMessageCallback;
  };

  private abortController: AbortController | null = null;

  constructor(config: WeixinBotConfig) {
    const stateDir = config.stateDir ?? path.join(os.homedir(), ".weixinbot");
    const mediaDir = config.mediaDir ?? path.join(stateDir, "media");

    this.config = {
      baseUrl: config.baseUrl ?? DEFAULT_BASE_URL,
      cdnBaseUrl: config.cdnBaseUrl ?? DEFAULT_CDN_BASE_URL,
      token: config.token ?? "",
      stateDir,
      mediaDir,
      logLevel: config.logLevel ?? "INFO",
      botAgent: config.botAgent ?? "WeixinBot",
      longPollTimeoutMs: config.longPollTimeoutMs ?? 35_000,
      onMessage: config.onMessage,
    };

    // 初始化状态目录
    setStateDir(stateDir);

    // 设置日志级别
    setLogLevel(this.config.logLevel);
  }

  // -------------------------------------------------------------------------
  // 登录
  // -------------------------------------------------------------------------

  /** QR 码登录 */
  async loginWithQRCode(opts?: {
    botType?: string;
    onQRCode?: (qrcodeUrl: string) => void;
  }): Promise<{
    success: boolean;
    botToken?: string;
    accountId?: string;
    baseUrl?: string;
    userId?: string;
    message: string;
  }> {
    const result = await doLoginWithQRCode(opts);
    if (result.success && result.botToken) {
      const accountId = result.accountId ?? "default";
      this.setSession(result.botToken, {
        accountId,
        baseUrl: result.baseUrl,
        userId: result.userId,
      });
    }
    return result;
  }

  // -------------------------------------------------------------------------
  // 消息监听
  // -------------------------------------------------------------------------

  /** 启动监听前验证服务端会话，避免 HTTP 200 + errcode=-14 被误报为在线。 */
  async verifySession(): Promise<void> {
    if (!this.config.token) {
      throw new Error("Weixin token not set, login first");
    }
    const resp = await notifyStart({
      baseUrl: this.config.baseUrl,
      token: this.config.token,
      botAgent: this.config.botAgent,
    });
    assertSessionActive(resp);
  }

  /** 启动消息监听（长轮询） */
  async start(accountId?: string, options?: StartOptions): Promise<void> {
    const id = accountId ?? "default";

    // 恢复 context tokens
    restoreContextTokens(id);

    // 通知上线
    if (options?.notify !== false) {
      await this.verifySession();
    }

    this.abortController = new AbortController();

    await startMonitor({
      baseUrl: this.config.baseUrl,
      cdnBaseUrl: this.config.cdnBaseUrl,
      token: this.config.token,
      accountId: id,
      botAgent: this.config.botAgent,
      longPollTimeoutMs: this.config.longPollTimeoutMs,
      mediaDir: this.config.mediaDir,
      abortSignal: this.abortController.signal,
      onConnected: options?.onConnected,
      onMessage: this.config.onMessage,
    });
  }

  /** 停止消息监听 */
  async stop(): Promise<void> {
    this.abortController?.abort();
    this.abortController = null;

    try {
      await notifyStop({
        baseUrl: this.config.baseUrl,
        token: this.config.token,
        botAgent: this.config.botAgent,
      });
    } catch (err) {
      console.warn(`notifyStop failed: ${String(err)}`);
    }
  }

  // -------------------------------------------------------------------------
  // 主动发消息
  // -------------------------------------------------------------------------

  /** 发送文本消息 */
  async sendText(to: string, text: string, contextToken?: string): Promise<{ messageId: string }> {
    return sendTextMessage({
      to,
      text,
      opts: {
        baseUrl: this.config.baseUrl,
        token: this.config.token,
        contextToken,
        botAgent: this.config.botAgent,
      },
    });
  }

  /** 发送媒体文件（自动判断类型） */
  async sendMedia(
    filePath: string,
    to: string,
    text?: string,
    contextToken?: string,
  ): Promise<{ messageId: string }> {
    return sendMediaFile({
      filePath,
      to,
      text: text ?? "",
      opts: {
        baseUrl: this.config.baseUrl,
        token: this.config.token,
        contextToken,
        botAgent: this.config.botAgent,
      },
      cdnBaseUrl: this.config.cdnBaseUrl,
    });
  }

  // -------------------------------------------------------------------------
  // 配置
  // -------------------------------------------------------------------------

  /** 更新 token（登录后使用） */
  setToken(token: string): void {
    this.setSession(token);
  }

  /** 原子更新扫码会话，确保 token 与该账号实际分配的 API 网关一致。 */
  setSession(
    token: string,
    opts?: { baseUrl?: string; accountId?: string; userId?: string },
  ): void {
    this.config.token = token;
    if (opts?.baseUrl?.trim()) {
      this.config.baseUrl = opts.baseUrl.trim();
    }
    if (opts?.accountId) {
      saveAccount(opts.accountId, {
        token,
        baseUrl: this.config.baseUrl,
        userId: opts.userId,
      });
    }
  }

  /** 获取当前 token */
  getToken(): string {
    return this.config.token;
  }

  getBaseUrl(): string {
    return this.config.baseUrl;
  }
}

// -------------------------------------------------------------------------
// 便捷导出：直接使用 API 函数
// -------------------------------------------------------------------------

export { getUpdates, sendMessage, getConfig, sendTyping, notifyStart, notifyStop } from "./api.js";
export { sendTextMessage, sendMediaFile, sendImageMessage, sendVideoMessage, sendFileMessage } from "./messaging.js";
export { downloadAndDecryptBuffer, uploadFileToWeixin, uploadVideoToWeixin } from "./cdn.js";
export { loginWithQRCode, startQRLogin, waitForQRLogin, displayQRCode } from "./auth.js";
export { StreamingMarkdownFilter } from "./markdown-filter.js";
export { setLogLevel } from "./logger.js";
