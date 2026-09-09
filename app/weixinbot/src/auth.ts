/**
 * QR 码登录认证
 */
import { randomUUID } from "node:crypto";

import { apiGetFetch, apiPostFetch } from "./api.js";
import { logger } from "./logger.js";
import { redactUrl, safeError } from "./utils.js";

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

const FIXED_BASE_URL = "https://ilinkai.weixin.qq.com";
const ACTIVE_LOGIN_TTL_MS = 5 * 60_000;
const QR_LONG_POLL_TIMEOUT_MS = 35_000;
const DEFAULT_ILINK_BOT_TYPE = "3";

// ---------------------------------------------------------------------------
// 类型
// ---------------------------------------------------------------------------

type ActiveLogin = {
  sessionKey: string;
  id: string;
  qrcode: string;
  qrcodeUrl: string;
  startedAt: number;
  botToken?: string;
  status?: "wait" | "scaned" | "confirmed" | "expired" | "scaned_but_redirect" | "need_verifycode" | "verify_code_blocked" | "binded_redirect";
  error?: string;
  currentApiBaseUrl?: string;
  pendingVerifyCode?: string;
};

interface QRCodeResponse {
  qrcode: string;
  qrcode_img_content: string;
}

interface StatusResponse {
  status: "wait" | "scaned" | "confirmed" | "expired" | "scaned_but_redirect" | "need_verifycode" | "verify_code_blocked" | "binded_redirect";
  bot_token?: string;
  ilink_bot_id?: string;
  baseurl?: string;
  ilink_user_id?: string;
  redirect_host?: string;
}

const activeLogins = new Map<string, ActiveLogin>();

// ---------------------------------------------------------------------------
// QR 码登录流程
// ---------------------------------------------------------------------------

function isLoginFresh(login: ActiveLogin): boolean {
  return Date.now() - login.startedAt < ACTIVE_LOGIN_TTL_MS;
}

function purgeExpiredLogins(): void {
  for (const [id, login] of activeLogins) {
    if (!isLoginFresh(login)) activeLogins.delete(id);
  }
}

async function fetchQRCode(apiBaseUrl: string, botType: string): Promise<QRCodeResponse> {
  logger.info(`Fetching QR code from: ${redactUrl(apiBaseUrl)} bot_type=${botType}`);
  const rawText = await apiPostFetch({
    baseUrl: apiBaseUrl,
    endpoint: `ilink/bot/get_bot_qrcode?bot_type=${encodeURIComponent(botType)}`,
    body: JSON.stringify({ local_token_list: [] }),
    label: "fetchQRCode",
  });
  return JSON.parse(rawText) as QRCodeResponse;
}

async function pollQRStatus(apiBaseUrl: string, qrcode: string, verifyCode?: string): Promise<StatusResponse> {
  try {
    let endpoint = `ilink/bot/get_qrcode_status?qrcode=${encodeURIComponent(qrcode)}`;
    if (verifyCode) {
      endpoint += `&verify_code=${encodeURIComponent(verifyCode)}`;
    }
    const rawText = await apiGetFetch({
      baseUrl: apiBaseUrl,
      endpoint,
      timeoutMs: QR_LONG_POLL_TIMEOUT_MS,
      label: "pollQRStatus",
    });
    return JSON.parse(rawText) as StatusResponse;
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      return { status: "wait" };
    }
    logger.warn(`pollQRStatus: network error, will retry: ${safeError(err)}`);
    return { status: "wait" };
  }
}

/** 在终端展示二维码 */
export async function displayQRCode(qrcodeUrl: string): Promise<void> {
  try {
    const qrterm = await import("qrcode-terminal");
    qrterm.default.generate(qrcodeUrl, { small: true });
  } catch {
    // qrcode-terminal 不可用，只显示链接
  }
  process.stdout.write(`\n若二维码无法显示，请访问以下链接：\n${qrcodeUrl}\n`);
}

// ---------------------------------------------------------------------------
// 公开 API
// ---------------------------------------------------------------------------

export interface QRLoginResult {
  success: boolean;
  botToken?: string;
  accountId?: string;
  baseUrl?: string;
  userId?: string;
  message: string;
}

/** 发起 QR 码登录（获取二维码） */
export async function startQRLogin(opts?: {
  botType?: string;
}): Promise<{ qrcodeUrl: string; sessionKey: string; message: string }> {
  const sessionKey = randomUUID();
  purgeExpiredLogins();

  const botType = opts?.botType || DEFAULT_ILINK_BOT_TYPE;
  const qrResp = await fetchQRCode(FIXED_BASE_URL, botType);

  const login: ActiveLogin = {
    sessionKey,
    id: sessionKey,
    qrcode: qrResp.qrcode,
    qrcodeUrl: qrResp.qrcode_img_content,
    startedAt: Date.now(),
    currentApiBaseUrl: FIXED_BASE_URL,
  };
  activeLogins.set(sessionKey, login);

  return {
    qrcodeUrl: qrResp.qrcode_img_content,
    sessionKey,
    message: "请用手机微信扫描二维码",
  };
}

/** 等待 QR 码扫描确认（阻塞直到登录成功或超时） */
export async function waitForQRLogin(sessionKey: string): Promise<QRLoginResult> {
  const login = activeLogins.get(sessionKey);
  if (!login) {
    return { success: false, message: "无效的 sessionKey" };
  }

  let apiBaseUrl = login.currentApiBaseUrl ?? FIXED_BASE_URL;

  while (true) {
    if (!isLoginFresh(login)) {
      return { success: false, message: "二维码已过期" };
    }

    const status = await pollQRStatus(apiBaseUrl, login.qrcode, login.pendingVerifyCode);
    login.status = status.status;

    if (status.status === "confirmed") {
      activeLogins.delete(sessionKey);
      return {
        success: true,
        botToken: status.bot_token,
        accountId: status.ilink_bot_id,
        baseUrl: status.baseurl || FIXED_BASE_URL,
        userId: status.ilink_user_id,
        message: "登录成功",
      };
    }

    if (status.status === "binded_redirect") {
      activeLogins.delete(sessionKey);
      return {
        success: true,
        message: "已连接过此机器人，无需重复登录",
      };
    }

    if (status.status === "expired") {
      activeLogins.delete(sessionKey);
      return { success: false, message: "二维码已过期，请重新获取" };
    }

    if (status.status === "scaned_but_redirect" && status.redirect_host) {
      apiBaseUrl = `https://${status.redirect_host}`;
      login.currentApiBaseUrl = apiBaseUrl;
      logger.info(`Redirecting to ${redactUrl(apiBaseUrl)}`);
      continue;
    }

    // wait / scaned / need_verifycode / verify_code_blocked → 继续轮询
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/** 一键 QR 码登录（获取二维码 + 等待确认） */
export async function loginWithQRCode(opts?: {
  botType?: string;
  onQRCode?: (qrcodeUrl: string) => void;
}): Promise<QRLoginResult> {
  const startResult = await startQRLogin(opts);

  // 显示二维码
  if (opts?.onQRCode) {
    opts.onQRCode(startResult.qrcodeUrl);
  } else {
    await displayQRCode(startResult.qrcodeUrl);
  }

  // 等待确认
  return waitForQRLogin(startResult.sessionKey);
}
