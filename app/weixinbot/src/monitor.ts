/**
 * 长轮询消息监听器
 */
import { getUpdates, getConfig, sendTyping as sendTypingApi } from "./api.js";
import { downloadMediaFromItem } from "./media.js";
import { sendTextMessage, sendMediaFile } from "./messaging.js";
import { logger } from "./logger.js";
import { loadSyncBuf, saveSyncBuf, setContextToken, getContextToken } from "./state.js";
import { fingerprint, redactUrl, safeError, sleep } from "./utils.js";
import type { WeixinMessage, MessageContext, OnMessageCallback } from "./types.js";
import { MessageItemType, TypingStatus } from "./types.js";

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

const DEFAULT_LONG_POLL_TIMEOUT_MS = 35_000;
const MAX_CONSECUTIVE_FAILURES = 3;
const BACKOFF_DELAY_MS = 30_000;
const RETRY_DELAY_MS = 2_000;
const SESSION_EXPIRED_ERRCODE = -14;

// ---------------------------------------------------------------------------
// Config 缓存
// ---------------------------------------------------------------------------

interface CachedConfig {
  typingTicket: string;
}

const CONFIG_CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const CONFIG_CACHE_INITIAL_RETRY_MS = 2_000;
const CONFIG_CACHE_MAX_RETRY_MS = 60 * 60 * 1000;

interface ConfigCacheEntry {
  config: CachedConfig;
  everSucceeded: boolean;
  nextFetchAt: number;
  retryDelayMs: number;
}

class WeixinConfigManager {
  private cache = new Map<string, ConfigCacheEntry>();

  constructor(
    private apiOpts: { baseUrl: string; token?: string; botAgent?: string },
  ) {}

  async getForUser(userId: string, contextToken?: string): Promise<CachedConfig> {
    const now = Date.now();
    const entry = this.cache.get(userId);
    const shouldFetch = !entry || now >= entry.nextFetchAt;

    if (shouldFetch) {
      let fetchOk = false;
      try {
        const resp = await getConfig({
          baseUrl: this.apiOpts.baseUrl,
          token: this.apiOpts.token,
          ilinkUserId: userId,
          contextToken,
          botAgent: this.apiOpts.botAgent,
        });
        if (resp.ret === 0) {
          this.cache.set(userId, {
            config: { typingTicket: resp.typing_ticket ?? "" },
            everSucceeded: true,
            nextFetchAt: now + Math.random() * CONFIG_CACHE_TTL_MS,
            retryDelayMs: CONFIG_CACHE_INITIAL_RETRY_MS,
          });
          fetchOk = true;
        }
      } catch (err) {
        logger.warn(`getConfig failed for user=${fingerprint(userId)}: ${safeError(err)}`);
      }
      if (!fetchOk) {
        const prevDelay = entry?.retryDelayMs ?? CONFIG_CACHE_INITIAL_RETRY_MS;
        const nextDelay = Math.min(prevDelay * 2, CONFIG_CACHE_MAX_RETRY_MS);
        if (entry) {
          entry.nextFetchAt = now + nextDelay;
          entry.retryDelayMs = nextDelay;
        } else {
          this.cache.set(userId, {
            config: { typingTicket: "" },
            everSucceeded: false,
            nextFetchAt: now + CONFIG_CACHE_INITIAL_RETRY_MS,
            retryDelayMs: CONFIG_CACHE_INITIAL_RETRY_MS,
          });
        }
      }
    }

    return this.cache.get(userId)?.config ?? { typingTicket: "" };
  }
}

// ---------------------------------------------------------------------------
// 消息文本提取
// ---------------------------------------------------------------------------

function extractTextBody(itemList?: WeixinMessage["item_list"]): string {
  if (!itemList?.length) return "";
  for (const item of itemList) {
    if (item.type === MessageItemType.TEXT && item.text_item?.text != null) {
      return String(item.text_item.text);
    }
    if (item.type === MessageItemType.VOICE && item.voice_item?.text) {
      return item.voice_item.text;
    }
  }
  return "";
}

// ---------------------------------------------------------------------------
// Monitor 配置
// ---------------------------------------------------------------------------

export interface MonitorOpts {
  baseUrl: string;
  cdnBaseUrl: string;
  token?: string;
  accountId: string;
  botAgent?: string;
  longPollTimeoutMs?: number;
  mediaDir: string;
  abortSignal?: AbortSignal;
  onConnected?: () => void;
  onMessage: OnMessageCallback;
}

// ---------------------------------------------------------------------------
// 长轮询主循环
// ---------------------------------------------------------------------------

/** 启动长轮询消息监听 */
export async function startMonitor(opts: MonitorOpts): Promise<void> {
  const {
    baseUrl,
    cdnBaseUrl,
    token,
    accountId,
    botAgent,
    longPollTimeoutMs,
    mediaDir,
    abortSignal,
    onConnected,
    onMessage,
  } = opts;

  const aLog = logger.withAccount(accountId);
  aLog.info(`Monitor started: api=${redactUrl(baseUrl)}`);

  // 恢复同步游标
  let getUpdatesBuf = loadSyncBuf(accountId) ?? "";
  if (getUpdatesBuf) {
    aLog.info(`Resuming from previous sync buf (${getUpdatesBuf.length} bytes)`);
  }

  const configManager = new WeixinConfigManager({ baseUrl, token, botAgent });
  let nextTimeoutMs = longPollTimeoutMs ?? DEFAULT_LONG_POLL_TIMEOUT_MS;
  let consecutiveFailures = 0;
  let connectedReported = false;

  while (!abortSignal?.aborted) {
    try {
      const resp = await getUpdates({
        baseUrl,
        token,
        get_updates_buf: getUpdatesBuf,
        timeoutMs: nextTimeoutMs,
        botAgent,
      });

      if (resp.longpolling_timeout_ms != null && resp.longpolling_timeout_ms > 0) {
        nextTimeoutMs = resp.longpolling_timeout_ms;
      }

      const isApiError =
        (resp.ret !== undefined && resp.ret !== 0) ||
        (resp.errcode !== undefined && resp.errcode !== 0);

      if (isApiError) {
        const isSessionExpired =
          resp.errcode === SESSION_EXPIRED_ERRCODE || resp.ret === SESSION_EXPIRED_ERRCODE;

        if (isSessionExpired) {
          aLog.error("Session expired, stopping monitor and requiring QR login");
          const error = new Error(resp.errmsg || "Weixin session expired") as Error & { code: number };
          error.name = "WeixinAuthExpiredError";
          error.code = SESSION_EXPIRED_ERRCODE;
          throw error;
        }

        consecutiveFailures += 1;
        aLog.error(`getUpdates failed: ret=${resp.ret} errcode=${resp.errcode} (${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES})`);
        if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
          consecutiveFailures = 0;
          await sleep(BACKOFF_DELAY_MS, abortSignal);
        } else {
          await sleep(RETRY_DELAY_MS, abortSignal);
        }
        continue;
      }

      consecutiveFailures = 0;
      if (!connectedReported) {
        connectedReported = true;
        onConnected?.();
      }

      // 保存同步游标
      if (resp.get_updates_buf) {
        saveSyncBuf(accountId, resp.get_updates_buf);
        getUpdatesBuf = resp.get_updates_buf;
      }

      // 处理消息
      const msgs = resp.msgs ?? [];
      for (const full of msgs) {
        aLog.info(
          `inbound: from=${fingerprint(full.from_user_id)} types=${full.item_list?.map((i) => i.type).join(",") ?? "none"}`
        );

        await processOneMessage(full, {
          accountId,
          baseUrl,
          cdnBaseUrl,
          token,
          botAgent,
          mediaDir,
          configManager,
          onMessage,
        });
      }
    } catch (err) {
      if (abortSignal?.aborted) {
        aLog.info("Monitor stopped (aborted)");
        return;
      }
      if (err && typeof err === "object" && "code" in err && (err as { code?: number }).code === SESSION_EXPIRED_ERRCODE) {
        throw err;
      }
      consecutiveFailures += 1;
      aLog.error(
        `getUpdates error (${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES}): ${safeError(err)}`
      );
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        consecutiveFailures = 0;
        await sleep(BACKOFF_DELAY_MS, abortSignal);
      } else {
        await sleep(RETRY_DELAY_MS, abortSignal);
      }
    }
  }

  aLog.info("Monitor loop exited");
}

// ---------------------------------------------------------------------------
// 单条消息处理
// ---------------------------------------------------------------------------

async function processOneMessage(
  full: WeixinMessage,
  deps: {
    accountId: string;
    baseUrl: string;
    cdnBaseUrl: string;
    token?: string;
    botAgent?: string;
    mediaDir: string;
    configManager: WeixinConfigManager;
    onMessage: OnMessageCallback;
  },
): Promise<void> {
  const { accountId, baseUrl, cdnBaseUrl, token, botAgent, mediaDir, configManager, onMessage } = deps;

  const fromUserId = full.from_user_id ?? "";
  const textBody = extractTextBody(full.item_list);

  // 缓存 context token
  if (full.context_token && fromUserId) {
    setContextToken(accountId, fromUserId, full.context_token);
  }

  // 下载媒体
  const hasDownloadableMedia = (m?: { encrypt_query_param?: string; full_url?: string }) =>
    m?.encrypt_query_param || m?.full_url;

  const mediaItem =
    full.item_list?.find((i) => i.type === MessageItemType.IMAGE && hasDownloadableMedia(i.image_item?.media)) ??
    full.item_list?.find((i) => i.type === MessageItemType.VIDEO && hasDownloadableMedia(i.video_item?.media)) ??
    full.item_list?.find((i) => i.type === MessageItemType.FILE && hasDownloadableMedia(i.file_item?.media)) ??
    full.item_list?.find((i) => i.type === MessageItemType.VOICE && hasDownloadableMedia(i.voice_item?.media) && !i.voice_item?.text);

  let mediaResult: { imagePath?: string; voicePath?: string; voiceMediaType?: string; filePath?: string; fileMediaType?: string; videoPath?: string } = {};
  if (mediaItem) {
    mediaResult = await downloadMediaFromItem(mediaItem, cdnBaseUrl, mediaDir);
  }

  // 获取 typing ticket
  const cachedConfig = await configManager.getForUser(fromUserId, full.context_token);

  // 构建 MessageContext
  const contextToken = full.context_token ?? getContextToken(accountId, fromUserId);
  const apiOpts = { baseUrl, token, contextToken, botAgent };

  const messageContext: MessageContext = {
    fromUserId,
    toUserId: full.to_user_id ?? "",
    contextToken,
    accountId,
    text: textBody,
    imagePath: mediaResult.imagePath,
    voicePath: mediaResult.voicePath,
    filePath: mediaResult.filePath,
    videoPath: mediaResult.videoPath,
    voiceMediaType: mediaResult.voiceMediaType,
    fileMediaType: mediaResult.fileMediaType,
    replyText: async (text: string) => {
      return sendTextMessage({ to: fromUserId, text, opts: apiOpts });
    },
    replyImage: async (filePath: string, text?: string) => {
      return sendMediaFile({ filePath, to: fromUserId, text: text ?? "", opts: apiOpts, cdnBaseUrl });
    },
    replyVideo: async (filePath: string, text?: string) => {
      return sendMediaFile({ filePath, to: fromUserId, text: text ?? "", opts: apiOpts, cdnBaseUrl });
    },
    replyFile: async (filePath: string, _fileName?: string, text?: string) => {
      return sendMediaFile({ filePath, to: fromUserId, text: text ?? "", opts: apiOpts, cdnBaseUrl });
    },
    sendTyping: async () => {
      if (cachedConfig.typingTicket) {
        await sendTypingApi({
          baseUrl,
          token,
          botAgent,
          body: {
            ilink_user_id: fromUserId,
            typing_ticket: cachedConfig.typingTicket,
            status: TypingStatus.TYPING,
          },
        });
      }
    },
  };

  try {
    await onMessage(full, messageContext);
  } catch (err) {
    logger.error(`onMessage callback error: ${safeError(err)}`);
  }
}
