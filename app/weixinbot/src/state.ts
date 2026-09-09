/**
 * 状态持久化：账号数据、同步游标、context token
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { logger } from "./logger.js";
import { fingerprint, safeError } from "./utils.js";

// ---------------------------------------------------------------------------
// 状态目录
// ---------------------------------------------------------------------------

let customStateDir: string | undefined;

/** 设置状态存储根目录 */
export function setStateDir(dir: string): void {
  customStateDir = dir;
}

/** 获取状态存储根目录，默认 ~/.weixinbot */
export function getStateDir(): string {
  return customStateDir ?? path.join(os.homedir(), ".weixinbot");
}

function resolveWeixinStateDir(): string {
  return path.join(getStateDir(), "state");
}

// ---------------------------------------------------------------------------
// 账号数据
// ---------------------------------------------------------------------------

export interface AccountData {
  token?: string;
  savedAt?: string;
  baseUrl?: string;
  userId?: string;
}

function resolveAccountsDir(): string {
  return path.join(resolveWeixinStateDir(), "accounts");
}

function ensurePrivateDir(dir: string): void {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  try {
    fs.chmodSync(dir, 0o700);
  } catch {
    // 某些文件系统不支持 chmod；不影响状态读写。
  }
}

function writePrivateJson(filePath: string, value: unknown): void {
  const dir = path.dirname(filePath);
  ensurePrivateDir(dir);
  const tempPath = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(value, null, 2), { encoding: "utf-8", mode: 0o600 });
  fs.renameSync(tempPath, filePath);
  try {
    fs.chmodSync(filePath, 0o600);
  } catch {
    // 某些文件系统不支持 chmod；不影响状态读写。
  }
}

function accountFileStem(accountId: string): string {
  // accountId 来自远端登录结果或本地管理 API，不能直接作为路径片段。
  return encodeURIComponent(accountId);
}

function legacyAccountPath(accountId: string, suffix: string): string | null {
  // 兼容旧版直接使用 accountId 的文件名，但绝不接受目录分隔符或 . / ..。
  if (!accountId || accountId === "." || accountId === ".." || path.basename(accountId) !== accountId) {
    return null;
  }
  return path.join(resolveAccountsDir(), `${accountId}${suffix}`);
}

function resolveAccountPath(accountId: string): string {
  return path.join(resolveAccountsDir(), `${accountFileStem(accountId)}.json`);
}

function candidatePaths(accountId: string, suffix: string): string[] {
  const encoded = path.join(resolveAccountsDir(), `${accountFileStem(accountId)}${suffix}`);
  const legacy = legacyAccountPath(accountId, suffix);
  return legacy && legacy !== encoded ? [encoded, legacy] : [encoded];
}

/** 保存账号数据 */
export function saveAccount(accountId: string, data: AccountData): void {
  const dir = resolveAccountsDir();
  ensurePrivateDir(dir);

  const existing = loadAccount(accountId) ?? {};
  const merged: AccountData = {
    ...existing,
    ...data,
    savedAt: new Date().toISOString(),
  };

  const filePath = resolveAccountPath(accountId);
  writePrivateJson(filePath, merged);
  logger.debug(`saveAccount: saved ${fingerprint(accountId)}`);
}

/** 加载账号数据 */
export function loadAccount(accountId: string): AccountData | null {
  for (const filePath of candidatePaths(accountId, ".json")) {
    try {
      if (fs.existsSync(filePath)) {
        return JSON.parse(fs.readFileSync(filePath, "utf-8")) as AccountData;
      }
    } catch {
      // ignore and try compatibility path
    }
  }
  return null;
}

/** 列出所有已注册的账号 ID */
export function listAccountIds(): string[] {
  const dir = resolveAccountsDir();
  try {
    if (!fs.existsSync(dir)) return [];
    const ids = fs
      .readdirSync(dir)
      .filter((f) => f.endsWith(".json"))
      .filter((f) => !f.endsWith(".sync.json") && !f.endsWith(".context-tokens.json"))
      .map((f) => f.replace(/\.json$/, ""))
      .map((stem) => {
        try {
          return decodeURIComponent(stem);
        } catch {
          return stem;
        }
      });
    return [...new Set(ids)];
  } catch {
    return [];
  }
}

/** 删除账号凭证及该账号的同步游标/context token。 */
export function deleteAccount(accountId: string): void {
  const files = [
    ...candidatePaths(accountId, ".json"),
    ...candidatePaths(accountId, ".sync.json"),
    ...candidatePaths(accountId, ".context-tokens.json"),
  ];
  for (const filePath of files) {
    try {
      fs.rmSync(filePath, { force: true });
    } catch (err) {
      logger.warn(`deleteAccount: failed for account=${fingerprint(accountId)}: ${safeError(err)}`);
    }
  }
  const prefix = `${accountId}:`;
  for (const key of contextTokenStore.keys()) {
    if (key.startsWith(prefix)) contextTokenStore.delete(key);
  }
}

// ---------------------------------------------------------------------------
// 同步游标（get_updates_buf）
// ---------------------------------------------------------------------------

export interface SyncBufData {
  get_updates_buf: string;
}

/** 获取同步游标文件路径 */
export function getSyncBufFilePath(accountId: string): string {
  return path.join(resolveAccountsDir(), `${accountFileStem(accountId)}.sync.json`);
}

/** 加载同步游标 */
export function loadSyncBuf(accountId: string): string | undefined {
  for (const filePath of candidatePaths(accountId, ".sync.json")) {
    try {
      const raw = fs.readFileSync(filePath, "utf-8");
      const data = JSON.parse(raw) as { get_updates_buf?: string };
      if (typeof data.get_updates_buf === "string") {
        return data.get_updates_buf;
      }
    } catch {
      // ignore and try compatibility path
    }
  }
  return undefined;
}

/** 保存同步游标 */
export function saveSyncBuf(accountId: string, buf: string): void {
  const filePath = getSyncBufFilePath(accountId);
  try {
    writePrivateJson(filePath, { get_updates_buf: buf });
  } catch (err) {
    logger.warn(`saveSyncBuf: failed for account=${fingerprint(accountId)}: ${safeError(err)}`);
  }
}

// ---------------------------------------------------------------------------
// Context Token 存储
// ---------------------------------------------------------------------------

const contextTokenStore = new Map<string, string>();

function contextTokenKey(accountId: string, userId: string): string {
  return `${accountId}:${userId}`;
}

function resolveContextTokenFilePath(accountId: string): string {
  return path.join(resolveAccountsDir(), `${accountFileStem(accountId)}.context-tokens.json`);
}

/** 持久化 context tokens 到磁盘 */
function persistContextTokens(accountId: string): void {
  const prefix = `${accountId}:`;
  const tokens: Record<string, string> = {};
  for (const [k, v] of contextTokenStore) {
    if (k.startsWith(prefix)) {
      tokens[k.slice(prefix.length)] = v;
    }
  }
  const filePath = resolveContextTokenFilePath(accountId);
  try {
    writePrivateJson(filePath, tokens);
  } catch (err) {
    logger.warn(`persistContextTokens: failed for account=${fingerprint(accountId)}: ${safeError(err)}`);
  }
}

/** 恢复 context tokens（启动时调用） */
export function restoreContextTokens(accountId: string): void {
  for (const filePath of candidatePaths(accountId, ".context-tokens.json")) {
    try {
      if (!fs.existsSync(filePath)) continue;
      const raw = fs.readFileSync(filePath, "utf-8");
      const tokens = JSON.parse(raw) as Record<string, string>;
      let count = 0;
      for (const [userId, token] of Object.entries(tokens)) {
        if (typeof token === "string" && token) {
          contextTokenStore.set(contextTokenKey(accountId, userId), token);
          count++;
        }
      }
      logger.info(`restoreContextTokens: restored ${count} tokens for ${fingerprint(accountId)}`);
      return;
    } catch (err) {
      logger.warn(`restoreContextTokens: failed for account=${fingerprint(accountId)}: ${safeError(err)}`);
    }
  }
}

/** 存储 context token */
export function setContextToken(accountId: string, userId: string, token: string): void {
  contextTokenStore.set(contextTokenKey(accountId, userId), token);
  persistContextTokens(accountId);
}

/** 获取 context token */
export function getContextToken(accountId: string, userId: string): string | undefined {
  return contextTokenStore.get(contextTokenKey(accountId, userId));
}

/** 查找拥有指定用户 context token 的所有账号 */
export function findAccountIdsByContextToken(accountIds: string[], userId: string): string[] {
  return accountIds.filter((id) => contextTokenStore.has(contextTokenKey(id, userId)));
}
