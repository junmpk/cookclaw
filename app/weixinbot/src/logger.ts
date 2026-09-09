/**
 * 独立日志器 — 控制台输出 + 可选文件日志
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fingerprint } from "./utils.js";

const SUBSYSTEM = "weixinbot";
const LEVEL_IDS: Record<string, number> = {
  TRACE: 1,
  DEBUG: 2,
  INFO: 3,
  WARN: 4,
  ERROR: 5,
  FATAL: 6,
};

let minLevelId = LEVEL_IDS.INFO;

/** 设置最低日志级别 */
export function setLogLevel(level: string): void {
  const upper = level.toUpperCase();
  if (!(upper in LEVEL_IDS)) {
    throw new Error(`Invalid log level: ${level}. Valid: ${Object.keys(LEVEL_IDS).join(", ")}`);
  }
  minLevelId = LEVEL_IDS[upper];
}

/** 日期转为本地 ISO 格式字符串 */
function toLocalISO(now: Date): string {
  const offsetMs = -now.getTimezoneOffset() * 60_000;
  const sign = offsetMs >= 0 ? "+" : "-";
  const abs = Math.abs(now.getTimezoneOffset());
  const offStr = `${sign}${String(Math.floor(abs / 60)).padStart(2, "0")}:${String(abs % 60).padStart(2, "0")}`;
  return new Date(now.getTime() + offsetMs).toISOString().replace("Z", offStr);
}

export interface Logger {
  info(message: string): void;
  debug(message: string): void;
  warn(message: string): void;
  error(message: string): void;
  withAccount(accountId: string): Logger;
}

/** 创建日志器实例 */
export function createLogger(logDir?: string): Logger {
  const mainLogDir = logDir;
  let logDirEnsured = false;

  function ensureLogDir(): void {
    if (logDirEnsured || !mainLogDir) return;
    try {
      fs.mkdirSync(mainLogDir, { recursive: true });
    } catch {
      // ignore
    }
    logDirEnsured = true;
  }

  function writeLog(level: string, message: string, accountId?: string): void {
    const levelId = LEVEL_IDS[level] ?? LEVEL_IDS.INFO;
    if (levelId < minLevelId) return;

    const now = new Date();
    const prefix = accountId ? `[${fingerprint(accountId)}] ` : "";
    const timestamp = toLocalISO(now);
    const line = `${timestamp} [${level}] ${prefix}${message}`;

    // 控制台输出
    if (levelId >= LEVEL_IDS.WARN) {
      console.error(line);
    } else {
      console.log(line);
    }

    // 文件输出
    if (mainLogDir) {
      ensureLogDir();
      const dateKey = toLocalISO(now).slice(0, 10);
      const logPath = path.join(mainLogDir, `weixinbot-${dateKey}.log`);
      try {
        fs.appendFileSync(logPath, line + "\n", "utf-8");
      } catch {
        // ignore write errors
      }
    }
  }

  function createAccountLogger(accountId: string): Logger {
    return {
      info: (msg) => writeLog("INFO", msg, accountId),
      debug: (msg) => writeLog("DEBUG", msg, accountId),
      warn: (msg) => writeLog("WARN", msg, accountId),
      error: (msg) => writeLog("ERROR", msg, accountId),
      withAccount: () => createAccountLogger(accountId),
    };
  }

  return {
    info: (msg) => writeLog("INFO", msg),
    debug: (msg) => writeLog("DEBUG", msg),
    warn: (msg) => writeLog("WARN", msg),
    error: (msg) => writeLog("ERROR", msg),
    withAccount: createAccountLogger,
  };
}

/** 全局默认日志器 */
export const logger = createLogger();
