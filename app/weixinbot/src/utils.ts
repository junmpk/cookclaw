/**
 * 工具函数：随机 ID 生成、脱敏、MIME 类型映射
 */
import crypto from "node:crypto";
import path from "node:path";

// ---------------------------------------------------------------------------
// 随机 ID / 临时文件名
// ---------------------------------------------------------------------------

/** 生成带前缀的唯一 ID：`{prefix}:{timestamp}-{8位hex}` */
export function generateId(prefix: string): string {
  return `${prefix}:${Date.now()}-${crypto.randomBytes(4).toString("hex")}`;
}

/** 生成临时文件名：`{prefix}-{timestamp}-{8位hex}{ext}` */
export function tempFileName(prefix: string, ext: string): string {
  return `${prefix}-${Date.now()}-${crypto.randomBytes(4).toString("hex")}${ext}`;
}

// ---------------------------------------------------------------------------
// 脱敏
// ---------------------------------------------------------------------------

/** 截断字符串，超出部分用省略号 + 长度标注 */
export function truncate(s: string | undefined, max: number): string {
  if (!s) return "";
  if (s.length <= max) return s;
  return `${s.slice(0, max)}…(len=${s.length})`;
}

/** 脱敏 token：只显示前几个字符 + 总长度 */
export function redactToken(token: string | undefined, prefixLen = 6): string {
  if (!token) return "(none)";
  if (token.length <= prefixLen) return `****(len=${token.length})`;
  return `${token.slice(0, prefixLen)}…(len=${token.length})`;
}

/** 脱敏 JSON body 中的敏感字段并截断 */
export function redactBody(body: string | undefined, maxLen = 200): string {
  if (!body) return "(empty)";
  // 请求/响应中除 token 外还有正文、用户 ID、client_id、AES key 等敏感字段。
  // 日志只保留 JSON 顶层形状和长度，不做“替换几个字段后打印正文”的伪脱敏。
  try {
    const parsed = JSON.parse(body) as unknown;
    if (Array.isArray(parsed)) return `json(array,len=${parsed.length},chars=${body.length})`;
    if (parsed && typeof parsed === "object") {
      const keys = Object.keys(parsed as Record<string, unknown>).sort().slice(0, 20);
      return `json(object,keys=${keys.join("|") || "-"},chars=${body.length})`;
    }
    return `json(${typeof parsed},chars=${body.length})`;
  } catch {
    return `non-json(chars=${body.length})`;
  }
}

/** 脱敏 URL：移除 query string */
export function redactUrl(rawUrl: string): string {
  try {
    const u = new URL(rawUrl);
    return `${u.protocol}//${u.hostname}${u.port ? `:${u.port}` : ""}`;
  } catch {
    return "invalid-url";
  }
}

/** 稳定不可逆短指纹，只用于关联日志，不输出账号/消息原值。 */
export function fingerprint(value: unknown): string {
  const raw = String(value ?? "");
  return raw ? `sha256:${crypto.createHash("sha256").update(raw).digest("hex").slice(0, 10)}` : "-";
}

/** 异常日志只保留类型与数值状态，避免 Error.message 二次携带原始响应。 */
export function safeError(error: unknown): string {
  if (!error || typeof error !== "object") return `type=${typeof error}`;
  const value = error as { name?: unknown; code?: unknown; status?: unknown };
  const name = typeof value.name === "string" ? value.name : "Error";
  const code = [value.code, value.status].find(
    (item) => typeof item === "number" || (typeof item === "string" && /^[A-Z0-9_-]{1,32}$/.test(item)),
  );
  return `type=${name}${code !== undefined ? ` code=${String(code)}` : ""}`;
}

// ---------------------------------------------------------------------------
// MIME 类型映射
// ---------------------------------------------------------------------------

const EXTENSION_TO_MIME: Record<string, string> = {
  ".pdf": "application/pdf",
  ".doc": "application/msword",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".xls": "application/vnd.ms-excel",
  ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ".ppt": "application/vnd.ms-powerpoint",
  ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  ".txt": "text/plain",
  ".csv": "text/csv",
  ".zip": "application/zip",
  ".tar": "application/x-tar",
  ".gz": "application/gzip",
  ".mp3": "audio/mpeg",
  ".ogg": "audio/ogg",
  ".wav": "audio/wav",
  ".mp4": "video/mp4",
  ".mov": "video/quicktime",
  ".webm": "video/webm",
  ".mkv": "video/x-matroska",
  ".avi": "video/x-msvideo",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
};

const MIME_TO_EXTENSION: Record<string, string> = {
  "image/jpeg": ".jpg",
  "image/jpg": ".jpg",
  "image/png": ".png",
  "image/gif": ".gif",
  "image/webp": ".webp",
  "image/bmp": ".bmp",
  "video/mp4": ".mp4",
  "video/quicktime": ".mov",
  "video/webm": ".webm",
  "video/x-matroska": ".mkv",
  "video/x-msvideo": ".avi",
  "audio/mpeg": ".mp3",
  "audio/ogg": ".ogg",
  "audio/wav": ".wav",
  "application/pdf": ".pdf",
  "application/zip": ".zip",
  "application/x-tar": ".tar",
  "application/gzip": ".gz",
  "text/plain": ".txt",
  "text/csv": ".csv",
};

/** 根据文件扩展名获取 MIME 类型 */
export function getMimeFromFilename(filename: string): string {
  const ext = path.extname(filename).toLowerCase();
  return EXTENSION_TO_MIME[ext] ?? "application/octet-stream";
}

/** 根据 MIME 类型获取文件扩展名 */
export function getExtensionFromMime(mimeType: string): string {
  const ct = mimeType.split(";")[0].trim().toLowerCase();
  return MIME_TO_EXTENSION[ct] ?? ".bin";
}

/** 从 Content-Type 或 URL 推断文件扩展名 */
export function getExtensionFromContentTypeOrUrl(contentType: string | null, url: string): string {
  if (contentType) {
    const ext = getExtensionFromMime(contentType);
    if (ext !== ".bin") return ext;
  }
  const ext = path.extname(new URL(url).pathname).toLowerCase();
  const knownExts = new Set(Object.keys(EXTENSION_TO_MIME));
  return knownExts.has(ext) ? ext : ".bin";
}

// ---------------------------------------------------------------------------
// 其他
// ---------------------------------------------------------------------------

/** 确保路径以 / 结尾 */
export function ensureTrailingSlash(url: string): string {
  return url.endsWith("/") ? url : `${url}/`;
}

/** X-WECHAT-UIN header：随机 uint32 → 十进制字符串 → base64 */
export function randomWechatUin(): string {
  const uint32 = crypto.randomBytes(4).readUInt32BE(0);
  return Buffer.from(String(uint32), "utf-8").toString("base64");
}

/** sleep 工具函数 */
export function sleep(ms: number, abortSignal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (abortSignal?.aborted) {
      resolve();
      return;
    }
    const timer = setTimeout(resolve, ms);
    abortSignal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}
