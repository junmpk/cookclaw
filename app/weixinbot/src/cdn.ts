/**
 * CDN 操作：AES-128-ECB 加密/解密、上传、下载
 */
import { createCipheriv, createDecipheriv } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

import { getUploadUrl } from "./api.js";
import type { WeixinApiOptions } from "./api.js";
import { logger } from "./logger.js";
import { getExtensionFromContentTypeOrUrl, redactUrl, tempFileName } from "./utils.js";
import { UploadMediaType } from "./types.js";

// ---------------------------------------------------------------------------
// AES-128-ECB 加密/解密
// ---------------------------------------------------------------------------

/** AES-128-ECB 加密（PKCS7 填充） */
export function encryptAesEcb(plaintext: Buffer, key: Buffer): Buffer {
  const cipher = createCipheriv("aes-128-ecb", key, null);
  return Buffer.concat([cipher.update(plaintext), cipher.final()]);
}

/** AES-128-ECB 解密 */
export function decryptAesEcb(ciphertext: Buffer, key: Buffer): Buffer {
  const decipher = createDecipheriv("aes-128-ecb", key, null);
  return Buffer.concat([decipher.update(ciphertext), decipher.final()]);
}

/** 计算 AES-128-ECB 密文大小（PKCS7 填充到 16 字节边界） */
export function aesEcbPaddedSize(plaintextSize: number): number {
  return Math.ceil((plaintextSize + 1) / 16) * 16;
}

// ---------------------------------------------------------------------------
// CDN URL 构建
// ---------------------------------------------------------------------------

const DEFAULT_CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c";

/** 构建 CDN 下载 URL */
function buildCdnDownloadUrl(encryptedQueryParam: string, cdnBaseUrl: string): string {
  return `${cdnBaseUrl}/download?encrypted_query_param=${encodeURIComponent(encryptedQueryParam)}`;
}

/** 构建 CDN 上传 URL */
function buildCdnUploadUrl(params: { cdnBaseUrl: string; uploadParam: string; filekey: string }): string {
  return `${params.cdnBaseUrl}/upload?encrypted_query_param=${encodeURIComponent(params.uploadParam)}&filekey=${encodeURIComponent(params.filekey)}`;
}

// ---------------------------------------------------------------------------
// CDN 上传
// ---------------------------------------------------------------------------

const UPLOAD_MAX_RETRIES = 3;

/** 上传 buffer 到 CDN（AES-128-ECB 加密） */
export async function uploadBufferToCdn(params: {
  buf: Buffer;
  uploadFullUrl?: string;
  uploadParam?: string;
  filekey: string;
  cdnBaseUrl: string;
  label: string;
  aeskey: Buffer;
}): Promise<{ downloadParam: string }> {
  const { buf, uploadFullUrl, uploadParam, filekey, cdnBaseUrl, label, aeskey } = params;
  const ciphertext = encryptAesEcb(buf, aeskey);
  const trimmedFull = uploadFullUrl?.trim();
  let cdnUrl: string;
  if (trimmedFull) {
    cdnUrl = trimmedFull;
  } else if (uploadParam) {
    cdnUrl = buildCdnUploadUrl({ cdnBaseUrl, uploadParam, filekey });
  } else {
    throw new Error(`${label}: CDN upload URL missing`);
  }
  logger.debug(`${label}: CDN POST ciphertextSize=${ciphertext.length}`);

  let downloadParam: string | undefined;
  let lastError: unknown;

  for (let attempt = 1; attempt <= UPLOAD_MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(cdnUrl, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: new Uint8Array(ciphertext),
      });
      if (res.status >= 400 && res.status < 500) {
        const responseText = res.headers.get("x-error-message") ?? (await res.text());
        throw new Error(`CDN upload client error ${res.status} response_chars=${responseText.length}`);
      }
      if (res.status !== 200) {
        throw new Error(`CDN upload server error status=${res.status}`);
      }
      downloadParam = res.headers.get("x-encrypted-param") ?? undefined;
      if (!downloadParam) {
        throw new Error("CDN upload response missing x-encrypted-param header");
      }
      logger.debug(`${label}: CDN upload success attempt=${attempt}`);
      break;
    } catch (err) {
      lastError = err;
      if (err instanceof Error && err.message.includes("client error")) throw err;
      if (attempt < UPLOAD_MAX_RETRIES) {
        logger.error(`${label}: attempt ${attempt} failed, retrying...`);
      }
    }
  }

  if (!downloadParam) {
    throw lastError instanceof Error
      ? lastError
      : new Error(`CDN upload failed after ${UPLOAD_MAX_RETRIES} attempts`);
  }
  return { downloadParam };
}

// ---------------------------------------------------------------------------
// CDN 下载
// ---------------------------------------------------------------------------

/** 从 CDN 下载原始字节 */
async function fetchCdnBytes(url: string, label: string): Promise<Buffer> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${label}: CDN download status=${res.status} response_chars=${body.length}`);
  }
  return Buffer.from(await res.arrayBuffer());
}

/** 解析 AES 密钥（支持 base64 原始字节和 base64 hex 字符串两种格式） */
function parseAesKey(aesKeyBase64: string, label: string): Buffer {
  const decoded = Buffer.from(aesKeyBase64, "base64");
  if (decoded.length === 16) return decoded;
  if (decoded.length === 32 && /^[0-9a-fA-F]{32}$/.test(decoded.toString("ascii"))) {
    return Buffer.from(decoded.toString("ascii"), "hex");
  }
  throw new Error(`${label}: invalid aes_key format`);
}

/** 下载并解密 CDN 媒体文件 */
export async function downloadAndDecryptBuffer(
  encryptedQueryParam: string,
  aesKeyBase64: string,
  cdnBaseUrl: string,
  label: string,
  fullUrl?: string,
): Promise<Buffer> {
  const key = parseAesKey(aesKeyBase64, label);
  const url = fullUrl || buildCdnDownloadUrl(encryptedQueryParam, cdnBaseUrl);
  logger.debug(`${label}: downloading origin=${redactUrl(url)}`);
  const encrypted = await fetchCdnBytes(url, label);
  logger.debug(`${label}: downloaded ${encrypted.byteLength} bytes, decrypting`);
  return decryptAesEcb(encrypted, key);
}

/** 下载未加密的 CDN 资源 */
export async function downloadPlainCdnBuffer(
  encryptedQueryParam: string,
  cdnBaseUrl: string,
  label: string,
  fullUrl?: string,
): Promise<Buffer> {
  const url = fullUrl || buildCdnDownloadUrl(encryptedQueryParam, cdnBaseUrl);
  return fetchCdnBytes(url, label);
}

// ---------------------------------------------------------------------------
// 文件上传到 CDN（完整流程）
// ---------------------------------------------------------------------------

export interface UploadedFileInfo {
  filekey: string;
  downloadEncryptedQueryParam: string;
  aeskey: string;
  fileSize: number;
  fileSizeCiphertext: number;
}

/** 下载远程文件到临时目录 */
export async function downloadRemoteImageToTemp(url: string, destDir: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`remote media download failed: status=${res.status} origin=${redactUrl(url)}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  await fs.mkdir(destDir, { recursive: true });
  const ext = getExtensionFromContentTypeOrUrl(res.headers.get("content-type"), url);
  const name = tempFileName("weixin-remote", ext);
  const filePath = path.join(destDir, name);
  await fs.writeFile(filePath, buf);
  return filePath;
}

/** 上传媒体文件到 CDN（通用流程） */
async function uploadMediaToCdn(params: {
  filePath: string;
  toUserId: string;
  opts: WeixinApiOptions;
  cdnBaseUrl: string;
  mediaType: number;
  label: string;
}): Promise<UploadedFileInfo> {
  const { filePath, toUserId, opts, cdnBaseUrl, mediaType, label } = params;
  const crypto = await import("node:crypto");

  const plaintext = await fs.readFile(filePath);
  const rawsize = plaintext.length;
  const rawfilemd5 = crypto.createHash("md5").update(plaintext).digest("hex");
  const filesize = aesEcbPaddedSize(rawsize);
  const filekey = crypto.randomBytes(16).toString("hex");
  const aeskey = crypto.randomBytes(16);

  logger.debug(`${label}: extension=${path.extname(filePath) || ".bin"} rawsize=${rawsize} filesize=${filesize}`);

  const uploadUrlResp = await getUploadUrl({
    ...opts,
    filekey,
    media_type: mediaType,
    to_user_id: toUserId,
    rawsize,
    rawfilemd5,
    filesize,
    no_need_thumb: true,
    aeskey: aeskey.toString("hex"),
  });

  const uploadFullUrl = uploadUrlResp.upload_full_url?.trim();
  const uploadParam = uploadUrlResp.upload_param;
  if (!uploadFullUrl && !uploadParam) {
    throw new Error(`${label}: getUploadUrl returned no upload URL`);
  }

  const { downloadParam } = await uploadBufferToCdn({
    buf: plaintext,
    uploadFullUrl: uploadFullUrl || undefined,
    uploadParam: uploadParam ?? undefined,
    filekey,
    cdnBaseUrl,
    aeskey,
    label: `${label}[filekey=${filekey}]`,
  });

  return {
    filekey,
    downloadEncryptedQueryParam: downloadParam,
    aeskey: aeskey.toString("hex"),
    fileSize: rawsize,
    fileSizeCiphertext: filesize,
  };
}

/** 上传图片到微信 CDN */
export async function uploadFileToWeixin(params: {
  filePath: string;
  toUserId: string;
  opts: WeixinApiOptions;
  cdnBaseUrl: string;
}): Promise<UploadedFileInfo> {
  return uploadMediaToCdn({ ...params, mediaType: UploadMediaType.IMAGE, label: "uploadFileToWeixin" });
}

/** 上传视频到微信 CDN */
export async function uploadVideoToWeixin(params: {
  filePath: string;
  toUserId: string;
  opts: WeixinApiOptions;
  cdnBaseUrl: string;
}): Promise<UploadedFileInfo> {
  return uploadMediaToCdn({ ...params, mediaType: UploadMediaType.VIDEO, label: "uploadVideoToWeixin" });
}

/** 上传文件附件到微信 CDN */
export async function uploadFileAttachmentToWeixin(params: {
  filePath: string;
  fileName: string;
  toUserId: string;
  opts: WeixinApiOptions;
  cdnBaseUrl: string;
}): Promise<UploadedFileInfo> {
  return uploadMediaToCdn({ ...params, mediaType: UploadMediaType.FILE, label: "uploadFileAttachmentToWeixin" });
}
