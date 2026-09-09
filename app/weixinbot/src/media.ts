/**
 * 媒体下载和处理
 */
import fs from "node:fs";
import path from "node:path";

import { downloadAndDecryptBuffer, downloadPlainCdnBuffer } from "./cdn.js";
import { logger } from "./logger.js";
import { getMimeFromFilename, safeError } from "./utils.js";
import type { MessageItem } from "./types.js";
import { MessageItemType } from "./types.js";

const WEIXIN_MEDIA_MAX_BYTES = 100 * 1024 * 1024;

// ---------------------------------------------------------------------------
// SILK 语音转码
// ---------------------------------------------------------------------------

const SILK_SAMPLE_RATE = 24_000;

function pcmBytesToWav(pcm: Uint8Array, sampleRate: number): Buffer {
  const pcmBytes = pcm.byteLength;
  const totalSize = 44 + pcmBytes;
  const buf = Buffer.allocUnsafe(totalSize);
  let offset = 0;

  buf.write("RIFF", offset); offset += 4;
  buf.writeUInt32LE(totalSize - 8, offset); offset += 4;
  buf.write("WAVE", offset); offset += 4;
  buf.write("fmt ", offset); offset += 4;
  buf.writeUInt32LE(16, offset); offset += 4;
  buf.writeUInt16LE(1, offset); offset += 2;
  buf.writeUInt16LE(1, offset); offset += 2;
  buf.writeUInt32LE(sampleRate, offset); offset += 4;
  buf.writeUInt32LE(sampleRate * 2, offset); offset += 4;
  buf.writeUInt16LE(2, offset); offset += 2;
  buf.writeUInt16LE(16, offset); offset += 2;
  buf.write("data", offset); offset += 4;
  buf.writeUInt32LE(pcmBytes, offset); offset += 4;
  Buffer.from(pcm.buffer, pcm.byteOffset, pcm.byteLength).copy(buf, offset);

  return buf;
}

/** 尝试将 SILK 音频转为 WAV（需要 silk-wasm） */
async function silkToWav(silkBuf: Buffer): Promise<Buffer | null> {
  try {
    const { decode } = await import("silk-wasm");
    const result = await decode(silkBuf, SILK_SAMPLE_RATE);
    return pcmBytesToWav(result.data, SILK_SAMPLE_RATE);
  } catch {
    logger.warn("silkToWav: transcode unavailable, will use raw SILK");
    return null;
  }
}

// ---------------------------------------------------------------------------
// 媒体保存
// ---------------------------------------------------------------------------

/** 保存 buffer 到本地文件 */
export async function saveMediaBuffer(
  buffer: Buffer,
  mediaDir: string,
  contentType?: string,
  subdir?: string,
  originalFilename?: string,
): Promise<{ path: string }> {
  const dir = subdir ? path.join(mediaDir, subdir) : mediaDir;
  fs.mkdirSync(dir, { recursive: true });

  const timestamp = Date.now();
  const random = Math.random().toString(36).slice(2, 10);
  let filename: string;
  if (originalFilename) {
    filename = `${timestamp}-${random}-${originalFilename}`;
  } else {
    const ext = contentType ? getExtFromContentType(contentType) : ".bin";
    filename = `${timestamp}-${random}${ext}`;
  }

  const filePath = path.join(dir, filename);
  fs.writeFileSync(filePath, buffer);
  return { path: filePath };
}

function getExtFromContentType(contentType: string): string {
  const map: Record<string, string> = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/silk": ".silk",
    "application/pdf": ".pdf",
  };
  const ct = contentType.split(";")[0].trim().toLowerCase();
  return map[ct] ?? ".bin";
}

// ---------------------------------------------------------------------------
// 媒体下载
// ---------------------------------------------------------------------------

export interface MediaDownloadResult {
  /** 已下载的图片路径 */
  imagePath?: string;
  /** 已下载的语音路径 */
  voicePath?: string;
  /** 语音 MIME 类型 */
  voiceMediaType?: string;
  /** 已下载的文件路径 */
  filePath?: string;
  /** 文件 MIME 类型 */
  fileMediaType?: string;
  /** 已下载的视频路径 */
  videoPath?: string;
}

/** 从消息项下载媒体文件 */
export async function downloadMediaFromItem(
  item: MessageItem,
  cdnBaseUrl: string,
  mediaDir: string,
): Promise<MediaDownloadResult> {
  const result: MediaDownloadResult = {};

  if (item.type === MessageItemType.IMAGE) {
    const img = item.image_item;
    if (!img?.media?.encrypt_query_param && !img?.media?.full_url) return result;
    const aesKeyBase64 = img.aeskey
      ? Buffer.from(img.aeskey, "hex").toString("base64")
      : img.media.aes_key;
    try {
      const buf = aesKeyBase64
        ? await downloadAndDecryptBuffer(
            img.media.encrypt_query_param ?? "",
            aesKeyBase64,
            cdnBaseUrl,
            "image",
            img.media.full_url,
          )
        : await downloadPlainCdnBuffer(
            img.media.encrypt_query_param ?? "",
            cdnBaseUrl,
            "image-plain",
            img.media.full_url,
          );
      const saved = await saveMediaBuffer(buf, mediaDir, undefined, "inbound");
      result.imagePath = saved.path;
    } catch (err) {
      logger.error(`image download failed: ${safeError(err)}`);
    }
  } else if (item.type === MessageItemType.VOICE) {
    const voice = item.voice_item;
    if ((!voice?.media?.encrypt_query_param && !voice?.media?.full_url) || !voice?.media?.aes_key)
      return result;
    try {
      const silkBuf = await downloadAndDecryptBuffer(
        voice.media.encrypt_query_param ?? "",
        voice.media.aes_key,
        cdnBaseUrl,
        "voice",
        voice.media.full_url,
      );
      const wavBuf = await silkToWav(silkBuf);
      if (wavBuf) {
        const saved = await saveMediaBuffer(wavBuf, mediaDir, "audio/wav", "inbound");
        result.voicePath = saved.path;
        result.voiceMediaType = "audio/wav";
      } else {
        const saved = await saveMediaBuffer(silkBuf, mediaDir, "audio/silk", "inbound");
        result.voicePath = saved.path;
        result.voiceMediaType = "audio/silk";
      }
    } catch (err) {
      logger.error(`voice download failed: ${safeError(err)}`);
    }
  } else if (item.type === MessageItemType.FILE) {
    const fileItem = item.file_item;
    if ((!fileItem?.media?.encrypt_query_param && !fileItem?.media?.full_url) || !fileItem?.media?.aes_key)
      return result;
    try {
      const buf = await downloadAndDecryptBuffer(
        fileItem.media.encrypt_query_param ?? "",
        fileItem.media.aes_key,
        cdnBaseUrl,
        "file",
        fileItem.media.full_url,
      );
      const mime = getMimeFromFilename(fileItem.file_name ?? "file.bin");
      const saved = await saveMediaBuffer(
        buf,
        mediaDir,
        mime,
        "inbound",
        fileItem.file_name ?? undefined,
      );
      result.filePath = saved.path;
      result.fileMediaType = mime;
    } catch (err) {
      logger.error(`file download failed: ${safeError(err)}`);
    }
  } else if (item.type === MessageItemType.VIDEO) {
    const videoItem = item.video_item;
    if ((!videoItem?.media?.encrypt_query_param && !videoItem?.media?.full_url) || !videoItem?.media?.aes_key)
      return result;
    try {
      const buf = await downloadAndDecryptBuffer(
        videoItem.media.encrypt_query_param ?? "",
        videoItem.media.aes_key,
        cdnBaseUrl,
        "video",
        videoItem.media.full_url,
      );
      const saved = await saveMediaBuffer(buf, mediaDir, "video/mp4", "inbound");
      result.videoPath = saved.path;
    } catch (err) {
      logger.error(`video download failed: ${safeError(err)}`);
    }
  }

  return result;
}
