/**
 * 消息发送：文本、图片、视频、文件
 */
import path from "node:path";

import { sendMessage as sendMessageApi } from "./api.js";
import type { WeixinApiOptions } from "./api.js";
import { uploadFileToWeixin, uploadVideoToWeixin, uploadFileAttachmentToWeixin } from "./cdn.js";
import type { UploadedFileInfo } from "./cdn.js";
import { logger } from "./logger.js";
import { StreamingMarkdownFilter } from "./markdown-filter.js";
import { fingerprint, generateId, getMimeFromFilename } from "./utils.js";
import type { MessageItem, SendMessageReq } from "./types.js";
import { MessageItemType, MessageState, MessageType } from "./types.js";

// ---------------------------------------------------------------------------
// 消息构建
// ---------------------------------------------------------------------------

function generateClientId(): string {
  return generateId("weixinbot");
}

/** 构建文本消息请求 */
function buildTextMessageReq(params: {
  to: string;
  text: string;
  contextToken?: string;
  clientId: string;
}): SendMessageReq {
  const { to, text, contextToken, clientId } = params;
  const item_list: MessageItem[] = text
    ? [{ type: MessageItemType.TEXT, text_item: { text } }]
    : [];
  return {
    msg: {
      from_user_id: "",
      to_user_id: to,
      client_id: clientId,
      message_type: MessageType.BOT,
      message_state: MessageState.FINISH,
      item_list: item_list.length ? item_list : undefined,
      context_token: contextToken ?? undefined,
    },
  };
}

// ---------------------------------------------------------------------------
// 发送文本消息
// ---------------------------------------------------------------------------

/** 发送纯文本消息 */
export async function sendTextMessage(params: {
  to: string;
  text: string;
  opts: WeixinApiOptions & { contextToken?: string };
}): Promise<{ messageId: string }> {
  const { to, text, opts } = params;

  // Markdown 过滤
  const f = new StreamingMarkdownFilter();
  const filteredText = f.feed(text) + f.flush();

  const clientId = generateClientId();
  const req = buildTextMessageReq({
    to,
    text: filteredText,
    contextToken: opts.contextToken,
    clientId,
  });

  await sendMessageApi({ baseUrl: opts.baseUrl, token: opts.token, body: req, botAgent: opts.botAgent });
  logger.info(`sendTextMessage: to=${fingerprint(to)} client=${fingerprint(clientId)}`);
  return { messageId: clientId };
}

// ---------------------------------------------------------------------------
// 发送媒体消息（内部通用）
// ---------------------------------------------------------------------------

async function sendMediaItems(params: {
  to: string;
  text: string;
  mediaItem: MessageItem;
  opts: WeixinApiOptions & { contextToken?: string };
  label: string;
}): Promise<{ messageId: string }> {
  const { to, text, mediaItem, opts, label } = params;
  const items: MessageItem[] = [];
  if (text) {
    items.push({ type: MessageItemType.TEXT, text_item: { text } });
  }
  items.push(mediaItem);

  let lastClientId = "";
  for (const item of items) {
    lastClientId = generateClientId();
    const req: SendMessageReq = {
      msg: {
        from_user_id: "",
        to_user_id: to,
        client_id: lastClientId,
        message_type: MessageType.BOT,
        message_state: MessageState.FINISH,
        item_list: [item],
        context_token: opts.contextToken ?? undefined,
      },
    };
    await sendMessageApi({ baseUrl: opts.baseUrl, token: opts.token, body: req, botAgent: opts.botAgent });
  }

  logger.info(`${label}: success to=${fingerprint(to)} client=${fingerprint(lastClientId)}`);
  return { messageId: lastClientId };
}

// ---------------------------------------------------------------------------
// 发送图片消息
// ---------------------------------------------------------------------------

/** 发送图片消息（使用已上传的文件信息） */
export async function sendImageMessage(params: {
  to: string;
  text: string;
  uploaded: UploadedFileInfo;
  opts: WeixinApiOptions & { contextToken?: string };
}): Promise<{ messageId: string }> {
  const { to, text, uploaded, opts } = params;
  const imageItem: MessageItem = {
    type: MessageItemType.IMAGE,
    image_item: {
      media: {
        encrypt_query_param: uploaded.downloadEncryptedQueryParam,
        aes_key: Buffer.from(uploaded.aeskey).toString("base64"),
        encrypt_type: 1,
      },
      mid_size: uploaded.fileSizeCiphertext,
    },
  };
  return sendMediaItems({ to, text, mediaItem: imageItem, opts, label: "sendImageMessage" });
}

// ---------------------------------------------------------------------------
// 发送视频消息
// ---------------------------------------------------------------------------

/** 发送视频消息（使用已上传的文件信息） */
export async function sendVideoMessage(params: {
  to: string;
  text: string;
  uploaded: UploadedFileInfo;
  opts: WeixinApiOptions & { contextToken?: string };
}): Promise<{ messageId: string }> {
  const { to, text, uploaded, opts } = params;
  const videoItem: MessageItem = {
    type: MessageItemType.VIDEO,
    video_item: {
      media: {
        encrypt_query_param: uploaded.downloadEncryptedQueryParam,
        aes_key: Buffer.from(uploaded.aeskey).toString("base64"),
      },
      video_size: uploaded.fileSizeCiphertext,
    },
  };
  return sendMediaItems({ to, text, mediaItem: videoItem, opts, label: "sendVideoMessage" });
}

// ---------------------------------------------------------------------------
// 发送文件消息
// ---------------------------------------------------------------------------

/** 发送文件消息（使用已上传的文件信息） */
export async function sendFileMessage(params: {
  to: string;
  text: string;
  fileName: string;
  uploaded: UploadedFileInfo;
  opts: WeixinApiOptions & { contextToken?: string };
}): Promise<{ messageId: string }> {
  const { to, text, fileName, uploaded, opts } = params;
  const fileItem: MessageItem = {
    type: MessageItemType.FILE,
    file_item: {
      media: {
        encrypt_query_param: uploaded.downloadEncryptedQueryParam,
        aes_key: Buffer.from(uploaded.aeskey).toString("base64"),
      },
      file_name: fileName,
    },
  };
  return sendMediaItems({ to, text, mediaItem: fileItem, opts, label: "sendFileMessage" });
}

// ---------------------------------------------------------------------------
// 发送媒体文件（自动判断类型）
// ---------------------------------------------------------------------------

/** 发送媒体文件（自动根据 MIME 类型路由到图片/视频/文件上传） */
export async function sendMediaFile(params: {
  filePath: string;
  to: string;
  text: string;
  opts: WeixinApiOptions & { contextToken?: string };
  cdnBaseUrl: string;
}): Promise<{ messageId: string }> {
  const { filePath, to, text, opts, cdnBaseUrl } = params;
  const mime = getMimeFromFilename(filePath);
  const uploadOpts: WeixinApiOptions = { baseUrl: opts.baseUrl, token: opts.token, botAgent: opts.botAgent };

  if (mime.startsWith("video/")) {
    const uploaded = await uploadVideoToWeixin({ filePath, toUserId: to, opts: uploadOpts, cdnBaseUrl });
    return sendVideoMessage({ to, text, uploaded, opts });
  }

  if (mime.startsWith("image/")) {
    const uploaded = await uploadFileToWeixin({ filePath, toUserId: to, opts: uploadOpts, cdnBaseUrl });
    return sendImageMessage({ to, text, uploaded, opts });
  }

  const fileName = path.basename(filePath);
  const uploaded = await uploadFileAttachmentToWeixin({ filePath, fileName, toUserId: to, opts: uploadOpts, cdnBaseUrl });
  return sendFileMessage({ to, text, fileName, uploaded, opts });
}
