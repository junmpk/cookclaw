/**
 * 微信 iLink 协议类型定义
 * 从 @tencent-weixin/openclaw-weixin 提取，去除 OpenClaw 依赖
 */

// ---------------------------------------------------------------------------
// 通用请求元数据
// ---------------------------------------------------------------------------

/** 每次请求携带的基础信息 */
export interface BaseInfo {
  channel_version?: string;
  /** 自声明上游应用标识，类似 HTTP User-Agent */
  bot_agent?: string;
}

// ---------------------------------------------------------------------------
// 媒体上传类型
// ---------------------------------------------------------------------------

/** proto: UploadMediaType */
export const UploadMediaType = {
  IMAGE: 1,
  VIDEO: 2,
  FILE: 3,
  VOICE: 4,
} as const;

// ---------------------------------------------------------------------------
// 上传 URL 请求/响应
// ---------------------------------------------------------------------------

export interface GetUploadUrlReq {
  filekey?: string;
  media_type?: number;
  to_user_id?: string;
  rawsize?: number;
  rawfilemd5?: string;
  filesize?: number;
  thumb_rawsize?: number;
  thumb_rawfilemd5?: string;
  thumb_filesize?: number;
  no_need_thumb?: boolean;
  aeskey?: string;
}

export interface GetUploadUrlResp {
  upload_param?: string;
  thumb_upload_param?: string;
  upload_full_url?: string;
}

// ---------------------------------------------------------------------------
// 消息类型常量
// ---------------------------------------------------------------------------

export const MessageType = {
  NONE: 0,
  USER: 1,
  BOT: 2,
} as const;

export const MessageItemType = {
  NONE: 0,
  TEXT: 1,
  IMAGE: 2,
  VOICE: 3,
  FILE: 4,
  VIDEO: 5,
} as const;

export const MessageState = {
  NEW: 0,
  GENERATING: 1,
  FINISH: 2,
} as const;

// ---------------------------------------------------------------------------
// 消息项类型
// ---------------------------------------------------------------------------

export interface TextItem {
  text?: string;
}

/** CDN 媒体引用；aes_key 为 base64 编码 */
export interface CDNMedia {
  encrypt_query_param?: string;
  aes_key?: string;
  encrypt_type?: number;
  full_url?: string;
}

export interface ImageItem {
  media?: CDNMedia;
  thumb_media?: CDNMedia;
  /** AES-128 密钥的 hex 字符串（16 字节） */
  aeskey?: string;
  url?: string;
  mid_size?: number;
  thumb_size?: number;
  thumb_height?: number;
  thumb_width?: number;
  hd_size?: number;
}

export interface VoiceItem {
  media?: CDNMedia;
  /** 语音编码类型：1=pcm 2=adpcm 3=feature 4=speex 5=amr 6=silk 7=mp3 8=ogg-speex */
  encode_type?: number;
  bits_per_sample?: number;
  sample_rate?: number;
  playtime?: number;
  /** 语音转文字内容 */
  text?: string;
}

export interface FileItem {
  media?: CDNMedia;
  file_name?: string;
  md5?: string;
  len?: string;
}

export interface VideoItem {
  media?: CDNMedia;
  video_size?: number;
  play_length?: number;
  video_md5?: string;
  thumb_media?: CDNMedia;
  thumb_size?: number;
  thumb_height?: number;
  thumb_width?: number;
}

export interface RefMessage {
  message_item?: MessageItem;
  title?: string;
}

export interface MessageItem {
  type?: number;
  create_time_ms?: number;
  update_time_ms?: number;
  is_completed?: boolean;
  msg_id?: string;
  ref_msg?: RefMessage;
  text_item?: TextItem;
  image_item?: ImageItem;
  voice_item?: VoiceItem;
  file_item?: FileItem;
  video_item?: VideoItem;
}

// ---------------------------------------------------------------------------
// 统一消息结构
// ---------------------------------------------------------------------------

/** 统一消息（proto: WeixinMessage） */
export interface WeixinMessage {
  seq?: number;
  message_id?: number;
  from_user_id?: string;
  to_user_id?: string;
  client_id?: string;
  create_time_ms?: number;
  update_time_ms?: number;
  delete_time_ms?: number;
  session_id?: string;
  group_id?: string;
  message_type?: number;
  message_state?: number;
  item_list?: MessageItem[];
  context_token?: string;
}

// ---------------------------------------------------------------------------
// getUpdates 请求/响应
// ---------------------------------------------------------------------------

export interface GetUpdatesReq {
  sync_buf?: string;
  get_updates_buf?: string;
}

export interface GetUpdatesResp {
  ret?: number;
  errcode?: number;
  errmsg?: string;
  msgs?: WeixinMessage[];
  sync_buf?: string;
  get_updates_buf?: string;
  longpolling_timeout_ms?: number;
}

// ---------------------------------------------------------------------------
// sendMessage 请求/响应
// ---------------------------------------------------------------------------

export interface SendMessageReq {
  msg?: WeixinMessage;
}

export interface SendMessageResp {}

// ---------------------------------------------------------------------------
// typing 相关
// ---------------------------------------------------------------------------

export const TypingStatus = {
  TYPING: 1,
  CANCEL: 2,
} as const;

export interface SendTypingReq {
  ilink_user_id?: string;
  typing_ticket?: string;
  status?: number;
}

export interface SendTypingResp {
  ret?: number;
  errmsg?: string;
}

// ---------------------------------------------------------------------------
// getConfig 响应
// ---------------------------------------------------------------------------

export interface GetConfigResp {
  ret?: number;
  errmsg?: string;
  typing_ticket?: string;
}

// ---------------------------------------------------------------------------
// notifyStart / notifyStop
// ---------------------------------------------------------------------------

export interface NotifyStopResp {
  ret?: number;
  errcode?: number;
  errmsg?: string;
}

export interface NotifyStartResp {
  ret?: number;
  errcode?: number;
  errmsg?: string;
}

// ---------------------------------------------------------------------------
// SDK 回调类型
// ---------------------------------------------------------------------------

/** 收到消息时的回调 */
export type OnMessageCallback = (message: WeixinMessage, context: MessageContext) => Promise<void> | void;

/** 消息上下文，提供回复能力 */
export interface MessageContext {
  /** 发送者 ID */
  fromUserId: string;
  /** 接收者 ID（机器人 ID） */
  toUserId: string;
  /** 会话 context_token，回复时必须携带 */
  contextToken?: string;
  /** 账号 ID */
  accountId: string;
  /** 消息文本内容 */
  text: string;
  /** 已下载的图片本地路径 */
  imagePath?: string;
  /** 已下载的语音本地路径 */
  voicePath?: string;
  /** 已下载的文件本地路径 */
  filePath?: string;
  /** 已下载的视频本地路径 */
  videoPath?: string;
  /** 语音 MIME 类型 */
  voiceMediaType?: string;
  /** 文件 MIME 类型 */
  fileMediaType?: string;
  /** 回复文本消息 */
  replyText: (text: string) => Promise<{ messageId: string }>;
  /** 回复图片消息（本地文件路径） */
  replyImage: (filePath: string, text?: string) => Promise<{ messageId: string }>;
  /** 回复视频消息（本地文件路径） */
  replyVideo: (filePath: string, text?: string) => Promise<{ messageId: string }>;
  /** 回复文件消息（本地文件路径） */
  replyFile: (filePath: string, fileName?: string, text?: string) => Promise<{ messageId: string }>;
  /** 发送 typing 状态 */
  sendTyping: () => Promise<void>;
}

/** SDK 配置 */
export interface WeixinBotConfig {
  /** API 基础地址，默认 https://ilinkai.weixin.qq.com */
  baseUrl?: string;
  /** CDN 基础地址，默认 https://novac2c.cdn.weixin.qq.com/c2c */
  cdnBaseUrl?: string;
  /** 登录后的 bot_token（可后续通过 loginWithQRCode 获取或 setToken 设置） */
  token?: string;
  /** 状态存储目录，默认 ~/.weixinbot */
  stateDir?: string;
  /** 日志级别：TRACE | DEBUG | INFO | WARN | ERROR */
  logLevel?: string;
  /** bot_agent 标识，默认 "WeixinBot" */
  botAgent?: string;
  /** 长轮询超时（毫秒），默认 35000 */
  longPollTimeoutMs?: number;
  /** 媒体保存目录，默认 {stateDir}/media */
  mediaDir?: string;
  /** 收到消息的回调 */
  onMessage: OnMessageCallback;
}
