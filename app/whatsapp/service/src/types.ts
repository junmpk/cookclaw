/**
 * WhatsApp 微服务类型定义
 */

/** 连接状态 */
export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'logged_out';

/** 消息类型 */
export type MessageType = 'text' | 'image' | 'video' | 'audio' | 'document' | 'sticker';

/** 发送消息请求 */
export interface SendMessageRequest {
  /** 目标号码（E.164 格式，如 +15551234567）或群组 JID */
  to: string;
  /** 消息文本内容 */
  text?: string;
  /** 媒体 URL（图片/视频/音频/文档） */
  mediaUrl?: string;
  /** 消息类型（默认自动检测） */
  type?: MessageType;
  /** 引用回复的消息 ID */
  quotedMessageId?: string;
}

/** 发送消息响应 */
export interface SendMessageResponse {
  success: boolean;
  messageId?: string;
  error?: string;
}

/** Webhook 消息事件 */
export interface WebhookMessageEvent {
  /** 事件类型 */
  event: 'message' | 'message.reaction' | 'message.read' | 'connection.update';
  /** 时间戳 */
  timestamp: number;
  /** 消息数据 */
  data: {
    /** 消息 ID */
    messageId?: string;
    /** 发送者 JID */
    from?: string;
    /** 发送者手机号（E.164） */
    fromNumber?: string;
    /** 聊天类型 */
    chatType?: 'dm' | 'group';
    /** 消息文本 */
    text?: string;
    /** 媒体 URL 列表 */
    mediaUrls?: string[];
    /** 媒体类型 */
    mediaTypes?: string[];
    /** 引用消息 ID */
    quotedMessageId?: string;
    /** 群组 JID（如果是群组消息） */
    groupJid?: string;
    /** 群组中发送者 JID */
    participant?: string;
    /** 反应 emoji */
    reaction?: string;
    /** 连接状态 */
    connectionState?: ConnectionState;
  };
}

/** 服务状态 */
export interface ServiceStatus {
  /** 服务运行状态 */
  running: boolean;
  /** WhatsApp 连接状态 */
  connected: boolean;
  /** 连接状态详情 */
  connectionState: ConnectionState;
  /** 当前登录的手机号 */
  phoneNumber?: string;
  /** 最后连接时间 */
  lastConnectedAt?: string;
  /** 运行时长（秒） */
  uptime: number;
  /** Webhook URL */
  webhookUrl?: string;
}

/** Webhook 配置 */
export interface WebhookConfig {
  /** Webhook URL */
  url: string;
  /** Webhook 认证密钥 */
  secret?: string;
}
