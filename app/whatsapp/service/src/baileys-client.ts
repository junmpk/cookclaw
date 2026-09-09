/**
 * Baileys WhatsApp 客户端
 *
 * 封装 @whiskeysockets/baileys，实现 WhatsApp 连接管理、消息收发、
 * 凭据持久化、断线重连等功能。
 */

import {
  default as makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
  type WASocket,
  type ConnectionState as BaileysConnectionState,
  type WAMessage,
  type proto,
  type GroupMetadata,
  getContentType,
  downloadMediaMessage,
} from '@whiskeysockets/baileys';
import { Boom } from '@hapi/boom';
import QRCode from 'qrcode-terminal';
import fs from 'fs';
import path from 'path';
import axios from 'axios';
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { HttpsProxyAgent } = require('https-proxy-agent');
import {
  createLogger,
  errorType,
  logFingerprint,
  redactNetworkEndpoint,
} from './logger.js';
import type {
  ConnectionState,
  SendMessageRequest,
  SendMessageResponse,
  WebhookMessageEvent,
  WebhookConfig,
  ServiceStatus,
} from './types.js';

const logger = createLogger('baileys-client');

export class BaileysWhatsAppClient {
  private sock: WASocket | null = null;
  private connectionState: ConnectionState = 'disconnected';
  private phoneNumber: string | undefined;
  private lastConnectedAt: Date | undefined;
  private startTime: Date = new Date();
  private webhookConfig: WebhookConfig | null = null;
  private authFolder: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private isShuttingDown = false;
  private qrCodeData: string | null = null;
  private latestVersion: string | null = null;
  private connectionGeneration = 0;
  private loginResetPromise: Promise<void> | null = null;

  constructor(authFolder?: string) {
    this.authFolder = authFolder || process.env.AUTH_FOLDER || './auth';
  }

  /**
   * 启动 WhatsApp 客户端
   */
  async start(): Promise<void> {
    try {
      const { version } = await fetchLatestBaileysVersion();
      this.latestVersion = version.join('.');
      logger.info(`Baileys 版本: ${this.latestVersion}`);
    } catch {
      logger.warn('无法获取最新 Baileys 版本，使用默认版本');
    }

    await this._connect();
  }

  /**
   * 停止 WhatsApp 客户端
   */
  async stop(): Promise<void> {
    this.isShuttingDown = true;
    ++this.connectionGeneration;
    if (this.sock) {
      this.sock.end(undefined);
      this.sock = null;
    }
    this.qrCodeData = null;
    this.connectionState = 'disconnected';
    logger.info('WhatsApp 客户端已停止');
  }

  /**
   * 获取 QR 码数据
   */
  getQRCode(): string | null {
    return this.qrCodeData;
  }

  /**
   * 退出当前账号、清理本地凭据并启动一轮新的扫码登录。
   *
   * 页面上的“生成二维码 / 更换账号”会调用此方法。使用连接代次隔离旧
   * socket 的异步事件，避免旧连接关闭事件覆盖新连接状态。
   */
  async startQrLogin(): Promise<void> {
    if (this.loginResetPromise) {
      return this.loginResetPromise;
    }

    this.loginResetPromise = this._resetAndConnect();
    try {
      await this.loginResetPromise;
    } finally {
      this.loginResetPromise = null;
    }
  }

  /**
   * 获取服务状态
   */
  getStatus(): ServiceStatus {
    return {
      running: this.sock !== null,
      connected: this.connectionState === 'connected',
      connectionState: this.connectionState,
      phoneNumber: this.phoneNumber,
      lastConnectedAt: this.lastConnectedAt?.toISOString(),
      uptime: Math.floor((Date.now() - this.startTime.getTime()) / 1000),
      webhookUrl: this.webhookConfig?.url,
    };
  }

  /**
   * 配置 Webhook
   */
  setWebhook(config: WebhookConfig): void {
    this.webhookConfig = config;
    logger.info(`Webhook 已配置: ${redactNetworkEndpoint(config.url)}`);
  }

  /**
   * 发送文本消息
   */
  async sendMessage(req: SendMessageRequest): Promise<SendMessageResponse> {
    if (!this.sock || this.connectionState !== 'connected') {
      return { success: false, error: 'WhatsApp 未连接' };
    }

    try {
      const jid = this._resolveJid(req.to);
      let sent: proto.IWebMessageInfo | undefined;

      if (req.mediaUrl) {
        // 发送媒体消息
        sent = await this._sendMediaMessage(jid, req);
      } else if (req.text) {
        // 发送文本消息
        sent = await this.sock.sendMessage(jid, { text: req.text });
      } else {
        return { success: false, error: '必须提供 text 或 mediaUrl' };
      }

      const messageId = sent?.key?.id;
      logger.info(
        `消息已发送 → recipient=${logFingerprint(req.to)} message=${logFingerprint(messageId)}`
      );
      return { success: true, messageId: messageId || undefined };
    } catch (error: any) {
      logger.error(`发送消息失败: error_type=${errorType(error)}`);
      return { success: false, error: error.message };
    }
  }

  /**
   * 发送已读回执
   */
  async sendReadReceipt(jid: string, messageIds: string[]): Promise<void> {
    if (!this.sock || this.connectionState !== 'connected') return;
    try {
      const resolvedJid = this._resolveJid(jid);
      for (const id of messageIds) {
        await this.sock.readMessages([{ remoteJid: resolvedJid, id } as any]);
      }
    } catch (error: unknown) {
      logger.error(`发送已读回执失败: error_type=${errorType(error)}`);
    }
  }

  /**
   * 发送 emoji 反应
   */
  async sendReaction(jid: string, messageId: string, emoji: string): Promise<void> {
    if (!this.sock || this.connectionState !== 'connected') return;
    try {
      const resolvedJid = this._resolveJid(jid);
      await this.sock.sendMessage(resolvedJid, {
        react: { text: emoji, key: { remoteJid: resolvedJid, id: messageId } },
      });
    } catch (error: unknown) {
      logger.error(`发送反应失败: error_type=${errorType(error)}`);
    }
  }

  // ─── 私有方法 ──────────────────────────────────────────────────────────

  /**
   * 建立 WhatsApp 连接
   */
  private async _connect(): Promise<void> {
    if (this.isShuttingDown) return;

    const generation = ++this.connectionGeneration;

    // 确保凭据目录存在
    if (!fs.existsSync(this.authFolder)) {
      fs.mkdirSync(this.authFolder, { recursive: true });
    }

    const { state, saveCreds } = await useMultiFileAuthState(this.authFolder);

    const { version } = await fetchLatestBaileysVersion().catch(() => ({
      version: [2, 3000, 1023716180] as [number, number, number],
    }));

    // 代理配置
    const proxyUrl = process.env.PROXY_URL || process.env.HTTPS_PROXY || process.env.https_proxy;
    const agent = proxyUrl ? new HttpsProxyAgent(proxyUrl) : undefined;
    if (agent) {
      logger.info(`使用代理: ${redactNetworkEndpoint(proxyUrl)}`);
    }

    const socket = makeWASocket({
      agent,
      version,
      auth: {
        creds: state.creds,
        keys: makeCacheableSignalKeyStore(state.keys, {
          info: () => {},
          error: () => {},
          warn: () => {},
          debug: () => {},
          trace: () => {},
          child: () => ({
            info: () => {},
            error: () => {},
            warn: () => {},
            debug: () => {},
            trace: () => {},
            level: 'silent',
          } as any),
          level: 'silent',
        } as any),
      },
      printQRInTerminal: false,
      logger: {
        info: () => {},
        error: () => {},
        warn: () => {},
        debug: () => {},
        trace: () => {},
        child: () =>
          ({
            info: () => {},
            error: () => {},
            warn: () => {},
            debug: () => {},
            trace: () => {},
            level: 'silent',
          } as any),
        level: 'silent',
      } as any,
      browser: ['CookClaw WhatsApp Service', 'Chrome', '1.0.0'],
      connectTimeoutMs: 30_000,
      keepAliveIntervalMs: 25_000,
      retryRequestDelayMs: 500,
      maxMsgRetryCount: 3,
    });
    if (generation !== this.connectionGeneration || this.isShuttingDown) {
      socket.end(undefined);
      return;
    }
    this.sock = socket;

    // ─── 凭据保存 ─────────────────────────────────────────────────
    socket.ev.on('creds.update', async () => {
      if (generation === this.connectionGeneration) {
        await saveCreds();
      }
    });

    // ─── 连接状态更新 ──────────────────────────────────────────────
    socket.ev.on('connection.update', async (update) => {
      if (generation !== this.connectionGeneration) return;
      const { connection, lastDisconnect, qr } = update;

      if (qr) {
        this.qrCodeData = qr;
        logger.info('QR 码已生成，请在 WhatsApp 中扫描');
        QRCode.generate(qr, { small: true }, (qrcode) => {
          console.log('\n' + qrcode + '\n');
          console.log('📱 请打开 WhatsApp → 设置 → 关联设备 → 扫描此 QR 码\n');
        });
      }

      if (connection === 'close') {
        const statusCode = (lastDisconnect?.error as Boom)?.output?.statusCode;
        const shouldReconnect =
          statusCode !== DisconnectReason.loggedOut &&
          statusCode !== DisconnectReason.badSession &&
          !this.isShuttingDown;

        this.connectionState = shouldReconnect ? 'disconnected' : 'logged_out';
        this.sock = null;

        if (shouldReconnect) {
          this.reconnectAttempts++;
          const delay = Math.min(1000 * 2 ** this.reconnectAttempts, 30_000);
          logger.warn(
            `连接断开 (code: ${statusCode})，${delay / 1000}s 后重连 (第 ${this.reconnectAttempts} 次)`
          );
          await this._sleep(delay);
          if (generation === this.connectionGeneration && !this.isShuttingDown) {
            void this._connect();
          }
        } else {
          logger.error(`连接断开 (code: ${statusCode})，需要重新扫码登录`);
          // 通知 Webhook
          await this._fireWebhook({
            event: 'connection.update',
            timestamp: Date.now(),
            data: { connectionState: 'logged_out' },
          });
        }
      } else if (connection === 'open') {
        this.connectionState = 'connected';
        this.lastConnectedAt = new Date();
        this.reconnectAttempts = 0;
        this.qrCodeData = null;

        // 获取登录手机号
        if (socket.user?.id) {
          this.phoneNumber = '+' + socket.user.id.split('@')[0].split(':')[0];
        }

        logger.info(`✅ WhatsApp 已连接 (account=${logFingerprint(this.phoneNumber)})`);

        // 通知 Webhook
        await this._fireWebhook({
          event: 'connection.update',
          timestamp: Date.now(),
          data: { connectionState: 'connected' },
        });
      } else if (connection === 'connecting') {
        this.connectionState = 'connecting';
        logger.info('正在连接 WhatsApp...');
      }
    });

    // ─── 接收消息 ──────────────────────────────────────────────────
    socket.ev.on('messages.upsert', async ({ messages, type }) => {
      if (generation !== this.connectionGeneration) return;
      if (type !== 'notify') return;

      for (const msg of messages) {
        try {
          await this._handleIncomingMessage(msg);
        } catch (error: unknown) {
          logger.error(`处理消息失败: error_type=${errorType(error)}`);
        }
      }
    });

    // ─── 消息反应 ──────────────────────────────────────────────────
    socket.ev.on('messages.reaction', async (reactions) => {
      if (generation !== this.connectionGeneration) return;
      for (const reaction of reactions) {
        await this._fireWebhook({
          event: 'message.reaction',
          timestamp: Date.now(),
          data: {
            messageId: reaction.key.id || undefined,
            from: reaction.key.remoteJid || undefined,
            reaction: (reaction as any).text || undefined,
          },
        });
      }
    });
  }

  private async _resetAndConnect(): Promise<void> {
    logger.info('正在重置 WhatsApp 登录会话...');
    this.isShuttingDown = true;
    ++this.connectionGeneration;

    const oldSocket = this.sock;
    this.sock = null;
    this.qrCodeData = null;
    this.phoneNumber = undefined;
    this.lastConnectedAt = undefined;
    this.reconnectAttempts = 0;

    if (oldSocket) {
      try {
        await Promise.race([
          oldSocket.logout(),
          this._sleep(5_000),
        ]);
      } catch (error: unknown) {
        logger.warn(`退出旧 WhatsApp 会话失败，将直接清理本地凭据: error_type=${errorType(error)}`);
      }
      oldSocket.end(undefined);
    }

    fs.rmSync(this.authFolder, { recursive: true, force: true });
    fs.mkdirSync(this.authFolder, { recursive: true });
    this.connectionState = 'connecting';
    this.isShuttingDown = false;
    await this._connect();
  }

  /**
   * 处理入站消息
   */
  private async _handleIncomingMessage(msg: WAMessage): Promise<void> {
    // 忽略自己发送的消息
    if (msg.key.fromMe) return;

    const jid = msg.key.remoteJid;
    if (!jid) return;

    const isGroup = jid.endsWith('@g.us');
    const chatType = isGroup ? 'group' : 'dm';
    const fromNumber = this._jidToPhone(jid);
    const participant = msg.key.participant
      ? this._jidToPhone(msg.key.participant)
      : undefined;

    // 解析消息内容
    const content = msg.message;
    if (!content) return;

    const contentType = getContentType(content);
    let text = '';
    const mediaUrls: string[] = [];
    const mediaTypes: string[] = [];

    // 提取文本
    if (content.conversation) {
      text = content.conversation;
    } else if (content.extendedTextMessage?.text) {
      text = content.extendedTextMessage.text;
    } else if (content.imageMessage) {
      text = content.imageMessage.caption || '';
      mediaTypes.push('image');
    } else if (content.videoMessage) {
      text = content.videoMessage.caption || '';
      mediaTypes.push('video');
    } else if (content.audioMessage) {
      mediaTypes.push('audio');
    } else if (content.documentMessage) {
      text = content.documentMessage.caption || '';
      mediaTypes.push('document');
    } else if (content.stickerMessage) {
      mediaTypes.push('sticker');
    } else if (content.contactMessage) {
      text = `[联系人] ${content.contactMessage.displayName || ''}`;
      mediaTypes.push('contact');
    } else if (content.locationMessage) {
      text = `[位置] ${content.locationMessage.degreesLatitude},${content.locationMessage.degreesLongitude}`;
      mediaTypes.push('location');
    }

    // 忽略空消息（协议消息等）
    if (!text && mediaTypes.length === 0) return;

    // 下载媒体并保存为本地文件（如果有媒体）
    if (
      contentType &&
      ['imageMessage', 'videoMessage', 'audioMessage', 'documentMessage', 'stickerMessage'].includes(
        contentType
      )
    ) {
      try {
        const mediaBuffer = await downloadMediaMessage(msg, 'buffer', {});
        if (mediaBuffer) {
          const mediaDir = path.join(this.authFolder, 'media');
          if (!fs.existsSync(mediaDir)) {
            fs.mkdirSync(mediaDir, { recursive: true });
          }
          const ext = this._getMediaExtension(contentType);
          const filename = `${msg.key.id || Date.now()}.${ext}`;
          const filepath = path.join(mediaDir, filename);
          fs.writeFileSync(filepath, mediaBuffer);
          mediaUrls.push(filepath);
        }
      } catch (error: unknown) {
        logger.warn(`下载媒体失败: error_type=${errorType(error)}`);
      }
    }

    // 引用消息
    const quotedMessageId =
      content.extendedTextMessage?.contextInfo?.stanzaId || undefined;

    const webhookEvent: WebhookMessageEvent = {
      event: 'message',
      timestamp: msg.messageTimestamp ? Number(msg.messageTimestamp) * 1000 : Date.now(),
      data: {
        messageId: msg.key.id || undefined,
        from: jid,
        fromNumber: isGroup ? participant : fromNumber,
        chatType,
        text,
        mediaUrls: mediaUrls.length > 0 ? mediaUrls : undefined,
        mediaTypes: mediaTypes.length > 0 ? mediaTypes : undefined,
        quotedMessageId,
        groupJid: isGroup ? jid : undefined,
        participant: isGroup ? msg.key.participant || undefined : undefined,
      },
    };

    logger.info(
      `📨 收到消息: sender=${logFingerprint(fromNumber || jid)} text_chars=${text.length}`
    );

    // 发送已读回执
    if (jid && msg.key.id) {
      await this.sendReadReceipt(jid, [msg.key.id]);
    }

    // 转发到 Webhook
    await this._fireWebhook(webhookEvent);
  }

  /**
   * 发送媒体消息
   */
  private async _sendMediaMessage(
    jid: string,
    req: SendMessageRequest
  ): Promise<proto.IWebMessageInfo | undefined> {
    if (!this.sock) throw new Error('未连接');

    // 下载媒体
    const response = await axios.get(req.mediaUrl!, {
      responseType: 'arraybuffer',
      timeout: 60_000,
    });

    const buffer = Buffer.from(response.data);
    const ct = String(response.headers['content-type'] || 'application/octet-stream');

    // 根据媒体类型发送
    if (ct.startsWith('image/')) {
      return this.sock.sendMessage(jid, {
        image: buffer,
        caption: req.text,
      });
    } else if (ct.startsWith('video/')) {
      return this.sock.sendMessage(jid, {
        video: buffer,
        caption: req.text,
      });
    } else if (ct.startsWith('audio/')) {
      return this.sock.sendMessage(jid, {
        audio: buffer,
        mimetype: ct,
      });
    } else {
      // 作为文档发送
      const filename = req.mediaUrl!.split('/').pop() || 'file';
      return this.sock.sendMessage(jid, {
        document: buffer,
        fileName: filename,
        mimetype: ct,
        caption: req.text,
      });
    }
  }

  /**
   * 解析 JID（手机号 → WhatsApp JID）
   */
  private _resolveJid(to: string): string {
    // 已经是 JID 格式
    if (to.includes('@')) return to;

    // E.164 格式 → JID
    const number = to.replace(/[^0-9]/g, '');
    // 判断是否是群组 ID（纯数字较长且不含 +）
    if (to.includes('-') && !to.startsWith('+')) {
      return `${number}@g.us`;
    }
    return `${number}@s.whatsapp.net`;
  }

  /**
   * JID → 手机号
   */
  private _jidToPhone(jid: string): string {
    const number = jid.split('@')[0].split(':')[0];
    return '+' + number;
  }

  /**
   * 获取引用消息对象
   */
  private async _getMessageForQuote(
    messageId: string
  ): Promise<proto.IContextInfo['quotedMessage'] | undefined> {
    // Baileys 需要完整的消息对象来引用
    // 这里简化处理，返回基本的引用信息
    return undefined;
  }

  /**
   * 获取媒体文件扩展名
   */
  private _getMediaExtension(contentType: string): string {
    const map: Record<string, string> = {
      imageMessage: 'jpg',
      videoMessage: 'mp4',
      audioMessage: 'ogg',
      documentMessage: 'bin',
      stickerMessage: 'webp',
    };
    return map[contentType] || 'bin';
  }

  /**
   * 触发 Webhook
   */
  private async _fireWebhook(event: WebhookMessageEvent): Promise<void> {
    if (!this.webhookConfig?.url) return;

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (this.webhookConfig.secret) {
        headers['X-Webhook-Secret'] = this.webhookConfig.secret;
      }

      await axios.post(this.webhookConfig.url, event, {
        headers,
        timeout: 10_000,
      });
      logger.debug(`Webhook 已发送: ${event.event}`);
    } catch (error: unknown) {
      logger.error(`Webhook 发送失败: error_type=${errorType(error)}`);
    }
  }

  private _sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
