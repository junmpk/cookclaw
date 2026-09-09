/**
 * HTTP API 服务器
 *
 * 提供 REST API 供 CookClaw 调用，实现消息发送、状态查询等功能。
 */

import express, { type Request, type Response, type NextFunction } from 'express';
import { createLogger, errorType } from './logger.js';
import { BaileysWhatsAppClient } from './baileys-client.js';
import type { SendMessageRequest, WebhookConfig } from './types.js';

const logger = createLogger('http-server');

/**
 * 创建 HTTP API 服务器
 */
export function createHttpServer(client: BaileysWhatsAppClient): express.Application {
  const app = express();

  // ─── 中间件 ─────────────────────────────────────────────────────
  app.use(express.json({ limit: '50mb' }));

  // 请求日志
  app.use((req: Request, _res: Response, next: NextFunction) => {
    logger.info(`${req.method} ${req.path}`);
    next();
  });

  // API Token 认证
  const apiToken = process.env.API_TOKEN;
  if (apiToken) {
    app.use('/api', (req: Request, res: Response, next: NextFunction) => {
      const auth = req.headers.authorization;
      const token = auth?.replace('Bearer ', '') || req.query.token;

      if (token !== apiToken) {
        res.status(401).json({ error: '未授权：无效的 API Token' });
        return;
      }
      next();
    });
  }

  // ─── API 路由 ───────────────────────────────────────────────────

  /**
   * GET /health — 健康检查
   */
  app.get('/health', (_req: Request, res: Response) => {
    res.json({ status: 'ok', service: 'cookclaw-whatsapp' });
  });

  /**
   * GET /api/status — 服务状态
   */
  app.get('/api/status', (_req: Request, res: Response) => {
    res.json(client.getStatus());
  });

  /**
   * GET /api/qr — 获取 QR 码
   */
  app.get('/api/qr', (_req: Request, res: Response) => {
    const qr = client.getQRCode();
    if (qr) {
      res.json({ qr, message: '请使用 WhatsApp 扫描此 QR 码' });
    } else {
      const status = client.getStatus();
      if (status.connected) {
        res.json({ connected: true, message: '已连接，无需扫码' });
      } else {
        res.json({ qr: null, message: 'QR 码尚未生成，请等待...' });
      }
    }
  });

  /**
   * POST /api/login/qr/start — 清理当前会话并生成新的登录二维码
   */
  app.post('/api/login/qr/start', async (_req: Request, res: Response) => {
    await client.startQrLogin();
    res.json({ success: true, message: '正在生成 WhatsApp 登录二维码' });
  });

  /**
   * POST /api/send — 发送消息
   *
   * Body: { to: string, text?: string, mediaUrl?: string, type?: string, quotedMessageId?: string }
   */
  app.post('/api/send', async (req: Request, res: Response) => {
    const { to, text, mediaUrl, type, quotedMessageId } = req.body as SendMessageRequest;

    if (!to) {
      res.status(400).json({ error: '缺少必填参数: to' });
      return;
    }

    if (!text && !mediaUrl) {
      res.status(400).json({ error: '必须提供 text 或 mediaUrl' });
      return;
    }

    const result = await client.sendMessage({
      to,
      text,
      mediaUrl,
      type: type as any,
      quotedMessageId,
    });

    if (result.success) {
      res.json(result);
    } else {
      res.status(500).json(result);
    }
  });

  /**
   * POST /api/read — 发送已读回执
   *
   * Body: { jid: string, messageIds: string[] }
   */
  app.post('/api/read', async (req: Request, res: Response) => {
    const { jid, messageIds } = req.body;
    if (!jid || !messageIds) {
      res.status(400).json({ error: '缺少参数: jid, messageIds' });
      return;
    }
    await client.sendReadReceipt(jid, messageIds);
    res.json({ success: true });
  });

  /**
   * POST /api/react — 发送 emoji 反应
   *
   * Body: { jid: string, messageId: string, emoji: string }
   */
  app.post('/api/react', async (req: Request, res: Response) => {
    const { jid, messageId, emoji } = req.body;
    if (!jid || !messageId || !emoji) {
      res.status(400).json({ error: '缺少参数: jid, messageId, emoji' });
      return;
    }
    await client.sendReaction(jid, messageId, emoji);
    res.json({ success: true });
  });

  /**
   * POST /api/webhook — 配置 Webhook
   *
   * Body: { url: string, secret?: string }
   */
  app.post('/api/webhook', (req: Request, res: Response) => {
    const { url, secret } = req.body as WebhookConfig;
    if (!url) {
      res.status(400).json({ error: '缺少参数: url' });
      return;
    }
    client.setWebhook({ url, secret });
    res.json({ success: true, message: 'Webhook 已配置' });
  });

  // ─── 错误处理 ───────────────────────────────────────────────────
  app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
    logger.error(`HTTP 错误: error_type=${errorType(err)}`);
    res.status(500).json({ error: '内部服务器错误' });
  });

  return app;
}
