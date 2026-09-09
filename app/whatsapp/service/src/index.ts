/**
 * CookClaw WhatsApp 微服务入口
 *
 * 基于 @whiskeysockets/baileys 的 WhatsApp 消息网关。
 * 提供 HTTP API 供 CookClaw 调用，支持消息收发、QR 码登录、Webhook 转发。
 *
 * 用法:
 *   npm run dev          # 开发模式（热重载）
 *   npm start            # 生产模式
 *   npm run qr           # 仅显示 QR 码
 */

import dotenv from 'dotenv';
dotenv.config();

import { createHttpServer } from './http-server.js';
import { BaileysWhatsAppClient } from './baileys-client.js';
import { createLogger, errorType, redactNetworkEndpoint } from './logger.js';

const logger = createLogger('main');

async function main() {
  const port = parseInt(process.env.PORT || '3001', 10);
  const webhookUrl = process.env.WEBHOOK_URL;
  const webhookSecret = process.env.WEBHOOK_SECRET;

  logger.info('🚀 CookClaw WhatsApp 微服务启动中...');
  logger.info(`PROXY_URL: ${redactNetworkEndpoint(process.env.PROXY_URL)}`);
  logger.info(`HTTPS_PROXY: ${redactNetworkEndpoint(process.env.HTTPS_PROXY)}`);

  // 创建 Baileys 客户端
  const client = new BaileysWhatsAppClient();

  // 配置 Webhook（如果环境变量中有）
  if (webhookUrl) {
    client.setWebhook({ url: webhookUrl, secret: webhookSecret });
    logger.info(`Webhook 已配置: ${redactNetworkEndpoint(webhookUrl)}`);
  }

  // 创建 HTTP API 服务器
  const app = createHttpServer(client);

  // 启动 HTTP 服务器
  app.listen(port, () => {
    logger.info(`📡 HTTP API 已启动: http://localhost:${port}`);
    logger.info(`   健康检查: GET  /health`);
    logger.info(`   服务状态: GET  /api/status`);
    logger.info(`   获取 QR:  GET  /api/qr`);
    logger.info(`   发送消息: POST /api/send`);
    logger.info(`   已读回执: POST /api/read`);
    logger.info(`   发送反应: POST /api/react`);
    logger.info(`   配置钩子: POST /api/webhook`);
  });

  // 启动 WhatsApp 连接
  try {
    await client.start();
  } catch (error: unknown) {
    logger.error(`WhatsApp 启动失败: error_type=${errorType(error)}`);
    logger.info('将仅提供 HTTP API，等待手动连接...');
  }

  // 优雅关闭
  const shutdown = async (signal: string) => {
    logger.info(`\n收到 ${signal}，正在关闭...`);
    await client.stop();
    process.exit(0);
  };

  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
}

main().catch((error: unknown) => {
  logger.error(`启动失败: error_type=${errorType(error)}`);
  process.exit(1);
});
