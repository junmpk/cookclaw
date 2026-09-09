/**
 * 日志工具
 */

import pino from 'pino';
import { createHash } from 'crypto';

export function logFingerprint(value: unknown): string {
  const raw = String(value ?? '');
  return raw ? `sha256:${createHash('sha256').update(raw).digest('hex').slice(0, 10)}` : '-';
}

export function redactNetworkEndpoint(value: unknown): string {
  const raw = String(value ?? '').trim();
  if (!raw) return '(未设置)';
  try {
    const url = new URL(raw);
    return `${url.protocol}//${url.hostname}${url.port ? `:${url.port}` : ''}`;
  } catch {
    return '(已配置，格式不可解析)';
  }
}

export function errorType(error: unknown): string {
  return error instanceof Error ? error.name : typeof error;
}

export function createLogger(name: string) {
  const level = process.env.LOG_LEVEL || 'info';

  return pino({
    name,
    level,
    transport:
      level === 'debug'
        ? {
            target: 'pino/file',
            options: { destination: 1 }, // stdout
          }
        : undefined,
    formatters: {
      level(label) {
        return { level: label };
      },
    },
    timestamp: pino.stdTimeFunctions.isoTime,
  });
}
