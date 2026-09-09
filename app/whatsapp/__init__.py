"""
CookClaw WhatsApp 适配器模块

通过 HTTP API 与 app/whatsapp/service（Baileys 微服务）通信，
实现 WhatsApp 消息的收发。
"""

from app.whatsapp.adapter import WhatsAppAdapter

__all__ = ["WhatsAppAdapter"]
