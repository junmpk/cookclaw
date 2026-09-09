"""
QQ Bot 通道模块

基于 QQ Bot 官方 API v2: https://bot.q.qq.com/wiki/develop/api-v2/
"""

from app.qqbot.constants import QQBOT_VERSION
from app.qqbot.adapter import QQAdapter

__all__ = ["QQAdapter", "QQBOT_VERSION"]
__version__ = QQBOT_VERSION
