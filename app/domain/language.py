"""不依赖通道或检索实现的语言判定规则。"""

from __future__ import annotations


def detect_lang(text: str) -> str:
    """仅区分 zh/en；中文字符占比大于 0.3 时判定为中文。"""
    if not text:
        return "zh"
    zh = sum(1 for char in text if 0x4E00 <= ord(char) <= 0x9FFF)
    total = len(text.replace(" ", ""))
    if total == 0:
        return "zh"
    return "zh" if (zh / total) > 0.3 else "en"


__all__ = ["detect_lang"]
