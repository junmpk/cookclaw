"""
AES-256-GCM 密钥生成和解密

用于扫码配置流程中解密 Bot 凭证。
密文布局: IV (12 bytes) ‖ Ciphertext (N bytes) ‖ AuthTag (16 bytes)
"""

import base64
import os
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def generate_aes_key() -> str:
    """
    生成 256 位 AES 密钥，返回 Base64 编码字符串。

    Returns:
        str: Base64 编码的 AES-256 密钥
    """
    key_bytes = os.urandom(32)  # 256 bits
    return base64.b64encode(key_bytes).decode("utf-8")


def decrypt_secret(encrypted_base64: str, key_base64: str) -> str:
    """
    使用 AES-256-GCM 解密凭证。

    Args:
        encrypted_base64: 加密的 client_secret (Base64 编码)
        key_base64: AES 密钥 (Base64 编码)

    Returns:
        str: 解密后的 client_secret 字符串

    Raises:
        ValueError: 解密失败时抛出
    """
    try:
        key_bytes = base64.b64decode(key_base64)
        encrypted_bytes = base64.b64decode(encrypted_base64)

        # 密文布局: IV (12 bytes) ‖ Ciphertext (N bytes) ‖ AuthTag (16 bytes)
        nonce = encrypted_bytes[:12]
        ciphertext_with_tag = encrypted_bytes[12:]

        aesgcm = AESGCM(key_bytes)
        plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, None)

        return plaintext.decode("utf-8")
    except Exception as e:
        raise ValueError(f"AES-256-GCM 解密失败: {e}") from e
