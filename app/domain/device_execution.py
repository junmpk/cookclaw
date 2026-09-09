"""设备执行任务的稳定标识。

``action_id`` 是 CookClaw 任务状态主键；``msg_id`` 随设备请求透传。
同一执行记录在任何安全重试中都必须复用原标识。
"""

from __future__ import annotations

import secrets
import uuid


# JavaScript/JSON 可精确表达的最大整数，避免上游将 msgId 当 Number 时丢精度。
MAX_DEVICE_MSG_ID = 9_007_199_254_740_991


def new_device_action_identity() -> tuple[str, int]:
    action_id = uuid.uuid4().hex
    msg_id = secrets.randbelow(MAX_DEVICE_MSG_ID - 1) + 1
    return action_id, msg_id


def normalize_device_msg_id(value: object) -> int:
    try:
        msg_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("device msg_id must be an integer") from exc
    if msg_id <= 0 or msg_id > MAX_DEVICE_MSG_ID:
        raise ValueError(
            f"device msg_id must be between 1 and {MAX_DEVICE_MSG_ID}"
        )
    return msg_id


__all__ = [
    "MAX_DEVICE_MSG_ID",
    "new_device_action_identity",
    "normalize_device_msg_id",
]
