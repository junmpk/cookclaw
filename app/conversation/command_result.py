"""画像与偏好命令可向编排层公开的脱敏状态变更。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MemoryStateScope = Literal["session", "profile"]
MemoryStateOperation = Literal["set", "clear"]


@dataclass(frozen=True, slots=True)
class MemoryStateChange:
    """已经由 Memory 领域完成的状态变化；只记录字段名，不包含字段值。"""

    scope: MemoryStateScope
    operation: MemoryStateOperation
    keys: tuple[str, ...]
    reason_code: str
