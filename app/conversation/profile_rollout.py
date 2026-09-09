"""用户资料记忆的独立灰度策略，避免和 Deep Agent 开关耦合。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class ProfileMemoryRolloutDecision:
    enabled: bool
    cohort: str
    channel: str


def _split_csv(value: object, *, lower: bool = False) -> set[str]:
    values = {
        item.strip()
        for item in str(value or "").split(",")
        if item.strip()
    }
    return {item.lower() for item in values} if lower else values


def _thread_identity(thread_id: str) -> tuple[str, str]:
    parts = str(thread_id or "").split(":")
    channel = parts[0].strip().lower() if parts else ""
    user_id = parts[-1].strip() if len(parts) >= 2 else ""
    return channel, user_id


def _stable_percentage_bucket(channel: str, user_id: str) -> float:
    digest = hashlib.sha256(f"profile:{channel}:{user_id}".encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") % 10_000) / 100


def profile_memory_rollout_decision(
    thread_id: str,
) -> ProfileMemoryRolloutDecision:
    channel, user_id = _thread_identity(thread_id)
    if not settings.CONVERSATION_PROFILE_MEMORY_ENABLED:
        return ProfileMemoryRolloutDecision(False, "master_disabled", channel)
    channels = _split_csv(
        settings.CONVERSATION_PROFILE_MEMORY_CHANNELS,
        lower=True,
    )
    if not channel or (channel not in channels and "*" not in channels):
        return ProfileMemoryRolloutDecision(False, "channel_excluded", channel)
    if channel == "qq":
        allow_users = _split_csv(
            settings.CONVERSATION_PROFILE_MEMORY_QQ_ALLOW_FROM,
        )
        if user_id and user_id in allow_users:
            return ProfileMemoryRolloutDecision(True, "qq_allowlist", channel)
    percentage = max(
        0.0,
        min(
            100.0,
            float(settings.CONVERSATION_PROFILE_MEMORY_ROLLOUT_PERCENT),
        ),
    )
    if user_id and percentage > 0:
        if _stable_percentage_bucket(channel, user_id) < percentage:
            return ProfileMemoryRolloutDecision(
                True,
                "stable_percentage",
                channel,
            )
    return ProfileMemoryRolloutDecision(False, "not_selected", channel)


def profile_memory_enabled(thread_id: str) -> bool:
    return profile_memory_rollout_decision(thread_id).enabled


def general_profile_memory_enabled(thread_id: str) -> bool:
    """通用事实复用用户资料灰度边界，并额外受独立总开关控制。"""
    return bool(
        settings.CONVERSATION_GENERAL_MEMORY_ENABLED
        and profile_memory_enabled(thread_id)
    )
