"""受控 Deep Agent 的确定性灰度策略。

主开关、通道、QQ 用户白名单和稳定百分比分桶必须同时在 Python 层收敛，
避免通道入口、推荐回复和候选选择各自判断出不同结果。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class DeepAgentRolloutDecision:
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
    digest = hashlib.sha256(f"{channel}:{user_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10_000
    return bucket / 100


def deep_agent_rollout_decision(thread_id: str) -> DeepAgentRolloutDecision:
    """返回本轮是否允许进入受控 Agent；不读取或记录用户消息正文。"""
    channel, user_id = _thread_identity(thread_id)
    if not settings.CONVERSATION_DEEP_AGENT_ENABLED:
        return DeepAgentRolloutDecision(False, "master_disabled", channel)

    channels = _split_csv(
        settings.CONVERSATION_DEEP_AGENT_CHANNELS,
        lower=True,
    )
    if not channel or (channel not in channels and "*" not in channels):
        return DeepAgentRolloutDecision(False, "channel_excluded", channel)

    if channel == "qq":
        allow_users = _split_csv(
            settings.CONVERSATION_DEEP_AGENT_QQ_ALLOW_FROM,
        )
        if user_id and user_id in allow_users:
            return DeepAgentRolloutDecision(True, "qq_allowlist", channel)

    percentage = max(
        0.0,
        min(100.0, float(settings.CONVERSATION_DEEP_AGENT_ROLLOUT_PERCENT)),
    )
    if user_id and percentage > 0:
        if _stable_percentage_bucket(channel, user_id) < percentage:
            return DeepAgentRolloutDecision(True, "stable_percentage", channel)

    return DeepAgentRolloutDecision(False, "not_selected", channel)


def controlled_deep_agent_enabled(thread_id: str) -> bool:
    return deep_agent_rollout_decision(thread_id).enabled
