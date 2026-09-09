"""Bounded Planner active 模式的确定性灰度策略。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True, slots=True)
class PlannerRolloutDecision:
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
    digest = hashlib.sha256(
        f"turn-planner:{channel}:{user_id}".encode("utf-8")
    ).digest()
    return (int.from_bytes(digest[:4], "big") % 10_000) / 100


def planner_active_actions() -> set[str]:
    """返回 active Executor 的配置动作白名单。"""
    return _split_csv(getattr(settings, "TURN_PLANNER_ACTIVE_ACTIONS", ""))


def planner_rollout_decision(
    thread_id: str,
    *,
    message_type: str = "text",
) -> PlannerRolloutDecision:
    """active 必须显式启用并命中灰度；语音仍不进入模型规划。"""
    channel, user_id = _thread_identity(thread_id)
    mode = str(getattr(settings, "TURN_PLANNER_MODE", "off") or "off")
    if mode.strip().lower() != "active":
        return PlannerRolloutDecision(False, "mode_disabled", channel)
    if str(message_type or "text").strip().lower() not in {"text", "image"}:
        return PlannerRolloutDecision(False, "message_type_excluded", channel)

    channels = _split_csv(
        getattr(settings, "TURN_PLANNER_ACTIVE_CHANNELS", "qq"),
        lower=True,
    )
    if not channel or (channel not in channels and "*" not in channels):
        return PlannerRolloutDecision(False, "channel_excluded", channel)

    if channel == "qq":
        allow_users = _split_csv(
            getattr(settings, "TURN_PLANNER_ACTIVE_QQ_ALLOW_FROM", ""),
        )
        if user_id and user_id in allow_users:
            return PlannerRolloutDecision(True, "qq_allowlist", channel)

    allow_identities = _split_csv(
        getattr(settings, "TURN_PLANNER_ACTIVE_ALLOW_IDENTITIES", ""),
    )
    if user_id and f"{channel}:{user_id}" in allow_identities:
        return PlannerRolloutDecision(
            True,
            "identity_allowlist",
            channel,
        )

    percentage = max(
        0.0,
        min(
            100.0,
            float(
                getattr(
                    settings,
                    "TURN_PLANNER_ACTIVE_ROLLOUT_PERCENT",
                    0.0,
                )
            ),
        ),
    )
    if user_id and percentage > 0:
        if _stable_percentage_bucket(channel, user_id) < percentage:
            return PlannerRolloutDecision(
                True,
                "stable_percentage",
                channel,
            )
    return PlannerRolloutDecision(False, "not_selected", channel)
