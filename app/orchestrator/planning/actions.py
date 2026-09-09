"""Planner 可见动作的封闭注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import FrozenSet, Mapping

from app.orchestrator.planning.models import PlannerAction, RiskLevel


@dataclass(frozen=True)
class ActionSpec:
    name: PlannerAction
    risk: RiskLevel
    allowed_args: FrozenSet[str] = frozenset()
    required_args: FrozenSet[str] = frozenset()


_ACTION_REGISTRY: dict[str, ActionSpec] = {
    "conversation.respond": ActionSpec(
        name="conversation.respond",
        risk="low",
    ),
    "conversation.clarify": ActionSpec(
        name="conversation.clarify",
        risk="low",
        allowed_args=frozenset({"slot"}),
        required_args=frozenset({"slot"}),
    ),
    "recipe.search": ActionSpec(
        name="recipe.search",
        risk="low",
        allowed_args=frozenset({"query"}),
    ),
    "recipe.recommend": ActionSpec(
        name="recipe.recommend",
        risk="low",
        allowed_args=frozenset({"query"}),
    ),
    "recipe.detail": ActionSpec(
        name="recipe.detail",
        risk="low",
        allowed_args=frozenset({"recipe_id"}),
        required_args=frozenset({"recipe_id"}),
    ),
    "candidate.select": ActionSpec(
        name="candidate.select",
        risk="medium",
        allowed_args=frozenset({"recipe_id", "position"}),
    ),
    "candidate.compare": ActionSpec(
        name="candidate.compare",
        risk="low",
    ),
    "candidate.restore_previous": ActionSpec(
        name="candidate.restore_previous",
        risk="low",
    ),
    "menu.plan": ActionSpec(
        name="menu.plan",
        risk="low",
        allowed_args=frozenset({"query"}),
    ),
    "device.prepare": ActionSpec(
        name="device.prepare",
        risk="medium",
        allowed_args=frozenset({"recipe_id"}),
        required_args=frozenset({"recipe_id"}),
    ),
    "device.status": ActionSpec(
        name="device.status",
        risk="low",
    ),
    "web.search": ActionSpec(
        name="web.search",
        risk="low",
        allowed_args=frozenset({"query"}),
    ),
}

ACTION_REGISTRY: Mapping[str, ActionSpec] = MappingProxyType(_ACTION_REGISTRY)


def planner_capabilities() -> list[str]:
    return list(ACTION_REGISTRY)
