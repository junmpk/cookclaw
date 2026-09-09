"""Bounded Turn Planner 的稳定输入、输出与审计协议。"""

from __future__ import annotations

import re
from typing import Any, Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from app.orchestrator.turn.runtime_models import DerivedTurnContext


PlannerPhase = Literal[
    "explore",
    "clarify",
    "search_or_recommend",
    "candidate_selection",
    "recipe_detail",
    "device_prepare",
    "awaiting_confirmation",
    "running",
    "follow_up",
]

PlannerAction = Literal[
    "conversation.respond",
    "conversation.clarify",
    "recipe.search",
    "recipe.recommend",
    "recipe.detail",
    "candidate.select",
    "candidate.compare",
    "candidate.restore_previous",
    "menu.plan",
    "device.prepare",
    "device.status",
    "web.search",
]

ReplyAct = Literal[
    "acknowledge",
    "ask_one_question",
    "answer",
    "recommend",
    "explain",
    "offer_next_step",
    "report_state",
]

RiskLevel = Literal["low", "medium", "high"]
PlannerStatus = Literal[
    "success",
    "timeout",
    "model_error",
    "parse_error",
]

NonEmptyShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
]
ShortFact = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=180),
]
EvidenceRef = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]


class RecipeRef(BaseModel):
    """Planner 可见的最小菜谱引用，不包含步骤、设备参数或完整 metadata。"""

    model_config = ConfigDict(extra="forbid")

    recipe_id: str = Field(default="", max_length=128)
    name: str = Field(default="", max_length=80)
    position: int | None = Field(default=None, ge=1, le=20)


class TurnContext(BaseModel):
    """一轮 Planner 所需的裁剪上下文。

    ``utterance``、近期轮次和约束只发送给 Planner，不允许写入 Shadow 日志。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["turn_context_v1"] = "turn_context_v1"
    trace_id: str = Field(min_length=1, max_length=64)
    # 纯图片消息可以没有 caption；这时 Planner 只能使用明确标注来源的派生事实。
    utterance: str = Field(default="", max_length=2_000)
    channel: str = Field(default="unknown", max_length=24)
    message_type: str = Field(default="text", max_length=24)
    pending_action: str | None = Field(default=None, max_length=80)
    pending_device_start: bool = False
    device_choice_required: bool = False
    active_cooking: bool = False
    pending_clarification: str | None = Field(default=None, max_length=80)
    latest_candidates: list[RecipeRef] = Field(default_factory=list, max_length=10)
    previous_candidate_count: int = Field(default=0, ge=0, le=20)
    latest_focus: RecipeRef | None = None
    selected_recipe: RecipeRef | None = None
    current_task: str | None = Field(default=None, max_length=80)
    menu_task_active: bool = False
    active_constraints: dict[str, list[str]] = Field(default_factory=dict)
    stable_constraints: dict[str, list[str]] = Field(default_factory=dict)
    recent_user_turns: list[str] = Field(default_factory=list, max_length=4)
    capabilities: list[str] = Field(default_factory=list, max_length=16)
    derived_context: DerivedTurnContext | None = Field(
        default=None,
        repr=False,
    )

    def planner_payload(self) -> dict[str, Any]:
        """返回模型输入；明确排除仅用于链路关联的 trace_id。"""
        return self.model_dump(exclude={"trace_id"})

    def context_ref(self) -> dict[str, Any]:
        """返回可安全记录的上下文摘要，不包含原话、菜名、ID 或约束值。"""
        result = {
            "schema_version": self.schema_version,
            "channel": self.channel,
            "message_type": self.message_type,
            "has_pending_action": self.pending_action is not None,
            "pending_device_start": self.pending_device_start,
            "device_choice_required": self.device_choice_required,
            "active_cooking": self.active_cooking,
            "has_pending_clarification": self.pending_clarification is not None,
            "candidate_count": len(self.latest_candidates),
            "previous_candidate_count": self.previous_candidate_count,
            "has_focus": self.latest_focus is not None,
            "has_selected_recipe": self.selected_recipe is not None,
            "has_current_task": self.current_task is not None,
            "menu_task_active": self.menu_task_active,
            "active_constraint_count": sum(
                len(values) for values in self.active_constraints.values()
            ),
            "stable_constraint_count": sum(
                len(values) for values in self.stable_constraints.values()
            ),
            "recent_user_turn_count": len(self.recent_user_turns),
            "capability_count": len(self.capabilities),
        }
        if self.derived_context is not None:
            result["derived_context"] = self.derived_context.context_ref()
        return result

    def known_recipe_ids(self) -> set[str]:
        values = {
            item.recipe_id
            for item in self.latest_candidates
            if item.recipe_id
        }
        for item in (self.latest_focus, self.selected_recipe):
            if item is not None and item.recipe_id:
                values.add(item.recipe_id)
        return values


class PlanStep(BaseModel):
    """Planner 建议的一个低风险业务步骤。"""

    model_config = ConfigDict(extra="forbid")

    action: PlannerAction
    args: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=8)


class TurnPlan(BaseModel):
    """Planner 的唯一输出；它是建议，不是副作用授权。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["turn_plan_v1"] = "turn_plan_v1"
    goal: NonEmptyShortText
    phase: PlannerPhase
    known_facts: list[ShortFact] = Field(default_factory=list, max_length=12)
    missing_slots: list[ShortFact] = Field(default_factory=list, max_length=4)
    steps: list[PlanStep] = Field(default_factory=list, max_length=2)
    reply_act: ReplyAct
    risk: RiskLevel = "low"
    reason_code: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0, le=1)
    requires_deterministic_handler: bool = False

    @field_validator("reason_code")
    @classmethod
    def normalize_reason_code(cls, value: str) -> str:
        normalized = "_".join(str(value or "").strip().upper().split())
        if not normalized:
            raise ValueError("reason_code is required")
        if not re.fullmatch(r"[A-Z0-9_]+", normalized):
            raise ValueError("reason_code must contain only A-Z, 0-9 and underscore")
        return normalized[:80]


class PlanValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=80)
    step_index: int | None = Field(default=None, ge=0, le=1)


class PlanValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    execution_allowed: bool
    issues: list[PlanValidationIssue] = Field(default_factory=list, max_length=20)

    @property
    def issue_codes(self) -> list[str]:
        return [item.code for item in self.issues]


class PlannerCallResult(BaseModel):
    """一次 Planner 模型调用的脱敏结果。"""

    model_config = ConfigDict(extra="forbid")

    status: PlannerStatus
    plan: TurnPlan | None = None
    error_code: str | None = Field(default=None, max_length=80)
    model: str = Field(default="", max_length=120)
    duration_ms: float = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class LegacyRouteSnapshot(BaseModel):
    """当前路由的可比较摘要，不包含 SearchRequest、结果或回复正文。"""

    model_config = ConfigDict(extra="forbid")

    outcome: str = Field(default="unknown", max_length=80)
    action: str = Field(default="unknown", max_length=80)
    category: str = Field(default="unknown", max_length=80)
    reason_code: str = Field(default="UNKNOWN", max_length=80)
    risk: RiskLevel = "low"
    needs_clarification: bool = False


class ShadowComparison(BaseModel):
    """Legacy 与 Planner 的机械差异，不能替代人工正确性判断。"""

    model_config = ConfigDict(extra="forbid")

    expected_planner_action: str | None = Field(default=None, max_length=80)
    planner_action: str | None = Field(default=None, max_length=80)
    action_match: bool | None = None
    risk_match: bool | None = None
    clarification_match: bool | None = None
    deterministic_handoff_match: bool | None = None
    mismatch_codes: list[str] = Field(default_factory=list, max_length=12)
