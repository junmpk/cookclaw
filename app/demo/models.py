"""先读这里：外部数据用 Pydantic 校验，图节点只交换可序列化数据。"""
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class Brief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: str = Field(min_length=1, max_length=2000)
    people: int = Field(default=3, ge=1, le=10)
    dishes: int = Field(default=3, ge=1, le=5)
    soups: int = Field(default=1, ge=0, le=2)
    exclusions: list[str] = Field(default_factory=list, max_length=12)
    available_ingredients: list[str] = Field(default_factory=list, max_length=20)
    budget_yuan: int | None = Field(default=None, ge=1, le=5000)
    max_minutes: int | None = Field(default=None, ge=5, le=1440)
    equipment: list[str] = Field(default_factory=list, max_length=8)
    # 局部菜单修订信息，不属于用户长期饮食约束。
    replace_index: int | None = Field(default=None, ge=0, le=6)
    replace_recipe_ids: list[str] = Field(default_factory=list, max_length=7)
    mode: Literal["live", "rehearsal"] = "rehearsal"


class Recipe(BaseModel):
    id: str
    name: str
    source: str
    image_url: str | None = None
    ingredients: list[str]
    steps: list[str] = Field(default_factory=list)
    kind: Literal["dish", "soup"]
    nutrition: dict[str, str] = Field(default_factory=dict)
    # Imported ingredient labels are not an allergen certification.
    ingredients_complete: bool = False
    # source=真实来源详情；demo_template=仅用于展示排期的虚拟模板；
    # missing=当前没有步骤证据。前端必须显式展示该边界。
    detail_basis: Literal["source", "demo_template", "missing"] = "missing"


class ResearchDecision(BaseModel):
    selected_ids: list[str] = Field(max_length=16)
    missing: list[str] = Field(default_factory=list, max_length=6)


class DietDecision(BaseModel):
    advice: list[str] = Field(max_length=6)
    unknowns: list[str] = Field(max_length=6)
    rule_ids: list[str] = Field(max_length=6)


class MenuDecision(BaseModel):
    recipe_ids: list[str] = Field(max_length=7)
    explanation: str = Field(max_length=1000)


class ReviewDecision(BaseModel):
    needs_revision: bool
    findings: list[str] = Field(max_length=6)
    unknowns: list[str] = Field(max_length=6)


AgentName = Literal["research", "dietary", "inventory", "menu", "scheduler"]


class ComplexityDimensions(BaseModel):
    """多维任务复杂度；每一维都能映射到一个业务原因。"""

    menu: int = Field(default=0, ge=0, le=4)
    dietary: int = Field(default=0, ge=0, le=3)
    inventory: int = Field(default=0, ge=0, le=3)
    scheduling: int = Field(default=0, ge=0, le=4)


class ComplexityProfile(BaseModel):
    level: Literal["simple", "coordinated", "advanced"] = "simple"
    total_score: int = Field(default=0, ge=0, le=20)
    dimensions: ComplexityDimensions = Field(default_factory=ComplexityDimensions)
    selected_agents: list[AgentName] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list, max_length=12)


class CandidateCoverage(BaseModel):
    recipe_id: str
    matched_ingredients: list[str] = Field(default_factory=list, max_length=12)
    missing_preview: list[str] = Field(default_factory=list, max_length=8)
    coverage_percent: int = Field(default=0, ge=0, le=100)


class InventoryDecision(BaseModel):
    available_ingredients: list[str] = Field(default_factory=list, max_length=20)
    candidate_coverage: list[CandidateCoverage] = Field(default_factory=list, max_length=16)
    preferred_recipe_ids: list[str] = Field(default_factory=list, max_length=8)
    explanation: str = Field(default="", max_length=600)
    purchase_items: list[str] = Field(default_factory=list, max_length=12)
    budget_yuan: int | None = Field(default=None, ge=1, le=5000)
    unknowns: list[str] = Field(default_factory=list, max_length=8)


class InventoryAgentDecision(BaseModel):
    preferred_recipe_ids: list[str] = Field(default_factory=list, max_length=8)
    explanation: str = Field(max_length=600)
    unknowns: list[str] = Field(default_factory=list, max_length=6)


class ScheduleTask(BaseModel):
    order: int = Field(ge=1, le=20)
    recipe_id: str
    recipe_name: str
    equipment: str = "待确认"
    start_minute: int | None = Field(default=None, ge=0, le=1440)
    duration_minutes: int | None = Field(default=None, ge=1, le=1440)
    evidence: str = Field(default="", max_length=300)
    evidence_basis: Literal["source", "demo_template", "missing"] = "missing"


class ScheduleDecision(BaseModel):
    tasks: list[ScheduleTask] = Field(default_factory=list, max_length=12)
    parallel_groups: list[list[str]] = Field(default_factory=list, max_length=8)
    deadline_status: Literal["not_requested", "verifiable", "needs_confirmation"] = "not_requested"
    total_minutes: int | None = Field(default=None, ge=1, le=1440)
    unknowns: list[str] = Field(default_factory=list, max_length=8)


class PlanValidationReport(BaseModel):
    status: Literal["ready", "needs_confirmation", "blocked"] = "ready"
    blocking_issues: list[str] = Field(default_factory=list, max_length=12)
    warnings: list[str] = Field(default_factory=list, max_length=12)
    checks: dict[str, Literal["passed", "warning", "failed", "not_applicable"]] = (
        Field(default_factory=dict)
    )


class DemoState(TypedDict, total=False):
    workflow_identity: dict
    brief: dict
    complexity_profile: dict
    preloaded_candidates: list[dict]
    version: int
    candidates: list[dict]
    dietary: dict
    inventory: dict
    menu: dict
    schedule: dict
    detail_hydration: dict
    plan_validation: dict
    issues: list[str]
    review: dict
    revision_count: int
    research_count: int
    approved: bool
    execution: dict
    status: str
    replacement: dict
