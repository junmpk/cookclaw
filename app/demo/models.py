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


class DemoState(TypedDict, total=False):
    brief: dict
    version: int
    candidates: list[dict]
    dietary: dict
    menu: dict
    issues: list[str]
    review: dict
    revision_count: int
    research_count: int
    approved: bool
    execution: dict
    status: str
    replacement: dict
