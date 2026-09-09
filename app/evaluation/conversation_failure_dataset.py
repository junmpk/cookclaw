"""版本化 Conversation Failure Dataset 的加载与硬规则校验。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ResponseHardRules(BaseModel):
    nonempty: bool = True
    must_contain: list[str] = Field(default_factory=list)
    must_contain_any: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    max_questions: int | None = Field(default=None, ge=0, le=3)


class ExpectedTurn(BaseModel):
    route: list[str] = Field(default_factory=list)
    reference: str | None = None
    state: dict = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    response: ResponseHardRules = Field(default_factory=ResponseHardRules)
    judge_rubric: dict[str, int] = Field(default_factory=dict)


class FailureTurn(BaseModel):
    input: str = Field(min_length=1, max_length=8000)
    expected: ExpectedTurn


class FailureCase(BaseModel):
    case_id: str = Field(pattern=r"^[A-Z]+-[0-9]{2}$")
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    severity: Literal["P0", "P1", "P2"]
    tags: list[str] = Field(min_length=1)
    channel: Literal["web", "qq", "weixin", "whatsapp", "all"]
    locale: Literal["zh", "en"]
    thread_policy: Literal["fresh", "reuse", "new_thread_same_account"]
    pre_state: dict = Field(default_factory=dict)
    turns: list[FailureTurn] = Field(min_length=1)
    cleanup: dict = Field(default_factory=dict)


class ConversationFailureDataset(BaseModel):
    schema_version: Literal["conversation_failure_dataset_v1"]
    dataset_version: str = Field(pattern=r"^[0-9]{4}\.[0-9]{2}\.[0-9]+$")
    cases: list[FailureCase] = Field(min_length=1)

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> ConversationFailureDataset:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id values must be unique")
        return self


class ResponseRuleResult(BaseModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)
    question_count: int = 0


def load_failure_dataset(path: Path) -> ConversationFailureDataset:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ConversationFailureDataset.model_validate(payload)


def evaluate_response_rules(
    response: str,
    rules: ResponseHardRules,
) -> ResponseRuleResult:
    text = str(response or "").strip()
    failures: list[str] = []
    if rules.nonempty and not text:
        failures.append("response_empty")
    for value in rules.must_contain:
        if value not in text:
            failures.append(f"missing_required:{value}")
    if rules.must_contain_any and not any(
        value in text for value in rules.must_contain_any
    ):
        failures.append("missing_required_any")
    for value in rules.must_not_contain:
        if value in text:
            failures.append(f"forbidden_text:{value}")
    question_count = len(re.findall(r"[?？]", text))
    if (
        rules.max_questions is not None
        and question_count > rules.max_questions
    ):
        failures.append(
            f"too_many_questions:{question_count}>{rules.max_questions}"
        )
    return ResponseRuleResult(
        passed=not failures,
        failures=failures,
        question_count=question_count,
    )
