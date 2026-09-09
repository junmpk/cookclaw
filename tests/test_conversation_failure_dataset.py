"""Conversation Failure Dataset 的版本、完整性与硬规则回归。"""
from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.conversation_failure_dataset import (
    ResponseHardRules,
    evaluate_response_rules,
    load_failure_dataset,
)
from scripts.replay_conversation_failure_dataset import _web_replay_thread_id

_ROOT = Path(__file__).parent / "fixtures" / "conversation_failure_dataset"
_CASES = _ROOT / "v1" / "cases.yaml"


def test_failure_dataset_v1_is_valid_unique_and_covers_priority_dimensions():
    dataset = load_failure_dataset(_CASES)

    assert dataset.schema_version == "conversation_failure_dataset_v1"
    assert dataset.dataset_version == "2026.08.1"
    assert len(dataset.cases) == 20
    assert len({case.case_id for case in dataset.cases}) == 20
    assert {
        case.case_id.split("-", 1)[0] for case in dataset.cases
    } == {"CTX", "MEM", "STATE", "RESP", "REL"}
    assert all(case.turns for case in dataset.cases)
    assert all(
        turn.expected.forbidden_tools is not None
        and turn.expected.judge_rubric
        for case in dataset.cases
        for turn in case.turns
    )


def test_failure_dataset_schema_is_machine_readable_and_version_aligned():
    schema = json.loads(
        (_ROOT / "schema.v1.json").read_text(encoding="utf-8")
    )

    assert schema["properties"]["schema_version"]["const"] == (
        "conversation_failure_dataset_v1"
    )
    assert schema["properties"]["cases"]["minItems"] == 1


def test_response_hard_rules_keep_safety_separate_from_naturalness_judge():
    rules = ResponseHardRules(
        must_contain=["没有正在等确认的步骤"],
        must_contain_any=["想先做什么", "想接着刚才哪一点"],
        must_not_contain=["不能替你猜", "待办动作", "已开始"],
        max_questions=1,
    )

    natural = evaluate_response_rules(
        "可以，不过我这边现在没有正在等确认的步骤。你想先做什么？",
        rules,
    )
    mechanical = evaluate_response_rules(
        "没有可对应的待办动作，不能替你猜。你确认吗？到底是哪一步？",
        rules,
    )

    assert natural.passed is True
    assert natural.question_count == 1
    assert mechanical.passed is False
    assert "forbidden_text:不能替你猜" in mechanical.failures
    assert "too_many_questions:2>1" in mechanical.failures


def test_failure_dataset_contains_no_obvious_secret_or_raw_identity_fields():
    text = _CASES.read_text(encoding="utf-8").lower()

    assert "api_key" not in text
    assert "password" not in text
    assert "trace_id:" not in text
    assert "raw_user_id" not in text


def test_web_replay_thread_id_matches_public_api_contract():
    first = _web_replay_thread_id("run-1", "CTX-01")
    repeated = _web_replay_thread_id("run-1", "CTX-01")
    another = _web_replay_thread_id("run-1", "CTX-02")

    assert first == repeated
    assert first != another
    assert first.startswith("web:")
    assert len(first.removeprefix("web:")) == 32
    assert all(char in "0123456789abcdef" for char in first.removeprefix("web:"))
