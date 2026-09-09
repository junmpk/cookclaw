"""澄清归约器的纯状态语义回归。"""

import pytest

from app.orchestrator.clarification_reducer import (
    ConstraintReply,
    parse_constraint_reply,
    reduce_constraint_clarification,
)


@pytest.mark.parametrize(
    "text",
    [
        "没有",
        "没有什么忌口，请推荐",
        "没啥过敏",
        "没有任何饮食限制",
        "都能吃",
        "no restrictions",
    ],
)
def test_no_constraints_requires_pending_safety_dimension(text):
    assert parse_constraint_reply(text, None) is ConstraintReply.UNKNOWN
    assert (
        parse_constraint_reply(text, "flavor_preferences")
        is ConstraintReply.UNKNOWN
    )
    transition = reduce_constraint_clarification(text, "party_constraints")
    assert transition.reply is ConstraintReply.NO_CONSTRAINTS
    assert transition.constraints_confirmed is True
    assert transition.next_dimension is None


@pytest.mark.parametrize("text", ["有", "有的", "yes"])
def test_bare_yes_moves_to_constraint_detail(text):
    transition = reduce_constraint_clarification(text, "party_constraints")
    assert transition.reply is ConstraintReply.HAS_CONSTRAINTS
    assert transition.constraints_confirmed is False
    assert transition.next_dimension == "party_constraints_detail"


def test_constraint_detail_is_recognized_only_inside_safety_chain():
    assert (
        parse_constraint_reply("我对花生过敏", "party_constraints_detail")
        is ConstraintReply.CONSTRAINT_DETAILS
    )
    assert (
        parse_constraint_reply("我对花生过敏", "available_ingredients")
        is ConstraintReply.UNKNOWN
    )


def test_unrelated_short_reply_does_not_advance_safety_state():
    transition = reduce_constraint_clarification(
        "清淡一点",
        "party_constraints",
    )
    assert transition.reply is ConstraintReply.UNKNOWN
    assert transition.constraints_confirmed is None
