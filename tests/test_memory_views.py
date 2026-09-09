"""RuntimeMemorySnapshot 的类型化、深层只读消费视图。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from app.conversation.models import ConversationMemory, ConversationTurn
from app.conversation.preference_parser import PreferenceMutation
from app.conversation.runtime_memory import (
    LongTermMemorySnapshot,
    QAMemoryView,
    RuntimeMemorySnapshot,
    SafetyMemoryView,
    build_planner_memory_view,
    build_qa_memory_view,
    build_safety_memory_view,
)


def _snapshot(
    *,
    short_status: str = "loaded",
    long_status: str = "loaded",
) -> RuntimeMemorySnapshot:
    memory = ConversationMemory(
        thread_id="qq:dm:view-user:view-user",
        channel="qq",
        user_id="view-user",
        preferences={
            "likes": ["家常菜"],
            "dislikes": ["香菜"],
            "allergens": ["花生"],
            "dietary_constraints": ["素食"],
            "available_ingredients": ["鸡蛋"],
        },
        recent_turns=[
            ConversationTurn(role="user", content="下午好", created_at=1),
            ConversationTurn(role="assistant", content="下午好呀", created_at=2),
            ConversationTurn(
                role="user",
                content="晚上帮我整点肉类菜",
                created_at=3,
            ),
        ],
        summary={
            "conversation_digest": ["用户：之前聊过晚饭", "助手：给过建议"],
        },
    )
    first_search = {
        "created_at": 10,
        "recipes": [{"id": "r1"}, {"id": "r2"}],
    }
    latest_search = {
        "created_at": 20,
        "recipes": [{"id": "r3"}],
    }
    memory.search_history = [first_search, latest_search]
    memory.latest_search = latest_search

    long_term = LongTermMemorySnapshot(
        status=long_status,  # type: ignore[arg-type]
        profile_version=7,
        preferred_name="范老师",
        display_name="范先生",
        preferences={
            "likes": ["清淡"],
            "dislikes": ["肥肉"],
            "allergens": ["虾"],
            "dietary_constraints": ["清真"],
            "available_ingredients": [],
        },
        temporal_dietary_constraints=[{
            "value": "减脂",
            "source": "user_explicit",
            "updated_at": 100,
            "expires_at": 200,
            "mutable_metadata": {"must_not_escape": True},
        }],
        account_digest=["偏好清淡", "对虾过敏"],
        invalidated_preferences={
            "likes": [],
            "dislikes": ["旧忌口"],
            "allergens": ["旧过敏原"],
            "dietary_constraints": [],
            "available_ingredients": [],
        },
        general_facts=[{
            "id": "fact-job",
            "category": "work",
            "key": "occupation",
            "value": "AI应用开发工程师",
            "subject": "self",
            "scope": "stable",
            "source": "user_explicit",
            "updated_at": 300,
            "expires_at": 0,
            "mutable_metadata": {"must_not_escape": True},
        }],
    )
    return RuntimeMemorySnapshot(
        thread_id=memory.thread_id,
        channel="qq",
        user_id="view-user",
        short_term=memory,
        short_term_status=short_status,  # type: ignore[arg-type]
        long_term=long_term,
        current_turn_preferences={
            "likes": ["牛肉"],
            "dislikes": [],
            "allergens": [],
            "dietary_constraints": [],
            "available_ingredients": [],
        },
        current_turn_preference_mutations=(
            PreferenceMutation(
                operation="remove",
                bucket="dislikes",
                value="香菜",
                source_span="我没有不喜欢香菜了",
                scope="long_term_candidate",
            ),
        ),
        is_new_session=False,
        hydrated_at=400,
    )


def _assert_deeply_readonly(value: object) -> None:
    assert not isinstance(value, (ConversationMemory, dict, list, set))
    if is_dataclass(value):
        for item in fields(value):
            _assert_deeply_readonly(getattr(value, item.name))
    elif isinstance(value, tuple):
        for item in value:
            _assert_deeply_readonly(item)


def test_views_distinguish_not_provided_loaded_empty_and_unavailable():
    missing = build_safety_memory_view(None)
    assert missing.load_state.provided is False
    assert missing.load_state.short_term_status is None
    assert missing.load_state.long_term_status is None
    assert missing.to_context_dict()["provided"] is False

    loaded_empty = build_planner_memory_view(_snapshot(long_status="empty"))
    assert loaded_empty.load_state.provided is True
    assert loaded_empty.load_state.long_term_loaded_empty is True
    assert loaded_empty.load_state.long_term_available is True
    assert loaded_empty.load_state.long_term_failed is False

    unavailable_snapshot = _snapshot(
        short_status="unavailable",
        long_status="unavailable",
    )
    unavailable = build_qa_memory_view(unavailable_snapshot)
    assert unavailable.load_state.provided is True
    assert unavailable.load_state.short_term_available is False
    assert unavailable.load_state.long_term_failed is True
    # 即使异常快照误带正文，失败状态也必须 fail closed。
    assert unavailable.preferred_name is None
    assert unavailable.long_term_preferences.has_values is False
    assert unavailable.session_preferences.has_values is False
    assert unavailable.general_facts == ()
    assert unavailable.recent_turns == ()


def test_runtime_builds_typed_deeply_readonly_views():
    snapshot = _snapshot()

    safety = snapshot.safety_view()
    planner = snapshot.planner_view()
    qa = snapshot.qa_view()

    assert isinstance(safety, SafetyMemoryView)
    assert isinstance(qa, QAMemoryView)
    _assert_deeply_readonly(safety)
    _assert_deeply_readonly(planner)
    _assert_deeply_readonly(qa)

    with pytest.raises(FrozenInstanceError):
        planner.previous_candidate_count = 99  # type: ignore[misc]
    with pytest.raises(AttributeError):
        planner.recent_user_turns.append("污染")  # type: ignore[attr-defined]

    assert "范老师" not in repr(qa)
    assert "花生" not in repr(safety)


def test_views_copy_source_values_and_ignore_later_source_mutation():
    snapshot = _snapshot()
    safety = build_safety_memory_view(snapshot)
    planner = build_planner_memory_view(snapshot)
    qa = build_qa_memory_view(snapshot)

    snapshot.short_term.preferences["allergens"].append("后加短期过敏原")
    snapshot.short_term.recent_turns[0].content = "篡改历史"
    snapshot.long_term.preferences["allergens"].append("后加长期过敏原")
    snapshot.long_term.temporal_dietary_constraints[0]["value"] = "已改变"
    snapshot.long_term.general_facts[0]["value"] = "已改变职业"
    snapshot.current_turn_preferences["likes"].append("后加本轮喜好")

    assert safety.session_constraints.allergens == ("花生",)
    assert safety.long_term_constraints.allergens == ("虾",)
    assert safety.temporal_dietary_constraints[0].value == "减脂"
    assert planner.current_turn_preferences.likes == ("牛肉",)
    assert qa.recent_turns[0].content == "下午好"
    assert qa.general_facts[0].value == "AI应用开发工程师"


def test_to_context_dict_returns_a_fresh_deep_copy_each_time():
    qa = build_qa_memory_view(_snapshot())
    first = qa.to_context_dict()
    second = qa.to_context_dict()

    first["long_term_preferences"]["allergens"].append("污染")
    first["general_facts"][0]["value"] = "污染"
    first["recent_turns"][0]["content"] = "污染"

    assert second["long_term_preferences"]["allergens"] == ["虾"]
    assert second["general_facts"][0]["value"] == "AI应用开发工程师"
    assert second["recent_turns"][0]["content"] == "下午好"
    assert qa.long_term_preferences.allergens == ("虾",)
    assert qa.general_facts[0].value == "AI应用开发工程师"


def test_each_view_exposes_only_its_allowed_typed_projection():
    snapshot = _snapshot()
    safety = build_safety_memory_view(snapshot)
    planner = build_planner_memory_view(snapshot)
    qa = build_qa_memory_view(snapshot)

    assert safety.long_term_constraints.allergens == ("虾",)
    assert safety.long_term_constraints.dislikes == ("肥肉",)
    assert not hasattr(safety.long_term_constraints, "likes")
    assert not hasattr(safety.long_term_constraints, "available_ingredients")
    assert safety.invalidated_long_term_constraints.allergens == (
        "旧过敏原",
    )

    assert planner.long_term_preferences.likes == ("清淡",)
    assert planner.session_preferences.available_ingredients == ("鸡蛋",)
    assert planner.current_turn_preferences.likes == ("牛肉",)
    assert planner.recent_user_turns == (
        "下午好",
        "晚上帮我整点肉类菜",
    )
    assert planner.previous_candidate_count == 2
    assert planner.current_turn_mutations[0].operation == "remove"

    assert qa.preferred_name == "范老师"
    assert qa.account_digest == ("偏好清淡", "对虾过敏")
    assert qa.conversation_digest == (
        "用户：之前聊过晚饭",
        "助手：给过建议",
    )
    assert qa.general_facts[0].fact_id == "fact-job"
    fact_context = qa.to_context_dict()["general_facts"][0]
    assert "mutable_metadata" not in fact_context
    mutation_context = qa.to_context_dict()["current_turn_mutations"][0]
    assert "source_span" not in mutation_context


def test_runtime_view_methods_match_pure_builders():
    snapshot = _snapshot()

    assert snapshot.safety_view() == build_safety_memory_view(snapshot)
    assert snapshot.planner_view() == build_planner_memory_view(snapshot)
    assert snapshot.qa_view() == build_qa_memory_view(snapshot)


def test_resolved_views_export_effective_policy_values_without_aliasing():
    snapshot = _snapshot()
    resolved = {
        "preferred_name": "会话称呼",
        "display_name": "显示名",
        "preferences": {
            "likes": ["鸡肉"],
            "dislikes": [],
            "allergens": ["虾"],
            "dietary_constraints": [],
            "available_ingredients": ["土豆"],
        },
        "temporal_dietary_constraints": [{"value": "减脂"}],
        "account_digest": ["喜欢鸡肉"],
    }

    safety = build_safety_memory_view(snapshot, resolved_context=resolved)
    planner = build_planner_memory_view(snapshot, resolved_context=resolved)
    qa = build_qa_memory_view(snapshot, resolved_context=resolved)

    assert safety.to_router_context_dict()["preferences"]["likes"] == ["鸡肉"]
    assert planner.to_planner_context_dict()["preferences"]["allergens"] == ["虾"]
    assert qa.preferred_name == "会话称呼"
    assert qa.account_digest == ("喜欢鸡肉",)

    first = planner.to_planner_context_dict()
    second = planner.to_planner_context_dict()
    first["preferences"]["likes"].append("污染")
    assert second["preferences"]["likes"] == ["鸡肉"]
    assert planner.effective_preferences.likes == ("鸡肉",)


def test_unresolved_raw_view_cannot_be_used_as_policy_result():
    planner = build_planner_memory_view(_snapshot())

    with pytest.raises(RuntimeError, match="ConversationService"):
        planner.to_planner_context_dict()
