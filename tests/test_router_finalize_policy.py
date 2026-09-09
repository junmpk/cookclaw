"""Router 的“交给系统决定”与安全澄清政策回归。"""

from __future__ import annotations

import asyncio

import pytest

from app.orchestrator import router
from app.orchestrator.intent import IntentResult
from app.orchestrator.recommendation_response import RecommendationNarrative
from app.orchestrator.search_request import SearchRequest


def _search_results(count: int = 10) -> dict:
    return {
        "success": True,
        "results": [
            {
                "id": f"recipe-{index}",
                "score": 0.95 - index * 0.01,
                "metadata": {
                    "recipe_id": f"recipe-{index}",
                    "name": f"已核验菜品{index}",
                    "ingredients": [f"食材{index}"],
                    "tags": ["家常"],
                },
            }
            for index in range(1, count + 1)
        ],
    }


def _stub_grounded_outputs(monkeypatch) -> list[tuple[str, int]]:
    searches: list[tuple[str, int]] = []

    async def search(query: str, top_k: int = 3, lang: str | None = None) -> dict:
        del lang
        searches.append((query, top_k))
        return _search_results()

    async def narrative(**_kwargs) -> RecommendationNarrative:
        return RecommendationNarrative("", "", {}, "")

    async def clarify(*_args, dimension=None, **_kwargs) -> str:
        return f"clarify:{dimension}"

    monkeypatch.setattr(router, "_run_search_subprocess", search)
    monkeypatch.setattr(router, "generate_recommendation_narrative", narrative)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarify)
    return searches


def _pending_party_request(*, constraints_confirmed: bool = True) -> SearchRequest:
    return SearchRequest(
        original_text="六个人聚餐，推荐六道菜",
        query="朋友聚餐",
        scenes=["朋友聚餐"],
        party_size=6,
        result_limit=6,
        explicit_result_limit=True,
        constraints_confirmed=constraints_confirmed,
        task_operation="fuzzy_recommend",
    )


def test_semantic_finalize_merges_pending_request_instead_of_abandoning_it(monkeypatch):
    searches = _stub_grounded_outputs(monkeypatch)
    intent = IntentResult.from_raw(
        {
            "r": False,
            "c": "off_topic",
            "k": [],
            "s": {},
            "signals": {"finalize": True},
        },
        original_text="就这样",
    )

    outcome = asyncio.run(router.route_fast_path(
        "就这样",
        pending_search_request=_pending_party_request().model_dump(),
        pending_clarification_dimension="flavor_preferences",
        intent_override=intent,
    ))

    assert outcome.kind == "search"
    assert outcome.intent.category.value == "recipe_recommend"
    assert outcome.search_request.party_size == 6
    assert outcome.search_request.result_limit == 6
    assert outcome.search_request.explicit_result_limit is True
    assert outcome.search_request.constraints_confirmed is True
    assert searches and "朋友聚餐" in searches[0][0]


@pytest.mark.parametrize(
    "text",
    ["无需澄清，按默认来", "你看着搭配", "不用再问，直接推荐"],
)
def test_initial_delegation_skips_only_soft_preference_questions(
    monkeypatch,
    text: str,
):
    searches = _stub_grounded_outputs(monkeypatch)
    intent = IntentResult.from_raw(
        {
            "r": True,
            "c": "recipe_recommend",
            "k": ["晚餐"],
            "s": {"q": "晚餐", "meal": ["晚餐"]},
            # 即使分类器漏掉 finalize，统一字面 helper 也应生效。
            "signals": {"finalize": False, "no_constraints": True},
        },
        original_text=f"晚餐没忌口，{text}",
    )

    outcome = asyncio.run(router.route_fast_path(
        f"晚餐没忌口，{text}",
        intent_override=intent,
    ))

    assert outcome.kind == "search"
    assert outcome.search_request.constraints_confirmed is True
    assert searches


@pytest.mark.parametrize("dimension", ["party_constraints", "party_constraints_detail"])
def test_finalize_never_bypasses_unknown_party_safety(
    monkeypatch,
    dimension: str,
):
    searches = _stub_grounded_outputs(monkeypatch)
    intent = IntentResult.from_raw(
        {
            "r": False,
            "c": "unknown",
            "k": [],
            "signals": {"finalize": True},
        },
        original_text="你安排",
    )

    outcome = asyncio.run(router.route_fast_path(
        "你安排",
        pending_search_request=_pending_party_request(
            constraints_confirmed=False,
        ).model_dump(),
        pending_clarification_dimension=dimension,
        intent_override=intent,
    ))

    assert outcome.kind == "clarify"
    assert outcome.clarification_dimension == dimension
    assert outcome.search_request.constraints_confirmed is False
    assert searches == []


@pytest.mark.parametrize("text", ["都行", "都可以", "什么都行"])
def test_all_fine_confirms_no_constraints_in_party_safety_context(
    monkeypatch,
    text: str,
):
    searches = _stub_grounded_outputs(monkeypatch)
    intent = IntentResult.from_raw(
        {
            "r": False,
            "c": "off_topic",
            "k": [],
            # 即使分类器误判，安全问题下的确定性短答仍生效。
            "signals": {"no_constraints": False},
        },
        original_text=text,
    )

    outcome = asyncio.run(router.route_fast_path(
        text,
        pending_search_request=_pending_party_request(
            constraints_confirmed=False,
        ).model_dump(),
        pending_clarification_dimension="party_constraints",
        intent_override=intent,
    ))

    assert outcome.kind == "search"
    assert outcome.search_request.constraints_confirmed is True
    assert outcome.search_request.result_limit == 6
    assert searches


@pytest.mark.parametrize("dimension", ["party_constraints", "party_constraints_detail"])
@pytest.mark.parametrize("text", ["随便", "你安排"])
def test_delegation_does_not_confirm_safety_even_if_classifier_claims_no_constraints(
    monkeypatch,
    dimension: str,
    text: str,
):
    searches = _stub_grounded_outputs(monkeypatch)
    intent = IntentResult.from_raw(
        {
            "r": False,
            "c": "off_topic",
            "k": [],
            "signals": {"finalize": True, "no_constraints": True},
        },
        original_text=text,
    )

    outcome = asyncio.run(router.route_fast_path(
        text,
        pending_search_request=_pending_party_request(
            constraints_confirmed=False,
        ).model_dump(),
        pending_clarification_dimension=dimension,
        intent_override=intent,
    ))

    assert outcome.kind == "clarify"
    assert outcome.clarification_dimension == dimension
    assert outcome.search_request.constraints_confirmed is False
    assert searches == []


def test_explicit_no_restrictions_and_finalize_completes_pending_safety(
    monkeypatch,
):
    searches = _stub_grounded_outputs(monkeypatch)
    intent = IntentResult.from_raw(
        {"r": False, "c": "off_topic", "k": [], "signals": {}},
        original_text="没啥忌口，直接推荐吧",
    )

    outcome = asyncio.run(router.route_fast_path(
        "没啥忌口，直接推荐吧",
        pending_search_request=_pending_party_request(
            constraints_confirmed=False,
        ).model_dump(),
        pending_clarification_dimension="party_constraints",
        intent_override=intent,
    ))

    assert outcome.kind == "search"
    assert outcome.search_request.constraints_confirmed is True
    assert outcome.search_request.party_size == 6
    assert outcome.search_request.result_limit == 6
    assert searches


@pytest.mark.parametrize("dimension", ["party_constraints", "party_constraints_detail"])
def test_explicit_no_restrictions_text_overrides_false_classifier_signal(
    monkeypatch,
    dimension: str,
):
    searches = _stub_grounded_outputs(monkeypatch)
    intent = IntentResult.from_raw(
        {
            "r": False,
            "c": "off_topic",
            "k": [],
            "signals": {"finalize": False, "no_constraints": False},
        },
        original_text="没啥忌口，直接推荐吧",
    )

    outcome = asyncio.run(router.route_fast_path(
        "没啥忌口，直接推荐吧",
        pending_search_request=_pending_party_request(
            constraints_confirmed=False,
        ).model_dump(),
        pending_clarification_dimension=dimension,
        intent_override=intent,
    ))

    assert outcome.kind == "search"
    assert outcome.search_request.constraints_confirmed is True
    assert outcome.search_request.party_size == 6
    assert searches


def test_has_constraints_plus_delegation_advances_to_safety_detail(monkeypatch):
    searches = _stub_grounded_outputs(monkeypatch)
    intent = IntentResult.from_raw(
        {
            "r": False,
            "c": "off_topic",
            "k": [],
            "signals": {"finalize": True, "no_constraints": False},
        },
        original_text="有忌口，但具体你安排",
    )

    outcome = asyncio.run(router.route_fast_path(
        "有忌口，但具体你安排",
        pending_search_request=_pending_party_request(
            constraints_confirmed=False,
        ).model_dump(),
        pending_clarification_dimension="party_constraints",
        intent_override=intent,
    ))

    assert outcome.kind == "clarify"
    assert outcome.clarification_dimension == "party_constraints_detail"
    assert outcome.search_request.constraints_confirmed is False
    assert searches == []


@pytest.mark.parametrize(
    "dimension",
    ["recommendation_basics", "available_ingredients"],
)
def test_repeated_complete_request_executes_on_second_turn(
    monkeypatch,
    dimension: str,
):
    searches = _stub_grounded_outputs(monkeypatch)
    original = "今天晚饭吃什么，给我推荐一下"
    pending = SearchRequest(
        original_text=original,
        query="晚餐",
        meals=["晚餐"],
        task_operation="fuzzy_recommend",
    )
    intent = IntentResult.from_raw(
        {"r": False, "c": "off_topic", "k": [], "signals": {}},
        original_text=original,
    )

    outcome = asyncio.run(router.route_fast_path(
        original,
        pending_search_request=pending.model_dump(),
        pending_clarification_dimension=dimension,
        pending_asked_dimensions=[dimension],
        intent_override=intent,
    ))

    assert outcome.kind == "search"
    assert outcome.intent.category.value == "recipe_recommend"
    assert outcome.search_request.meals == ["晚餐"]
    assert searches


def test_repeated_complete_request_overrides_false_topic_change_and_merges_pending(
    monkeypatch,
):
    searches = _stub_grounded_outputs(monkeypatch)
    pending = _pending_party_request(constraints_confirmed=True)
    original = pending.original_text
    intent = IntentResult.from_raw(
        {
            "r": False,
            "c": "off_topic",
            "k": [],
            "signals": {"topic_change": True, "finalize": False},
        },
        original_text=original,
    )

    outcome = asyncio.run(router.route_fast_path(
        original,
        pending_search_request=pending.model_dump(),
        pending_clarification_dimension="flavor_preferences",
        intent_override=intent,
    ))

    assert outcome.kind == "search"
    assert outcome.search_request.constraints_confirmed is True
    assert outcome.search_request.party_size == 6
    assert outcome.search_request.result_limit == 6
    assert outcome.search_request.explicit_result_limit is True
    assert searches and "朋友聚餐" in searches[0][0]


def test_condition_change_merges_pending_instead_of_abandoning_it(monkeypatch):
    searches = _stub_grounded_outputs(monkeypatch)
    text = "口味换成清淡的，按默认直接推荐"
    intent = IntentResult.from_raw(
        {
            "r": True,
            "c": "recipe_recommend",
            "k": ["清淡"],
            "s": {"q": "清淡", "flavor": ["清淡"]},
            "signals": {"finalize": False, "topic_change": False},
        },
        original_text=text,
    )

    outcome = asyncio.run(router.route_fast_path(
        text,
        pending_search_request=_pending_party_request().model_dump(),
        pending_clarification_dimension="flavor_preferences",
        intent_override=intent,
    ))

    assert router._abandons_pending_request(text) is False
    assert outcome.kind == "search"
    assert outcome.search_request.party_size == 6
    assert outcome.search_request.result_limit == 6
    assert outcome.search_request.constraints_confirmed is True
    assert "清淡" in outcome.search_request.flavors
    assert searches


def test_explicit_abandonment_wins_over_embedded_delegation_phrase(monkeypatch):
    searches = _stub_grounded_outputs(monkeypatch)
    text = "算了，换个话题，你看着安排"
    intent = IntentResult.from_raw(
        {"r": False, "c": "off_topic", "k": [], "signals": {}},
        original_text=text,
    )

    outcome = asyncio.run(router.route_fast_path(
        text,
        pending_search_request=_pending_party_request().model_dump(),
        pending_clarification_dimension="flavor_preferences",
        intent_override=intent,
    ))

    assert router._abandons_pending_request(text) is True
    assert router.is_finalize_recommendation_request(text) is True
    assert outcome.kind == "agent"
    assert searches == []
