"""结构化搜索请求、硬约束和候选多样性测试。"""
import asyncio
import pytest

import app.orchestrator.router as router
from app.conversation.service import persistent_preferences_from_text
from app.orchestrator.condition_reducer import (
    looks_like_new_recipe_task,
    reduce_recipe_conditions,
)
from app.orchestrator.intent import IntentResult
from app.orchestrator.search_request import (
    SearchRequest,
    delegates_recommendation_choice,
    exact_recipe_synonym_queries,
    filter_hard_constraint_violations,
    has_explicit_no_dietary_restrictions,
    hard_constraint_notice,
    is_menu_quantity_dissatisfaction,
    remembered_preference_conflicts,
    requested_allergen_conflicts,
)
from app.orchestrator.search_selection import (
    prioritize_exact_matches,
    prioritize_ingredient_coverage,
    prioritize_soft_preferences,
    select_diverse_results,
)
from app.orchestrator.menu_plan import _group_query, build_menu_plan


@pytest.fixture(autouse=True)
def _disable_narrative_model(monkeypatch):
    from app.orchestrator.recommendation_response import RecommendationNarrative

    async def empty_narrative(*_args, **_kwargs):
        return RecommendationNarrative("", "", {}, "")

    monkeypatch.setattr(router, "generate_recommendation_narrative", empty_narrative)


def test_structured_request_keeps_all_constraints():
    raw = {
        "r": True,
        "c": "recipe_search",
        "k": ["湖南", "减脂", "简单"],
        "s": {
            "q": "湖南口味 减脂 简单",
            "cuisine": ["湘菜"],
            "flavor": ["辣"],
            "scene": ["减脂", "简单"],
            "exclude": ["花生"],
        },
    }
    intent = IntentResult.from_raw(raw, original_text="最近减脂，想吃湖南口味，不吃花生，简单一点")
    request = intent.search_request
    assert request.cuisines == ["湘菜"]
    assert request.flavors == ["辣"]
    assert request.exclude == ["花生"]
    query = request.retrieval_query(lang="zh")
    assert "湖南口味 减脂 简单" in query
    assert "花生" not in query


def test_party_allergy_is_atomized_and_unresolved_allergy_still_needs_clarification():
    text = (
        "周末六个人聚餐，想要四菜一汤，家常一点，其中一个朋友不吃辣，"
        "还有一个人对花生过敏，至少要有一道鱼，别太复杂。"
    )
    request = SearchRequest.from_intent_raw({}, original_text=text)
    assert request.exclude == ["花生"]
    assert request.avoid == ["辣"]
    assert request.constraints_confirmed is True

    unresolved = SearchRequest.from_intent_raw({}, original_text="我有食物过敏")
    assert unresolved.exclude == []
    assert unresolved.constraints_confirmed is False

    prawn = SearchRequest.from_intent_raw({}, original_text="对虾过敏")
    assert prawn.exclude == ["对虾"]
    assert prawn.constraints_confirmed is True


def test_hot_weather_directly_recommends_dishes_and_filters_drinks(monkeypatch):
    searches = []

    async def classify(question):
        return IntentResult.from_raw(
            {
                "r": True,
                "c": "recipe_recommend",
                "k": ["菜"],
                "s": {"q": "菜"},
            },
            original_text=question,
        )

    async def search(query, **_kwargs):
        searches.append(query)
        return {
            "success": True,
            "results": [
                {
                    "id": "drink-1",
                    "score": 0.95,
                    "metadata": {
                        "recipe_id": "drink-1",
                        "name": "火龙果蔬果汁",
                        "tags": ["饮品"],
                        "facets": {"meal": ["饮品"]},
                    },
                },
                {
                    "id": "dish-1",
                    "score": 0.9,
                    "metadata": {
                        "recipe_id": "dish-1",
                        "name": "凉拌黄瓜",
                        "tags": ["凉拌"],
                    },
                },
                {
                    "id": "dish-2",
                    "score": 0.85,
                    "metadata": {
                        "recipe_id": "dish-2",
                        "name": "清蒸鲈鱼",
                        "tags": ["蒸"],
                    },
                },
            ],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)

    outcome = asyncio.run(router.route_fast_path(
        "今天天气好晴朗，有些热，吃那些菜好呢，出出主意吧",
    ))

    assert outcome.kind == "search"
    assert outcome.search_request.scenes == ["天热"]
    assert "天热" in searches[0]
    assert [
        item["metadata"]["recipe_id"]
        for item in outcome.search_result["results"]
    ] == ["dish-1", "dish-2"]


def test_temporal_diet_is_confirmed_before_use_and_can_be_skipped(monkeypatch):
    # 该测试 pin 旧固定话术中的具体术语，需关闭 LLM 澄清走 fallback 路径
    import app.orchestrator.recommendation_response as rec_response
    monkeypatch.setattr(rec_response, "_CLARIFICATION_LLM_ENABLED", False)
    searches = []

    async def classify(question):
        if "天气" in question:
            raw = {
                "r": True,
                "c": "recipe_recommend",
                "k": ["菜"],
                "s": {"q": "菜"},
            }
        else:
            raw = {"r": False, "c": "off_topic", "k": []}
        return IntentResult.from_raw(raw, original_text=question)

    async def load_context(_scope):
        return {
            "preferences": {},
            "temporal_dietary_constraints": [{
                "value": "减肥",
                "expires_at": 9_999_999_999,
            }],
            "recent_turns": [],
            "events": [],
        }

    async def search(query, **_kwargs):
        searches.append(query)
        return {
            "success": True,
            "results": [{
                "id": "dish-1",
                "score": 0.9,
                "metadata": {
                    "recipe_id": "dish-1",
                    "name": "凉拌黄瓜",
                    "tags": ["凉拌"],
                },
            }],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)

    first = asyncio.run(router.route_fast_path(
        "今天天气有些热，吃哪些菜好呢",
        memory_context_loader=load_context,
    ))
    assert first.kind == "clarify"
    assert first.clarification_dimension == "remembered_dietary_constraint"
    assert "最近在减肥" in first.clarification_message
    assert searches == []

    accepted = asyncio.run(router.route_fast_path(
        "是的",
        memory_context_loader=load_context,
        pending_search_request=first.search_request.model_dump(),
        pending_clarification_dimension=first.clarification_dimension,
    ))
    assert accepted.kind == "search"
    assert accepted.search_request.scenes == ["天热", "减肥"]

    skipped = asyncio.run(router.route_fast_path(
        "不用了",
        memory_context_loader=load_context,
        pending_search_request=first.search_request.model_dump(),
        pending_clarification_dimension=first.clarification_dimension,
    ))
    assert skipped.kind == "search"
    assert skipped.search_request.scenes == ["天热"]
    assert "减肥" not in skipped.search_query


def test_remembered_dislike_conflict_asks_and_current_exception_is_scoped(
    monkeypatch,
):
    searches = []

    async def classify(question):
        if question == "这顿想吃西餐":
            raw = {
                "r": True,
                "c": "recipe_recommend",
                "k": ["西餐"],
                "s": {"q": "西餐", "cuisine": ["西餐"]},
            }
        else:
            raw = {"r": False, "c": "off_topic", "k": []}
        return IntentResult.from_raw(raw, original_text=question)

    async def load_context(_scope):
        return {
            "preferences": {"dislikes": ["西餐"]},
            "recent_turns": [],
            "events": [],
        }

    async def clarify(**kwargs):
        assert kwargs["dimension"] == "remembered_preference_conflict"
        return "你之前说过不喜欢西餐，这顿要临时作为例外吗？"

    async def search(query, **_kwargs):
        searches.append(query)
        return {
            "success": True,
            "results": [{
                "id": "western-1",
                "score": 0.9,
                "metadata": {
                    "recipe_id": "western-1",
                    "name": "番茄意面",
                    "facets": {"cuisine": ["western"]},
                },
            }],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)

    first = asyncio.run(router.route_fast_path(
        "这顿想吃西餐",
        memory_context_loader=load_context,
    ))
    assert first.kind == "clarify"
    assert first.clarification_dimension == "remembered_preference_conflict"
    assert searches == []

    accepted = asyncio.run(router.route_fast_path(
        "是的",
        memory_context_loader=load_context,
        pending_search_request=first.search_request.model_dump(),
        pending_clarification_dimension=first.clarification_dimension,
    ))
    assert accepted.kind == "search"
    assert accepted.search_request.suppressed_inherited == ["西餐"]
    assert accepted.search_request.exclude_cuisines == []
    assert searches

    declined = asyncio.run(router.route_fast_path(
        "不要",
        memory_context_loader=load_context,
        pending_search_request=first.search_request.model_dump(),
        pending_clarification_dimension=first.clarification_dimension,
    ))
    assert declined.kind == "ambiguous"
    assert declined.clarification_dimension == (
        "remembered_preference_conflict_declined"
    )
    assert "继续避开西餐" in declined.direct_message


def test_old_intent_format_still_extracts_explicit_exclusion():
    request = SearchRequest.from_intent_raw(
        {"r": True, "c": "recipe_search", "k": ["鸡肉", "清淡"]},
        original_text="想吃清淡鸡肉，不吃香菜和花生",
        keywords=["鸡肉", "清淡"],
    )
    assert request.query == "鸡肉 清淡"
    assert request.exclude == ["香菜", "花生"]


def test_broad_flavor_is_repaired_from_ingredient_and_expanded_for_retrieval():
    request = SearchRequest.from_intent_raw(
        {
            "r": True,
            "c": "recipe_search",
            "s": {
                "q": "重口味",
                "ingredient": ["重口味"],
            },
        },
        original_text="推荐点重口味的菜",
    )

    assert request.ingredients == []
    assert request.dishes == []
    assert request.flavors == ["重口味"]
    assert request.has_broad_flavor_direction() is True
    assert all(
        term in request.retrieval_query()
        for term in ("重口味", "香辣", "咸香", "下饭菜")
    )


def test_broad_flavor_search_uses_soft_threshold_and_keeps_grounded_results(monkeypatch):
    searches = []

    async def classify(question):
        return IntentResult.from_raw(
            {
                "r": True,
                "c": "recipe_search",
                "s": {"q": "重口味", "ingredient": ["重口味"]},
            },
            original_text=question,
        )

    async def search(query, **_kwargs):
        searches.append(query)
        return {
            "success": True,
            "results": [
                {
                    "id": "bold-1",
                    "score": 0.18,
                    "metadata": {
                        "recipe_id": "bold-1",
                        "name": "剁椒炒大白菜",
                        "ingredients": ["大白菜", "剁椒酱"],
                        "tags": ["香辣", "咸香", "下饭菜"],
                    },
                },
                {
                    "id": "weak-1",
                    "score": 0.16,
                    "metadata": {
                        "recipe_id": "weak-1",
                        "name": "清炒时蔬",
                        "ingredients": ["时蔬"],
                        "tags": ["清淡"],
                    },
                },
            ],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)
    monkeypatch.setattr(router, "_relevance_threshold", lambda: 0.20)
    monkeypatch.setattr(router, "_broad_relevance_threshold", lambda: 0.17)

    outcome = asyncio.run(router.route_fast_path("推荐点重口味的菜"))

    assert outcome.kind == "search"
    assert searches and all(term in searches[0] for term in ("香辣", "咸香", "下饭菜"))
    assert [item["id"] for item in outcome.search_result["results"]] == ["bold-1"]
    assert outcome.search_result["_relevance_threshold"] == 0.17


def test_current_method_does_not_inherit_previous_dish():
    request = SearchRequest(
        original_text="我想吃炒菜，有啥推荐的",
        query="炒菜",
        methods=["炒"],
    )
    merged = request.merge_recent_context([
        {"role": "user", "content": "番茄炒蛋的做法"},
        {"role": "user", "content": "我想吃炒菜，有啥推荐的"},
    ])
    assert merged.query == "炒菜"
    assert merged.context_note == {}


def test_detail_request_requires_all_joined_ingredients():
    request = SearchRequest.from_intent_raw(
        {
            "s": {
                "q": "西葫芦 鸡蛋",
                "ingredient": ["西葫芦", "鸡蛋"],
            },
        },
        original_text="西葫芦和鸡蛋怎么做",
    )
    assert request.detail_requested is True
    assert request.required_ingredients == ["西葫芦", "鸡蛋"]
    result = filter_hard_constraint_violations({
        "results": [
            {"id": "1", "metadata": {"name": "酸辣西葫芦", "ingredients": ["西葫芦", "辣椒"]}},
            {"id": "2", "metadata": {"name": "西葫芦炒鸡蛋", "ingredients": ["西葫芦", "鸡蛋"]}},
        ],
    }, request)
    assert [item["id"] for item in result["results"]] == ["2"]


def test_disliked_western_cuisine_is_filtered_by_facet_and_not_retrieved():
    # 改法 A：dislikes 走软过滤（avoid），不再硬过滤
    # 旧行为：西餐 → exclude_cuisines → 硬过滤掉
    # 新行为：西餐 → avoid → 软过滤，结果保留但排序靠后
    request = SearchRequest(
        original_text="推荐辣菜",
        query="辣",
        flavors=["辣"],
    ).merge_preferences({"dislikes": ["西餐"]})
    assert request.exclude_cuisines == []  # 不再硬过滤
    assert "西餐" in request.avoid          # 改为软过滤
    assert "西餐" not in request.retrieval_query()
    result = filter_hard_constraint_violations({
        "results": [{
            "id": "western-1",
            "metadata": {
                "name": "西班牙蒜香辣虾",
                "ingredients": ["虾", "蒜"],
                "facets": {"cuisine": ["western"]},
                "tags": ["西餐", "香辣"],
            },
        }],
    }, request)
    # 软过滤不丢弃结果
    assert [r["id"] for r in result["results"]] == ["western-1"]


def test_negative_cuisine_statement_cannot_remain_a_positive_search_direction():
    request = SearchRequest.from_intent_raw(
        {
            "r": True,
            "c": "recipe_recommend",
            "s": {
                "q": "西餐",
                "cuisine": ["西餐"],
            },
        },
        original_text="我不喜欢吃西餐，以后别给我推荐",
    )
    assert request.cuisines == []
    assert request.exclude_cuisines == ["western"]


def test_remembered_preference_conflict_is_detected_before_preferences_merge():
    request = SearchRequest(
        original_text="这顿想吃西餐",
        query="西餐",
        cuisines=["西餐"],
    )
    conflicts = remembered_preference_conflicts(
        request,
        {"dislikes": ["西餐"]},
    )
    assert conflicts == [{
        "bucket": "dislikes",
        "dimension": "cuisine",
        "value": "西餐",
        "canonical": "western",
    }]
    overridden = request.model_copy(
        update={"suppressed_inherited": ["西餐"]},
    )
    assert remembered_preference_conflicts(
        overridden,
        {"dislikes": ["西餐"]},
    ) == []
    spicy = SearchRequest(
        original_text="这顿想吃辣",
        query="辣",
        flavors=["辣"],
    )
    assert remembered_preference_conflicts(
        spicy,
        {"dislikes": ["辣的"]},
    )[0]["value"] == "辣的"
    reverse = SearchRequest(
        original_text="这顿不要西餐",
        query="家常菜",
        exclude_cuisines=["western"],
    )
    assert remembered_preference_conflicts(
        reverse,
        {"likes": ["西餐"]},
    ) == [{
        "bucket": "likes",
        "dimension": "cuisine",
        "value": "西餐",
        "canonical": "western",
    }]
    reverse_override = reverse.model_copy(
        update={"suppressed_inherited": ["西餐"]},
    ).merge_preferences({"likes": ["西餐"]})
    assert reverse_override.soft_preferences == []


def test_applied_preferences_keep_user_visible_provenance():
    # 改法 A：dislikes 走软过滤，审计字段改为 soft_dislike
    request = SearchRequest(
        original_text="推荐几个家常菜",
        query="家常菜",
    ).merge_preferences({
        "likes": ["蒜香"],
        "dislikes": ["西餐"],
    })
    assert request.exclude_cuisines == []      # 不再硬过滤
    assert "西餐" in request.avoid              # 改走软过滤
    assert {
        (item["kind"], item["display"])
        for item in request.applied_memory_constraints
    } == {
        ("soft_dislike", "西餐"),
        ("soft_preference", "蒜香"),
    }
    from app.orchestrator.recommendation_response import (
        RecommendationNarrative,
        _with_core_fact_grounding,
    )

    disclosed = _with_core_fact_grounding(
        RecommendationNarrative(
            "这几道先从做法上比较。",
            "",
            {},
            "",
        ),
        request,
        "zh",
    )
    assert "按你之前明确说过的" in disclosed.opening
    # 改法 A：dislikes 走软过滤，披露文案改为"参考了不喜欢 X"而非"避开 X"
    assert "不喜欢西餐" in disclosed.opening
    assert "喜欢蒜香" in disclosed.opening


def test_cilantro_exclusion_covers_chinese_and_english_aliases_and_seasonings():
    request = SearchRequest(
        original_text="不吃香菜，推荐家常菜",
        query="家常菜",
        exclude=["香菜"],
    )
    result = filter_hard_constraint_violations({
        "results": [
            {
                "id": "zh-alias",
                "metadata": {
                    "name": "芫荽拌鸡丝",
                    "ingredients": ["鸡肉"],
                    "seasonings": ["芫荽"],
                },
            },
            {
                "id": "en-alias",
                "metadata": {
                    "name": "Lime chicken",
                    "ingredients": ["chicken"],
                    "seasonings": ["coriander leaves"],
                },
            },
            {
                "id": "safe",
                "metadata": {
                    "name": "葱油鸡丝",
                    "ingredients": ["鸡肉"],
                    "seasonings": ["葱", "盐"],
                },
            },
        ],
    }, request)
    assert [item["id"] for item in result["results"]] == ["safe"]


def test_halal_request_is_deterministic_and_filters_pork_and_alcohol():
    request = SearchRequest.from_intent_raw(
        {
            "c": "recipe_recommend",
            "s": {"q": "家常菜", "diet": []},
        },
        original_text="我是回族，推荐几道家常菜",
    )
    assert "halal" in request.dietary_constraints
    result = filter_hard_constraint_violations({
        "results": [
            {
                "id": "pork",
                "metadata": {
                    "name": "木须肉",
                    "ingredients": ["猪肉", "鸡蛋", "木耳"],
                },
            },
            {
                "id": "wine",
                "metadata": {
                    "name": "红烧鸡块",
                    "ingredients": ["鸡肉"],
                    "seasonings": ["生抽", "料酒"],
                },
            },
            {
                "id": "safe",
                "metadata": {
                    "name": "孜然羊肉",
                    "ingredients": ["羊肉", "洋葱"],
                    "seasonings": ["孜然", "盐"],
                },
            },
        ],
    }, request)
    assert [item["id"] for item in result["results"]] == ["safe"]
    notice = hard_constraint_notice(request, "zh")
    assert "不具备清真认证" in notice
    assert "猪肉、猪油、猪骨和酒类" in notice


def test_guest_light_preference_uses_one_scoped_slot_and_keeps_user_spicy_preference():
    request = SearchRequest.from_intent_raw(
        {
            "c": "recipe_recommend",
            "s": {
                "q": "三人聚餐 清淡",
                "flavor": ["清淡"],
                "scene": ["聚餐"],
            },
        },
        original_text="三个人吃饭，其中一人清淡，另外两人随意",
    )
    assert request.party_size == 3
    assert request.flavors == []
    assert request.scoped_preferences == [{
        "scope": "guest_subset",
        "count": 1,
        "max_slots": 1,
        "cuisines": [],
        "flavors": ["清淡"],
        "source_text": "三个人吃饭，其中一人清淡，另外两人随意",
    }]

    planned = request.merge_preferences({"likes": ["辣"]}).with_recommendation_menu_defaults()
    assert planned.soft_preferences == ["辣"]
    assert planned.menu_dish_count == 3
    scoped_query = _group_query(planned, "scoped_dish", planned.scoped_preferences[0], "zh")
    regular_query = _group_query(planned, "dish", None, "zh")
    assert "清淡" in scoped_query
    assert "辣" not in scoped_query
    assert "辣" in regular_query
    assert "清淡" not in regular_query


def test_business_operation_is_frozen_after_standardization():
    exact = SearchRequest.from_intent_raw(
        {"c": "recipe_search", "s": {"q": "红烧肉", "dish": ["红烧肉"]}},
        original_text="红烧肉怎么做",
    ).with_task_operation("search").freeze_canonical_question()
    fuzzy_search = SearchRequest.from_intent_raw(
        {"c": "recipe_search", "s": {"q": "鸡肉", "ingredient": ["鸡肉"]}},
        original_text="找一些鸡肉菜",
    ).with_task_operation("search").freeze_canonical_question()
    fuzzy_recommend = SearchRequest.from_intent_raw(
        {"c": "recipe_recommend", "s": {"q": "辣", "flavor": ["辣"]}},
        original_text="推荐辣菜",
    ).with_task_operation("recommend").freeze_canonical_question()
    assert exact.task_operation == "exact_search"
    assert exact.canonical_question == "红烧肉"
    assert fuzzy_search.task_operation == "fuzzy_search"
    assert fuzzy_recommend.task_operation == "fuzzy_recommend"


def test_negative_only_fuzzy_search_requires_a_positive_fact():
    request = SearchRequest.from_intent_raw(
        {"c": "recipe_search", "s": {"q": "", "exclude": ["香菜"]}},
        original_text="不吃香菜，给我找菜谱",
    ).with_task_operation("search")
    assert request.exclude == ["香菜"]
    assert request.canonical_question == ""
    assert request.search_clarification_dimension() == "ingredient_or_flavor"


def test_explicit_detail_reference_freezes_recent_topic_as_dish():
    request = SearchRequest.from_intent_raw(
        {"c": "recipe_search", "s": {"q": ""}},
        original_text="这个怎么做",
    )
    merged = request.merge_recent_context([
        {"role": "user", "content": "红烧肉"},
        {"role": "user", "content": "这个怎么做"},
    ]).with_task_operation("search").freeze_canonical_question()
    assert merged.context_note["term"] == "红烧肉"
    assert merged.dishes == ["红烧肉"]
    assert merged.task_operation == "exact_search"
    assert merged.canonical_question == "红烧肉"


def test_change_batch_feedback_turns_not_down_rice_into_current_search_preference():
    request = SearchRequest(
        original_text="我想吃牛肉类的菜",
        query="牛肉",
        ingredients=["牛肉"],
    )
    refined = request.refine("一点都不下饭，换一批")
    assert "下饭" in refined.flavors
    assert "下饭" in refined.retrieval_query(lang="zh")
    # 这是对当前候选的反馈，不应在这里伪装成全局长期画像。
    assert refined.soft_preferences == []


def test_result_limit_follows_party_size_or_explicit_count_and_caps_at_ten():
    family = SearchRequest.from_intent_raw(
        {"r": True, "c": "recipe_search", "k": ["湘菜"]},
        original_text="我们一家四口，都是湖南人，推荐些湘菜",
        keywords=["湘菜"],
    )
    assert family.party_size == 4
    assert family.result_limit == 4
    assert family.ingredients == []

    explicit = SearchRequest.from_intent_raw(
        {"r": True, "c": "recipe_search", "k": ["家常菜"]},
        original_text="给我推荐12道家常菜",
        keywords=["家常菜"],
    )
    assert explicit.result_limit == 10


def test_recipe_list_container_is_not_misread_as_one_recipe():
    recipe_list = SearchRequest.from_intent_raw(
        {"c": "recipe_recommend", "s": {}},
        original_text="请推荐一个食谱清单给我",
    )
    assert recipe_list.result_limit == 3
    assert recipe_list.explicit_result_limit is False

    one_recipe = SearchRequest.from_intent_raw(
        {"c": "recipe_recommend", "s": {}},
        original_text="请推荐一个食谱给我",
    )
    assert one_recipe.result_limit == 1
    assert one_recipe.explicit_result_limit is True


@pytest.mark.parametrize(
    "text",
    [
        "大家都没什么忌口",
        "大家没有啥忌口",
        "所有人都没有任何过敏",
        "饮食没什么限制",
    ],
)
def test_no_dietary_restriction_variants_are_confirmed(text):
    request = SearchRequest.from_intent_raw({}, original_text=text)
    assert request.constraints_confirmed is True


@pytest.mark.parametrize(
    "text",
    [
        "不知道有没有过敏",
        "不确定有没有忌口",
        "不清楚有没有人过敏",
        "有没有忌口我还不知道",
        "忌口还没问清楚",
        "不是没有忌口",
    ],
)
def test_unknown_or_negated_safety_facts_are_not_confirmed(text):
    request = SearchRequest.from_intent_raw({}, original_text=text)
    assert has_explicit_no_dietary_restrictions(text) is False
    assert request.constraints_confirmed is False
    assert request.exclude == []


@pytest.mark.parametrize(
    "text",
    ["没有忌口", "没过敏", "我们啥都不忌口，你安排就好"],
)
def test_explicit_no_restriction_helper_keeps_clear_confirmations(text):
    request = SearchRequest.from_intent_raw({}, original_text=text)
    assert has_explicit_no_dietary_restrictions(text) is True
    assert request.constraints_confirmed is True
    assert request.exclude == []


def test_constraint_extraction_stops_before_delegation_tail():
    for text in ("不要花生别再问", "不要花生，别再问"):
        request = SearchRequest.from_intent_raw({}, original_text=text)
        assert request.exclude == ["花生"]

    unresolved = SearchRequest.from_intent_raw(
        {},
        original_text="六个人聚餐，有忌口但你安排",
    )
    assert unresolved.constraints_confirmed is False
    assert unresolved.exclude == []
    assert unresolved.recommendation_clarification_dimension() == "party_constraints"


def test_exact_baijiu_party_request_keeps_menu_scale_and_safety_facts():
    text = (
        "我在杭州，今天晚上有10个朋友来我家吃饭，都是江西人，"
        "要喝点白酒，所有人都没什么忌口，请推荐一个食谱清单给我"
    )
    request = SearchRequest.from_intent_raw(
        {"c": "recipe_recommend", "s": {}},
        original_text=text,
    )

    assert request.party_size == 11
    assert request.constraints_confirmed is True
    assert request.scenes == ["下酒"]
    assert request.result_limit == 10
    assert request.explicit_result_limit is False

    planned = request.with_recommendation_menu_defaults()
    assert planned.menu_dish_count == 6
    assert planned.menu_soup_count == 2
    assert planned.result_limit == 8


@pytest.mark.parametrize(
    "text",
    [
        "10个人推荐五个菜。。。太少了",
        "10个人才五道菜，根本不够吃",
        "only five dishes is not enough for 10 people",
    ],
)
def test_menu_quantity_dissatisfaction_is_not_a_new_explicit_limit(text):
    request = SearchRequest.from_intent_raw(
        {
            "c": "recipe_recommend",
            "s": {"q": "五个菜"},
        },
        original_text=text,
    )

    assert is_menu_quantity_dissatisfaction(text) is True
    assert request.explicit_result_limit is False
    assert request.result_limit != 5


def test_explicit_requested_count_still_wins_without_dissatisfaction():
    request = SearchRequest.from_intent_raw(
        {"c": "recipe_recommend", "s": {}},
        original_text="10个人吃，请给我八道菜",
    )

    assert request.party_size == 10
    assert request.result_limit == 8
    assert request.explicit_result_limit is True


def test_location_and_origin_do_not_infer_cuisine_or_flavor():
    text = (
        "我在杭州，今天晚上有10个朋友来我家吃饭，都是江西人，"
        "要喝点白酒，所有人都没什么忌口，请推荐一个食谱清单给我"
    )
    request = SearchRequest.from_intent_raw(
        {
            "c": "recipe_recommend",
            "s": {
                "q": "杭州 江西口味 白酒配餐 10人 晚餐",
                "cuisine": ["赣菜"],
                "flavor": ["辣"],
                "scene": ["杭州", "江西人", "白酒"],
                "meal": ["晚餐"],
            },
        },
        original_text=text,
    )

    assert request.cuisines == []
    assert request.flavors == []
    assert request.scenes == ["下酒"]
    assert request.meals == ["晚餐"]
    assert request.query == ""
    assert request.retrieval_query() == "下酒 晚餐"
    assert request.freeze_canonical_question().canonical_question == "下酒 晚餐"
    assert request.recommendation_clarification_dimension() is None


def test_explicit_actionable_scenes_survive_scene_sanitization():
    request = SearchRequest.from_intent_raw(
        {
            "c": "recipe_recommend",
            "s": {
                "q": "运动后 快手 晚餐",
                "scene": ["运动后", "快手"],
                "meal": ["晚餐"],
            },
        },
        original_text="运动后想要快手晚餐，直接推荐",
    )

    assert request.scenes == ["运动后", "快手"]
    assert request.meals == ["晚餐"]
    assert request.retrieval_query() == "运动后 快手 晚餐"


def test_explicit_regional_food_request_still_keeps_normalized_cuisine():
    request = SearchRequest.from_intent_raw(
        {
            "c": "recipe_recommend",
            "s": {
                "q": "湖南口味",
                "cuisine": ["湘菜"],
                "flavor": ["辣"],
            },
        },
        original_text="最近减脂，想吃湖南口味",
    )

    assert request.cuisines == ["湘菜"]
    assert request.flavors == ["辣"]


def test_past_drinking_discomfort_is_not_a_drinks_pairing_scene():
    request = SearchRequest.from_intent_raw(
        {},
        original_text="昨天喝白酒喝多了，今天胃不舒服",
    )
    assert "下酒" not in request.scenes


def test_english_number_words_parse_party_size_and_menu_counts():
    request = SearchRequest.from_intent_raw(
        {},
        original_text=(
            "There will be three of us. Please plan three dishes and one soup."
        ),
        keywords=[],
    )
    assert request.party_size == 3
    assert request.menu_dish_count == 3
    assert request.menu_soup_count == 1
    assert request.is_menu_plan


def test_explicit_english_ingredient_survives_sparse_intent_payload():
    request = SearchRequest.from_intent_raw(
        {"s": {"q": "beef", "ingredient": ["beef"]}},
        original_text="Recommend three beef recipes I can cook at home.",
        keywords=["beef"],
    )
    assert request.ingredients == ["beef"]
    assert request.recommendation_clarification_dimension() is None


def test_guest_count_includes_host_and_corrects_model_query():
    request = SearchRequest.from_intent_raw(
        {"s": {"q": "三人餐", "scene": ["聚会"]}},
        original_text="我有三个朋友来吃饭餐，我应该准备什么？",
        keywords=["三人餐"],
    )
    assert request.party_size == 4
    assert request.result_limit == 4
    assert request.query == "4人餐"


def test_friends_coming_for_dinner_without_chifan_word_is_still_party_size():
    request = SearchRequest.from_intent_raw(
        {"s": {"q": "朋友聚会 晚餐", "scene": ["朋友聚会", "晚餐"]}},
        original_text="今晚会来五个朋友，吃那些菜呢",
        keywords=["朋友聚会", "晚餐"],
    )
    assert request.party_size == 6
    assert request.result_limit == 6


@pytest.mark.parametrize(
    ("search_request", "expected_dimension"),
    [
        (
            SearchRequest(
                original_text="晚饭吃什么",
                query="晚饭",
                meals=["晚饭"],
            ),
            "available_ingredients",
        ),
        (
            SearchRequest(
                original_text="我有三个朋友来聚餐，该准备什么",
                query="朋友聚餐",
                scenes=["朋友聚餐"],
                party_size=4,
            ),
            "party_preferences",
        ),
    ],
)
def test_generic_meal_or_party_recommendation_needs_clarification(
    search_request, expected_dimension,
):
    assert search_request.clarification_dimension() == expected_dimension


def test_party_context_asks_only_safety_then_uses_default_preferences():
    initial = SearchRequest.from_intent_raw(
        {
            "c": "recipe_recommend",
            "s": {
                "q": "上海 夏天 朋友聚餐",
                "scene": ["上海", "夏天", "朋友聚餐"],
            },
        },
        original_text="夏天在上海招待三个朋友，推荐吃什么",
    )
    assert initial.party_size == 4
    assert initial.recommendation_clarification_dimension() == "party_constraints"

    no_restrictions = SearchRequest.from_intent_raw(
        {"c": "cooking_qa", "s": {"q": ""}},
        original_text="没有忌口，也没有过敏",
    )
    after_constraints = initial.merge_clarification(no_restrictions)
    assert after_constraints.constraints_confirmed is True
    assert after_constraints.recommendation_clarification_dimension() is None


@pytest.mark.parametrize(
    "text",
    [
        "随便推荐",
        "你安排",
        "直接推荐",
        "无需澄清",
        "不需要澄清",
        "按默认来",
        "你看着安排",
        "不要再问了",
        "口味无所谓",
    ],
)
def test_delegation_phrases_share_one_policy(text):
    assert delegates_recommendation_choice(text) is True
    assert SearchRequest(
        original_text=text,
        query="推荐",
    ).clarification_dimension() is None
    assert SearchRequest(
        original_text=text,
        query="推荐",
    ).search_clarification_dimension() is None


@pytest.mark.parametrize(
    "text",
    ["随便", "你安排", "直接推荐", "无需澄清", "按默认来", "不要再问了"],
)
def test_delegation_never_bypasses_unknown_party_safety(text):
    request = SearchRequest(
        original_text=f"六个人聚餐，{text}",
        query="六人餐",
        party_size=6,
        scenes=["聚餐"],
    )
    assert request.recommendation_clarification_dimension() == "party_constraints"


def test_confirmed_no_restrictions_directly_recommends_without_party_size():
    request = SearchRequest.from_intent_raw(
        {"c": "recipe_recommend", "s": {}},
        original_text="没忌口，请推荐一个食谱清单",
    )
    assert request.party_size is None
    assert request.constraints_confirmed is True
    assert request.recommendation_clarification_dimension() is None


def test_stop_clarifying_phrase_is_not_parsed_as_excluded_ingredient():
    request = SearchRequest.from_intent_raw(
        {"c": "recipe_recommend", "s": {}},
        original_text="不要再问了，给我菜单",
    )
    assert delegates_recommendation_choice(request.original_text) is True
    assert request.exclude == []
    assert request.avoid == []

    mixed = SearchRequest.from_intent_raw(
        {"c": "recipe_recommend", "s": {}},
        original_text="不要花生，别再问了，直接推荐",
    )
    assert mixed.exclude == ["花生"]


@pytest.mark.parametrize(
    "search_request",
    [
        SearchRequest(
            original_text="想吃清淡一点的鸡肉菜",
            query="清淡 鸡肉",
            ingredients=["鸡肉"],
            flavors=["清淡"],
        ),
        SearchRequest(
            original_text="随便，直接推荐几道菜",
            query="推荐",
        ),
    ],
)
def test_specific_or_explicitly_random_recommendation_skips_clarification(search_request):
    assert search_request.clarification_dimension() is None


def test_negative_constraints_are_not_kept_as_positive_facets():
    request = SearchRequest.from_intent_raw(
        {
            "r": True,
            "c": "recipe_search",
            "k": ["家常菜"],
            "s": {"q": "家常菜", "meal": ["汤"], "flavor": ["甜"]},
        },
        original_text="想吃家常菜，不要汤，也不要甜的",
        keywords=["家常菜"],
    )
    assert "汤" not in request.meals
    assert "甜" not in request.flavors


def test_scene_does_not_infer_health_goals_or_exclusions():
    request = SearchRequest.from_intent_raw(
        {
            "r": True,
            "c": "recipe_recommend",
            "k": ["运动后"],
            "s": {
                "q": "运动后 晚餐 鸡胸肉",
                "ingredient": ["鸡胸肉"],
                "scene": ["运动后"],
            },
        },
        original_text="运动后晚饭吃什么",
        keywords=["运动后"],
    )
    assert request.query == "运动后"
    assert request.scenes == ["运动后"]
    assert request.ingredients == []
    assert request.exclude == []
    assert request.avoid == []


def test_explicit_food_leads_query_across_arbitrary_scenes():
    request = SearchRequest.from_intent_raw(
        {
            "r": True,
            "c": "recipe_recommend",
            "k": ["鸡胸肉", "简单"],
            "s": {
                "q": "加班 简单 晚餐 鸡胸肉",
                "ingredients": ["鸡胸肉"],
                "scene": ["加班", "简单"],
                "meal": ["晚餐"],
            },
        },
        original_text="加班回家，冰箱有鸡胸肉，想简单做点晚饭",
        keywords=["鸡胸肉", "简单"],
    )

    assert request.query == "加班 简单 鸡胸肉"
    assert request.ingredients == ["鸡胸肉"]
    assert request.retrieval_query(lang="zh") == "鸡胸肉 加班 简单 晚餐"
    assert request.exclude == []


def test_soft_negative_flavor_is_not_treated_as_allergen():
    request = SearchRequest.from_intent_raw(
        {"r": True, "c": "recipe_search", "k": ["鸡肉"]},
        original_text="想吃鸡肉，但是不要太辣",
        keywords=["鸡肉"],
    )
    assert request.exclude == []
    assert request.avoid == ["太辣"]
    assert "不要太辣" not in request.retrieval_query(lang="zh")
    assert "微辣" in request.retrieval_query(lang="zh")


def test_absolute_no_spicy_filter_removes_spicy_candidates():
    request = SearchRequest(
        original_text="推荐鸡肉菜，不要辣",
        query="鸡肉",
        ingredients=["鸡肉"],
        avoid=["辣"],
    )
    result = filter_hard_constraint_violations({
        "success": True,
        "results": [
            {
                "id": "spicy",
                "metadata": {
                    "recipe_id": "spicy",
                    "name": "香辣鸡丁",
                    "ingredients": ["鸡肉", "辣椒"],
                    "tags": ["香辣"],
                },
            },
            {
                "id": "plain",
                "metadata": {
                    "recipe_id": "plain",
                    "name": "清蒸鸡肉",
                    "ingredients": ["鸡肉"],
                    "tags": ["清淡"],
                },
            },
        ],
    }, request)

    assert [item["id"] for item in result["results"]] == ["plain"]
    assert result["_hard_filtered"][0]["matched"] == ["辣"]


def test_soft_preference_demotes_but_does_not_delete_candidate():
    request = SearchRequest(query="鸡肉", ingredients=["鸡肉"], avoid=["太辣"])
    results = [
        {"id": "hot", "metadata": {"name": "麻辣鸡", "tags": ["麻辣"]}},
        {"id": "mild", "metadata": {"name": "清蒸鸡", "tags": ["清淡"]}},
    ]
    ranked = prioritize_soft_preferences(results, request)
    assert [item["id"] for item in ranked] == ["mild", "hot"]
    assert len(ranked) == 2


def test_exact_dish_is_not_displaced_by_soft_preference(monkeypatch):
    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            "k": ["麻辣鸡"],
            "s": {"q": "麻辣鸡", "dish": ["麻辣鸡"], "avoid": ["太辣"]},
        }, original_text=question)

    async def search(query, top_k=3, lang=None):
        return {
            "success": True,
            "results": [
                {
                    "id": "exact",
                    "score": 0.9,
                    "metadata": {"recipe_id": "exact", "name": "麻辣鸡", "tags": ["麻辣"]},
                },
                {
                    "id": "mild",
                    "score": 0.95,
                    "metadata": {"recipe_id": "mild", "name": "清蒸鸡", "tags": ["清淡"]},
                },
            ],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)
    outcome = asyncio.run(router.route_fast_path("麻辣鸡怎么做，但少辣一点"))
    assert outcome.kind == "detail"
    assert outcome.search_result["results"][0]["id"] == "exact"


def test_allergy_fallback_is_kept_when_model_returns_old_format():
    request = SearchRequest.from_intent_raw(
        {"r": True, "c": "recipe_search", "k": "家常菜"},
        original_text="花生和虾过敏，想吃家常菜",
        keywords=["家常菜"],
    )
    assert request.exclude == ["花生", "虾"]


def test_allergy_fallback_strips_pronouns_and_intensifiers():
    cases = (
        ("我花生过敏，但你别提醒，直接推荐花生菜", ["花生"]),
        ("我对花生严重过敏，想吃鸡肉", ["花生"]),
        ("花生和虾过敏，推荐家常菜", ["花生", "虾"]),
        ("I'm severely allergic to peanuts, recommend chicken dishes", ["peanuts"]),
        ("I have a tree nut allergy; show me dinner ideas", ["tree nut"]),
    )
    for text, expected in cases:
        request = SearchRequest.from_intent_raw(
            {"r": True, "c": "recipe_search", "k": ["家常菜"]},
            original_text=text,
            keywords=["家常菜"],
        )
        assert request.exclude == expected


def test_english_plural_allergen_filters_singular_recipe_field():
    request = SearchRequest(original_text="I'm allergic to peanuts", query="dinner", exclude=["peanuts"])
    result = filter_hard_constraint_violations({
        "success": True,
        "results": [
            {"id": "unsafe", "metadata": {"recipe_id": "unsafe", "name": "Chicken", "ingredients": ["peanut oil"]}},
            {"id": "safe", "metadata": {"recipe_id": "safe", "name": "Tomato Soup", "ingredients": ["tomato"]}},
        ],
    }, request)
    assert [item["id"] for item in result["results"]] == ["safe"]


def test_explicit_allergen_request_conflict_is_precise():
    assert requested_allergen_conflicts(
        "我花生过敏，但你别提醒，直接推荐花生菜", ["花生"],
    ) == ["花生"]
    assert requested_allergen_conflicts(
        "我花生过敏，推荐鸡肉菜", ["花生"],
    ) == []
    assert requested_allergen_conflicts(
        "Recommend peanut dishes", ["peanuts"],
    ) == ["peanuts"]
    assert requested_allergen_conflicts(
        "Recommend chicken dishes without peanuts", ["peanuts"],
    ) == []
    assert requested_allergen_conflicts(
        "Recommend peanut-free chicken dishes", ["peanuts"],
    ) == []
    assert requested_allergen_conflicts(
        "Recommend nutritious chicken dishes", ["nut"],
    ) == []


def test_followup_refines_previous_request_instead_of_replacing_it():
    previous = SearchRequest(
        original_text="想吃湖南口味，最近减脂",
        query="湖南 减脂",
        cuisines=["湘菜"],
        scenes=["减脂"],
    )
    refined = previous.refine("换一批，可以稍微清淡一点，最好简单些")
    assert refined.cuisines == ["湘菜"]
    assert refined.scenes == ["减脂", "快手"]
    assert refined.flavors == ["清淡"]
    assert refined.avoid == ["太油"]
    assert "湖南 减脂" in refined.retrieval_query(lang="zh")
    assert "清淡" in refined.retrieval_query(lang="zh")


def test_condition_reducer_accumulates_recipe_task_filters():
    request = SearchRequest(
        original_text="推荐几个鸡肉菜",
        query="鸡肉",
        ingredients=["鸡肉"],
    )

    for followup in ("不要辣的", "孩子也能吃", "简单一点"):
        reduction = reduce_recipe_conditions(request, followup)
        assert reduction is not None
        request = reduction.request

    assert request.ingredients == ["鸡肉"]
    assert request.exclude == []
    assert request.avoid == ["辣"]
    assert request.scenes == ["儿童", "快手"]
    assert request.retrieval_query(lang="zh") == "鸡肉 儿童 快手"


def test_explicit_new_recommendation_does_not_refine_previous_search():
    previous = SearchRequest(
        original_text="我想吃点辣菜，但是热量别太高，最近在减肥",
        query="辣 减肥 热量",
        task_operation="fuzzy_recommend",
        flavors=["辣"],
        scenes=["减肥"],
    )
    question = (
        "有两个朋友一会儿过来，给我推荐一些菜吧。我们都习惯吃辣，"
        "还要适合下酒，而且简单一点。要有一道鱼"
    )

    assert looks_like_new_recipe_task(question) is True
    assert reduce_recipe_conditions(previous, question) is None

    request = SearchRequest.from_intent_raw(
        {
            "c": "recipe_recommend",
            "s": {
                "q": "辣 下酒 简单 鱼",
                "ingredient": ["鱼"],
                "flavor": ["辣"],
                "scene": ["下酒", "简单"],
            },
        },
        original_text=question,
    )
    assert request.party_size == 3
    assert request.required_ingredients == ["鱼"]
    assert request.ingredients == []
    assert request.flavors == ["辣"]
    assert "下酒" in request.scenes
    assert any(item in request.scenes for item in ("简单", "快手"))

    merged = request.merge_preferences({
        "dislikes": ["辣的"],
        "dietary_constraints": ["减肥"],
    }).with_task_operation("recommend")
    assert merged.exclude == []
    assert merged.suppressed_inherited == ["辣的"]
    assert "减肥" not in merged.scenes
    assert merged.task_operation == "fuzzy_recommend"

    planned = merged.with_recommendation_menu_defaults()
    assert planned.menu_dish_count == 3
    assert planned.menu_soup_count == 1
    assert planned.result_limit == 4


def test_group_meal_flavor_does_not_become_account_preference():
    extracted = persistent_preferences_from_text(
        "有两个朋友一会儿过来，我们都习惯吃辣，要简单一点"
    )
    assert extracted == {
        "likes": [],
        "dislikes": [],
        "allergens": [],
        "dietary_constraints": [],
    }


def test_spicy_current_request_suppresses_old_dislike_but_generic_request_filters_it():
    current = SearchRequest(
        original_text="这顿想吃辣",
        query="辣",
        flavors=["辣"],
    ).merge_preferences({"dislikes": ["辣的"]})
    assert current.exclude == []
    assert current.suppressed_inherited == ["辣的"]

    generic = SearchRequest(
        original_text="推荐几个菜",
        query="家常菜",
    ).merge_preferences({"dislikes": ["辣的"]})
    # 改法 A：dislikes 走软过滤（avoid），不再硬过滤
    assert "辣的" in generic.avoid
    assert "辣的" not in generic.exclude
    filtered = filter_hard_constraint_violations({
        "results": [{
            "id": "spicy",
            "metadata": {
                "name": "轻盈版辣炒青菜",
                "ingredients": ["干辣椒段"],
                "tags": ["香辣"],
            },
        }],
    }, generic)
    # 软过滤不丢弃结果
    assert [r["id"] for r in filtered["results"]] == ["spicy"]


def test_menu_plan_allocates_one_verified_required_ingredient_slot():
    request = SearchRequest(
        original_text="三个人吃，想吃辣的简单菜，要有一道鱼",
        query="辣 简单",
        task_operation="fuzzy_recommend",
        flavors=["辣"],
        scenes=["快手"],
        required_ingredients=["鱼"],
        party_size=3,
        result_limit=4,
        menu_dish_count=3,
        menu_soup_count=1,
    )

    async def search(query, _top_k, _lang):
        if query.startswith("鱼 "):
            results = [{
                "id": "fish",
                "metadata": {
                    "recipe_id": "fish",
                    "name": "香辣鲈鱼",
                    "ingredients": ["鲈鱼", "辣椒"],
                    "facets": {
                        "main_ingredient": ["fish"],
                        "meal": ["main_course"],
                    },
                },
            }]
        elif query.startswith("汤 "):
            results = [{
                "id": "soup",
                "metadata": {
                    "recipe_id": "soup",
                    "name": "番茄蛋花汤",
                    "ingredients": ["番茄", "鸡蛋"],
                    "facets": {"meal": ["soup"]},
                },
            }]
        else:
            results = [
                {
                    "id": "dish-1",
                    "metadata": {
                        "recipe_id": "dish-1",
                        "name": "辣炒青菜",
                        "ingredients": ["青菜", "辣椒"],
                        "facets": {"meal": ["main_course"]},
                    },
                },
                {
                    "id": "dish-2",
                    "metadata": {
                        "recipe_id": "dish-2",
                        "name": "香辣鸡丝",
                        "ingredients": ["鸡肉", "辣椒"],
                        "facets": {"meal": ["main_course"]},
                    },
                },
            ]
        return {"success": True, "results": results}

    result = asyncio.run(build_menu_plan(request, search=search))
    assert result["_menu_plan"]["complete"] is True
    assert len(result["results"]) == 4
    fish = next(item for item in result["results"] if item["id"] == "fish")
    assert fish["menu_role"] == "required_dish"
    assert fish["menu_requirement"] == "鱼"


def test_condition_reducer_replaces_spicy_constraint_then_adds_exclusion():
    request = SearchRequest(
        original_text="推荐不辣的鸡肉菜",
        query="鸡肉 不辣",
        ingredients=["鸡肉"],
        avoid=["辣"],
    )

    changed = reduce_recipe_conditions(request, "算了，稍微辣一点也可以")
    assert changed is not None
    assert changed.operation == "replace"
    request = changed.request
    assert request.ingredients == ["鸡肉"]
    assert request.flavors == ["微辣"]
    assert request.avoid == ["太辣"]
    assert "不辣" not in request.retrieval_query(lang="zh")

    excluded = reduce_recipe_conditions(request, "不要牛肉")
    assert excluded is not None
    request = excluded.request
    assert request.ingredients == ["鸡肉"]
    assert request.exclude == ["牛肉"]
    assert request.flavors == ["微辣"]


def test_english_spicy_revision_removes_previous_absolute_avoidance():
    request = SearchRequest(
        original_text="easy chicken, not spicy",
        query="chicken not spicy",
        ingredients=["chicken"],
        avoid=["spicy"],
    )

    changed = reduce_recipe_conditions(
        request,
        "actually, mildly spicy is okay",
    )
    assert changed is not None
    assert changed.request.flavors == ["mild"]
    assert changed.request.avoid == ["too spicy"]
    assert "not spicy" not in changed.request.retrieval_query(lang="en")


def test_memory_preferences_only_enrich_generic_request():
    # 改法 A：dislikes 走软过滤（avoid），不再硬过滤
    preferences = {
        "likes": ["辣"],
        "dislikes": ["香菜"],
        "dietary_constraints": ["减脂"],
        "available_ingredients": ["鸡蛋"],
    }
    generic = SearchRequest(original_text="晚饭吃什么", query="晚饭").merge_preferences(preferences)
    assert generic.soft_preferences == ["辣"]
    assert generic.ingredients == ["鸡蛋"]
    assert generic.exclude == []          # 不再硬过滤
    assert "香菜" in generic.avoid         # 改走软过滤

    explicit = SearchRequest(
        original_text="想吃清淡鸡肉", query="清淡 鸡肉", flavors=["清淡"], ingredients=["鸡肉"]
    ).merge_preferences(preferences)
    assert "辣" not in explicit.soft_preferences
    assert explicit.ingredients == ["鸡肉"]
    assert explicit.exclude == []          # 不再硬过滤
    assert "香菜" in explicit.avoid         # 改走软过滤
    assert explicit.dietary_constraints == []


def test_food_history_is_audit_only_and_does_not_override_current_request():
    memory = {
        "events": [{
            "kind": "cooked",
            "recipe": {
                "id": "f1", "name": "清蒸鲈鱼",
                "ingredients": ["鲈鱼", "姜"], "tags": ["清蒸"],
            },
        }],
    }
    garlic = SearchRequest(
        original_text="我想吃大蒜多的菜", query="大蒜多的菜", ingredients=["大蒜"]
    ).merge_food_memory(memory)
    assert garlic.memory_terms == ["鱼"]
    assert garlic.memory_note["kind"] == "cooked"
    assert "大蒜" in garlic.retrieval_query(lang="zh")
    assert "鱼" not in garlic.retrieval_query(lang="zh")

    explicit = SearchRequest(
        original_text="我想吃大蒜牛肉", query="大蒜 牛肉", ingredients=["大蒜", "牛肉"]
    ).merge_food_memory(memory)
    assert explicit.memory_terms == []
    assert "鱼" not in explicit.retrieval_query(lang="zh")


def test_english_food_history_uses_english_memory_term():
    request = SearchRequest(
        original_text="something with lots of garlic", query="lots of garlic", ingredients=["garlic"]
    ).merge_food_memory({
        "events": [{
            "kind": "searched", "query": "fish dinner", "original_question": "show me fish",
            "recipes": [{"name": "Steamed Fish", "ingredients": ["fish"]}],
        }],
    })
    assert request.memory_terms == ["fish"]
    assert request.retrieval_query(lang="en") == "garlic lots of"


def test_old_spicy_search_can_personalize_new_garlic_request():
    request = SearchRequest(
        original_text="我想吃大蒜，有啥菜可以推荐", query="大蒜", ingredients=["大蒜"]
    ).merge_food_memory({
        "events": [{
            "kind": "searched",
            "query": "辣 下饭菜 减肥",
            "original_question": "我想吃点比较辣的下饭菜",
            "recipes": [{"name": "酸辣大白菜"}],
        }],
    })
    assert request.memory_terms == ["辣"]
    assert request.memory_note["query"] == "辣 下饭菜 减肥"
    assert request.retrieval_query(lang="zh") == "大蒜"


def test_specific_soup_or_cuisine_does_not_inherit_old_beef_search():
    memory = {
        "events": [{
            "kind": "searched",
            "query": "牛肉",
            "original_question": "我想吃牛肉",
            "recipes": [{"name": "炒牛肉", "ingredients": ["牛肉"]}],
        }],
    }
    soup = SearchRequest(
        original_text="想喝点儿汤", query="汤", meals=["汤"],
    ).merge_food_memory(memory)
    hangzhou = SearchRequest(
        original_text="我想吃杭州菜", query="杭州菜", cuisines=["杭帮菜"],
    ).merge_food_memory(memory)
    assert soup.retrieval_query(lang="zh") == "汤"
    assert hangzhou.retrieval_query(lang="zh") == "杭州菜 杭帮菜"
    assert soup.memory_note == {} and hangzhou.memory_note == {}


def test_recent_topic_beats_older_food_history_for_elliptical_search():
    turns = [
        {"role": "user", "content": "我想吃鱼"},
        {"role": "assistant", "content": "给你推荐几道鱼。"},
        {"role": "user", "content": "地衣"},
        {"role": "assistant", "content": "地衣是真菌和藻类的共生体。"},
        {"role": "user", "content": "可以做什么菜？"},
    ]
    fish_history = {
        "events": [{
            "kind": "searched",
            "query": "鱼",
            "original_question": "我想吃鱼",
            "recipes": [{"name": "清蒸鱼"}],
        }],
    }

    request = SearchRequest(
        original_text="可以做什么菜？",
        query="菜",
    ).merge_recent_context(turns).merge_food_memory(fish_history)

    assert request.retrieval_query(lang="zh") == "地衣"
    assert request.context_note == {
        "kind": "recent_topic",
        "term": "地衣",
        "original_query": "菜",
    }
    assert request.memory_note == {}


def test_generic_search_without_recent_topic_does_not_treat_one_search_as_preference():
    request = SearchRequest(
        original_text="推荐几道菜",
        query="推荐",
    ).merge_recent_context([
        {"role": "user", "content": "设备在线吗"},
        {"role": "user", "content": "推荐几道菜"},
    ]).merge_food_memory({
        "events": [{
            "kind": "searched",
            "query": "鱼",
            "original_question": "我想吃鱼",
            "recipes": [{"name": "清蒸鱼"}],
        }],
    })

    assert request.context_note == {}
    assert request.retrieval_query(lang="zh") == "推荐"


def test_explicit_current_target_beats_reference_word_but_pronoun_uses_recent_topic():
    turns = [
        {"role": "user", "content": "地衣"},
        {"role": "assistant", "content": "地衣是一种共生体。"},
    ]

    explicit = SearchRequest(
        original_text="这个白菜可以做什么菜？",
        query="白菜",
    ).merge_recent_context(turns)
    pronoun = SearchRequest(
        original_text="它能做什么菜？",
        query="它 菜",
    ).merge_recent_context(turns)

    assert explicit.query == "白菜"
    assert explicit.context_note == {}
    assert pronoun.query == "地衣"
    assert pronoun.context_note["term"] == "地衣"


def test_english_whatsapp_request_keeps_diet_and_exclusions():
    request = SearchRequest.from_intent_raw(
        {"r": True, "c": "recipe_search", "k": ["dinner"]},
        original_text="Recommend dinner, I am vegetarian and I don't eat peanuts",
        keywords=["dinner"],
    )
    assert request.dietary_constraints == ["vegetarian"]
    assert request.exclude == ["peanuts"]

    remembered = request.merge_preferences({"likes": ["spicy"], "dietary_constraints": ["vegetarian"]})
    assert remembered.soft_preferences == ["spicy"]
    assert remembered.dietary_constraints == ["vegetarian"]

    refined = remembered.refine("Show me more, but make it lighter")
    assert refined.flavors == ["light"]
    assert refined.avoid == ["too oily"]


def test_hard_exclusion_filters_recipe_again_after_retrieval():
    request = SearchRequest(original_text="不吃花生", query="家常菜", exclude=["花生"])
    result = {
        "success": True,
        "results": [
            {"id": "1", "metadata": {"recipe_id": "1", "name": "花生拌菠菜", "ingredients": ["菠菜", "花生"]}},
            {"id": "2", "metadata": {"recipe_id": "2", "name": "蒜蓉菠菜", "ingredients": ["菠菜", "蒜"]}},
        ],
    }
    filtered = filter_hard_constraint_violations(result, request)
    assert [item["id"] for item in filtered["results"]] == ["2"]
    assert filtered["_hard_filtered"][0]["matched"] == ["花生"]


def test_explicit_ingredient_filters_unrelated_results_for_any_scene():
    request = SearchRequest(
        original_text="朋友聚餐，手里有两条鱼，怎么做",
        query="鱼 朋友聚餐",
        ingredients=["鱼"],
        scenes=["朋友聚餐"],
    )
    result = {
        "success": True,
        "results": [
            {
                "id": "unrelated",
                "metadata": {
                    "recipe_id": "unrelated",
                    "name": "肉末炒双笋",
                    "ingredients": ["猪肉", "竹笋"],
                },
            },
            {
                "id": "fish",
                "metadata": {
                    "recipe_id": "fish",
                    "name": "清蒸鲈鱼",
                    "ingredients": ["鲈鱼", "姜"],
                },
            },
        ],
    }

    filtered = filter_hard_constraint_violations(result, request)
    assert [item["id"] for item in filtered["results"]] == ["fish"]
    assert filtered["_positive_filtered"] == ["unrelated"]


def test_specific_ingredient_is_not_relaxed_to_a_broader_candidate():
    request = SearchRequest(
        original_text="家里有草鱼，推荐一个做法",
        query="草鱼",
        ingredients=["草鱼"],
    )
    result = {
        "success": True,
        "results": [
            {
                "id": "generic",
                "metadata": {"name": "家常鱼块", "ingredients": ["鱼"]},
            },
            {
                "id": "grounded",
                "metadata": {"name": "红烧草鱼", "ingredients": ["草鱼"]},
            },
        ],
    }

    filtered = filter_hard_constraint_violations(result, request)
    assert [item["id"] for item in filtered["results"]] == ["grounded"]


def test_diversity_selector_keeps_top_one_but_avoids_three_near_duplicates():
    results = [
        {"id": "1", "metadata": {"name": "剁椒鱼头", "tags": ["湘菜", "辣", "蒸"]}},
        {"id": "2", "metadata": {"name": "剁椒鱼块", "tags": ["湘菜", "辣", "蒸"]}},
        {"id": "3", "metadata": {"name": "辣椒炒菌菇", "tags": ["湘菜", "辣", "炒", "蔬菜"]}},
        {"id": "4", "metadata": {"name": "剁椒蒸茄子", "tags": ["湘菜", "辣", "蒸"]}},
    ]
    selected = select_diverse_results(results, limit=3)
    assert selected[0]["id"] == "1"
    assert selected[1]["id"] == "3"


def test_multi_ingredient_coverage_stays_ahead_of_partial_matches_after_diversity():
    request = SearchRequest(
        original_text="番茄和鸡蛋还可以做哪些菜",
        query="番茄 鸡蛋",
        ingredients=["番茄", "鸡蛋"],
    )
    results = [
        {
            "id": "tomato",
            "metadata": {
                "recipe_id": "tomato",
                "name": "番茄炖豆腐",
                "ingredients": ["番茄", "豆腐"],
                "tags": ["炖"],
            },
        },
        {
            "id": "both-1",
            "metadata": {
                "recipe_id": "both-1",
                "name": "番茄蛋花汤",
                "ingredients": ["番茄", "鸡蛋"],
                "tags": ["汤"],
            },
        },
        {
            "id": "egg",
            "metadata": {
                "recipe_id": "egg",
                "name": "青椒炒蛋",
                "ingredients": ["青椒", "鸡蛋"],
                "tags": ["炒"],
            },
        },
        {
            "id": "both-2",
            "metadata": {
                "recipe_id": "both-2",
                "name": "番茄鸡蛋饼",
                "ingredients": ["番茄", "鸡蛋"],
                "tags": ["煎"],
            },
        },
    ]
    prioritized = prioritize_ingredient_coverage(results, request)
    selected = select_diverse_results(prioritized, limit=3)
    assert [item["id"] for item in selected[:2]] == ["both-1", "both-2"]
    assert all(
        item["_ingredient_match"]["match_type"] == "all"
        for item in selected[:2]
    )
    assert selected[2]["_ingredient_match"]["match_type"] == "partial"


def test_menu_plan_mildly_demotes_processing_food_for_a_normal_party_table():
    request = SearchRequest(
        original_text="四个人聚餐，一菜一汤，家常简单一点，没有忌口",
        query="家常 快手",
        task_operation="fuzzy_recommend",
        scenes=["快手"],
        party_size=4,
        menu_dish_count=1,
        menu_soup_count=1,
        result_limit=2,
        constraints_confirmed=True,
    )

    async def search(query, _top_k, _lang):
        if query.startswith("汤 "):
            return {
                "success": True,
                "results": [{
                    "id": "soup",
                    "metadata": {
                        "recipe_id": "soup",
                        "name": "番茄蛋花汤",
                        "ingredients": ["番茄", "鸡蛋"],
                        "facets": {"meal": ["soup"]},
                    },
                }],
            }
        return {
            "success": True,
            "results": [
                {
                    "id": "puree",
                    "metadata": {
                        "recipe_id": "puree",
                        "name": "牛肉泥",
                        "ingredients": ["牛肉"],
                        "facets": {
                            "meal": ["main_course"],
                            "method": ["blended"],
                            "scene": ["child_friendly"],
                        },
                    },
                },
                {
                    "id": "chicken",
                    "metadata": {
                        "recipe_id": "chicken",
                        "name": "家常炒鸡",
                        "ingredients": ["鸡肉"],
                        "difficulty": "简单",
                        "facets": {
                            "meal": ["main_course"],
                            "main_ingredient": ["chicken"],
                            "method": ["stir_fried"],
                            "scene": ["quick", "rice_companion"],
                        },
                    },
                },
            ],
        }

    result = asyncio.run(build_menu_plan(request, search=search))
    assert result["_menu_plan"]["complete"] is True
    assert result["results"][0]["id"] == "chicken"


def test_structured_dish_is_prioritized_without_runtime_alias_rewrite():
    request = SearchRequest.from_intent_raw(
        {
            "r": True,
            "c": "recipe_search",
            "k": ["番茄炒蛋"],
            "s": {"q": "番茄炒蛋", "dish": ["番茄炒蛋"]},
        },
        original_text="我想吃番茄炒蛋",
        keywords=["番茄炒蛋"],
    )
    assert request.dishes == ["番茄炒蛋"]
    assert request.retrieval_query(lang="zh").startswith("番茄炒蛋")
    ranked = [
        {"id": "1", "metadata": {"name": "番茄虾仁鸡蛋汤"}},
        {"id": "2", "metadata": {"name": "番茄炒蛋"}},
    ]
    prioritized = prioritize_exact_matches(ranked, request.dishes)
    assert prioritized[0]["id"] == "2"


def test_router_uses_structured_request_and_filters_excluded_recipe(monkeypatch):
    captured = {}

    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            "k": ["湖南", "减脂"],
            "s": {"q": "湖南 减脂", "cuisine": ["湘菜"], "scene": ["减脂"], "exclude": ["花生"]},
        }, original_text=question)

    async def search(query, top_k=3, lang=None):
        captured.update(query=query, top_k=top_k, lang=lang)
        return {
            "success": True,
            "results": [
                    {"id": "1", "score": 0.9, "metadata": {"recipe_id": "1", "name": "花生辣鸡", "ingredients": ["鸡肉", "花生"]}},
                    {"id": "2", "score": 0.9, "metadata": {"recipe_id": "2", "name": "剁椒蒸鸡", "ingredients": ["鸡肉", "剁椒"], "tags": ["湘菜"]}},
            ],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)
    outcome = asyncio.run(router.route_fast_path("想吃湖南减脂菜，不吃花生"))
    assert captured["top_k"] == 10
    assert "花生" not in captured["query"]
    assert [item["id"] for item in outcome.search_result["results"]] == ["2"]
    assert outcome.search_result["_search_request"]["exclude"] == ["花生"]


def test_router_returns_dynamic_number_for_family_scene(monkeypatch):
    captured = {}

    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            "k": ["湘菜"],
            "s": {"q": "湘菜", "cuisine": ["湘菜"]},
        }, original_text=question)

    async def search(query, top_k=3, lang=None):
        captured["top_k"] = top_k
        return {
            "success": True,
            "results": [
                    {"id": str(i), "score": 0.9, "metadata": {"recipe_id": str(i), "name": f"湘菜{i}", "tags": [f"做法{i}"]}}
                for i in range(1, 10)
            ],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)
    outcome = asyncio.run(router.route_fast_path("我们一家四口，都是湖南人，推荐些湘菜"))
    assert captured["top_k"] == 10
    # 模糊搜索默认返回最多 5 条；只有推荐/菜单意图才按人数决定菜数。
    assert len(outcome.search_result["results"]) == 5
    assert outcome.search_result["_display_limit"] == 5


def test_router_resolves_elliptical_search_from_same_thread_before_food_memory(monkeypatch):
    captured = {}

    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            "k": ["菜"],
            "s": {"q": "菜"},
        }, original_text=question)

    async def search(query, top_k=3, lang=None):
        captured.update(query=query, top_k=top_k, lang=lang)
        return {
            "success": True,
            "results": [{
                "id": "1",
                "metadata": {
                    "recipe_id": "1",
                    "name": "地衣炒鸡蛋",
                    "ingredients": ["地衣", "鸡蛋"],
                },
            }],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)
    outcome = asyncio.run(router.route_fast_path(
        "可以做什么菜？",
        recent_turns=[
            {"role": "user", "content": "我想吃鱼"},
            {"role": "assistant", "content": "给你推荐几道鱼。"},
            {"role": "user", "content": "地衣"},
            {"role": "assistant", "content": "地衣是真菌和藻类的共生体。"},
            {"role": "user", "content": "可以做什么菜？"},
        ],
        food_memory={
            "events": [{
                "kind": "searched",
                "query": "鱼",
                "original_question": "我想吃鱼",
                "recipes": [{"name": "清蒸鱼"}],
            }],
        },
    ))

    assert outcome.kind == "search"
    assert captured["query"] == "地衣"
    assert outcome.search_request.context_note["term"] == "地衣"
    assert outcome.search_request.memory_note == {}


def test_router_clarifies_before_search_then_merges_pending_reply(monkeypatch):
    searches = []
    progress_events = []

    async def classify(question):
        if question == "晚饭吃什么":
            return IntentResult.from_raw({
                "r": True,
                "c": "recipe_search",
                "k": ["晚饭"],
                "s": {"q": "晚饭", "meal": ["晚饭"]},
            }, original_text=question)
        # 短补充即使被轻量模型分成 cooking_qa，也应在 pending 请求存在时
        # 合回上一轮，而不是当作一条孤立问答或再次追问。
        return IntentResult.from_raw({
            "r": True,
            "c": "cooking_qa",
            "k": ["鸡肉", "清淡"],
            "s": {
                "q": "鸡肉 清淡",
                "ingredient": ["鸡肉"],
                "flavor": ["清淡"],
            },
        }, original_text=question)

    async def clarify(*_args, **_kwargs):
        return "今晚先别费脑子。家里现在有什么主料？"

    async def search(query, top_k=3, lang=None):
        searches.append((query, top_k, lang))
        return {
            "success": True,
            "results": [{
                "id": "chicken-1",
                "metadata": {
                    "recipe_id": "chicken-1",
                    "name": "清蒸鸡肉",
                    "ingredients": ["鸡肉"],
                    "tags": ["清淡"],
                },
            }],
        }

    async def on_search_start(query, lang):
        progress_events.append((query, lang))

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)

    first = asyncio.run(router.route_fast_path(
        "晚饭吃什么",
        on_search_start=on_search_start,
    ))
    assert first.kind == "clarify"
    assert first.clarification_dimension == "available_ingredients"
    assert first.clarification_message == "今晚先别费脑子。家里现在有什么主料？"
    assert searches == []
    assert progress_events == []

    second = asyncio.run(router.route_fast_path(
        "家里有鸡肉，想吃清淡点",
        on_search_start=on_search_start,
        pending_search_request=first.search_request.model_dump(),
    ))
    assert second.kind == "search"
    assert len(searches) == 1
    assert len(progress_events) == 1
    assert second.search_request.meals == ["晚饭"]
    assert second.search_request.ingredients == ["鸡肉"]
    assert second.search_request.flavors == ["清淡"]
    assert all(term in searches[0][0] for term in ("晚饭", "鸡肉", "清淡"))


def test_lunch_clarification_searches_once_savory_party_facts_are_complete(monkeypatch):
    menu_requests = []

    async def classify(question):
        if question == "午餐吃什么呢":
            raw = {
                "r": True,
                "c": "recipe_recommend",
                "k": ["午餐"],
                "s": {"q": "午餐", "meal": ["午餐"]},
            }
        else:
            # 复现线上轻量模型输出：“下饭菜”只落在 q，而没有 flavor 槽位。
            raw = {
                "r": True,
                "c": "recipe_search",
                "k": ["下饭菜"],
                "s": {
                    "q": "下饭菜",
                    "scene": ["两人餐"],
                    "exclude": ["香菜"],
                },
            }
        return IntentResult.from_raw(raw, original_text=question)

    async def clarify(*_args, dimension=None, **_kwargs):
        return f"clarify:{dimension}"

    async def menu_plan(request, **_kwargs):
        menu_requests.append(request)
        return {
            "success": True,
            "results": [{
                "id": "savory-1",
                "metadata": {
                    "recipe_id": "savory-1",
                    "name": "杏鲍菇卤肉饭",
                    "ingredients": ["杏鲍菇", "五花肉"],
                    "tags": ["下饭"],
                },
            }],
            "_menu_plan": {"queries": [{"query": request.retrieval_query()}]},
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarify)
    monkeypatch.setattr(router, "build_menu_plan", menu_plan)

    first = asyncio.run(router.route_fast_path("午餐吃什么呢"))
    assert first.kind == "clarify"
    assert first.clarification_dimension == "recommendation_basics"

    second = asyncio.run(router.route_fast_path(
        "两个人吃，下饭菜，不吃香菜",
        pending_search_request=first.search_request.model_dump(),
        pending_clarification_dimension=first.clarification_dimension,
        pending_asked_dimensions=[first.clarification_dimension],
    ))

    assert second.kind == "menu_plan"
    assert len(menu_requests) == 1
    request = second.search_request
    assert request.party_size == 2
    assert request.meals == ["午餐"]
    assert request.flavors == ["下饭"]
    assert request.exclude == ["香菜"]
    assert request.constraints_confirmed is True
    assert all(term in request.retrieval_query() for term in ("午餐", "下饭"))


def test_repeated_non_safety_clarification_has_bounded_exit(monkeypatch):
    searches = []

    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            "k": ["午餐"],
            "s": {"q": "午餐", "meal": ["午餐"]},
        }, original_text=question)

    async def search(query, **_kwargs):
        searches.append(query)
        return {
            "success": True,
            "results": [{
                "id": "lunch-1",
                "metadata": {
                    "recipe_id": "lunch-1",
                    "name": "番茄炒蛋",
                    "ingredients": ["番茄", "鸡蛋"],
                },
            }],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)

    pending = SearchRequest(
        original_text="午餐吃什么",
        query="午餐",
        meals=["午餐"],
        task_operation="fuzzy_search",
    )
    outcome = asyncio.run(router.route_fast_path(
        "午餐",
        pending_search_request=pending.model_dump(),
        pending_clarification_dimension="available_ingredients",
        pending_asked_dimensions=[
            "available_ingredients",
            "available_ingredients",
        ],
    ))

    assert outcome.kind == "search"
    assert searches == ["午餐"]


def test_router_party_recommendation_collects_safety_then_uses_defaults(monkeypatch):
    searches = []

    async def classify(question):
        if "招待" in question:
            raw = {
                "r": True,
                "c": "recipe_recommend",
                "s": {
                    "q": "夏天 上海 朋友聚餐",
                    "scene": ["夏天", "上海", "朋友聚餐"],
                },
            }
        elif "忌口" in question:
            raw = {"r": True, "c": "cooking_qa", "s": {"q": ""}}
        else:
            raw = {
                "r": True,
                "c": "cooking_qa",
                "s": {"q": "清淡", "flavor": ["清淡"]},
            }
        return IntentResult.from_raw(raw, original_text=question)

    async def clarify(*_args, dimension=None, **_kwargs):
        return f"clarify:{dimension}"

    async def menu_plan(request, **_kwargs):
        searches.append(request.retrieval_query())
        return {
            "success": True,
            "results": [{
                "id": "light-1",
                "metadata": {
                    "recipe_id": "light-1",
                    "name": "清蒸鸡肉",
                    "ingredients": ["鸡肉"],
                    "tags": ["清淡"],
                },
            }],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarify)
    monkeypatch.setattr(router, "build_menu_plan", menu_plan)

    first = asyncio.run(router.route_fast_path(
        "夏天在上海招待三个朋友，推荐吃什么",
    ))
    assert first.kind == "clarify"
    assert first.clarification_dimension == "party_constraints"
    assert searches == []

    second = asyncio.run(router.route_fast_path(
        "没有忌口，也没有过敏",
        pending_search_request=first.search_request.model_dump(),
        pending_clarification_dimension=first.clarification_dimension,
    ))
    assert second.kind == "menu_plan"
    assert second.search_request.constraints_confirmed is True
    assert searches


def test_quantity_complaint_continues_party_pending_and_uses_ten_person_defaults(
    monkeypatch,
):
    menu_requests = []
    clarifications = []
    original_text = (
        "我在杭州，今天晚上有10个朋友来我家吃饭，都是江西人，"
        "要喝点白酒，所有人都没什么忌口，请推荐一个食谱清单给我"
    )
    pending = SearchRequest.from_intent_raw(
        {
            "c": "recipe_recommend",
            "s": {
                "q": "杭州 江西口味 白酒配餐 10人 晚餐",
                "cuisine": ["赣菜"],
                "flavor": ["辣"],
                "scene": ["杭州", "江西人", "白酒"],
                "meal": ["晚餐"],
            },
        },
        original_text=original_text,
    )

    async def classify(question):
        # 复现线上误判：数量抱怨被当成新搜索/话题切换。
        return IntentResult.from_raw(
            {
                "r": True,
                "c": "recipe_recommend",
                "a": "new_search",
                "k": ["10人", "五个菜"],
                "s": {"q": "10人餐 五个菜"},
                "signals": {"topic_change": True},
            },
            original_text=question,
        )

    async def clarify(*_args, dimension=None, **_kwargs):
        clarifications.append(dimension)
        return f"clarify:{dimension}"

    async def menu_plan(request, **_kwargs):
        menu_requests.append(request)
        return {
            "success": True,
            "results": [{
                "id": "dish-1",
                "metadata": {
                    "recipe_id": "dish-1",
                    "name": "真实菜谱一",
                    "ingredients": ["豆腐"],
                },
            }],
            "_menu_plan": {"queries": [{"query": request.retrieval_query()}]},
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarify)
    monkeypatch.setattr(router, "build_menu_plan", menu_plan)

    outcome = asyncio.run(router.route_fast_path(
        "10个人推荐五个菜 。。。太少了",
        pending_search_request=pending.model_dump(),
        pending_clarification_dimension="party_preferences",
    ))

    assert outcome.kind == "menu_plan"
    assert clarifications == []
    assert len(menu_requests) == 1
    request = outcome.search_request
    assert request.party_size == 10
    assert request.constraints_confirmed is True
    assert request.explicit_result_limit is False
    assert request.menu_dish_count == 6
    assert request.menu_soup_count == 2
    assert request.result_limit == 8
    assert request.cuisines == []
    assert request.flavors == []
    assert request.scenes == ["下酒"]
    assert "五个菜" not in request.retrieval_query()


@pytest.mark.parametrize(
    "constraint_reply",
    [
        "没有",
        "没有什么忌口，请推荐",
        "没啥忌口，直接推荐吧",
        "没有任何过敏和忌口",
    ],
)
def test_party_constraints_bare_no_uses_pending_dimension_and_enters_menu(
    monkeypatch,
    constraint_reply,
):
    menu_requests = []

    async def classify(question):
        if "两个人" in question:
            raw = {
                "r": True,
                "c": "recipe_recommend",
                "s": {
                    "q": "晚饭 辣",
                    "meal": ["晚饭"],
                    "flavor": ["辣"],
                },
            }
        else:
            # 裸“没有”本身不携带饮食限制宾语；必须由 pending dimension 补齐。
            raw = {"r": False, "c": "off_topic", "s": {}}
        return IntentResult.from_raw(raw, original_text=question)

    async def clarify(*_args, dimension=None, **_kwargs):
        return f"clarify:{dimension}"

    async def menu_plan(request, **_kwargs):
        menu_requests.append(request)
        return {
            "success": True,
            "results": [{
                "id": "spicy-1",
                "metadata": {
                    "recipe_id": "spicy-1",
                    "name": "辣椒炒肉",
                    "ingredients": ["辣椒", "猪肉"],
                    "tags": ["香辣", "下饭"],
                },
            }],
            "_menu_plan": {"queries": [{"query": request.retrieval_query()}]},
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarify)
    monkeypatch.setattr(router, "build_menu_plan", menu_plan)

    first = asyncio.run(router.route_fast_path("两个人吃，都喜欢吃辣的，准备晚饭"))

    assert first.kind == "clarify"
    assert first.clarification_dimension == "party_constraints"
    assert first.search_request.party_size == 2
    assert first.search_request.flavors == ["辣"]

    # 没有待答上下文时，裸“没有”仍不能被全局解释成无饮食限制。
    standalone = SearchRequest.from_intent_raw(
        {"r": False, "c": "off_topic", "s": {}},
        original_text="没有",
    )
    assert standalone.constraints_confirmed is False
    assert router._sanitize_no_constraints_followup(
        standalone,
        "没有",
        None,
    ).constraints_confirmed is False

    second = asyncio.run(router.route_fast_path(
        constraint_reply,
        pending_search_request=first.search_request.model_dump(),
        pending_clarification_dimension=first.clarification_dimension,
        pending_asked_dimensions=[first.clarification_dimension],
    ))

    assert second.kind == "menu_plan"
    assert second.search_request.constraints_confirmed is True
    assert second.search_request.party_size == 2
    assert second.search_request.flavors == ["辣"]
    assert len(menu_requests) == 1


@pytest.mark.parametrize("reply", ["没有", "没", "无", "none"])
def test_bare_no_constraint_reply_requires_party_constraints_context(reply):
    standalone = SearchRequest.from_intent_raw(
        {"r": False, "c": "off_topic", "s": {}},
        original_text=reply,
    )

    assert standalone.constraints_confirmed is False
    assert router._sanitize_no_constraints_followup(
        standalone,
        reply,
        None,
    ).constraints_confirmed is False
    assert router._sanitize_no_constraints_followup(
        standalone,
        reply,
        "party_constraints",
    ).constraints_confirmed is True


def test_party_constraints_bare_yes_asks_for_details_before_menu(monkeypatch):
    menu_requests = []

    async def classify(question):
        if "两个人" in question:
            raw = {
                "r": True,
                "c": "recipe_recommend",
                "s": {"q": "晚饭 辣", "meal": ["晚饭"], "flavor": ["辣"]},
            }
        elif question == "有":
            # 即使模型擅自补出具体限制，状态规则也只能记录“存在限制”。
            raw = {
                "r": True,
                "c": "recipe_recommend",
                "s": {
                    "q": "无饮食限制 宗教饮食要求",
                    "diet": ["无饮食限制"],
                    "exclude": ["宗教饮食要求"],
                },
            }
        else:
            raw = {"r": False, "c": "off_topic", "s": {}}
        return IntentResult.from_raw(raw, original_text=question)

    async def clarify(*_args, dimension=None, **_kwargs):
        return f"clarify:{dimension}"

    async def menu_plan(request, **_kwargs):
        menu_requests.append(request)
        return {
            "success": True,
            "results": [{
                "id": "safe-spicy-1",
                "metadata": {
                    "recipe_id": "safe-spicy-1",
                    "name": "香辣鸡丁",
                    "ingredients": ["鸡肉", "辣椒"],
                    "tags": ["香辣"],
                },
            }],
            "_menu_plan": {"queries": [{"query": request.retrieval_query()}]},
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarify)
    monkeypatch.setattr(router, "build_menu_plan", menu_plan)

    first = asyncio.run(router.route_fast_path("两个人吃，都喜欢吃辣的，准备晚饭"))
    second = asyncio.run(router.route_fast_path(
        "有",
        pending_search_request=first.search_request.model_dump(),
        pending_clarification_dimension=first.clarification_dimension,
        pending_asked_dimensions=[first.clarification_dimension],
    ))

    assert second.kind == "clarify"
    assert second.clarification_dimension == "party_constraints_detail"
    assert second.clarification_message == "clarify:party_constraints_detail"
    assert second.search_request.constraints_confirmed is False
    assert second.search_request.dietary_constraints == []
    assert second.search_request.exclude == []
    assert menu_requests == []

    third = asyncio.run(router.route_fast_path(
        "我对花生过敏",
        pending_search_request=second.search_request.model_dump(),
        pending_clarification_dimension=second.clarification_dimension,
        pending_asked_dimensions=[
            first.clarification_dimension,
            second.clarification_dimension,
        ],
    ))

    assert third.kind == "menu_plan"
    assert third.search_request.constraints_confirmed is True
    assert "花生" in third.search_request.exclude
    assert len(menu_requests) == 1


@pytest.mark.parametrize(
    ("text", "dimension", "expected"),
    [
        ("家里有鸡蛋和番茄", "available_ingredients", True),
        ("鸡蛋和番茄", "available_ingredients", True),
        ("都行", "party_preferences", True),
        ("没有忌口", "party_preferences", True),
        ("没有忌口，也没有过敏", "party_constraints", True),
        ("没有", "party_constraints", True),
        ("有", "party_constraints", True),
        ("我对花生过敏", "party_constraints_detail", True),
        ("清淡一点", "party_constraints", False),
        ("你定吧", "available_ingredients", True),
        ("人生好难", "party_preferences", False),
        ("I have a headache", "available_ingredients", False),
        ("Why is my phone hot?", "available_ingredients", False),
    ],
)
def test_clarification_reply_detection_is_dimension_safe(text, dimension, expected):
    assert router._looks_like_clarification_reply(text, dimension) is expected


def test_final_arrangement_uses_pending_facts_and_returns_requested_six(monkeypatch):
    searches = []

    async def classify(question):
        return IntentResult.from_raw({
            "r": False,
            "c": "off_topic",
            "k": ["最终答案"],
            "s": {"q": "最终答案"},
        }, original_text=question)

    async def search(query, top_k=3, lang=None):
        searches.append((query, top_k))
        return {
            "success": True,
            "results": [
                {
                    "id": f"party-{index}",
                    "score": 0.9 - index * 0.01,
                    "metadata": {
                        "recipe_id": f"party-{index}",
                        "name": f"聚餐菜{index}",
                        "ingredients": [f"食材{index}"],
                        "tags": [f"风格{index}"],
                    },
                }
                for index in range(1, 9)
            ],
        }

    pending = SearchRequest(
        original_text="六个人聚餐，推荐六道菜",
        query="朋友聚餐",
        scenes=["朋友聚餐"],
        party_size=6,
        result_limit=6,
        explicit_result_limit=True,
        constraints_confirmed=True,
        task_operation="fuzzy_recommend",
    )
    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)

    outcome = asyncio.run(router.route_fast_path(
        "具体你来安排，给我一个最终答案",
        pending_search_request=pending.model_dump(),
        pending_clarification_dimension="flavor_preferences",
        pending_asked_dimensions=["party_constraints", "flavor_preferences"],
    ))

    assert outcome.kind == "search"
    assert outcome.intent.category.value == "recipe_recommend"
    assert len(outcome.search_result["results"]) == 6
    assert outcome.search_request.result_limit == 6
    assert outcome.search_request.explicit_result_limit is True
    assert "最终答案" not in outcome.search_query
    assert searches and "朋友聚餐" in searches[0][0]


def test_final_arrangement_does_not_bypass_missing_safety_facts():
    assert router._looks_like_clarification_reply(
        "你来安排，直接给最终答案",
        "party_constraints",
    ) is False
    assert router._looks_like_clarification_reply(
        "你来安排，直接给最终答案",
        "flavor_preferences",
    ) is True


def test_party_and_dish_counts_are_not_selection_preferences():
    request = SearchRequest(
        original_text="六个人聚餐，推荐六道菜",
        query="六个人 聚餐 六道菜",
        scenes=["聚餐"],
        party_size=6,
        result_limit=6,
        explicit_result_limit=True,
        constraints_confirmed=True,
        task_operation="fuzzy_recommend",
    )
    assert request.has_selection_direction() is False
    assert request.recommendation_clarification_dimension() is None


def test_no_restriction_reply_does_not_turn_safety_words_into_search_direction():
    followup = SearchRequest.from_intent_raw(
        {
            "c": "cooking_qa",
            "k": ["过敏", "忌口", "宗教饮食"],
        },
        original_text="没有过敏、忌口或宗教饮食要求",
        keywords=["过敏", "忌口", "宗教饮食"],
    )
    merged = SearchRequest(
        original_text="六个人聚餐，推荐六道菜",
        query="六道菜",
        party_size=6,
        result_limit=6,
        explicit_result_limit=True,
        task_operation="fuzzy_recommend",
    ).merge_clarification(followup)

    assert followup.constraints_confirmed is True
    assert followup.query == ""
    assert followup.exclude == []
    assert merged.has_selection_direction() is False
    assert merged.recommendation_clarification_dimension() is None


def test_router_sanitizes_direct_recommend_classification_for_safety_reply(monkeypatch):
    searches = []

    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_recommend",
            "k": ["过敏", "忌口", "宗教饮食"],
            "s": {
                "q": "过敏 忌口 宗教饮食要求",
                "diet": ["无饮食限制"],
                "exclude": ["宗教饮食要求"],
            },
        }, original_text=question)

    async def search(*args, **kwargs):
        searches.append((args, kwargs))
        return {"success": True, "results": []}

    async def clarify(*_args, dimension=None, **_kwargs):
        return f"clarify:{dimension}"

    pending = SearchRequest(
        original_text="六个人聚餐，推荐六道菜",
        query="六道菜",
        party_size=6,
        result_limit=6,
        explicit_result_limit=True,
        task_operation="fuzzy_recommend",
    )
    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarify)

    outcome = asyncio.run(router.route_fast_path(
        "没有过敏、忌口或宗教饮食要求",
        pending_search_request=pending.model_dump(),
        pending_clarification_dimension="party_constraints",
        pending_asked_dimensions=["party_constraints"],
    ))

    assert outcome.kind == "search"
    assert outcome.search_request.constraints_confirmed is True
    assert outcome.search_request.dietary_constraints == []
    assert outcome.search_request.exclude == []
    assert searches


def test_explicit_topic_change_replaces_pending_recommendation(monkeypatch):
    captured = {}

    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            "k": ["红烧肉"],
            "s": {"q": "红烧肉", "dish": ["红烧肉"]},
        }, original_text=question)

    async def search(query, top_k=3, lang=None):
        captured["query"] = query
        return {
            "success": True,
                "results": [{
                    "id": "pork-1",
                    "score": 0.95,
                    "metadata": {"recipe_id": "pork-1", "name": "红烧肉"},
                }],
        }

    pending = SearchRequest(
        original_text="四个人朋友聚餐吃什么",
        query="朋友聚餐",
        scenes=["朋友聚餐"],
        party_size=4,
        result_limit=4,
    )
    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)

    outcome = asyncio.run(router.route_fast_path(
        "算了，我想吃红烧肉",
        pending_search_request=pending.model_dump(),
        pending_clarification_dimension="party_preferences",
    ))

    assert outcome.kind == "detail"
    assert outcome.search_request.party_size is None
    assert outcome.search_request.scenes == []
    assert "朋友聚餐" not in captured["query"]
    assert "红烧肉" in captured["query"]


def test_explicit_how_to_returns_grounded_detail_outcome(monkeypatch):
    memory_scopes = []

    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            "k": ["西葫芦炒鸡蛋"],
            "s": {"q": "西葫芦炒鸡蛋", "dish": ["西葫芦炒鸡蛋"]},
        }, original_text=question)

    async def search(query, top_k=3, lang=None):
        return {
            "success": True,
            "results": [{
                "id": "zucchini-egg",
                "score": 0.9,
                "metadata": {
                    "recipe_id": "zucchini-egg",
                    "name": "西葫芦炒鸡蛋",
                    "ingredients": ["西葫芦", "鸡蛋"],
                },
            }],
        }

    async def load_memory(scope):
        memory_scopes.append(scope)
        return {
            "preferences": {"allergens": []},
            "recent_turns": [],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)
    outcome = asyncio.run(router.route_fast_path(
        "西葫芦炒鸡蛋怎么做",
        memory_context_loader=load_memory,
    ))
    assert outcome.kind == "detail"
    assert outcome.search_result["results"][0]["id"] == "zucchini-egg"
    assert memory_scopes == ["preferences"]


def test_explicit_dish_search_returns_detail_without_extra_how_to_words(monkeypatch):
    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            "k": ["红烧肉"],
            "s": {"q": "红烧肉", "dish": ["红烧肉"]},
        }, original_text=question)

    async def search(query, top_k=3, lang=None):
        return {
            "success": True,
            "results": [{
                "id": "braised-pork",
                "score": 0.9,
                "metadata": {
                    "recipe_id": "braised-pork",
                    "name": "红烧肉",
                    "ingredients": ["五花肉"],
                },
            }],
        }

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)
    outcome = asyncio.run(router.route_fast_path("红烧肉"))
    assert outcome.kind == "detail"
    assert outcome.search_request.task_operation == "exact_search"
    assert outcome.search_request.canonical_question == "红烧肉"
    assert outcome.intent.business_operation.value == "exact_search"
    assert outcome.search_result["_interaction_mode"] == "detail"


def test_missing_explicit_recipe_fails_closed_without_web_reference(monkeypatch):
    bridge_calls = []

    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            # 模拟 Active Planner 的 intent override：保留用户原话，不依赖
            # 分类模型提前填充 dishes 槽位。
            "q": question,
            "k": [question],
            "a": "new_search",
        }, original_text=question)

    async def search(query, top_k=3, lang=None):
        return {"success": True, "results": []}

    async def bridge(*args, **kwargs):
        bridge_calls.append((args, kwargs))
        return "不应调用"

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)
    monkeypatch.setattr(router, "generate_search_bridge", bridge)
    outcome = asyncio.run(router.route_fast_path("烤红薯怎么做"))
    assert outcome.kind == "search"
    assert outcome.search_result["results"] == []
    assert outcome.search_result["_exact_not_found"] is True
    assert outcome.search_result["_grounded_output"] is True
    assert outcome.search_query == "烤红薯"
    assert bridge_calls == []
    assert not hasattr(router, "search_recipe_web")


def test_exact_recipe_retries_deterministic_synonym_before_not_found(monkeypatch):
    searches = []

    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            "k": ["烤红薯"],
            "s": {"q": "烤红薯", "dish": ["烤红薯"]},
        }, original_text=question)

    async def search(query, top_k=3, lang=None):
        searches.append(query)
        if query == "烤地瓜":
            return {
                "success": True,
                "results": [{
                    "id": "sweet-potato-1",
                    "score": 0.95,
                    "metadata": {
                        "recipe_id": "sweet-potato-1",
                        "name": "烤地瓜",
                        "ingredients": ["地瓜"],
                    },
                }],
            }
        return {"success": True, "results": []}

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)

    outcome = asyncio.run(router.route_fast_path("烤红薯怎么做"))
    assert exact_recipe_synonym_queries(outcome.search_request, limit=2) == ["烤地瓜", "烤番薯"]
    assert outcome.kind == "detail"
    assert searches == ["烤红薯", "烤地瓜"]
    assert outcome.search_result["_exact_alias_query"] == "烤地瓜"
    assert outcome.search_result["results"][0]["metadata"]["name"] == "烤地瓜"


def test_router_clarifies_negative_only_search_before_retrieval(monkeypatch):
    searches = []

    async def classify(question):
        return IntentResult.from_raw({
            "r": True,
            "c": "recipe_search",
            "k": [],
            "s": {"q": "", "exclude": ["香菜"]},
        }, original_text=question)

    async def search(*args, **kwargs):
        searches.append((args, kwargs))
        return {"success": True, "results": []}

    async def clarify(*args, **kwargs):
        return "除了不放香菜，你更想用什么主料，或者偏什么口味？"

    monkeypatch.setattr(router, "classify_intent", classify)
    monkeypatch.setattr(router, "_run_search_subprocess", search)
    monkeypatch.setattr(router, "generate_recommendation_clarification", clarify)
    outcome = asyncio.run(router.route_fast_path("不吃香菜，给我找菜谱"))
    assert outcome.kind == "clarify"
    assert outcome.search_request.task_operation == "fuzzy_search"
    assert outcome.search_request.canonical_question == ""
    assert outcome.clarification_dimension == "ingredient_or_flavor"
    assert searches == []
