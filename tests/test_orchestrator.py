"""
阶段1 重构的行为固化测试（纯函数，不依赖网络/Milvus）。

跑法（项目根目录）：
  .venv/bin/python tests/test_orchestrator.py          # 直接当脚本跑
  .venv/bin/python -m pytest tests/test_orchestrator.py # 有 pytest 时

固化两件事：
  1. IntentResult/IntentCategory 解析与旧 _intent_check 的 r/c/k 默认值等价；
  2. route_fast_path 的路由分支（search/greeting/agent）与旧 chat_stream/qqbot_chat 内联逻辑一致。
渲染函数 + detect_lang 也一并 pin（重构未动它们，回归用）。
"""
import asyncio
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.orchestrator.intent import IntentCategory, IntentResult
import app.orchestrator.router as R
import app.orchestrator.cook as cook_module
from app.orchestrator.cook import (
    _device_text,
    _public_device_error,
    _summarize_status,
    evaluate_device_readiness,
)
from app.agent.fast_path import (
    SearchQueryPlan,
    apply_search_plan,
    detect_lang,
    _filter_non_recipe_records,
    _format_search_response,
    format_menu_plan_response,
    format_search_markdown,
    plan_search_query,
)
import app.orchestrator.recommendation_response as recommendation_response
from app.orchestrator.recommendation_response import (
    RecommendationNarrative,
    generate_search_bridge,
    generate_recommendation_narrative,
)
from app.orchestrator.search_request import SearchRequest
from app.core.config import resolve_model_profile


def test_model_profiles_and_role_overrides():
    assert resolve_model_profile("qwen") == {
        "profile": "qwen", "main": "qwen3.7-plus", "qa": "qwen-plus",
    }
    assert resolve_model_profile("deepseek") == {
        "profile": "deepseek", "main": "deepseek-v4-pro", "qa": "deepseek-v4-pro",
    }
    assert resolve_model_profile("zhipu") == {
        "profile": "zhipu", "main": "glm-5.2", "qa": "glm-5.2",
    }
    assert resolve_model_profile("deepseek", {"QA_MODEL": "deepseek-v4-flash"})["qa"] == "deepseek-v4-flash"


def test_unknown_model_profile_fails_fast():
    try:
        resolve_model_profile("not-a-model")
        raise AssertionError("未知模型档位不应静默回退")
    except ValueError as exc:
        assert "qwen, deepseek, zhipu" in str(exc)


def test_recipe_result_policy_filters_device_program_records():
    original = {
        "success": True,
        "count": 3,
        "results": [
            {
                "id": "program-1",
                "metadata": {
                    "name": "研磨",
                    "facets": {"record_type": ["device_program"]},
                },
            },
            {
                "id": "recipe-1",
                "metadata": {
                    "name": "番茄炒蛋",
                    "record_type": "recipe",
                },
            },
            {
                "id": "legacy-recipe",
                "metadata": {"name": "清蒸鲈鱼"},
            },
        ],
    }

    filtered = _filter_non_recipe_records(original)

    assert filtered is not original
    assert filtered["count"] == 2
    assert filtered["_non_recipe_filtered_count"] == 1
    assert [item["id"] for item in filtered["results"]] == [
        "recipe-1",
        "legacy-recipe",
    ]


# ── 意图解析：与旧 _intent_check 默认值等价 ──────────────────────────
def test_intent_basic():
    r = IntentResult.from_raw({"r": True, "c": "recipe_search", "k": ["红烧肉"]})
    assert r.related is True
    assert r.category is IntentCategory.recipe_search
    assert r.keywords == ["红烧肉"]


def test_intent_defaults_match_old():
    # 旧 _intent_check 失败/缺字段默认 {"r":True,"c":"unknown","k":[]}
    r = IntentResult.from_raw({})
    assert (r.related, r.category, r.keywords) == (True, IntentCategory.unknown, [])


def test_intent_null_keywords_to_empty():
    r = IntentResult.from_raw({"r": True, "c": "greeting", "k": None})
    assert r.category is IntentCategory.greeting and r.keywords == []


def test_intent_unknown_category_falls_unknown():
    assert IntentResult.from_raw({"c": "weird_value"}).category is IntentCategory.unknown
    assert IntentCategory.from_str("recipe_execute") is IntentCategory.recipe_execute
    assert IntentCategory.from_str("device_manage") is IntentCategory.device_manage


# ── detect_lang（未改，回归 pin）────────────────────────────────────
def test_detect_lang():
    assert detect_lang("红烧肉怎么做") == "zh"
    assert detect_lang("how to cook chicken") == "en"
    assert detect_lang("") == "zh"


# ── 渲染函数（未改，回归 pin）──────────────────────────────────────
# 注：_format_search_response 现为 async（Phase 1.1 LLM 润色），
# 测试通过 FAST_PATH_LLM_POLISH=0 关闭润色，保持模板行为可预测。
os.environ.setdefault("FAST_PATH_LLM_POLISH", "0")


def test_format_search_response_empty():
    d = json.loads(asyncio.run(_format_search_response("x", {"results": []}, lang="zh")))
    assert d["type"] == "recipe_search" and d["data"]["total"] == 0
    assert "不拿别的菜硬凑" not in d["message"]


def test_empty_search_response_explains_actual_filter_reason():
    exact = json.loads(asyncio.run(_format_search_response("帮我找一道番茄炒蛋", {
        "results": [],
        "_exact_not_found": True,
    }, lang="zh")))
    assert "番茄炒蛋" in exact["message"]
    assert "帮我找一道番茄炒蛋" not in exact["message"]
    assert "不会从网上补一份做法" in exact["message"]

    d = json.loads(asyncio.run(_format_search_response("鸡肉", {
        "results": [],
        "_positive_filtered": ["other-1"],
        "_search_request": {"ingredients": ["鸡肉"]},
    }, lang="zh")))
    assert "鸡肉" in d["message"]
    assert "确认包含" in d["message"]

    broad = json.loads(asyncio.run(_format_search_response("随便来点菜", {
        "results": [],
        "_relevance_filtered_count": 4,
    }, lang="zh")))
    assert "范围比较宽" in broad["message"]
    assert "主料、菜系或做法" in broad["message"]


def test_format_search_response_en_filters_tags():
    d = json.loads(asyncio.run(_format_search_response("sea food", {
        "results": [{
            "id": "1",
            "score": 0.9,
            "metadata": {
                "recipe_id": "r1",
                "name": "Seafood Chowder Soup",
                "image_url": "https://example.com/seafood.png",
                "ingredients": ["shrimp", "clam"],
                "tags": ["海鲜类 / Seafood", "面食类 / Noodle-or-Flour", "咸鲜口味", "Savory"],
            },
        }]
    }, lang="en")))
    tags = d["data"]["recipes"][0]["tags"]
    assert d["lang"] == "en"
    assert tags == ["Seafood", "Noodle-or-Flour", "Savory"]
    assert not any(any("\u4e00" <= c <= "\u9fff" for c in tag) for tag in tags)


def test_english_search_response_never_falls_back_to_chinese_metadata():
    data = json.loads(asyncio.run(_format_search_response("Recommend chicken dishes", {
        "_original_query": "鸡肉 晚餐 简单",
        "_planned_query": "鸡肉 晚餐 简单",
        "results": [
            {
                "id": "good", "score": 0.9,
                "metadata": {
                    "recipe_id": "good", "name": "Simple Chicken",
                    "ingredients": ["chicken", "生姜"],
                    "tags": ["家常菜", "easy"],
                    "description": "这是一道家常菜",
                },
            },
            {
                "id": "dirty", "score": 0.8,
                "metadata": {"recipe_id": "dirty", "name": "香辣鸡肉", "tags": ["家常菜"]},
            },
        ],
    }, lang="en")))
    rendered = json.dumps(data, ensure_ascii=False)
    assert data["data"]["recipes"][0]["ingredients"] == ["chicken"]
    assert data["data"]["recipes"][0]["tags"] == ["easy"]
    assert data["data"]["recipes"][0]["description"] == ""
    assert "香辣鸡肉" not in rendered
    assert not any("\u4e00" <= char <= "\u9fff" for char in rendered)


def test_format_search_markdown_empty():
    assert "No matching" in format_search_markdown("x", {"results": []}, lang="en")


def test_search_plan_preserves_all_structured_constraints_without_inference():
    plan = plan_search_query("我是湖南人，喜欢吃辣，但是最近在减肥", "湖南 辣 减肥", lang="zh")
    assert plan.search_query == "湖南 辣 减肥"
    assert plan.reasoning == ""


def test_search_plan_never_rewrites_structured_query_for_different_scenes():
    cases = (
        ("加班回家，冰箱有鸡胸肉，想简单做点晚饭", "鸡胸肉 简单 晚餐 加班"),
        ("朋友聚餐，手里有两条鱼，怎么做", "鱼 朋友聚餐"),
        ("运动后想吃牛肉", "牛肉 运动后"),
    )
    for question, structured_query in cases:
        plan = plan_search_query(question, structured_query, lang="zh")
        assert plan.search_query == structured_query
        assert plan.reasoning == ""
        assert plan.blocked_terms == ()


def test_search_plan_blocked_terms_fail_closed_when_all_results_match():
    result = apply_search_plan(
        {
            "success": True,
            "results": [
                {
                    "id": "peanut-1",
                    "metadata": {
                        "name": "花生拌菜",
                        "ingredients": ["花生", "蔬菜"],
                    },
                }
            ],
        },
        SearchQueryPlan(
            original_query="凉菜",
            search_query="凉菜",
            blocked_terms=("花生",),
        ),
    )

    assert result["results"] == []


def test_recommendation_narrative_does_not_add_region_weather(monkeypatch):
    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": json.dumps({
                "opening": "忙了一天，晚饭的选择尽量简单一点。",
                "strategy": "先比较真实菜谱里的做法标签和步骤数。",
                "recipe_reasons": {"chicken-1": "【清蒸鸡胸肉】用到鸡胸肉。"},
                "closing": "想先看第一道的详情吗？",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    narrative = asyncio.run(generate_recommendation_narrative(
        "加班回家，冰箱有鸡胸肉，想简单做点晚饭",
        "鸡胸肉 简单 晚餐 加班",
        {
            "results": [{
                "id": "chicken-1",
                "metadata": {
                    "recipe_id": "chicken-1",
                    "name": "清蒸鸡胸肉",
                    "ingredients": ["鸡胸肉"],
                    "tags": ["蒸"],
                },
            }],
        },
        lang="zh",
        search_request=SearchRequest(
            original_text="加班回家，冰箱有鸡胸肉，想简单做点晚饭",
            query="鸡胸肉 简单 晚餐 加班",
            ingredients=["鸡胸肉"],
            scenes=["加班"],
            meals=["晚餐"],
        ),
    ))

    assert "鸡胸肉" in narrative.opening
    assert "杭州" not in narrative.opening
    assert "天气" not in narrative.opening


def test_hot_weather_reason_uses_recipe_fact_and_drops_health_invention(
    monkeypatch,
):
    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": json.dumps({
                "opening": "天热就先从做法上挑，不硬加健康结论。",
                "strategy": "【凉拌黄瓜】有凉拌标签，可以先看这道。",
                "recipe_reasons": {
                    "dish-1": "【凉拌黄瓜】有凉拌标签，天热时可以先看它。",
                    "drink-1": "【黄瓜汁】能补水、当代餐，还很顶饱。",
                },
                "closing": "想先看哪一道的详情？",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    narrative = asyncio.run(generate_recommendation_narrative(
        "今天天气有些热，吃哪些菜好呢",
        "菜 天热",
        {
            "results": [
                {
                    "id": "dish-1",
                    "metadata": {
                        "recipe_id": "dish-1",
                        "name": "凉拌黄瓜",
                        "tags": ["凉拌"],
                        "ingredients": ["黄瓜"],
                    },
                },
                {
                    "id": "drink-1",
                    "metadata": {
                        "recipe_id": "drink-1",
                        "name": "黄瓜汁",
                        "tags": ["饮品"],
                        "ingredients": ["黄瓜"],
                    },
                },
            ],
        },
        lang="zh",
        search_request=SearchRequest(
            original_text="今天天气有些热，吃哪些菜好呢",
            query="菜",
            scenes=["天热"],
        ),
    ))

    assert narrative.recipe_reasons["dish-1"] == (
        "【凉拌黄瓜】有凉拌标签，天热时可以先看它。"
    )
    assert "drink-1" not in narrative.recipe_reasons


def test_recommendation_narrative_is_dynamic_and_uses_only_real_facts(monkeypatch):
    result = {
        "results": [{
            "id": "1",
            "metadata": {
                "recipe_id": "r1",
                "name": "番茄鸡蛋汤",
                "tags": ["快手", "汤羹"],
                "ingredients": ["番茄", "鸡蛋"],
            },
        }],
    }
    captured = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["system"] = messages[0].content
            captured["input"] = messages[1].content
            return type("Response", (), {"content": json.dumps({
                "opening": "今天别和厨房较劲。",
                "strategy": "【番茄鸡蛋汤】更贴合你想简单吃点的状态。",
                "recipe_reasons": {"r1": "有快手标签，适合今晚少折腾。"},
                "closing": "你要是赶时间，我就接着帮你看它的真实步骤。",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    narrative = asyncio.run(generate_recommendation_narrative(
        "最近有点累，晚上想吃点简单的", "简单 晚餐", result, lang="zh",
    ))
    assert narrative.opening == "今天别和厨房较劲。"
    assert narrative.recipe_reasons["r1"] == "有快手标签，适合今晚少折腾。"
    assert "高情商" in captured["system"]
    assert "番茄鸡蛋汤" in captured["input"]


def test_party_menu_narrative_receives_recent_context_roles_and_completion(monkeypatch):
    captured = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["payload"] = json.loads(messages[1].content)
            return type("Response", (), {"content": json.dumps({
                "opening": "六个人这桌按四菜一汤来配。",
                "strategy": "先看【葱油鳜鱼】这个必备主料菜。",
                "recipe_reasons": {
                    "fish": "【葱油鳜鱼】是这桌的必备主料菜，主料有鳜鱼。",
                },
                "closing": "想先看哪道的详情？",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    request = SearchRequest(
        original_text="六个人，四菜一汤，不吃辣，对花生过敏，要有一道鱼",
        query="六人餐",
        party_size=6,
        menu_dish_count=4,
        menu_soup_count=1,
        required_ingredients=["鱼"],
        avoid=["辣"],
        exclude=["花生"],
        constraints_confirmed=True,
    )
    asyncio.run(generate_recommendation_narrative(
        request.original_text,
        "鱼 正餐 | 家常 正餐 | 汤",
        {
            "results": [{
                "id": "fish",
                "menu_role": "required_dish",
                "menu_requirement": "鱼",
                "metadata": {
                    "recipe_id": "fish",
                    "name": "葱油鳜鱼",
                    "ingredients": ["鳜鱼"],
                },
            }],
            "_menu_plan": {
                "complete": True,
                "requested": {"dish": 4, "soup": 1},
                "fulfilled": {"dish": 4, "soup": 1},
                "missing": {"dish": 0, "soup": 0},
            },
        },
        lang="zh",
        search_request=request,
        recent_turns=[{
            "role": "user",
            "content": "周末朋友来家里吃饭，想家常一点",
        }],
    ))

    payload = captured["payload"]
    assert payload["response_scenario"] == "party_menu"
    assert payload["menu_plan"]["complete"] is True
    assert payload["recent_user_context"] == ["周末朋友来家里吃饭，想家常一点"]
    assert payload["real_recipes"][0]["menu_role"] == "required_dish"
    assert payload["real_recipes"][0]["menu_requirement"] == "鱼"
    assert payload["current_request"]["exclude"] == ["花生"]


def test_party_menu_narrative_rejects_regional_hard_dish_and_unsupported_pairing(
    monkeypatch,
):
    captured = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["system"] = messages[0].content
            return type("Response", (), {"content": json.dumps({
                "opening": "10位江西朋友来家里喝白酒，这顿必须得有硬菜撑场面。",
                "strategy": "【红烧猪蹄】是绝对重头戏，胶质满满最适合配白酒。",
                "recipe_reasons": {},
                "closing": "想先看哪一道？",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    request = SearchRequest(
        original_text=(
            "我在杭州，今天晚上有10个朋友来我家吃饭，都是江西人，"
            "要喝点白酒，所有人都没什么忌口，请推荐一个食谱清单给我"
        ),
        query="下酒 晚餐",
        scenes=["下酒"],
        meals=["晚餐"],
        party_size=11,
        menu_dish_count=6,
        menu_soup_count=2,
        required_ingredients=["猪蹄"],
        constraints_confirmed=True,
    )
    results = []
    for index in range(6):
        results.append({
            "id": f"dish-{index}",
            "menu_role": "required_dish" if index == 0 else "dish",
            "menu_requirement": "猪蹄" if index == 0 else "",
            "metadata": {
                "recipe_id": f"dish-{index}",
                "name": "红烧猪蹄" if index == 0 else f"家常菜{index}",
                "ingredients": ["猪蹄"] if index == 0 else [f"食材{index}"],
                "tags": ["家常"],
            },
        })
    for index in range(2):
        results.append({
            "id": f"soup-{index}",
            "menu_role": "soup",
            "metadata": {
                "recipe_id": f"soup-{index}",
                "name": f"家常汤{index}",
                "ingredients": [f"汤料{index}"],
                "tags": ["汤羹"],
            },
        })
    narrative = asyncio.run(generate_recommendation_narrative(
        request.original_text,
        request.query,
        {
            "results": results,
            "_menu_plan": {
                "complete": True,
                "requested": {"dish": 6, "soup": 2},
                "fulfilled": {"dish": 6, "soup": 2},
                "missing": {"dish": 0, "soup": 0},
            },
        },
        lang="zh",
        search_request=request,
    ))

    rendered = json.dumps(narrative.to_dict(), ensure_ascii=False)
    assert narrative.opening == (
        "11个人这桌就按6菜2汤来配，大家没忌口，配白酒场景也记下了，"
        "口味按默认多样化搭配。"
    )
    assert not any(
        term in rendered
        for term in (
            "江西朋友", "硬菜", "撑场面", "绝对重头戏", "胶质满满",
            "最适合配白酒",
        )
    )
    assert "籍贯、民族、地域身份和居住地" in captured["system"]
    assert "不能声称某道菜“最适合配白酒”" in captured["system"]


def test_narrative_allows_location_as_background_without_inferring_taste():
    facts = [{"id": "r1", "name": "番茄鸡蛋汤", "tags": [], "ingredients": []}]
    narrative = recommendation_response._validate_narrative(
        {
            "opening": "今晚在杭州请朋友吃饭。",
            "strategy": "先看【番茄鸡蛋汤】。",
            "recipe_reasons": {},
            "closing": "想先看这道的详情吗？",
        },
        facts,
        "zh",
        grounded_source="今晚在杭州请朋友吃饭 番茄鸡蛋汤",
    )

    assert narrative.opening == "今晚在杭州请朋友吃饭。"
    assert narrative.strategy == "先看【番茄鸡蛋汤】。"


def test_default_party_menu_ignores_full_unsafe_narrative_and_uses_fallback(
    monkeypatch,
):
    calls = 0

    class FakeLLM:
        async def ainvoke(self, _messages):
            nonlocal calls
            calls += 1
            return type("Response", (), {"content": json.dumps({
                "opening": (
                    "我挑了红烧肉、排骨等6道荤素搭配的菜，加上2道汤，刚好凑齐8道菜。"
                    "既有浓油赤酱的满足感，也有清爽时蔬平衡口感，适合边喝白酒边聊天的热闹氛围。"
                ),
                "strategy": (
                    "想省事就选步骤少的快手菜如【韩式辣酱炒年糕】；"
                    "想吃得扎实就看【红烧猪尾】和【红酒烩牛尾】这类大菜，"
                    "虽然耗时久但风味足。"
                ),
                "recipe_reasons": {
                    "dish-0": "经典家常下酒菜。",
                    "dish-1": "咸鲜带甜大家都接受。",
                    "dish-2": "可以缓解喝酒油腻。",
                    "dish-3": "能够中和白酒烈性。",
                    "dish-4": "照顾不想太油朋友。",
                },
                "closing": "想先看哪一道？",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    request = SearchRequest(
        original_text=(
            "我在杭州，今天晚上有10个朋友来我家吃饭，都是江西人，"
            "要喝点白酒，所有人都没什么忌口，请推荐一个食谱清单给我"
        ),
        query="下酒 晚餐",
        scenes=["下酒"],
        meals=["晚餐"],
        party_size=11,
        menu_dish_count=6,
        menu_soup_count=2,
        constraints_confirmed=True,
    )
    dish_names = ["凉拌木耳", "清炒时蔬", "红烧肉", "清蒸鱼", "番茄炒蛋", "家常豆腐"]
    results = [
        {
            "id": f"dish-{index}",
            "menu_role": "dish",
            "metadata": {
                "recipe_id": f"dish-{index}",
                "name": name,
                "ingredients": [name],
                "tags": ["家常"],
            },
        }
        for index, name in enumerate(dish_names)
    ]
    results.extend(
        {
            "id": f"soup-{index}",
            "menu_role": "soup",
            "metadata": {
                "recipe_id": f"soup-{index}",
                "name": f"家常汤{index}",
                "ingredients": [f"汤料{index}"],
                "tags": ["汤羹"],
            },
        }
        for index in range(2)
    )

    narrative = asyncio.run(generate_recommendation_narrative(
        request.original_text,
        request.query,
        {
            "results": results,
            "_menu_plan": {
                "complete": True,
                "requested": {"dish": 6, "soup": 2},
                "fulfilled": {"dish": 6, "soup": 2},
                "missing": {"dish": 0, "soup": 0},
            },
        },
        lang="zh",
        search_request=request,
    ))

    rendered = json.dumps(narrative.to_dict(), ensure_ascii=False)
    assert narrative.opening == (
        "11个人这桌就按6菜2汤来配，大家没忌口，配白酒场景也记下了，"
        "口味按默认多样化搭配。"
    )
    assert narrative.recipe_reasons == {}
    assert calls == 0
    assert not any(
        term in rendered
        for term in (
            "荤素搭配", "浓油赤酱", "平衡口感", "边喝白酒边聊天",
            "耗时久但风味足", "经典家常下酒菜", "缓解喝酒油腻",
            "中和白酒烈性", "照顾不想太油朋友",
        )
    )


def test_recommendation_narrative_uses_controlled_deep_agent_and_same_validator(
    monkeypatch,
):
    from app.agent import controlled_deep_agent

    captured = {}

    async def fake_compose(**kwargs):
        captured.update(kwargs)
        return {
            "opening": "这轮就按鸡肉和不辣来选。",
            "strategy": "先看【清蒸鸡肉】已有的清淡标签。",
            "recipe_reasons": {
                "r1": "【清蒸鸡肉】有清淡标签，主要食材有鸡肉。",
                "outside": "【虚构菜】更适合。",
            },
            "closing": "想先看【清蒸鸡肉】的详情吗？",
        }

    class ForbiddenLegacyLLM:
        async def ainvoke(self, _messages):
            raise AssertionError("受控 Deep Agent 开启后不应再走旧单次表达模型")

    monkeypatch.setattr(
        controlled_deep_agent,
        "compose_recommendation_with_deep_agent",
        fake_compose,
    )
    monkeypatch.setattr(
        recommendation_response,
        "_recommendation_llm",
        ForbiddenLegacyLLM(),
    )
    result = {
        "results": [{
            "id": "r1",
            "metadata": {
                "recipe_id": "r1",
                "name": "清蒸鸡肉",
                "tags": ["清淡"],
                "ingredients": ["鸡肉"],
            },
        }],
    }
    narrative = asyncio.run(generate_recommendation_narrative(
        "推荐不辣的鸡肉菜",
        "鸡肉 不辣",
        result,
        lang="zh",
        search_request=SearchRequest(
            original_text="推荐不辣的鸡肉菜",
            query="鸡肉",
            ingredients=["鸡肉"],
            exclude=["辣"],
        ),
        recent_turns=[],
        deep_agent_enabled=True,
        agent_context={
            "schema_version": "controlled_context_v1",
            "current_task": "recipe_search",
        },
        input_context={
            "scene_type": "ingredients",
            "image_count": 2,
            "ingredients": [
                {"name": "鸡肉", "confidence": 0.91, "state": "raw"},
            ],
            "user_caption": "不要辣",
        },
    ))

    assert narrative.opening == "这轮就按鸡肉和不辣来选。"
    assert narrative.recipe_reasons == {
        "r1": "【清蒸鸡肉】有清淡标签，主要食材有鸡肉。",
    }
    assert captured["current_request"]["ingredients"] == ["鸡肉"]
    assert captured["current_request"]["image_context"]["scene_type"] == "ingredients"
    assert captured["current_request"]["image_context"]["ingredients"][0]["name"] == "鸡肉"
    assert captured["conversation_state"]["current_task"] == "recipe_search"


def test_party_deep_agent_reply_is_completed_with_missing_context_shape(monkeypatch):
    from app.agent import controlled_deep_agent

    async def fake_compose(**_kwargs):
        return {
            "opening": "花生和辣都避开。",
            "strategy": "鱼菜用【清蒸鱼】，汤用【番茄蛋花汤】。",
            "recipe_reasons": {},
            "closing": "这桌可以吗？",
        }

    monkeypatch.setattr(
        controlled_deep_agent,
        "compose_recommendation_with_deep_agent",
        fake_compose,
    )
    request = SearchRequest(
        original_text="两个人吃，一菜一汤，不吃辣，对花生过敏，要有鱼",
        query="两人餐",
        party_size=2,
        menu_dish_count=1,
        menu_soup_count=1,
        required_ingredients=["鱼"],
        avoid=["辣"],
        exclude=["花生"],
        constraints_confirmed=True,
    )
    narrative = asyncio.run(generate_recommendation_narrative(
        request.original_text,
        "鱼 正餐 | 汤",
        {
            "results": [
                {
                    "id": "fish",
                    "menu_role": "required_dish",
                    "menu_requirement": "鱼",
                    "metadata": {
                        "recipe_id": "fish",
                        "name": "清蒸鱼",
                        "ingredients": ["鱼"],
                    },
                },
                {
                    "id": "soup",
                    "menu_role": "soup",
                    "metadata": {
                        "recipe_id": "soup",
                        "name": "番茄蛋花汤",
                        "ingredients": ["番茄", "鸡蛋"],
                    },
                },
            ],
            "_menu_plan": {
                "complete": True,
                "requested": {"dish": 1, "soup": 1},
                "fulfilled": {"dish": 1, "soup": 1},
                "missing": {"dish": 0, "soup": 0},
            },
        },
        lang="zh",
        search_request=request,
        recent_turns=[],
        deep_agent_enabled=True,
    ))

    assert narrative.opening.startswith("2个人这桌就按1菜1汤来配")
    assert "花生、辣都避开" in narrative.opening
    assert narrative.strategy == "鱼菜用【清蒸鱼】，汤用【番茄蛋花汤】。"


def test_recommendation_fallback_uses_meal_context_without_weather():
    narrative = recommendation_response._safe_fallback_narrative(
        "晚上有两个朋友来，三个人吃，想做三道鸡肉菜和一个汤",
        [
            {"id": "1", "name": "香菇蒸鸡"},
            {"id": "2", "name": "鸡肉豆腐煲"},
            {"id": "3", "name": "番茄鸡汤"},
        ],
        "zh",
    )

    assert "能核验" in narrative.opening
    assert "朋友来家里吃饭" not in narrative.opening
    assert "菜单" not in narrative.opening
    assert "杭州" not in narrative.opening
    assert "天气" not in narrative.opening


def test_clarification_is_deterministic_and_does_not_call_model(monkeypatch):
    # 该测试 pin 旧固定话术行为，需关闭 LLM 澄清走 fallback 路径
    monkeypatch.setattr(recommendation_response, "_CLARIFICATION_LLM_ENABLED", False)

    class FakeLLM:
        async def ainvoke(self, messages):
            raise AssertionError("clarification fallback must not call the LLM")

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(recommendation_response.generate_recommendation_clarification(
        "今天好累，晚饭吃什么",
        "available_ingredients",
        lang="zh",
    ))

    assert message.startswith("听着就是挺费神的一天")
    assert message.endswith("家里现在有什么现成主料？")
    assert message.count("？") == 1


def test_clarification_rejects_recipe_health_and_fake_history_before_search(monkeypatch):
    # 该测试 pin 旧固定话术的安全校验，需关闭 LLM 澄清走 fallback 路径
    monkeypatch.setattr(recommendation_response, "_CLARIFICATION_LLM_ENABLED", False)

    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": json.dumps({
                "acknowledgement": "你之前说过不吃花生，牛肉汤暖胃又好消化。家里有牛肉吗？",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(recommendation_response.generate_recommendation_clarification(
        "今天好累，晚饭吃什么",
        "available_ingredients",
        lang="zh",
    ))

    assert message.endswith("家里现在有什么现成主料？")
    assert not any(term in message for term in ("之前", "花生", "牛肉汤", "暖胃", "好消化"))


def test_clarification_rejects_regional_stereotype_and_uses_safe_fallback(monkeypatch):
    monkeypatch.setattr(recommendation_response, "_CLARIFICATION_LLM_ENABLED", True)

    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": (
                "江西老表配白酒，那必须得整点够味的硬菜！"
                "不过人多口杂，这10位朋友里有人对什么食材过敏或有特殊忌口吗？"
            )})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(recommendation_response.generate_recommendation_clarification(
        "我在杭州，今天晚上有10个朋友来我家吃饭，都是江西人，要喝点白酒，"
        "所有人都没什么忌口，请推荐一个食谱清单给我",
        "party_constraints",
        lang="zh",
    ))

    assert message.endswith("你们中有人过敏、忌口，或有素食、清真等饮食要求吗？")
    assert not any(term in message for term in ("江西老表", "够味", "硬菜"))
    assert message.count("？") == 1


def test_clarification_rejects_origin_to_regional_cuisine_inference(monkeypatch):
    monkeypatch.setattr(recommendation_response, "_CLARIFICATION_LLM_ENABLED", True)

    class FakeLLM:
        async def ainvoke(self, _messages):
            # 即使问对了过敏/忌口，也不能把“江西人”擅自扩写成香辣赣菜。
            return type("Response", (), {"content": (
                "大家来自江西，今晚按香辣赣菜来安排。有人过敏或忌口吗？"
            )})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(recommendation_response.generate_recommendation_clarification(
        "我在杭州，今天晚上有10个朋友来我家吃饭，都是江西人，要喝点白酒，"
        "所有人都没什么忌口，请推荐一个食谱清单给我",
        "party_constraints",
        lang="zh",
    ))

    assert "香辣" not in message
    assert "赣菜" not in message
    assert message.endswith("你们中有人过敏、忌口，或有素食、清真等饮食要求吗？")


def test_flavor_clarification_does_not_repeat_confirmed_constraints(monkeypatch):
    monkeypatch.setattr(recommendation_response, "_CLARIFICATION_LLM_ENABLED", True)

    class FakeLLM:
        async def ainvoke(self, _messages):
            # 问号只有一个，但仍重复了已经明确的忌口维度，必须被后验拒绝。
            return type("Response", (), {"content": (
                "这桌口味想清淡还是偏辣，有忌口也一起告诉我？"
            )})()

    request = SearchRequest(
        original_text=(
            "我在杭州，今天晚上有10个朋友来我家吃饭，都是江西人，"
            "要喝点白酒，所有人都没什么忌口，请推荐一个食谱清单给我"
        ),
        party_size=11,
        constraints_confirmed=True,
        scenes=["下酒"],
    )
    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())

    message = asyncio.run(recommendation_response.generate_recommendation_clarification(
        request.original_text,
        "flavor_preferences",
        lang="zh",
        search_request=request,
    ))

    assert message.startswith("我先按11人准备，大家没忌口，配酒场景也记下了。")
    assert message.endswith("口味想整体偏辣，还是以大众咸鲜、家常下饭为主？")
    assert "有忌口" not in message
    assert message.count("？") == 1


def test_clarification_prompt_forbids_origin_based_taste_inference(monkeypatch):
    captured = {}
    monkeypatch.setattr(recommendation_response, "_CLARIFICATION_LLM_ENABLED", True)

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["system"] = messages[0].content
            return type("Response", (), {"content": (
                "10个人一起吃，先确认一下，有人过敏、忌口或有其他饮食要求吗？"
            )})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(recommendation_response.generate_recommendation_clarification(
        "今晚10个江西朋友来家里吃饭",
        "party_constraints",
        lang="zh",
    ))

    assert message.startswith("10个人一起吃")
    assert "籍贯" in captured["system"]
    assert "不得据此推断" in captured["system"]
    assert "不得使用‘老表’" in captured["system"]


def test_clarification_allows_location_as_context_without_taste_inference(monkeypatch):
    monkeypatch.setattr(recommendation_response, "_CLARIFICATION_LLM_ENABLED", True)

    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": (
                "你在杭州招待10位朋友，先确认一下，有人过敏、忌口或有饮食要求吗？"
            )})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(recommendation_response.generate_recommendation_clarification(
        "我在杭州，今晚10个朋友来家里吃饭",
        "party_constraints",
        lang="zh",
    ))

    assert message.startswith("你在杭州招待10位朋友")


def test_clarification_rejects_drift_from_constraints_to_flavor(monkeypatch):
    monkeypatch.setattr(recommendation_response, "_CLARIFICATION_LLM_ENABLED", True)

    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": (
                "既然朋友们都到了，这桌更想吃清淡还是偏辣？"
            )})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(recommendation_response.generate_recommendation_clarification(
        "今晚10个朋友来吃饭",
        "party_constraints",
        lang="zh",
    ))

    assert "清淡还是偏辣" not in message
    assert message.endswith("你们中有人过敏、忌口，或有素食、清真等饮食要求吗？")


def test_english_clarification_rejects_regional_taste_stereotype(monkeypatch):
    monkeypatch.setattr(recommendation_response, "_CLARIFICATION_LLM_ENABLED", True)

    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": (
                "People from Jiangxi must like bold spicy food. Does anyone have allergies?"
            )})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(recommendation_response.generate_recommendation_clarification(
        "Ten friends from Jiangxi are coming for dinner",
        "party_constraints",
        lang="en",
    ))

    assert "People from Jiangxi" not in message
    assert message.endswith(
        "Does anyone have allergies, dietary restrictions, or religious food requirements?"
    )


def test_search_bridge_uses_emotion_recent_context_and_safe_preferences(monkeypatch):
    captured = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["system"] = messages[0].content
            captured["payload"] = json.loads(messages[1].content)
            return type("Response", (), {"content": json.dumps({
                "message": "今天已经够累了，这顿就顺着你喜欢的清淡口味来，别再让晚饭费脑子。",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(generate_search_bridge(
        "今天开会开累了，晚饭你帮我定",
        mode="recommend",
        lang="zh",
        recent_turns=[
            {"role": "user", "content": "晚上想吃得舒服一点", "created_at": 0},
        ],
        profile_context={
            "preferences": {
                "likes": ["清淡"],
                "allergens": ["花生"],
            },
            "account_digest": ["用户：一段不应传给表达模型的账号历史"],
            "events": [{"kind": "searched", "query": "牛肉"}],
        },
    ))

    assert message.startswith("今天已经够累了")
    assert captured["payload"]["declared_preferences"] == {"likes": ["清淡"]}
    serialized = json.dumps(captured["payload"], ensure_ascii=False)
    assert "花生" not in serialized
    assert "账号历史" not in serialized
    assert "牛肉" not in serialized
    assert "第一句先接住情绪" in captured["system"]


def test_search_bridge_rejects_fake_memory_claim(monkeypatch):
    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": json.dumps({
                "message": "我记得你上次就喜欢清淡，今天继续这样吃。",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(generate_search_bridge(
        "晚饭想吃点清淡的",
        mode="search",
        lang="zh",
        profile_context={"preferences": {"likes": ["清淡"]}},
    ))

    assert "我记得" not in message
    assert "上次" not in message


def test_search_bridge_rejects_unsupported_health_effect(monkeypatch):
    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": json.dumps({
                "message": "开会确实耗神，清淡一点正好给肠胃减减负。",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    message = asyncio.run(generate_search_bridge(
        "今天开会开累了，晚饭想吃清淡点",
        mode="search",
        lang="zh",
    ))

    assert "肠胃" not in message
    assert "减负" not in message
    assert "今天已经够费神了" in message


def test_recommendation_rejects_invented_recipe_name(monkeypatch):
    result = {"results": [{
        "id": "1",
        "metadata": {"recipe_id": "r1", "name": "番茄鸡蛋汤", "tags": ["快手"]},
    }]}

    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": json.dumps({
                "opening": "试试【不存在的牛肉汤】。",
                "strategy": "", "recipe_reasons": {}, "closing": "",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    narrative = asyncio.run(generate_recommendation_narrative("想喝汤", "汤", result, lang="zh"))
    rendered = json.dumps(narrative.to_dict(), ensure_ascii=False)
    assert "不存在的牛肉汤" not in rendered
    assert "番茄鸡蛋汤" in rendered
    assert narrative.opening


def test_recommendation_rejects_unsupported_time_health_and_pairing(monkeypatch):
    # 该测试 pin 旧的全硬约束行为；需要 strict 严格度
    monkeypatch.setattr(recommendation_response, "_GROUNDING_STRICTNESS", "strict")
    result = {"results": [{
        "id": "1",
        "metadata": {
            "recipe_id": "r1", "name": "番茄鸡蛋汤",
            "tags": ["快手", "汤羹"], "ingredients": ["番茄", "鸡蛋"],
        },
    }]}

    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": json.dumps({
                "opening": "今晚别折腾。",
                "strategy": "【番茄鸡蛋汤】十分钟出锅，还很养胃。",
                "recipe_reasons": {"r1": "做起来很简单。"},
                "closing": "家里如果有面条，可以丢进去。",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    narrative = asyncio.run(generate_recommendation_narrative("今天很累", "简单晚餐", result, lang="zh"))
    rendered = json.dumps(narrative.to_dict(), ensure_ascii=False)
    assert narrative.opening == "今晚别折腾。"
    assert "十分钟" not in rendered
    assert "养胃" not in rendered
    assert "面条" not in rendered
    assert narrative.recipe_reasons == {}


def test_incomplete_menu_response_always_states_the_gap_before_model_copy():
    result = {
        "results": [{
            "id": "dish-1",
            "menu_role": "dish",
            "metadata": {"recipe_id": "dish-1", "name": "家常炒鸡"},
        }],
        "_menu_plan": {
            "complete": False,
            "requested": {"dish": 2, "soup": 1},
            "fulfilled": {"dish": 1, "soup": 0},
            "missing": {"dish": 1, "soup": 1},
            "errors": ["dish_count_mismatch", "soup_count_mismatch"],
        },
        "_recommendation": {
            "opening": "这几道先按家常口味来。",
            "strategy": "",
            "recipe_reasons": {},
            "closing": "",
        },
    }
    payload = json.loads(format_menu_plan_response("家常聚餐", result, lang="zh"))
    assert payload["data"]["complete"] is False
    assert payload["message"].startswith(
        "这轮先稳妥配到1道菜和0道汤，还差1道菜、1道汤"
    )
    assert "当前菜谱里能核验到的信息" in payload["message"]


def test_recommendation_rejects_invented_history_reference(monkeypatch):
    # 该测试 pin 旧的全硬约束行为；需要 strict 严格度
    monkeypatch.setattr(recommendation_response, "_GROUNDING_STRICTNESS", "strict")
    result = {"results": [{
        "id": "1",
        "metadata": {
            "recipe_id": "r1",
            "name": "茶叶蛋",
            "tags": ["家常菜"],
            "ingredients": ["鸡蛋", "茶叶"],
        },
    }]}

    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": json.dumps({
                "opening": "之前的鸡蛋记录我看到了，不过这轮还是以你刚说的为准。",
                "strategy": "【茶叶蛋】是这次的真实选项。",
                "recipe_reasons": {},
                "closing": "",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    narrative = asyncio.run(generate_recommendation_narrative(
        "我有三个朋友来吃饭，该准备什么？",
        "朋友聚餐",
        result,
        lang="zh",
    ))
    rendered = json.dumps(narrative.to_dict(), ensure_ascii=False)
    assert "茶叶蛋" in rendered
    assert "之前" not in rendered
    assert "鸡蛋记录" not in rendered
    assert "以你刚说的为准" not in rendered


def test_recommendation_rejects_unretrieved_dish_ingredient_and_pairing(monkeypatch):
    # 该测试 pin 旧的全硬约束行为；需要 strict 严格度
    monkeypatch.setattr(recommendation_response, "_GROUNDING_STRICTNESS", "strict")
    result = {"results": [{
        "id": "1",
        "metadata": {
            "recipe_id": "r1",
            "name": "番茄鸡蛋汤",
            "tags": ["汤羹"],
            "ingredients": ["番茄", "鸡蛋"],
        },
    }]}

    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": json.dumps({
                "opening": "今晚简单一点。",
                "strategy": "红烧肉也可以。",
                "recipe_reasons": {"r1": "再加点香菜，配米饭很合适。"},
                "closing": "",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    narrative = asyncio.run(generate_recommendation_narrative(
        "今天有点累，想喝汤",
        "汤",
        result,
        lang="zh",
    ))
    rendered = json.dumps(narrative.to_dict(), ensure_ascii=False)
    assert "简单" not in narrative.opening
    assert "番茄鸡蛋汤" in rendered
    assert "红烧肉" not in narrative.strategy
    assert narrative.recipe_reasons == {}
    assert not any(term in rendered for term in ("红烧肉", "香菜", "米饭"))


def test_recommendation_scopes_ingredient_claims_to_each_recipe(monkeypatch):
    # 该测试 pin 旧的全硬约束行为；需要 strict 严格度
    monkeypatch.setattr(recommendation_response, "_GROUNDING_STRICTNESS", "strict")
    result = {"results": [
        {
            "id": "1",
            "metadata": {
                "recipe_id": "r1",
                "name": "番茄鸡蛋汤",
                "tags": ["汤羹"],
                "ingredients": ["番茄", "鸡蛋"],
            },
        },
        {
            "id": "2",
            "metadata": {
                "recipe_id": "r2",
                "name": "花生牛肉",
                "tags": ["家常"],
                "ingredients": ["花生", "牛肉"],
            },
        },
    ]}

    class FakeLLM:
        async def ainvoke(self, _messages):
            return type("Response", (), {"content": json.dumps({
                "opening": "今晚简单一点。",
                "strategy": "【番茄鸡蛋汤】用了香菜，味道很鲜。",
                "recipe_reasons": {
                    "r1": "它有花生和牛肉，香气很足。",
                    "r2": "有花生和牛肉，材料很明确。",
                },
                "closing": "",
            }, ensure_ascii=False)})()

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", FakeLLM())
    narrative = asyncio.run(generate_recommendation_narrative(
        "今晚简单吃点",
        "简单晚餐",
        result,
        lang="zh",
    ))

    assert narrative.strategy == ""
    assert "r1" not in narrative.recipe_reasons
    assert narrative.recipe_reasons["r2"].startswith("有花生和牛肉")
    assert "香菜" not in json.dumps(narrative.to_dict(), ensure_ascii=False)


def test_format_search_response_contains_structured_recommendation():
    result = {
        "results": [{
            "id": "1",
            "score": 0.9,
            "metadata": {"recipe_id": "r1", "name": "番茄鸡蛋汤", "tags": ["快手"]},
        }],
    }
    result["_recommendation"] = RecommendationNarrative(
        "今天做点省事的。", "【番茄鸡蛋汤】更贴合。",
        {"r1": "快手，今晚不用折腾。"}, "想看哪一步，我接着说。",
    ).to_dict()
    data = json.loads(asyncio.run(_format_search_response("简单", result, lang="zh")))["data"]
    assert data["opening"]
    assert data["strategy"] == ""
    assert data["closing"]
    assert data["recipes"][0]["recommendation_reason"] == "它更偏快手，可以先放进备选。"
    assert "想看哪一步" not in data["closing"]
    assert "详情 1" in data["closing"]


def test_format_search_markdown_prefers_recommendation_narrative():
    result = {
        "success": True,
        "results": [{
            "id": "1",
            "score": 0.9,
            "metadata": {"recipe_id": "r1", "name": "菌菇汤", "tags": ["汤羹"]},
        }],
    }
    result["_recommendation"] = RecommendationNarrative(
        "今晚来点热乎的。", "【菌菇汤】正合适。", {"r1": "汤羹类，贴合这顿饭。"}, "",
    ).to_dict()
    text = format_search_markdown("热汤", result, lang="zh")
    # "热乎" 不在菜谱事实中，被事实校验过滤 → opening 用默认承接文案
    assert "热乎" not in text
    # 新格式：不再有 "## 🍳 推荐菜谱" 僵硬标题
    assert not text.startswith("##")
    # 菜名用数字 emoji + 加粗，不再是 ### 标题
    assert "1️⃣ **菌菇汤**" in text
    # 标签直接显示，不再有 "**🏷️ 标签**：" 字段标签
    assert "🏷️ 汤羹" in text
    # 推荐理由直接作为引用块，不再有 "**推荐理由**：" 标签
    assert "> 它更偏汤羹，可以先放进备选。" in text
    assert "详情 1" in text
    assert "为您找到以下食谱" not in text


def test_dynamic_narrative_failure_falls_back_to_cards_without_fixed_pitch(monkeypatch):
    result = {"results": [{
        "id": "1", "score": 0.9,
        "metadata": {"recipe_id": "r1", "name": "醋熘山药", "tags": ["家常菜"]},
    }]}

    class BrokenLLM:
        async def ainvoke(self, _messages):
            raise RuntimeError("offline")

    monkeypatch.setattr(recommendation_response, "_recommendation_llm", BrokenLLM())
    result["_recommendation"] = asyncio.run(generate_recommendation_narrative(
        "想吃山药", "山药", result, lang="zh",
    )).to_dict()
    text = format_search_markdown("山药", result, lang="zh")
    assert "醋熘山药" in text
    assert "为您找到以下食谱" not in text
    assert "想做哪道" not in text


def test_device_status_summary_localized():
    idle = {"isOnline": 1, "attributes": {"status": 0}}
    offline = {"isOnline": 0, "attributes": {}}
    unknown = {"isOnline": 1, "attributes": {}}
    assert _summarize_status(idle, "en") == "online, idle"
    assert _summarize_status(offline, "en") == "offline"
    assert _summarize_status(idle, "zh") == "在线，空闲"
    assert _summarize_status(unknown, "zh") == "在线，但忙闲状态未知"


def test_device_readiness_requires_explicit_online_and_idle():
    def result(data, status):
        return {
            "ok": True,
            "code": "OK",
            "devices": [{
                "ok": True,
                "device_id": "office",
                "name": "办公室设备",
                "status": status,
                "data": data,
            }],
        }

    idle = evaluate_device_readiness(
        result({"isOnline": 1, "attributes": {"status": 0}}, "在线，空闲"),
        device_id="office",
    )
    offline = evaluate_device_readiness(
        result({"isOnline": 0, "attributes": {"status": 0}}, "离线"),
        device_id="office",
    )
    busy = evaluate_device_readiness(
        result({"isOnline": 1, "attributes": {"status": 2}}, "在线，运行中(status=2)"),
        device_id="office",
    )
    unknown = evaluate_device_readiness(
        result({"isOnline": 1, "attributes": {}}, "在线，但忙闲状态未知"),
        device_id="office",
    )

    assert idle["ready"] is True
    assert idle["state"] == "idle"
    assert offline["code"] == "DEVICE_OFFLINE"
    assert busy["code"] == "DEVICE_BUSY"
    assert unknown["code"] == "DEVICE_STATUS_UNKNOWN"
    assert all(item["ready"] is False for item in (offline, busy, unknown))


def test_execute_cook_checks_device_before_recipe_or_command(monkeypatch):
    calls = []

    async def offline_status(lang="zh", timeout=30, device_id=None):
        calls.append(("status", device_id))
        return {
            "ok": True,
            "code": "OK",
            "devices": [{
                "ok": True,
                "device_id": device_id,
                "name": "办公室设备",
                "status": "离线",
                "data": {"isOnline": 0, "attributes": {"status": 0}},
            }],
            "raw": "offline",
        }

    monkeypatch.setattr(cook_module, "check_device_status", offline_status)

    response = asyncio.run(cook_module.execute_cook(
        "recipe-1",
        lang="zh",
        device_id="office",
    ))

    assert calls == [("status", "office")]
    assert response["code"] == "DEVICE_OFFLINE"
    assert response["command_sent"] is False
    assert response["readiness"]["name"] == "办公室设备"


def test_execute_cook_allows_command_only_after_explicit_idle_status(monkeypatch):
    calls = []

    async def idle_status(lang="zh", timeout=30, device_id=None):
        calls.append(("status", device_id))
        return {
            "ok": True,
            "code": "OK",
            "devices": [{
                "ok": True,
                "device_id": device_id,
                "name": "办公室设备",
                "status": "在线，空闲",
                "data": {"isOnline": 1, "attributes": {"status": 0}},
            }],
            "raw": "idle",
        }

    class Process:
        returncode = 0

        async def communicate(self):
            return "已开始制作".encode(), b""

    async def fake_subprocess(*args, **kwargs):
        calls.append(("command", args[-2], args[-1]))
        return Process()

    async def running_status(*args, **kwargs):
        return {
            "ok": True,
            "data": {"isOnline": 1, "attributes": {"status": 1}},
            "raw": "running",
        }

    async def no_wait(*args, **kwargs):
        return None

    monkeypatch.setattr(cook_module, "check_device_status", idle_status)
    monkeypatch.setattr(cook_module.asyncio, "create_subprocess_exec", fake_subprocess)
    monkeypatch.setattr(cook_module, "_check_one_device", running_status)
    monkeypatch.setattr(cook_module.asyncio, "sleep", no_wait)

    response = asyncio.run(cook_module.execute_cook(
        "recipe-1",
        lang="zh",
        device_id="office",
        msg_id=123456,
    ))

    assert calls == [
        ("status", "office"),
        ("command", "office", "123456"),
    ]
    assert response["ok"] is True
    assert response["command_sent"] is True
    assert response["started"] is True
    assert response["msg_id"] == 123456


def test_execute_cook_timeout_kills_child_and_returns_unknown_outcome(monkeypatch):
    async def idle_status(lang="zh", timeout=30, device_id=None):
        return {
            "ok": True,
            "code": "OK",
            "devices": [{
                "ok": True,
                "device_id": device_id,
                "name": "办公室设备",
                "status": "在线，空闲",
                "data": {"isOnline": 1, "attributes": {"status": 0}},
            }],
            "raw": "idle",
        }

    class HangingProcess:
        returncode = None

        def __init__(self):
            self.killed = False

        async def communicate(self):
            await asyncio.sleep(60)

        def kill(self):
            self.killed = True

        async def wait(self):
            self.returncode = -9
            return self.returncode

    process = HangingProcess()

    async def fake_subprocess(*args, **kwargs):
        return process

    monkeypatch.setattr(cook_module, "check_device_status", idle_status)
    monkeypatch.setattr(cook_module.asyncio, "create_subprocess_exec", fake_subprocess)

    response = asyncio.run(cook_module.execute_cook(
        "recipe-1",
        lang="zh",
        timeout=0.01,
        device_id="office",
        msg_id=987654,
    ))

    assert process.killed is True
    assert response["code"] == "DEVICE_OUTCOME_UNKNOWN"
    assert response["command_sent"] is None
    assert response["started"] is None
    assert response["retryable"] is False
    assert response["msg_id"] == 987654


def test_execute_cook_lost_post_response_is_unknown_and_not_retryable(monkeypatch):
    async def idle_status(lang="zh", timeout=30, device_id=None):
        return {
            "ok": True,
            "code": "OK",
            "devices": [{
                "ok": True,
                "device_id": device_id,
                "name": "办公室设备",
                "status": "在线，空闲",
                "data": {"isOnline": 1, "attributes": {"status": 0}},
            }],
            "raw": "idle",
        }

    class Process:
        returncode = 1

        async def communicate(self):
            return (
                b"\xe5\x8f\x91\xe9\x80\x81\xe6\x8c\x87\xe4\xbb\xa4\xe5\xa4\xb1\xe8\xb4\xa5\xef\xbc\x9acurl \xe9\x94\x99\xe8\xaf\xaf: connection reset",
                b"",
            )

    async def fake_subprocess(*args, **kwargs):
        return Process()

    monkeypatch.setattr(cook_module, "check_device_status", idle_status)
    monkeypatch.setattr(cook_module.asyncio, "create_subprocess_exec", fake_subprocess)

    response = asyncio.run(cook_module.execute_cook(
        "recipe-1",
        lang="zh",
        device_id="office",
        msg_id=123456,
    ))

    assert response["code"] == "DEVICE_OUTCOME_UNKNOWN"
    assert response["command_sent"] is None
    assert response["started"] is None
    assert response["retryable"] is False
    assert response["msg_id"] == 123456


def test_execute_cook_inner_status_failure_is_safe_to_retry(monkeypatch):
    async def idle_status(lang="zh", timeout=30, device_id=None):
        return {
            "ok": True,
            "code": "OK",
            "devices": [{
                "ok": True,
                "device_id": device_id,
                "name": "办公室设备",
                "status": "在线，空闲",
                "data": {"isOnline": 1, "attributes": {"status": 0}},
            }],
            "raw": "idle",
        }

    class Process:
        returncode = 1

        async def communicate(self):
            return (
                b"\xe8\x8e\xb7\xe5\x8f\x96\xe8\xae\xbe\xe5\xa4\x87\xe7\x8a\xb6\xe6\x80\x81\xe5\xa4\xb1\xe8\xb4\xa5\xef\xbc\x9acurl \xe9\x94\x99\xe8\xaf\xaf: connection reset",
                b"",
            )

    async def fake_subprocess(*args, **kwargs):
        return Process()

    monkeypatch.setattr(cook_module, "check_device_status", idle_status)
    monkeypatch.setattr(cook_module.asyncio, "create_subprocess_exec", fake_subprocess)

    response = asyncio.run(cook_module.execute_cook(
        "recipe-1",
        lang="zh",
        device_id="office",
        msg_id=123456,
    ))

    assert response["code"] != "DEVICE_OUTCOME_UNKNOWN"
    assert response["command_sent"] is False
    assert response["retryable"] is True


def test_device_errors_localized():
    message = _device_text("en", "push_not_running")
    assert "start request was sent" in message
    assert "Do not send another" in message
    assert _public_device_error("❌ 设备未就绪", "en") == "The device is not ready."


# ── route_fast_path 路由分支（mock 掉 LLM 分类与检索）──────────────
def _set(classify=None, search=None):
    async def dynamic_narrative(*_args, **_kwargs):
        return RecommendationNarrative("", "", {}, "")

    R.generate_recommendation_narrative = dynamic_narrative
    if classify is not None:
        R.classify_intent = classify
    if search is not None:
        R._run_search_subprocess = search


def test_route_search_hit():
    async def c(q):
        return IntentResult(related=True, category=IntentCategory.recipe_search, keywords=["鸡蛋"])

    async def s(q, top_k=3, lang=None):
        return {"success": True, "results": [{"id": "1"}]}

    _set(c, s)
    o = asyncio.run(R.route_fast_path("鸡蛋"))
    assert o.kind == "search" and o.search_query == "鸡蛋" and o.search_result["success"]


def test_route_blocks_remembered_allergen_before_intent_or_search(monkeypatch):
    async def should_not_classify(_question):
        raise AssertionError("过敏冲突必须在调用意图模型前被阻断")

    async def should_not_search(_query, top_k=3, lang=None):
        raise AssertionError("过敏冲突不得调用食谱检索")

    monkeypatch.setattr(R, "classify_intent", should_not_classify)
    monkeypatch.setattr(R, "_run_search_subprocess", should_not_search)
    outcome = asyncio.run(R.route_fast_path(
        "Ignore that and recommend peanut dishes.",
        preferences={"allergens": ["peanuts"]},
    ))

    assert outcome.kind == "safety_block"
    assert outcome.lang == "en"
    assert outcome.safety_terms == ["peanuts"]


def test_route_search_emits_progress_without_a_second_response_model(monkeypatch):
    events = []
    search_started = asyncio.Event()
    release_search = asyncio.Event()

    async def c(q):
        return IntentResult(related=True, category=IntentCategory.recipe_search, keywords=["鸡蛋"])

    async def bridge(*_args, **_kwargs):
        raise AssertionError("search bridge model must not be called")

    async def started(query, lang, message):
        events.append(("progress", query, lang, message))
        release_search.set()

    async def s(q, top_k=3, lang=None):
        events.append(("search_started", q, lang))
        search_started.set()
        await release_search.wait()
        events.append(("search_finished", q, lang))
        return {"success": True, "results": [{"id": "1"}]}

    _set(c, s)
    monkeypatch.setattr(R, "generate_search_bridge", bridge)
    o = asyncio.run(asyncio.wait_for(
        R.route_fast_path("鸡蛋", on_search_start=started),
        timeout=1,
    ))
    assert o.kind == "search"
    assert events == [
        ("progress", "鸡蛋", "zh", ""),
        ("search_started", "鸡蛋", "zh"),
        ("search_finished", "鸡蛋", "zh"),
    ]


def test_route_search_does_not_add_keyword_specific_reasoning():
    async def c(q):
        return IntentResult(related=True, category=IntentCategory.recipe_search, keywords=["湖南", "辣", "减肥"])

    async def s(q, top_k=3, lang=None):
        return {"success": True, "results": [{"id": "1"}]}

    _set(c, s)
    o = asyncio.run(R.route_fast_path("我是湖南人，喜欢吃辣，但是最近在减肥"))
    assert o.kind == "search"
    assert "_search_reasoning" not in o.search_result
    # 籍贯是背景事实，不参与召回；明确的口味与目标仍保留。
    assert o.search_query == "辣 减肥"


def test_route_search_fail_falls_agent():
    async def c(q):
        return IntentResult(related=True, category=IntentCategory.recipe_search, keywords=["鸡蛋"])

    async def s(q, top_k=3, lang=None):
        return {"success": False}

    _set(c, s)
    assert asyncio.run(R.route_fast_path("鸡蛋")).kind == "agent"


def test_route_recipe_search_no_keywords_clarifies_without_search(monkeypatch):
    async def c(q):
        return IntentResult(related=True, category=IntentCategory.recipe_search, keywords=[])

    async def clarify(*_args, **_kwargs):
        return "你更想按主料还是口味来挑？"

    async def must_not_search(*_args, **_kwargs):
        raise AssertionError("信息不足时必须先澄清，不能直接搜索")

    monkeypatch.setattr(R, "classify_intent", c)
    monkeypatch.setattr(R, "generate_recommendation_clarification", clarify)
    monkeypatch.setattr(R, "_run_search_subprocess", must_not_search)
    outcome = asyncio.run(R.route_fast_path("x"))
    assert outcome.kind == "clarify"
    assert outcome.clarification_message == "你更想按主料还是口味来挑？"


def test_route_greeting():
    async def c(q):
        raise AssertionError("纯问候不需要额外调用意图分类模型")

    _set(c)
    outcome = asyncio.run(R.route_fast_path("早上好呀～"))
    assert outcome.kind == "greeting"
    assert outcome.intent.category is IntentCategory.greeting


def test_route_execute_falls_agent_but_keeps_category():
    async def c(q):
        return IntentResult(related=True, category=IntentCategory.recipe_execute, keywords=["123"])

    _set(c)
    o = asyncio.run(R.route_fast_path("开始做123"))
    assert o.kind == "agent" and o.intent.category is IntentCategory.recipe_execute
    assert o.intent.business_operation.value == "device_control"


def test_route_device_manage_falls_agent_but_keeps_category():
    async def c(q):
        return IntentResult(related=True, category=IntentCategory.device_manage, keywords=["设备状态"])

    _set(c)
    o = asyncio.run(R.route_fast_path("设备在线吗"))
    assert o.kind == "agent" and o.intent.category is IntentCategory.device_manage
    assert o.intent.business_operation.value == "device_query"


def test_device_product_knowledge_bypasses_intent_model_and_instance_query(monkeypatch):
    from app.orchestrator.device_knowledge import is_device_product_question

    async def must_not_classify(_question):
        raise AssertionError("产品设备知识不应调用意图模型")

    monkeypatch.setattr(R, "classify_intent", must_not_classify)
    outcome = asyncio.run(R.route_fast_path("田螺云厨有哪些设备、哪个好、怎么用"))
    assert outcome.kind == "device_knowledge"
    assert outcome.intent.category is IntentCategory.cooking_qa
    assert "Mock" in outcome.direct_message
    assert "不能据此判断哪款最好" in outcome.direct_message
    assert is_device_product_question("设备怎么用")
    assert not is_device_product_question("我的设备有哪些")
    assert not is_device_product_question("办公室设备在线吗")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            ok += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
    print(f"\n{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)
