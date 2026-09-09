"""Mock Device 菜谱详情规范化、保存和展示回归。"""
import asyncio
import json

import app.agent.participle_agent as agent_module
import app.orchestrator.cook as cook_module
import app.orchestrator.recipe_completion as completion_module
import app.recipe_detail_store as detail_store_module
from app.orchestrator.recipe_completion import (
    build_recipe_detail_seed,
    complete_ingredient_quantities,
    complete_missing_recipe_fields,
)
from app.orchestrator.ingredient_quantities import parse_grounded_ingredient
from app.orchestrator.recipe_detail import (
    format_recipe_detail,
    normalize_recipe_detail,
    sanitize_recipe_detail_payload,
)


def _remote_recipe() -> dict:
    ingredients = {
        "mainMaterials": [
            {"foodIngredientName": "20 ml sesame oil ", "amount": None},
            {"foodIngredientName": "400 - 600 g chicken wings (approx. 8)"},
        ],
        "recipeAccessories": [],
        "recipeSeasoning": [],
        "other": [],
    }
    return {
        "id": "2509088617308622850",
        "name": "Braised Chicken Wings",
        "landscapeImageUrl": "https://example.com/landscape.jpeg",
        "portraitImageUrl": "https://example.com/portrait.jpeg",
        "recipeIngredientsVoList": ingredients,
        # 与 VoList 重复；规范化后不能出现四条食材。
        "recipeIngredientsApkVo": ingredients,
        "recipeStepVoList": [
            {
                "serialNumb": 1,
                "stepType": 0,
                "stepDesc": "Install saute blade. Secure the jug lid.",
                "stepTime": 0,
                "recipeStepParameterVoList": None,
            },
            {
                "serialNumb": 2,
                "stepType": 1,
                "stepDesc": "Saute",
                "stepTime": 420,
                "recipeStepParameterVoList": [{
                    "stepId": "step-2",
                    "time": 420,
                    "speed": "1",
                    "power": 10,
                    "temperature": 120,
                    "accessoriesType": 8,
                    "deviceModels": [""],
                }],
            },
        ],
        "cookingTime": 2100,
        "serviceSize": 3,
        "deviceModelIds": "1,2497129465493565442",
        "accessoryIds": "8",
        "updateTime": 1782099043000,
        "createTime": 1736873413000,
        "isCustomFood": True,
        "isCollect": True,
        "isPurchase": False,
    }


def test_normalize_recipe_detail_maps_real_api_fields_without_duplicate_ingredients():
    detail = normalize_recipe_detail(_remote_recipe(), language="en")

    assert detail["schema_version"] == "recipe_detail_v1"
    assert detail["recipe_id"] == "2509088617308622850"
    assert detail["name"] == "Braised Chicken Wings"
    assert [item["name"] for item in detail["ingredients"]] == [
        "20 ml sesame oil",
        "400 - 600 g chicken wings (approx. 8)",
    ]
    assert [step["description"] for step in detail["steps"]] == [
        "Install saute blade. Secure the jug lid.",
        "Saute",
    ]
    assert detail["steps"][1]["parameters"][0] == {
        "parameter_id": "step-2",
        "time_seconds": 420,
        "temperature_c": 120,
        "speed": "1",
        "power": 10,
        "turn": None,
        "weight": None,
        "preset_pressure": None,
        "cook_pressure": None,
        "accessory_type": 8,
        "accessory_image_url": None,
        "device_models": [],
    }
    assert detail["cooking_time_seconds"] == 2100
    assert detail["servings"] == 3
    assert detail["executable"] is True


def test_recipe_detail_raw_payload_removes_account_specific_state():
    raw = sanitize_recipe_detail_payload(_remote_recipe())
    assert "isCollect" not in raw
    assert "isPurchase" not in raw
    assert raw["id"] == "2509088617308622850"


def test_format_recipe_detail_uses_only_normalized_facts():
    text = format_recipe_detail(
        normalize_recipe_detail(_remote_recipe(), language="en"),
        lang="en",
    )
    assert "## 🍳 Braised Chicken Wings" in text
    assert "I’ve organized the saved ingredients and cooking order" in text
    assert "### Recipe description" in text
    assert "No saved description is available" in text
    assert "![Braised Chicken Wings #320px #240px](https://example.com/landscape.jpeg)" in text
    assert "### Ingredients" in text
    assert "20 ml sesame oil" in text
    assert "### Seasonings" in text
    assert "No separate seasonings are listed" in text
    assert "### Cooking steps" in text
    assert "Saute (7 min; 120°C; speed 1; power 10)" in text
    assert "### Cooking tips" in text
    assert "No separate cooking tip is recorded, so I haven’t added one." in text
    assert "Recipe ID" not in text
    assert "calorie" not in text.lower()


def test_ai_completion_fills_only_missing_display_fields(monkeypatch):
    detail = normalize_recipe_detail(_remote_recipe(), language="en")
    detail["steps"] = []
    detail["cooking_time_seconds"] = None
    detail["servings"] = None
    detail["executable"] = False

    class Reply:
        content = json.dumps({
            "steps": [
                {
                    "description": "Heat the sesame oil and dissolve the sugar.",
                    "duration_seconds": 300,
                },
                {
                    "description": "Add the chicken wings and braise until cooked through.",
                    "duration_seconds": 1500,
                },
            ],
            "total_time_seconds": 2100,
            "servings": 3,
            "cooking_tip": "Slice across the grain and stop heating as soon as the meat is cooked.",
        })

    class FakeLLM:
        async def ainvoke(self, messages):
            payload = json.loads(messages[-1].content)
            assert payload["name"] == "Braised Chicken Wings"
            assert payload["ingredients"] == [
                "20 ml sesame oil",
                "400 - 600 g chicken wings (approx. 8)",
            ]
            return Reply()

    monkeypatch.setattr(completion_module, "_recipe_completion_llm", FakeLLM())
    completed = asyncio.run(complete_missing_recipe_fields(detail))

    assert completed["ai_generated"] is True
    assert completed["generated_fields"] == [
        "steps",
        "cooking_time_seconds",
        "servings",
        "tips",
    ]
    assert completed["cooking_time_seconds"] == 2100
    assert completed["servings"] == 3
    assert completed["steps"][0]["type"] == "ai_generated_manual"
    assert completed["steps"][0]["parameters"] == []
    assert completed["tips"].startswith("Slice across the grain")
    assert completed["executable"] is False

    text = format_recipe_detail(completed, lang="en")
    assert "**AI-completed draft**" not in text
    assert "estimated 5 min" in text
    assert "### Cooking tips" in text


def test_ai_completion_does_not_overwrite_real_detail(monkeypatch):
    detail = normalize_recipe_detail(_remote_recipe(), language="en")
    detail["tips"] = "Keep the heat steady."

    class MustNotRun:
        async def ainvoke(self, messages):
            raise AssertionError("complete real detail must not call the model")

    monkeypatch.setattr(completion_module, "_recipe_completion_llm", MustNotRun())
    completed = asyncio.run(complete_missing_recipe_fields(detail))
    assert completed is detail
    assert "ai_generated" not in completed


def test_ai_completion_rewrites_steps_that_add_unlisted_ingredients(monkeypatch):
    detail = normalize_recipe_detail(_remote_recipe(), language="zh")
    detail["steps"] = []
    detail["cooking_time_seconds"] = None
    detail["servings"] = None

    class Reply:
        def __init__(self, content):
            self.content = json.dumps(content, ensure_ascii=False)

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return Reply({
                    "steps": [
                        {"description": "加入鸡翅和姜翻炒。", "duration_seconds": 300},
                        {"description": "加水焖煮。", "duration_seconds": 900},
                    ],
                    "total_time_seconds": 1500,
                    "servings": 3,
                    "cooking_tip": "鸡翅下锅后勤翻动，受热会更均匀。",
                })
            assert '"姜"' in messages[0].content
            return Reply({
                "steps": [
                    {"description": "加入鸡翅翻炒。", "duration_seconds": 300},
                    {"description": "加水焖煮。", "duration_seconds": 900},
                ],
                "total_time_seconds": 1500,
                "servings": 3,
                "cooking_tip": "鸡翅下锅后勤翻动，受热会更均匀。",
            })

    llm = FakeLLM()
    monkeypatch.setattr(completion_module, "_recipe_completion_llm", llm)
    completed = asyncio.run(complete_missing_recipe_fields(detail))

    assert llm.calls == 2
    assert completed["ai_generated"] is True
    assert all("姜" not in step["description"] for step in completed["steps"])


def test_ai_completion_adds_missing_ingredient_quantities_and_tip(monkeypatch):
    detail = build_recipe_detail_seed({
        "cookId": "recipe-beef-1",
        "name": "青椒炒牛肉",
        "ingredients": ["牛肉", "青椒", "食用油"],
    }, "zh")
    detail["steps"] = [{
        "number": 1,
        "type": "ai_generated_manual",
        "description": "牛肉和青椒炒熟。",
        "duration_seconds": 300,
        "parameters": [],
    }]
    detail["cooking_time_seconds"] = 900
    detail["servings"] = 2

    class Reply:
        content = json.dumps({
            "ingredient_quantities": [
                {"name": "牛肉", "amount": 250, "unit": "g"},
                {"name": "青椒", "amount": 150, "unit": "g"},
                {"name": "食用油", "amount": 20, "unit": "ml"},
            ],
            "cooking_tip": "牛肉逆着纹理切薄片，炒到刚变色就盛出，回锅后快速合炒。",
        }, ensure_ascii=False)

    class FakeLLM:
        async def ainvoke(self, messages):
            payload = json.loads(messages[-1].content)
            assert payload["ingredients_needing_quantities"] == [
                "牛肉", "青椒", "食用油",
            ]
            return Reply()

    monkeypatch.setattr(completion_module, "_recipe_completion_llm", FakeLLM())
    completed = asyncio.run(complete_missing_recipe_fields(detail))

    assert [
        (item["name"], item["amount"], item["unit"])
        for item in completed["ingredients"]
    ] == [
        ("牛肉", 250, "g"),
        ("青椒", 150, "g"),
        ("食用油", 20, "ml"),
    ]
    text = format_recipe_detail(completed, lang="zh")
    assert "- 250g 牛肉" in text
    assert "### 烹饪小技巧" in text
    assert "逆着纹理切薄片" in text


def test_milvus_candidate_builds_non_executable_completion_seed():
    detail = build_recipe_detail_seed({
        "cookId": "recipe-1",
        "name": "鸡翅",
        "image": "https://example.com/chicken.jpeg",
        "ingredients": ["鸡翅 500g", "生抽 20ml"],
        "tags": ["家常菜"],
    }, "zh")

    assert detail["recipe_id"] == "recipe-1"
    assert detail["name"] == "鸡翅"
    assert [
        (item["name"], item["amount"], item["unit"])
        for item in detail["ingredients"]
    ] == [
        ("鸡翅", 500, "g"),
        ("生抽", 20, "ml"),
    ]
    assert detail["steps"] == []
    assert detail["executable"] is False


def test_detail_card_uses_saved_description_and_exact_seasoning_column():
    detail = build_recipe_detail_seed({
        "cookId": "recipe-1",
        "name": "贡菜炒牛肉",
        "description": "贡菜脆爽，牛肉鲜嫩，是一道快手炒菜。",
        "image": "https://example.com/beef.jpg",
        "ingredients": [
            "红椒 50克", "蒜 20克", "盐 5克", "生抽 10克",
            "贡菜 50克", "牛肉 200克",
        ],
        "seasonings": ["蒜 20克", "盐 5克", "生抽 10克"],
    }, "zh")
    detail["steps"] = [{
        "number": 1,
        "type": "manual",
        "description": "大火快速翻炒至牛肉变色。",
        "duration_seconds": 180,
        "parameters": [],
    }]
    detail["tips"] = "牛肉逆纹切片，变色后尽快出锅。"

    text = format_recipe_detail(
        detail,
        lang="zh",
        seasonings=["蒜 20克", "盐 5克", "生抽 10克"],
    )

    headings = [
        "## 🍳 贡菜炒牛肉",
        "### 菜谱描述",
        "![贡菜炒牛肉",
        "### 食材",
        "### 调料",
        "### 烹饪步骤",
        "### 烹饪小技巧",
    ]
    assert [text.index(value) for value in headings] == sorted(
        text.index(value) for value in headings
    )
    assert "我把【贡菜炒牛肉】现有的用料和操作顺序整理好了" in text
    assert text.endswith(
        "照着上面已保存的步骤依次做就好，操作时也留意菜谱里这条小技巧。"
    )
    food_section = text.split("### 食材", 1)[1].split("### 调料", 1)[0]
    seasoning_section = text.split("### 调料", 1)[1].split("### 烹饪步骤", 1)[0]
    assert "贡菜" in food_section
    assert "牛肉" in food_section
    assert "盐" not in food_section
    assert "蒜" in seasoning_section
    assert "盐" in seasoning_section
    assert "生抽" in seasoning_section


def test_detail_wrapper_does_not_invent_missing_steps_tips_or_ingredients():
    detail = {
        "schema_version": "recipe_detail_v1",
        "recipe_id": "minimal-1",
        "name": "清蒸豆腐",
        "introduction": None,
        "tips": None,
        "media": {},
        "ingredients": [],
        "steps": [],
    }
    text = format_recipe_detail(detail, lang="zh")
    assert "下面是【清蒸豆腐】当前能够核验的信息" in text
    assert "暂无已保存的食材" in text
    assert "暂无已保存的烹饪步骤" in text
    assert "当前没有保存的步骤或小技巧" in text
    assert all(
        invented not in text
        for invented in ("葱", "姜", "蒜", "生抽", "蒸10分钟", "低脂", "养胃")
    )


def test_milvus_candidate_prefers_raw_ingredients_with_quantities():
    detail = build_recipe_detail_seed({
        "cookId": "recipe-2",
        "name": "牛肉炒芹菜",
        # 召回字段只有名称，不能用于覆盖详情用量。
        "ingredients": ["牛肉", "芹菜", "盐"],
        "ingredients_raw": ["牛肉 250克", "芹菜 150克", "盐 3克"],
    }, "zh")

    assert [
        (item["name"], item["amount"], item["unit"], item["quantity_source"])
        for item in detail["ingredients"]
    ] == [
        ("牛肉", 250, "克", "ingredients_raw"),
        ("芹菜", 150, "克", "ingredients_raw"),
        ("盐", 3, "克", "ingredients_raw"),
    ]


def test_grounded_ingredient_parser_keeps_source_quantities():
    samples = {
        "400 - 600 g chicken wings (approx. 8)": (
            "chicken wings (approx. 8)", "400-600", "g"
        ),
        "1/2 tsp ground white pepper": (
            "ground white pepper", 0.5, "tsp"
        ),
        "水 200.0克": ("水", 200, "克"),
        "15克糖": ("糖", 15, "克"),
        "2 spring onions, cut into lengths": (
            "spring onions, cut into lengths", 2, "pcs"
        ),
    }

    for source, expected in samples.items():
        item, unresolved = parse_grounded_ingredient(source)
        assert unresolved is None
        assert (item["name"], item["amount"], item["unit"]) == expected


def test_quantity_only_completion_keeps_names_and_scope(monkeypatch):
    class Reply:
        content = json.dumps({
            "ingredient_quantities": [
                {"name": "盐适量", "amount": 3, "unit": "g"},
                {"name": "Avocado, peeled and sliced", "amount": 200, "unit": "g"},
            ],
            # 调用方必须忽略模型越界返回的详情字段。
            "steps": [{"description": "must be ignored"}],
            "servings": 99,
        }, ensure_ascii=False)

    class FakeLLM:
        async def ainvoke(self, messages):
            payload = json.loads(messages[-1].content)
            assert payload["ingredients_needing_quantities"] == [
                "盐适量",
                "Avocado, peeled and sliced",
            ]
            return Reply()

    monkeypatch.setattr(completion_module, "_recipe_completion_llm", FakeLLM())
    result = asyncio.run(complete_ingredient_quantities(
        recipe_id="recipe-3",
        recipe_name="测试菜谱",
        ingredient_names=["盐适量", "Avocado, peeled and sliced"],
        lang="zh",
    ))

    assert result == [
        {
            "group": "main",
            "name": "盐适量",
            "amount": 3,
            "unit": "g",
            "remark": None,
            "quantity_source": "llm_fallback",
            "fallback_for": "盐适量",
        },
        {
            "group": "main",
            "name": "Avocado, peeled and sliced",
            "amount": 200,
            "unit": "g",
            "remark": None,
            "quantity_source": "llm_fallback",
            "fallback_for": "Avocado, peeled and sliced",
        },
    ]


def test_quantity_only_completion_allows_grounded_composite_split(monkeypatch):
    source = "Cherry tomatoes, lettuce, onion slices, cucumbers slices."

    class Reply:
        content = json.dumps({
            "ingredient_quantities": [
                {"name": "Cherry tomatoes", "amount": 100, "unit": "g"},
                {"name": "lettuce", "amount": 50, "unit": "g"},
                {"name": "onion slices", "amount": 30, "unit": "g"},
                {"name": "cucumbers slices", "amount": 50, "unit": "g"},
            ],
        })

    class FakeLLM:
        async def ainvoke(self, messages):
            return Reply()

    monkeypatch.setattr(completion_module, "_recipe_completion_llm", FakeLLM())
    result = asyncio.run(complete_ingredient_quantities(
        recipe_id="recipe-4",
        recipe_name="Potato pancakes",
        ingredient_names=[source],
        lang="en",
    ))

    assert [item["name"] for item in result] == [
        "Cherry tomatoes", "lettuce", "onion slices", "cucumbers slices",
    ]
    assert {item["fallback_for"] for item in result} == {source}


def test_fetch_recipe_details_normalizes_and_saves_remote_payload(monkeypatch):
    saved = []
    payload = {"code": 200, "data": _remote_recipe(), "msg": "操作 成功"}

    class Process:
        returncode = 0

        async def communicate(self):
            return json.dumps(payload).encode(), b""

    async def fake_subprocess(*args, **kwargs):
        return Process()

    async def fake_save(detail, raw):
        saved.append((detail, raw))

    monkeypatch.setattr(
        cook_module.asyncio,
        "create_subprocess_exec",
        fake_subprocess,
    )
    monkeypatch.setattr(detail_store_module, "save_recipe_detail", fake_save)

    result = asyncio.run(cook_module.fetch_recipe_details(
        "2509088617308622850",
        lang="en",
    ))

    assert result["ok"] is True
    assert result["source"] == "remote"
    assert result["recipe"]["steps"][1]["description"] == "Saute"
    assert len(saved) == 1
    assert saved[0][0]["recipe_id"] == "2509088617308622850"
    assert "isCollect" not in saved[0][1]


def test_detail_followup_renders_saved_schema_without_device_cta(monkeypatch):
    detail = normalize_recipe_detail(_remote_recipe(), language="en")

    async def fake_fetch(*args, **kwargs):
        assert kwargs["prefer_cache"] is True
        return {"ok": True, "code": "OK", "recipe": detail}

    monkeypatch.setattr(cook_module, "fetch_recipe_details", fake_fetch)
    text = asyncio.run(agent_module._recipe_detail_response(
        {"cookId": detail["recipe_id"], "name": detail["name"]},
        "en",
        pending_start=True,
    ))

    assert "### Recipe description" in text
    assert "### Ingredients" in text
    assert "### Seasonings" in text
    assert "### Cooking steps" in text
    assert "### Cooking tips" in text
    assert "start confirmation" not in text
    assert "cook this one" not in text
