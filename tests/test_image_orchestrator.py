"""图片识别 orchestrator 的纯逻辑测试（不碰网络/VL/Milvus）。"""

import time

from app.orchestrator import image as I


def test_ingredient_names_support_new_and_old_shapes():
    rec = {
        "ingredients": [
            {"name": "鸡蛋", "confidence": 0.9},
            {"name": "鸡蛋", "confidence": 0.8},
            "西红柿",
            {"ingredient": "青椒"},
            {"name": ""},
        ]
    }
    assert I._ingredient_names(rec) == ["鸡蛋", "西红柿", "青椒"]


def test_build_query_for_fridge_ingredients():
    rec = {
        "is_food": True,
        "scene_type": "ingredients",
        "dish_name": "",
        "ingredients": [
            {"name": "鸡蛋"},
            {"name": "西红柿"},
            {"name": "青椒"},
        ],
    }
    assert I._build_image_search_query(rec, lang="zh") == "鸡蛋 西红柿 青椒 家常菜"


def test_build_query_prefers_dish_name_for_prepared_dish():
    rec = {
        "is_food": True,
        "scene_type": "dish",
        "dish_name": "番茄炒蛋",
        "ingredients": [{"name": "鸡蛋"}, {"name": "番茄"}],
    }
    assert I._build_image_search_query(rec, lang="zh") == "番茄炒蛋"


def test_build_query_uses_model_search_query_and_short_hint():
    rec = {
        "is_food": True,
        "scene_type": "ingredients",
        "search_query": "鸡胸肉 西兰花 家常菜",
        "ingredients": [{"name": "鸡胸肉"}, {"name": "西兰花"}],
    }
    assert I._build_image_search_query(rec, lang="zh", user_hint="低脂一点") == "鸡胸肉 西兰花 家常菜 低脂一点"


def test_build_query_drops_generic_hint():
    rec = {
        "is_food": True,
        "scene_type": "ingredients",
        "ingredients": [{"name": "土豆"}, {"name": "牛肉"}],
    }
    assert I._build_image_search_query(rec, lang="zh", user_hint="帮我看看") == "土豆 牛肉 家常菜"


def test_build_query_drops_long_generic_caption():
    rec = {
        "is_food": True,
        "scene_type": "ingredients",
        "ingredients": [{"name": "土豆"}, {"name": "牛肉"}],
    }
    assert I._build_image_search_query(rec, lang="zh", user_hint="这些食材能做什么") == "土豆 牛肉 家常菜"


def test_recognized_summary_prefers_ingredients():
    rec = {
        "scene_type": "ingredients",
        "dish_name": "疑似炒菜",
        "ingredients": [{"name": "土豆"}, {"name": "牛肉"}],
    }
    assert I._recognized_food_summary(rec, lang="zh") == "土豆、牛肉"


def test_sequential_image_inventory_appends_and_keeps_caption_context():
    previous = {
        "kind": "image_inventory",
        "ts": time.time(),
        "payload": {
            "scene_type": "ingredients",
            "ingredients": [
                {"name": "鸡蛋", "confidence": 0.91, "state": "raw"},
                {"name": "番茄", "confidence": 0.88, "state": "raw"},
            ],
            "image_count": 1,
            "user_hints": ["清淡一点"],
        },
    }
    recognized = {
        "is_food": True,
        "scene_type": "ingredients",
        "ingredients": [
            {"name": "豆腐", "confidence": 0.94, "state": "raw"},
            {"name": "鸡蛋", "confidence": 0.72, "state": "packaged"},
        ],
        "search_query": "豆腐 鸡蛋 家常菜",
        "confidence": 0.9,
    }

    merged, payload = I.merge_image_inventory(
        recognized,
        previous_action=previous,
        image_count=2,
        user_hint="晚餐",
    )

    assert [item["name"] for item in merged["ingredients"]] == ["鸡蛋", "番茄", "豆腐"]
    assert merged["ingredients"][0]["confidence"] == 0.91
    assert merged["search_query"] == ""
    assert payload["image_count"] == 3
    assert payload["user_hints"] == ["清淡一点", "晚餐"]


def test_image_search_request_and_agent_context_are_source_neutral():
    from app.agent.controlled_deep_agent import _compact_search_request

    recognized = {
        "is_food": True,
        "scene_type": "ingredients",
        "ingredients": [
            {"name": "鸡胸肉", "confidence": 0.93, "state": "packaged"},
            {"name": "西兰花", "confidence": 0.89, "state": "raw"},
        ],
        "search_query": "",
        "confidence": 0.91,
    }

    request = I.build_image_search_request(
        recognized,
        lang="zh",
        user_hint="低脂晚餐",
    )
    context = I.build_image_agent_context(
        recognized,
        image_count=2,
        user_hint="低脂晚餐",
    )
    question = I.image_recommendation_question(context, lang="zh")

    assert request.ingredients == ["鸡胸肉", "西兰花"]
    assert "低脂" in request.scenes
    assert request.retrieval_query(lang="zh").startswith("鸡胸肉 西兰花")
    assert context["scene_type"] == "ingredients"
    assert context["image_count"] == 2
    assert "鸡胸肉" in question
    assert "图片" in question
    assert "冰箱" not in question
    compact = _compact_search_request({
        **request.public_dict(),
        "image_context": context,
    })
    assert compact["image_context"]["scene_type"] == "ingredients"
    assert compact["image_context"]["ingredients"][0]["name"] == "鸡胸肉"
    assert "image_url" not in compact["image_context"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            ok += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    raise SystemExit(0 if ok == len(fns) else 1)
