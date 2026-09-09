"""视觉识别结构归一化测试（不调用外部模型）。"""

from app.agent.vision import _normalize_recognition


def test_ingredients_override_contradictory_is_food_false():
    rec = _normalize_recognition(
        {
            "is_food": False,
            "scene_type": "ingredients",
            "dish_name": "",
            "ingredients": [
                {"name": "eggs", "confidence": 0.95, "state": "raw"},
                {"name": "tomatoes", "confidence": 0.9, "state": "raw"},
            ],
            "search_query": "eggs tomatoes recipe",
            "confidence": 0.98,
        },
        lang="en",
    )

    assert rec["is_food"] is True
    assert rec["scene_type"] == "ingredients"
    assert [item["name"] for item in rec["ingredients"]] == ["eggs", "tomatoes"]


def test_food_evidence_repairs_not_food_scene():
    rec = _normalize_recognition(
        {
            "is_food": False,
            "scene_type": "not_food",
            "ingredients": [{"name": "西红柿", "confidence": 0.9}],
        },
        lang="zh",
    )

    assert rec["is_food"] is True
    assert rec["scene_type"] == "ingredients"
    assert rec["search_query"] == "西红柿 家常菜"


def test_genuine_non_food_remains_non_food():
    rec = _normalize_recognition(
        {
            "is_food": False,
            "scene_type": "not_food",
            "dish_name": "",
            "ingredients": [],
            "confidence": 0.95,
        },
        lang="en",
    )

    assert rec["is_food"] is False
    assert rec["scene_type"] == "not_food"
    assert rec["search_query"] == ""


def test_multi_image_normalization_keeps_up_to_twelve_ingredients():
    rec = _normalize_recognition(
        {
            "is_food": True,
            "scene_type": "ingredients",
            "ingredients": [
                {"name": f"食材{index}", "confidence": 0.9}
                for index in range(1, 15)
            ],
        },
        lang="zh",
    )

    assert len(rec["ingredients"]) == 12
    assert rec["ingredients"][-1]["name"] == "食材12"
