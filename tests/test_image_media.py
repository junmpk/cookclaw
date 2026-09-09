"""通道图片回复分流测试（不调用视觉模型或 Milvus）。"""

import pytest
from types import SimpleNamespace

from app import main
from app.conversation.models import ConversationTurn
from app.orchestrator import image as image_orchestrator
from app.orchestrator.turn.image_followup import (
    merge_confirmed_ingredients,
    parse_image_ingredient_supplements,
)


def test_all_ingested_recipe_image_hosts_are_allowed_for_qq_upload():
    hosts = {
        "images.example.invalid",
        "images.example.invalid",
        "images.example.invalid",
        "cloudkit-prod.oss-cn-hangzhou.aliyuncs.com",
        "cloudkit-prod.oss-accelerate.aliyuncs.com",
    }
    assert all(main._recipe_image_host_allowed(host) for host in hosts)
    assert not main._recipe_image_host_allowed("127.0.0.1")


def test_recent_conversation_lang_prefers_chinese_context_over_short_ok():
    turns = [
        ConversationTurn(role="user", content="我想找几道适合晚上做的家常菜"),
        ConversationTurn(role="assistant", content="可以，想吃清淡还是香辣？"),
        ConversationTurn(role="user", content="清淡一点"),
        ConversationTurn(role="assistant", content="好的。"),
        ConversationTurn(role="user", content="ok"),
    ]

    assert main._recent_conversation_lang(turns) == "zh"


def test_recent_conversation_lang_keeps_english_context():
    turns = [
        ConversationTurn(role="user", content="I want something quick for dinner"),
        ConversationTurn(role="assistant", content="What ingredients do you have?"),
        ConversationTurn(role="user", content="Something light and easy"),
    ]

    assert main._recent_conversation_lang(turns) == "en"


def test_recent_conversation_lang_ignores_synthetic_image_placeholder():
    turns = [
        ConversationTurn(role="user", content="Show me some easy English recipes"),
        ConversationTurn(role="user", content="[用户发送了1张图片]"),
    ]

    assert main._recent_conversation_lang(turns) == "en"


def test_qq_four_item_menu_splits_before_soup_and_keeps_indices():
    recipes = [
        {
            "id": str(index),
            "name": name,
            "menu_role": role,
            "image": f"https://example.com/{index}.jpg",
            "ingredients": ["chicken"] if role == "dish" else ["water"],
        }
        for index, name, role in (
            (1, "Chicken One", "dish"),
            (2, "Chicken Two", "dish"),
            (3, "Chicken Three", "dish"),
            (4, "Soup Stock", "soup"),
        )
    ]
    messages = main._qq_markdown_messages(
        {
            "type": "menu_plan",
            "lang": "en",
            "data": {
                "lang": "en",
                "recipes": recipes,
                "closing": 'Reply "details 1–4" to see the full recipe.',
            },
        }
    )

    assert len(messages) == 2
    assert messages[0].count("![") == 3
    assert "### 3. Chicken Three" in messages[0]
    assert "## 🍲 Soup" in messages[1]
    assert "### 4. Soup Stock" in messages[1]
    assert messages[1].count("![") == 1
    assert 'Reply "details 1–4"' in messages[1]


def test_recipe_delivery_builds_real_markdown_cards():
    delivery = main._recipe_search_delivery({
        "type": "recipe_search",
        "lang": "zh",
        "data": {
            "opening": "按你的口味挑了两道。",
            "strategy": "想快一点选第一道。",
            "closing": "回复详情编号即可。",
            "recipes": [{
                "id": "r1",
                "name": "香菜拌牛肉",
                "image": "https://images.example.invalid/r1.jpg",
                "ingredients": ["卤牛肉 100克", "香菜 100克"],
                "tags": ["香辣", "凉拌", "牛肉"],
                "recommendation_reason": "三分钟拌好，适合先做。",
                "recipe_detail": {
                    "cooking_time_seconds": 180,
                    "servings": 2,
                    "steps": [{}],
                },
            }],
        },
    })

    assert delivery is not None
    assert delivery["intro"].startswith("## 🍳 推荐菜谱")
    card = delivery["cards"][0]["markdown"]
    assert card.startswith("**1. 香菜拌牛肉**")
    assert "{{QQ_RECIPE_IMAGE}}" in card
    assert "**主要食材**： 卤牛肉 100克、香菜 100克" in card
    assert "**标签**： 香辣 · 凉拌 · 牛肉" in card
    assert "**推荐理由**： 三分钟拌好，适合先做。" in card
    assert "**做饭参考**： 约3分钟 · 2人份 · 1步" in card
    assert "查看详情" not in card
    assert delivery["closing"].count("回复") == 1
    assert delivery["closing"].startswith("## 怎么选")


def test_plaintext_recipe_fallback_keeps_qq_core_fields():
    data = {
        "type": "recipe_search",
        "lang": "zh",
        "data": {
            "recipes": [{
                "id": "r1",
                "name": "番茄炒蛋",
                "ingredients": ["番茄", "鸡蛋"],
                "tags": ["家常"],
                "recommendation_reason": "符合当前条件",
                "recipe_detail": {
                    "cooking_time_seconds": 600,
                    "steps": [{}],
                },
            }],
        },
    }
    text = main._json_to_plaintext(data, encode_filter_urls=False)
    for expected in (
        "番茄炒蛋",
        "番茄、鸡蛋",
        "标签：家常",
        "符合当前条件",
        "做饭参考：约10分钟 · 1步",
        "查看详情：回复「详情 1」",
    ):
        assert expected in text


def test_multi_recipe_delivery_keeps_one_collection_cta():
    data = {
        "type": "recipe_search",
        "lang": "zh",
        "data": {
            "opening": "给你挑了两道。",
            "recipes": [
                {"id": "r1", "name": "番茄炒蛋"},
                {"id": "r2", "name": "蒸水蛋"},
            ],
        },
    }

    delivery = main._recipe_search_delivery(data)
    plaintext = main._json_to_plaintext(data, encode_filter_urls=False)

    assert delivery is not None
    assert all("查看详情" not in card["text"] for card in delivery["cards"])
    assert all("查看详情" not in card["markdown"] for card in delivery["cards"])
    assert delivery["closing"].count("详情 + 序号") == 1
    assert plaintext.count("详情 + 序号") == 1


def test_explicit_ingredient_supplement_replaces_last_visual_item_when_full():
    assert parse_image_ingredient_supplements("Add some tofu", "en") == ["tofu"]
    merged = merge_confirmed_ingredients(
        [
            "eggs", "tomatoes", "bell peppers", "broccoli",
            "mushrooms", "carrots", "lettuce", "apples",
        ],
        ["tofu"],
    )
    assert merged[0] == "tofu"
    assert len(merged) == 8
    assert "apples" not in merged


@pytest.mark.asyncio
async def test_qq_image_wrapper_passes_structured_image_request(monkeypatch):
    captured = {}

    async def fake_handle(media_urls, chat_id, user_text=""):
        captured.update({
            "media_urls": media_urls,
            "chat_id": chat_id,
            "user_text": user_text,
        })
        return {"kind": "recipe_search", "response": {"type": "recipe_search"}}

    monkeypatch.setattr(main, "_handle_image_media", fake_handle)
    event = SimpleNamespace(
        source=SimpleNamespace(
            chat_type="dm",
            chat_id="qq-chat",
            user_id="qq-user",
        ),
        media_urls=["/tmp/one.jpg", "/tmp/two.jpg"],
        text="清淡一点",
    )

    result = await main._handle_image_message(event)

    assert result["kind"] == "recipe_search"
    assert captured["media_urls"] == ["/tmp/one.jpg", "/tmp/two.jpg"]
    assert captured["chat_id"].startswith("qq:")
    assert captured["user_text"] == "清淡一点"


@pytest.mark.asyncio
async def test_qq_recipe_recommendation_is_one_markdown_with_inline_images(monkeypatch):
    class Result:
        success = True

    class Adapter:
        def __init__(self):
            self.images = []
            self.texts = []

        async def send_image(self, chat_id, image_url, caption="", reply_to=None):
            self.images.append((chat_id, image_url, caption, reply_to))
            return Result()

        async def send(self, chat_id, content, reply_to=None):
            self.texts.append((chat_id, content, reply_to))
            return Result()

    async def no_sleep(_seconds):
        return None

    adapter = Adapter()
    monkeypatch.setattr(main, "_qq_adapter", adapter)
    monkeypatch.setattr(main.asyncio, "sleep", no_sleep)
    event = SimpleNamespace(
        source=SimpleNamespace(chat_id="qq-user"),
        message_id="qq-message",
    )
    data = {
        "type": "recipe_search",
        "lang": "en",
        "data": {
            "lang": "en",
            "opening": "Here are three ideas.",
            "closing": 'Reply "details 1", "details 2", or "details 3".',
            "recipes": [
                {
                    "id": str(index),
                    "name": name,
                    "image": f"https://images.example.invalid/{index}.jpg",
                    "ingredients": ["tofu", "tomato"],
                }
                for index, name in enumerate(
                    ["Tofu One", "Tofu Two", "Tofu Three"],
                    1,
                )
            ],
        },
    }

    remembered = await main._send_qq_recipe_search(
        event,
        data,
        lead="Final ingredients: tofu and tomato.",
    )

    assert remembered is not None
    assert adapter.images == []
    assert len(adapter.texts) == 1
    document = adapter.texts[0][1]
    assert document.startswith("## 🍳 Recipe ideas")
    assert "Final ingredients: tofu and tomato." not in document
    assert document.count("Here are three ideas.") == 1
    assert document.count("![") == 3
    assert document.index("**1. Tofu One**") < document.index("![Tofu One")
    assert document.index("![Tofu One") < document.index("**Main ingredients**:")
    assert "**2. Tofu Two**" in document
    assert "**3. Tofu Three**" in document
    assert "{{QQ_RECIPE_IMAGE}}" not in document
    assert "## How to choose" in document
    assert 'Reply "details 1", "details 2", or "details 3".' in document


@pytest.mark.asyncio
async def test_visual_service_error_is_not_reported_as_non_food(monkeypatch):
    async def fake_recognize(images, lang="zh"):
        return {
            "is_food": False,
            "scene_type": "not_food",
            "ingredients": [],
            "search_query": "",
            "error": "APIConnectionError",
        }

    monkeypatch.setattr(
        image_orchestrator,
        "recognize_ingredients_by_images",
        fake_recognize,
    )

    text = await main._handle_image_media(
        ["https://example.com/fridge.jpg"],
        chat_id="qq:test:image-error",
        user_text="please identify these ingredients",
    )

    assert "visual recognition took a brief kitchen break" in text
    assert "staying out of the pan" not in text


@pytest.mark.asyncio
async def test_genuine_non_food_keeps_humorous_fallback(monkeypatch):
    async def fake_recognize(images, lang="zh"):
        return {
            "is_food": False,
            "scene_type": "not_food",
            "ingredients": [],
            "search_query": "",
        }

    monkeypatch.setattr(
        image_orchestrator,
        "recognize_ingredients_by_images",
        fake_recognize,
    )

    text = await main._handle_image_media(
        ["https://example.com/not-food.jpg"],
        chat_id="qq:test:not-food",
        user_text="what can I cook",
    )

    assert "staying out of the pan" in text


@pytest.mark.asyncio
async def test_image_success_uses_deep_agent_with_merged_visual_context(monkeypatch):
    from app.agent import fast_path
    from app.orchestrator import recommendation_response
    from app.orchestrator.recommendation_response import RecommendationNarrative
    from app.conversation.task_state_workspace import clear_thread, get_pending_action

    recognitions = iter([
        {
            "is_food": True,
            "scene_type": "ingredients",
            "dish_name": "",
            "dish_names": [],
            "ingredients": [
                {"name": "鸡蛋", "confidence": 0.92, "state": "raw"},
                {"name": "番茄", "confidence": 0.88, "state": "raw"},
            ],
            "search_query": "鸡蛋 番茄 家常菜",
            "confidence": 0.91,
        },
        {
            "is_food": True,
            "scene_type": "ingredients",
            "dish_name": "",
            "dish_names": [],
            "ingredients": [
                {"name": "豆腐", "confidence": 0.95, "state": "raw"},
            ],
            "search_query": "豆腐 家常菜",
            "confidence": 0.95,
        },
    ])
    narrative_calls = []

    async def fake_recognize(images, lang="zh"):
        assert images
        return next(recognitions)

    async def fake_search(query, top_k=3, lang=None):
        assert top_k == 9
        return {
            "success": True,
            "results": [
                {
                    "id": f"recipe-{index}",
                    "score": 0.9 - index * 0.01,
                    "metadata": {
                        "recipe_id": f"recipe-{index}",
                        "name": name,
                        "ingredients": ingredients,
                        "tags": ["饮品"] if name == "番茄汁" else ["家常"],
                    },
                }
                for index, (name, ingredients) in enumerate(
                    [
                        ("番茄炒蛋", ["番茄", "鸡蛋"]),
                        ("家常豆腐", ["豆腐"]),
                        ("番茄豆腐", ["番茄", "豆腐"]),
                        ("番茄汁", ["番茄"]),
                    ],
                    1,
                )
            ],
        }

    async def fake_narrative(*args, **kwargs):
        narrative_calls.append(kwargs)
        return RecommendationNarrative(
            "我把图片里的食材合在一起看了，这几道更贴近现在这批食材。",
            "可以先按主要食材对照着选。",
            {},
            "想先看哪一道的详情？",
        )

    async def fake_lang(_chat_id, _user_text=""):
        return "zh"

    async def fake_context(_chat_id, _question):
        return (
            [{"role": "user", "content": "今晚想吃得清淡一点"}],
            {"schema_version": "controlled_context_v1"},
        )

    thread_id = "unit-image-agent"
    clear_thread(thread_id)
    monkeypatch.setattr(
        image_orchestrator,
        "recognize_ingredients_by_images",
        fake_recognize,
    )
    monkeypatch.setattr(fast_path, "_run_search_subprocess", fake_search)
    monkeypatch.setattr(
        recommendation_response,
        "generate_recommendation_narrative",
        fake_narrative,
    )
    monkeypatch.setattr(main, "_image_response_lang", fake_lang)
    monkeypatch.setattr(main, "_image_agent_memory_context", fake_context)

    try:
        first = await main._handle_image_media(
            ["https://example.com/ingredients-1.jpg"],
            chat_id=thread_id,
            user_text="清淡一点",
        )
        second = await main._handle_image_media(
            ["https://example.com/ingredients-2.jpg"],
            chat_id=thread_id,
        )

        assert first["kind"] == "recipe_search"
        assert second["kind"] == "recipe_search"
        assert len(narrative_calls) == 2
        second_call = narrative_calls[-1]
        assert second_call["deep_agent_enabled"] is True
        assert second_call["recent_turns"][0]["content"] == "今晚想吃得清淡一点"
        assert [
            item["name"]
            for item in second_call["input_context"]["ingredients"]
        ] == ["鸡蛋", "番茄", "豆腐"]
        assert second_call["input_context"]["image_count"] == 2
        assert second_call["input_context"]["user_caption"] == "清淡一点"
        assert second_call["search_request"].ingredients == ["鸡蛋", "番茄", "豆腐"]
        assert "冰箱" not in second_call["original_question"]
        assert all(
            (item.get("metadata") or {}).get("name") != "番茄汁"
            for item in second_call["search_result"]["results"]
        )

        pending = get_pending_action(thread_id)
        assert pending["kind"] == "image_inventory"
        assert [
            item["name"]
            for item in pending["payload"]["ingredients"]
        ] == ["鸡蛋", "番茄", "豆腐"]
    finally:
        clear_thread(thread_id)
