"""阶段 4.4：图片派生事实、Planner 门禁和统一 Envelope。"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.conversation.models import ConversationMemory
from app.conversation.service import ConversationService, build_qq_thread_id
from app.conversation.store import InMemoryConversationStore
from app.observability.trace import (
    ensure_turn_trace,
    finish_turn_trace,
    record_tool_call,
)
from app.orchestrator.planning.models import PlanStep, TurnPlan
from app.orchestrator.planning.plan_validator import PlanValidator
from app.orchestrator.recommendation_response import RecommendationNarrative
from app.orchestrator.routing_models import RoutingContext
from app.orchestrator.turn import image_handler
from app.orchestrator.turn.application_service import TurnApplicationService
from app.orchestrator.turn.context_loader import build_turn_context
from app.orchestrator.turn.execution_journal import TurnExecutionJournal
from app.orchestrator.turn.runtime_models import DerivedTurnContext, TurnRequest
from app.ports.dialogue_state import clear_thread


class _NoopTaskStateRepository:
    def __init__(self):
        self.scope_entries = 0

    @asynccontextmanager
    async def turn_scope(self, _thread_id):
        self.scope_entries += 1
        yield None

    async def refresh_current_scope(self):
        return None

    async def flush_current_scope(self):
        return None


def _derived() -> DerivedTurnContext:
    return DerivedTurnContext(
        scene_type="ingredients",
        media_count=2,
        ingredients=["番茄", "鸡蛋"],
        dishes=[],
        recognition_uncertain=True,
    )


def _image_request(thread_id: str, utterance: str = "") -> TurnRequest:
    return TurnRequest(
        utterance=utterance,
        thread_id=thread_id,
        channel="qq",
        message_type="image",
    )


def _recipe_plan(action: str = "recipe.recommend") -> TurnPlan:
    return TurnPlan(
        goal="根据图片里的食材推荐真实菜谱",
        phase="search_or_recommend",
        known_facts=["视觉工具识别到食材线索"],
        steps=[
            PlanStep(
                action=action,
                args={},
                evidence_refs=["derived_facts"],
            )
        ],
        reply_act="recommend",
        risk="low",
        reason_code="IMAGE_RECIPE_RECOMMENDATION",
        confidence=0.8,
    )


def test_derived_context_is_visible_to_planner_but_redacted_from_refs():
    derived = _derived()
    context = build_turn_context(
        "",
        RoutingContext(channel="qq", message_type="image"),
        trace_id="trace-phase44",
        derived_context=derived,
    )
    request = TurnRequest(
        utterance="",
        thread_id="qq:secret-user",
        channel="qq",
        message_type="image",
        derived_context=derived,
    )

    payload = context.planner_payload()
    assert payload["utterance"] == ""
    assert payload["derived_context"]["ingredients"] == ["番茄", "鸡蛋"]
    assert "番茄" not in json.dumps(context.context_ref(), ensure_ascii=False)
    assert "鸡蛋" not in json.dumps(request.context_ref(), ensure_ascii=False)
    assert "secret-user" not in json.dumps(request.context_ref(), ensure_ascii=False)
    assert "番茄" not in repr(context)
    assert context.context_ref()["derived_context"]["ingredient_count"] == 2


def test_validator_accepts_sourced_derived_facts():
    context = build_turn_context(
        "",
        RoutingContext(channel="qq", message_type="image"),
        derived_context=_derived(),
    )

    validation = PlanValidator().validate(_recipe_plan(), context)

    assert validation.valid is True
    assert validation.execution_allowed is True


async def _lang(_thread_id: str, _utterance: str) -> str:
    return "zh"


async def _memory(_thread_id: str, _question: str):
    return [], {"schema_version": "controlled_context_v1"}


def _converter(media: str):
    return {"image_url": media}


def _recognition():
    return {
        "is_food": True,
        "scene_type": "ingredients",
        "ingredients": [
            {"name": "番茄", "confidence": 0.91, "state": "raw"},
            {"name": "鸡蛋", "confidence": 0.88, "state": "raw"},
        ],
        "dish_name": "",
        "dish_names": [],
        "search_query": "番茄 鸡蛋 家常菜",
        "image_count": 1,
    }


def _search_result():
    return {
        "success": True,
        "results": [
            {
                "id": "recipe-1",
                "score": 0.95,
                "metadata": {
                    "recipe_id": "recipe-1",
                    "name": "番茄炒蛋",
                    "ingredients": ["番茄", "鸡蛋"],
                    "tags": ["家常"],
                },
            }
        ],
    }


async def _narrative(*_args, **_kwargs):
    return RecommendationNarrative(
        "我按图片里识别到的食材找了真实菜谱。",
        "这道最贴近现有食材。",
        {},
        "想看详情就告诉我。",
    )


@pytest.mark.asyncio
async def test_image_recipe_filter_inherits_saved_allergens(monkeypatch):
    import app.conversation.service as service_module

    user_id = "image-allergen-user"
    thread_id = build_qq_thread_id("dm", user_id, user_id)
    short_store = InMemoryConversationStore()
    profile_store = InMemoryConversationStore()
    await profile_store.save(
        ConversationMemory(
            thread_id=f"profile:qq:{user_id}",
            channel="qq",
            user_id=user_id,
            preferences={
                "likes": [],
                "dislikes": [],
                "allergens": ["花生"],
                "dietary_constraints": [],
                "available_ingredients": [],
            },
            version=3,
        ),
        expires_at=time.time() + 3600,
    )
    service = ConversationService(
        short_store,
        profile_store=profile_store,
    )
    monkeypatch.setattr(service_module, "_service", service)

    async def fake_recognize(_images, lang="zh"):
        return {
            **_recognition(),
            "ingredients": [
                {"name": "花生", "confidence": 0.9, "state": "raw"},
                {"name": "番茄", "confidence": 0.9, "state": "raw"},
            ],
            "search_query": "花生 番茄",
        }

    async def fake_search(*_args, **_kwargs):
        return {
            "success": True,
            "results": [
                {
                    "id": "unsafe",
                    "score": 0.99,
                    "metadata": {
                        "recipe_id": "unsafe",
                        "name": "花生拌菜",
                        "ingredients": ["花生", "青菜"],
                    },
                },
                {
                    "id": "safe",
                    "score": 0.8,
                    "metadata": {
                        "recipe_id": "safe",
                        "name": "番茄炒蛋",
                        "ingredients": ["番茄", "鸡蛋"],
                    },
                },
            ],
        }

    monkeypatch.setattr(
        image_handler,
        "planner_rollout_decision",
        lambda *_args, **_kwargs: SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(
        image_handler.image_orchestrator,
        "recognize_ingredients_by_images",
        fake_recognize,
    )
    monkeypatch.setattr(
        image_handler.agent_fast_path,
        "_run_search_subprocess",
        fake_search,
    )
    monkeypatch.setattr(
        image_handler.recommendation_response,
        "generate_recommendation_narrative",
        _narrative,
    )
    clear_thread(thread_id)
    try:
        result = await image_handler.handle_image_turn(
            ["https://example.com/ingredients.jpg"],
            request=_image_request(thread_id),
            image_input_converter=_converter,
            language_resolver=_lang,
            memory_loader=_memory,
        )

        assert result.envelope.response_type == "recipe_search"
        recipes = result.envelope.data.get("recipes") or []
        assert [item["id"] for item in recipes] == ["safe"]
        assert all("花生" not in item.get("name", "") for item in recipes)
    finally:
        clear_thread(thread_id)


@pytest.mark.asyncio
async def test_image_recipe_search_fails_closed_when_profile_is_unreadable(
    monkeypatch,
):
    import app.conversation.service as service_module

    class UnavailableProfileStore(InMemoryConversationStore):
        async def load(self, thread_id: str):
            raise TimeoutError(f"profile unavailable: {thread_id}")

    user_id = "image-profile-down"
    thread_id = build_qq_thread_id("dm", user_id, user_id)
    service = ConversationService(
        InMemoryConversationStore(),
        profile_store=UnavailableProfileStore(),
    )
    monkeypatch.setattr(service_module, "_service", service)
    searches = 0

    async def fake_recognize(_images, lang="zh"):
        return _recognition()

    async def forbidden_search(*_args, **_kwargs):
        nonlocal searches
        searches += 1
        return _search_result()

    monkeypatch.setattr(
        image_handler.image_orchestrator,
        "recognize_ingredients_by_images",
        fake_recognize,
    )
    monkeypatch.setattr(
        image_handler.agent_fast_path,
        "_run_search_subprocess",
        forbidden_search,
    )
    clear_thread(thread_id)
    try:
        result = await image_handler.handle_image_turn(
            ["https://example.com/ingredients.jpg"],
            request=_image_request(thread_id),
            image_input_converter=_converter,
            language_resolver=_lang,
            memory_loader=_memory,
        )

        assert result.envelope.response_type == "text"
        assert "读不到" in result.envelope.message
        assert "过敏" in result.envelope.message
        assert searches == 0
    finally:
        clear_thread(thread_id)


@pytest.mark.asyncio
async def test_image_uses_same_planner_and_executes_grounded_search_once(monkeypatch):
    thread_id = "qq:phase44:planner-user"
    clear_thread(thread_id)
    searches = 0
    captured_context = None
    journal_starts = 0
    original_journal_start = TurnExecutionJournal.start.__func__

    def counting_journal_start(cls, thread_id, *, trace_event_start=None):
        nonlocal journal_starts
        journal_starts += 1
        return original_journal_start(
            cls,
            thread_id,
            trace_event_start=trace_event_start,
        )

    monkeypatch.setattr(
        TurnExecutionJournal,
        "start",
        classmethod(counting_journal_start),
    )

    async def fake_recognize(_images, lang="zh"):
        assert lang == "zh"
        record_tool_call(
            "vision_recognition",
            duration_ms=2,
            success=True,
        )
        return _recognition()

    async def fake_search(_query, top_k=3, lang=None):
        nonlocal searches
        searches += 1
        assert top_k == 9
        assert lang == "zh"
        record_tool_call(
            "recipe_search",
            duration_ms=3,
            success=True,
            result_count=1,
        )
        return _search_result()

    async def fake_plan(context):
        nonlocal captured_context
        captured_context = context
        return SimpleNamespace(
            execution_allowed=True,
            plan=_recipe_plan(),
            action="recipe.recommend",
            fallback_code=None,
        )

    monkeypatch.setattr(
        image_handler,
        "planner_rollout_decision",
        lambda *_args, **_kwargs: SimpleNamespace(enabled=True),
    )
    monkeypatch.setattr(image_handler, "evaluate_active_plan", fake_plan)
    monkeypatch.setattr(
        image_handler,
        "planner_active_actions",
        lambda: {"recipe.search", "recipe.recommend"},
    )
    monkeypatch.setattr(
        image_handler.image_orchestrator,
        "recognize_ingredients_by_images",
        fake_recognize,
    )
    monkeypatch.setattr(
        image_handler.agent_fast_path,
        "_run_search_subprocess",
        fake_search,
    )
    monkeypatch.setattr(
        image_handler.recommendation_response,
        "generate_recommendation_narrative",
        _narrative,
    )

    _trace, trace_token = ensure_turn_trace(
        channel="qq",
        thread_id=thread_id,
        message_type="image",
    )
    try:
        request = TurnRequest(
            utterance="清淡一点",
            thread_id=thread_id,
            channel="qq",
            message_type="image",
            trace_id=_trace.trace_id,
        )
        domain_result = None

        async def image_stage(context):
            nonlocal domain_result
            domain_result = await image_handler.handle_image_turn(
                ["https://example.com/ingredients.jpg"],
                request=request,
                image_input_converter=_converter,
                language_resolver=_lang,
                memory_loader=_memory,
                short_term_snapshot=context.short_term_snapshot,
            )
            return domain_result.envelope

        async def runtime_loader(_context):
            raise AssertionError("图片 handler 不应加载文本 runtime")

        repository = _NoopTaskStateRepository()
        envelope = await TurnApplicationService(
            task_state_repository=repository,
        ).handle(
            request,
            runtime_loader=runtime_loader,
            handlers={"image_handler": image_stage},
        )
        assert domain_result is not None
        result = domain_result.with_envelope(envelope)

        assert searches == 1
        assert repository.scope_entries == 1
        assert journal_starts == 1
        assert captured_context is not None
        assert captured_context.message_type == "image"
        assert captured_context.derived_context.ingredients == ["番茄", "鸡蛋"]
        assert result.envelope.response_type == "recipe_search"
        assert result.envelope.handled_by == "image_handler"
        assert [tool.tool for tool in result.envelope.tool_results] == [
            "vision_recognition",
            "recipe_search",
        ]
        assert result.channel_value()["response"]["data"]["recipes"][0]["name"] == "番茄炒蛋"
    finally:
        finish_turn_trace(
            trace_token,
            success=True,
            response_type="image_recipe_search",
        )
        clear_thread(thread_id)


@pytest.mark.asyncio
async def test_image_rejects_device_plan_and_never_executes_device(monkeypatch):
    thread_id = "qq:phase44:no-device"
    clear_thread(thread_id)
    searches = 0

    async def fake_search(*_args, **_kwargs):
        nonlocal searches
        searches += 1
        return _search_result()

    async def fake_plan(_context):
        return SimpleNamespace(
            execution_allowed=True,
            plan=_recipe_plan("device.prepare"),
            action="device.prepare",
            fallback_code=None,
        )

    async def fake_recognize(_images, lang="zh"):
        return _recognition()

    monkeypatch.setattr(
        image_handler,
        "planner_rollout_decision",
        lambda *_args, **_kwargs: SimpleNamespace(enabled=True),
    )
    monkeypatch.setattr(image_handler, "evaluate_active_plan", fake_plan)
    monkeypatch.setattr(
        image_handler.image_orchestrator,
        "recognize_ingredients_by_images",
        fake_recognize,
    )
    monkeypatch.setattr(
        image_handler.agent_fast_path,
        "_run_search_subprocess",
        fake_search,
    )
    monkeypatch.setattr(
        image_handler.recommendation_response,
        "generate_recommendation_narrative",
        _narrative,
    )

    try:
        result = await image_handler.handle_image_turn(
            ["https://example.com/ingredients.jpg"],
            request=_image_request(thread_id, "立刻开机做"),
            image_input_converter=_converter,
            language_resolver=_lang,
            memory_loader=_memory,
        )

        assert searches == 1
        assert result.envelope.response_type == "recipe_search"
        assert all(
            not tool.tool.startswith("device")
            for tool in result.envelope.tool_results
        )
    finally:
        clear_thread(thread_id)


@pytest.mark.asyncio
async def test_image_planner_execution_failure_does_not_repeat_search(monkeypatch):
    thread_id = "qq:phase44:single-search"
    clear_thread(thread_id)
    searches = 0

    async def fake_recognize(_images, lang="zh"):
        return _recognition()

    async def fake_search(*_args, **_kwargs):
        nonlocal searches
        searches += 1
        return _search_result()

    async def broken_narrative(*_args, **_kwargs):
        raise RuntimeError("synthetic narrative failure")

    async def fake_plan(_context):
        return SimpleNamespace(
            execution_allowed=True,
            plan=_recipe_plan(),
            action="recipe.recommend",
            fallback_code=None,
        )

    monkeypatch.setattr(
        image_handler,
        "planner_rollout_decision",
        lambda *_args, **_kwargs: SimpleNamespace(enabled=True),
    )
    monkeypatch.setattr(image_handler, "evaluate_active_plan", fake_plan)
    monkeypatch.setattr(
        image_handler,
        "planner_active_actions",
        lambda: {"recipe.search", "recipe.recommend"},
    )
    monkeypatch.setattr(
        image_handler.image_orchestrator,
        "recognize_ingredients_by_images",
        fake_recognize,
    )
    monkeypatch.setattr(
        image_handler.agent_fast_path,
        "_run_search_subprocess",
        fake_search,
    )
    monkeypatch.setattr(
        image_handler.recommendation_response,
        "generate_recommendation_narrative",
        broken_narrative,
    )

    try:
        result = await image_handler.handle_image_turn(
            ["https://example.com/ingredients.jpg"],
            request=_image_request(thread_id),
            image_input_converter=_converter,
            language_resolver=_lang,
            memory_loader=_memory,
        )

        assert searches == 1
        assert result.envelope.response_type == "text"
        assert "不会重复检索" in result.envelope.message
    finally:
        clear_thread(thread_id)
