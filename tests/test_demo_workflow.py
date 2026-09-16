"""独立演示链的行为测试。下列数据只用于测试，不作为演示食谱。"""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.demo.agents import Agents
from app.demo.chat_intent import graph_routing_decision, might_need_graph
from app.demo.details import hydrate_schedule_details
from app.demo.engine import Engine
from app.demo.graph import build_graph
from app.demo.im_bridge import IMGraphBridge
from app.demo.models import Brief, DietDecision, Recipe, ResearchDecision
from app.demo.retrieval import recipe_from_search_hit
from app.demo.seed import parse_recipe
from app.demo.server import create_app
from app.demo.storage import DemoStorage
from app.demo.validation import validate_menu, validate_plan


def fixtures():
    return [Recipe(id="fixture-a", name="TEST FIXTURE A", source="https://example.org/a",
                   ingredients=["test vegetable"], kind="dish"),
            Recipe(id="fixture-b", name="TEST FIXTURE B", source="https://example.org/b",
                   ingredients=["test egg"], kind="dish"),
            Recipe(id="fixture-c", name="TEST FIXTURE C", source="https://example.org/c",
            ingredients=["test broth"], kind="soup")]


def replacement_fixtures():
    return [*fixtures(), Recipe(id="fixture-d", name="TEST FIXTURE D", source="https://example.org/d",
                                ingredients=["test potato"], kind="dish")]


def grounded_search_result(recipes):
    return {
        "results": [
            {
                "id": recipe.id,
                "metadata": {
                    "name": recipe.name,
                    "image_url": recipe.image_url,
                    "ingredients": recipe.ingredients,
                    "tags": [recipe.kind],
                    "recipe_detail": {
                        "source": recipe.source,
                        "ingredients": recipe.ingredients,
                        "steps": recipe.steps,
                    },
                },
            }
            for recipe in recipes
        ]
    }


async def handoff_fixture(
    callback,
    *,
    message="test fixture workflow",
    dishes=1,
    soups=1,
    recipes=None,
):
    request = {
        "original_text": message,
        "party_size": 2,
        "menu_dish_count": dishes,
        "menu_soup_count": soups,
        "exclude": [],
    }
    return await callback(
        None,
        SimpleNamespace(
            kind="menu_plan",
            search_request=request,
            search_result=grounded_search_result(recipes or fixtures()),
        ),
    )


class FakeStore:
    async def search(self, query, limit=12):
        return fixtures()

    def close(self):
        pass


async def silent(*args):
    pass


def initial(**overrides):
    brief = Brief(request="test fixture workflow", dishes=1, soups=1, **overrides)
    return {"brief": brief.model_dump(), "version": 1, "research_count": 0, "revision_count": 0}


def test_graph_router_uses_structured_complexity_instead_of_magic_phrase():
    complex_brief = Brief(
        request="两个人吃，一菜一汤，不要花生",
        people=2,
        dishes=1,
        soups=1,
        exclusions=["花生"],
    )
    decision = graph_routing_decision(
        "两个人吃，想做一菜一汤，不要花生",
        complex_brief,
    )

    assert decision.use_graph is True
    assert decision.trigger == "complexity"
    assert "需要同时组合菜和汤" in decision.reasons
    assert "存在必须排除的食材" in decision.reasons
    assert might_need_graph("给三口之家安排三菜一汤，偏清淡") is True

    simple = graph_routing_decision(
        "鸡胸肉怎么做才不柴",
        Brief(request="鸡胸肉怎么做才不柴", people=1, dishes=1, soups=0),
    )
    assert simple.use_graph is False
    assert simple.trigger == "simple"


def test_complexity_profile_selects_inventory_and_scheduler_by_dimension():
    brief = Brief(
        request="四个人三菜一汤，使用现有鸡肉，45分钟内用灶台和空气炸锅做好",
        people=4,
        dishes=3,
        soups=1,
        exclusions=["花生"],
        available_ingredients=["鸡肉"],
        budget_yuan=50,
        max_minutes=45,
        equipment=["灶台", "空气炸锅"],
    )
    decision = graph_routing_decision(brief.request, brief)

    assert decision.use_graph is True
    assert decision.profile is not None
    assert decision.profile.level == "advanced"
    assert decision.profile.dimensions.inventory == 3
    assert decision.profile.dimensions.scheduling == 4
    assert decision.profile.selected_agents == [
        "research",
        "dietary",
        "inventory",
        "menu",
        "scheduler",
    ]
    assert "需要核对现有食材与采购缺口" in decision.profile.reasons


@pytest.mark.asyncio
async def test_dynamic_graph_runs_optional_inventory_and_scheduler(tmp_path):
    brief = Brief(
        request="两个人一菜一汤，用现有食材，30分钟内做好",
        people=2,
        dishes=1,
        soups=1,
        available_ingredients=["test vegetable"],
        max_minutes=60,
        equipment=["灶台"],
        mode="rehearsal",
    )
    profile = graph_routing_decision(brief.request, brief).profile
    graph = build_graph(
        Agents(FakeStore(), silent, "rehearsal"),
        InMemorySaver(),
        DemoStorage(tmp_path / "dynamic.sqlite"),
        "dynamic",
        silent,
        profile,
    )
    result = await graph.ainvoke(
        {
            "brief": brief.model_dump(),
            "complexity_profile": profile.model_dump(),
            "version": 1,
            "research_count": 0,
            "revision_count": 0,
        },
        {"configurable": {"thread_id": "dynamic"}},
    )

    assert result["__interrupt__"]
    assert result["inventory"]["candidate_coverage"]
    assert len(result["schedule"]["tasks"]) == 2
    assert all(
        task["evidence_basis"] == "demo_template"
        for task in result["schedule"]["tasks"]
    )
    assert result["plan_validation"]["status"] == "needs_confirmation"
    assert result["complexity_profile"]["selected_agents"][-1] == "scheduler"


def test_scheduler_is_optional_without_time_or_equipment_constraint():
    brief = Brief(
        request="给三个人安排三菜一汤",
        people=3,
        dishes=3,
        soups=1,
    )
    profile = graph_routing_decision(brief.request, brief).profile

    assert profile is not None
    assert "scheduler" not in profile.selected_agents
    assert profile.dimensions.scheduling == 1


def test_virtual_steps_are_explicit_and_plan_validation_keeps_boundary(
    monkeypatch,
):
    monkeypatch.setenv("DEMO_VIRTUAL_SCHEDULE", "true")
    brief = Brief(
        request="一菜一汤，60分钟内用灶台做好",
        dishes=1,
        soups=1,
        max_minutes=60,
        equipment=["灶台"],
    )
    hydrated, detail = hydrate_schedule_details(
        brief,
        fixtures(),
        ["fixture-a", "fixture-c"],
    )
    selected = [recipe for recipe in hydrated if recipe.id in {"fixture-a", "fixture-c"}]
    schedule = Agents(
        FakeStore(), silent, "rehearsal"
    )._schedule_evidence(brief, selected).model_dump()
    report = validate_plan(
        brief,
        hydrated,
        {"recipe_ids": ["fixture-a", "fixture-c"]},
        schedule=schedule,
    )

    assert detail["demo_template_count"] == 2
    assert all(recipe.detail_basis == "demo_template" for recipe in selected)
    assert all("演示模板" in recipe.steps[0] for recipe in selected)
    assert report.status == "needs_confirmation"
    assert any("虚拟步骤" in warning for warning in report.warnings)


def test_validator_blocks_fabrication_duplicates_and_exclusions():
    brief = Brief(request="test", dishes=2, soups=1, exclusions=["鸡蛋"])
    issues = validate_menu(brief, fixtures(), ["fixture-b", "fixture-b", "invented"])
    assert any("之外" in issue for issue in issues)
    assert any("重复" in issue for issue in issues)
    assert any("排除" in issue for issue in issues)
    assert any("汤" in issue for issue in issues)


def test_hybrid_hit_is_hydrated_without_model_invented_identity():
    recipe = recipe_from_search_hit({
        "id": "grounded-1",
        "metadata": {
            "name": "番茄鸡蛋汤",
            "image_url": "https://example.org/recipe.jpg",
            "ingredients": ["番茄", "鸡蛋"],
            "tags": ["家常汤"],
            "recipe_detail": {
                "source": "https://example.org/grounded-1",
                "nutrition": {"protein": "10g"},
            },
        },
    })
    assert recipe.id == "grounded-1"
    assert recipe.name == "番茄鸡蛋汤"
    assert recipe.kind == "soup"
    assert recipe.ingredients == ["番茄", "鸡蛋"]


@pytest.mark.asyncio
async def test_graph_interrupt_then_idempotent_mock(tmp_path):
    storage = DemoStorage(tmp_path / "runs.sqlite")
    graph = build_graph(Agents(FakeStore(), silent, "rehearsal"), InMemorySaver(), storage, "run", silent)
    config = {"configurable": {"thread_id": "test"}}
    result = await graph.ainvoke(initial(), config=config)
    assert result["__interrupt__"]
    assert "execution" not in result
    assert result["revision_count"] == 1
    result = await graph.ainvoke(Command(resume={"approved": True, "version": 1}), config=config)
    assert result["status"] == "completed"
    assert result["execution"] == storage.execute_mock("run:v1", ["fixture-a", "fixture-c"])
    with storage.connect() as db:
        assert db.execute("SELECT count(*) FROM executions").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_shortage_refetches_then_stops(tmp_path):
    storage = DemoStorage(tmp_path / "runs.sqlite")
    graph = build_graph(Agents(FakeStore(), silent, "rehearsal"), InMemorySaver(), storage, "run", silent)
    state = initial()
    state["brief"]["dishes"] = 5
    result = await graph.ainvoke(state, {"configurable": {"thread_id": "shortage"}})
    assert result["status"] == "blocked"
    assert result["research_count"] == 2
    assert result["revision_count"] == 3
    assert "execution" not in result


@pytest.mark.asyncio
async def test_parallel_research_and_diet(tmp_path):
    research_started, diet_started = asyncio.Event(), asyncio.Event()
    class ParallelAgents(Agents):
        async def research(self, *args, **kwargs):
            research_started.set()
            await asyncio.wait_for(diet_started.wait(), 1)
            return [r.model_dump() for r in fixtures()]

        async def diet(self, *args):
            diet_started.set()
            await asyncio.wait_for(research_started.wait(), 1)
            return {"advice": [], "unknowns": [], "rule_ids": []}
    graph = build_graph(ParallelAgents(FakeStore(), silent, "rehearsal"), InMemorySaver(),
                        DemoStorage(tmp_path / "db"), "parallel", silent)
    result = await graph.ainvoke(initial(), {"configurable": {"thread_id": "parallel"}})
    assert result["__interrupt__"]


def wait_run(client, run_id):
    for _ in range(200):
        row = client.get(f"/api/runs/{run_id}").json()
        if row["status"] != "running":
            return row
        time.sleep(.02)
    raise AssertionError("run did not finish")


def test_restart_resume_and_duplicate_confirmation(tmp_path):
    with TestClient(create_app(tmp_path, FakeStore())) as client:
        run_id = client.post("/api/runs", json=initial()["brief"]).json()["id"]
        assert wait_run(client, run_id)["status"] == "awaiting_confirmation"
    with TestClient(create_app(tmp_path, FakeStore())) as client:
        assert client.get(f"/api/runs/{run_id}").json()["status"] == "awaiting_confirmation"
        assert client.post(f"/api/runs/{run_id}/decision", json={"version": 1, "approved": True}).status_code == 202
        row = wait_run(client, run_id)
        assert row["status"] == "completed"
        again = client.post(f"/api/runs/{run_id}/decision", json={"version": 1, "approved": True})
        assert again.status_code == 202
        assert again.json()["state"]["execution"] == row["state"]["execution"]


def test_revision_invalidates_old_approval_and_cancel(tmp_path):
    with TestClient(create_app(tmp_path, FakeStore())) as client:
        run_id = client.post("/api/runs", json=initial()["brief"]).json()["id"]
        wait_run(client, run_id)
        updated = initial(exclusions=["鸡蛋"])["brief"]
        assert client.post(f"/api/runs/{run_id}/revise", json={"version": 1, "brief": updated}).status_code == 202
        assert wait_run(client, run_id)["version"] == 2
        assert client.post(f"/api/runs/{run_id}/decision", json={"version": 1, "approved": True}).status_code == 409
        assert client.post(f"/api/runs/{run_id}/decision", json={"version": 2, "approved": False}).status_code == 202
        row = wait_run(client, run_id)
        assert row["status"] == "cancelled"
        assert "execution" not in row["state"]


def test_tool_failure_safe_error_retry_and_sse(tmp_path):
    class FlakyStore(FakeStore):
        def __init__(self):
            self.fail = True

        async def search(self, *args):
            if self.fail:
                self.fail = False
                raise RuntimeError("secret-provider-message-must-not-leak")
            return fixtures()
    with TestClient(create_app(tmp_path, FlakyStore())) as client:
        run_id = client.post("/api/runs", json=initial()["brief"]).json()["id"]
        assert wait_run(client, run_id)["status"] == "error"
        events = client.get(f"/api/runs/{run_id}/events").text
        assert "secret-provider" not in events
        assert "event: done" in events
        retried = client.post(f"/api/runs/{run_id}/retry")
        assert retried.status_code == 202
        assert retried.json()["attempt"] == 2
        recovered = wait_run(client, run_id)
        assert recovered["status"] == "awaiting_confirmation"
        assert recovered["metrics"]["attempt_count"] == 2
        assert recovered["metrics"]["recovered"] is True
        assert recovered["metrics"]["error_count"] == 1
        recovery_events = [
            item
            for item in client.get(f"/api/runs/{run_id}/events").text.splitlines()
            if "recovery_start" in item
        ]
        assert recovery_events


def test_interrupted_before_first_checkpoint_rebuilds_initial_input(tmp_path):
    storage = DemoStorage(tmp_path / "runs.sqlite")
    storage.create(
        "interrupted-run",
        initial()["brief"],
        thread_id="web:" + "c" * 32,
    )
    storage.update("interrupted-run", "running", initial())

    with TestClient(create_app(tmp_path, FakeStore())) as client:
        interrupted = client.get("/api/runs/interrupted-run").json()
        assert interrupted["status"] == "interrupted"
        retried = client.post("/api/runs/interrupted-run/retry")
        assert retried.status_code == 202
        recovered = wait_run(client, "interrupted-run")
        assert recovered["status"] == "awaiting_confirmation"
        assert recovered["metrics"]["attempt_count"] == 2
        events = client.get("/api/runs/interrupted-run/events").text
        assert "recovery_fallback" in events


def test_live_requires_credentials_and_valid_brief(tmp_path, monkeypatch):
    # 显式空值应优先于本机 .env，避免测试意外消耗真实模型额度。
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(create_app(tmp_path, FakeStore())) as client:
        assert client.post("/api/runs", json=initial(mode="live")["brief"]).status_code == 409
        assert client.post("/api/runs", json={"request": "", "dishes": 0}).status_code == 422
        assert client.get("/api/runs/unknown").status_code == 404
        page = client.get("/")
        assert page.status_code == 200
        assert "动态多 Agent" in page.text
        assert "最终烹饪排期" in page.text
        assert client.get("/assets/app.js").status_code == 200


def test_chat_preserves_session_and_uses_shared_web_entry(tmp_path, monkeypatch):
    from app.agent import participle_agent
    from app.orchestrator.turn.runtime_models import ResponseEnvelope
    seen = []
    async def handle(message, thread_id, **_kwargs):
        seen.append((message, thread_id))
        return ResponseEnvelope(message="shared reply", handled_by="test_shared")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.setattr(participle_agent, "handle_web_turn", handle)
    with TestClient(create_app(tmp_path, FakeStore())) as client:
        response = client.post("/api/chat", json={"message": "鸡胸肉怎么做才不柴？"})
        assert response.status_code == 200
        thread = response.json()["thread_id"]
        second = client.post("/api/chat", json={"message": "两个人", "thread_id": thread})
        assert second.json()["reply"] == "shared reply"
        assert seen[0][1] == seen[1][1] == thread
        assert client.post("/api/chat", json={"message":"你好", "thread_id":"qq:someone"}).status_code == 422
        assert client.post("/api/chat", json={"message": ""}).status_code == 422
        assert client.post("/api/chat", json={"message": "  "}).status_code == 422


def test_web_image_upload_keeps_thread_and_returns_confirmation(tmp_path, monkeypatch):
    from app import main

    async def handle_image(media_urls, chat_id, user_text=""):
        assert media_urls[0].startswith("data:image/png;base64,")
        assert user_text == "看看冰箱"
        return {
            "kind": "image_confirmation",
            "response": {
                "type": "image_ingredients_confirmation",
                "intent": "image_ingredients",
                "lang": "zh",
                "data": {
                    "ingredients": ["番茄", "鸡蛋"],
                    "requires_confirmation": True,
                },
                "message": "请确认识别到的食材。",
            },
        }

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.setattr(main, "_handle_image_media", handle_image)
    image = "data:image/png;base64," + "iVBORw0KGgo="
    with TestClient(create_app(tmp_path, FakeStore())) as client:
        response = client.post(
            "/api/chat/image",
            json={"data_url": image, "caption": "看看冰箱"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["thread_id"].startswith("web:")
        assert payload["response"]["response_type"] == "image_ingredients_confirmation"
        assert payload["response"]["data"]["ingredients"] == ["番茄", "鸡蛋"]

        invalid = client.post(
            "/api/chat/image",
            json={"data_url": "data:text/plain;base64,aGVsbG8="},
        )
        assert invalid.status_code == 422


def test_run_storage_persists_canonical_thread_relation(tmp_path):
    storage = DemoStorage(tmp_path / "runs.sqlite")
    storage.create("run-a", initial()["brief"], thread_id="web:" + "a" * 32)
    row = storage.get("run-a")
    assert row["thread_id"] == "web:" + "a" * 32
    assert storage.chat_run_id(row["thread_id"]) == "run-a"
    assert row["checkpoint_thread_id"] == (
        row["thread_id"] + ":workflow:run-a:v1"
    )


def test_thread_binding_reuses_active_run_atomically(tmp_path):
    storage = DemoStorage(tmp_path / "runs.sqlite")
    thread_id = "web:" + "b" * 32
    first, created = storage.create_for_thread(
        "run-first",
        initial()["brief"],
        thread_id=thread_id,
    )
    second, second_created = storage.create_for_thread(
        "run-second",
        initial()["brief"],
        thread_id=thread_id,
    )

    assert (first, created) == ("run-first", True)
    assert (second, second_created) == ("run-first", False)
    assert storage.get("run-second") is None
    assert storage.chat_run_id(thread_id) == "run-first"


@pytest.mark.asyncio
async def test_im_bridge_uses_same_configurable_recipe_backend(
    tmp_path,
    monkeypatch,
):
    from app.demo import im_bridge

    store = FakeStore()
    store.backend_name = "hybrid_rag"
    selected = {}

    def build(path, backend):
        selected.update({"path": path, "backend": backend})
        return store

    monkeypatch.setenv("DEMO_RECIPE_BACKEND", "hybrid")
    monkeypatch.setattr(im_bridge, "build_recipe_store", build)
    bridge = await IMGraphBridge.create(tmp_path)
    try:
        assert bridge.recipe_backend == "hybrid_rag"
        assert selected["backend"] == "hybrid"
        assert selected["path"] == tmp_path / "recipes.db"
    finally:
        await bridge.close()


def test_chat_graph_bridge_reuses_thread_and_confirmation(tmp_path, monkeypatch):
    from app.agent import participle_agent
    from app.orchestrator.turn.runtime_models import ResponseEnvelope

    async def handle(_message, _thread_id, **kwargs):
        callback = kwargs.get("workflow_handoff")
        if callback is not None:
            result = await handoff_fixture(
                callback,
                message="两个人吃，想做一菜一汤，不要花生",
            )
            if result is not None:
                return result
        return ResponseEnvelope(message="已解析当前菜单需求", handled_by="shared")

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.setattr(participle_agent, "handle_web_turn", handle)
    with TestClient(create_app(tmp_path, FakeStore())) as client:
        response = client.post(
            "/api/chat",
            json={"message": "两个人吃，想做一菜一汤，不要花生", "mode": "rehearsal"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "langgraph_bridge"
        assert payload["graph_status"] == "running"
        assert payload["response"]["data"]["routing"]["trigger"] == "complexity"
        thread = payload["thread_id"]
        run_id = payload["graph_run_id"]
        assert wait_run(client, run_id)["status"] == "awaiting_confirmation"

        confirmation = client.post(
            "/api/chat",
            json={"message": "确认", "thread_id": thread, "mode": "rehearsal"},
        )
        assert confirmation.status_code == 200
        assert confirmation.json()["graph_run_id"] == run_id
        assert wait_run(client, run_id)["status"] == "completed"


def test_chat_graph_bridge_replaces_one_menu_slot(tmp_path, monkeypatch):
    from app.agent import participle_agent
    from app.orchestrator.turn.runtime_models import ResponseEnvelope

    class ReplacementStore(FakeStore):
        async def search(self, query, limit=12):
            return replacement_fixtures()

    async def handle(_message, _thread_id, **kwargs):
        callback = kwargs.get("workflow_handoff")
        if callback is not None:
            result = await handoff_fixture(
                callback,
                message="请让三位专家协作配餐",
                dishes=2,
                soups=1,
                recipes=replacement_fixtures(),
            )
            if result is not None:
                return result
        return ResponseEnvelope(message="已解析当前菜单需求", handled_by="shared")

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.setattr(participle_agent, "handle_web_turn", handle)
    with TestClient(create_app(tmp_path, ReplacementStore())) as client:
        response = client.post(
            "/api/chat",
            json={"message": "请让三位专家协作配餐", "mode": "rehearsal"},
        )
        thread = response.json()["thread_id"]
        old_run = response.json()["graph_run_id"]
        old_row = wait_run(client, old_run)
        assert old_row["status"] == "awaiting_confirmation"
        old_ids = old_row["state"]["menu"]["recipe_ids"]

        revised = client.post(
            "/api/chat",
            json={"message": "第二道换掉", "thread_id": thread, "mode": "rehearsal"},
        )
        assert revised.status_code == 200
        new_run = revised.json()["graph_run_id"]
        assert new_run != old_run
        new_row = wait_run(client, new_run)
        assert new_row["status"] == "awaiting_confirmation"
        new_ids = new_row["state"]["menu"]["recipe_ids"]
        assert new_ids[0] == old_ids[0]
        assert new_ids[1] != old_ids[1]
        assert new_ids[2] == old_ids[2]
        assert client.post(
            f"/api/runs/{old_run}/decision",
            json={"version": 1, "approved": True},
        ).status_code == 409
        assert client.post(
            f"/api/runs/{new_run}/decision",
            json={"version": 1, "approved": True},
        ).status_code == 202
        assert wait_run(client, new_run)["status"] == "completed"


def test_repeated_hydration_preserves_demo_evidence(monkeypatch):
    monkeypatch.setenv("DEMO_VIRTUAL_SCHEDULE", "true")
    brief = Brief(request="test", dishes=1, soups=0)
    first, _ = hydrate_schedule_details(brief, fixtures(), ["fixture-a"])
    second, summary = hydrate_schedule_details(brief, first, ["fixture-a"])
    assert second[0].detail_basis == "demo_template"
    assert second[0].steps == first[0].steps
    assert summary["demo_template_count"] == 1
    assert summary["source_count"] == 0


@pytest.mark.parametrize("thread_id", ["qq:group:g1:u1", "whatsapp:dm:u1:u1"])
@pytest.mark.asyncio
async def test_im_channels_share_graph_and_natural_slot_adjustment(
    tmp_path,
    monkeypatch,
    thread_id,
):
    from app.orchestrator.turn.runtime_models import ResponseEnvelope

    class ReplacementStore(FakeStore):
        async def search(self, query, limit=12):
            return replacement_fixtures()

    class Connection:
        async def close(self):
            pass

    class TaskState:
        def __init__(self):
            self.menu_task = {
                "request": {
                    "original_text": "两个人，两菜一汤，不要花生",
                    "party_size": 2,
                    "menu_dish_count": 2,
                    "menu_soup_count": 1,
                    "exclude": ["花生"],
                }
            }
            self.active_search_request = None

    store = ReplacementStore()
    bridge = IMGraphBridge(
        Engine(
            DemoStorage(tmp_path / "runs.sqlite"),
            InMemorySaver(),
            store,
        ),
        Connection(),
        store,
    )
    monkeypatch.setenv("MULTI_AGENT_BRIDGE_MODE", "rehearsal")

    async def fallback():
        return ResponseEnvelope(message="共享 Runtime 已解析", handled_by="shared")

    async def load_state(_thread_id):
        return TaskState()

    first = await bridge.handle(
        "两个人吃，安排两菜一汤，不要花生",
        thread_id,
        fallback=fallback,
        state_loader=load_state,
    )
    assert first.response_type == "menu_plan"
    old_ids = [item["id"] for item in first.data["recipes"]]

    revised = await bridge.handle(
        "第二道换清淡一点",
        thread_id,
        fallback=fallback,
        state_loader=load_state,
    )
    new_ids = [item["id"] for item in revised.data["recipes"]]
    assert new_ids[0] == old_ids[0]
    assert new_ids[1] != old_ids[1]
    assert new_ids[2] == old_ids[2]
    assert revised.handled_by == "langgraph_bridge"

    confirmed = await bridge.handle(
        "确认",
        thread_id,
        fallback=fallback,
        state_loader=load_state,
    )
    assert confirmed.response_type == "workflow"
    assert "未连接或控制真实设备" in confirmed.message
    await bridge.close()


@pytest.mark.asyncio
async def test_im_workflow_handoff_reuses_grounded_candidates(tmp_path, monkeypatch):
    class NoSecondSearchStore(FakeStore):
        async def search(self, query, limit=12):
            raise AssertionError("初始 Graph 不应重复执行食谱检索")

    class Connection:
        async def close(self):
            pass

    store = NoSecondSearchStore()
    bridge = IMGraphBridge(
        Engine(
            DemoStorage(tmp_path / "handoff.sqlite"),
            InMemorySaver(),
            store,
        ),
        Connection(),
        store,
    )
    monkeypatch.setenv("MULTI_AGENT_BRIDGE_MODE", "rehearsal")
    outcome = SimpleNamespace(
        kind="menu_plan",
        search_request={
            "original_text": "两个人一菜一汤",
            "party_size": 2,
            "menu_dish_count": 1,
            "menu_soup_count": 1,
        },
        search_result=grounded_search_result(fixtures()),
    )

    response = await bridge.workflow_handoff(
        "两个人一菜一汤",
        "qq:group:g1:u1",
        outcome,
    )

    assert response is not None
    assert response.response_type == "menu_plan"
    assert response.handled_by == "langgraph_bridge"
    assert [item["id"] for item in response.data["recipes"]] == [
        "fixture-a",
        "fixture-c",
    ]
    await bridge.close()


def test_slot_adjustment_parser_supports_natural_phrases():
    from app.demo.chat_intent import (
        requests_unspecified_adjustment,
        slot_adjustment_from_chat,
    )

    specified = slot_adjustment_from_chat("把第二道换成西兰花烤土豆")
    lighter = slot_adjustment_from_chat("第一道清淡一点")
    assert specified and specified.slot_index == 1
    assert lighter and lighter.slot_index == 0
    assert requests_unspecified_adjustment("这道菜我不想要了，帮我换一下")


@pytest.mark.asyncio
async def test_shared_qqbot_entry_uses_bound_graph_bridge():
    from app.agent import participle_agent
    from app.orchestrator.turn.runtime_models import ResponseEnvelope

    class BoundBridge:
        async def handle(self, question, thread_id, *, fallback, state_loader):
            assert question == "请让三位专家协作配餐"
            assert thread_id == "qq:group:g1:u1"
            return ResponseEnvelope(
                response_type="workflow",
                message="bridge reply",
                handled_by="langgraph_bridge",
            )

    participle_agent.bind_im_graph_bridge(BoundBridge())
    try:
        raw = await participle_agent.qqbot_chat(
            "请让三位专家协作配餐",
            "qq:group:g1:u1",
        )
    finally:
        participle_agent.bind_im_graph_bridge(None)
    payload = json.loads(raw)
    assert payload["type"] == "workflow"
    assert payload["message"] == "bridge reply"


def test_seed_requires_source_and_license():
    data = {"@type": "Recipe", "name": "test fixture", "recipeIngredient": ["test"],
            "isBasedOn": "https://example.org", "image": ["https://example.org/test.jpg"],
            "license": "https://creativecommons.org/publicdomain/mark/1.0/"}
    page = '<script type="application/ld+json">'+json.dumps(data)+'</script>'
    parsed = parse_recipe(page, "fixture", "dish")
    assert parsed.id == "fixture"
    assert parsed.image_url == "https://example.org/test.jpg"
    data.pop("license")
    with pytest.raises(ValueError):
        parse_recipe('<script type="application/ld+json">'+json.dumps(data)+'</script>', "fixture", "dish")


@pytest.mark.asyncio
async def test_research_retains_retrieved_candidates(monkeypatch):
    agents = Agents(FakeStore(), silent, "live")
    async def invoke(role, schema, tools, payload, instruction):
        await tools[0].ainvoke({"query": "test"})
        return ResearchDecision(selected_ids=["fixture-b"])
    monkeypatch.setattr(agents, "invoke", invoke)
    result = await agents.research(Brief(**initial()["brief"]), [], 1)
    assert [r["id"] for r in result] == ["fixture-b", "fixture-a", "fixture-c"]


@pytest.mark.asyncio
async def test_initial_diet_cannot_publish_unretrieved_recipes(monkeypatch):
    agents = Agents(FakeStore(), silent, "live")
    async def invoke(role, schema, tools, payload, instruction):
        await tools[0].ainvoke({})
        return DietDecision(advice=["unretrieved-dish"], unknowns=[], rule_ids=["diversity"])
    monkeypatch.setattr(agents, "invoke", invoke)
    result = await agents.diet(Brief(**initial()["brief"]))
    assert "unretrieved-dish" not in str(result)
    assert result["rule_ids"] == ["diversity"]


@pytest.mark.asyncio
async def test_menu_replace_precedes_candidate_exclusion(monkeypatch):
    from app.agent import participle_agent as core
    monkeypatch.setattr(core, "get_pending_action", lambda _: {"kind":"select_recipe", "lang":"zh"})
    monkeypatch.setattr(core, "get_menu_task", lambda _: {"request": {}})
    monkeypatch.setattr(core, "resolve_selection", lambda *a: None)
    async def replace(thread, message, **kwargs):
        assert message == "第二道换掉"
        return "replaced"
    monkeypatch.setattr(core, "_replace_pending_menu_slot", replace)
    assert await core._handle_pending_action("web:test", "第二道换掉") == "replaced"


@pytest.mark.asyncio
async def test_public_adapter_preserves_source_and_menu_types():
    from app.demo.shared_chat import bind_recipe_store, search_public_recipes
    from app.orchestrator.menu_plan import _is_verified_dish, _is_verified_soup
    bind_recipe_store(FakeStore())
    result = await search_public_recipes("test", 3, "en")
    rows = result["results"]
    assert _is_verified_soup(rows[2])
    assert not _is_verified_dish(rows[2])
    assert rows[0]["metadata"]["recipe_detail"]["source"] == "https://example.org/a"
    assert "score" not in rows[0]  # Do not invent a similarity score.
