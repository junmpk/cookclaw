"""独立演示链的行为测试。下列数据只用于测试，不作为演示食谱。"""
import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.demo.agents import Agents
from app.demo.engine import Engine
from app.demo.graph import build_graph
from app.demo.im_bridge import IMGraphBridge
from app.demo.models import Brief, DietDecision, Recipe, ResearchDecision
from app.demo.seed import parse_recipe
from app.demo.server import create_app
from app.demo.storage import DemoStorage
from app.demo.validation import validate_menu


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


def test_validator_blocks_fabrication_duplicates_and_exclusions():
    brief = Brief(request="test", dishes=2, soups=1, exclusions=["鸡蛋"])
    issues = validate_menu(brief, fixtures(), ["fixture-b", "fixture-b", "invented"])
    assert any("之外" in issue for issue in issues)
    assert any("重复" in issue for issue in issues)
    assert any("排除" in issue for issue in issues)
    assert any("汤" in issue for issue in issues)


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
        async def research(self, *args):
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
        assert client.post(f"/api/runs/{run_id}/retry").status_code == 202
        assert wait_run(client, run_id)["status"] == "awaiting_confirmation"


def test_live_requires_credentials_and_valid_brief(tmp_path, monkeypatch):
    # 显式空值应优先于本机 .env，避免测试意外消耗真实模型额度。
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(create_app(tmp_path, FakeStore())) as client:
        assert client.post("/api/runs", json=initial(mode="live")["brief"]).status_code == 409
        assert client.post("/api/runs", json={"request": "", "dishes": 0}).status_code == 422
        assert client.get("/api/runs/unknown").status_code == 404
        assert client.get("/").status_code == 200
        assert client.get("/assets/app.js").status_code == 200


def test_chat_preserves_session_and_uses_shared_web_entry(tmp_path, monkeypatch):
    from app.agent import participle_agent
    from app.orchestrator.turn.runtime_models import ResponseEnvelope
    seen = []
    async def handle(message, thread_id):
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


def test_chat_graph_bridge_reuses_thread_and_confirmation(tmp_path, monkeypatch):
    from app.agent import participle_agent
    from app.demo import server
    from app.orchestrator.turn.runtime_models import ResponseEnvelope

    async def handle(_message, _thread_id):
        return ResponseEnvelope(message="已解析当前菜单需求", handled_by="shared")

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.setattr(participle_agent, "handle_web_turn", handle)
    monkeypatch.setattr(
        server,
        "brief_from_task_state",
        lambda _state, mode="live": Brief(
            request="test fixture workflow",
            dishes=1,
            soups=1,
            mode=mode,
        ),
    )
    with TestClient(create_app(tmp_path, FakeStore())) as client:
        response = client.post(
            "/api/chat",
            json={"message": "请让三位专家协作配餐", "mode": "rehearsal"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "langgraph_bridge"
        assert payload["graph_status"] == "running"
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
    from app.demo import server
    from app.orchestrator.turn.runtime_models import ResponseEnvelope

    class ReplacementStore(FakeStore):
        async def search(self, query, limit=12):
            return replacement_fixtures()

    async def handle(_message, _thread_id):
        return ResponseEnvelope(message="已解析当前菜单需求", handled_by="shared")

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.setattr(participle_agent, "handle_web_turn", handle)
    monkeypatch.setattr(
        server,
        "brief_from_task_state",
        lambda _state, mode="live": Brief(
            request="test fixture workflow",
            dishes=2,
            soups=1,
            mode=mode,
        ),
    )
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
        "请让三位专家协作配餐",
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
