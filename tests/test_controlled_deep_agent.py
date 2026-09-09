"""受控 Deep Agent 的候选事实、上下文隔离和输出校验。"""
import asyncio
import json
from types import SimpleNamespace

from app.agent import controlled_deep_agent as agent_module
from app.observability.trace import ensure_turn_trace, finish_turn_trace


def test_agent_middleware_blocks_builtin_tools_and_allows_candidate_read():
    called = []

    async def handler(request):
        called.append(request.tool_call["name"])
        return "allowed"

    blocked = asyncio.run(
        agent_module._allow_candidate_tool_only.awrap_tool_call(
            SimpleNamespace(tool_call={"name": "execute", "id": "blocked-1"}),
            handler,
        )
    )
    assert blocked.status == "error"
    assert called == []

    reads_token = agent_module._tool_reads.set({})
    allowed_token = agent_module._allowed_tools.set(
        frozenset({"get_current_recipe_candidates"})
    )
    try:
        allowed = asyncio.run(
            agent_module._allow_candidate_tool_only.awrap_tool_call(
                SimpleNamespace(
                    tool_call={
                        "name": "get_current_recipe_candidates",
                        "id": "allowed-1",
                    },
                ),
                handler,
            )
        )
    finally:
        agent_module._allowed_tools.reset(allowed_token)
        agent_module._tool_reads.reset(reads_token)
    assert allowed == "allowed"
    assert called == ["get_current_recipe_candidates"]


def test_compact_candidate_facts_exposes_only_grounded_fields():
    facts = agent_module.compact_candidate_facts([{
        "cookId": "r1",
        "name": "清蒸鸡肉",
        "tags": ["清淡", "家常"],
        "ingredients": ["鸡肉", "姜"],
        "seasonings": ["盐"],
        "description": "真实描述",
        "recipe_detail": {"steps": ["步骤一", "步骤二"]},
        "internal_token": "must-not-leak",
        "device_id": "must-not-leak",
    }])

    assert facts == [{
        "position": 1,
        "recipe_id": "r1",
        "name": "清蒸鸡肉",
        "tags": ["清淡", "家常"],
        "ingredients": ["鸡肉", "姜"],
        "seasonings": ["盐"],
        "description": "真实描述",
        "step_count": 2,
    }]


def test_compact_conversation_state_keeps_task_facts_and_hides_device_ids():
    compact = agent_module.compact_conversation_state({
        "current_task": "device_start",
        "active_search_request": {
            "ingredients": ["鸡肉"],
            "exclude": ["辣"],
            "memory_note": {"secret": "hidden"},
        },
        "selected_recipe": {"cookId": "r2", "name": "清蒸鸡肉"},
        "excluded_recipe_ids": ["r1"],
        "pending_action": {
            "kind": "confirm_device_start",
            "payload": {"token": "hidden"},
        },
        "pending_device_start": {
            "cookId": "r2",
            "name": "清蒸鸡肉",
            "device_id": "secret-device-id",
            "action_id": "secret-action-id",
        },
        "language": "zh",
    }, recent_conversation="用户：不要辣")

    assert compact["active_search_request"] == {
        "ingredients": ["鸡肉"],
        "exclude": ["辣"],
    }
    assert compact["selected_recipe"]["recipe_id"] == "r2"
    assert compact["excluded_recipe_ids"] == ["r1"]
    assert compact["pending_action"] == {"kind": "confirm_device_start"}
    assert compact["pending_device_start"] == {
        "waiting_confirmation": True,
        "recipe_id": "r2",
        "recipe_name": "清蒸鸡肉",
        "device_choice_required": False,
        "device_already_selected": True,
    }
    assert "secret-device-id" not in json.dumps(compact, ensure_ascii=False)
    assert "secret-action-id" not in json.dumps(compact, ensure_ascii=False)
    assert compact["recent_conversation"] == "用户：不要辣"


def test_candidate_agent_context_is_isolated_and_output_is_validated(monkeypatch):
    observed = {}

    class FakeAgent:
        async def ainvoke(self, inputs, config):
            observed["tool"] = agent_module.get_current_recipe_candidates.invoke({})
            observed["state"] = agent_module.get_current_conversation_state.invoke({})
            observed["payload"] = json.loads(inputs["messages"][0].content)
            observed["config"] = config
            return {
                "structured_response": {
                    "selected_recipe_id": "r2",
                    "reason": "facts support it",
                },
            }

    monkeypatch.setattr(agent_module, "_get_candidate_agent", lambda _model: FakeAgent())

    async def run():
        selected = await agent_module.choose_candidate_with_deep_agent(
            model=object(),
            question="帮我选一道清淡的",
            conversation_context="用户上一轮说不要太油",
            recipes=[
                {"cookId": "r1", "name": "香辣鸡丁", "tags": ["香辣"]},
                {"cookId": "r2", "name": "清蒸鸡肉", "tags": ["清淡"]},
            ],
            lang="zh",
            conversation_state={
                "current_task": "recipe_selection",
                "active_search_request": {"ingredients": ["鸡肉"]},
                "excluded_recipe_ids": ["r1"],
            },
        )
        assert selected == "r2"

    asyncio.run(run())
    assert observed["tool"]["candidates"][1]["recipe_id"] == "r2"
    assert observed["state"]["data"]["current_task"] == "recipe_selection"
    assert observed["state"]["data"]["excluded_recipe_ids"] == ["r1"]
    assert "recent_conversation" not in observed["payload"]
    assert observed["config"]["recursion_limit"] == 8
    assert agent_module.get_current_recipe_candidates.invoke({}) == {"candidates": []}
    assert agent_module.get_current_conversation_state.invoke({}) == {
        "ok": True,
        "data": {},
    }


def test_candidate_agent_rejects_recipe_id_outside_current_candidates(monkeypatch):
    class FakeAgent:
        async def ainvoke(self, _inputs, config):
            return {
                "structured_response": {
                    "selected_recipe_id": "invented-id",
                    "reason": "invalid",
                },
            }

    monkeypatch.setattr(agent_module, "_get_candidate_agent", lambda _model: FakeAgent())

    async def run():
        selected = await agent_module.choose_candidate_with_deep_agent(
            model=object(),
            question="帮我选一个",
            conversation_context="",
            recipes=[{"cookId": "r1", "name": "真实菜"}],
            lang="zh",
        )
        assert selected is None

    asyncio.run(run())


def test_candidate_agent_marks_actual_deep_agent_participation(monkeypatch):
    class FakeAgent:
        async def ainvoke(self, _inputs, config):
            return {
                "structured_response": {
                    "selected_recipe_id": "r1",
                    "reason": "verified",
                },
            }

    monkeypatch.setattr(agent_module, "_get_candidate_agent", lambda _model: FakeAgent())
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")

    async def run():
        _trace, token = ensure_turn_trace(
            channel="qq",
            thread_id="qq:dm:trace:trace",
            deep_agent_enabled=True,
            deep_agent_cohort="qq_allowlist",
        )
        selected = await agent_module.choose_candidate_with_deep_agent(
            model=object(),
            question="选一个",
            conversation_context="",
            recipes=[{"cookId": "r1", "name": "真实菜"}],
            lang="zh",
        )
        payload = finish_turn_trace(token, success=True)
        return selected, payload

    selected, payload = asyncio.run(run())
    assert selected == "r1"
    assert payload["deep_agent_enabled"] is True
    assert payload["deep_agent_used"] is True
    assert payload["deep_agent_cohort"] == "qq_allowlist"


def test_verified_recipe_detail_is_limited_to_current_results():
    candidate_token = agent_module._candidate_context.set((
        {
            "recipe_id": "r1",
            "name": "清蒸鸡肉",
            "ingredients": ["鸡肉"],
            "step_count": 3,
        },
    ))
    try:
        found = agent_module.get_verified_recipe_detail.invoke({"recipe_id": "r1"})
        missing = agent_module.get_verified_recipe_detail.invoke({"recipe_id": "r2"})
    finally:
        agent_module._candidate_context.reset(candidate_token)

    assert found["ok"] is True
    assert found["data"]["recipe"]["step_count"] == 3
    assert missing["ok"] is False
    assert missing["error"]["code"] == "RECIPE_NOT_IN_CURRENT_RESULTS"


def test_recommendation_agent_returns_only_current_recipe_reason_ids(monkeypatch):
    observed = {}

    class FakeAgent:
        async def ainvoke(self, inputs, config):
            observed["state"] = agent_module.get_current_conversation_state.invoke({})
            observed["results"] = agent_module.get_current_recipe_search_results.invoke({})
            observed["config"] = config
            return {
                "structured_response": {
                    "opening": "这轮按鸡肉来选。",
                    "strategy": "先比较这两道。",
                    "recipe_reasons": {
                        "r1": "有鸡肉。",
                        "invented": "不存在。",
                    },
                    "closing": "想看哪一道？",
                },
            }

    monkeypatch.setattr(
        agent_module,
        "_get_recommendation_agent",
        lambda _model: FakeAgent(),
    )

    async def run():
        result = await agent_module.compose_recommendation_with_deep_agent(
            model=object(),
            current_question="推荐鸡肉菜",
            recent_user_context=["不要辣"],
            current_request={"ingredients": ["鸡肉"], "exclude": ["辣"]},
            recipes=[
                {"id": "r1", "name": "清蒸鸡肉", "ingredients": ["鸡肉"]},
                {"id": "r2", "name": "鸡肉炖菜", "ingredients": ["鸡肉"]},
            ],
            lang="zh",
            conversation_state={"current_task": "recipe_search"},
        )
        assert result["recipe_reasons"] == {"r1": "有鸡肉。"}

    asyncio.run(run())
    assert observed["state"]["data"]["active_search_request"] == {
        "ingredients": ["鸡肉"],
        "exclude": ["辣"],
    }
    assert len(observed["results"]["data"]["recipes"]) == 2
    assert observed["config"]["recursion_limit"] == 10
