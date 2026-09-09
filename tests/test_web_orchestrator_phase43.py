"""阶段 4.3 Web 接入统一 Turn facade 的回归。"""

from __future__ import annotations

import asyncio
import gc

from app.orchestrator.turn.memory_adapter import MemoryHandlerResult
from app.orchestrator.turn.response_renderer import WebResponseRenderer
from app.orchestrator.turn.runtime_models import ResponseEnvelope
from app.orchestrator.intent import IntentCategory, IntentResult
from app.orchestrator.router import FastPathOutcome


def _recipe_envelope() -> ResponseEnvelope:
    return ResponseEnvelope(
        response_type="recipe_search",
        intent="recommend",
        lang="zh",
        message="按你现有的鸡腿和土豆，先看这道。",
        data={
            "recipes": [
                {
                    "id": "private-recipe-id",
                    "name": "土豆炖鸡",
                    "image": "https://example.com/chicken.jpg",
                    "ingredients": ["鸡腿", "土豆"],
                    "tags": ["炖", "家常"],
                    "recommendation_reason": "同时用到本轮两个主料。",
                }
            ],
            "closing": "想看步骤就回复“详情 1”。",
        },
        handled_by="recipe_handler",
        trace_id="private-trace",
    )


def test_web_renderer_builds_markdown_from_envelope_only():
    markdown = WebResponseRenderer().render(_recipe_envelope())

    # 新格式：不再有 "## 推荐菜谱" 僵硬标题
    assert "## 推荐菜谱" not in markdown
    # 菜名用数字 emoji + 加粗，不再是 ### 标题
    assert "1️⃣ **土豆炖鸡**" in markdown
    assert "![土豆炖鸡](https://example.com/chicken.jpg)" in markdown
    assert "鸡腿、土豆" in markdown
    # 推荐理由直接作为引用块，不再有 "**推荐理由**：" 标签
    assert "> 同时用到本轮两个主料" in markdown
    assert "详情 1" in markdown
    assert "private-recipe-id" not in markdown
    assert "private-trace" not in markdown
    assert "handled_by" not in markdown


def test_web_renderer_ignores_non_http_image_and_uses_plain_message():
    unsafe = _recipe_envelope().model_copy(
        update={
            "data": {
                "recipes": [
                    {
                        "name": "测试菜",
                        "image": "javascript:alert(1)",
                    }
                ]
            }
        }
    )
    plain = ResponseEnvelope(message="普通回答", handled_by="planner")

    markdown = WebResponseRenderer().render(unsafe)

    assert "javascript:" not in markdown
    assert WebResponseRenderer().render(plain) == "普通回答"


def test_web_uses_unified_core_and_keeps_async_chunk_contract(monkeypatch):
    import app.agent.participle_agent as agent_module

    captured = {}

    async def fake_core(question, *, thread_id, channel, **_kwargs):
        captured.update(
            question=question,
            thread_id=thread_id,
            channel=channel,
        )
        return _recipe_envelope()

    monkeypatch.setattr(agent_module, "_run_turn_orchestrator", fake_core)

    async def collect():
        return [
            chunk
            async for chunk in agent_module.chat_stream(
                "用鸡腿和土豆推荐一道菜",
                "web:phase43",
            )
        ]

    chunks = asyncio.run(collect())

    assert len(chunks) == 1
    assert "土豆炖鸡" in chunks[0]
    assert captured == {
        "question": "用鸡腿和土豆推荐一道菜",
        "thread_id": "web:phase43",
        "channel": "web",
    }


def test_web_turn_persists_user_before_core_and_assistant_after_reply(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation.service import ConversationService
    from app.conversation.store import InMemoryConversationStore

    service = ConversationService(
        InMemoryConversationStore(),
        profile_store=InMemoryConversationStore(),
    )
    monkeypatch.setattr(service_module, "_service", service)
    observed_before_core = []

    async def fake_core(question, *, thread_id, channel, **_kwargs):
        snapshot = await service.load_runtime_memory(thread_id, channel=channel)
        observed_before_core.extend(
            (turn.role, turn.content)
            for turn in snapshot.short_term.recent_turns
        )
        return ResponseEnvelope(
            response_type="chat",
            intent="chat",
            lang="zh",
            message="装饰器可以在不改原函数主体的情况下扩展行为。",
            handled_by="conversation_fallback",
        )

    monkeypatch.setattr(agent_module, "_run_turn_orchestrator", fake_core)

    async def run():
        chunks = [
            chunk
            async for chunk in agent_module.chat_stream(
                "Python 装饰器是什么？",
                "web:memory-lifecycle",
            )
        ]
        snapshot = await service.load_runtime_memory("web:memory-lifecycle")
        return chunks, snapshot

    chunks, snapshot = asyncio.run(run())

    assert observed_before_core == [("user", "Python 装饰器是什么？")]
    assert chunks == ["装饰器可以在不改原函数主体的情况下扩展行为。"]
    assert [
        (turn.role, turn.content)
        for turn in snapshot.short_term.recent_turns
    ] == [
        ("user", "Python 装饰器是什么？"),
        ("assistant", "装饰器可以在不改原函数主体的情况下扩展行为。"),
    ]


def test_same_web_session_serializes_complete_turns(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation.service import ConversationService
    from app.conversation.store import InMemoryConversationStore

    service = ConversationService(InMemoryConversationStore())
    monkeypatch.setattr(service_module, "_service", service)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    entered: list[str] = []

    async def fake_core(question, **_kwargs):
        entered.append(question)
        if question == "第一轮":
            first_entered.set()
            await release_first.wait()
        return ResponseEnvelope(
            response_type="chat",
            intent="chat",
            lang="zh",
            message=f"回复：{question}",
            handled_by="conversation_fallback",
        )

    monkeypatch.setattr(agent_module, "_run_turn_orchestrator", fake_core)

    async def collect(question):
        return [
            chunk
            async for chunk in agent_module.chat_stream(
                question,
                "web:serialized-turn",
            )
        ]

    async def run():
        first = asyncio.create_task(collect("第一轮"))
        await first_entered.wait()
        second = asyncio.create_task(collect("第二轮"))
        await asyncio.sleep(0)
        assert entered == ["第一轮"]
        release_first.set()
        await asyncio.gather(first, second)
        return await service.load("web:serialized-turn")

    memory = asyncio.run(run())

    assert entered == ["第一轮", "第二轮"]
    assert [(turn.role, turn.content) for turn in memory.recent_turns] == [
        ("user", "第一轮"),
        ("assistant", "回复：第一轮"),
        ("user", "第二轮"),
        ("assistant", "回复：第二轮"),
    ]


def test_web_turn_lock_is_released_before_sse_chunk_is_yielded(monkeypatch):
    import app.agent.participle_agent as agent_module
    import app.conversation.service as service_module
    from app.conversation.service import ConversationService
    from app.conversation.store import InMemoryConversationStore

    service = ConversationService(InMemoryConversationStore())
    monkeypatch.setattr(service_module, "_service", service)

    async def fake_core(_question, **_kwargs):
        return ResponseEnvelope(
            response_type="chat",
            intent="chat",
            lang="zh",
            message="已经生成",
            handled_by="conversation_fallback",
        )

    monkeypatch.setattr(agent_module, "_run_turn_orchestrator", fake_core)

    async def run():
        thread_id = "web:sse-backpressure"
        lock = service.turn_lock(thread_id)
        stream = agent_module.chat_stream("继续", thread_id)
        chunk = await anext(stream)
        locked_after_yield = lock.locked()
        del lock
        gc.collect()
        retained_after_yield = thread_id in service._turn_locks
        await stream.aclose()
        return chunk, locked_after_yield, retained_after_yield

    chunk, locked_after_yield, retained_after_yield = asyncio.run(run())

    assert chunk == "已经生成"
    assert not locked_after_yield
    assert not retained_after_yield


def test_unified_core_accepts_web_turn_request(monkeypatch):
    import app.agent.participle_agent as agent_module

    captured = {}

    async def fake_execute_turn(request, **_kwargs):
        captured["request"] = request
        return ResponseEnvelope(message="web handled", handled_by="test")

    monkeypatch.setattr(
        agent_module,
        "execute_turn",
        fake_execute_turn,
    )

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "继续",
            thread_id="web:phase43-core",
            channel="web",
            source_message_type="text",
            trace_id="phase43-trace",
        )
    )

    assert envelope.message == "web handled"
    assert captured["request"].channel == "web"
    assert captured["request"].thread_id == "web:phase43-core"


def test_web_unified_core_recipe_handler_keeps_grounded_recipe_data(
    monkeypatch,
):
    import app.agent.participle_agent as agent_module

    calls = []

    async def empty_memory(_question, **_kwargs):
        return MemoryHandlerResult()

    async def skip(_runtime):
        return None

    async def skip_domain(_runtime, **_kwargs):
        return None

    async def fake_route(*_args, **_kwargs):
        calls.append("route")
        return FastPathOutcome(
            kind="search",
            lang="zh",
            intent=IntentResult(
                related=True,
                category=IntentCategory.recipe_search,
                action="new_search",
                source="exact_rule",
                reason_code="PHASE43_GROUNDED_SEARCH",
            ),
            search_query="鸡腿 土豆",
            search_result={
                "success": True,
                "results": [
                    {
                        "id": "recipe-1",
                        "score": 0.9,
                        "metadata": {
                            "recipe_id": "recipe-1",
                            "name": "土豆炖鸡",
                            "ingredients": ["鸡腿", "土豆"],
                            "tags": ["炖", "家常"],
                        },
                    }
                ],
            },
        )

    async def fake_commit(*_args, **_kwargs):
        calls.append("commit")

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("Recipe Handler 命中后不得进入 Legacy Router")

    monkeypatch.setattr(agent_module, "_run_im_memory_prelude", empty_memory)
    monkeypatch.setattr(agent_module, "_handle_qq_exact_command", skip)
    monkeypatch.setattr(agent_module, "_handle_qq_device_pending", skip_domain)
    monkeypatch.setattr(agent_module, "_handle_qq_pending_state", skip)
    monkeypatch.setattr(agent_module, "_route_conversation_turn", fake_route)
    monkeypatch.setattr(agent_module, "_commit_search_state", fake_commit)
    monkeypatch.setattr(agent_module, "_handle_qq_conversation_fallback", forbidden)

    envelope = asyncio.run(
        agent_module._run_turn_orchestrator(
            "用鸡腿和土豆推荐一道菜",
            thread_id="web:phase43-grounded",
            channel="web",
            source_message_type="text",
            trace_id="phase43-grounded-trace",
        )
    )
    markdown = WebResponseRenderer().render(envelope)

    assert calls == ["route", "commit"]
    assert envelope.handled_by == "recipe_handler"
    assert envelope.data["recipes"][0]["id"] == "recipe-1"
    assert "土豆炖鸡" in markdown
    assert "鸡腿、土豆" in markdown
