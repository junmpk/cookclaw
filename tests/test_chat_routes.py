"""Web chat API 的会话隔离与输入边界测试；不启动应用 lifespan。"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.routes.chat as chat_routes


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(chat_routes.router, prefix="/api/v1")
    return TestClient(app)


def test_legacy_request_gets_unique_web_thread_id_and_passes_it_to_stream(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_chat_stream(question: str, thread_id: str):
        calls.append((question, thread_id))
        yield "收到"

    monkeypatch.setattr(chat_routes, "chat_stream", fake_chat_stream)
    client = _client()

    first = client.post("/api/v1/chat", json={"question": "  晚饭吃什么  "})
    second = client.post("/api/v1/chat", json={"question": "再推荐一个"})

    first_id = first.headers["X-CookClaw-Thread-Id"]
    second_id = second.headers["X-CookClaw-Thread-Id"]
    assert first.status_code == 200
    assert second.status_code == 200
    assert first_id.startswith("web:")
    assert second_id.startswith("web:")
    assert first_id != second_id
    assert calls == [("晚饭吃什么", first_id), ("再推荐一个", second_id)]
    assert 'data: {"content": "收到"}' in first.text
    assert first.text.endswith("data: [DONE]\n\n")


def test_explicit_thread_id_is_reused_and_returned(monkeypatch):
    seen: list[str] = []

    async def fake_chat_stream(_question: str, thread_id: str):
        seen.append(thread_id)
        yield "继续"

    monkeypatch.setattr(chat_routes, "chat_stream", fake_chat_stream)
    response = _client().post(
        "/api/v1/chat",
        json={
            "question": "家里有鸡肉",
            "thread_id": "web:0123456789abcdef0123456789abcdef",
        },
    )

    assert response.status_code == 200
    assert response.headers["X-CookClaw-Thread-Id"] == (
        "web:0123456789abcdef0123456789abcdef"
    )
    assert response.headers["Cache-Control"] == "no-store"
    assert seen == ["web:0123456789abcdef0123456789abcdef"]


def test_web_thread_id_cannot_enter_an_im_namespace(monkeypatch):
    seen: list[str] = []

    async def fake_chat_stream(_question: str, thread_id: str):
        seen.append(thread_id)
        yield "继续"

    monkeypatch.setattr(chat_routes, "chat_stream", fake_chat_stream)
    response = _client().post(
        "/api/v1/chat",
        json={"question": "继续", "thread_id": "qq:dm:someone:someone"},
    )

    assert response.status_code == 422
    assert seen == []


def test_web_thread_id_rejects_low_entropy_or_empty_tokens(monkeypatch):
    async def must_not_stream(*_args, **_kwargs):
        raise AssertionError("invalid session tokens must not reach chat_stream")
        yield  # pragma: no cover

    monkeypatch.setattr(chat_routes, "chat_stream", must_not_stream)
    client = _client()

    for thread_id in ("web:", "web:demo", "client-session_01"):
        response = client.post(
            "/api/v1/chat",
            json={"question": "继续", "thread_id": thread_id},
        )
        assert response.status_code == 422


def test_null_thread_id_is_backward_compatible(monkeypatch):
    seen: list[str] = []

    async def fake_chat_stream(_question: str, thread_id: str):
        seen.append(thread_id)
        yield "好"

    monkeypatch.setattr(chat_routes, "chat_stream", fake_chat_stream)
    response = _client().post(
        "/api/v1/chat",
        json={"question": "你好", "thread_id": None},
    )

    assert response.status_code == 200
    assert seen == [response.headers["X-CookClaw-Thread-Id"]]
    assert seen[0].startswith("web:")


def test_health_exposes_non_secret_orchestration_modes(monkeypatch):
    monkeypatch.setattr(chat_routes.settings, "TURN_PLANNER_MODE", "active")
    monkeypatch.setattr(
        chat_routes.settings,
        "TURN_PLANNER_ACTIVE_ROLLOUT_PERCENT",
        12.5,
    )

    response = _client().get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["orchestration"] == {
        "turn_mode": "unified",
        "planner_mode": "active",
        "planner_model": chat_routes.settings.TURN_PLANNER_MODEL,
        "planner_rollout_percent": 12.5,
    }


def test_chat_request_rejects_empty_or_oversized_question(monkeypatch):
    async def must_not_stream(*_args, **_kwargs):
        raise AssertionError("invalid requests must not reach chat_stream")
        yield  # pragma: no cover - keeps this an async generator

    monkeypatch.setattr(chat_routes, "chat_stream", must_not_stream)
    client = _client()

    empty = client.post("/api/v1/chat", json={"question": "   "})
    oversized = client.post(
        "/api/v1/chat",
        json={"question": "a" * (chat_routes._MAX_QUESTION_CHARS + 1)},
    )

    assert empty.status_code == 422
    assert oversized.status_code == 422


def test_chat_request_rejects_oversized_or_header_unsafe_thread_id(monkeypatch):
    async def must_not_stream(*_args, **_kwargs):
        raise AssertionError("invalid requests must not reach chat_stream")
        yield  # pragma: no cover - keeps this an async generator

    monkeypatch.setattr(chat_routes, "chat_stream", must_not_stream)
    client = _client()

    oversized = client.post(
        "/api/v1/chat",
        json={
            "question": "你好",
            "thread_id": "a" * (chat_routes._MAX_THREAD_ID_CHARS + 1),
        },
    )
    unsafe = client.post(
        "/api/v1/chat",
        json={"question": "你好", "thread_id": "web:会话"},
    )

    assert oversized.status_code == 422
    assert unsafe.status_code == 422
