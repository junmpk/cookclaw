"""WhatsApp/微信内部 Webhook 的共享密钥边界。"""

from fastapi.testclient import TestClient

import app.main as main_module


class _RecordingAdapter:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def handle_webhook(self, payload: dict) -> None:
        self.payloads.append(dict(payload))


def _client() -> TestClient:
    return TestClient(main_module.app)


def test_whatsapp_webhook_rejects_missing_or_wrong_secret(monkeypatch):
    adapter = _RecordingAdapter()
    monkeypatch.setattr(main_module, "_wa_adapter", adapter)
    monkeypatch.setattr(
        main_module.settings,
        "WHATSAPP_WEBHOOK_SECRET",
        "wa-test-secret",
    )

    missing = _client().post("/api/v1/whatsapp/webhook", json={"id": "m1"})
    wrong = _client().post(
        "/api/v1/whatsapp/webhook",
        json={"id": "m1"},
        headers={"X-Webhook-Secret": "wrong"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert adapter.payloads == []


def test_whatsapp_webhook_accepts_matching_secret(monkeypatch):
    adapter = _RecordingAdapter()
    monkeypatch.setattr(main_module, "_wa_adapter", adapter)
    monkeypatch.setattr(
        main_module.settings,
        "WHATSAPP_WEBHOOK_SECRET",
        "wa-test-secret",
    )

    response = _client().post(
        "/api/v1/whatsapp/webhook",
        json={"id": "m1"},
        headers={"X-Webhook-Secret": "wa-test-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert adapter.payloads == [{"id": "m1"}]


def test_weixin_webhook_rejects_unconfigured_secret(monkeypatch):
    adapter = _RecordingAdapter()
    monkeypatch.setattr(main_module, "_wx_adapter", adapter)
    monkeypatch.setattr(main_module.settings, "WEIXIN_WEBHOOK_SECRET", "")

    response = _client().post(
        "/api/v1/weixin/webhook",
        json={"message_id": "m2"},
    )

    assert response.status_code == 503
    assert adapter.payloads == []


def test_weixin_webhook_accepts_matching_secret(monkeypatch):
    adapter = _RecordingAdapter()
    monkeypatch.setattr(main_module, "_wx_adapter", adapter)
    monkeypatch.setattr(
        main_module.settings,
        "WEIXIN_WEBHOOK_SECRET",
        "wx-test-secret",
    )

    response = _client().post(
        "/api/v1/weixin/webhook",
        json={"message_id": "m2"},
        headers={"X-Webhook-Secret": "wx-test-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert adapter.payloads == [{"message_id": "m2"}]
