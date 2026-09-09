"""HTTP 存活探测路由测试；不启动应用 lifespan。"""

from fastapi.testclient import TestClient

from app.main import app


def test_root_head_probe_returns_ok_without_body():
    response = TestClient(app).head("/")

    assert response.status_code == 200
    assert response.content == b""
