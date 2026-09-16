"""离线验证面试演示契约；只使用虚拟食谱，不访问模型、Milvus 或设备。"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.demo.models import Recipe
from app.demo.server import create_app


class ContractRecipeStore:
    backend_name = "contract_fixture"

    def __init__(self) -> None:
        self.recipes = [
            Recipe(
                id=f"contract-dish-{index}",
                name=f"演示菜 {index}",
                source=f"contract://dish/{index}",
                ingredients=["演示食材", f"配料{index}"],
                kind="dish",
            )
            for index in range(1, 5)
        ] + [
            Recipe(
                id="contract-soup-1",
                name="演示汤 1",
                source="contract://soup/1",
                ingredients=["演示食材", "汤料"],
                kind="soup",
            )
        ]

    async def search(self, _query: str, limit: int = 12) -> list[Recipe]:
        return self.recipes[:limit]

    def close(self) -> None:
        return None


def wait_stable(client: TestClient, run_id: str) -> dict:
    for _ in range(300):
        row = client.get(f"/api/runs/{run_id}").json()
        if row["status"] != "running":
            return row
        time.sleep(0.02)
    raise RuntimeError("演示工作流未在预期时间内到达稳定状态")


def main() -> None:
    brief = {
        "request": "四个人三菜一汤，使用现有食材，45分钟内用两个设备完成",
        "people": 4,
        "dishes": 3,
        "soups": 1,
        "exclusions": ["花生"],
        "available_ingredients": ["演示食材"],
        "budget_yuan": 50,
        "max_minutes": 45,
        "equipment": ["灶台", "空气炸锅"],
        "mode": "rehearsal",
    }
    with tempfile.TemporaryDirectory(prefix="cookclaw-demo-contract-") as temp:
        app = create_app(Path(temp), ContractRecipeStore())
        with TestClient(app) as client:
            created = client.post("/api/runs", json=brief)
            created.raise_for_status()
            run_id = created.json()["id"]
            awaiting = wait_stable(client, run_id)
            identity = awaiting["workflow_identity"]
            metrics = awaiting["metrics"]
            selected = awaiting["state"]["complexity_profile"]["selected_agents"]
            assert awaiting["status"] == "awaiting_confirmation"
            assert selected == [
                "research",
                "dietary",
                "inventory",
                "menu",
                "scheduler",
            ]
            assert identity["thread_id"] == created.json()["thread_id"]
            assert identity["run_id"] == run_id
            assert identity["checkpoint_thread_id"].endswith(f":{run_id}:v1")
            assert awaiting["state"]["plan_validation"]["status"] in {
                "ready",
                "needs_confirmation",
            }
            assert metrics["node_count"] >= 8

            confirmed = client.post(
                f"/api/runs/{run_id}/decision",
                json={"version": 1, "approved": True},
            )
            confirmed.raise_for_status()
            completed = wait_stable(client, run_id)
            assert completed["status"] == "completed"
            assert completed["state"]["execution"]["provider"] == "mock"

            print(json.dumps({
                "contract": "cookclaw_interview_demo_v1",
                "status": "passed",
                "recipe_data": "synthetic_contract_fixture",
                "external_calls": False,
                "selected_agents": selected,
                "workflow_identity": identity,
                "metrics": completed["metrics"],
                "execution_provider": "mock",
            }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
