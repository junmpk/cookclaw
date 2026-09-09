"""Isolated ASGI/SSE contract and Web default-thread isolation probe."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CONVERSATION_STORE", "memory")

import httpx

from app.main import app
from app.conversation.task_state_workspace import clear_thread


def sse_text(body: str) -> str:
    chunks = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunks.append(str(json.loads(line[6:]).get("content") or ""))
    return "".join(chunks)


async def main() -> int:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://cookclaw.test") as client_a:
        health = await client_a.get("/api/v1/health")
        invalid = await client_a.post("/api/v1/chat", json={})
        blank = await client_a.post("/api/v1/chat", json={"question": "   "})

        clear_thread("default")
        first = await client_a.post("/api/v1/chat", json={"question": "推荐三道简单鸡肉菜"})
        first_text = sse_text(first.text)

    # Simulate a different browser; API has no session/user field or cookie.
    async with httpx.AsyncClient(transport=transport, base_url="http://cookclaw.test") as client_b:
        second = await client_b.post("/api/v1/chat", json={"question": "为什么推荐第二道？"})
        second_text = sse_text(second.text)

    first_names = re.findall(r"^\d+\. \*\*(.+?)\*\*$", first_text, flags=re.MULTILINE)
    leaked = any(name in second_text for name in first_names)
    report = {
        "health": {"status": health.status_code, "json": health.json()},
        "invalid_request": {"status": invalid.status_code, "body": invalid.json()},
        "blank_request": {"status": blank.status_code, "text": sse_text(blank.text), "done": "data: [DONE]" in blank.text},
        "recipe_sse": {
            "status": first.status_code,
            "content_type": first.headers.get("content-type"),
            "done": "data: [DONE]" in first.text,
            "recipes": first_names,
            "text": first_text,
        },
        "cross_browser_default_thread": {
            "leaked": leaked,
            "browser_b_text": second_text,
            "matched_recipe_names": [name for name in first_names if name in second_text],
        },
    }
    output = Path("/tmp/cookclaw-demo-e2e/http-probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "health": health.status_code,
        "invalid": invalid.status_code,
        "blank_status": blank.status_code,
        "recipes": first_names,
        "cross_browser_leaked": leaked,
        "matched": report["cross_browser_default_thread"]["matched_recipe_names"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
