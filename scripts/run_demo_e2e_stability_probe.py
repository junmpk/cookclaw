"""Long-input and same-thread rapid-message probes (no real device writes)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CONVERSATION_STORE", "memory")

from app.agent.participle_agent import qqbot_chat
from app.conversation.task_state_workspace import clear_thread


def compact(raw: str) -> dict:
    try:
        data = json.loads(raw)
        return {
            "type": data.get("type"),
            "message": data.get("message"),
            "query": (data.get("data") or {}).get("query"),
            "recipes": [item.get("name") for item in (data.get("data") or {}).get("recipes", [])],
        }
    except Exception as exc:
        return {"parse_error": repr(exc), "raw": raw[:1000]}


async def main() -> int:
    long_text = ("鸡肉" * 1800) + "，请推荐三道菜"
    long_raw = await qqbot_chat(long_text, thread_id="qq:dm:long-input:long-input")

    thread_id = "qq:dm:rapid-user:rapid-user"
    clear_thread(thread_id)
    seed = await qqbot_chat("推荐三道简单鸡肉菜", thread_id=thread_id)
    rapid = await asyncio.gather(
        qqbot_chat("换一批", thread_id=thread_id),
        qqbot_chat("为什么推荐第二道？", thread_id=thread_id),
        qqbot_chat("还是上一批", thread_id=thread_id),
        return_exceptions=True,
    )
    report = {
        "long_input": {
            "input_chars": len(long_text),
            "actual": compact(long_raw),
            "no_crash": bool(long_raw),
        },
        "rapid_same_thread": {
            "seed": compact(seed),
            "actual": [compact(item) if isinstance(item, str) else {"error": repr(item)} for item in rapid],
            "no_crash": all(not isinstance(item, Exception) for item in rapid),
        },
    }
    output = Path("/tmp/cookclaw-demo-e2e/stability-probe.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "long_chars": len(long_text),
        "long_type": report["long_input"]["actual"].get("type"),
        "rapid_no_crash": report["rapid_same_thread"]["no_crash"],
        "rapid_types": [item.get("type") for item in report["rapid_same_thread"]["actual"]],
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
