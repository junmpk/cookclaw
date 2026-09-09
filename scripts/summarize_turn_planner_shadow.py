#!/usr/bin/env python3
"""汇总 Turn Planner Shadow JSONL 或应用日志。

用法：
  .venv/bin/python scripts/summarize_turn_planner_shadow.py \
    outputs/turn-planner-shadow.jsonl
  journalctl -u cookclaw --since today -o cat > /tmp/cookclaw.log
  .venv/bin/python scripts/summarize_turn_planner_shadow.py \
    /tmp/cookclaw.log --channel qq
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.orchestrator.planning.shadow_report import summarize_shadow_records


def _extract_rows(value: Any) -> list[dict]:
    if isinstance(value, dict):
        if value.get("schema_version") == "turn_planner_shadow_v1":
            return [value]
        rows: list[dict] = []
        for child in value.values():
            rows.extend(_extract_rows(child))
        return rows
    if isinstance(value, list):
        rows = []
        for child in value:
            rows.extend(_extract_rows(child))
        return rows
    return []


def _load_rows(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        document = None
    if document is not None:
        return _extract_rows(document)

    rows: list[dict] = []
    marker = "turn_planner_shadow "
    for line in raw.splitlines():
        text = line.strip()
        if marker in text:
            text = text.split(marker, 1)[1].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        rows.extend(_extract_rows(payload))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="汇总 CookClaw Turn Planner Shadow 记录"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Shadow JSONL 或包含 turn_planner_shadow 的应用日志",
    )
    parser.add_argument("--channel", default="", help="只统计指定通道，如 qq")
    args = parser.parse_args()

    rows = _load_rows(args.path)
    if args.channel:
        rows = [
            row
            for row in rows
            if (row.get("context_ref") or {}).get("channel") == args.channel
        ]
    print(
        json.dumps(
            summarize_shadow_records(rows),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
