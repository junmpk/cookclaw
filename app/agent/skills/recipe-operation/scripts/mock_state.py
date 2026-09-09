from __future__ import annotations

import json
import tempfile
from pathlib import Path


STATE_PATH = Path(tempfile.gettempdir()) / "cookclaw_public_mock_device.json"


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"running": False, "recipe_id": None}


def save_state(*, running: bool, recipe_id: str | None = None) -> None:
    STATE_PATH.write_text(
        json.dumps({"running": running, "recipe_id": recipe_id}),
        encoding="utf-8",
    )
