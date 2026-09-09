from __future__ import annotations

import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT = {
    "default_device": "demo",
    "devices": {
        "demo": {
            "provider_device_id": "mock-device",
            "device_type": 0,
            "name": "Demo cooker",
        }
    },
}


def load_config() -> dict:
    path = _ROOT / "config.json"
    if not path.exists():
        return _DEFAULT
    return json.loads(path.read_text(encoding="utf-8"))


def get_default_device_id(config: dict) -> str | None:
    devices = config.get("devices") or {}
    if not devices:
        return None
    return str(config.get("default_device") or next(iter(devices)))


def resolve_device(config: dict, device_id: str | None = None) -> dict:
    devices = config.get("devices") or {}
    selected = str(device_id or get_default_device_id(config) or "")
    if selected not in devices:
        raise ValueError(f"unknown device_id: {selected}")
    return {"device_id": selected, **devices[selected]}


def list_devices(config: dict) -> list[tuple[str, str]]:
    devices = config.get("devices") or {}
    default = get_default_device_id(config)
    ordered = sorted(devices, key=lambda item: (item != default, item))
    return [(item, str(devices[item].get("name") or item)) for item in ordered]
