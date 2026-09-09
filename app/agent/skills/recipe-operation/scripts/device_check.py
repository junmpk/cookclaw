from __future__ import annotations

import json
import sys

from mock_state import load_state


def main() -> int:
    state = load_state()
    payload = {
        "code": 200,
        "success": True,
        "data": {
            "isOnline": 1,
            "attributes": {"status": 1 if state.get("running") else 0},
        },
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
