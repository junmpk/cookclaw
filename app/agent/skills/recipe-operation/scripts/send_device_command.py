from __future__ import annotations

import json
import sys

from mock_state import save_state


def main() -> int:
    action = str(sys.argv[1] if len(sys.argv) > 1 else "")
    if action not in {"start", "stop"}:
        print(json.dumps({"code": 400, "success": False}))
        return 2
    save_state(running=action == "start")
    print(json.dumps({"code": 200, "success": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
