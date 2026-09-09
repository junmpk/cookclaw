from __future__ import annotations

import json
import sys


def main() -> int:
    recipe_id = str(sys.argv[1] if len(sys.argv) > 1 else "").strip()
    if not recipe_id:
        print(json.dumps({"code": 404, "success": False}))
        return 0
    # The mock verifies only the identifier. Recipe facts still come from RAG.
    print(json.dumps({"code": 200, "success": True, "data": {"id": recipe_id}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
