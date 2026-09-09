from __future__ import annotations

import sys

from mock_state import save_state


def main() -> int:
    recipe_id = str(sys.argv[1] if len(sys.argv) > 1 else "").strip()
    if not recipe_id:
        print("执行失败：missing recipe id", file=sys.stderr)
        return 2
    save_state(running=True, recipe_id=recipe_id)
    print("已开始制作（公开版 Mock，不会控制真实设备）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
