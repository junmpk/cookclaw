#!/usr/bin/env python3
"""
将工作簿1.xlsx（Milvus auto-ID → device code 映射）导出为 JSON。

用法：
    python3 scripts/export_id_code_mapping.py [输入xlsx] [输出json]

默认：
    输入: ~/Downloads/工作簿1.xlsx
    输出: app/agent/skills/recipe-search/data/id_to_code.json
"""
import json
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = Path.home() / "Downloads" / "工作簿1.xlsx"
DEFAULT_OUTPUT = ROOT / "app" / "agent" / "skills" / "recipe-search" / "data" / "id_to_code.json"


def export(input_path: Path, output_path: Path) -> None:
    wb = openpyxl.load_workbook(input_path, read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        print("❌ Excel 为空", file=sys.stderr)
        sys.exit(1)

    # 找到 id 和 code 列的索引
    header = [str(c).strip().lower() if c else "" for c in rows[0]]
    try:
        id_idx = header.index("id")
        code_idx = header.index("code")
    except ValueError:
        print(f"❌ 找不到 id/code 列，实际列名: {header}", file=sys.stderr)
        sys.exit(1)

    mapping: dict[str, int] = {}
    skipped = 0
    for row in rows[1:]:
        milvus_id = row[id_idx]
        code = row[code_idx]
        if milvus_id is None or code is None:
            skipped += 1
            continue
        mapping[str(int(milvus_id))] = int(code)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)

    print(f"✅ 导出 {len(mapping)} 条映射 → {output_path}")
    if skipped:
        print(f"   跳过 {skipped} 条空行", file=sys.stderr)


if __name__ == "__main__":
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    export(input_path, output_path)
