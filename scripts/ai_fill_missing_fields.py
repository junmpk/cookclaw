"""AI 补全菜谱缺失字段。

用法：
  # 50 行样本验证
  python ai_fill_missing_fields.py recipes.xlsx --sample 50 --output sample_filled.xlsx

  # 全量执行
  python ai_fill_missing_fields.py recipes.xlsx --output recipes_filled.xlsx

  # 从断点续跑（跳过 output 中已有的行）
  python ai_fill_missing_fields.py recipes.xlsx --output recipes_filled.xlsx --resume
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import openpyxl

# 加载 .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# ── 配置 ──────────────────────────────────────────────────────────────

DASHSCOPE_API_KEY = os.getenv(
    "DASHSCOPE_API_KEY",
    "",
)
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AI_FILL_MODEL", "qwen-plus")
MAX_RETRIES = 2
BATCH_SIZE = 5  # 并发请求数
RATE_LIMIT_DELAY = 0.3  # 请求间隔（秒）

# 需要 AI 补全的字段及其生成 prompt
FIELDS_TO_FILL = {
    "标签": {
        "prompt": (
            "根据菜名和食材，生成 5-8 个英文标签（用、分隔），"
            "描述口味、做法、食材类型、场景等。"
            "例：Sweet、Creamy、Blended、Fruit、Beverage、Quick"
        ),
    },
    "标签标准编码": {
        "prompt": (
            "根据菜名、食材和标签，生成结构化 JSON 标签编码。\n"
            "格式：{\"record_type\":[],\"cuisine\":[],\"flavor\":[],\"method\":[],"
            "\"main_ingredient\":[],\"meal\":[],\"nutrition\":[],\"scene\":[],"
            "\"diet\":[],\"difficulty\":[]}\n"
            "每个数组填英文小写值，没有就留空数组。不要编造不存在的属性。"
        ),
    },
    "菜谱描述": {
        "prompt": (
            "根据菜名和食材，用一句话（30-60字）描述这道菜的特点。"
            "用与 lang 字段相同的语言写。不要提及具体用量。"
        ),
    },
    "小贴士": {
        "prompt": (
            "根据菜名和烹饪步骤，写一条实用的小贴士（20-50字）。"
            "用与 lang 字段相同的语言写。可以是技巧、替代食材或注意事项。"
        ),
    },
    "调料": {
        "prompt": (
            "根据菜名和食材列表，推断可能需要的调料（盐、油、酱油等）。"
            "用纯文本输出，每行一个调料（含用量），不要 JSON 数组。用与 lang 字段相同的语言写。"
        ),
    },
    "营养成分": {
        "prompt": (
            "根据食材和菜名，估算每 100g 的大致营养成分。输出 JSON：\n"
            "{\"basis\":\"per_100g_estimated\",\"calorie_kcal\":数字,"
            "\"protein_g\":数字,\"fat_g\":数字,\"carbohydrate_g\":数字,"
            "\"fiber_g\":数字}\n"
            "数字为合理估算值，不要编造过于精确的数字。"
        ),
    },
    "食材": {
        "prompt": (
            "根据菜名和烹饪步骤，反推主要食材列表（含大概用量）。"
            "用纯文本输出，每行一个食材（如 '200g avocado'），不要 JSON 数组。用与 lang 字段相同的语言写。"
        ),
    },
    "预计用时": {
        "prompt": (
            "根据菜名和烹饪步骤，估算总用时（分钟）。只输出一个数字。"
        ),
    },
}


def build_prompt(row_data: dict, fields_needed: list[str]) -> str:
    """构建单次 AI 调用的 prompt，一次性补全多个缺失字段。"""
    parts = [f"你是一位专业的菜谱编辑。请根据已有信息补全缺失字段。\n"]

    # 已有信息
    parts.append(f"菜名：{row_data.get('名称', '')}")
    parts.append(f"语言：{row_data.get('lang', 'zh')}")
    if row_data.get("食材"):
        parts.append(f"食材：{row_data['食材'][:200]}")
    if row_data.get("标签"):
        parts.append(f"标签：{row_data['标签'][:200]}")
    if row_data.get("烹饪步骤"):
        steps = str(row_data["烹饪步骤"])[:300]
        parts.append(f"烹饪步骤（摘要）：{steps}")
    if row_data.get("调料"):
        parts.append(f"调料：{row_data['调料'][:200]}")

    parts.append(f"\n请补全以下缺失字段。输出 JSON，key 必须严格使用以下中文名称（不要代码块）：")

    for field in fields_needed:
        field_info = FIELDS_TO_FILL.get(field, {})
        parts.append(f'- "{field}"：{field_info.get("prompt", "请根据上下文补全")}')

    return "\n".join(parts)


def call_ai(prompt: str) -> dict | None:
    """调用 DashScope API，返回解析后的 JSON。"""
    url = f"{DASHSCOPE_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是专业菜谱编辑，只输出 JSON，不要代码块和解释。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 800,
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(url, headers=headers, json=body)
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Clean markdown code blocks
                if content.startswith("```"):
                    content = content.split("\n", 1)[-1]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                return json.loads(content)
        except (json.JSONDecodeError, httpx.HTTPError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            print(f"  AI call failed: {type(e).__name__}: {e}")
            return None
    return None


def process_row(row_data: dict, headers: list[str]) -> dict:
    """处理单行，返回需要补全的字段值。"""
    # 确定哪些字段缺失
    fields_needed = []
    for field in FIELDS_TO_FILL:
        if field not in headers:
            fields_needed.append(field)
        else:
            val = row_data.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                fields_needed.append(field)

    if not fields_needed:
        return {}

    prompt = build_prompt(row_data, fields_needed)
    result = call_ai(prompt)
    if result is None:
        return {}

    # Key 映射兜底：AI 可能用英文 key 返回
    key_alias = {
        "标签": ["tags", "tag", "labels"],
        "标签标准编码": ["structured_tags", "tag_encoding", "label_encoding"],
        "菜谱描述": ["description", "desc", "recipe_description"],
        "小贴士": ["tips", "tip", "note", "notes"],
        "调料": ["seasoning", "seasonings", "condiments"],
        "营养成分": ["nutrition", "nutritional_info", "nutrients"],
        "食材": ["ingredients", "ingredient"],
        "预计用时": ["estimated_time", "cook_time", "time", "prep_time"],
    }

    filled = {}
    for field in fields_needed:
        val = result.get(field)
        # 尝试英文别名
        if val is None:
            for alias in key_alias.get(field, []):
                if alias in result:
                    val = result[alias]
                    break
        if val is not None:
            if isinstance(val, (dict, list)):
                filled[field] = json.dumps(val, ensure_ascii=False)
            else:
                filled[field] = str(val) if val is not None else ""

    # 格式修正：调料和食材应为换行分隔纯文本，不是 JSON 数组
    for text_field in ['调料', '食材']:
        if text_field in filled:
            raw = filled[text_field]
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    filled[text_field] = "\n".join(str(item) for item in parsed)
            except (json.JSONDecodeError, TypeError):
                pass  # 已经是纯文本，不需要转换

    # 格式修正：预计用时转为数字
    if '预计用时' in filled:
        try:
            filled['预计用时'] = int(float(filled['预计用时']))
        except (ValueError, TypeError):
            pass

    return filled


def _write_output(all_rows, std_headers, output_path, total):
    """写出结果到 Excel 文件。"""
    wb_out = openpyxl.Workbook()
    ws_out = wb_out.active
    ws_out.title = "Sheet1"
    for col_idx, header in enumerate(std_headers, 1):
        ws_out.cell(row=1, column=col_idx, value=header)
    for row_idx in range(total):
        row = all_rows[row_idx]
        for col_idx, header in enumerate(std_headers, 1):
            val = row.get(header)
            ws_out.cell(row=row_idx + 2, column=col_idx, value=val)
    wb_out.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="AI 补全菜谱缺失字段")
    parser.add_argument("input", help="输入 Excel 文件")
    parser.add_argument("--output", "-o", required=True, help="输出 Excel 文件")
    parser.add_argument("--sample", type=int, default=0, help="只处理 N 行样本")
    parser.add_argument("--resume", action="store_true", help="从断点续跑")
    args = parser.parse_args()

    # Load input
    print(f"Loading {args.input}...")
    wb_in = openpyxl.load_workbook(args.input, read_only=True)
    ws_in = wb_in.active
    headers = [cell.value for cell in next(ws_in.iter_rows(min_row=1, max_row=1))]

    # Build target headers (standard order + 标签标准编码)
    std_headers = [
        "id", "名称", "食材", "标签", "标签标准编码", "营养成分",
        "图片地址", "lang", "菜谱描述", "预计用时", "难易程度",
        "小贴士", "调料", "烹饪步骤", "机型",
    ]
    # Add any headers not in standard
    for h in headers:
        if h not in std_headers:
            std_headers.append(h)

    # Read all rows
    all_rows = []
    for row in ws_in.iter_rows(min_row=2, values_only=True):
        row_dict = {}
        for i, val in enumerate(row):
            if i < len(headers):
                row_dict[headers[i]] = val
        all_rows.append(row_dict)
    wb_in.close()

    total = len(all_rows)
    print(f"Loaded {total} rows, {len(headers)} columns")
    print(f"Target columns ({len(std_headers)}): {std_headers}")

    # Select rows to process
    if args.sample > 0:
        # Pick rows with most missing fields
        scored = []
        for idx, row in enumerate(all_rows):
            missing = sum(
                1 for f in FIELDS_TO_FILL
                if f not in row or row[f] is None or (isinstance(row[f], str) and not row[f].strip())
            )
            scored.append((missing, idx))
        scored.sort(reverse=True)
        process_indices = [idx for _, idx in scored[:args.sample]]
        print(f"Selected {len(process_indices)} rows with most missing fields for sample")
    else:
        process_indices = list(range(total))

    # Check resume: load existing output
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from {args.output}...")
        wb_out = openpyxl.load_workbook(args.output, read_only=True)
        ws_out = wb_out.active
        out_headers = [cell.value for cell in next(ws_out.iter_rows(min_row=1, max_row=1))]
        id_col_out = out_headers.index("id")
        lang_col_out = out_headers.index("lang")
        # 用标签标准编码判断是否已被 AI 处理（该字段原始不存在，只有 AI 填充后才有值）
        tag_enc_col = out_headers.index("标签标准编码") if "标签标准编码" in out_headers else None
        for row in ws_out.iter_rows(min_row=2, values_only=True):
            key = (str(row[id_col_out]), str(row[lang_col_out]))
            # 只有标签标准编码非空才算已处理
            if tag_enc_col is not None:
                val = row[tag_enc_col]
                if val and str(val).strip() and str(val).strip() != "null":
                    processed_ids.add(key)
            else:
                processed_ids.add(key)
        wb_out.close()
        print(f"Already processed: {len(processed_ids)} rows")

    # Process rows
    print(f"\nProcessing {len(process_indices)} rows...")
    filled_count = 0
    failed_count = 0

    # 中断时保存已处理的结果
    import signal

    def _save_partial(signum, frame):
        print(f"\n收到中断信号，保存已处理的结果...")
        _write_output(all_rows, std_headers, args.output, total)
        print(f"已保存 {filled_count} 行到 {args.output}")
        sys.exit(0)

    signal.signal(signal.SIGUSR1, _save_partial)
    print(f"  (发送 kill -USR1 {os.getpid()} 可随时保存并退出)")

    for progress, row_idx in enumerate(process_indices):
        row = all_rows[row_idx]
        row_key = (str(row.get("id", "")), str(row.get("lang", "")))

        if row_key in processed_ids:
            continue

        # Process
        filled = process_row(row, headers)
        if filled:
            row.update(filled)
            filled_count += 1
        else:
            failed_count += 1

        # Progress
        if (progress + 1) % 10 == 0 or progress == len(process_indices) - 1:
            print(f"  [{progress + 1}/{len(process_indices)}] "
                  f"filled={filled_count} failed={failed_count}", flush=True)

        # 每 100 行自动保存一次，防止中断丢失进度
        if (progress + 1) % 100 == 0:
            _write_output(all_rows, std_headers, args.output, total)
            print(f"  💾 自动保存: {progress + 1}/{len(process_indices)} 行已写入 {args.output}", flush=True)

        # Rate limit
        time.sleep(RATE_LIMIT_DELAY)

    # Write output
    print(f"\nWriting {args.output}...")
    _write_output(all_rows, std_headers, args.output, total)
    print(f"Saved {total} rows to {args.output}")
    print(f"AI filled: {filled_count}, Failed: {failed_count}")


if __name__ == "__main__":
    main()
