#!/usr/bin/env bash
# ============================================================
#  CookClaw 灌库脚本 —— 部署时在服务器执行一次，把菜谱数据写入向量库。
#
#  不分方案，统一执行：脚本读 .env 里的 RECIPE_MILVUS_URI，
#    指向本地文件 → 方案A（Milvus Lite，在服务器本地重建数据）
#    指向 http/tcp → 方案B（Milvus Server，灌到远程服务）
#  三步：Excel 校验清洗 → recipe_collection（纯语义）→ recipe_hybrid（线上混合集合）。
#
#  用法（项目根目录）：
#    ./scripts/ingest.sh                        # 用默认 xlsx，重建集合
#    ./scripts/ingest.sh path/to/recipes.xlsx   # 指定数据文件
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

XLSX="${1:-菜品清单_2026.04.08_enriched.xlsx}"
SKILL_DIR="app/agent/skills/recipe-search"

# ── 选 python：优先根 .venv（精简、无 torch），否则 uv run ──
if [[ -x ".venv/bin/python" ]]; then
  PY=(".venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PY=("uv" "run" "python")
else
  echo "[ingest] 找不到 .venv/bin/python，也没有 uv。请先在项目根执行 'uv sync'。" >&2
  exit 1
fi

# ── 前置检查 ──
[[ -f ".env" ]] || { echo "[ingest] 缺 .env（cp deploy/cookclaw.prod.env.example .env 后填值）" >&2; exit 1; }

# 数据文件：默认名找不到时，自动选项目根唯一的 .xlsx（省得在服务器上敲中文名）
if [[ ! -f "$XLSX" ]]; then
  shopt -s nullglob; _xlsx=( *.xlsx ); shopt -u nullglob
  if [[ ${#_xlsx[@]} -eq 1 ]]; then
    XLSX="${_xlsx[0]}"
    echo "[ingest] 默认数据文件不在，自动选用：$XLSX"
  fi
fi
[[ -f "$XLSX" ]] || { echo "[ingest] 找不到数据文件：$XLSX（项目根 .xlsx：$(ls -1 *.xlsx 2>/dev/null | tr '\n' ' ' || echo 无)）" >&2; exit 1; }

# 仅为打印目标：从 .env 取 RECIPE_MILVUS_URI（不 source 整个 .env，避免副作用；
# 真正的连接信息由两个 Python 脚本各自 load_dotenv 读取）
TARGET="$(grep -E '^[[:space:]]*RECIPE_MILVUS_URI=' .env | tail -1 | cut -d= -f2- | tr -d ' "')"
TARGET="${TARGET:-recipe_milvus.db}"
case "$TARGET" in
  http*|tcp*|unix*) PLAN="方案B（Milvus Server）" ;;
  *)                PLAN="方案A（Milvus Lite 本地文件）" ;;
esac

if [[ "$PLAN" == 方案B* ]] \
  && [[ -z "${MILVUS_ANALYZER_TYPE:-}" ]] \
  && ! grep -Eq '^[[:space:]]*MILVUS_ANALYZER_TYPE=' .env; then
  export MILVUS_ANALYZER_TYPE=standard
  ANALYZER_NOTE="standard（Milvus Server 默认兼容）"
else
  ANALYZER_NOTE="${MILVUS_ANALYZER_TYPE:-$(grep -E '^[[:space:]]*MILVUS_ANALYZER_TYPE=' .env | tail -1 | cut -d= -f2- | tr -d ' "' || true)}"
  ANALYZER_NOTE="${ANALYZER_NOTE:-默认(jieba)}"
fi

echo "============================================================"
echo "  CookClaw 灌库"
echo "  识别方案：$PLAN"
echo "  目标 URI：$TARGET"
echo "  Analyzer：$ANALYZER_NOTE"
echo "  数据文件：$XLSX"
echo "  说明：会调用百炼 embedding，耗时几分钟、有少量 API 费用（一次性）。"
echo "============================================================"

VALIDATED_DIR="tmp"
BASE_NAME="$(basename "$XLSX")"
STEM_NAME="${BASE_NAME%.*}"
VALIDATED_XLSX="$VALIDATED_DIR/${STEM_NAME}.validated.xlsx"
INVALID_REPORT="$VALIDATED_DIR/${STEM_NAME}.invalid_rows.csv"

# ── 1) Excel 校验/清洗：删除缺食材、缺图片、非法语言等不标准行 ──
echo "[1/3] 校验并清洗 Excel ..."
"${PY[@]}" "$SKILL_DIR/validate_ingest_excel.py" "$XLSX" --out "$VALIDATED_XLSX" --report "$INVALID_REPORT"
XLSX="$VALIDATED_XLSX"
echo "[ingest] 后续入库使用清洗副本：$XLSX"
echo "[ingest] 不合规行报告：$INVALID_REPORT"

# ── 2) Excel → recipe_collection（纯语义基线）──
echo "[2/3] 向量化并写入 recipe_collection ..."
"${PY[@]}" "$SKILL_DIR/ingest_local.py" "$XLSX" --recreate

# ── 3) recipe_collection → recipe_hybrid（线上混合检索集合）──
echo "[3/3] 迁移生成 recipe_hybrid（线上集合）..."
"${PY[@]}" "$SKILL_DIR/migrate_hybrid.py" --recreate

echo
echo "✅ 灌库完成。可验证检索："
echo "   ${PY[*]} $SKILL_DIR/recipe_search.py \"红烧肉\" 3"
