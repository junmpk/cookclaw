"""
本地灌库脚本 — 将 Excel 菜谱向量化后写入本地 Milvus Lite。

用法（在本技能目录下，用独立 venv 运行）：
  # 1) 先看 Excel 的列名（不写库），据此核对脚本顶部 COLUMN_MAP
  .venv/bin/python ingest_local.py /path/to/recipes.xlsx --inspect

  # 2) 正式灌库（首次或重建用 --recreate 清空旧集合）
  .venv/bin/python ingest_local.py /path/to/recipes.xlsx --recreate

依赖：openpyxl, pymilvus, milvus-lite, openai(百炼 Embedding)
需要环境变量：DASHSCOPE_API_KEY（向量化用，已自动从项目根 .env 读取）

写入的集合 schema：
  id(自增主键) + embedding(FLOAT_VECTOR, dim) + metadata(JSON)
  metadata 含：recipe_id / name / ingredients[] / tags[] / facets / nutrition /
  image_url / description / recipe_detail
  —— 与查询侧 dataset/milvus.py 的读取契约一致。
"""
import os
import re
import sys
import json
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


# 加载项目根 .env（standalone 运行也能拿到 DASHSCOPE_API_KEY）
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(_PROJECT_ROOT / ".env")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MILVUS_CONFIG, EMBEDDING_CONFIG  # noqa: E402
from module.embedding import get_embedding_model     # noqa: E402
from dataset.milvus import build_milvus_client, safe_milvus_uri        # noqa: E402
from pymilvus import MilvusClient, DataType           # noqa: E402
from loguru import logger                             # noqa: E402
import openpyxl                                        # noqa: E402


# ─────────────────────────────────────────────────────────────
#  ⭐ 列映射：逻辑字段 → 你 Excel 里的实际列名
#  先用 `--inspect` 看真实列名，再改这里。正式入库前需通过 validate_ingest_excel.py 校验。
# ─────────────────────────────────────────────────────────────
REQUIRED_COLUMN_MAP = {
    "recipe_id":   "id",       # 菜谱唯一 ID
    "name":        "名称",      # 菜名（必填）
    "ingredients": "食材",      # 带用量的原始食材
    "tags":        "标签",      # 仅当前菜谱语言的展示标签
    "image_url":   "图片地址",   # 图片 URL（必填）
    "lang":        "lang",      # 语言标记 zh/en
}

OPTIONAL_COLUMN_MAP = {
    "facets": "标签标准编码",     # 语言无关 canonical code
    "nutrition": "营养成分",      # 每 100g 估算营养
    "description": "菜谱描述",
    "estimated_time": "预计用时", # 分钟
    "difficulty": "难易程度",
    "tips": "小贴士",
    "seasonings": "调料",
    "steps": "烹饪步骤",
    "device_code": "食谱编号",   # Mock Device 设备执行用编号
}

COLUMN_MAP = {**REQUIRED_COLUMN_MAP, **OPTIONAL_COLUMN_MAP}

_TAG_SPLIT_RE = re.compile(r"[、,，;；\|\r\n]+")
_ITEM_SPLIT_RE = re.compile(r"[\r\n;；\|]+")
_VALID_LANGS = {"zh", "en"}
_BUSINESS_KEY_FIELDS = ("recipe_id", "lang")


def _split_multi(val, *, tags=False):
    """拆分多值字段。

    食材不能按逗号或斜杠拆分，否则 ``1/2 tsp``、``salt, to taste`` 会被破坏；
    标签则按受控标签常用分隔符拆分。
    """
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return []
    splitter = _TAG_SPLIT_RE if tags else _ITEM_SPLIT_RE
    return [p.strip() for p in splitter.split(s) if p.strip()]


def _json_object(value):
    if isinstance(value, dict):
        return value
    if value is None or not str(value).strip():
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value):
    if isinstance(value, list):
        return value
    if value is None or not str(value).strip():
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return _split_multi(value)
    return parsed if isinstance(parsed, list) else []


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _build_recipe_detail(rec):
    """把标准化 Excel 字段转成稳定的展示详情，不生成任何设备指令。"""
    from app.orchestrator.ingredient_quantities import build_grounded_ingredients

    main_ingredients, main_unresolved = build_grounded_ingredients(rec["ingredients_raw"])
    seasonings, seasoning_unresolved = build_grounded_ingredients(rec["seasonings"])
    for item in main_ingredients:
        item["group"] = "main"
    for item in seasonings:
        item["group"] = "seasoning"
    ingredients = [
        *main_ingredients,
        *({
            "group": "main",
            "name": name,
            "amount": None,
            "unit": None,
            "remark": None,
            "quantity_source": "source_without_quantity",
        } for name in main_unresolved),
    ]
    # 「调料」通常是「食材」的子集；只补充食材列里不存在的项，避免详情重复。
    seen_names = {str(item.get("name") or "").strip().casefold() for item in ingredients}
    for item in seasonings:
        key = str(item.get("name") or "").strip().casefold()
        if key and key not in seen_names:
            ingredients.append(item)
            seen_names.add(key)
    for name in seasoning_unresolved:
        key = name.casefold()
        if key and key not in seen_names:
            ingredients.append({
                "group": "seasoning",
                "name": name,
                "amount": None,
                "unit": None,
                "remark": None,
                "quantity_source": "source_without_quantity",
            })
            seen_names.add(key)
    steps = []
    for raw in rec["steps"]:
        description = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not description:
            continue
        position = len(steps) + 1
        steps.append({
            "number": position,
            "type": "manual",
            "description": description,
            "image_url": None,
            "video_url": None,
            "duration_seconds": None,
            "parameters": [],
        })
    cooking_minutes = _number(rec.get("estimated_time"))
    nutrition = rec.get("nutrition") if isinstance(rec.get("nutrition"), dict) else {}
    calorie = _number(nutrition.get("calorie_kcal"))
    return {
        "schema_version": "recipe_detail_v1",
        "recipe_id": rec["recipe_id"],
        "cookId": rec.get("device_code") or rec["recipe_id"],
        "language": rec["lang"],
        "name": rec["name"],
        "introduction": rec.get("description") or None,
        "tips": rec.get("tips") or None,
        "media": {
            "landscape_image_url": rec.get("image_url") or None,
            "portrait_image_url": None,
            "intro_video_url": None,
        },
        "tags": rec["tags"],
        "ingredients": ingredients,
        "steps": steps,
        "cooking_time_seconds": (
            int(float(cooking_minutes) * 60) if cooking_minutes is not None else None
        ),
        "servings": None,
        "challenge_level": rec.get("difficulty") or None,
        "calorie_number": calorie,
        "category_ids": [],
        "accessory_ids": [],
        "device_model_ids": [],
        "is_custom_food": False,
        "executable": False,
        "source_created_at": None,
        "source_updated_at": None,
    }


def read_rows(path, sheet=None):
    """读 Excel → (表头list, 行dict列表)"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return [], []
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    records = []
    for r in rows[1:]:
        if r is None or all(c is None for c in r):
            continue
        records.append(dict(zip(header, r)))
    return header, records


def to_recipe(row):
    """按 COLUMN_MAP 把一行转成标准菜谱 dict；无名称的行跳过。"""
    def g(key):
        col = COLUMN_MAP.get(key)
        return row.get(col) if col else None

    name = g("name")
    if name is None or not str(name).strip():
        return None
    ingredients_raw = _split_multi(g("ingredients"))
    seasonings = _split_multi(g("seasonings"))
    rec = {
        "recipe_id": str(g("recipe_id") or "").strip(),
        "name": str(name).strip(),
        "ingredients": ingredients_raw,
        "ingredients_raw": ingredients_raw,
        "seasonings": seasonings,
        "tags": _split_multi(g("tags"), tags=True),
        "facets": _json_object(g("facets")),
        "nutrition": _json_object(g("nutrition")),
        "image_url": str(g("image_url") or "").strip(),
        "description": str(g("description") or "").strip(),
        "lang": str(g("lang") or "").strip().lower(),
        "estimated_time": _number(g("estimated_time")),
        "difficulty": str(g("difficulty") or "").strip(),
        "tips": str(g("tips") or "").strip(),
        "steps": _json_list(g("steps")),
        "device_code": _number(g("device_code")),
    }
    rec["recipe_detail"] = _build_recipe_detail(rec)
    return rec


def build_embed_text(rec):
    """拼向量文本：本地化内容为主，追加 canonical code 以统一跨语言检索。"""
    parts = [rec["name"]]
    if rec["tags"]:
        parts.append(" ".join(rec["tags"]))
    if rec["ingredients"]:
        parts.append(" ".join(rec["ingredients"]))
    if rec["description"]:
        parts.append(rec["description"])
    facets = rec.get("facets") or {}
    if isinstance(facets, dict):
        canonical = [
            str(value)
            for values in facets.values()
            if isinstance(values, list)
            for value in values
            if str(value).strip()
        ]
        if canonical:
            parts.append(" ".join(canonical))
    return " ".join(parts)


def _valid_url(value):
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _business_key(rec):
    recipe_id = str(rec.get("recipe_id") or "").strip()
    lang = str(rec.get("lang") or "").strip().lower()
    if not recipe_id or lang not in _VALID_LANGS:
        return None
    return recipe_id, lang


def _business_key_text(rec):
    key = _business_key(rec)
    return "" if key is None else f"{key[0]}:{key[1]}"


_PLACEHOLDER_INGREDIENT_RE = re.compile(
    r"^(?:food|ingredient|item|材料|食材|配料)\s*\d+$",
    re.IGNORECASE,
)
_PLACEHOLDER_NAME_RE = re.compile(
    r"^(?:chefyap|recipe|dish|menu)\s*\d+$",
    re.IGNORECASE,
)


def _is_placeholder_recipe(rec: dict) -> bool:
    """检测占位/测试数据：菜名或食材匹配 food1/ingredient2/chefyap0430 等模式。"""
    name = str(rec.get("name") or "").strip()
    if name and _PLACEHOLDER_NAME_RE.match(name):
        return True
    ingredients = rec.get("ingredients")
    if isinstance(ingredients, str):
        items = [i.strip() for i in ingredients.replace(";", ",").split(",") if i.strip()]
    elif isinstance(ingredients, (list, tuple)):
        items = [str(i).strip() for i in ingredients if str(i).strip()]
    else:
        items = []
    if items:
        placeholder_count = sum(1 for i in items if _PLACEHOLDER_INGREDIENT_RE.match(i))
        if placeholder_count >= len(items) * 0.5 and placeholder_count >= 2:
            return True
    return False


def validate_recipes_for_ingest(recipes):
    """底层防线：直接调用 ingest_local.py 时也拒绝写入脏数据。"""
    seen_business_keys = set()
    invalid = []
    for idx, rec in enumerate(recipes, start=2):
        reasons = []
        recipe_id = rec.get("recipe_id", "")
        if not recipe_id:
            reasons.append("missing_recipe_id")

        if not rec.get("ingredients"):
            reasons.append("missing_ingredients")
        if not _valid_url(rec.get("image_url")):
            reasons.append("invalid_image_url")
        lang = str(rec.get("lang") or "").strip().lower()
        if lang not in _VALID_LANGS:
            reasons.append("invalid_lang")
        if _is_placeholder_recipe(rec):
            reasons.append("placeholder_data")
        business_key = _business_key(rec)
        if business_key:
            if business_key in seen_business_keys:
                reasons.append("duplicate_business_key")
            seen_business_keys.add(business_key)
        if reasons:
            invalid.append((idx, rec, reasons))
    return invalid


def ensure_collection(client, name, dim, recreate):
    """建集合（id + embedding + metadata(JSON)，AUTOINDEX + COSINE）。"""
    exists = name in client.list_collections()
    if exists and recreate:
        client.drop_collection(name)
        logger.info(f"已删除旧集合：{name}")
        exists = False
    if exists:
        logger.info(f"集合 {name} 已存在，将追加写入（需重建请加 --recreate）")
        return

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("metadata", DataType.JSON)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    logger.info(f"已创建集合：{name} (dim={dim}, metric=COSINE)")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    path = args[0]
    inspect = "--inspect" in args
    recreate = "--recreate" in args

    if not os.path.exists(path):
        print(f"❌ 找不到文件：{path}")
        sys.exit(1)

    header, rows = read_rows(path)

    # ── --inspect：只看列名，不写库 ──
    if inspect:
        print("📋 Excel 列名：", header)
        if rows:
            print("📄 首行示例：", json.dumps(rows[0], ensure_ascii=False, default=str))
        print(f"📊 数据行数：{len(rows)}")
        print("\n👉 请据此核对/修改脚本顶部 COLUMN_MAP，再去掉 --inspect 正式灌库。")
        return

    # ── 校验 COLUMN_MAP 是否命中真实列 ──
    missing = [c for c in REQUIRED_COLUMN_MAP.values() if c not in header]
    if missing:
        print(f"❌ COLUMN_MAP 里这些列在 Excel 中不存在：{missing}")
        print(f"   Excel 实际列名：{header}")
        print("   请先用 --inspect 看列名并修正脚本顶部 COLUMN_MAP。")
        sys.exit(1)

    recipes = [r for r in (to_recipe(x) for x in rows) if r]
    logger.info(f"解析到 {len(recipes)} 条有效菜谱")
    if not recipes:
        print("❌ 没有有效数据（检查 name 列是否为空）")
        sys.exit(1)
    invalid = validate_recipes_for_ingest(recipes)
    if invalid:
        print(f"❌ Excel 含 {len(invalid)} 条不合规菜谱，已停止入库。业务主键：{' + '.join(_BUSINESS_KEY_FIELDS)}。请先运行 validate_ingest_excel.py 生成清洗副本。")
        for row_no, rec, reasons in invalid[:10]:
            print(json.dumps({
                "approx_excel_row": row_no,
                "business_key": _business_key_text(rec),
                "id": rec.get("recipe_id"),
                "name": rec.get("name"),
                "lang": rec.get("lang"),
                "reasons": reasons,
                "ingredients": rec.get("ingredients"),
                "image_url": rec.get("image_url"),
            }, ensure_ascii=False, default=str))
        sys.exit(1)

    # ── 向量化（百炼 text-embedding-v4）──
    if not EMBEDDING_CONFIG.get("api_key"):
        print("❌ 未读到 DASHSCOPE_API_KEY，请先在项目根 .env 填入真实 Key。")
        sys.exit(1)

    model = get_embedding_model(
        api_key=EMBEDDING_CONFIG.get("api_key"),
        base_url=EMBEDDING_CONFIG.get("base_url"),
        model=EMBEDDING_CONFIG.get("model"),
        timeout=EMBEDDING_CONFIG.get("timeout", 30),
    )
    texts = [build_embed_text(r) for r in recipes]
    logger.info("正在向量化 ...")
    vectors = model.encode(texts)            # np.ndarray [n, dim]
    dim = int(vectors.shape[1])
    logger.info(f"向量化完成，维度 {dim}")

    # ── 写入本地 Milvus Lite ──
    uri = MILVUS_CONFIG["uri"]
    name = MILVUS_CONFIG["collection_name"]
    client = build_milvus_client(uri)
    ensure_collection(client, name, dim, recreate)

    data = [{"embedding": vectors[i].tolist(), "metadata": recipes[i]} for i in range(len(recipes))]
    client.insert(collection_name=name, data=data)
    client.close()

    logger.info(f"✅ 已写入 {len(data)} 条到集合 [{name}] @ {safe_milvus_uri(uri)}")
    print(f"\n✅ 灌库完成：{len(data)} 条 → {safe_milvus_uri(uri)}")
    print("   现在可以测试检索：")
    print('   .venv/bin/python recipe_search.py "红烧肉" 3')


if __name__ == "__main__":
    main()
