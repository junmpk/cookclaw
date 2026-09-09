#!/usr/bin/env python3
"""只修复 Milvus ``recipe_detail.ingredients``，不改其它详情字段。

数据优先级：
1. 经过入库校验的 Excel ``食材`` 原文（保留重量）；
2. Excel 原文确实缺用量的食材，才调用模型补 amount/unit。

修复通过 staging 集合完成，复用原有 text/dense，不重新 embedding。校验会逐条
确认 recipe_detail 除 ingredients 外完全不变，再允许原子切换生产集合。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PROJECT_ROOT / "app/agent/skills/recipe-search"
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SKILL_ROOT))
os.environ.setdefault("MILVUS_ANALYZER_TYPE", "standard")

from app.orchestrator.ingredient_quantities import (  # noqa: E402
    parse_grounded_ingredient,
)
from app.orchestrator.recipe_completion import (  # noqa: E402
    complete_ingredient_quantities,
)
from dataset.milvus import (  # noqa: E402
    build_milvus_client,
    create_hybrid_collection,
    safe_milvus_uri,
)


QUERY_LIMIT = 16_384
FALLBACK_SCHEMA = "ingredient_quantity_v2"
DEFAULT_XLSX = (
    PROJECT_ROOT
    / "tmp/recipes_merged_anonymized_enriched_for_ingest.validated.xlsx"
)
DEFAULT_CACHE = PROJECT_ROOT / "outputs/milvus_ingredient_quantity_fallback.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/milvus_ingredient_repair_report.json"


def _metadata(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _business_key(metadata: dict) -> str:
    recipe_id = str(metadata.get("recipe_id") or "").strip()
    lang = "en" if metadata.get("lang") == "en" else "zh"
    return f"{recipe_id}:{lang}" if recipe_id else ""


def _source_parts(value: Any) -> list[str]:
    # 原 Excel 以换行/分号分食材；不能按英文逗号切，否则会把
    # “avocado, peeled and sliced”拆成伪食材。
    return [
        part.strip()
        for part in re.split(r"[\r\n；;|]+", str(value or ""))
        if part.strip()
    ]


def _source_hash(parts: list[str]) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_excel_sources(path: Path) -> dict[str, dict]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    header = [str(value or "").strip() for value in next(rows)]
    positions = {name: index for index, name in enumerate(header)}
    missing = [name for name in ("id", "名称", "食材", "lang") if name not in positions]
    if missing:
        workbook.close()
        raise ValueError(f"Excel 缺少列：{missing}")

    records: dict[str, dict] = {}
    for row in rows:
        recipe_id = str(row[positions["id"]] or "").strip()
        lang = "en" if str(row[positions["lang"]] or "").strip() == "en" else "zh"
        if not recipe_id:
            continue
        key = f"{recipe_id}:{lang}"
        if key in records:
            workbook.close()
            raise ValueError(f"Excel 存在重复业务键：{key}")
        parts = _source_parts(row[positions["食材"]])
        if not parts:
            workbook.close()
            raise ValueError(f"Excel 食材为空：{key}")
        records[key] = {
            "recipe_id": recipe_id,
            "lang": lang,
            "name": str(row[positions["名称"]] or "").strip(),
            "parts": parts,
            "source_hash": _source_hash(parts),
        }
    workbook.close()
    return records


def _fetch_rows(client, collection: str) -> list[dict]:
    if collection not in client.list_collections():
        raise RuntimeError(f"集合不存在：{collection}")
    client.load_collection(collection)
    rows = client.query(
        collection_name=collection,
        filter="id >= 0",
        output_fields=["text", "dense", "metadata"],
        limit=QUERY_LIMIT,
    )
    if len(rows) >= QUERY_LIMIT:
        raise RuntimeError(f"集合达到查询上限 {QUERY_LIMIT}")
    keys = [_business_key(_metadata(row.get("metadata"))) for row in rows]
    if "" in keys or len(keys) != len(set(keys)):
        raise RuntimeError("Milvus 业务键缺失或重复")
    return sorted(rows, key=lambda row: _business_key(_metadata(row.get("metadata"))))


def _ingredient_plan(parts: list[str]) -> tuple[list[dict], list[str]]:
    ordered: list[dict] = []
    unresolved: list[str] = []
    seen_missing: set[str] = set()
    for raw in parts:
        item, missing_name = parse_grounded_ingredient(raw)
        if item:
            ordered.append({"kind": "source", "item": item})
        elif missing_name:
            folded = missing_name.casefold()
            if folded not in seen_missing:
                seen_missing.add(folded)
                unresolved.append(missing_name)
            ordered.append({
                "kind": "fallback",
                "name": missing_name,
                "source_text": raw,
            })
    if not ordered:
        raise ValueError("原始食材无法解析")
    return ordered, unresolved


def _valid_fallback_items(items: Any, names: list[str]) -> bool:
    if not isinstance(items, list) or not items:
        return False
    expected = set(names)
    covered = set()
    for item in items:
        if not isinstance(item, dict):
            return False
        name = str(item.get("name") or "").strip()
        source_name = str(item.get("fallback_for") or name).strip()
        if not name or source_name not in expected:
            return False
        amount = item.get("amount")
        unit = str(item.get("unit") or "").strip()
        if amount in (None, "", 0, "0") or not unit:
            return False
        covered.add(source_name)
    return covered == expected


def _read_cache(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(item.get("business_key") or "")
            names = item.get("names") or []
            values = item.get("items")
            if (
                key
                and item.get("fallback_schema") == FALLBACK_SCHEMA
                and item.get("source_hash")
                and _valid_fallback_items(values, names)
            ):
                records[key] = item
    return records


async def _generate_fallbacks(
    targets: list[dict],
    cache_path: Path,
    *,
    concurrency: int,
    max_attempts: int,
) -> dict[str, dict]:
    cached = _read_cache(cache_path)
    reusable = {
        target["key"]: cached[target["key"]]
        for target in targets
        if target["key"] in cached
        and cached[target["key"]].get("source_hash") == target["source_hash"]
        and cached[target["key"]].get("names") == target["unresolved"]
    }
    pending = [target for target in targets if target["key"] not in reusable]
    print(
        f"[fallback] recipes={len(targets)} cached={len(reusable)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return reusable

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    failures: list[str] = []
    done = 0
    started = time.monotonic()

    async def generate(target: dict) -> None:
        nonlocal done
        async with semaphore:
            result_items: list[dict] = []
            for attempt in range(1, max_attempts + 1):
                result_items = await complete_ingredient_quantities(
                    recipe_id=target["key"].split(":", 1)[0],
                    recipe_name=str(target["detail"].get("name") or ""),
                    ingredient_names=target["unresolved"],
                    lang=target["key"].rsplit(":", 1)[-1],
                )
                if _valid_fallback_items(result_items, target["unresolved"]):
                    break
                if attempt < max_attempts:
                    await asyncio.sleep(min(2**attempt, 5))
            if not _valid_fallback_items(result_items, target["unresolved"]):
                async with write_lock:
                    failures.append(target["key"])
                return

            for item in result_items:
                item["quantity_source"] = "llm_fallback"
            record = {
                "fallback_schema": FALLBACK_SCHEMA,
                "business_key": target["key"],
                "source_hash": target["source_hash"],
                "names": target["unresolved"],
                "items": result_items,
            }
            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            async with write_lock:
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")
                    handle.flush()
                reusable[target["key"]] = record
                done += 1
                if done == 1 or done % 20 == 0 or done == len(pending):
                    elapsed = max(time.monotonic() - started, 0.001)
                    rate = done / elapsed
                    eta = (len(pending) - done) / rate if rate else 0
                    print(
                        f"[fallback] {done}/{len(pending)} "
                        f"rate={rate:.2f}/s eta={eta/60:.1f}m",
                        flush=True,
                    )

    await asyncio.gather(*(generate(target) for target in pending))
    if failures:
        raise RuntimeError(
            f"模型仍有 {len(failures)} 道菜未完整补量：{failures[:20]}"
        )
    return reusable


def _merge_plan(plan: list[dict], fallback: dict[str, list[dict]]) -> list[dict]:
    final: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in plan:
        if entry["kind"] == "source":
            item = deepcopy(entry["item"])
        else:
            items = deepcopy(fallback.get(entry["name"]) or [])
            if not items:
                raise ValueError(f"缺少模型兜底：{entry['name']}")
            for item in items:
                item.pop("fallback_for", None)
                item["source_text"] = entry["source_text"]
                key = (
                    str(item.get("name") or "").casefold(),
                    str(item.get("amount") or ""),
                    str(item.get("unit") or "").casefold(),
                )
                if key not in seen:
                    seen.add(key)
                    final.append(item)
            continue
        key = (
            str(item.get("name") or "").casefold(),
            str(item.get("amount") or ""),
            str(item.get("unit") or "").casefold(),
        )
        if key not in seen:
            seen.add(key)
            final.append(item)
    return final


def _detail_without_ingredients(detail: dict) -> dict:
    value = deepcopy(detail)
    value.pop("ingredients", None)
    return value


def _validate_repair(
    source_rows: list[dict],
    repaired: dict[str, list[dict]],
) -> dict:
    missing = []
    invalid = []
    other_detail_changed = []
    source_count = 0
    llm_count = 0
    for row in source_rows:
        md = _metadata(row.get("metadata"))
        key = _business_key(md)
        ingredients = repaired.get(key)
        if ingredients is None:
            missing.append(key)
            continue
        if not ingredients or any(
            not str(item.get("name") or "").strip()
            or item.get("amount") in (None, "", 0, "0")
            or not str(item.get("unit") or "").strip()
            for item in ingredients
            if isinstance(item, dict)
        ) or any(not isinstance(item, dict) for item in ingredients):
            invalid.append(key)
        source_count += sum(
            item.get("quantity_source") == "ingredients_raw"
            for item in ingredients
        )
        llm_count += sum(
            item.get("quantity_source") == "llm_fallback"
            for item in ingredients
        )
        old_detail = md.get("recipe_detail") or {}
        candidate = deepcopy(old_detail)
        candidate["ingredients"] = ingredients
        if _detail_without_ingredients(candidate) != _detail_without_ingredients(old_detail):
            other_detail_changed.append(key)
    result = {
        "rows": len(source_rows),
        "missing_repairs": len(missing),
        "invalid_repairs": len(invalid),
        "other_detail_changed": len(other_detail_changed),
        "source_quantity_items": source_count,
        "llm_fallback_items": llm_count,
        "missing_preview": missing[:10],
        "invalid_preview": invalid[:10],
    }
    if missing or invalid or other_detail_changed:
        raise RuntimeError(f"修复内容校验失败：{json.dumps(result, ensure_ascii=False)}")
    return result


def _build_staging(
    client,
    source_rows: list[dict],
    repaired: dict[str, list[dict]],
    staging: str,
    *,
    recreate: bool,
    insert_batch: int,
) -> None:
    if staging in client.list_collections() and not recreate:
        raise RuntimeError(f"staging 已存在：{staging}")
    create_hybrid_collection(client, staging, 1024, recreate=recreate)
    inserted = 0
    for offset in range(0, len(source_rows), insert_batch):
        batch = []
        for row in source_rows[offset:offset + insert_batch]:
            md = deepcopy(_metadata(row.get("metadata")))
            detail = deepcopy(md.get("recipe_detail") or {})
            detail["ingredients"] = repaired[_business_key(md)]
            md["recipe_detail"] = detail
            batch.append({
                "text": row["text"],
                "dense": row["dense"],
                "metadata": md,
            })
        client.insert(collection_name=staging, data=batch)
        inserted += len(batch)
        print(f"[build] inserted={inserted}/{len(source_rows)}", flush=True)
    client.flush(staging)
    client.load_collection(staging)


def _validate_collection(
    client,
    source: str,
    staging: str,
    expected: int,
) -> dict:
    source_rows = _fetch_rows(client, source)
    staged_rows = _fetch_rows(client, staging)
    source_by_key = {
        _business_key(_metadata(row.get("metadata"))): row for row in source_rows
    }
    invalid = []
    other_changed = []
    for row in staged_rows:
        md = _metadata(row.get("metadata"))
        key = _business_key(md)
        old_md = _metadata(source_by_key.get(key, {}).get("metadata"))
        detail = md.get("recipe_detail") or {}
        ingredients = detail.get("ingredients") or []
        if not ingredients or any(
            not isinstance(item, dict)
            or item.get("amount") in (None, "", 0, "0")
            or not str(item.get("unit") or "").strip()
            for item in ingredients
        ):
            invalid.append(key)
        if _detail_without_ingredients(detail) != _detail_without_ingredients(
            old_md.get("recipe_detail") or {}
        ):
            other_changed.append(key)
    dense = client.search(
        collection_name=staging,
        data=[[0.0] * 1024],
        anns_field="dense",
        limit=1,
        output_fields=["metadata"],
        search_params={"metric_type": "COSINE"},
    )
    sparse = client.search(
        collection_name=staging,
        data=["recipe"],
        anns_field="sparse",
        limit=1,
        output_fields=["metadata"],
        search_params={"metric_type": "BM25"},
    )
    result = {
        "expected": expected,
        "source_rows": len(source_rows),
        "staged_rows": len(staged_rows),
        "invalid_ingredient_rows": len(invalid),
        "other_detail_changed": len(other_changed),
        "dense_probe_hits": len(dense[0]) if dense else 0,
        "sparse_probe_hits": len(sparse[0]) if sparse else 0,
    }
    if (
        len(source_rows) != expected
        or len(staged_rows) != expected
        or invalid
        or other_changed
        or result["dense_probe_hits"] < 1
        or result["sparse_probe_hits"] < 1
    ):
        raise RuntimeError(f"staging 校验失败：{json.dumps(result, ensure_ascii=False)}")
    return result


def _promote(client, source: str, staging: str, expected: int) -> str:
    _validate_collection(client, source, staging, expected)
    backup = f"{source}_before_ingredient_repair_{datetime.now():%Y%m%d_%H%M%S}"
    client.release_collection(source)
    client.release_collection(staging)
    client.rename_collection(old_name=source, new_name=backup)
    try:
        client.rename_collection(old_name=staging, new_name=source)
    except Exception:
        client.rename_collection(old_name=backup, new_name=source)
        raise
    client.load_collection(source)
    return backup


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("analyze", "generate", "build", "validate", "promote", "all"),
        default="analyze",
    )
    parser.add_argument("--source", default="recipe_hybrid")
    parser.add_argument("--staging", default="recipe_hybrid_ingredients_repaired")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--insert-batch", type=int, default=250)
    parser.add_argument("--recreate-staging", action="store_true")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32:
        parser.error("--concurrency 必须在 1..32")
    return args


async def main() -> int:
    args = _args()
    uri = os.getenv("RECIPE_MILVUS_URI", "").strip()
    if not uri:
        raise RuntimeError("RECIPE_MILVUS_URI 未配置")
    client = build_milvus_client(uri)
    print(f"[preflight] milvus={safe_milvus_uri(uri)} source={args.source}")
    rows = _fetch_rows(client, args.source)
    excel = _load_excel_sources(args.xlsx)
    milvus_keys = {_business_key(_metadata(row.get("metadata"))) for row in rows}
    if milvus_keys != set(excel):
        raise RuntimeError(
            f"Excel/Milvus 业务键不一致：milvus={len(milvus_keys)} excel={len(excel)}"
        )

    targets = []
    repaired: dict[str, list[dict]] = {}
    plans: dict[str, list[dict]] = {}
    source_items = 0
    unresolved_items = 0
    for row in rows:
        md = _metadata(row.get("metadata"))
        key = _business_key(md)
        plan, unresolved = _ingredient_plan(excel[key]["parts"])
        plans[key] = plan
        source_items += sum(entry["kind"] == "source" for entry in plan)
        unresolved_items += len(unresolved)
        if unresolved:
            targets.append({
                "key": key,
                "source_hash": excel[key]["source_hash"],
                "unresolved": unresolved,
                "detail": md.get("recipe_detail") or {},
            })

    analysis = {
        "rows": len(rows),
        "source_quantity_items": source_items,
        "recipes_needing_llm": len(targets),
        "llm_fallback_items": unresolved_items,
    }
    print(f"[analyze] {json.dumps(analysis, ensure_ascii=False)}")
    if args.phase == "analyze":
        client.close()
        return 0

    fallbacks = await _generate_fallbacks(
        targets,
        args.cache,
        concurrency=args.concurrency,
        max_attempts=args.max_attempts,
    )
    for row in rows:
        md = _metadata(row.get("metadata"))
        key = _business_key(md)
        fallback_items: dict[str, list[dict]] = {}
        for item in (fallbacks.get(key) or {}).get("items", []):
            source_name = str(item.get("fallback_for") or item.get("name") or "")
            fallback_items.setdefault(source_name, []).append(item)
        repaired[key] = _merge_plan(plans[key], fallback_items)
    repair_validation = _validate_repair(rows, repaired)
    print(f"[repair] {json.dumps(repair_validation, ensure_ascii=False)}")

    if args.phase in {"build", "all"}:
        _build_staging(
            client,
            rows,
            repaired,
            args.staging,
            recreate=args.recreate_staging,
            insert_batch=args.insert_batch,
        )

    collection_validation = None
    if args.phase in {"validate", "all"}:
        collection_validation = _validate_collection(
            client, args.source, args.staging, len(rows)
        )
        print(f"[validate] {json.dumps(collection_validation, ensure_ascii=False)}")

    backup = None
    if args.phase == "promote" or args.promote:
        backup = _promote(client, args.source, args.staging, len(rows))
        print(f"[promote] production={args.source} backup={backup}")

    report = {
        "finished_at": datetime.now().astimezone().isoformat(),
        "source": args.source,
        "staging": args.staging,
        "xlsx": str(args.xlsx),
        "analysis": analysis,
        "repair_validation": repair_validation,
        "collection_validation": collection_validation,
        "backup": backup,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    client.close()
    print(f"[done] report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
