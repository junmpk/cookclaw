#!/usr/bin/env python3
"""并发校验菜谱工作簿中的图片 URL，并输出可交接的异常清单 JSON。"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx


_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_IMAGE_MAGIC = (
    b"\xff\xd8\xff",  # JPEG
    b"\x89PNG\r\n\x1a\n",
    b"GIF87a",
    b"GIF89a",
    b"RIFF",  # WebP 在 8-12 字节另行判断
    b"BM",
)


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if not letters:
        return -1
    value = 0
    for char in letters.group():
        value = value * 26 + ord(char) - 64
    return value - 1


def _cell_value(cell: ET.Element) -> str:
    inline = cell.find("m:is/m:t", _NS)
    value = cell.find("m:v", _NS)
    return (
        inline.text
        if inline is not None
        else value.text
        if value is not None
        else ""
    ) or ""


def read_recipe_refs(
    path: Path,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    """只读解析首个工作表，按图片 URL 聚合菜谱引用。"""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = root.findall(".//m:sheetData/m:row", _NS)
    if not rows:
        return {}, []

    headers: list[str] = []
    for cell in rows[0].findall("m:c", _NS):
        index = _column_index(cell.get("r", ""))
        if index < 0:
            continue
        headers.extend([""] * (index - len(headers) + 1))
        headers[index] = _cell_value(cell).strip()

    by_url: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_urls: list[dict[str, str]] = []
    for row in rows[1:]:
        record = {header: "" for header in headers}
        for cell in row.findall("m:c", _NS):
            index = _column_index(cell.get("r", ""))
            if 0 <= index < len(headers):
                record[headers[index]] = _cell_value(cell).strip()
        recipe_ref = {
            "id": record.get("id", ""),
            "name": record.get("名称", ""),
            "lang": record.get("lang", ""),
        }
        url = record.get("图片地址", "")
        if url:
            by_url[url].append(recipe_ref)
        else:
            missing_urls.append(recipe_ref)
    return dict(by_url), missing_urls


def _looks_like_image(first_chunk: bytes, content_type: str) -> bool:
    if content_type.startswith("image/"):
        return True
    if first_chunk.startswith(b"RIFF") and first_chunk[8:12] == b"WEBP":
        return True
    return any(first_chunk.startswith(magic) for magic in _IMAGE_MAGIC)


async def check_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    refs: list[dict[str, str]],
) -> dict:
    started = time.monotonic()
    result = {
        "url": url,
        "host": urlparse(url).netloc,
        "ok": False,
        "status_code": None,
        "content_type": "",
        "final_url": "",
        "error": "",
        "elapsed_ms": 0,
        "references": refs,
    }
    try:
        async with semaphore:
            async with client.stream(
                "GET",
                url,
                headers={
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                    "Range": "bytes=0-4095",
                    "User-Agent": "Mozilla/5.0 CookClawImageAudit/1.0",
                },
            ) as response:
                result["status_code"] = response.status_code
                result["content_type"] = (
                    response.headers.get("content-type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                result["final_url"] = str(response.url)
                first_chunk = b""
                async for chunk in response.aiter_bytes():
                    first_chunk += chunk
                    if len(first_chunk) >= 32:
                        break
                if response.status_code >= 400:
                    result["error"] = f"HTTP {response.status_code}"
                elif not first_chunk:
                    result["error"] = "响应正文为空"
                elif not _looks_like_image(first_chunk, result["content_type"]):
                    result["error"] = (
                        f"响应不是图片（Content-Type={result['content_type'] or 'missing'}）"
                    )
                else:
                    result["ok"] = True
    except httpx.TimeoutException:
        result["error"] = "请求超时"
    except httpx.HTTPError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - 诊断脚本保留具体错误类型
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
    return result


async def run(args: argparse.Namespace) -> int:
    refs, missing_urls = read_recipe_refs(args.input)
    urls = sorted(refs)
    print(f"IMAGE_URL_AUDIT total_refs={sum(map(len, refs.values()))} unique_urls={len(urls)}")
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=min(args.concurrency, 30),
    )
    timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 10.0))
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        limits=limits,
        trust_env=True,
    ) as client:
        tasks = [
            asyncio.create_task(check_one(client, semaphore, url, refs[url]))
            for url in urls
        ]
        results = []
        completed = 0
        for task in asyncio.as_completed(tasks):
            results.append(await task)
            completed += 1
            if completed % 250 == 0 or completed == len(tasks):
                failures = sum(not item["ok"] for item in results)
                print(
                    f"IMAGE_URL_AUDIT progress={completed}/{len(tasks)} "
                    f"failures={failures}",
                    flush=True,
                )
        # 高并发下个别图片源可能偶发超时。异常项再低并发复检两轮，只有持续
        # 失败才进入交接清单，避免把正常图片误报给数据同事。
        for retry_round in range(1, 3):
            failed_urls = [item["url"] for item in results if not item["ok"]]
            if not failed_urls:
                break
            print(
                f"IMAGE_URL_AUDIT retry_round={retry_round} "
                f"candidates={len(failed_urls)}",
                flush=True,
            )
            retry_semaphore = asyncio.Semaphore(5)
            retry_results = await asyncio.gather(*[
                check_one(client, retry_semaphore, url, refs[url])
                for url in failed_urls
            ])
            replacements = {item["url"]: item for item in retry_results}
            results = [
                replacements.get(item["url"], item)
                if not item["ok"] else item
                for item in results
            ]

    failures = sorted(
        (item for item in results if not item["ok"]),
        key=lambda item: (item["host"], item["status_code"] or 0, item["url"]),
    )
    payload = {
        "source_file": str(args.input),
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_references": sum(map(len, refs.values())),
        "unique_urls": len(urls),
        "healthy_urls": len(urls) - len(failures),
        "abnormal_urls": len(failures),
        "abnormal_recipe_references": sum(
            len(item["references"]) for item in failures
        ) + len(missing_urls),
        "missing_image_urls": missing_urls,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "IMAGE_URL_AUDIT complete "
        f"healthy={payload['healthy_urls']} abnormal={payload['abnormal_urls']} "
        f"missing={len(missing_urls)} "
        f"output={args.output}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
