"""公共联网搜索服务。

QQ、微信、WhatsApp 与 Web 都从 orchestrator 层调用这里，通道适配器只负责渲染。
实时事实查询使用百炼 DashScope 原生接口，并在后台确认搜索是否执行及来源是否有效；
用户侧只返回答案正文，不展示来源、链接或引用编号。
菜谱搜索不经过这里，仍由 Milvus RAG 独占。
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from app.core.config import settings
from app.observability.trace import (
    observe_model_call,
    record_timeout,
    record_tool_call,
)

_EXPLICIT_SEARCH_MARKERS = (
    "联网搜索", "联网搜", "上网搜索", "上网搜", "网上查", "搜索一下", "帮我查",
    "查一下", "核实一下", "web search", "search online", "look it up", "look up",
)

_FRESHNESS_MARKERS = (
    "今天", "今日", "明天", "后天", "昨天", "现在", "目前", "当前", "刚刚",
    "最近", "近期", "最新", "实时", "今年", "本周", "下周", "本月", "下月",
    "下一次", "下一个", "下个", "即将", "何时", "什么时候", "几号",
    "today", "tomorrow", "yesterday", "now", "current", "currently", "latest",
    "recent", "recently", "this year", "this week", "next week", "next",
)

_TIME_SENSITIVE_TOPICS = (
    "天气", "气温", "台风", "空气质量", "节日", "节气", "节假日", "法定假日",
    "法定节假日", "假期", "放假", "调休",
    "新闻", "热点", "价格", "股价", "股票", "基金", "汇率", "油价", "金价",
    "政策", "法规", "法律", "规定", "标准", "赛程", "比赛", "比分", "排名",
    "发布", "版本", "更新", "航班", "车次", "演出", "票房", "展会", "活动",
    "选举", "总统", "总理", "CEO", "ceo", "负责人",
    "weather", "forecast", "holiday", "festival", "news", "price", "exchange rate",
    "stock", "policy", "law", "regulation", "schedule", "score", "ranking", "release",
    "version", "flight", "train", "election", "president",
)


@dataclass(frozen=True)
class WebSearchSource:
    index: int
    title: str
    url: str
    site_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "url": self.url,
            "site_name": self.site_name,
        }


@dataclass(frozen=True)
class WebSearchResult:
    success: bool
    answer: str = ""
    sources: tuple[WebSearchSource, ...] = ()
    searched_at: str = ""
    request_id: str = ""
    error: str = ""
    cache_hit: bool = False

    def to_data(self) -> dict[str, Any]:
        """返回通道安全元数据；来源只用于后台核验，不下发给用户。"""
        return {
            "searched_at": self.searched_at,
            "verified": self.success,
            "cache_hit": self.cache_hit,
            **({"request_id": self.request_id} if self.request_id else {}),
            **({"error": self.error} if self.error else {}),
        }


_CACHE: dict[str, tuple[float, WebSearchResult]] = {}
_REFERENCE_MARKER_RE = re.compile(
    r"\s*(?:\[\s*ref[_\s-]*\d+\s*\]|\(\s*ref[_\s-]*\d+\s*\)|\bref_\d+\b)",
    flags=re.IGNORECASE,
)


def _strip_reference_markers(answer: str) -> str:
    """移除模型偶发返回的引用编号；来源元数据仍可用于后台校验。"""
    return _REFERENCE_MARKER_RE.sub("", str(answer or "")).strip()


def clear_web_search_cache() -> None:
    """清空进程内联网结果缓存，主要用于测试与运维诊断。"""
    _CACHE.clear()


def should_search_web(question: str) -> bool:
    """确定性判断普通问答是否需要实时核验。

    显式要求联网时直接命中；否则必须同时出现时效词和时效主题，避免把普通闲聊
    大面积导向付费搜索。业务路由会先处理菜谱/设备意图，因此不会替代 Milvus RAG。
    """
    normalized = " ".join(str(question or "").strip().lower().split())
    if not normalized:
        return False
    if any(marker in normalized for marker in _EXPLICIT_SEARCH_MARKERS):
        return True
    return (
        any(marker in normalized for marker in _FRESHNESS_MARKERS)
        and any(topic.lower() in normalized for topic in _TIME_SENSITIVE_TOPICS)
    )


def _now() -> datetime:
    try:
        zone = ZoneInfo(settings.WEB_SEARCH_TIMEZONE)
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("Asia/Shanghai")
    return datetime.now(zone)


def _cache_key(question: str, lang: str, *, mode: str = "verified") -> str:
    normalized = " ".join(str(question or "").strip().lower().split())
    return f"{mode}:{lang}:{normalized}"


def _get_cached(key: str) -> WebSearchResult | None:
    cached = _CACHE.get(key)
    if not cached:
        return None
    stored_at, result = cached
    if time.monotonic() - stored_at > settings.WEB_SEARCH_CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return replace(result, cache_hit=True)


def _put_cached(key: str, result: WebSearchResult) -> None:
    if not result.success or settings.WEB_SEARCH_CACHE_TTL_SECONDS <= 0:
        return
    if len(_CACHE) >= 128:
        oldest_key = min(_CACHE, key=lambda item: _CACHE[item][0])
        _CACHE.pop(oldest_key, None)
    _CACHE[key] = (time.monotonic(), result)


def _system_prompt(lang: str, now: datetime) -> str:
    language = "Answer in English." if lang == "en" else "始终使用中文回答。"
    return (
        "你是 CookClaw 的实时信息核验助手。只根据本次联网搜索得到的信息回答，"
        "不能用记忆猜测实时事实。若来源冲突或信息不足，要直接说明。\n"
        f"{language}\n"
        f"当前服务器时间：{now.isoformat()}，时区：{settings.WEB_SEARCH_TIMEZONE}。\n"
        "涉及相对日期（今天、下一次、今年等）时必须换算成明确年月日。"
        "涉及节日或假期时，应区分传统节日与法定节假日；中文提问未指定地区时，"
        "默认按中国大陆解释并明确这个假设。回答简洁，不编造来源。"
        "最终答案不要输出来源列表、链接、ref 编号、引用标记或核验时间。"
    )


def _build_payload(question: str, lang: str, now: datetime) -> dict[str, Any]:
    return {
        "model": settings.WEB_SEARCH_MODEL,
        "input": {
            "messages": [
                {"role": "system", "content": _system_prompt(lang, now)},
                {"role": "user", "content": question},
            ]
        },
        "parameters": {
            "enable_search": True,
            "result_format": "message",
            "search_options": {
                "forced_search": True,
                "search_strategy": settings.WEB_SEARCH_STRATEGY,
                "enable_source": True,
                # 来源只用于服务端验证搜索确实有依据，不让模型在答案中生成 ref 标记。
                "enable_citation": False,
            },
        },
    }


def _recipe_system_prompt(lang: str) -> str:
    language = "Answer in concise English." if lang == "en" else "使用简洁中文回答。"
    return (
        "你是 CookClaw 的网络菜谱整理助手。根据联网搜索结果直接整理一道可读的"
        "家庭做法，只输出：食材、调料、编号步骤和大致耗时。不要列来源、链接、"
        "引用编号、核验时间、背景介绍或设备操作建议。信息冲突时采用常见家常做法，"
        f"不要声称已经实际制作或验证。{language}"
    )


def _build_recipe_payload(question: str, lang: str) -> dict[str, Any]:
    """菜谱参考只需要可读步骤，不承担实时事实核验与来源展示。"""
    return {
        "model": settings.WEB_SEARCH_MODEL,
        "input": {
            "messages": [
                {"role": "system", "content": _recipe_system_prompt(lang)},
                {"role": "user", "content": question},
            ]
        },
        "parameters": {
            "enable_search": True,
            "result_format": "message",
            # 限制输出长度并关闭来源生成，减少搜索后整理和传输耗时。
            "max_tokens": 650,
            "search_options": {
                "forced_search": True,
                "search_strategy": settings.WEB_SEARCH_STRATEGY,
                "enable_source": False,
                "enable_citation": False,
            },
        },
    }


async def _request_dashscope(payload: dict[str, Any]) -> dict[str, Any]:
    url = (
        f"{settings.DASHSCOPE_NATIVE_BASE_URL.rstrip('/')}"
        "/services/aigc/text-generation/generation"
    )
    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=settings.WEB_SEARCH_TIMEOUT_SECONDS) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, dict):
        raise ValueError("DashScope 联网搜索返回了非对象响应")
    return data


def _valid_source_url(url: str) -> bool:
    parsed = urlparse(str(url or ""))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _parse_sources(raw_sources: Any, answer: str) -> tuple[WebSearchSource, ...]:
    if not isinstance(raw_sources, list):
        return ()
    all_sources: list[WebSearchSource] = []
    seen_urls: set[str] = set()
    for position, item in enumerate(raw_sources, 1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not _valid_source_url(url) or url in seen_urls:
            continue
        seen_urls.add(url)
        try:
            index = int(item.get("index") or position)
        except (TypeError, ValueError):
            index = position
        all_sources.append(WebSearchSource(
            index=index,
            title=str(item.get("title") or item.get("site_name") or url).strip(),
            url=url,
            site_name=str(item.get("site_name") or "").strip(),
        ))

    max_sources = max(1, settings.WEB_SEARCH_MAX_SOURCES)
    referenced = {int(value) for value in re.findall(r"\[ref_(\d+)\]", answer)}
    selected = [source for source in all_sources if source.index in referenced]
    for source in all_sources:
        if len(selected) >= max_sources:
            break
        if source not in selected:
            selected.append(source)
    return tuple(selected)


def _parse_response(
    data: dict[str, Any],
    searched_at: str,
    *,
    require_sources: bool = True,
) -> WebSearchResult:
    output = data.get("output") if isinstance(data.get("output"), dict) else {}
    choices = output.get("choices") if isinstance(output.get("choices"), list) else []
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    raw_answer = str(message.get("content") or "").strip() if isinstance(message, dict) else ""
    answer = _strip_reference_markers(raw_answer)
    search_info = output.get("search_info") if isinstance(output.get("search_info"), dict) else {}
    sources = _parse_sources(search_info.get("search_results"), raw_answer)
    request_id = str(data.get("request_id") or "")

    if not answer:
        return WebSearchResult(
            success=False,
            searched_at=searched_at,
            request_id=request_id,
            error="empty_answer",
        )
    if require_sources and not sources:
        return WebSearchResult(
            success=False,
            searched_at=searched_at,
            request_id=request_id,
            error="no_verifiable_sources",
        )
    return WebSearchResult(
        success=True,
        answer=answer,
        sources=sources if require_sources else (),
        searched_at=searched_at,
        request_id=request_id,
    )


async def search_web(question: str, lang: str = "zh") -> WebSearchResult:
    """强制联网搜索；后台校验来源，用户展示层只返回答案正文。"""
    started = time.monotonic()
    now = _now()
    searched_at = now.isoformat()
    if not settings.WEB_SEARCH_ENABLED:
        result = WebSearchResult(
            success=False,
            searched_at=searched_at,
            error="disabled",
        )
        record_tool_call(
            "web_search",
            duration_ms=(time.monotonic() - started) * 1000,
            success=False,
            kind="search",
            error_type="disabled",
            backend="config",
        )
        return result
    if not settings.DASHSCOPE_API_KEY:
        result = WebSearchResult(
            success=False,
            searched_at=searched_at,
            error="missing_api_key",
        )
        record_tool_call(
            "web_search",
            duration_ms=(time.monotonic() - started) * 1000,
            success=False,
            kind="search",
            error_type="missing_api_key",
            backend="config",
        )
        return result

    key = _cache_key(question, lang)
    cached = _get_cached(key)
    if cached:
        record_tool_call(
            "web_search",
            duration_ms=(time.monotonic() - started) * 1000,
            success=True,
            kind="search",
            backend="cache",
        )
        return cached

    try:
        data = await observe_model_call(
            lambda: _request_dashscope(_build_payload(question, lang, now)),
            stage="web_search_model",
            model=settings.WEB_SEARCH_MODEL,
            timeout_seconds=settings.WEB_SEARCH_TIMEOUT_SECONDS,
            reply_generation=True,
        )
        result = _parse_response(data, searched_at)
    except asyncio.TimeoutError:
        result = WebSearchResult(success=False, searched_at=searched_at, error="timeout")
    except httpx.TimeoutException:
        record_timeout("web_search_model", error_type="HTTPTimeout")
        result = WebSearchResult(success=False, searched_at=searched_at, error="timeout")
    except httpx.HTTPStatusError as exc:
        result = WebSearchResult(
            success=False,
            searched_at=searched_at,
            error=f"http_{exc.response.status_code}",
        )
    except Exception as exc:
        result = WebSearchResult(
            success=False,
            searched_at=searched_at,
            error=f"request_failed:{type(exc).__name__}",
        )

    _put_cached(key, result)
    record_tool_call(
        "web_search",
        duration_ms=(time.monotonic() - started) * 1000,
        success=result.success,
        kind="search",
        error_type=result.error or None,
        backend="dashscope",
    )
    return result


async def search_recipe_web(question: str, lang: str = "zh") -> WebSearchResult:
    """快速返回网络菜谱参考；不请求、不校验也不保存展示来源。"""
    started = time.monotonic()
    now = _now()
    searched_at = now.isoformat()
    if not settings.WEB_SEARCH_ENABLED:
        result = WebSearchResult(
            success=False,
            searched_at=searched_at,
            error="disabled",
        )
        record_tool_call(
            "recipe_web_search",
            duration_ms=(time.monotonic() - started) * 1000,
            success=False,
            kind="search",
            error_type="disabled",
            backend="config",
        )
        return result
    if not settings.DASHSCOPE_API_KEY:
        result = WebSearchResult(
            success=False,
            searched_at=searched_at,
            error="missing_api_key",
        )
        record_tool_call(
            "recipe_web_search",
            duration_ms=(time.monotonic() - started) * 1000,
            success=False,
            kind="search",
            error_type="missing_api_key",
            backend="config",
        )
        return result

    key = _cache_key(question, lang, mode="recipe_fast")
    cached = _get_cached(key)
    if cached:
        record_tool_call(
            "recipe_web_search",
            duration_ms=(time.monotonic() - started) * 1000,
            success=True,
            kind="search",
            backend="cache",
        )
        return cached

    try:
        data = await observe_model_call(
            lambda: _request_dashscope(_build_recipe_payload(question, lang)),
            stage="recipe_web_search_model",
            model=settings.WEB_SEARCH_MODEL,
            timeout_seconds=settings.RECIPE_WEB_SEARCH_TIMEOUT_SECONDS,
            reply_generation=True,
        )
        result = _parse_response(data, searched_at, require_sources=False)
    except asyncio.TimeoutError:
        result = WebSearchResult(success=False, searched_at=searched_at, error="timeout")
    except httpx.TimeoutException:
        record_timeout("recipe_web_search_model", error_type="HTTPTimeout")
        result = WebSearchResult(success=False, searched_at=searched_at, error="timeout")
    except httpx.HTTPStatusError as exc:
        result = WebSearchResult(
            success=False,
            searched_at=searched_at,
            error=f"http_{exc.response.status_code}",
        )
    except Exception as exc:
        result = WebSearchResult(
            success=False,
            searched_at=searched_at,
            error=f"request_failed:{type(exc).__name__}",
        )

    _put_cached(key, result)
    record_tool_call(
        "recipe_web_search",
        duration_ms=(time.monotonic() - started) * 1000,
        success=result.success,
        kind="search",
        error_type=result.error or None,
        backend="dashscope",
    )
    return result


def web_search_failure_text(lang: str = "zh") -> str:
    if lang == "en":
        return "Web search is temporarily unavailable, so I can't reliably verify this real-time information. Please try again shortly."
    return "联网搜索暂时不可用，我现在无法可靠核实这条实时信息。请稍后再试。"


def format_web_search_text(result: WebSearchResult, lang: str = "zh", *, markdown: bool = True) -> str:
    """Web/SSE 展示只返回答案正文，不公开后台来源或核验时间。"""
    if not result.success:
        return web_search_failure_text(lang)
    return _strip_reference_markers(result.answer)
