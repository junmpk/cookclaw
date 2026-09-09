"""公共联网搜索与多通道路由测试，不调用真实外网。"""
import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from app.core.config import resolve_dashscope_native_base_url, settings
from app.orchestrator.intent import IntentCategory, IntentResult
from app.orchestrator.web_search import (
    WebSearchResult,
    WebSearchSource,
    clear_web_search_cache,
    format_web_search_text,
    search_recipe_web,
    search_web,
    should_search_web,
)


def test_dashscope_native_url_is_derived_in_same_region():
    assert resolve_dashscope_native_base_url(
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ) == "https://dashscope.aliyuncs.com/api/v1"
    assert resolve_dashscope_native_base_url(
        "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    ) == "https://workspace.cn-beijing.maas.aliyuncs.com/api/v1"


def test_realtime_questions_and_explicit_search_are_detected():
    assert should_search_web("下一次节日是什么时候？")
    assert should_search_web("下一个节假日是什么时候啊")
    assert should_search_web("下个法定假日是哪天？")
    assert should_search_web("今天上海天气怎么样")
    assert should_search_web("帮我联网搜索阿里云最近的新闻")
    assert should_search_web("What is the latest exchange rate?")
    assert not should_search_web("Python 装饰器是什么？")
    assert not should_search_web("今天推荐三个鸡肉菜")
    assert not should_search_web("2024 年端午节是哪天？")


def test_native_search_verifies_sources_but_omits_citations(monkeypatch):
    import app.orchestrator.web_search as web_module

    calls = []
    fixed_now = datetime(2026, 7, 20, 14, 9, tzinfo=ZoneInfo("Asia/Shanghai"))

    async def fake_request(payload):
        calls.append(payload)
        return {
            "request_id": "req-web-1",
            "output": {
                "choices": [{"message": {"content": "下一次法定假期是……[ref_2]"}}],
                "search_info": {
                    "search_results": [
                        {"index": 1, "title": "普通来源", "url": "https://example.com/a"},
                        {"index": 2, "title": "国务院通知", "url": "https://gov.example/holiday", "site_name": "gov"},
                    ]
                },
            },
        }

    async def run():
        clear_web_search_cache()
        monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(settings, "WEB_SEARCH_CACHE_TTL_SECONDS", 300)
        monkeypatch.setattr(web_module, "_now", lambda: fixed_now)
        monkeypatch.setattr(web_module, "_request_dashscope", fake_request)

        first = await search_web("下一次节日是什么时候？", "zh")
        second = await search_web("下一次节日是什么时候？", "zh")

        assert first.success is True
        assert first.searched_at == "2026-07-20T14:09:00+08:00"
        assert first.sources[0].index == 2  # 被正文引用的来源优先保留
        assert first.answer == "下一次法定假期是……"
        assert first.request_id == "req-web-1"
        assert second.cache_hit is True
        assert second.answer == first.answer
        assert second.sources == first.sources
        assert len(calls) == 1
        payload = calls[0]
        assert payload["parameters"]["enable_search"] is True
        assert payload["parameters"]["search_options"]["forced_search"] is True
        assert payload["parameters"]["search_options"]["enable_source"] is True
        assert payload["parameters"]["search_options"]["enable_citation"] is False
        assert "citation_format" not in payload["parameters"]["search_options"]
        assert "2026-07-20T14:09:00+08:00" in payload["input"]["messages"][0]["content"]

    asyncio.run(run())


def test_search_without_verifiable_sources_fails_closed(monkeypatch):
    import app.orchestrator.web_search as web_module

    async def fake_request(payload):
        return {
            "output": {
                "choices": [{"message": {"content": "模型给了一个没有来源的答案"}}],
                "search_info": {"search_results": []},
            }
        }

    async def run():
        clear_web_search_cache()
        monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(web_module, "_request_dashscope", fake_request)
        result = await search_web("今天的最新新闻", "zh")
        assert result.success is False
        assert result.answer == ""
        assert result.error == "no_verifiable_sources"

    asyncio.run(run())


def test_search_timeout_fails_closed(monkeypatch):
    import app.orchestrator.web_search as web_module

    async def fake_request(payload):
        raise httpx.ReadTimeout("slow")

    async def run():
        clear_web_search_cache()
        monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(web_module, "_request_dashscope", fake_request)
        result = await search_web("明天天气", "zh")
        assert result.success is False
        assert result.error == "timeout"

    asyncio.run(run())


def test_recipe_web_search_skips_sources_and_reuses_fast_cache(monkeypatch):
    import app.orchestrator.web_search as web_module

    calls = []

    async def fake_request(payload):
        calls.append(payload)
        return {
            "request_id": "recipe-fast-1",
            "output": {
                "choices": [{
                    "message": {
                        "content": "食材：西红柿、鸡蛋。[ref_1]\n步骤：1. 炒鸡蛋；2. 炒西红柿。",
                    },
                }],
                "search_info": {"search_results": []},
            },
        }

    async def run():
        clear_web_search_cache()
        monkeypatch.setattr(settings, "WEB_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(settings, "WEB_SEARCH_CACHE_TTL_SECONDS", 300)
        monkeypatch.setattr(web_module, "_request_dashscope", fake_request)

        first = await search_recipe_web("西红柿炒鸡蛋", "zh")
        second = await search_recipe_web("西红柿炒鸡蛋", "zh")

        assert first.success is True
        assert first.sources == ()
        assert "[ref_" not in first.answer
        assert second.cache_hit is True
        assert len(calls) == 1
        options = calls[0]["parameters"]["search_options"]
        assert options["forced_search"] is True
        assert options["enable_source"] is False
        assert options["enable_citation"] is False
        assert calls[0]["parameters"]["max_tokens"] == 650

    asyncio.run(run())


def test_recipe_web_reference_output_omits_verification_metadata():
    from app.agent.participle_agent import (
        _recipe_web_reference_msg,
        _recipe_web_reference_text,
    )

    result = WebSearchResult(
        success=True,
        answer="食材：西红柿、鸡蛋。\n步骤：先炒蛋，再炒西红柿。",
        searched_at="2026-07-25T18:00:00+08:00",
        request_id="hidden-request-id",
    )
    text = _recipe_web_reference_text(result, "zh", markdown=True)
    payload = json.loads(_recipe_web_reference_msg(result, "zh"))

    assert "食材：西红柿、鸡蛋" in text
    assert "核验来源" not in text
    assert "核验时间" not in text
    assert "hidden-request-id" not in json.dumps(payload, ensure_ascii=False)
    assert "sources" not in payload["data"]
    assert "verified" not in payload["data"]
    assert payload["data"]["device_eligible"] is False


def test_router_returns_one_shared_web_search_outcome(monkeypatch):
    import app.orchestrator.router as router_module

    result = WebSearchResult(
        success=True,
        answer="核验后的统一答案[ref_1]",
        sources=(WebSearchSource(1, "权威来源", "https://example.com/source"),),
        searched_at="2026-07-20T14:09:00+08:00",
    )

    async def fake_classify(question):
        return IntentResult(related=False, category=IntentCategory.off_topic)

    async def fake_search(question, lang):
        assert question == "下一次节日是什么时候？"
        assert lang == "zh"
        return result

    async def run():
        monkeypatch.setattr(router_module, "classify_intent", fake_classify)
        monkeypatch.setattr(router_module, "search_web", fake_search)
        outcome = await router_module.route_fast_path("下一次节日是什么时候？")
        assert outcome.kind == "web_search"
        assert outcome.web_search_result is result

    asyncio.run(run())


def test_all_channels_render_answer_without_sources_or_reference_markers():
    from app.agent.participle_agent import _web_search_msg
    from app.main import _json_to_markdown, _json_to_plaintext

    result = WebSearchResult(
        success=True,
        answer="核验后的统一答案[ref_1]",
        sources=(WebSearchSource(1, "权威来源", "https://example.com/source"),),
        searched_at="2026-07-20T14:09:00+08:00",
    )
    common = json.loads(_web_search_msg(result, "zh"))
    qq_text = _json_to_markdown(common)
    weixin_text = _json_to_plaintext(common)

    assert "核验后的统一答案" in qq_text
    assert "核验后的统一答案" in weixin_text
    assert "ref_" not in qq_text
    assert "ref_" not in weixin_text
    assert "https://example.com/source" not in qq_text
    assert "https://example.com/source" not in weixin_text
    assert "核验来源" not in qq_text
    assert "核验来源" not in weixin_text
    assert "sources" not in common["data"]
    assert common["data"]["verified"] is True


def test_web_search_text_and_im_payload_strip_synthetic_reference_markers():
    from app.agent.participle_agent import _web_search_msg
    from app.main import _json_to_plaintext

    result = WebSearchResult(
        success=True,
        answer="已核验[ref_1]",
        sources=(WebSearchSource(1, "来源", "https://example.com/source_file"),),
        searched_at="2026-07-20T14:09:00+08:00",
    )
    common = json.loads(_web_search_msg(result, "zh"))
    assert format_web_search_text(result, "zh", markdown=True) == "已核验"
    assert common["message"] == "已核验"
    assert "sources" not in common["data"]
    assert _json_to_plaintext(common, encode_filter_urls=True) == "已核验"
    assert _json_to_plaintext(common, encode_filter_urls=False) == "已核验"
