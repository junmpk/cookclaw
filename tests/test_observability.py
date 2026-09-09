"""对话 Trace 与基线汇总的纯本地测试。"""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage

from app.observability.baseline import summarize_traces
from app.observability.health import evaluate_health_alerts
from app.observability.trace import (
    collect_turn_traces,
    current_trace,
    ensure_turn_trace,
    finish_turn_trace,
    observe_model_call,
    record_domain_result,
    record_quality_check,
    record_reply_validation,
    trace_async_tool,
)
from app.orchestrator.routing_models import RouteDecision
from scripts.summarize_conversation_traces import _load_rows


def test_turn_trace_counts_models_tokens_tools_and_quality(monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")

    async def run():
        _trace, token = ensure_turn_trace(
            channel="qq",
            thread_id="qq:dm:chat-user:private-user",
        )

        async def fake_model():
            return AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": 21,
                    "output_tokens": 7,
                    "total_tokens": 28,
                },
            )

        @trace_async_tool("device_status")
        async def fake_tool():
            return {"ok": True, "code": "OK"}

        response = await observe_model_call(
            fake_model,
            stage="intent_classifier",
            model="test-model",
            intent=True,
            reply_generation=True,
        )
        assert response.content == "ok"
        assert await fake_tool() == {"ok": True, "code": "OK"}
        record_reply_validation(checked=3, kept=2)
        record_quality_check("grounded_reply", True)
        return finish_turn_trace(token, success=True, response_type="recipe_search")

    payload = asyncio.run(run())
    assert payload is not None
    assert payload["model_call_count"] == 1
    assert payload["intent_call_count"] == 1
    assert payload["tool_call_count"] == 1
    assert payload["token_usage"] == {
        "input_tokens": 21,
        "output_tokens": 7,
        "total_tokens": 28,
    }
    assert payload["context"] == {"max_model_input_tokens": 21}
    assert payload["reply_generation_duration_ms"] >= 0
    assert payload["reply_validation"] == {"checked": 3, "kept": 2}
    assert payload["quality"]["effective"] is True
    assert payload["status"] == "success"


def test_turn_trace_context_is_isolated_between_concurrent_tasks(monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")

    async def worker(thread_id: str, delay: float):
        trace, token = ensure_turn_trace(channel="qq", thread_id=thread_id)
        await asyncio.sleep(delay)
        assert current_trace() is trace
        payload = finish_turn_trace(token, success=True)
        assert current_trace() is None
        return payload

    async def run():
        return await asyncio.gather(
            worker("qq:dm:a:a", 0.01),
            worker("qq:dm:b:b", 0),
        )

    first, second = asyncio.run(run())
    assert first["trace_id"] != second["trace_id"]
    assert first["thread_hash"] != second["thread_hash"]


def test_trace_collector_receives_only_traces_from_its_context(monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")
    with collect_turn_traces() as rows:
        _trace, token = ensure_turn_trace(channel="qq", thread_id="qq:dm:x:x")
        expected = finish_turn_trace(token, success=True)
    assert rows == [expected]


def test_trace_payload_does_not_store_thread_id_or_message(monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")
    raw_thread = "qq:dm:private-chat-id:private-user-id"
    raw_message = "我不吃花生，设备编号 kitchen-secret"
    _trace, token = ensure_turn_trace(channel="qq", thread_id=raw_thread)
    payload = finish_turn_trace(token, success=True, response_type="text")
    encoded = json.dumps(payload, ensure_ascii=False)
    assert raw_thread not in encoded
    assert raw_message not in encoded
    assert payload["thread_hash"]


def test_domain_result_is_observed_without_incrementing_external_tool_count(
    monkeypatch,
):
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")
    _trace, token = ensure_turn_trace(channel="qq", thread_id="qq:dm:x:x")
    record_domain_result(
        "profile_memory",
        duration_ms=3.5,
        success=True,
        result_code="PROFILE_SAVED",
        state_patch_count=1,
    )
    payload = finish_turn_trace(token, success=True)

    assert payload["tool_call_count"] == 0
    event = next(
        item for item in payload["events"] if item["kind"] == "domain_result"
    )
    assert event == {
        "kind": "domain_result",
        "at_ms": event["at_ms"],
        "name": "profile_memory",
        "success": True,
        "result_code": "PROFILE_SAVED",
        "duration_ms": 3.5,
        "state_patch_count": 1,
        "continuation": False,
    }


def test_model_timeout_is_counted_once(monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")

    async def run():
        _trace, token = ensure_turn_trace(channel="qq", thread_id="qq:dm:x:x")

        async def slow_model():
            await asyncio.sleep(0.2)

        with pytest.raises(asyncio.TimeoutError):
            await observe_model_call(
                slow_model,
                stage="qa",
                timeout_seconds=0.001,
            )
        return finish_turn_trace(token, success=False, error_type="TimeoutError")

    payload = asyncio.run(run())
    assert payload["model_call_count"] == 1
    assert payload["timeout_count"] == 1
    assert payload["error_type"] == "TimeoutError"


def test_route_decision_reuses_active_turn_trace_id(monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")
    trace, token = ensure_turn_trace(channel="qq", thread_id="qq:dm:x:x")
    decision = RouteDecision(
        category="recipe",
        action="recipe_search",
        source="exact_rule",
        reason_code="TEST",
    )
    assert decision.trace_id == trace.trace_id
    finish_turn_trace(token, success=True)


def test_jsonl_sink_and_log_parser(monkeypatch, tmp_path):
    path = tmp_path / "conversation-trace.jsonl"
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "true")
    monkeypatch.setenv("CONVERSATION_TRACE_JSONL_PATH", str(path))
    _trace, token = ensure_turn_trace(channel="qq", thread_id="qq:dm:x:x")
    expected = finish_turn_trace(token, success=True)

    rows = _load_rows(path)
    assert rows == [expected]

    journal_path = tmp_path / "journal.log"
    journal_path.write_text(
        "unrelated log\n"
        f"conversation_turn_trace {json.dumps(expected, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    assert _load_rows(journal_path) == [expected]

    report_path = tmp_path / "qq-report.json"
    report_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {"results": [{"trace": expected}]},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    assert _load_rows(report_path) == [expected]


def test_baseline_summary_reports_measured_values_without_invented_target():
    rows = [
        {
            "schema_version": "conversation_trace_v1",
            "status": "success",
            "total_duration_ms": 100,
            "model_call_count": 1,
            "tool_call_count": 1,
            "intent_call_count": 1,
            "search_call_count": 1,
            "timeout_count": 0,
            "tool_failure_count": 0,
            "token_usage": {"total_tokens": 50},
            "context": {"max_model_input_tokens": 40},
            "search_duration_ms": 30,
            "reply_generation_duration_ms": 50,
            "fallback_reasons": [],
            "quality": {
                "assessed_checks": 1,
                "effective": True,
            },
            "route": {"action": "recipe_search"},
            "events": [{
                "kind": "profile_memory",
                "action": "set",
                "status": "saved",
            }],
        },
        {
            "schema_version": "conversation_trace_v1",
            "status": "error",
            "error_type": "TimeoutError",
            "total_duration_ms": 300,
            "model_call_count": 3,
            "tool_call_count": 2,
            "intent_call_count": 1,
            "search_call_count": 1,
            "timeout_count": 1,
            "tool_failure_count": 1,
            "token_usage": {"total_tokens": 150},
            "context": {"max_model_input_tokens": 120},
            "search_duration_ms": 90,
            "reply_generation_duration_ms": 150,
            "fallback_reasons": ["deep_agent_timeout"],
            "quality": {
                "assessed_checks": 1,
                "effective": False,
            },
            "route": {"action": "candidate_choice"},
            "events": [{
                "kind": "profile_memory",
                "action": "change",
                "status": "write_failed",
            }],
        },
    ]

    summary = summarize_traces(rows)
    assert summary["trace_count"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["latency_ms"] == {"p50": 200.0, "p95": 290.0, "max": 300.0}
    assert summary["model_calls"]["average"] == 2.0
    assert summary["tokens"]["average"] == 100.0
    assert summary["context_tokens"]["average_max_per_turn"] == 80.0
    assert summary["search_duration_ms"]["p50"] == 60.0
    assert summary["reply_generation_duration_ms"]["p95"] == 145.0
    assert summary["timeout_rate"] == 0.5
    assert summary["tool_failure_rate"] == 0.5
    assert summary["quality"]["effective_rate"] == 0.5
    assert summary["route_actions"] == {
        "candidate_choice": 1,
        "recipe_search": 1,
    }
    assert summary["profile_memory"] == {
        "event_count": 2,
        "actions": {"change": 1, "set": 1},
        "statuses": {"saved": 1, "write_failed": 1},
        "failure_rate": 0.5,
    }
    assert "target" not in summary


def test_operational_summary_exposes_planner_handler_tool_and_safety_metrics():
    rows = [
        {
            "schema_version": "conversation_trace_v1",
            "status": "success",
            "message_type": "text",
            "total_duration_ms": 100,
            "fallback_reasons": [],
            "events": [
                {"kind": "planner_active", "status": "ready"},
                {
                    "kind": "planner_execution",
                    "status": "executed",
                    "action": "recipe.search",
                },
                {
                    "kind": "turn_orchestrator",
                    "handled_by": "planner",
                },
                {
                    "kind": "tool_call",
                    "name": "recipe_search",
                    "success": True,
                },
            ],
        },
        {
            "schema_version": "conversation_trace_v1",
            "status": "success",
            "message_type": "image",
            "total_duration_ms": 300,
            "fallback_reasons": ["PLANNER_TIMEOUT"],
            "route": {"action": "safety_block"},
            "events": [
                {"kind": "planner_active", "status": "fallback"},
                {
                    "kind": "planner_execution",
                    "status": "failed_closed",
                    "action": "device.start",
                },
                {
                    "kind": "turn_orchestrator",
                    "handled_by": "image_recipe_handler",
                },
                {
                    "kind": "tool_call",
                    "name": "device_status",
                    "success": False,
                    "state_unknown": True,
                },
                {
                    "kind": "tool_call",
                    "name": "device_start",
                    "success": True,
                    "command_sent": True,
                },
                {
                    "kind": "tool_call",
                    "name": "device_start",
                    "success": True,
                    "command_sent": True,
                },
            ],
        },
    ]

    summary = summarize_traces(rows)

    assert summary["planner"] == {
        "attempt_count": 2,
        "statuses": {"fallback": 1, "ready": 1},
        "fallback_rate": 0.5,
        "execution_count": 2,
        "execution_statuses": {"executed": 1, "failed_closed": 1},
        "execution_failure_rate": 0.5,
    }
    assert summary["handlers"] == {
        "image_recipe_handler": 1,
        "planner": 1,
    }
    assert summary["message_types"] == {"image": 1, "text": 1}
    assert summary["fallback_reasons"] == {"PLANNER_TIMEOUT": 1}
    assert summary["tool_failures_by_name"] == {"device_status": 1}
    assert summary["safety"] == {
        "safety_block_turns": 1,
        "duplicate_device_command_turns": 1,
        "forbidden_planner_action_events": 1,
        "device_state_unknown_turns": 1,
        "device_state_unknown_rate": 0.5,
    }


def test_health_alerts_have_fixed_safety_redlines_and_explicit_quality_limits():
    summary = {
        "trace_count": 2,
        "success_rate": 0.94,
        "latency_ms": {"p95": 4200},
        "timeout_rate": 0.02,
        "tool_failure_rate": 0.03,
        "planner": {
            "fallback_rate": 0.2,
            "execution_failure_rate": 0.1,
        },
        "safety": {
            "duplicate_device_command_turns": 1,
            "forbidden_planner_action_events": 1,
            "device_state_unknown_rate": 0.5,
        },
    }

    alerts = evaluate_health_alerts(
        summary,
        {
            "min_trace_count": 10,
            "min_success_rate": 0.98,
            "max_p95_latency_ms": 3000,
            "max_timeout_rate": 0.01,
            "max_tool_failure_rate": 0.02,
            "max_planner_fallback_rate": 0.1,
            "max_planner_execution_failure_rate": 0.05,
            "max_device_state_unknown_rate": 0.1,
        },
    )

    assert {item["code"] for item in alerts} == {
        "DUPLICATE_DEVICE_COMMAND",
        "PLANNER_FORBIDDEN_ACTION",
        "TRACE_COUNT_LOW",
        "TRACE_SUCCESS_RATE_LOW",
        "TRACE_P95_LATENCY_HIGH",
        "TRACE_TIMEOUT_RATE_HIGH",
        "TOOL_FAILURE_RATE_HIGH",
        "PLANNER_FALLBACK_RATE_HIGH",
        "PLANNER_EXECUTION_FAILURE_RATE_HIGH",
        "DEVICE_STATE_UNKNOWN_RATE_HIGH",
    }
    critical = [item for item in alerts if item["severity"] == "critical"]
    assert len(critical) == 2


def test_health_alerts_do_not_invent_unconfigured_performance_thresholds():
    alerts = evaluate_health_alerts(
        {
            "success_rate": 0,
            "latency_ms": {"p95": 999999},
            "safety": {
                "duplicate_device_command_turns": 0,
                "forbidden_planner_action_events": 0,
            },
        }
    )

    assert alerts == []


@pytest.mark.parametrize(
    ("thresholds", "expected_fragment"),
    [
        ({"min_success_rate": 1.1}, "between 0 and 1"),
        ({"max_p95_latency_ms": -1}, "non-negative"),
        ({"min_trace_count": 1.5}, "non-negative integer"),
        ({"max_typo_rate": 0.1}, "unsupported health thresholds"),
    ],
)
def test_health_alerts_reject_invalid_or_unknown_thresholds(
    thresholds,
    expected_fragment,
):
    with pytest.raises(ValueError, match=expected_fragment):
        evaluate_health_alerts({}, thresholds)


def test_tool_failure_code_is_not_counted_as_success(monkeypatch):
    monkeypatch.setenv("CONVERSATION_TRACE_ENABLED", "false")

    async def run():
        _trace, token = ensure_turn_trace(channel="qq", thread_id="qq:dm:x:x")

        @trace_async_tool("device_start")
        async def failed_tool():
            return {"ok": False, "code": "IOT_TIMEOUT"}

        assert (await failed_tool())["ok"] is False
        return finish_turn_trace(token, success=True)

    payload = asyncio.run(run())
    assert payload["tool_call_count"] == 1
    assert payload["tool_failure_count"] == 1
    assert payload["timeout_count"] == 1
    event = next(item for item in payload["events"] if item["kind"] == "tool_call")
    assert event["success"] is False
    assert event["error_type"] == "IOT_TIMEOUT"
