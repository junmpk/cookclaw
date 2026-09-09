"""灰度基线比较必须只使用显式阈值，并在数据缺失时 fail closed。"""

from scripts.compare_conversation_baselines import compare


def _summary(*, latency=100, tokens=50, used=0, failures=0, timeouts=0):
    return {
        "trace_count": 10,
        "latency_ms": {"p95": latency},
        "tokens": {"average": tokens},
        "model_calls": {"average": 1},
        "tool_calls": {"average": 1},
        "tool_failure_rate": failures,
        "timeout_rate": timeouts,
        "deep_agent_used_rate": used,
    }


def test_compare_without_thresholds_reports_data_but_invents_no_gate():
    report = compare(_summary(), _summary(latency=120, tokens=60))
    assert report["passed"] is None
    assert report["release_decision"] == "not_evaluated"
    assert report["gate_count"] == 0
    assert report["deltas"]["p95_latency_percent"] == 20.0
    assert report["deltas"]["average_tokens_percent"] == 20.0
    assert report["note"] == "未配置门禁阈值，仅完成数据对比。"


def test_compare_applies_only_explicit_quality_and_performance_thresholds():
    report = compare(
        _summary(),
        _summary(latency=108, tokens=55, used=0.5),
        candidate_business={
            "effective_turn_rate": 0.95,
            "intent_accuracy": 0.97,
            "route_accuracy": 0.96,
        },
        thresholds={
            "max_p95_latency_regression_percent": 10,
            "max_average_token_regression_percent": 12,
            "min_effective_turn_rate": 0.9,
            "min_intent_accuracy": 0.95,
            "min_route_accuracy": 0.95,
        },
        require_deep_agent_used=True,
    )
    assert report["passed"] is True
    assert report["gate_count"] == 6


def test_compare_fails_when_threshold_metric_is_missing():
    report = compare(
        _summary(),
        _summary(),
        thresholds={"min_effective_turn_rate": 0.9},
    )
    assert report["passed"] is False
    assert report["gates"][0]["reason"] == "metric_unavailable"


def test_compare_fails_when_candidate_has_no_trace_data():
    candidate = _summary()
    candidate["trace_count"] = 0
    report = compare(_summary(), candidate)
    assert report["has_required_trace_data"] is False
    assert report["passed"] is False


def test_compare_can_require_actual_deep_agent_participation():
    report = compare(
        _summary(),
        _summary(used=0),
        require_deep_agent_used=True,
    )
    assert report["passed"] is False
    assert report["gates"][0]["reason"] == "deep_agent_not_observed"
