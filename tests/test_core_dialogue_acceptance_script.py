"""核心对话验收清单与报告生成器的纯逻辑测试。"""

from pathlib import Path

from scripts.run_core_dialogue_acceptance import (
    SCENARIOS,
    _build_report,
    _parse_args,
    _selected_scenarios,
)


def test_acceptance_manifest_covers_exactly_ten_unique_scenarios():
    assert len(SCENARIOS) == 10
    assert {scenario.scenario_id for scenario in SCENARIOS} == {
        f"AC-{index:02d}" for index in range(1, 11)
    }
    assert len({scenario.name for scenario in SCENARIOS}) == 10
    assert all(scenario.dialogue for scenario in SCENARIOS)
    assert all(scenario.expected for scenario in SCENARIOS)
    assert all(scenario.tests for scenario in SCENARIOS)


def test_acceptance_manifest_references_existing_test_files():
    for scenario in SCENARIOS:
        for nodeid in scenario.tests:
            path, separator, test_name = nodeid.partition("::")
            assert separator == "::"
            assert test_name.startswith("test_")
            source = Path(path)
            assert source.is_file()
            assert f"def {test_name}(" in source.read_text(encoding="utf-8")


def test_acceptance_report_fails_closed_when_test_is_missing():
    scenario = SCENARIOS[0]
    report = _build_report([scenario], {}, pytest_exit_code=0)
    assert report["passed"] is False
    assert report["scenarios"][0]["checks"][0]["status"] == "missing"
    assert report["real_device_called"] is False
    assert report["external_service_called"] is False


def test_acceptance_report_passes_only_when_every_mapped_test_passes():
    scenario = SCENARIOS[4]
    test_results = {
        nodeid.rsplit("::", 1)[-1]: {
            "status": "passed",
            "duration_seconds": 0.01,
            "detail": "",
        }
        for nodeid in scenario.tests
    }
    report = _build_report([scenario], test_results, pytest_exit_code=0)
    assert report["passed"] is True
    assert report["scenario_pass_rate"] == 1.0


def test_acceptance_cli_supports_scenario_filter_and_no_report():
    args = _parse_args(["--scenario", "AC-02", "--no-report"])
    assert args.scenario == ["AC-02"]
    assert args.no_report is True
    assert [item.scenario_id for item in _selected_scenarios(args.scenario)] == [
        "AC-02"
    ]
