"""阶段 1 Turn Planner 协议、校验、Shadow 与旁路集成测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage

import app.orchestrator.planning.planner as planner_module
from app.observability.trace import (
    current_trace,
    ensure_turn_trace,
    finish_turn_trace,
)
from app.orchestrator.planning.actions import ACTION_REGISTRY
from app.orchestrator.turn.context_loader import build_turn_context
from app.orchestrator.planning.models import (
    PlannerCallResult,
    PlanStep,
    TurnPlan,
)
from app.orchestrator.planning.plan_validator import PlanValidator
from app.orchestrator.planning.planner import TurnPlanner
from app.orchestrator.planning.shadow_compare import (
    build_shadow_bypass_record,
    build_shadow_record,
    compare_shadow_plan,
    drain_shadow_tasks,
    legacy_route_snapshot,
    record_shadow_result,
    schedule_shadow_evaluation,
)
from app.orchestrator.planning.shadow_report import summarize_shadow_records
from app.orchestrator.routing_models import RoutingContext


def _context(**updates):
    routing = RoutingContext(
        latest_candidates=[
            {
                "cookId": "r1",
                "name": "番茄炒蛋",
                "recipe_detail": {"steps": ["private-step"]},
            },
            {"cookId": "r2", "name": "清蒸鲈鱼"},
        ],
        latest_focus={"cookId": "r1", "name": "番茄炒蛋"},
        selected_recipe={"cookId": "r2", "name": "清蒸鲈鱼"},
        channel="qq",
        message_type="text",
        **updates,
    )
    return build_turn_context(
        "帮我看看第二道",
        routing,
        trace_id="trace-test",
    )


def _legacy_outcome(
    *,
    kind: str = "search",
    action: str = "new_search",
    category: str = "recipe_search",
    risk: str = "low",
    needs_clarification: bool = False,
):
    decision = SimpleNamespace(
        action=action,
        category=category,
        reason_code="LEGACY_TEST",
        risk=risk,
        needs_clarification=needs_clarification,
    )
    return SimpleNamespace(kind=kind, decision=decision, intent=None)


def _valid_search_plan() -> TurnPlan:
    return TurnPlan(
        goal="搜索用户指定的真实菜谱",
        phase="search_or_recommend",
        known_facts=["用户明确要求搜索"],
        steps=[
            PlanStep(
                action="recipe.search",
                args={"query": "红烧肉"},
                evidence_refs=["utterance"],
            )
        ],
        reply_act="offer_next_step",
        risk="low",
        reason_code="EXPLICIT_RECIPE_SEARCH",
        confidence=0.92,
    )


def test_default_planner_model_uses_deterministic_json_mode(monkeypatch):
    captured: dict[str, object] = {}
    bound_model = object()

    class FakeChatQwen:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def bind(self, **kwargs):
            captured["bind"] = kwargs
            return bound_model

    monkeypatch.setattr(planner_module, "ChatQwen", FakeChatQwen)
    monkeypatch.setattr(planner_module, "_planner_llm", None)
    monkeypatch.setattr(planner_module, "_planner_llm_signature", None)

    assert planner_module.get_planner_llm() is bound_model
    assert captured["init"]["temperature"] == 0
    assert captured["bind"] == {"response_format": {"type": "json_object"}}
    assert "`missing_slots` may list other information" in (
        planner_module._PLANNER_PROMPT
    )


def test_context_loader_exposes_only_compact_business_state():
    routing = RoutingContext(
        pending_action={
            "kind": "select_recipe",
            "payload": {"password": "must-not-leak"},
        },
        pending_device_start={
            "cookId": "r1",
            "device_id": "private-device-id",
            "devices": [("private-device-id", "厨房设备")],
        },
        active_cooking={
            "cookId": "r1",
            "device_id": "another-private-device-id",
        },
        latest_candidates=[
            {
                "cookId": "r1",
                "name": "番茄炒蛋",
                "recipe_detail": {"steps": ["private-step"]},
            }
        ],
        recent_user_turns=["旧轮次"],
        stable_constraints={"allergens": ["花生"]},
        channel="qq",
    )
    context = build_turn_context(
        "这轮不吃辣",
        routing,
        memory_context={
            "recent_user_turns": ["最近一轮"],
            "preferences": {"allergens": ["花生"]},
            "temporal_dietary_constraints": [{"value": "减脂"}],
            "previous_candidate_count": 3,
            "private_profile": {"name": "must-not-leak"},
        },
        trace_id="trace-1",
    )

    payload = json.dumps(context.planner_payload(), ensure_ascii=False)
    log_ref = json.dumps(context.context_ref(), ensure_ascii=False)
    assert context.pending_action == "select_recipe"
    assert context.pending_device_start is True
    assert context.device_choice_required is True
    assert context.active_cooking is True
    assert context.latest_candidates[0].recipe_id == "r1"
    assert context.previous_candidate_count == 3
    assert context.recent_user_turns == ["最近一轮"]
    assert context.active_constraints["temporal_dietary_constraints"] == ["减脂"]
    assert "private-device-id" not in payload
    assert "another-private-device-id" not in payload
    assert "private-step" not in payload
    assert "must-not-leak" not in payload
    assert "这轮不吃辣" not in log_ref
    assert "番茄炒蛋" not in log_ref
    assert "花生" not in log_ref


def test_plan_validator_accepts_bounded_candidate_and_menu_actions():
    context = build_turn_context(
        "回到上一批，再比较一下",
        RoutingContext(
            latest_candidates=[
                {"cookId": "r1", "name": "番茄炒蛋"},
                {"cookId": "r2", "name": "清蒸鲈鱼"},
            ],
            current_task="menu_plan",
            channel="qq",
        ),
        memory_context={"previous_candidate_count": 3},
        trace_id="trace-bounded-actions",
    )
    plans = [
        TurnPlan(
            goal="比较当前候选",
            phase="candidate_selection",
            known_facts=["当前有多个候选"],
            steps=[
                PlanStep(
                    action="candidate.compare",
                    evidence_refs=["candidate:r1"],
                )
            ],
            reply_act="explain",
            reason_code="COMPARE_CURRENT_CANDIDATES",
            confidence=0.8,
        ),
        TurnPlan(
            goal="恢复上一批候选",
            phase="follow_up",
            known_facts=["存在上一批候选"],
            steps=[
                PlanStep(
                    action="candidate.restore_previous",
                    evidence_refs=["previous_candidates"],
                )
            ],
            reply_act="report_state",
            reason_code="RESTORE_PREVIOUS_CANDIDATES",
            confidence=0.8,
        ),
        TurnPlan(
            goal="继续规划聚餐菜单",
            phase="search_or_recommend",
            known_facts=["当前任务是菜单规划"],
            steps=[
                PlanStep(
                    action="menu.plan",
                    args={"query": "四菜一汤"},
                    evidence_refs=["menu_task"],
                )
            ],
            reply_act="recommend",
            reason_code="CONTINUE_MENU_PLAN",
            confidence=0.8,
        ),
    ]

    results = [PlanValidator().validate(plan, context) for plan in plans]

    assert context.menu_task_active is True
    assert all(result.valid for result in results)


def test_plan_validator_rejects_candidate_actions_without_required_context():
    context = build_turn_context(
        "回到上一批，再比较一下",
        RoutingContext(channel="qq"),
        trace_id="trace-missing-candidate-context",
    )
    compare_plan = TurnPlan(
        goal="比较候选",
        phase="candidate_selection",
        known_facts=["用户要求比较"],
        steps=[
            PlanStep(
                action="candidate.compare",
                evidence_refs=["utterance"],
            )
        ],
        reply_act="explain",
        reason_code="COMPARE_CANDIDATES",
        confidence=0.7,
    )
    restore_plan = TurnPlan(
        goal="恢复上一批",
        phase="follow_up",
        known_facts=["用户要求上一批"],
        steps=[
            PlanStep(
                action="candidate.restore_previous",
                evidence_refs=["utterance"],
            )
        ],
        reply_act="report_state",
        reason_code="RESTORE_PREVIOUS",
        confidence=0.7,
    )

    compare_result = PlanValidator().validate(compare_plan, context)
    restore_result = PlanValidator().validate(restore_plan, context)

    assert "CANDIDATE_COMPARISON_REQUIRES_MULTIPLE" in compare_result.issue_codes
    assert "PREVIOUS_CANDIDATE_CONTEXT_REQUIRED" in restore_result.issue_codes


def test_plan_validator_accepts_grounded_recipe_detail():
    context = _context()
    plan = TurnPlan(
        goal="展示第二道菜的真实详情",
        phase="recipe_detail",
        known_facts=["第二道候选是已核验菜谱"],
        steps=[
            PlanStep(
                action="recipe.detail",
                args={"recipe_id": "r2"},
                evidence_refs=["candidate:r2"],
            )
        ],
        reply_act="explain",
        risk="low",
        reason_code="VERIFIED_CANDIDATE_DETAIL",
        confidence=0.9,
    )

    result = PlanValidator().validate(plan, context)

    assert result.valid is True
    assert result.execution_allowed is True
    assert result.issue_codes == []


def test_plan_validator_rejects_recipe_outside_context_and_device_args():
    context = _context()
    plan = TurnPlan(
        goal="准备设备制作",
        phase="device_prepare",
        known_facts=["用户想制作一道菜"],
        steps=[
            PlanStep(
                action="device.prepare",
                args={
                    "recipe_id": "invented-id",
                    "device_id": "private-device",
                },
                evidence_refs=["utterance"],
            )
        ],
        reply_act="offer_next_step",
        risk="medium",
        reason_code="DEVICE_PREPARE",
        confidence=0.8,
    )

    result = PlanValidator().validate(plan, context)

    assert result.valid is False
    assert result.execution_allowed is False
    assert "ACTION_ARGS_NOT_ALLOWED" in result.issue_codes
    assert "FORBIDDEN_SIDE_EFFECT_ARG" in result.issue_codes
    assert "RECIPE_ID_OUTSIDE_CONTEXT" in result.issue_codes


def test_plan_validator_allows_high_risk_deterministic_handoff_only():
    context = _context(active_cooking={"cookId": "r2"})
    safe_handoff = TurnPlan(
        goal="停止当前运行任务",
        phase="running",
        known_facts=["存在活动任务"],
        steps=[],
        reply_act="report_state",
        risk="high",
        reason_code="DEFER_TO_DETERMINISTIC_HANDLER",
        confidence=0.99,
        requires_deterministic_handler=True,
    )
    unsafe_step = safe_handoff.model_copy(update={
        "steps": [
            PlanStep(
                action="device.status",
                evidence_refs=["active_cooking"],
            )
        ]
    })
    wrong_reason = safe_handoff.model_copy(update={"reason_code": "OTHER"})

    safe_result = PlanValidator().validate(safe_handoff, context)
    unsafe_result = PlanValidator().validate(unsafe_step, context)
    wrong_reason_result = PlanValidator().validate(wrong_reason, context)

    assert safe_result.valid is True
    assert safe_result.execution_allowed is False
    assert unsafe_result.valid is False
    assert "DETERMINISTIC_HANDOFF_MUST_NOT_HAVE_STEPS" in unsafe_result.issue_codes
    assert "DETERMINISTIC_HANDOFF_REASON_INVALID" in wrong_reason_result.issue_codes


def test_plan_validator_rejects_conflicting_candidate_references():
    context = _context()
    plan = TurnPlan(
        goal="选择候选菜谱",
        phase="candidate_selection",
        known_facts=["存在两个候选"],
        steps=[
            PlanStep(
                action="candidate.select",
                args={"recipe_id": "r1", "position": 2},
                evidence_refs=["candidate:r1"],
            )
        ],
        reply_act="offer_next_step",
        risk="medium",
        reason_code="SELECT_CANDIDATE",
        confidence=0.8,
    )

    result = PlanValidator().validate(plan, context)

    assert result.valid is False
    assert "CANDIDATE_REFERENCE_CONFLICT" in result.issue_codes


def test_plan_validator_allows_one_question_selected_from_missing_slots():
    context = _context()
    plan = TurnPlan(
        goal="确认聚餐人数和忌口",
        phase="clarify",
        missing_slots=["人数", "忌口"],
        steps=[
            PlanStep(
                action="conversation.clarify",
                args={"slot": "人数"},
                evidence_refs=["utterance"],
            )
        ],
        reply_act="ask_one_question",
        risk="low",
        reason_code="MISSING_PARTY_CONSTRAINTS",
        confidence=0.7,
    )

    result = PlanValidator().validate(plan, context)

    assert result.valid is True
    assert result.execution_allowed is True


def test_plan_validator_rejects_clarification_slot_not_declared_missing():
    context = _context()
    plan = TurnPlan(
        goal="确认聚餐人数",
        phase="clarify",
        missing_slots=["人数", "忌口"],
        steps=[
            PlanStep(
                action="conversation.clarify",
                args={"slot": "预算"},
                evidence_refs=["utterance"],
            )
        ],
        reply_act="ask_one_question",
        risk="low",
        reason_code="MISSING_PARTY_CONSTRAINTS",
        confidence=0.7,
    )

    result = PlanValidator().validate(plan, context)

    assert result.valid is False
    assert "CLARIFY_SLOT_MISMATCH" in result.issue_codes


def test_plan_validator_rejects_empty_step_and_unknown_evidence():
    context = _context()
    empty = TurnPlan(
        goal="回答问题",
        phase="explore",
        steps=[],
        reply_act="answer",
        risk="low",
        reason_code="ANSWER",
        confidence=0.8,
    )
    bad_evidence = _valid_search_plan().model_copy(update={
        "steps": [
            PlanStep(
                action="recipe.search",
                args={"query": "红烧肉"},
                evidence_refs=["private_database_row"],
            )
        ]
    })

    empty_result = PlanValidator().validate(empty, context)
    evidence_result = PlanValidator().validate(bad_evidence, context)

    assert "NON_DETERMINISTIC_PLAN_REQUIRES_STEP" in empty_result.issue_codes
    assert "EVIDENCE_REF_OUTSIDE_CONTEXT" in evidence_result.issue_codes


def test_turn_planner_parses_fenced_structured_output_without_trace_id():
    captured = []

    class FakeModel:
        async def ainvoke(self, messages):
            captured.extend(messages)
            return AIMessage(
                content=(
                    "```json\n"
                    + json.dumps(
                        _valid_search_plan().model_dump(),
                        ensure_ascii=False,
                    )
                    + "\n```"
                ),
                usage_metadata={
                    "input_tokens": 30,
                    "output_tokens": 20,
                    "total_tokens": 50,
                },
            )

    result = asyncio.run(
        TurnPlanner(
            FakeModel(),
            model_name="fake-planner",
            timeout_seconds=1,
        ).plan(_context())
    )

    assert result.status == "success"
    assert result.plan is not None
    assert result.plan.steps[0].action == "recipe.search"
    assert result.input_tokens == 30
    assert result.output_tokens == 20
    assert "trace-test" not in str(captured[-1].content)


def test_turn_planner_fails_closed_on_unknown_action():
    class FakeModel:
        async def ainvoke(self, _messages):
            data = _valid_search_plan().model_dump()
            data["steps"][0]["action"] = "device.start"
            return AIMessage(content=json.dumps(data, ensure_ascii=False))

    result = asyncio.run(
        TurnPlanner(FakeModel(), timeout_seconds=1).plan(_context())
    )

    assert result.status == "parse_error"
    assert result.error_code == "PLANNER_SCHEMA_ERROR"
    assert result.plan is None


def test_turn_planner_schema_mutation_corpus_fails_closed():
    base = _valid_search_plan().model_dump()
    payloads = []

    unknown_phase = json.loads(json.dumps(base, ensure_ascii=False))
    unknown_phase["phase"] = "autonomous_loop"
    payloads.append(unknown_phase)

    too_many_steps = json.loads(json.dumps(base, ensure_ascii=False))
    too_many_steps["steps"] = [base["steps"][0]] * 3
    payloads.append(too_many_steps)

    extra_field = json.loads(json.dumps(base, ensure_ascii=False))
    extra_field["execute_immediately"] = True
    payloads.append(extra_field)

    unsafe_reason = json.loads(json.dumps(base, ensure_ascii=False))
    unsafe_reason["reason_code"] = "用户说了私密内容"
    payloads.append(unsafe_reason)

    invalid_confidence = json.loads(json.dumps(base, ensure_ascii=False))
    invalid_confidence["confidence"] = 2
    payloads.append(invalid_confidence)

    empty_goal = json.loads(json.dumps(base, ensure_ascii=False))
    empty_goal["goal"] = ""
    payloads.append(empty_goal)

    class FakeModel:
        def __init__(self, payload):
            self.payload = payload

        async def ainvoke(self, _messages):
            return AIMessage(
                content=json.dumps(self.payload, ensure_ascii=False)
            )

    async def run():
        return [
            await TurnPlanner(FakeModel(payload), timeout_seconds=1).plan(
                _context()
            )
            for payload in payloads
        ]

    results = asyncio.run(run())

    assert all(result.status == "parse_error" for result in results)
    assert all(result.plan is None for result in results)


def test_turn_planner_timeout_is_a_result_not_an_exception():
    class SlowModel:
        async def ainvoke(self, _messages):
            await asyncio.sleep(0.1)

    result = asyncio.run(
        TurnPlanner(SlowModel(), timeout_seconds=0.001).plan(_context())
    )

    assert result.status == "timeout"
    assert result.error_code == "PLANNER_TIMEOUT"
    assert result.plan is None


def test_shadow_record_contains_only_sanitized_plan_summary():
    context = _context()
    legacy = legacy_route_snapshot(_legacy_outcome())
    plan = _valid_search_plan()
    validation = PlanValidator().validate(plan, context)
    comparison = compare_shadow_plan(legacy, plan, validation)
    record = build_shadow_record(
        context=context,
        legacy=legacy,
        planner_result=PlannerCallResult(
            status="success",
            plan=plan,
            model="fake",
            duration_ms=12,
        ),
        validation=validation,
        comparison=comparison,
    )
    encoded = json.dumps(record, ensure_ascii=False)

    assert record["comparison"]["action_match"] is True
    assert record["planner"]["step_actions"] == ["recipe.search"]
    assert "帮我看看第二道" not in encoded
    assert "番茄炒蛋" not in encoded
    assert "清蒸鲈鱼" not in encoded
    assert "红烧肉" not in encoded
    assert "EXPLICIT_RECIPE_SEARCH" not in encoded
    assert "r1" not in encoded
    assert "r2" not in encoded


def test_shadow_bypass_record_has_trace_and_reason_without_private_content():
    context = _context()

    record = build_shadow_bypass_record(
        context=context,
        bypass_reason="pending action handler",
        legacy_action="candidate_followup",
        category="recipe_search",
    )
    encoded = json.dumps(record, ensure_ascii=False)

    assert record["trace_id"] == "trace-test"
    assert record["status"] == "bypassed"
    assert record["bypass_reason"] == "PENDING_ACTION_HANDLER"
    assert record["legacy"]["action"] == "candidate_followup"
    assert record["planner"] is None
    assert "帮我看看第二道" not in encoded
    assert "番茄炒蛋" not in encoded
    assert "r1" not in encoded


def test_shadow_record_writes_independent_jsonl_without_private_content(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "planner-shadow.jsonl"
    context = _context()
    legacy = legacy_route_snapshot(_legacy_outcome())
    plan = _valid_search_plan()
    validation = PlanValidator().validate(plan, context)
    comparison = compare_shadow_plan(legacy, plan, validation)
    record = build_shadow_record(
        context=context,
        legacy=legacy,
        planner_result=PlannerCallResult(
            status="success",
            plan=plan,
            model="fake",
        ),
        validation=validation,
        comparison=comparison,
    )
    monkeypatch.setenv("TURN_PLANNER_SHADOW_JSONL_PATH", str(path))

    record_shadow_result(record)

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [record]
    encoded = path.read_text(encoding="utf-8")
    assert "帮我看看第二道" not in encoded
    assert "番茄炒蛋" not in encoded
    assert "红烧肉" not in encoded


def test_shadow_scheduler_runs_only_in_shadow_mode_and_isolates_turn_trace(
    monkeypatch,
):
    import app.orchestrator.planning.shadow_compare as shadow_module

    records = []
    planner_trace_values = []

    class FakePlanner:
        async def plan(self, _context):
            planner_trace_values.append(current_trace())
            return PlannerCallResult(
                status="success",
                plan=_valid_search_plan(),
                model="fake",
            )

    async def run():
        context = _context()
        outcome = _legacy_outcome()

        monkeypatch.setattr(shadow_module.settings, "TURN_PLANNER_MODE", "off")
        assert schedule_shadow_evaluation(context, outcome) is None

        monkeypatch.setattr(shadow_module.settings, "TURN_PLANNER_MODE", "active")
        assert schedule_shadow_evaluation(context, outcome) is None

        monkeypatch.setattr(shadow_module.settings, "TURN_PLANNER_MODE", "shadow")
        monkeypatch.setattr(shadow_module, "_get_planner", lambda: FakePlanner())
        monkeypatch.setattr(
            shadow_module,
            "record_shadow_result",
            records.append,
        )
        _trace, token = ensure_turn_trace(
            channel="qq",
            thread_id="qq:dm:shadow:shadow",
        )
        task = schedule_shadow_evaluation(context, outcome)
        assert task is not None
        await drain_shadow_tasks(timeout_seconds=1)
        payload = finish_turn_trace(token, success=True)
        return payload

    payload = asyncio.run(run())

    assert len(records) == 1
    assert records[0]["status"] == "success"
    assert records[0]["comparison"]["action_match"] is True
    assert planner_trace_values == [None]
    assert payload["model_call_count"] == 0


def test_turn_orchestrator_off_mode_does_not_build_context(monkeypatch):
    import app.orchestrator.turn.turn_orchestrator as turn_module
    import app.orchestrator.planning.shadow_compare as shadow_module

    monkeypatch.setattr(shadow_module.settings, "TURN_PLANNER_MODE", "off")

    def forbidden_context(*_args, **_kwargs):
        raise AssertionError("off 模式不应构造 Planner 上下文")

    monkeypatch.setattr(turn_module, "build_turn_context", forbidden_context)

    assert turn_module.schedule_turn_plan_shadow(
        utterance="找红烧肉",
        routing_context=RoutingContext(channel="qq"),
        legacy_outcome=_legacy_outcome(),
    ) is None

    monkeypatch.setattr(shadow_module.settings, "TURN_PLANNER_MODE", "active")
    assert turn_module.schedule_turn_plan_shadow(
        utterance="找红烧肉",
        routing_context=RoutingContext(channel="qq"),
        legacy_outcome=_legacy_outcome(),
    ) is None


def test_turn_orchestrator_shadow_failure_never_changes_legacy_flow(monkeypatch):
    import app.orchestrator.turn.turn_orchestrator as turn_module
    import app.orchestrator.planning.shadow_compare as shadow_module

    monkeypatch.setattr(shadow_module.settings, "TURN_PLANNER_MODE", "shadow")

    def broken_context(*_args, **_kwargs):
        raise ValueError("private details must not propagate")

    monkeypatch.setattr(turn_module, "build_turn_context", broken_context)

    assert turn_module.schedule_turn_plan_shadow(
        utterance="找红烧肉",
        routing_context=RoutingContext(channel="qq"),
        legacy_outcome=_legacy_outcome(),
    ) is None


def test_turn_orchestrator_records_early_bypass_only_in_shadow(monkeypatch):
    import app.orchestrator.turn.turn_orchestrator as turn_module
    import app.orchestrator.planning.shadow_compare as shadow_module

    captured = []
    monkeypatch.setattr(turn_module, "record_shadow_bypass", lambda *args, **kwargs: captured.append((args, kwargs)) or {})
    monkeypatch.setattr(shadow_module.settings, "TURN_PLANNER_MODE", "off")
    assert turn_module.record_turn_plan_bypass(
        utterance="可以",
        routing_context=RoutingContext(channel="qq"),
        bypass_reason="affirmative_without_pending",
    ) is None
    assert captured == []

    monkeypatch.setattr(shadow_module.settings, "TURN_PLANNER_MODE", "shadow")
    result = turn_module.record_turn_plan_bypass(
        utterance="可以",
        routing_context=RoutingContext(channel="qq"),
        bypass_reason="affirmative_without_pending",
        needs_clarification=True,
    )

    assert result == {}
    assert len(captured) == 1
    context = captured[0][0][0]
    assert context.trace_id
    assert captured[0][0][1] == "affirmative_without_pending"
    assert captured[0][1]["needs_clarification"] is True


def test_shadow_capacity_limit_skips_extra_model_call(monkeypatch):
    import app.orchestrator.planning.shadow_compare as shadow_module

    async def run():
        monkeypatch.setattr(shadow_module.settings, "TURN_PLANNER_MODE", "shadow")
        monkeypatch.setattr(
            shadow_module.settings,
            "TURN_PLANNER_SHADOW_MAX_INFLIGHT",
            1,
        )
        blocker = asyncio.create_task(asyncio.sleep(60))
        shadow_module._shadow_tasks.add(blocker)
        try:
            assert schedule_shadow_evaluation(
                _context(),
                _legacy_outcome(),
            ) is None
        finally:
            shadow_module._shadow_tasks.discard(blocker)
            blocker.cancel()
            await asyncio.gather(blocker, return_exceptions=True)

    asyncio.run(run())


def test_route_wrapper_returns_legacy_outcome_unchanged_in_shadow(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    captured = []
    route_calls = []
    expected = _legacy_outcome()
    thread_id = "qq:dm:planner-shadow:planner-shadow"
    session.clear_thread(thread_id)

    async def fake_route(_question, **_kwargs):
        route_calls.append(_question)
        return expected

    def fake_schedule(**kwargs):
        captured.append(kwargs)
        return None

    async def run():
        return await agent_module._route_conversation_turn(
            "帮我找红烧肉",
            thread_id,
        )

    monkeypatch.setattr(agent_module, "route_fast_path", fake_route)
    monkeypatch.setattr(
        agent_module,
        "supports_session_memory",
        lambda _thread_id: False,
    )
    monkeypatch.setattr(
        agent_module,
        "schedule_turn_plan_shadow",
        fake_schedule,
    )

    result = asyncio.run(run())

    assert result is expected
    assert route_calls == ["帮我找红烧肉"]
    assert len(captured) == 1
    assert captured[0]["legacy_outcome"] is expected
    assert captured[0]["routing_context"].channel == "qq"
    session.clear_thread(thread_id)


def test_early_affirmative_records_bypass_without_entering_router(monkeypatch):
    import app.agent.participle_agent as agent_module
    from app.conversation import task_state_workspace as session

    captured = []
    thread_id = "planner-bypass-affirmative"
    session.clear_thread(thread_id)

    async def forbidden_route(*_args, **_kwargs):
        raise AssertionError("无 pending 肯定词不应进入 Legacy router")

    monkeypatch.setattr(agent_module, "route_fast_path", forbidden_route)
    monkeypatch.setattr(
        agent_module,
        "record_turn_plan_bypass",
        lambda **kwargs: captured.append(kwargs),
    )

    response = asyncio.run(agent_module.qqbot_chat("可以", thread_id=thread_id))

    message = json.loads(response)["message"]
    assert "没有正在等确认的步骤" in message
    assert message.count("？") == 1
    assert "待办动作" not in message
    assert "不能替你猜" not in message
    assert not any(word in message for word in ("已确认", "已开始", "已执行"))
    assert len(captured) == 1
    assert captured[0]["bypass_reason"] == "AFFIRMATIVE_WITHOUT_PENDING"
    assert captured[0]["routing_context"].channel == "unknown"
    assert captured[0]["needs_clarification"] is True
    session.clear_thread(thread_id)


def test_golden_cases_cover_core_actions_and_deterministic_handoff():
    path = Path(__file__).parent / "fixtures" / "turn_planner_golden.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    ids = [item["id"] for item in cases]
    actions = {
        action
        for item in cases
        for action in item["expected_actions"]
    }

    assert len(ids) == len(set(ids))
    assert len(cases) >= 10
    assert "recipe.search" in actions
    assert "recipe.recommend" in actions
    assert "recipe.detail" in actions
    assert "candidate.select" in actions
    assert "candidate.compare" in actions
    assert "candidate.restore_previous" in actions
    assert "menu.plan" in actions
    assert "device.prepare" in actions
    assert "web.search" in actions
    assert "deterministic_handler" in actions
    assert actions - set(ACTION_REGISTRY) == {"deterministic_handler"}


def test_shadow_summary_reports_measured_rates_without_targets():
    rows = [
        {
            "schema_version": "turn_planner_shadow_v1",
            "status": "success",
            "context_ref": {"channel": "qq"},
            "legacy": {"action": "new_search"},
            "planner": {"step_actions": ["recipe.search"]},
            "validation": {"valid": True},
            "comparison": {
                "action_match": True,
                "risk_match": True,
                "clarification_match": True,
                "deterministic_handoff_match": True,
                "mismatch_codes": [],
            },
            "model": {
                "duration_ms": 100,
                "input_tokens": 20,
                "output_tokens": 10,
            },
        },
        {
            "schema_version": "turn_planner_shadow_v1",
            "status": "parse_error",
            "context_ref": {"channel": "qq"},
            "legacy": {"action": "stop"},
            "planner": None,
            "validation": None,
            "comparison": None,
            "model": {
                "duration_ms": 300,
                "input_tokens": 30,
                "output_tokens": 20,
                "error_code": "PLANNER_SCHEMA_ERROR",
            },
        },
    ]

    summary = summarize_shadow_records(rows)

    assert summary["record_count"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["validation_valid_rate"] == 1.0
    assert summary["comparison"]["action_match_rate"] == 1.0
    assert summary["latency_ms"]["p50"] == 200.0
    assert summary["tokens"]["total"] == 80
    assert summary["errors"] == {"PLANNER_SCHEMA_ERROR": 1}
    assert "target" not in summary


def test_shadow_summary_excludes_bypasses_from_planner_success_rate():
    rows = [
        {
            "schema_version": "turn_planner_shadow_v1",
            "status": "success",
            "context_ref": {"channel": "qq"},
            "legacy": {"action": "new_search"},
            "planner": {"step_actions": ["recipe.search"]},
            "validation": {"valid": True},
            "comparison": None,
            "model": {
                "duration_ms": 100,
                "input_tokens": 10,
                "output_tokens": 5,
            },
        },
        {
            "schema_version": "turn_planner_shadow_v1",
            "status": "bypassed",
            "bypass_reason": "PENDING_ACTION_HANDLER",
            "context_ref": {"channel": "qq"},
            "legacy": {"action": "candidate_followup"},
            "planner": None,
            "validation": None,
            "comparison": None,
            "model": {
                "duration_ms": 9999,
                "input_tokens": 9999,
                "output_tokens": 9999,
            },
        },
    ]

    summary = summarize_shadow_records(rows)

    assert summary["record_count"] == 2
    assert summary["evaluated_record_count"] == 1
    assert summary["bypass_count"] == 1
    assert summary["planner_evaluation_rate"] == 0.5
    assert summary["success_rate"] == 1.0
    assert summary["bypass_reasons"] == {"PENDING_ACTION_HANDLER": 1}
    assert summary["latency_ms"]["p50"] == 100.0
    assert summary["tokens"]["total"] == 15
    assert summary["tokens"]["average_total"] == 15.0
