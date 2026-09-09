"""会话态 + 选择解析的单测（纯逻辑，不碰设备/网络）。
跑：.venv/bin/python tests/test_session.py  或  pytest tests/test_session.py
"""
from app.conversation.models import ConversationTaskState
from app.conversation import task_state_workspace as S


def _mk(*id_name):
    return {"results": [{"metadata": {"recipe_id": i, "name": n}} for i, n in id_name]}


def _seed(tid="t", *id_name):
    S._last.clear()
    S.remember_candidates(tid, _mk(*id_name))


def test_remember_recall():
    _seed("t", ("111", "一键红烧肉"), ("222", "稻香红烧肉"), ("333", "东坡肉"))
    items = S.recall_candidates("t")
    assert items and len(items) == 3
    assert items[0] == {"cookId": "111", "name": "一键红烧肉"}


def test_remember_keeps_candidate_context_fields():
    S._last.clear()
    S.remember_candidates("t", {
        "results": [{
            "metadata": {
                "recipe_id": "111",
                "name": "剁椒炒藕片",
                "tags": ["湘菜", "香辣"],
                "ingredients": ["藕", "剁椒"],
                "image_url": "https://example.com/a.png",
            }
        }]
    }, lang="zh")
    ctx = S.recall_candidate_context("t")
    assert ctx["lang"] == "zh"
    assert ctx["items"][0]["tags"] == ["湘菜", "香辣"]
    assert ctx["items"][0]["ingredients"] == ["藕", "剁椒"]


def test_candidate_context_keeps_decision_id():
    S._last.clear()
    result = _mk(("111", "番茄炒蛋"))
    result["_decision_id"] = "decision-1"
    S.remember_candidates("decision-thread", result, lang="zh")
    assert S.recall_candidate_context("decision-thread")["decision_id"] == "decision-1"


def test_task_state_mutable_fields_are_isolated():
    first = ConversationTaskState()
    second = ConversationTaskState()
    first.candidate_recipes.append({"cookId": "r1", "name": "菜一"})
    first.excluded_recipe_ids.append("r1")

    assert second.candidate_recipes == []
    assert second.excluded_recipe_ids == []


def test_task_state_snapshot_restore_keeps_stable_candidate_indexes():
    tid = "task-state-roundtrip"
    S.clear_thread(tid)
    S.remember_candidates(
        tid,
        {
            **_mk(("r1", "菜一"), ("r2", "菜二"), ("r3", "菜三")),
            "_decision_id": "decision-roundtrip",
            "_search_request": {"query": "鸡肉", "exclude": ["辣椒"]},
        },
        lang="zh",
    )
    S.set_selected_recipe(
        tid,
        {"cookId": "r2", "name": "菜二"},
        lang="zh",
    )
    S.set_candidate_excluded(tid, "r1")

    snapshot = S.snapshot_thread_state(tid)
    S.clear_thread(tid)
    assert S.recall_candidates(tid) is None

    S.restore_thread_state(tid, snapshot)
    restored = S.recall_candidate_context(tid)
    assert [item["cookId"] for item in restored["items"]] == ["r1", "r2", "r3"]
    assert restored["decision_id"] == "decision-roundtrip"
    assert restored["search_request"]["exclude"] == ["辣椒"]
    assert S.get_selected_recipe(tid)["cookId"] == "r2"
    assert S.get_excluded_recipe_ids(tid) == ["r1"]
    assert S.resolve_selection(tid, "第二个")["cookId"] == "r2"
    S.clear_thread(tid)


def test_active_search_request_survives_candidate_invalidation_and_roundtrip():
    tid = "active-search-roundtrip"
    S.clear_thread(tid)
    request = {
        "original_text": "推荐鸡肉菜",
        "query": "鸡肉",
        "ingredients": ["鸡肉"],
        "avoid": ["辣"],
    }
    S.set_active_search_request(tid, request, lang="zh")
    S.remember_candidates(
        tid,
        {
            **_mk(("r1", "菜一"), ("r2", "菜二")),
            "_search_request": request,
        },
        lang="zh",
    )

    S.clear_candidate_context(tid, keep_search_request=True)
    assert S.recall_candidate_context(tid) is None
    assert S.get_active_search_request(tid)["request"]["ingredients"] == ["鸡肉"]
    snapshot = S.snapshot_thread_state(tid)
    assert snapshot.current_task == "recipe_search"
    assert snapshot.active_search_request["avoid"] == ["辣"]

    S.clear_thread(tid)
    S.restore_thread_state(tid, snapshot)
    assert S.get_active_search_request(tid)["request"]["query"] == "鸡肉"
    assert S.snapshot_thread_state(tid).language == "zh"
    S.clear_thread(tid)


def test_routing_state_exposes_only_pruned_active_constraints():
    tid = "active-search-routing"
    S.clear_thread(tid)
    S.set_active_search_request(tid, {
        "original_text": "这段原文不应进入路由摘要",
        "query": "鸡肉",
        "ingredients": ["鸡肉"],
        "avoid": ["辣"],
        "memory_note": {"private": "not-for-router"},
    }, lang="zh")

    state = S.routing_state_snapshot(tid)
    assert state["current_task"] == "recipe_search"
    assert state["active_constraints"] == {
        "ingredients": ["鸡肉"],
        "avoid": ["辣"],
    }
    assert "active_search_request" not in state
    S.clear_thread(tid)


def test_focus_remembers_latest_topic_by_thread():
    S._focus.clear()
    S.remember_focus("a", "鸡蛋炒西红柿", lang="zh", source="qa")
    S.remember_focus("b", "红烧肉", lang="zh", source="search")
    assert S.recall_focus("a")["topic"] == "鸡蛋炒西红柿"
    assert S.recall_focus("a")["source"] == "qa"
    assert S.recall_focus("b")["topic"] == "红烧肉"
    S.remember_focus("a", "番茄炒蛋", lang="zh", source="qa")
    assert S.recall_focus("a")["topic"] == "番茄炒蛋"


def test_query_focus_is_typed_and_not_a_verified_recipe():
    S._focus.clear()
    S.remember_focus(
        "query-thread",
        "晚饭 清淡",
        source="search_history",
        focus_kind="query",
    )
    focus = S.recall_focus("query-thread")
    assert focus["focus_kind"] == "query"
    assert not focus.get("verified_recipe")


def test_search_clarification_is_isolated_and_cleared_with_thread():
    S.clear_thread("clarify-a")
    S.clear_thread("clarify-b")
    request = {"original_text": "晚饭吃什么", "query": "晚饭"}
    S.set_search_clarification(
        "clarify-a", request, dimension="available_ingredients", lang="zh",
    )
    assert S.get_search_clarification("clarify-a")["request"] == request
    assert S.get_search_clarification("clarify-a")["dimension"] == "available_ingredients"
    assert S.get_search_clarification("clarify-b") is None
    S.clear_thread("clarify-a")
    assert S.get_search_clarification("clarify-a") is None


def test_search_clarification_tracks_rounds_and_asked_facts():
    S.clear_thread("clarify-rounds")
    S.set_search_clarification(
        "clarify-rounds",
        {"query": "午餐"},
        dimension="flavor_preferences",
        lang="zh",
        asked_dimensions=["recommendation_basics", "flavor_preferences"],
        round_count=2,
    )

    state = S.get_search_clarification("clarify-rounds")
    assert state["asked_dimensions"] == [
        "recommendation_basics",
        "flavor_preferences",
    ]
    assert state["round_count"] == 2


def test_ordinal_arabic():
    _seed("t", ("111", "A"), ("222", "B"), ("333", "C"))
    assert S.resolve_selection("t", "1")["cookId"] == "111"
    assert S.resolve_selection("t", "做第二个")["cookId"] == "222"
    assert S.resolve_selection("t", "第3道")["cookId"] == "333"


def test_ordinal_chinese():
    _seed("t", ("111", "A"), ("222", "B"))
    assert S.resolve_selection("t", "第一个")["cookId"] == "111"
    assert S.resolve_selection("t", "二")["cookId"] == "222"


def test_ordinal_english_and_lang():
    S._last.clear()
    S.remember_candidates("t", _mk(("111", "A"), ("222", "B")), lang="en")
    sel = S.resolve_selection("t", "make the first one")
    assert sel["cookId"] == "111" and sel["lang"] == "en"
    assert S.resolve_selection("t", "second")["cookId"] == "222"


def test_long_number_is_not_ordinal():
    _seed("t", ("111", "A"), ("222", "B"))
    # 19 位长数字是 cookId，不当序号；也不匹配名字 → None
    assert S.resolve_selection("t", "2369728950351216642") is None


def test_out_of_range():
    _seed("t", ("111", "A"), ("222", "B"))
    assert S.resolve_selection("t", "5") is None


def test_name_match():
    _seed("t", ("111", "一键红烧肉"), ("222", "东坡肉"))
    assert S.resolve_selection("t", "开始做一键红烧肉")["cookId"] == "111"


def test_no_candidates():
    S._last.clear()
    assert S.resolve_selection("none", "1") is None


def test_isolation_by_thread():
    S._last.clear()
    S.remember_candidates("a", _mk(("111", "X")))
    S.remember_candidates("b", _mk(("222", "Y")))
    assert S.resolve_selection("a", "1")["cookId"] == "111"
    assert S.resolve_selection("b", "1")["cookId"] == "222"


def test_pending():
    S.clear_pending("t")
    assert S.get_pending("t") is None
    S.set_pending("t", "999", "X", lang="en")
    pending = S.get_pending("t")
    assert pending["cookId"] == "999"
    assert pending["lang"] == "en"
    assert pending["action_id"]
    assert isinstance(pending["msg_id"], int)
    claimed = S.consume_pending(
        "t",
        expected_action_id=pending["action_id"],
    )
    assert claimed["msg_id"] == pending["msg_id"]
    assert S.consume_pending("t") is None
    S.clear_pending("t")
    assert S.get_pending("t") is None


def test_is_confirm():
    assert S.is_confirm("确认") and S.is_confirm("确认开火")
    assert S.is_confirm("confirm") and S.is_confirm("start cooking")
    assert not S.is_confirm("好") and not S.is_confirm("开始做")
    assert not S.is_confirm("想吃面条") and not S.is_confirm("取消")
    assert not S.is_confirm("Please don't start cooking")


def test_voice_control_normalization_accepts_only_unambiguous_commands():
    assert S.normalize_voice_control_text("[Voice] 确认。") == "确认"
    assert S.normalize_voice_control_text("确认，嗯。") == "确认"
    assert S.normalize_voice_control_text("嗯，确认。") == "确认"
    assert S.normalize_voice_control_text("[Voice] 停止。") == "停止"
    assert S.normalize_voice_control_text("烹饪，开始烹饪。") == "开始烹饪"
    assert S.normalize_voice_control_text("Please confirm, okay.") == "confirm"


def test_voice_control_normalization_does_not_turn_long_or_wrong_asr_into_commands():
    assert S.normalize_voice_control_text("[Voice] 承认第三道。") == "承认第三道。"
    assert S.normalize_voice_control_text("确认一下今天菜单") == "确认一下今天菜单"
    assert S.normalize_voice_control_text("[Voice] 今天有客户来访，推荐几道菜。") == "今天有客户来访，推荐几道菜。"
    assert not S.is_confirm(S.normalize_voice_control_text("[Voice] 承认第三道。"))
    assert not S.is_confirm(S.normalize_voice_control_text("确认一下今天菜单"))


def test_is_cancel():
    assert S.is_cancel("取消") and S.is_cancel("算了") and S.is_cancel("不")
    assert S.is_cancel("cancel")
    assert not S.is_cancel("确认")


def test_abandonment_is_not_an_explicit_device_stop():
    assert S.is_abandonment("好吧，我放弃")
    assert S.is_abandonment("算了吧")
    assert S.is_abandonment("I give up")
    assert not S.is_abandonment("停止烹饪")
    assert not S.is_stop("好吧，我放弃")


def test_is_stop():
    assert S.is_stop("停止") and S.is_stop("别做了") and S.is_stop("暂停")
    assert S.is_stop("stop") and S.is_stop("stop cooking")
    assert not S.is_stop("开始做") and not S.is_stop("红烧肉")


def test_resolve_device_choice():
    devs = [("default", "办公室设备"), ("second", "展厅设备")]
    assert S.resolve_device_choice("1", devs) == "default"
    assert S.resolve_device_choice("2", devs) == "second"
    assert S.resolve_device_choice("展厅设备", devs) == "second"
    assert S.resolve_device_choice("用第一台", devs) == "default"
    assert S.resolve_device_choice("用办公室那台", devs) == "default"
    assert S.resolve_device_choice("就用展厅那台", devs) == "second"
    assert S.resolve_device_choice("Use the Office Device", devs) == "default"
    assert S.resolve_device_choice("Choose Showroom Device.", devs) == "second"
    assert S.resolve_device_choice("第一步做什么", devs) is None
    assert S.resolve_device_choice("第二道怎么做", devs) is None
    assert S.resolve_device_choice("I work in the office", devs) is None
    assert S.resolve_device_choice("xyz", devs) is None
    assert S.resolve_device_choice("1", []) is None


def test_set_pending_with_devices():
    S.clear_pending("t")
    S.set_pending("t", "111", "金钱蛋", devices=[("default", "办公室设备"), ("second", "展厅设备")])
    p = S.get_pending("t")
    assert p["cookId"] == "111" and len(p["devices"]) == 2 and p["device_id"] is None
    S.set_pending("t", "111", "金钱蛋", device_id="default")
    p = S.get_pending("t")
    assert p["device_id"] == "default" and p["devices"] == []


def test_active_cooking_remembers_device():
    S.clear_active_cooking("t")
    assert S.get_active_cooking("t") is None
    S.set_active_cooking("t", "111", "红烧肉", device_id="second", lang="zh")
    rec = S.get_active_cooking("t")
    assert rec["cookId"] == "111"
    assert rec["device_id"] == "second"
    assert rec["lang"] == "zh"
    S.clear_active_cooking("t")
    assert S.get_active_cooking("t") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            ok += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    raise SystemExit(0 if ok == len(fns) else 1)
