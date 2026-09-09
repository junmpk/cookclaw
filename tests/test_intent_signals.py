"""语义信号（signals）单元测试。

验证：
1. IntentResult.from_raw 正确解析 signals 字段
2. router 的决策函数优先消费 signals、正则兜底
"""
import pytest
from app.orchestrator.intent import IntentResult, IntentCategory


# ── 1. IntentResult 解析 signals ──────────────────────────────────────


class TestIntentResultSignalParsing:
    """from_raw 应正确解析 signals 字段到对应的 IntentResult 属性。"""

    def test_no_signals_field_defaults_to_none(self):
        data = {"r": True, "c": "recipe_search", "a": "new_search", "k": ["红烧肉"]}
        result = IntentResult.from_raw(data)
        assert result.is_finalize_request is None
        assert result.is_no_constraints is None
        assert result.is_constraint_reply is None
        assert result.is_topic_change is None
        assert result.is_preference_affirm is None
        assert result.is_preference_negate is None

    def test_empty_signals_defaults_to_none(self):
        data = {"r": True, "c": "recipe_search", "a": "new_search", "k": [], "signals": {}}
        result = IntentResult.from_raw(data)
        assert result.is_finalize_request is None
        assert result.is_topic_change is None

    def test_finalize_signal_parsed(self):
        data = {
            "r": True, "c": "recipe_recommend", "a": "refine_search",
            "k": [], "signals": {"finalize": True},
        }
        result = IntentResult.from_raw(data)
        assert result.is_finalize_request is True
        assert result.is_no_constraints is None

    def test_no_constraints_signal_parsed(self):
        data = {
            "r": True, "c": "recipe_recommend", "a": "refine_search",
            "k": [], "signals": {"no_constraints": True, "constraint_reply": True},
        }
        result = IntentResult.from_raw(data)
        assert result.is_no_constraints is True
        assert result.is_constraint_reply is True

    def test_topic_change_signal_parsed(self):
        data = {
            "r": True, "c": "recipe_search", "a": "new_search",
            "k": ["鸡胸肉"],
            "s": {"q": "鸡胸肉 清淡", "ingredient": ["鸡胸肉"], "flavor": ["清淡"]},
            "signals": {"topic_change": True},
        }
        result = IntentResult.from_raw(data)
        assert result.is_topic_change is True
        assert result.is_finalize_request is None

    def test_preference_signals_parsed(self):
        data = {
            "r": True, "c": "recipe_recommend", "a": "refine_search",
            "k": [], "signals": {"preference_affirm": True},
        }
        result = IntentResult.from_raw(data)
        assert result.is_preference_affirm is True
        assert result.is_preference_negate is None

    def test_false_signals_are_not_none(self):
        """明确 false 的信号应保留为 False（不是 None），表示分类器已判断。"""
        data = {
            "r": True, "c": "recipe_search", "a": "new_search",
            "k": ["红烧肉"], "signals": {"finalize": False, "topic_change": False},
        }
        result = IntentResult.from_raw(data)
        assert result.is_finalize_request is False
        assert result.is_topic_change is False

    def test_invalid_signals_type_ignored(self):
        """signals 字段如果不是 dict，应被忽略。"""
        data = {"r": True, "c": "recipe_search", "a": "new_search", "k": [], "signals": "invalid"}
        result = IntentResult.from_raw(data)
        assert result.is_finalize_request is None


# ── 2. Router 决策函数消费 signals ────────────────────────────────────


class TestRouterSignalConsumption:
    """router 的决策函数应优先使用语义信号，信号缺失时回退正则。"""

    def test_is_clarification_reply_finalize_signal_overrides_regex(self):
        """finalize=True 时，即使正则匹配也不视为澄清回复。"""
        from app.orchestrator.router import _is_clarification_reply

        intent = IntentResult(
            category=IntentCategory.recipe_recommend,
            is_finalize_request=True,
        )
        # "清淡一点" 在正则中匹配 taste_or_constraint，但 finalize 信号应覆盖
        result = _is_clarification_reply(intent, "清淡一点", "flavor_preferences")
        assert result is False

    def test_is_clarification_reply_constraint_signal_true(self):
        """constraint_reply=True 时直接返回 True。"""
        from app.orchestrator.router import _is_clarification_reply

        intent = IntentResult(
            category=IntentCategory.recipe_recommend,
            is_constraint_reply=True,
        )
        result = _is_clarification_reply(intent, "随便什么都行", "party_constraints")
        assert result is True

    def test_is_clarification_reply_topic_change_overrides(self):
        """topic_change=True 时不视为澄清回复。"""
        from app.orchestrator.router import _is_clarification_reply

        intent = IntentResult(
            category=IntentCategory.recipe_search,
            is_topic_change=True,
        )
        result = _is_clarification_reply(intent, "鸡胸肉能做什么", "flavor_preferences")
        assert result is False

    def test_is_clarification_reply_falls_back_to_regex(self):
        """所有信号为 None 时回退正则。"""
        from app.orchestrator.router import _is_clarification_reply

        intent = IntentResult(category=IntentCategory.recipe_recommend)
        # "辣" 在正则中匹配 taste_or_constraint
        result = _is_clarification_reply(intent, "辣", "flavor_preferences")
        assert result is True

    def test_no_constraints_signal_overrides_regex(self):
        """is_no_constraints=True 时，即使正则不匹配也返回 True。"""
        from app.orchestrator.router import _is_contextual_no_constraints_reply

        intent = IntentResult(is_no_constraints=True)
        # 一段正则不匹配的文本
        result = _is_contextual_no_constraints_reply(
            "我们啥都不忌口，你安排就好",
            "party_constraints",
            intent,
        )
        assert result is True

    def test_no_constraints_signal_false_blocks_regex(self):
        """is_no_constraints=False 时，即使正则匹配也返回 False。"""
        from app.orchestrator.router import _is_contextual_no_constraints_reply

        intent = IntentResult(is_no_constraints=False)
        result = _is_contextual_no_constraints_reply(
            "没有忌口",
            "party_constraints",
            intent,
        )
        assert result is False

    def test_no_constraints_fallback_to_regex_when_signal_none(self):
        """信号为 None 时回退正则。"""
        from app.orchestrator.router import _is_contextual_no_constraints_reply

        result = _is_contextual_no_constraints_reply(
            "没有忌口",
            "party_constraints",
            None,
        )
        assert result is True

    def test_preference_affirm_signal(self):
        """preference_affirm=True 时视为澄清回复。"""
        from app.orchestrator.router import _is_clarification_reply

        intent = IntentResult(
            category=IntentCategory.recipe_recommend,
            is_preference_affirm=True,
        )
        result = _is_clarification_reply(
            intent, "对，以后叫我老周", "remembered_preference_conflict",
        )
        assert result is True

    def test_preference_negate_signal(self):
        """preference_negate=True 时视为澄清回复。"""
        from app.orchestrator.router import _is_clarification_reply

        intent = IntentResult(
            category=IntentCategory.recipe_recommend,
            is_preference_negate=True,
        )
        result = _is_clarification_reply(
            intent, "算了不用了", "remembered_preference_conflict",
        )
        assert result is True
