"""受控 Deep Agent 灰度必须稳定、按用户隔离且默认不放量。"""

from app.agent import deep_agent_rollout as rollout


def _configure(
    monkeypatch,
    *,
    enabled=True,
    channels="qq",
    allow_users="",
    percentage=0,
):
    monkeypatch.setattr(
        rollout.settings,
        "CONVERSATION_DEEP_AGENT_ENABLED",
        enabled,
    )
    monkeypatch.setattr(
        rollout.settings,
        "CONVERSATION_DEEP_AGENT_CHANNELS",
        channels,
    )
    monkeypatch.setattr(
        rollout.settings,
        "CONVERSATION_DEEP_AGENT_QQ_ALLOW_FROM",
        allow_users,
    )
    monkeypatch.setattr(
        rollout.settings,
        "CONVERSATION_DEEP_AGENT_ROLLOUT_PERCENT",
        percentage,
    )


def test_rollout_is_off_when_master_switch_is_disabled(monkeypatch):
    _configure(
        monkeypatch,
        enabled=False,
        allow_users="selected-user",
        percentage=100,
    )
    decision = rollout.deep_agent_rollout_decision(
        "qq:dm:chat:selected-user",
    )
    assert decision.enabled is False
    assert decision.cohort == "master_disabled"


def test_qq_allowlist_selects_only_named_user(monkeypatch):
    _configure(monkeypatch, allow_users="alice,bob", percentage=0)
    selected = rollout.deep_agent_rollout_decision("qq:dm:chat:alice")
    excluded = rollout.deep_agent_rollout_decision("qq:dm:chat:carol")
    assert selected.enabled is True
    assert selected.cohort == "qq_allowlist"
    assert excluded.enabled is False
    assert excluded.cohort == "not_selected"


def test_channel_gate_precedes_user_allowlist(monkeypatch):
    _configure(
        monkeypatch,
        channels="weixin",
        allow_users="alice",
        percentage=100,
    )
    decision = rollout.deep_agent_rollout_decision("qq:dm:chat:alice")
    assert decision.enabled is False
    assert decision.cohort == "channel_excluded"


def test_zero_percent_is_safe_default_and_full_percent_is_explicit(monkeypatch):
    _configure(monkeypatch, percentage=0)
    disabled = rollout.deep_agent_rollout_decision("qq:dm:chat:any-user")
    assert disabled.enabled is False

    _configure(monkeypatch, percentage=100)
    enabled = rollout.deep_agent_rollout_decision("qq:dm:chat:any-user")
    assert enabled.enabled is True
    assert enabled.cohort == "stable_percentage"


def test_percentage_bucket_is_stable_for_same_user(monkeypatch):
    _configure(monkeypatch, percentage=37.5)
    first = rollout.deep_agent_rollout_decision("qq:dm:chat:stable-user")
    second = rollout.deep_agent_rollout_decision("qq:group:other:stable-user")
    assert first == second
    assert first.cohort in {"stable_percentage", "not_selected"}
