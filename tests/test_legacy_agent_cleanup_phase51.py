"""阶段 5.1：旧完整 Agent 与失真运行时资产清理回归。"""

from pathlib import Path

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_participle_agent_no_longer_constructs_shell_skill_agents():
    import app.agent.participle_agent as module

    source = (PROJECT_ROOT / "app/agent/participle_agent.py").read_text(
        encoding="utf-8"
    )

    assert not hasattr(module, "agent")
    assert not hasattr(module, "qqbot_agent")
    assert not hasattr(module, "chat")
    assert "LocalShellBackend" not in source
    assert "MemorySaver" not in source
    assert "create_deep_agent(" not in source
    assert "_controlled_agent_llm" in source


def test_obsolete_agent_prompt_and_memory_assets_are_removed():
    removed_files = [
        "app/core/system_prompt.md",
        "app/core/qqbot_prompt.md",
        "app/memory/AGENTS.md",
        "app/agent/skills/SKILL.md",
    ]
    removed_skill = (
        PROJECT_ROOT / "app/agent/skills/self-improving-agent-3.0.16"
    )

    assert all(not (PROJECT_ROOT / value).exists() for value in removed_files)
    assert not any(path.is_file() for path in removed_skill.rglob("*"))


def test_active_mode_does_not_emit_stale_not_implemented_error(
    monkeypatch,
    caplog,
):
    from app.orchestrator.planning.shadow_compare import planner_shadow_enabled

    monkeypatch.setattr(settings, "TURN_PLANNER_MODE", "active")

    assert planner_shadow_enabled() is False
    assert "not implemented" not in caplog.text.lower()
