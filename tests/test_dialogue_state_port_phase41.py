"""阶段 4.1 DialogueState Port 回归。"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from app.conversation.models import ConversationTaskState
from app.conversation.dialogue_state_workspace import (
    TurnScopedDialogueStatePort,
    WorkspaceDialogueStatePort,
)
from app.ports import dialogue_state


class _FakeStatePort(WorkspaceDialogueStatePort):
    def __init__(self, label: str):
        self.label = label
        self.pending = {}

    def set_pending(self, thread_id, cook_id, name, **kwargs):
        self.pending[thread_id] = {
            "cookId": cook_id,
            "name": name,
            "port": self.label,
            **kwargs,
        }

    def get_pending(self, thread_id, max_age=300):
        return self.pending.get(thread_id)


def test_state_port_scope_delegates_and_restores_default_backend():
    default_port = dialogue_state.get_dialogue_state_port()
    fake = _FakeStatePort("fake")

    with dialogue_state.dialogue_state_port_scope(fake):
        dialogue_state.set_pending("thread", "r1", "番茄炒蛋", lang="zh")
        assert dialogue_state.get_pending("thread") == {
            "cookId": "r1",
            "name": "番茄炒蛋",
            "port": "fake",
            "lang": "zh",
        }
        assert dialogue_state.get_dialogue_state_port() is fake

    assert dialogue_state.get_dialogue_state_port() is default_port


def test_state_port_context_isolated_between_async_turns():
    async def run(label: str):
        port = _FakeStatePort(label)
        with dialogue_state.dialogue_state_port_scope(port):
            dialogue_state.set_pending(label, "r1", label)
            await asyncio.sleep(0)
            return dialogue_state.get_pending(label)["port"]

    async def gather():
        return await asyncio.gather(run("turn-a"), run("turn-b"))

    assert asyncio.run(gather()) == ["turn-a", "turn-b"]


def test_orchestrator_entrypoints_do_not_import_task_workspace_directly():
    project_root = Path(__file__).resolve().parents[1]
    migrated = (
        project_root / "app/agent/participle_agent.py",
        project_root / "app/orchestrator/turn/application_service.py",
        project_root / "app/orchestrator/turn/execution_journal.py",
        project_root / "app/conversation/task_state_repository.py",
        project_root / "app/main.py",
    )

    for path in migrated:
        source = path.read_text(encoding="utf-8")
        assert "from app.conversation.task_state_workspace import" not in source
        assert "from app.conversation import task_state_workspace" not in source


def test_default_port_preserves_existing_snapshot_and_restore_semantics():
    thread_id = "phase41-default-port"
    dialogue_state.clear_thread(thread_id)
    dialogue_state.remember_candidates(
        thread_id,
        {
            "results": [
                {
                    "id": "recipe-1",
                    "metadata": {
                        "recipe_id": "recipe-1",
                        "name": "番茄炒蛋",
                    },
                }
            ]
        },
        lang="zh",
    )
    before = dialogue_state.snapshot_thread_state(thread_id)
    dialogue_state.clear_thread(thread_id)
    assert dialogue_state.recall_candidate_context(thread_id) is None

    dialogue_state.restore_thread_state(thread_id, before)
    restored = dialogue_state.recall_candidate_context(thread_id)

    assert restored["items"][0]["cookId"] == "recipe-1"
    assert dialogue_state.resolve_selection(thread_id, "第一个")["name"] == "番茄炒蛋"
    dialogue_state.clear_thread(thread_id)


def test_default_port_exposes_device_execution_lifecycle():
    thread_id = "phase41-device-execution"
    dialogue_state.clear_device_execution(thread_id)
    execution = {
        "cookId": "recipe-1",
        "name": "番茄炒蛋",
        "device_id": "office",
        "action_id": "phase41-action",
        "msg_id": 4101,
        "lang": "zh",
        "status": "dispatching",
        "updated_at": time.time(),
    }

    try:
        dialogue_state.set_device_execution(thread_id, execution)
        stored = dialogue_state.get_device_execution(thread_id)
        assert stored is not None
        assert stored["action_id"] == "phase41-action"
        assert stored["status"] == "dispatching"

        assert dialogue_state.update_device_execution(
            thread_id,
            expected_action_id="phase41-action",
            status="submitted_unverified",
            result_code="DEVICE_STATE_UNVERIFIED",
            command_sent=True,
            started=None,
        ) is True
        updated = dialogue_state.get_device_execution(thread_id)
        assert updated is not None
        assert updated["status"] == "submitted_unverified"
        assert updated["result_code"] == "DEVICE_STATE_UNVERIFIED"
        assert updated["command_sent"] is True
        assert updated["started"] is None
    finally:
        dialogue_state.clear_device_execution(thread_id)

    assert dialogue_state.get_device_execution(thread_id) is None


def test_turn_scoped_port_replace_and_close_clear_internal_device_execution():
    thread_id = "phase41-turn-device-execution"
    initial = ConversationTaskState(
        device_execution={
            "action_id": "initial-action",
            "device_id": "office",
            "status": "dispatching",
            "updated_at": time.time(),
        }
    )
    port = TurnScopedDialogueStatePort(thread_id, initial)
    internal_thread_id = port._internal_thread_id

    assert port.get_device_execution(thread_id)["action_id"] == "initial-action"
    port.replace(ConversationTaskState())
    assert port.get_device_execution(thread_id) is None

    port.set_device_execution(
        thread_id,
        {
            "action_id": "replacement-action",
            "device_id": "showroom",
            "status": "outcome_unknown",
            "updated_at": time.time(),
        },
    )
    assert port.get_device_execution(thread_id)["action_id"] == "replacement-action"
    port.close()

    backend = WorkspaceDialogueStatePort()
    assert backend.get_device_execution(internal_thread_id) is None
