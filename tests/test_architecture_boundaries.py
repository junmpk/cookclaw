"""单一编排根和包依赖方向的静态回归门禁。"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_files(relative: str) -> list[Path]:
    excluded = {".venv", "node_modules", "__pycache__"}
    return sorted(
        path
        for path in (ROOT / relative).rglob("*.py")
        if not excluded.intersection(path.relative_to(ROOT).parts)
    )


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _call_count(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        )
    )


def test_only_one_orchestration_package_exists():
    assert (ROOT / "app/orchestrator").is_dir()
    assert not (ROOT / "app/orchestration").exists()


def test_ports_do_not_import_concrete_packages():
    forbidden = ("app.conversation", "app.orchestrator", "app.agent")
    violations = []
    for path in _python_files("app/ports"):
        for imported in _imports(path):
            if imported.startswith(forbidden):
                violations.append(f"{path.relative_to(ROOT)} -> {imported}")
    assert violations == []


def test_conversation_does_not_depend_on_orchestration_or_agent():
    forbidden = ("app.orchestrator", "app.agent")
    violations = []
    for path in _python_files("app/conversation"):
        for imported in _imports(path):
            if imported.startswith(forbidden):
                violations.append(f"{path.relative_to(ROOT)} -> {imported}")
    assert violations == []


def test_runtime_has_one_turn_composition_root():
    application_calls = []
    orchestrator_calls = []
    turn_scope_calls = []
    for path in _python_files("app"):
        relative = str(path.relative_to(ROOT))
        application_calls.extend(
            [relative] * _call_count(path, "TurnApplicationService")
        )
        orchestrator_calls.extend([relative] * _call_count(path, "TurnOrchestrator"))
        turn_scope_calls.extend([relative] * _call_count(path, "turn_scope"))

    assert application_calls == ["app/orchestrator/turn/facade.py"]
    assert orchestrator_calls == ["app/orchestrator/turn/application_service.py"]
    assert turn_scope_calls == ["app/orchestrator/turn/application_service.py"]


def test_image_domain_handler_reuses_outer_turn_request():
    path = ROOT / "app/orchestrator/turn/image_handler.py"
    assert _call_count(path, "TurnRequest") == 0


def test_private_release_manifests_are_excluded_from_public_snapshot():
    for relative in ("scripts/package.sh", "scripts/replace_release.sh"):
        assert not (ROOT / relative).exists()
