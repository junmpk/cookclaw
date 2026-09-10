"""SQLite 保存演示会话、真实事件及 Mock 执行账本。与图 checkpoint 分工明确。"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class DemoStorage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, brief TEXT NOT NULL, version INTEGER NOT NULL,
                    status TEXT NOT NULL, state TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    version INTEGER NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL,
                    data TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_run ON events(run_id, id);
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY, result TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_links (
                    thread_id TEXT PRIMARY KEY, run_id TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, run_id: str, brief: dict):
        with self.connect() as db:
            db.execute("INSERT INTO runs(id, brief, version, status) VALUES (?, ?, 1, 'running')",
                       (run_id, json.dumps(brief)))

    def get(self, run_id: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["brief"] = json.loads(result["brief"])
        result["state"] = json.loads(result["state"])
        return result

    def update(self, run_id, status, state=None):
        with self.connect() as db:
            if state is None:
                db.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))
            else:
                db.execute("UPDATE runs SET status=?, state=? WHERE id=?", (status, json.dumps(state), run_id))

    def revise(self, run_id, brief):
        with self.connect() as db:
            db.execute("UPDATE runs SET brief=?, version=version+1, status='running', state='{}' WHERE id=?",
                       (json.dumps(brief), run_id))

    def event(self, run_id, version, role, kind, data):
        with self.connect() as db:
            db.execute("INSERT INTO events(run_id,version,role,kind,data,created) VALUES (?,?,?,?,?,?)",
                       (run_id, version, role, kind, json.dumps(data, ensure_ascii=False), time.time()))

    def events(self, run_id, after=0):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT 1000", (run_id, after)).fetchall()
        return [dict(row) | {"data": json.loads(row["data"])} for row in rows]

    def mark_interrupted(self):
        with self.connect() as db:
            db.execute("UPDATE runs SET status='interrupted' WHERE status='running'")

    def link_chat(self, thread_id: str, run_id: str):
        with self.connect() as db:
            db.execute(
                "INSERT INTO chat_links(thread_id, run_id) VALUES (?, ?) "
                "ON CONFLICT(thread_id) DO UPDATE SET run_id=excluded.run_id",
                (thread_id, run_id),
            )

    def chat_run_id(self, thread_id: str) -> str | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT run_id FROM chat_links WHERE thread_id=?", (thread_id,)
            ).fetchone()
        return str(row["run_id"]) if row else None

    def execute_mock(self, execution_id: str, recipe_ids: list[str]):
        """模拟操作和账本在同一事务内提交；重复确认返回同一条记录。"""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT result FROM executions WHERE execution_id=?", (execution_id,)).fetchone()
            if existing:
                return json.loads(existing["result"])
            result = {"execution_id": execution_id, "provider": "mock", "status": "completed",
                      "recipe_ids": recipe_ids, "message": "模拟执行完成，未连接或控制真实设备。"}
            db.execute("INSERT INTO executions VALUES (?, ?)", (execution_id, json.dumps(result)))
            return result
