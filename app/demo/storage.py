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
                    status TEXT NOT NULL, state TEXT NOT NULL DEFAULT '{}',
                    thread_id TEXT, parent_run_id TEXT,
                    checkpoint_thread_id TEXT,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    created REAL,
                    updated REAL
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
            columns = {
                str(row[1])
                for row in db.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "thread_id" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN thread_id TEXT")
            if "parent_run_id" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN parent_run_id TEXT")
            if "checkpoint_thread_id" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN checkpoint_thread_id TEXT")
            if "attempt" not in columns:
                db.execute(
                    "ALTER TABLE runs ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1"
                )
            if "created" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN created REAL")
            if "updated" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN updated REAL")
            now = time.time()
            db.execute("UPDATE runs SET attempt=1 WHERE attempt IS NULL")
            db.execute("UPDATE runs SET created=? WHERE created IS NULL", (now,))
            db.execute("UPDATE runs SET updated=created WHERE updated IS NULL")
            rows = db.execute(
                "SELECT id, version, thread_id FROM runs "
                "WHERE checkpoint_thread_id IS NULL"
            ).fetchall()
            for row in rows:
                db.execute(
                    "UPDATE runs SET checkpoint_thread_id=? WHERE id=?",
                    (
                        self.checkpoint_thread_id(
                            str(row["thread_id"] or f"web:{row['id']}"),
                            str(row["id"]),
                            int(row["version"]),
                        ),
                        row["id"],
                    ),
                )
            db.execute(
                "CREATE INDEX IF NOT EXISTS runs_thread ON runs(thread_id)"
            )
            latest = db.execute(
                "SELECT r.thread_id, r.id FROM runs r "
                "WHERE r.thread_id IS NOT NULL AND NOT EXISTS ("
                "SELECT 1 FROM chat_links c WHERE c.thread_id=r.thread_id) "
                "AND r.rowid=(SELECT MAX(r2.rowid) FROM runs r2 "
                "WHERE r2.thread_id=r.thread_id)"
            ).fetchall()
            for row in latest:
                db.execute(
                    "INSERT OR IGNORE INTO chat_links(thread_id, run_id) "
                    "VALUES (?, ?)",
                    (row["thread_id"], row["id"]),
                )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def checkpoint_thread_id(thread_id: str, run_id: str, version: int) -> str:
        return f"{thread_id}:workflow:{run_id}:v{version}"

    def create(
        self,
        run_id: str,
        brief: dict,
        *,
        thread_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        now = time.time()
        owner = str(thread_id or f"web:{run_id}")
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs(id, brief, version, status, thread_id, parent_run_id, "
                "checkpoint_thread_id, attempt, created, updated) "
                "VALUES (?, ?, 1, 'running', ?, ?, ?, 1, ?, ?)",
                (
                    run_id,
                    json.dumps(brief),
                    owner,
                    parent_run_id,
                    self.checkpoint_thread_id(owner, run_id, 1),
                    now,
                    now,
                ),
            )
            db.execute(
                "INSERT INTO chat_links(thread_id, run_id) VALUES (?, ?) "
                "ON CONFLICT(thread_id) DO UPDATE SET run_id=excluded.run_id",
                (owner, run_id),
            )

    def create_for_thread(
        self,
        run_id: str,
        brief: dict,
        *,
        thread_id: str,
        parent_run_id: str | None = None,
        replace_active: bool = False,
    ) -> tuple[str, bool]:
        """原子地绑定会话与 Run；同一会话只暴露一个当前运行。"""
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT runs.id, runs.status FROM chat_links "
                "JOIN runs ON runs.id=chat_links.run_id "
                "WHERE chat_links.thread_id=?",
                (thread_id,),
            ).fetchone()
            if (
                current
                and current["status"] in {"running", "awaiting_confirmation"}
                and not replace_active
            ):
                return str(current["id"]), False
            db.execute(
                "INSERT INTO runs(id, brief, version, status, thread_id, parent_run_id, "
                "checkpoint_thread_id, attempt, created, updated) "
                "VALUES (?, ?, 1, 'running', ?, ?, ?, 1, ?, ?)",
                (
                    run_id,
                    json.dumps(brief),
                    thread_id,
                    parent_run_id,
                    self.checkpoint_thread_id(thread_id, run_id, 1),
                    now,
                    now,
                ),
            )
            db.execute(
                "INSERT INTO chat_links(thread_id, run_id) VALUES (?, ?) "
                "ON CONFLICT(thread_id) DO UPDATE SET run_id=excluded.run_id",
                (thread_id, run_id),
            )
        return run_id, True

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
        now = time.time()
        with self.connect() as db:
            if state is None:
                db.execute(
                    "UPDATE runs SET status=?, updated=? WHERE id=?",
                    (status, now, run_id),
                )
            else:
                db.execute(
                    "UPDATE runs SET status=?, state=?, updated=? WHERE id=?",
                    (status, json.dumps(state), now, run_id),
                )

    def revise(self, run_id, brief):
        now = time.time()
        with self.connect() as db:
            row = db.execute(
                "SELECT thread_id, version FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None:
                return
            next_version = int(row["version"]) + 1
            owner = str(row["thread_id"] or f"web:{run_id}")
            db.execute(
                "UPDATE runs SET brief=?, version=?, status='running', state='{}', "
                "checkpoint_thread_id=?, attempt=1, updated=? WHERE id=?",
                (
                    json.dumps(brief),
                    next_version,
                    self.checkpoint_thread_id(owner, run_id, next_version),
                    now,
                    run_id,
                ),
            )

    def begin_retry(self, run_id: str) -> int:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status, attempt FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["status"] not in {"error", "interrupted"}:
                raise ValueError("run is not retryable")
            attempt = int(row["attempt"] or 1) + 1
            db.execute(
                "UPDATE runs SET status='running', attempt=?, updated=? WHERE id=?",
                (attempt, now, run_id),
            )
            return attempt

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
            db.execute(
                "UPDATE runs SET status='interrupted', updated=? WHERE status='running'",
                (time.time(),),
            )

    def link_chat(self, thread_id: str, run_id: str):
        with self.connect() as db:
            row = db.execute(
                "SELECT version FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            db.execute(
                "UPDATE runs SET thread_id=?, checkpoint_thread_id=?, updated=? "
                "WHERE id=?",
                (
                    thread_id,
                    self.checkpoint_thread_id(
                        thread_id,
                        run_id,
                        int(row["version"] if row else 1),
                    ),
                    time.time(),
                    run_id,
                ),
            )
            db.execute(
                "INSERT INTO chat_links(thread_id, run_id) VALUES (?, ?) "
                "ON CONFLICT(thread_id) DO UPDATE SET run_id=excluded.run_id",
                (thread_id, run_id),
            )

    def chat_run_id(self, thread_id: str) -> str | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT run_id FROM chat_links WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
            if row is None:
                row = db.execute(
                    "SELECT id AS run_id FROM runs WHERE thread_id=? "
                    "ORDER BY rowid DESC LIMIT 1",
                    (thread_id,),
                ).fetchone()
        return str(row["run_id"]) if row else None

    def metrics(self, run_id: str) -> dict:
        row = self.get(run_id)
        if row is None:
            return {}
        events = self.events(run_id)
        model_ends = [item for item in events if item["kind"] == "model_end"]
        node_ends = [item for item in events if item["kind"] == "node_end"]
        started = [item["created"] for item in events if item["kind"] == "run_start"]
        ended = [item["created"] for item in events if item["kind"] == "run_end"]
        wall_duration_ms = None
        if started:
            wall_duration_ms = round(
                (max(ended or [events[-1]["created"]]) - min(started)) * 1000
            )
        active_seconds = 0.0
        active_started = None
        for item in events:
            if item["kind"] == "run_start":
                active_started = float(item["created"])
            elif item["kind"] in {"run_end", "error"} and active_started is not None:
                active_seconds += max(0.0, float(item["created"]) - active_started)
                active_started = None
        if active_started is not None and events:
            active_seconds += max(0.0, float(events[-1]["created"]) - active_started)
        return {
            "attempt_count": int(row.get("attempt") or 1),
            "recovered": int(row.get("attempt") or 1) > 1,
            "event_count": len(events),
            "node_count": len(node_ends),
            "model_call_count": sum(
                1 for item in events if item["kind"] == "model_start"
            ),
            "tool_call_count": sum(
                1 for item in events if item["kind"] == "tool_start"
            ),
            "token_total": sum(
                int(item["data"].get("input_tokens") or 0)
                + int(item["data"].get("output_tokens") or 0)
                for item in model_ends
            ),
            "node_duration_ms": sum(
                int(item["data"].get("duration_ms") or 0) for item in node_ends
            ),
            "run_duration_ms": round(active_seconds * 1000) if started else None,
            "wall_duration_ms": wall_duration_ms,
            "error_count": sum(1 for item in events if item["kind"] == "error"),
        }

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
