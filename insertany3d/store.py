"""SQLite persistence for the local InsertAny3D control plane."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 4


class StoreError(RuntimeError):
    pass


class SchedulerStore:
    """A small single-controller SQLite store with explicit transactions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        # Multiple CLI processes may briefly open a new state database together
        # (the Farm wrapper starts its monitor beside the worker).  Wait for a
        # writer instead of failing immediately during WAL initialization.
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def __enter__(self) -> SchedulerStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
            except BaseException:
                self.connection.rollback()
                raise
            else:
                self.connection.commit()

    def event(
        self,
        connection: sqlite3.Connection,
        batch_id: str,
        kind: str,
        payload: dict[str, Any],
        created_at: float,
        *,
        stage_id: int | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO events(batch_id, stage_id, kind, payload_json, created_at) VALUES(?, ?, ?, ?, ?)",
            (batch_id, stage_id, kind, _json(payload), created_at),
        )

    def rows(self, query: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.connection.execute(query, parameters))

    def row(self, query: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.connection.execute(query, parameters).fetchone()

    def _migrate(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise StoreError(f"数据库版本 {version} 高于当前支持的 {SCHEMA_VERSION}")
        if version == 0:
            with self.transaction() as connection:
                connection.executescript(
                    """
                    CREATE TABLE batches (
                        batch_id TEXT PRIMARY KEY,
                        manifest_sha256 TEXT NOT NULL,
                        manifest_json TEXT NOT NULL,
                        root_path TEXT NOT NULL,
                        status TEXT NOT NULL,
                        edit_mode TEXT NOT NULL,
                        review_batch_size INTEGER NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE projects (
                        batch_id TEXT NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
                        project_id TEXT NOT NULL,
                        project_path TEXT NOT NULL,
                        scene_path TEXT NOT NULL,
                        sort_index INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        PRIMARY KEY(batch_id, project_id)
                    );
                    CREATE TABLE tasks (
                        batch_id TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        sort_index INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        last_dispatched_at REAL,
                        terminal_error_code TEXT,
                        PRIMARY KEY(batch_id, project_id, task_id),
                        FOREIGN KEY(batch_id, project_id) REFERENCES projects(batch_id, project_id) ON DELETE CASCADE
                    );
                    CREATE TABLE stages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        sort_index INTEGER NOT NULL,
                        state TEXT NOT NULL,
                        contract_version TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        effective_config_json TEXT NOT NULL,
                        max_attempts INTEGER NOT NULL,
                        timeout_seconds INTEGER NOT NULL,
                        not_before REAL NOT NULL DEFAULT 0,
                        last_error_code TEXT,
                        last_message TEXT,
                        updated_at REAL NOT NULL,
                        UNIQUE(batch_id, project_id, task_id, name),
                        FOREIGN KEY(batch_id, project_id, task_id) REFERENCES tasks(batch_id, project_id, task_id) ON DELETE CASCADE
                    );
                    CREATE INDEX stages_queue ON stages(batch_id, state, not_before, sort_index);
                    CREATE TABLE stage_dependencies (
                        stage_id INTEGER NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
                        depends_on_stage_id INTEGER NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
                        PRIMARY KEY(stage_id, depends_on_stage_id)
                    );
                    CREATE TABLE attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stage_id INTEGER NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
                        attempt_number INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        worker_id TEXT NOT NULL,
                        staging_dir TEXT NOT NULL,
                        output_dir TEXT NOT NULL,
                        started_at REAL,
                        finished_at REAL,
                        error_code TEXT,
                        message TEXT,
                        cleanup_status TEXT,
                        UNIQUE(stage_id, attempt_number)
                    );
                    CREATE TABLE leases (
                        stage_id INTEGER PRIMARY KEY REFERENCES stages(id) ON DELETE CASCADE,
                        attempt_id INTEGER UNIQUE NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
                        token TEXT UNIQUE NOT NULL,
                        worker_id TEXT NOT NULL,
                        resources_json TEXT NOT NULL,
                        acquired_at REAL NOT NULL,
                        heartbeat_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        pid INTEGER,
                        pgid INTEGER,
                        host_boot_id TEXT,
                        process_start_ticks INTEGER,
                        revoked_at REAL
                    );
                    CREATE TABLE artifacts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        artifact_id TEXT NOT NULL,
                        stage_id INTEGER NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
                        attempt_id INTEGER NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
                        type TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        created_at REAL NOT NULL,
                        UNIQUE(stage_id, artifact_id, sha256)
                    );
                    CREATE TABLE edit_reviews (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        edit_attempt INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        decision TEXT,
                        decided_by TEXT,
                        decided_at REAL,
                        policy_version TEXT,
                        note TEXT,
                        created_at REAL NOT NULL,
                        UNIQUE(batch_id, project_id, task_id, edit_attempt)
                    );
                    CREATE TABLE api_buckets (
                        bucket_key TEXT PRIMARY KEY,
                        token_fingerprint TEXT NOT NULL,
                        model TEXT NOT NULL,
                        current_limit INTEGER NOT NULL,
                        maximum_limit INTEGER NOT NULL,
                        clean_successes INTEGER NOT NULL DEFAULT 0,
                        cooldown_until REAL NOT NULL DEFAULT 0,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
                        stage_id INTEGER REFERENCES stages(id) ON DELETE CASCADE,
                        kind TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX events_batch ON events(batch_id, id);
                    CREATE TABLE gc_journal (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
                        owner_token TEXT NOT NULL,
                        target_path TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        dry_run INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        finished_at REAL
                    );
                    PRAGMA user_version = 1;
                    """
                )
            version = 1
        if version == 1:
            with self.transaction() as connection:
                connection.executescript(
                    """
                    CREATE TABLE artifact_commits (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stage_id INTEGER NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
                        attempt_id INTEGER UNIQUE NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
                        lease_token TEXT NOT NULL,
                        staging_dir TEXT NOT NULL,
                        output_dir TEXT NOT NULL,
                        artifacts_json TEXT NOT NULL,
                        state TEXT NOT NULL,
                        error TEXT,
                        prepared_at REAL NOT NULL,
                        finished_at REAL
                    );
                    CREATE INDEX artifact_commits_stage ON artifact_commits(stage_id, state);
                    CREATE TABLE api_bucket_owners (
                        bucket_key TEXT PRIMARY KEY REFERENCES api_buckets(bucket_key) ON DELETE CASCADE,
                        owner_token TEXT NOT NULL,
                        pid INTEGER NOT NULL,
                        host_boot_id TEXT NOT NULL,
                        process_start_ticks INTEGER NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    PRAGMA user_version = 2;
                    """
                )
            version = 2
        if version == 2:
            with self.transaction() as connection:
                connection.executescript(
                    """
                    ALTER TABLE attempts ADD COLUMN generation_group_id TEXT;
                    ALTER TABLE attempts ADD COLUMN generation_index INTEGER;
                    ALTER TABLE stages ADD COLUMN aggregate_started_at REAL;
                    ALTER TABLE stages ADD COLUMN aggregate_finished_at REAL;
                    CREATE TABLE IF NOT EXISTS image_edit_generations (
                        stage_id INTEGER NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
                        group_id TEXT NOT NULL,
                        generation_index INTEGER NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        attempt_id INTEGER REFERENCES attempts(id) ON DELETE SET NULL,
                        attempt_number INTEGER NOT NULL DEFAULT 0,
                        output_json TEXT,
                        error_code TEXT,
                        started_at REAL,
                        finished_at REAL,
                        PRIMARY KEY(stage_id, group_id, generation_index)
                    );
                    CREATE INDEX IF NOT EXISTS image_edit_generations_stage ON image_edit_generations(stage_id, group_id, status);
                    PRAGMA user_version = 3;
                    """
                )
            version = 3
        if version == 3:
            with self.transaction() as connection:
                connection.executescript(
                    """
                    CREATE TABLE leases_v4 (
                        stage_id INTEGER NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
                        attempt_id INTEGER UNIQUE NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
                        token TEXT UNIQUE NOT NULL,
                        worker_id TEXT NOT NULL,
                        resources_json TEXT NOT NULL,
                        acquired_at REAL NOT NULL,
                        heartbeat_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        pid INTEGER,
                        pgid INTEGER,
                        host_boot_id TEXT,
                        process_start_ticks INTEGER,
                        revoked_at REAL
                    );
                    INSERT INTO leases_v4 SELECT stage_id, attempt_id, token, worker_id, resources_json,
                        acquired_at, heartbeat_at, expires_at, pid, pgid, host_boot_id, process_start_ticks, revoked_at FROM leases;
                    DROP TABLE leases;
                    ALTER TABLE leases_v4 RENAME TO leases;
                    CREATE INDEX leases_stage ON leases(stage_id, revoked_at);
                    PRAGMA user_version = 4;
                    """
                )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
