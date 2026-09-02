"""Persistent fair scheduler with lease fencing and edit review gates."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import ContractError, validate_batch_manifest
from .contracts.models import canonical_sha256
from .dag import (
    REMOTE_EVIDENCE_END,
    REMOTE_EVIDENCE_START,
    REMOTE_PROCESS_STAGES,
    STAGE_BY_NAME,
    STAGE_INDEX,
    STAGES,
    image_api_limits,
)
from .processes import ProcessIdentity, ProcessSupervisor, current_boot_id as host_boot_id
from .store import SchedulerStore, StoreError
from .image_edit import generation_count


ACTIVE_STATES = frozenset({"leased", "running", "committing", "suspect", "recovering"})
TERMINAL_STATES = frozenset({"succeeded", "failed_terminal", "rejected", "canceled"})
RETRYABLE_ERRORS = frozenset({"transient_network", "resource_oom", "worker_crash", "stalled", "http_429", "http_503"})
TERMINAL_ERRORS = frozenset({"invalid_input", "compile_or_contract", "http_400", "http_403"})
EVALUATION_SKIPPED_ERROR = "evaluation_skipped_incomplete_batch"


# Status output groups stages by the resource that determines their queue.
# Stages that acquire multiple resources use the most user-visible resource
# (for example, Unity stages use the Unity slot rather than project_lock).
_STATUS_RESOURCE_BY_STAGE = {
    "image_edit": "image_api",
    "upload_inputs": "upload",
    "download_results": "download",
    "unity_anchor": "unity",
    "unity_apply": "unity",
    "unity_eval6": "unity",
    "evaluate_absolute": "evaluation_api",
}


def status_resource_for_stage(stage_name: str | None) -> str | None:
    """Return the queue/resource group used by the human status display."""

    if not stage_name:
        return None
    explicit = _STATUS_RESOURCE_BY_STAGE.get(stage_name)
    if explicit:
        return explicit
    spec = STAGE_BY_NAME.get(stage_name)
    if spec is None or not spec.resources:
        return None
    return spec.resources[0]


class LeaseFencedError(StoreError):
    """The caller no longer owns the stage lease."""


@dataclass(frozen=True)
class WorkItem:
    stage_id: int
    batch_id: str
    project_id: str
    task_id: str
    stage: str
    contract_version: str
    attempt: int
    attempt_id: int
    lease_token: str
    resources: dict[str, str]
    staging_dir: Path
    output_dir: Path
    expires_at: float
    generation_group_id: str | None = None
    generation_index: int | None = None


class BatchController:
    def __init__(
        self,
        store: SchedulerStore,
        *,
        clock: Callable[[], float] = time.time,
        lease_seconds: float = 60.0,
        process_supervisor: ProcessSupervisor | None = None,
    ):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds 必须大于 0")
        self.store = store
        self.clock = clock
        self.lease_seconds = lease_seconds
        self.process_supervisor = process_supervisor or ProcessSupervisor()

    def plan(self, manifest: Mapping[str, Any], root: str | Path, *, formal: bool = True) -> str:
        data = validate_batch_manifest(manifest, formal=formal)
        now = self.clock()
        batch_id = str(data["batchId"])
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        manifest_digest = canonical_sha256(data)
        with self.store.transaction() as connection:
            existing = connection.execute("SELECT manifest_sha256 FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            if existing:
                if existing["manifest_sha256"] != manifest_digest:
                    raise StoreError(f"batch {batch_id} 已存在且 manifest 不同；请使用新的 batchId")
                return batch_id
            policy = data["editPolicy"]
            connection.execute(
                "INSERT INTO batches VALUES(?, ?, ?, ?, 'planned', ?, ?, ?, ?)",
                (
                    batch_id,
                    manifest_digest,
                    _json(data),
                    str(root_path),
                    policy["mode"],
                    int(policy["reviewBatchSize"]),
                    now,
                    now,
                ),
            )
            for project_index, project in enumerate(data["projects"]):
                project_id = str(project["projectId"])
                connection.execute(
                    "INSERT INTO projects VALUES(?, ?, ?, ?, ?, 'pending')",
                    (batch_id, project_id, project["projectPath"], project["scenePath"], project_index),
                )
                for task_index, task_value in enumerate(project["tasks"]):
                    task_id = task_value if isinstance(task_value, str) else task_value["taskId"]
                    connection.execute(
                        "INSERT INTO tasks VALUES(?, ?, ?, ?, 'pending', NULL, NULL)",
                        (batch_id, project_id, task_id, task_index),
                    )
                    stage_ids: list[int] = []
                    for stage_index, spec in enumerate(STAGES):
                        config = self._effective_stage_config(data, project, task_value, spec.name)
                        idem = _idempotency_key(batch_id, project_id, task_id, spec.name, spec.contract_version, [], config)
                        cursor = connection.execute(
                            """INSERT INTO stages(
                                batch_id, project_id, task_id, name, sort_index, state, contract_version,
                                idempotency_key, effective_config_json, max_attempts, timeout_seconds, updated_at
                            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                batch_id,
                                project_id,
                                task_id,
                                spec.name,
                                stage_index,
                                "ready" if stage_index == 0 else "pending",
                                spec.contract_version,
                                idem,
                                _json(config),
                                spec.max_attempts,
                                spec.timeout_seconds,
                                now,
                            ),
                        )
                        stage_ids.append(int(cursor.lastrowid))
                    for index in range(1, len(stage_ids)):
                        connection.execute(
                            "INSERT INTO stage_dependencies(stage_id, depends_on_stage_id) VALUES(?, ?)",
                            (stage_ids[index], stage_ids[index - 1]),
                        )
            self.store.event(connection, batch_id, "batch_planned", {"manifestSha256": manifest_digest}, now)
        return batch_id

    def start(self, batch_id: str, *, formal: bool = True) -> None:
        now = self.clock()
        with self.store.transaction() as connection:
            batch = _require_batch(connection, batch_id)
            if batch["status"] in {"canceled", "succeeded", "failed"}:
                raise StoreError(f"batch {batch_id} 已处于终态 {batch['status']}")
            manifest = json.loads(batch["manifest_json"])
            validate_batch_manifest(manifest, formal=formal)
            connection.execute("UPDATE batches SET status='running', updated_at=? WHERE batch_id=?", (now, batch_id))
            connection.execute("UPDATE projects SET status='running' WHERE batch_id=? AND status='pending'", (batch_id,))
            connection.execute("UPDATE tasks SET status='running' WHERE batch_id=? AND status='pending'", (batch_id,))
            self.store.event(connection, batch_id, "batch_started", {}, now)

    def resume(self, batch_id: str, *, current_boot_id: str | None = None) -> int:
        self.reconcile_artifact_commits(batch_id)
        recovered = self.recover_expired(batch_id, current_boot_id=current_boot_id, reconcile=False)
        self.refresh(batch_id)
        return recovered

    def refresh(self, batch_id: str) -> None:
        now = self.clock()
        with self.store.transaction() as connection:
            batch = _require_batch(connection, batch_id)
            self._refresh_ready(connection, batch, now)
            self._update_aggregate_status(connection, batch_id, now)

    def lease_next(
        self,
        batch_id: str,
        worker_id: str,
        capacities: Mapping[str, int],
        *,
        project_id: str | None = None,
        task_id: str | None = None,
        stage_name: str | None = None,
    ) -> WorkItem | None:
        now = self.clock()
        with self.store.transaction() as connection:
            batch = _require_batch(connection, batch_id)
            if batch["status"] != "running":
                return None
            self._refresh_ready(connection, batch, now)
            clauses = ["s.batch_id=?", "s.not_before<=?", "(s.state='ready' OR (s.name='image_edit' AND s.state IN ('leased','running')))"]
            parameters: list[Any] = [batch_id, now]
            for column, value in (("s.project_id", project_id), ("s.task_id", task_id), ("s.name", stage_name)):
                if value is not None:
                    clauses.append(f"{column}=?")
                    parameters.append(value)
            rows = connection.execute(
                f"""SELECT s.*, t.last_dispatched_at, p.sort_index AS project_order, t.sort_index AS task_order
                   FROM stages s
                   JOIN tasks t USING(batch_id, project_id, task_id)
                   JOIN projects p USING(batch_id, project_id)
                   WHERE {' AND '.join(clauses)}
                   ORDER BY COALESCE(t.last_dispatched_at, -1), s.sort_index DESC, p.sort_index, t.sort_index""",
                tuple(parameters),
            ).fetchall()
            manifest = json.loads(batch["manifest_json"])
            for stage in rows:
                allocation = self._allocate_resources(connection, stage, capacities, manifest)
                if allocation is None:
                    continue
                attempt_number = int(
                    connection.execute("SELECT COUNT(*) FROM attempts WHERE stage_id=?", (stage["id"],)).fetchone()[0]
                ) + 1
                generation_index = None
                generation_group_id = None
                if stage["name"] == "image_edit":
                    config = json.loads(stage["effective_config_json"])
                    count = generation_count(config.get("task", config))
                    generation_group_id = str(config.get("editGenerationGroup", f"{stage['task_id']}-group-1"))
                    # Materialize the whole group up front.  This makes pending
                    # candidates visible to aggregate completion even when the
                    # API capacity leases them one at a time.
                    for index in range(1, count + 1):
                        connection.execute(
                            "INSERT OR IGNORE INTO image_edit_generations(stage_id, group_id, generation_index) VALUES(?, ?, ?)",
                            (stage["id"], generation_group_id, index),
                        )
                    existing = connection.execute(
                        "SELECT generation_index, status FROM image_edit_generations WHERE stage_id=? AND group_id=?",
                        (stage["id"], generation_group_id),
                    ).fetchall()
                    active = {int(row["generation_index"]) for row in existing if row["status"] in ("succeeded", "running")}
                    candidates = [int(row["generation_index"]) for row in existing if row["status"] in ("pending", "failed_retryable")]
                    generation_index = next((index for index in sorted(candidates) if index not in active), None)
                    if generation_index is None:
                        if any(row["status"] == "running" for row in existing):
                            continue
                        succeeded = [row for row in existing if row["status"] == "succeeded"]
                        if succeeded:
                            connection.execute(
                                "UPDATE stages SET state='succeeded', aggregate_finished_at=COALESCE(aggregate_started_at, ?), updated_at=? WHERE id=?",
                                (now, now, stage["id"]),
                            )
                            latest = max(
                                int(row["generation_index"])
                                for row in succeeded
                            )
                            latest_attempt = connection.execute(
                                """SELECT MAX(attempt_number) AS attempt FROM image_edit_generations
                                   WHERE stage_id=? AND group_id=? AND generation_index=?""",
                                (stage["id"], generation_group_id, latest),
                            ).fetchone()
                            if latest_attempt and latest_attempt["attempt"] is not None:
                                self._create_edit_review(connection, batch, stage, int(latest_attempt["attempt"]), now)
                        else:
                            connection.execute(
                                "UPDATE stages SET state='failed_terminal', last_error_code='attempts_exhausted', updated_at=? WHERE id=?",
                                (now, stage["id"]),
                            )
                        continue
                    generation_attempts = int(connection.execute(
                        """SELECT COUNT(*) FROM attempts
                           WHERE stage_id=? AND generation_group_id=? AND generation_index=?""",
                        (stage["id"], generation_group_id, generation_index),
                    ).fetchone()[0])
                    if generation_attempts >= int(stage["max_attempts"]):
                        connection.execute(
                            """UPDATE image_edit_generations SET status='failed_terminal',
                               error_code='attempts_exhausted', finished_at=?
                               WHERE stage_id=? AND group_id=? AND generation_index=?
                                 AND status NOT IN ('succeeded','running')""",
                            (now, stage["id"], generation_group_id, generation_index),
                        )
                        continue
                elif attempt_number > int(stage["max_attempts"]):
                    connection.execute(
                        "UPDATE stages SET state='failed_terminal', last_error_code='attempts_exhausted', updated_at=? WHERE id=?",
                        (now, stage["id"]),
                    )
                    continue
                task_root = Path(batch["root_path"]) / stage["project_id"] / stage["task_id"]
                attempt_root = task_root / "stages" / stage["name"] / f"attempt-{attempt_number:04d}"
                staging_dir = attempt_root / "output.staging"
                output_leaf = stage["idempotency_key"]
                if generation_index is not None:
                    output_leaf = f"{output_leaf}/generation-{generation_index:03d}"
                output_dir = task_root / "artifacts" / stage["name"] / output_leaf
                staging_dir.mkdir(parents=True, exist_ok=False)
                token = secrets.token_hex(24)
                cursor = connection.execute(
                    """INSERT INTO attempts(stage_id, attempt_number, status, worker_id, staging_dir, output_dir,
                       generation_group_id, generation_index)
                       VALUES(?, ?, 'leased', ?, ?, ?, ?, ?)""",
                    (stage["id"], attempt_number, worker_id, str(staging_dir), str(output_dir), generation_group_id, generation_index),
                )
                attempt_id = int(cursor.lastrowid)
                if generation_index is not None:
                    connection.execute(
                        "UPDATE image_edit_generations SET attempt_id=?, attempt_number=?, status='running', started_at=? WHERE stage_id=? AND group_id=? AND generation_index=?",
                        (attempt_id, attempt_number, now, stage["id"], generation_group_id, generation_index),
                    )
                expires_at = now + self.lease_seconds
                connection.execute(
                    "INSERT INTO leases VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)",
                    (stage["id"], attempt_id, token, worker_id, _json(allocation), now, now, expires_at),
                )
                if stage["name"] != "image_edit":
                    connection.execute("UPDATE stages SET state='leased', updated_at=? WHERE id=?", (now, stage["id"]))
                else:
                    connection.execute(
                        "UPDATE stages SET state='running', aggregate_started_at=COALESCE(aggregate_started_at, ?), updated_at=? WHERE id=?",
                        (now, now, stage["id"]),
                    )
                connection.execute(
                    "UPDATE tasks SET last_dispatched_at=? WHERE batch_id=? AND project_id=? AND task_id=?",
                    (now, batch_id, stage["project_id"], stage["task_id"]),
                )
                self.store.event(
                    connection,
                    batch_id,
                    "stage_leased",
                    {"attempt": attempt_number, "workerId": worker_id, "resources": allocation},
                    now,
                    stage_id=stage["id"],
                )
                return WorkItem(
                    int(stage["id"]),
                    batch_id,
                    str(stage["project_id"]),
                    str(stage["task_id"]),
                    str(stage["name"]),
                    str(stage["contract_version"]),
                    attempt_number,
                    attempt_id,
                    token,
                    allocation,
                    staging_dir,
                    output_dir,
                    expires_at,
                    generation_group_id,
                    generation_index,
                )
        return None

    def mark_running(
        self,
        item: WorkItem,
        *,
        pid: int,
        pgid: int,
        host_boot_id: str,
        process_start_ticks: int,
    ) -> None:
        now = self.clock()
        with self.store.transaction() as connection:
            self._require_lease(connection, item.stage_id, item.lease_token)
            connection.execute(
                """UPDATE leases SET pid=?, pgid=?, host_boot_id=?, process_start_ticks=?, heartbeat_at=?, expires_at=?
                   WHERE stage_id=? AND token=?""",
                (pid, pgid, host_boot_id, process_start_ticks, now, now + self.lease_seconds, item.stage_id, item.lease_token),
            )
            connection.execute("UPDATE attempts SET status='running', started_at=? WHERE id=?", (now, item.attempt_id))
            connection.execute("UPDATE stages SET state='running', updated_at=? WHERE id=?", (now, item.stage_id))

    def heartbeat(
        self,
        stage_id: int,
        lease_token: str,
        *,
        progress: Mapping[str, Any] | None = None,
    ) -> float:
        now = self.clock()
        with self.store.transaction() as connection:
            lease = self._require_lease(connection, stage_id, lease_token)
            expires_at = now + self.lease_seconds
            connection.execute(
                "UPDATE leases SET heartbeat_at=?, expires_at=? WHERE stage_id=? AND token=?",
                (now, expires_at, stage_id, lease_token),
            )
            stage = connection.execute("SELECT batch_id FROM stages WHERE id=?", (stage_id,)).fetchone()
            self.store.event(
                connection,
                stage["batch_id"],
                "stage_heartbeat",
                {"attemptId": lease["attempt_id"], "progress": dict(progress or {})},
                now,
                stage_id=stage_id,
            )
            return expires_at

    def commit_success(self, item: WorkItem, artifacts: list[Mapping[str, Any]]) -> Path:
        self.prepare_artifact_commit(item, artifacts)
        self.publish_prepared_commit(item.stage_id, item.lease_token)
        return item.output_dir

    def prepare_artifact_commit(self, item: WorkItem, artifacts: list[Mapping[str, Any]]) -> None:
        """Persist a recoverable intent before publishing files."""
        if not artifacts:
            raise ContractError("$.artifacts", "成功 stage 必须至少有一个 artifact")
        checked = self._check_artifacts(item.staging_dir, artifacts)
        normalized = [
            {
                "artifactId": artifact["artifactId"],
                "type": artifact.get("type", "stage_output"),
                "path": relative.as_posix(),
                "sha256": digest,
                "size": size,
            }
            for artifact, relative, size, digest in checked
        ]
        now = self.clock()
        with self.store.transaction() as connection:
            self._require_lease(connection, item.stage_id, item.lease_token)
            existing = connection.execute(
                "SELECT * FROM artifact_commits WHERE attempt_id=?", (item.attempt_id,)
            ).fetchone()
            if existing is not None:
                if existing["attempt_id"] != item.attempt_id or existing["lease_token"] != item.lease_token:
                    raise LeaseFencedError("该 stage 已有其他 attempt 的发布事务")
                if existing["state"] == "committed":
                    return
                raise StoreError("该 attempt 已有未完成发布；请运行 resume/reconcile")
            # A retry may legitimately produce a byte-identical file (for
            # example cameras.json or a stable manifest).  Keep the old
            # attempt's audit row and give this attempt a deterministic ID
            # suffix instead of failing the whole publish on the uniqueness
            # constraint.  Downstream wiring aliases by path, not this ID.
            for entry in normalized:
                original = str(entry["artifactId"])
                candidate = original
                suffix = 2
                while connection.execute(
                    "SELECT 1 FROM artifacts WHERE stage_id=? AND artifact_id=? AND sha256=? LIMIT 1",
                    (item.stage_id, candidate, entry["sha256"]),
                ).fetchone() is not None:
                    candidate = f"{original}_attempt{item.attempt}"
                    if suffix > 2:
                        candidate += f"_{suffix}"
                    suffix += 1
                entry["artifactId"] = candidate
            connection.execute(
                """INSERT INTO artifact_commits(
                       stage_id, attempt_id, lease_token, staging_dir, output_dir,
                       artifacts_json, state, prepared_at
                   ) VALUES(?, ?, ?, ?, ?, ?, 'prepared', ?)""",
                (
                    item.stage_id,
                    item.attempt_id,
                    item.lease_token,
                    str(item.staging_dir),
                    str(item.output_dir),
                    _json(normalized),
                    now,
                ),
            )
            connection.execute("UPDATE attempts SET status='committing' WHERE id=?", (item.attempt_id,))
            connection.execute("UPDATE stages SET state='committing', updated_at=? WHERE id=?", (now, item.stage_id))
            self.store.event(
                connection,
                item.batch_id,
                "artifact_commit_prepared",
                {"attempt": item.attempt, "artifactCount": len(normalized)},
                now,
                stage_id=item.stage_id,
            )

    def publish_prepared_commit(self, stage_id: int, lease_token: str) -> Path:
        row = self.store.row(
            "SELECT * FROM artifact_commits WHERE stage_id=? AND lease_token=? ORDER BY id DESC LIMIT 1",
            (stage_id, lease_token),
        )
        if row is None:
            raise StoreError("找不到已准备的 artifact 发布事务")
        if row["lease_token"] != lease_token:
            raise LeaseFencedError("artifact 发布 token 不匹配")
        self._publish_commit_files(row)
        self._finalize_artifact_commit(stage_id, lease_token, allow_expired=False)
        return Path(row["output_dir"])

    def reconcile_artifact_commits(self, batch_id: str) -> int:
        rows = self.store.rows(
            """SELECT c.* FROM artifact_commits c
               JOIN stages s ON s.id=c.stage_id
               JOIN leases l ON l.stage_id=s.id AND l.token=c.lease_token
               WHERE s.batch_id=? AND c.state IN ('prepared','published')
                 AND l.revoked_at IS NULL
               ORDER BY c.id""",
            (batch_id,),
        )
        reconciled = 0
        for row in rows:
            try:
                self._publish_commit_files(row)
                self._finalize_artifact_commit(row["stage_id"], row["lease_token"], allow_expired=True)
                reconciled += 1
            except (ContractError, OSError, StoreError) as exc:
                self._fail_artifact_commit(row, exc)
        return reconciled

    def fail(
        self,
        item: WorkItem,
        error_code: str,
        message: str,
        *,
        retry_after: float | None = None,
        cleanup_completed: bool = True,
        stage_status: str | None = None,
    ) -> str:
        if stage_status not in {None, "failed_retryable", "failed_terminal", "rejected", "canceled"}:
            raise ValueError(f"不能把 stage 结果状态 {stage_status!r} 作为失败提交")
        now = self.clock()
        with self.store.transaction() as connection:
            self._require_lease(connection, item.stage_id, item.lease_token)
            stage = connection.execute("SELECT * FROM stages WHERE id=?", (item.stage_id,)).fetchone()
            generation_attempt = None
            generation_retry_allowed = item.attempt < int(stage["max_attempts"])
            if item.stage == "image_edit":
                generation_attempt = connection.execute(
                    "SELECT generation_group_id, generation_index FROM attempts WHERE id=?",
                    (item.attempt_id,),
                ).fetchone()
                if generation_attempt and generation_attempt["generation_index"] is not None:
                    attempts_for_generation = connection.execute(
                        """SELECT COUNT(*) AS count FROM attempts
                           WHERE stage_id=? AND generation_group_id=? AND generation_index=?""",
                        (item.stage_id, generation_attempt["generation_group_id"], int(generation_attempt["generation_index"])),
                    ).fetchone()
                    generation_retry_allowed = int(attempts_for_generation["count"]) < int(stage["max_attempts"])
            if not cleanup_completed:
                state = "recovering"
            elif stage_status == "canceled":
                state = "canceled"
            elif stage_status == "rejected":
                state = "rejected"
            elif stage_status == "failed_terminal":
                state = "failed_terminal"
            elif error_code == "delivery_unknown":
                state = "waiting_manual"
            elif stage_status == "failed_retryable" and generation_retry_allowed:
                state = "ready"
            elif error_code == "quality_rejected":
                state = "rejected"
            elif error_code in TERMINAL_ERRORS:
                state = "failed_terminal"
            elif error_code in RETRYABLE_ERRORS and generation_retry_allowed:
                state = "ready"
            else:
                state = "failed_terminal"
            not_before = now + max(0.0, retry_after or 0.0) if state == "ready" else 0.0
            connection.execute(
                """UPDATE attempts SET status=?, finished_at=?, error_code=?, message=?, cleanup_status=? WHERE id=?""",
                (state, now, error_code, message, "clean" if cleanup_completed else "cleanup_failed", item.attempt_id),
            )
            aggregate_complete = False
            if item.stage == "image_edit":
                if generation_attempt and generation_attempt["generation_index"] is not None:
                    generation_state = "failed_retryable" if state == "ready" else "failed_terminal"
                    connection.execute(
                        "UPDATE image_edit_generations SET status=?, error_code=?, finished_at=? WHERE stage_id=? AND group_id=? AND generation_index=?",
                        (generation_state, error_code, now, item.stage_id, generation_attempt["generation_group_id"], int(generation_attempt["generation_index"])),
                    )
                    generation_rows = connection.execute(
                        """SELECT status FROM image_edit_generations
                           WHERE stage_id=? AND group_id=?""",
                        (item.stage_id, generation_attempt["generation_group_id"]),
                    ).fetchall()
                    config = json.loads(stage["effective_config_json"])
                    required = generation_count(config.get("task", config))
                    succeeded_count = sum(row["status"] == "succeeded" for row in generation_rows)
                    unfinished = {"pending", "running", "failed_retryable"}
                    aggregate_complete = succeeded_count >= required or not any(row["status"] in unfinished for row in generation_rows)
            allow_partial_review = False
            # A delivery-unknown result requires operator reconciliation.  It
            # must not be collapsed into aggregate success merely because all
            # generation rows are terminal (which is especially easy to hit
            # for a one-candidate group).  Keep the parent blocked and avoid
            # publishing a review until the delivery is resolved.
            if state == "waiting_manual":
                aggregate_complete = False
                stage_state = state
            else:
                # A terminal failure can enter the partial-candidate review
                # only when at least one candidate succeeded; an all-failed
                # group stays terminally failed.
                allow_partial_review = (
                    item.stage == "image_edit"
                    and aggregate_complete
                    and state == "failed_terminal"
                    and succeeded_count > 0
                ) if item.stage == "image_edit" else False
                stage_state = "succeeded" if allow_partial_review else state
            if item.stage == "image_edit" and not aggregate_complete and stage_state == "failed_terminal":
                # A failed candidate does not block the remaining generation
                # leases; review is allowed once all missing candidates exhaust.
                stage_state = "ready"
                not_before = 0.0
            connection.execute(
                "UPDATE stages SET state=?, not_before=?, last_error_code=?, last_message=?, updated_at=? WHERE id=?",
                (stage_state, not_before, error_code, message, now, item.stage_id),
            )
            if cleanup_completed:
                connection.execute("DELETE FROM leases WHERE stage_id=? AND token=?", (item.stage_id, item.lease_token))
            else:
                connection.execute(
                    "UPDATE leases SET revoked_at=? WHERE stage_id=? AND token=?",
                    (now, item.stage_id, item.lease_token),
                )
            self.store.event(
                connection,
                item.batch_id,
                "stage_failed",
                {"attempt": item.attempt, "errorCode": error_code, "nextState": state},
                now,
                stage_id=item.stage_id,
            )
            if stage_state in {"failed_terminal", "rejected"}:
                self._route_debug_bundle(connection, stage, error_code, now)
            if item.stage == "image_edit" and allow_partial_review and generation_attempt:
                latest = connection.execute(
                    """SELECT MAX(attempt_number) AS attempt FROM image_edit_generations
                       WHERE stage_id=? AND group_id=? AND status='succeeded'""",
                    (item.stage_id, generation_attempt["generation_group_id"]),
                ).fetchone()
                if latest is not None and latest["attempt"] is not None:
                    self._create_edit_review(connection, batch=connection.execute("SELECT * FROM batches WHERE batch_id=?", (item.batch_id,)).fetchone(), stage=stage, attempt=int(latest["attempt"]), now=now)
            self._update_aggregate_status(connection, item.batch_id, now)
            return stage_state

    def review_page(self, batch_id: str, *, page: int = 1, size: int | None = None) -> list[dict[str, Any]]:
        if page <= 0:
            raise ValueError("page 必须从 1 开始")
        batch = self.store.row("SELECT review_batch_size FROM batches WHERE batch_id=?", (batch_id,))
        if batch is None:
            raise StoreError(f"batch 不存在: {batch_id}")
        page_size = int(size or batch["review_batch_size"])
        if page_size <= 0:
            raise ValueError("size 必须大于 0")
        rows = self.store.rows(
            """SELECT batch_id, project_id, task_id, edit_attempt, status, decision, decided_by,
                      decided_at, policy_version, note
               FROM edit_reviews WHERE batch_id=? AND status='pending_review'
               ORDER BY id LIMIT ? OFFSET ?""",
            (batch_id, page_size, (page - 1) * page_size),
        )
        result = []
        batch_root_row = self.store.row("SELECT root_path FROM batches WHERE batch_id=?", (batch_id,))
        batch_root = Path(str(batch_root_row["root_path"])).resolve() if batch_root_row else None
        for row in rows:
            item = dict(row)
            image_stage = self.store.row(
                """SELECT * FROM stages WHERE batch_id=? AND project_id=? AND task_id=?
                   AND name='image_edit'""",
                (batch_id, row["project_id"], row["task_id"]),
            )
            group_id = None
            if image_stage is not None:
                generation_attempt = self.store.row(
                    """SELECT generation_group_id FROM attempts
                       WHERE stage_id=? AND attempt_number=?""",
                    (image_stage["id"], row["edit_attempt"]),
                )
                if generation_attempt is not None:
                    group_id = generation_attempt["generation_group_id"]

            item["editArtifacts"] = []
            candidates: list[dict[str, Any]] = []
            if image_stage is not None and group_id:
                generation_rows = self.store.rows(
                    """SELECT generation_index, attempt_id, attempt_number
                       FROM image_edit_generations
                       WHERE stage_id=? AND group_id=? AND status='succeeded'
                       ORDER BY generation_index""",
                    (image_stage["id"], group_id),
                )
                config = json.loads(image_stage["effective_config_json"])
                item["generationCount"] = generation_count(config.get("task", config))
                item["generationIndices"] = [int(generation["generation_index"]) for generation in generation_rows]
                for generation in generation_rows:
                    artifacts = [
                        dict(artifact)
                        for artifact in self.store.rows(
                            """SELECT a.artifact_id AS artifactId, a.type, a.relative_path AS path,
                                      a.sha256, a.size
                               FROM artifacts a
                               WHERE a.stage_id=? AND a.attempt_id=?
                               ORDER BY a.artifact_id""",
                            (image_stage["id"], generation["attempt_id"]),
                        )
                    ]
                    item["editArtifacts"].extend(artifacts)
                    image = next(
                        (
                            artifact
                            for artifact in artifacts
                            if artifact.get("type") == "edited_image"
                            or (
                                artifact.get("type") == "fake_stage_result"
                                and Path(str(artifact.get("path", ""))).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                            )
                        ),
                        None,
                    )
                    if image is not None and batch_root is not None:
                        full = (batch_root / str(image["path"])).resolve()
                        if full.is_file():
                            candidates.append({
                                "index": int(generation["generation_index"]),
                                "path": str(full),
                                "attempt": int(generation["attempt_number"]),
                            })
                if candidates:
                    item["reviewManifest"] = {"candidates": candidates}
            else:
                # Legacy/synthetic records may predate image_edit_generations.
                item["editArtifacts"] = [
                    dict(artifact)
                    for artifact in self.store.rows(
                        """SELECT a.artifact_id AS artifactId, a.type, a.relative_path AS path,
                                  a.sha256, a.size
                           FROM artifacts a
                           JOIN stages s ON s.id=a.stage_id
                           JOIN attempts attempt ON attempt.id=a.attempt_id
                           WHERE s.batch_id=? AND s.project_id=? AND s.task_id=?
                             AND s.name='image_edit' AND attempt.attempt_number=?
                           ORDER BY a.artifact_id""",
                        (batch_id, row["project_id"], row["task_id"], row["edit_attempt"]),
                    )
                ]
                for artifact in item["editArtifacts"]:
                    if artifact.get("type") != "image_edit_manifest" or batch_root is None:
                        continue
                    manifest_path = batch_root / str(artifact["path"])
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        index = int(manifest.get("generationIndex", 1))
                        output = manifest.get("output") or {}
                        output_path = output.get("fullPath") or output.get("path")
                        if output_path:
                            full = Path(str(output_path))
                            if not full.is_absolute():
                                full = manifest_path.parent / full
                            if not full.is_file() and output.get("path"):
                                name = Path(str(output["path"])).name
                                sibling = next((a.get("path") for a in item["editArtifacts"] if Path(str(a.get("path", ""))).name == name), None)
                                if sibling:
                                    full = batch_root / str(sibling)
                            if full.is_file():
                                candidates.append({"index": index, "path": str(full.resolve())})
                                item["generationIndex"] = index
                                item["generationCount"] = int(manifest.get("generationCount", 1))
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        continue
                if candidates:
                    item["reviewManifest"] = {"candidates": sorted(candidates, key=lambda value: value["index"])}
            result.append(item)
        return result

    @staticmethod
    def _generation_candidate(
        connection: Any,
        batch_root: Path,
        *,
        stage_id: int,
        edit_attempt: int,
        selected_index: int | None,
    ) -> dict[str, Any] | None:
        """Resolve one successful candidate from the review's generation group."""
        attempt = connection.execute(
            """SELECT generation_group_id FROM attempts
               WHERE stage_id=? AND attempt_number=?""",
            (stage_id, edit_attempt),
        ).fetchone()
        if attempt is None or not attempt["generation_group_id"]:
            return None
        group_id = str(attempt["generation_group_id"])
        condition = "AND g.generation_index=?" if selected_index is not None else ""
        parameters: tuple[Any, ...] = (stage_id, group_id, selected_index) if selected_index is not None else (stage_id, group_id)
        row = connection.execute(
            f"""SELECT g.generation_index, g.attempt_number,
                         a.relative_path AS path, a.sha256, a.size
                    FROM image_edit_generations g
                    JOIN artifacts a ON a.attempt_id=g.attempt_id AND a.stage_id=g.stage_id
                   WHERE g.stage_id=? AND g.group_id=? AND g.status='succeeded'
                     {condition} AND (a.type='edited_image' OR (a.type='fake_stage_result' AND (
                         lower(a.relative_path) LIKE '%.png'
                         OR lower(a.relative_path) LIKE '%.jpg'
                         OR lower(a.relative_path) LIKE '%.jpeg'
                         OR lower(a.relative_path) LIKE '%.webp'
                     )))
                   ORDER BY g.generation_index LIMIT 1""",
            parameters,
        ).fetchone()
        if row is None:
            return None
        path = (batch_root / str(row["path"])).resolve()
        if not path.is_file():
            return None
        return {
            "index": int(row["generation_index"]),
            "attempt": int(row["attempt_number"]),
            "path": str(path),
            "sha256": str(row["sha256"]),
            "size": int(row["size"]),
        }

    def decide_edit(
        self,
        batch_id: str,
        project_id: str,
        task_id: str,
        edit_attempt: int,
        decision: str,
        *,
        decided_by: str,
        note: str | None = None,
    ) -> None:
        decision = str(decision).strip()
        # CLI shorthand for the multi-candidate review gate.
        if decision.upper() == "N":
            decision = "canceled"
        elif decision.upper() == "R":
            decision = "regenerate"
        elif decision.isdigit():
            selected_index = int(decision)
            decision = "accepted"
        else:
            selected_index = None
        if decision not in {"accepted", "rejected", "canceled", "regenerate"}:
            raise ValueError("手动决定必须是 accepted、rejected、regenerate、R、N 或候选序号")
        now = self.clock()
        with self.store.transaction() as connection:
            batch = _require_batch(connection, batch_id)
            review = connection.execute(
                """SELECT * FROM edit_reviews WHERE batch_id=? AND project_id=? AND task_id=? AND edit_attempt=?""",
                (batch_id, project_id, task_id, edit_attempt),
            ).fetchone()
            if review is None:
                raise StoreError("找不到对应的图片编辑审核记录")
            if review["status"] != "pending_review":
                raise StoreError("该图片编辑 attempt 已经做过决定")
            connection.execute(
                """UPDATE edit_reviews SET status='decided', decision=?, decided_by=?, decided_at=?, note=? WHERE id=?""",
                (decision, decided_by, now, note, review["id"]),
            )
            gate = connection.execute(
                "SELECT * FROM stages WHERE batch_id=? AND project_id=? AND task_id=? AND name='edit_gate'",
                (batch_id, project_id, task_id),
            ).fetchone()
            if decision == "accepted":
                gate_config = json.loads(gate["effective_config_json"])
                gate_config["approvedEditAttempt"] = edit_attempt
                gate_config["reviewDecision"] = "accepted"
                image_stage_row = connection.execute(
                    "SELECT id FROM stages WHERE batch_id=? AND project_id=? AND task_id=? AND name='image_edit'",
                    (batch_id, project_id, task_id),
                ).fetchone()
                if image_stage_row is not None:
                    candidate = self._generation_candidate(
                        connection,
                        Path(str(batch["root_path"])).resolve(),
                        stage_id=int(image_stage_row["id"]),
                        edit_attempt=edit_attempt,
                        selected_index=selected_index,
                    )
                    if candidate is None:
                        if selected_index is None:
                            raise StoreError("没有可接受的成功图片候选")
                        raise StoreError(f"候选 {selected_index} 不存在或尚未成功")
                    gate_config["acceptedGenerationIndex"] = int(candidate["index"])
                    # Persist the exact selected file so downstream stages do
                    # not need to infer a candidate from artifact ordering.
                    gate_config["acceptedGenerationPath"] = candidate["path"]
                connection.execute(
                    "UPDATE stages SET state='succeeded', effective_config_json=?, updated_at=? WHERE id=?",
                    (_json(gate_config), now, gate["id"]),
                )
                gate_value = dict(gate)
                gate_value["effective_config_json"] = _json(gate_config)
                self._refresh_idempotency_key(connection, gate_value)
            elif decision == "rejected":
                connection.execute(
                    "UPDATE stages SET state='rejected', last_error_code='edit_rejected', updated_at=? WHERE id=?",
                    (now, gate["id"]),
                )
            elif decision == "canceled":
                connection.execute(
                    "UPDATE stages SET state='canceled', last_error_code='edit_canceled', updated_at=? WHERE id=?",
                    (now, gate["id"]),
                )
                connection.execute(
                    """UPDATE stages SET state='canceled', last_error_code='edit_canceled', updated_at=?
                       WHERE batch_id=? AND project_id=? AND task_id=? AND sort_index>?
                         AND state NOT IN ('succeeded','failed_terminal','rejected','canceled')""",
                    (now, batch_id, project_id, task_id, gate["sort_index"]),
                )
            else:
                image_stage = connection.execute(
                    "SELECT * FROM stages WHERE batch_id=? AND project_id=? AND task_id=? AND name='image_edit'",
                    (batch_id, project_id, task_id),
                ).fetchone()
                image_config = json.loads(image_stage["effective_config_json"])
                next_generation = int(image_config.get("editGeneration", 1)) + 1
                image_config["editGeneration"] = next_generation
                image_config["editGenerationGroup"] = f"{task_id}-group-{next_generation}"
                image_config["regeneratedFromAttempt"] = edit_attempt
                attempt_count = int(
                    connection.execute("SELECT COUNT(*) FROM attempts WHERE stage_id=?", (image_stage["id"],)).fetchone()[0]
                )
                connection.execute(
                    """UPDATE stages SET state='ready', not_before=0, effective_config_json=?,
                       max_attempts=MAX(max_attempts, ?), updated_at=? WHERE id=?""",
                    (_json(image_config), attempt_count + 1, now, image_stage["id"]),
                )
                image_value = dict(image_stage)
                image_value["effective_config_json"] = _json(image_config)
                self._refresh_idempotency_key(connection, image_value)
                # Preserve the old group's audit trail while preventing its
                # candidates from being considered by the new review.
                old_group = connection.execute(
                    "SELECT generation_group_id FROM attempts WHERE stage_id=? AND attempt_number=?",
                    (image_stage["id"], edit_attempt),
                ).fetchone()
                if old_group is not None and old_group["generation_group_id"]:
                    connection.execute(
                        "UPDATE image_edit_generations SET status='superseded' WHERE stage_id=? AND group_id=? AND status<>'succeeded'",
                        (image_stage["id"], old_group["generation_group_id"]),
                    )
                gate_config = json.loads(gate["effective_config_json"])
                gate_config.pop("approvedEditAttempt", None)
                gate_config.pop("reviewDecision", None)
                connection.execute(
                    "UPDATE stages SET state='pending', effective_config_json=?, updated_at=? WHERE id=?",
                    (_json(gate_config), now, gate["id"]),
                )
            self.store.event(
                connection,
                batch_id,
                "edit_decided",
                {"projectId": project_id, "taskId": task_id, "editAttempt": edit_attempt, "decision": decision},
                now,
                stage_id=gate["id"],
            )
            self._refresh_ready(connection, batch, now)
            self._update_aggregate_status(connection, batch_id, now)

    def retry(
        self,
        batch_id: str,
        *,
        project_id: str | None = None,
        task_id: str | None = None,
        stage_name: str | None = None,
    ) -> int:
        clauses = ["batch_id=?", "state IN ('failed_terminal','rejected','waiting_manual')"]
        parameters: list[Any] = [batch_id]
        for column, value in (("project_id", project_id), ("task_id", task_id), ("name", stage_name)):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(value)
        now = self.clock()
        with self.store.transaction() as connection:
            _require_batch(connection, batch_id)
            rows = connection.execute(
                f"SELECT id, batch_id, project_id, task_id, sort_index FROM stages WHERE {' AND '.join(clauses)}",
                tuple(parameters),
            ).fetchall()
            for row in rows:
                attempt_count = int(
                    connection.execute("SELECT COUNT(*) FROM attempts WHERE stage_id=?", (row["id"],)).fetchone()[0]
                )
                connection.execute(
                    """UPDATE stages SET state='ready', not_before=0, max_attempts=MAX(max_attempts, ?),
                       last_error_code=NULL, last_message=NULL, updated_at=? WHERE id=?""",
                    (attempt_count + 1, now, row["id"]),
                )
                connection.execute(
                    """UPDATE stages SET state='pending', not_before=0, last_error_code=NULL, last_message=NULL, updated_at=?
                       WHERE batch_id=? AND project_id=? AND task_id=? AND sort_index>?""",
                    (now, row["batch_id"], row["project_id"], row["task_id"], row["sort_index"]),
                )
            if rows:
                connection.execute("UPDATE batches SET status='running', updated_at=? WHERE batch_id=?", (now, batch_id))
            self.store.event(connection, batch_id, "manual_retry", {"count": len(rows)}, now)
            return len(rows)

    def get_recovering_item(
        self,
        batch_id: str,
        project_id: str,
        task_id: str,
        stage_name: str,
        attempt: int,
        lease_token: str,
    ) -> WorkItem:
        """Return one explicitly identified, fenced recovery attempt.

        This is read-only.  Callers must supply every stable identity field so a
        recovery command cannot silently select a newer attempt.
        """

        row = self.store.row(
            """SELECT s.id AS stage_id, s.batch_id, s.project_id, s.task_id,
                      s.name, s.contract_version, s.state, a.id AS attempt_id,
                      a.attempt_number, a.staging_dir, a.output_dir, a.status AS attempt_status,
                      a.generation_group_id, a.generation_index,
                      l.token, l.resources_json, l.expires_at, l.revoked_at
                 FROM stages s
                 JOIN attempts a ON a.stage_id=s.id
                 JOIN leases l ON l.stage_id=s.id AND l.attempt_id=a.id
                WHERE s.batch_id=? AND s.project_id=? AND s.task_id=? AND s.name=?
                  AND a.attempt_number=? AND l.token=?""",
            (batch_id, project_id, task_id, stage_name, attempt, lease_token),
        )
        if row is None:
            raise StoreError("找不到身份完全匹配的远端恢复 attempt")
        if row["state"] != "recovering" or row["revoked_at"] is None:
            raise StoreError("该 attempt 不是等待显式恢复的已撤销租约")
        if row["attempt_status"] != "recovering":
            raise StoreError("stage 与 attempt 的恢复状态不一致")
        return WorkItem(
            stage_id=int(row["stage_id"]),
            batch_id=str(row["batch_id"]),
            project_id=str(row["project_id"]),
            task_id=str(row["task_id"]),
            stage=str(row["name"]),
            contract_version=str(row["contract_version"]),
            attempt=int(row["attempt_number"]),
            attempt_id=int(row["attempt_id"]),
            lease_token=str(row["token"]),
            resources=json.loads(row["resources_json"]),
            staging_dir=Path(row["staging_dir"]),
            output_dir=Path(row["output_dir"]),
            expires_at=float(row["expires_at"]),
            generation_group_id=str(row["generation_group_id"]) if row["generation_group_id"] is not None else None,
            generation_index=int(row["generation_index"]) if row["generation_index"] is not None else None,
        )

    def reactivate_recovering_item(self, item: WorkItem) -> WorkItem:
        """Re-enable the same lease/attempt after a result was verified."""

        now = self.clock()
        expires_at = now + self.lease_seconds
        with self.store.transaction() as connection:
            row = connection.execute(
                """SELECT s.state, a.status AS attempt_status, l.revoked_at
                     FROM stages s
                     JOIN attempts a ON a.id=? AND a.stage_id=s.id
                     JOIN leases l ON l.stage_id=s.id AND l.attempt_id=a.id
                    WHERE s.id=? AND s.batch_id=? AND s.project_id=? AND s.task_id=?
                      AND s.name=? AND a.attempt_number=? AND l.token=?""",
                (
                    item.attempt_id,
                    item.stage_id,
                    item.batch_id,
                    item.project_id,
                    item.task_id,
                    item.stage,
                    item.attempt,
                    item.lease_token,
                ),
            ).fetchone()
            if row is None:
                raise LeaseFencedError("恢复 attempt 的身份或 lease token 已变化")
            if row["state"] != "recovering" or row["attempt_status"] != "recovering" or row["revoked_at"] is None:
                raise LeaseFencedError("恢复 attempt 已被其他操作处理")
            connection.execute(
                """UPDATE leases SET revoked_at=NULL, heartbeat_at=?, expires_at=?
                     WHERE stage_id=? AND attempt_id=? AND token=?""",
                (now, expires_at, item.stage_id, item.attempt_id, item.lease_token),
            )
            connection.execute(
                """UPDATE attempts SET status='leased', finished_at=NULL, error_code=NULL,
                       message=NULL, cleanup_status=NULL WHERE id=?""",
                (item.attempt_id,),
            )
            connection.execute(
                """UPDATE stages SET state='leased', last_error_code=NULL,
                       last_message=NULL, updated_at=? WHERE id=?""",
                (now, item.stage_id),
            )
            self.store.event(
                connection,
                item.batch_id,
                "remote_recovery_reactivated",
                {"attempt": item.attempt, "leaseTokenSha256": hashlib.sha256(item.lease_token.encode()).hexdigest()},
                now,
                stage_id=item.stage_id,
            )
        return WorkItem(
            **{**item.__dict__, "expires_at": expires_at},
        )

    def resolve_recovering_item(
        self,
        item: WorkItem,
        *,
        observed_remote_state: str,
        action: str,
        message: str | None = None,
    ) -> str:
        """Release a fenced remote resource after an explicit stopped-state decision."""

        if observed_remote_state not in {"EXITED", "MISSING"}:
            raise ValueError("只有确认 EXITED 或 MISSING 后才能释放恢复租约")
        if action not in {"retry", "terminal"}:
            raise ValueError("恢复决定必须是 retry 或 terminal")
        now = self.clock()
        with self.store.transaction() as connection:
            row = connection.execute(
                """SELECT s.*, a.status AS attempt_status, l.revoked_at
                     FROM stages s
                     JOIN attempts a ON a.id=? AND a.stage_id=s.id
                     JOIN leases l ON l.stage_id=s.id AND l.attempt_id=a.id
                    WHERE s.id=? AND s.batch_id=? AND s.project_id=? AND s.task_id=?
                      AND s.name=? AND a.attempt_number=? AND l.token=?""",
                (
                    item.attempt_id,
                    item.stage_id,
                    item.batch_id,
                    item.project_id,
                    item.task_id,
                    item.stage,
                    item.attempt,
                    item.lease_token,
                ),
            ).fetchone()
            if row is None:
                raise LeaseFencedError("恢复 attempt 的身份或 lease token 已变化")
            if row["state"] != "recovering" or row["attempt_status"] != "recovering" or row["revoked_at"] is None:
                raise LeaseFencedError("恢复 attempt 已被其他操作处理")
            next_state = "ready" if action == "retry" else "failed_terminal"
            error_code = "remote_stopped_retry" if action == "retry" else "remote_stopped_terminal"
            detail = message or f"显式恢复确认远端状态为 {observed_remote_state}"
            connection.execute(
                """UPDATE attempts SET status=?, finished_at=?, error_code=?, message=?,
                       cleanup_status='remote_confirmed_stopped' WHERE id=?""",
                ("failed_retryable" if action == "retry" else "failed_terminal", now, error_code, detail, item.attempt_id),
            )
            if action == "retry":
                connection.execute(
                    """UPDATE stages SET state='ready', not_before=0,
                           max_attempts=MAX(max_attempts, ?), last_error_code=?, last_message=?, updated_at=?
                         WHERE id=?""",
                    (item.attempt + 1, error_code, detail, now, item.stage_id),
                )
            else:
                connection.execute(
                    """UPDATE stages SET state='failed_terminal', not_before=0,
                           last_error_code=?, last_message=?, updated_at=? WHERE id=?""",
                    (error_code, detail, now, item.stage_id),
                )
                self._route_debug_bundle(connection, row, error_code, now)
            connection.execute(
                "DELETE FROM leases WHERE stage_id=? AND attempt_id=? AND token=?",
                (item.stage_id, item.attempt_id, item.lease_token),
            )
            batch = _require_batch(connection, item.batch_id)
            self.store.event(
                connection,
                item.batch_id,
                "remote_recovery_resolved",
                {"attempt": item.attempt, "remoteState": observed_remote_state, "action": action},
                now,
                stage_id=item.stage_id,
            )
            self._refresh_ready(connection, batch, now)
            self._update_aggregate_status(connection, item.batch_id, now)
        return next_state

    def cancel(self, batch_id: str, *, project_id: str | None = None, task_id: str | None = None) -> int:
        now = self.clock()
        clauses = ["batch_id=?", "state NOT IN ('succeeded','failed_terminal','rejected','canceled')"]
        parameters: list[Any] = [batch_id]
        if project_id:
            clauses.append("project_id=?")
            parameters.append(project_id)
        if task_id:
            clauses.append("task_id=?")
            parameters.append(task_id)
        leases: list[dict[str, Any]] = []
        with self.store.transaction() as connection:
            _require_batch(connection, batch_id)
            rows = connection.execute(
                f"SELECT id, name, state FROM stages WHERE {' AND '.join(clauses)}",
                tuple(parameters),
            ).fetchall()
            active_remote = [
                row
                for row in rows
                if row["name"] in REMOTE_PROCESS_STAGES
                and connection.execute(
                    "SELECT 1 FROM leases WHERE stage_id=?",
                    (row["id"],),
                ).fetchone()
                is not None
            ]
            if active_remote:
                names = ", ".join(
                    sorted({f"{row['name']}({row['state']})" for row in active_remote})
                )
                raise StoreError(
                    "远端步骤仍有占用记录，通用 cancel 不能证明服务器子进程已退出: "
                    f"{names}；请先使用 recover-remote 探测并处理原 attempt"
                )
            for row in rows:
                connection.execute("UPDATE stages SET state='canceled', updated_at=? WHERE id=?", (now, row["id"]))
                lease = connection.execute(
                    """SELECT l.*, a.staging_dir FROM leases l JOIN attempts a ON a.id=l.attempt_id
                       WHERE l.stage_id=?""",
                    (row["id"],),
                ).fetchone()
                if lease is not None:
                    leases.append(dict(lease))
                connection.execute("UPDATE leases SET revoked_at=? WHERE stage_id=?", (now, row["id"]))
            self.store.event(connection, batch_id, "canceled", {"count": len(rows)}, now)
        for lease in leases:
            cleanup = self._cleanup_lease_process(lease, reason="cancel")
            with self.store.transaction() as connection:
                current = connection.execute(
                    "SELECT token FROM leases WHERE stage_id=?", (lease["stage_id"],)
                ).fetchone()
                if current is None or current["token"] != lease["token"]:
                    continue
                connection.execute(
                    "UPDATE attempts SET status='canceled', finished_at=?, cleanup_status=? WHERE id=?",
                    (self.clock(), "clean" if cleanup else "residual_process", lease["attempt_id"]),
                )
                if cleanup:
                    connection.execute("DELETE FROM leases WHERE stage_id=?", (lease["stage_id"],))
                else:
                    connection.execute(
                        """UPDATE stages SET state='failed_terminal', last_error_code='cleanup_failed',
                           last_message='process group still has members', updated_at=? WHERE id=?""",
                        (self.clock(), lease["stage_id"]),
                    )
        with self.store.transaction() as connection:
            self._update_aggregate_status(connection, batch_id, self.clock())
        return len(rows)

    def recover_expired(
        self,
        batch_id: str,
        *,
        current_boot_id: str | None = None,
        reconcile: bool = True,
    ) -> int:
        if reconcile:
            self.reconcile_artifact_commits(batch_id)
        now = self.clock()
        recovered = 0
        effective_boot_id = current_boot_id or host_boot_id()
        rows = self.store.rows(
                """SELECT l.*, s.max_attempts, s.state, s.name AS stage_name, a.staging_dir
                   FROM leases l JOIN stages s ON s.id=l.stage_id JOIN attempts a ON a.id=l.attempt_id
                   WHERE s.batch_id=? AND l.revoked_at IS NULL
                     AND (l.expires_at<=? OR (? IS NOT NULL AND l.host_boot_id IS NOT NULL AND l.host_boot_id<>?))""",
                (batch_id, now, effective_boot_id, effective_boot_id),
            )
        for lease_row in rows:
            lease = dict(lease_row)
            with self.store.transaction() as connection:
                current = connection.execute(
                    "SELECT token, revoked_at FROM leases WHERE stage_id=?", (lease["stage_id"],)
                ).fetchone()
                if current is None or current["token"] != lease["token"] or current["revoked_at"] is not None:
                    continue
                connection.execute("UPDATE leases SET revoked_at=? WHERE stage_id=?", (now, lease["stage_id"]))
            cleanup = True
            if lease.get("host_boot_id") == effective_boot_id:
                cleanup = self._cleanup_lease_process(lease, reason="lease-expired")
            with self.store.transaction() as connection:
                current = connection.execute(
                    "SELECT token FROM leases WHERE stage_id=?", (lease["stage_id"],)
                ).fetchone()
                if current is None or current["token"] != lease["token"]:
                    continue
                attempt = connection.execute("SELECT attempt_number FROM attempts WHERE id=?", (lease["attempt_id"],)).fetchone()
                if lease["stage_name"] in REMOTE_PROCESS_STAGES:
                    connection.execute(
                        """UPDATE attempts SET status='recovering', finished_at=?, error_code='delivery_unknown',
                           message='local remote-wrapper lease expired; remote process status is unknown',
                           cleanup_status=? WHERE id=?""",
                        (
                            now,
                            "local_wrapper_clean_remote_unknown" if cleanup else "local_wrapper_residual_remote_unknown",
                            lease["attempt_id"],
                        ),
                    )
                    connection.execute(
                        """UPDATE stages SET state='recovering', last_error_code='delivery_unknown',
                           last_message='远端 wrapper 租约过期；必须探测原 attempt，禁止自动重跑', updated_at=?
                           WHERE id=?""",
                        (now, lease["stage_id"]),
                    )
                    self.store.event(
                        connection,
                        batch_id,
                        "remote_lease_fenced",
                        {"attemptId": lease["attempt_id"], "nextState": "recovering"},
                        now,
                        stage_id=lease["stage_id"],
                    )
                    recovered += 1
                    continue
                next_state = (
                    "ready"
                    if cleanup and int(attempt["attempt_number"]) < int(lease["max_attempts"])
                    else "failed_terminal"
                )
                connection.execute(
                    """UPDATE attempts SET status='lease_expired', finished_at=?, error_code='worker_lost',
                       cleanup_status=? WHERE id=?""",
                    (now, "clean" if cleanup else "residual_process", lease["attempt_id"]),
                )
                connection.execute(
                    "UPDATE stages SET state=?, last_error_code='worker_lost', updated_at=? WHERE id=?",
                    (next_state, now, lease["stage_id"]),
                )
                if cleanup:
                    connection.execute("DELETE FROM leases WHERE stage_id=?", (lease["stage_id"],))
                self.store.event(
                    connection,
                    batch_id,
                    "lease_recovered",
                    {"attemptId": lease["attempt_id"], "nextState": next_state},
                    now,
                    stage_id=lease["stage_id"],
                )
                recovered += 1
        with self.store.transaction() as connection:
            self._update_aggregate_status(connection, batch_id, now)
        return recovered

    def status(self, batch_id: str) -> dict[str, Any]:
        batch = self.store.row("SELECT * FROM batches WHERE batch_id=?", (batch_id,))
        if batch is None:
            raise StoreError(f"batch 不存在: {batch_id}")
        counts = {
            row["state"]: int(row["count"])
            for row in self.store.rows("SELECT state, COUNT(*) AS count FROM stages WHERE batch_id=? GROUP BY state", (batch_id,))
        }
        now = self.clock()
        # Keep the status view read-only: ready ordering mirrors lease_next's
        # deterministic ordering, while dependency refresh remains owned by
        # the worker loop.
        ready_rows = self.store.rows(
            """SELECT s.id, s.project_id, s.task_id, s.name, s.sort_index,
                      t.last_dispatched_at, p.sort_index AS project_order,
                      t.sort_index AS task_order
               FROM stages s JOIN tasks t USING(batch_id, project_id, task_id)
               JOIN projects p USING(batch_id, project_id)
               WHERE s.batch_id=? AND s.state='ready' AND s.not_before<=?
               ORDER BY COALESCE(t.last_dispatched_at, -1), s.sort_index DESC,
                        p.sort_index, t.sort_index""",
            (batch_id, now),
        )
        ready_positions = {int(row["id"]): index for index, row in enumerate(ready_rows, 1)}
        resource_ready_positions: dict[str, dict[int, int]] = {}
        resource_counts: dict[str, int] = {}
        for row in ready_rows:
            resource = status_resource_for_stage(str(row["name"]))
            if resource is None:
                continue
            resource_counts[resource] = resource_counts.get(resource, 0) + 1
            resource_ready_positions.setdefault(resource, {})[int(row["id"])] = resource_counts[resource]
        tasks = []
        for row in self.store.rows(
            """SELECT t.project_id, t.task_id, t.status, t.sort_index AS task_order,
                      (SELECT MIN(a2.started_at) FROM attempts a2
                       JOIN stages s2 ON s2.id=a2.stage_id
                       WHERE s2.batch_id=t.batch_id AND s2.project_id=t.project_id
                         AND s2.task_id=t.task_id) AS task_started_at,
                      (SELECT MAX(a2.finished_at) FROM attempts a2
                       JOIN stages s2 ON s2.id=a2.stage_id
                       WHERE s2.batch_id=t.batch_id AND s2.project_id=t.project_id
                         AND s2.task_id=t.task_id) AS task_finished_at,
                      s.id AS stage_id, s.name AS current_stage, s.state AS stage_state,
                      s.last_error_code, s.last_message,
                      s.updated_at AS stage_updated_at,
                      a.started_at AS stage_started_at,
                      a.finished_at AS stage_finished_at,
                      s.not_before AS stage_not_before,
                      l.heartbeat_at, l.resources_json, l.acquired_at
               FROM tasks t
               LEFT JOIN stages s ON s.batch_id=t.batch_id AND s.project_id=t.project_id
                 AND s.task_id=t.task_id AND s.state<>'succeeded'
                 AND s.sort_index=(SELECT MIN(s2.sort_index) FROM stages s2
                   WHERE s2.batch_id=t.batch_id AND s2.project_id=t.project_id
                     AND s2.task_id=t.task_id AND s2.state<>'succeeded')
               LEFT JOIN attempts a ON a.id=(SELECT a2.id FROM attempts a2
                 WHERE a2.stage_id=s.id ORDER BY a2.attempt_number DESC LIMIT 1)
               LEFT JOIN leases l ON l.stage_id=s.id AND l.revoked_at IS NULL
               WHERE t.batch_id=?
               GROUP BY t.project_id, t.task_id
               ORDER BY t.project_id, t.sort_index""",
            (batch_id,),
        ):
            item = dict(row)
            task_started = item.pop("task_started_at", None)
            task_finished = item.pop("task_finished_at", None)
            stage_started = item.pop("stage_started_at", None)
            stage_finished = item.pop("stage_finished_at", None)
            stage_not_before = item.pop("stage_not_before", None)
            item["taskElapsedSeconds"] = (
                max(0.0, (task_finished or now) - task_started) if task_started is not None else None
            )
            stage_end = (
                now
                if item.get("stage_state") in ACTIVE_STATES
                else (stage_finished or item.get("stage_updated_at"))
            )
            item["stageElapsedSeconds"] = (
                max(0.0, stage_end - stage_started) if stage_started is not None else None
            )
            item["readyQueuePosition"] = ready_positions.get(item.get("stage_id"))
            queue_resource = status_resource_for_stage(item.get("current_stage"))
            item["queueResource"] = queue_resource
            item["resourceQueuePosition"] = (
                resource_ready_positions.get(queue_resource, {}).get(item.get("stage_id"))
                if queue_resource
                else None
            )
            item["notBefore"] = stage_not_before
            resources = item.pop("resources_json", None)
            item["resources"] = json.loads(resources) if resources else {}
            item["heartbeatAt"] = item.pop("heartbeat_at", None)
            item["stageStartedAt"] = stage_started
            item["taskStartedAt"] = task_started
            item["taskFinishedAt"] = task_finished
            if item.get("stage_id") is not None and item.get("stage_state") == "pending":
                blocker = self._status_blocker(
                    int(item["stage_id"]),
                    ready_positions=ready_positions,
                    resource_ready_positions=resource_ready_positions,
                    now=now,
                )
                if blocker is not None:
                    item["blockedBy"] = blocker
            tasks.append(item)
        api_buckets = []
        for row in self.store.rows("SELECT * FROM api_buckets ORDER BY bucket_key"):
            bucket = dict(row)
            bucket["in_flight"] = sum(
                1 for lease in self.store.rows(
                    "SELECT resources_json FROM leases l JOIN stages s ON s.id=l.stage_id "
                    "WHERE s.batch_id=? AND l.revoked_at IS NULL AND s.name='image_edit'", (batch_id,)
                )
                if "image_api" in json.loads(lease["resources_json"])
            )
            api_buckets.append(bucket)
        return {
            "schemaVersion": 1,
            "batchId": batch_id,
            "status": batch["status"],
            "stageCounts": counts,
            "observedAt": now,
            "activeLeases": sum(1 for row in self.store.rows(
                "SELECT 1 FROM leases l JOIN stages s ON s.id=l.stage_id "
                "WHERE s.batch_id=? AND l.revoked_at IS NULL", (batch_id,)
            )),
            "apiBuckets": api_buckets,
            "tasks": tasks,
        }

    def _status_blocker(
        self,
        stage_id: int,
        *,
        ready_positions: Mapping[int, int],
        resource_ready_positions: Mapping[str, Mapping[int, int]],
        now: float,
    ) -> dict[str, Any] | None:
        """Resolve a pending stage to the first non-successful predecessor.

        The worker refreshes dependencies asynchronously, so a status poll can
        observe several downstream stages as ``pending``.  Walking the DAG
        here gives operators the actual stage/resource that is holding the
        selected stage, while retaining the raw stage state in the snapshot.
        """

        current_id = int(stage_id)
        visited: set[int] = set()
        while current_id not in visited:
            visited.add(current_id)
            current = self.store.row("SELECT * FROM stages WHERE id=?", (current_id,))
            if current is None:
                return None
            dependencies = self.store.rows(
                """SELECT p.* FROM stage_dependencies d
                   JOIN stages p ON p.id=d.depends_on_stage_id
                   WHERE d.stage_id=? AND p.state<>'succeeded'
                   ORDER BY p.sort_index, p.id""",
                (current_id,),
            )
            if current["state"] != "pending" or not dependencies:
                break
            current_id = int(dependencies[0]["id"])

        if current is None or current["state"] == "succeeded":
            return None
        blocker = {
            "stageId": int(current["id"]),
            "stage": str(current["name"]),
            "stageState": str(current["state"]),
            "errorCode": current["last_error_code"],
            "message": current["last_message"],
            "notBefore": current["not_before"],
            "queueResource": status_resource_for_stage(str(current["name"])),
            "readyQueuePosition": ready_positions.get(int(current["id"])),
        }
        resource = blocker["queueResource"]
        blocker["resourceQueuePosition"] = (
            resource_ready_positions.get(resource, {}).get(int(current["id"]))
            if resource
            else None
        )
        lease = self.store.row(
            "SELECT resources_json, heartbeat_at, acquired_at FROM leases WHERE stage_id=? AND revoked_at IS NULL",
            (int(current["id"]),),
        )
        blocker["resources"] = json.loads(lease["resources_json"]) if lease else {}
        blocker["heartbeatAt"] = lease["heartbeat_at"] if lease else None
        blocker["acquiredAt"] = lease["acquired_at"] if lease else None
        # Keep this explicit for callers that need to distinguish a delayed
        # retry from a normal ready queue entry without comparing timestamps.
        blocker["retryWaiting"] = (
            str(current["state"]) == "ready"
            and float(current["not_before"] or 0.0) > now
        )
        return blocker

    def build_stage_request(self, item: WorkItem) -> dict[str, Any]:
        """Join this lease with committed inputs into a worker request."""
        from .stage_wiring import build_stage_request

        return build_stage_request(self.store, item)

    def write_stage_request(self, item: WorkItem, path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
        """Persist a request under the batch root using an atomic replace."""
        from .stage_wiring import write_stage_request

        return write_stage_request(self.store, item, path)

    def build_stage_command(
        self,
        item: WorkItem,
        request_path: str | Path,
        *,
        unity_executable: str | None = None,
        adapter_path: str | Path | None = None,
        python_executable: str | None = None,
    ) -> list[str]:
        """Return the concrete Unity or remote adapter command for a lease."""
        from .stage_wiring import build_stage_command

        return build_stage_command(
            self.store,
            item,
            request_path,
            unity_executable=unity_executable,
            adapter_path=adapter_path,
            python_executable=python_executable,
        )

    def doctor(self, batch_id: str) -> dict[str, Any]:
        batch = self.store.row("SELECT * FROM batches WHERE batch_id=?", (batch_id,))
        if batch is None:
            raise StoreError(f"batch 不存在: {batch_id}")
        root = Path(batch["root_path"])
        disk = shutil.disk_usage(root)
        expired = int(
            self.store.row(
                "SELECT COUNT(*) AS count FROM leases l JOIN stages s ON s.id=l.stage_id WHERE s.batch_id=? AND l.expires_at<=?",
                (batch_id, self.clock()),
            )["count"]
        )
        foreign_key_errors = [tuple(row) for row in self.store.rows("PRAGMA foreign_key_check")]
        api_owners = [
            {
                "bucketKey": row["bucket_key"],
                "pid": row["pid"],
                "hostBootId": row["host_boot_id"],
                "processStartTicks": row["process_start_ticks"],
            }
            for row in self.store.rows("SELECT * FROM api_bucket_owners ORDER BY bucket_key")
        ]
        residual_leases = int(self.store.row("SELECT COUNT(*) AS count FROM leases WHERE revoked_at IS NOT NULL")["count"])
        return {
            "schemaVersion": 1,
            "batchId": batch_id,
            "database": "ok" if not foreign_key_errors else "invalid",
            "foreignKeyErrors": foreign_key_errors,
            "expiredLeases": expired,
            "residualProcessLeases": residual_leases,
            "apiQueue": {"singleControllerEnforced": True, "owners": api_owners},
            "disk": {"total": disk.total, "used": disk.used, "free": disk.free, "usedRatio": disk.used / disk.total},
        }

    def gc(self, batch_id: str, *, owner_token: str, dry_run: bool = True, older_than_seconds: float = 86400) -> list[str]:
        if not owner_token:
            raise ValueError("owner_token 不能为空")
        batch = self.store.row("SELECT root_path FROM batches WHERE batch_id=?", (batch_id,))
        if batch is None:
            raise StoreError(f"batch 不存在: {batch_id}")
        root = Path(batch["root_path"]).resolve()
        now = self.clock()
        active_attempts = {
            Path(row["staging_dir"]).resolve()
            for row in self.store.rows(
                "SELECT a.staging_dir FROM attempts a JOIN stages s ON s.id=a.stage_id JOIN leases l ON l.attempt_id=a.id WHERE s.batch_id=?",
                (batch_id,),
            )
        }
        candidates: list[Path] = []
        for row in self.store.rows(
            """SELECT a.staging_dir FROM attempts a JOIN stages s ON s.id=a.stage_id
               WHERE s.batch_id=? AND a.status<>'succeeded' AND COALESCE(a.finished_at, 0)<=?""",
            (batch_id, now - older_than_seconds),
        ):
            path = Path(row["staging_dir"]).resolve()
            if path in active_attempts or not _within(path, root) or path == root or not path.exists():
                continue
            candidates.append(path)
        with self.store.transaction() as connection:
            for path in candidates:
                cursor = connection.execute(
                    "INSERT INTO gc_journal(batch_id, owner_token, target_path, reason, dry_run, status, created_at) VALUES(?, ?, ?, 'stale_failed_staging', ?, 'planned', ?)",
                    (batch_id, owner_token, str(path), int(dry_run), now),
                )
                if not dry_run:
                    shutil.rmtree(path)
                    connection.execute("UPDATE gc_journal SET status='deleted', finished_at=? WHERE id=?", (self.clock(), cursor.lastrowid))
        return [str(path) for path in candidates]

    def _cleanup_lease_process(self, lease: Mapping[str, Any], *, reason: str) -> bool:
        required = ("pid", "pgid", "host_boot_id", "process_start_ticks")
        if any(lease.get(key) is None for key in required):
            return True
        identity = ProcessIdentity(
            int(lease["pid"]),
            int(lease["pgid"]),
            str(lease["host_boot_id"]),
            int(lease["process_start_ticks"]),
        )
        staging_dir = lease.get("staging_dir")
        diagnostics = Path(staging_dir).parent / "diagnostics" / reason if staging_dir else None
        result = self.process_supervisor.terminate(identity, diagnostics_dir=diagnostics)
        return result.completed

    def _refresh_ready(self, connection: Any, batch: Any, now: float) -> None:
        while True:
            rows = connection.execute(
                """SELECT s.* FROM stages s
                   WHERE s.batch_id=? AND s.state='pending'
                     AND NOT EXISTS (
                       SELECT 1 FROM stage_dependencies d JOIN stages p ON p.id=d.depends_on_stage_id
                       WHERE d.stage_id=s.id AND p.state<>'succeeded'
                     ) ORDER BY s.sort_index""",
                (batch["batch_id"],),
            ).fetchall()
            if not rows:
                break
            changed = False
            for stage in rows:
                if stage["name"] == "edit_gate":
                    review = connection.execute(
                        """SELECT * FROM edit_reviews WHERE batch_id=? AND project_id=? AND task_id=?
                           ORDER BY edit_attempt DESC LIMIT 1""",
                        (stage["batch_id"], stage["project_id"], stage["task_id"]),
                    ).fetchone()
                    if review is None:
                        continue
                    if review["decision"] in {"accepted", "accepted_by_policy"}:
                        new_state = "succeeded"
                    elif review["decision"] == "rejected":
                        new_state = "rejected"
                    else:
                        new_state = "waiting_review"
                else:
                    new_state = "ready"
                    self._refresh_idempotency_key(connection, stage)
                if stage["name"] == "edit_gate" and new_state == "succeeded":
                    gate_config = json.loads(stage["effective_config_json"])
                    gate_config["approvedEditAttempt"] = int(review["edit_attempt"])
                    gate_config["reviewDecision"] = review["decision"]
                    if review["decision"] == "accepted_by_policy":
                        image_stage = connection.execute(
                            """SELECT id FROM stages WHERE batch_id=? AND project_id=? AND task_id=?
                               AND name='image_edit'""",
                            (stage["batch_id"], stage["project_id"], stage["task_id"]),
                        ).fetchone()
                        if image_stage is not None:
                            candidate = self._generation_candidate(
                                connection,
                                Path(str(batch["root_path"])).resolve(),
                                stage_id=int(image_stage["id"]),
                                edit_attempt=int(review["edit_attempt"]),
                                selected_index=None,
                            )
                            if candidate is not None:
                                gate_config["acceptedGenerationIndex"] = int(candidate["index"])
                                gate_config["acceptedGenerationPath"] = candidate["path"]
                    connection.execute(
                        "UPDATE stages SET state=?, effective_config_json=?, updated_at=? WHERE id=?",
                        (new_state, _json(gate_config), now, stage["id"]),
                    )
                    stage_value = dict(stage)
                    stage_value["effective_config_json"] = _json(gate_config)
                    self._refresh_idempotency_key(connection, stage_value)
                else:
                    connection.execute("UPDATE stages SET state=?, updated_at=? WHERE id=?", (new_state, now, stage["id"]))
                changed = True
            if not changed:
                break

    def _create_edit_review(self, connection: Any, batch: Any, stage: Any, attempt: int, now: float) -> None:
        mode = batch["edit_mode"]
        decision = "accepted_by_policy" if mode == "automatic" else None
        status = "decided" if decision else "pending_review"
        manifest = json.loads(batch["manifest_json"])
        policy_version = manifest["editPolicy"].get("policyVersion") if decision else None
        connection.execute(
            """INSERT INTO edit_reviews(batch_id, project_id, task_id, edit_attempt, status, decision,
                                        decided_by, decided_at, policy_version, note, created_at)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
            (
                stage["batch_id"],
                stage["project_id"],
                stage["task_id"],
                attempt,
                status,
                decision,
                "automatic-policy" if decision else None,
                now if decision else None,
                policy_version,
                now,
            ),
        )

    def _refresh_idempotency_key(self, connection: Any, stage: Any) -> None:
        config = json.loads(stage["effective_config_json"])
        accepted_path = config.get("acceptedGenerationPath")
        if not accepted_path and int(stage["sort_index"]) > STAGE_INDEX["edit_gate"]:
            gate = connection.execute(
                """SELECT effective_config_json FROM stages
                   WHERE batch_id=? AND project_id=? AND task_id=? AND name='edit_gate'""",
                (stage["batch_id"], stage["project_id"], stage["task_id"]),
            ).fetchone()
            if gate is not None:
                accepted_path = json.loads(gate["effective_config_json"]).get("acceptedGenerationPath")
        root_row = connection.execute("SELECT root_path FROM batches WHERE batch_id=?", (stage["batch_id"],)).fetchone()
        root = Path(str(root_row["root_path"])).resolve() if root_row else None
        artifact_rows = connection.execute(
            """SELECT ('artifact:' || a.sha256) AS identity,
                      upstream.name AS source_stage, a.relative_path AS relative_path
                   FROM artifacts a
                   JOIN stages upstream ON upstream.id=a.stage_id
                   JOIN attempts attempt ON attempt.id=a.attempt_id
                   WHERE upstream.batch_id=? AND upstream.project_id=? AND upstream.task_id=?
                     AND upstream.sort_index<? AND attempt.status='succeeded'
                     AND (upstream.name='image_edit' OR attempt.attempt_number=(
                       SELECT MAX(latest.attempt_number) FROM attempts latest
                       WHERE latest.stage_id=upstream.id AND latest.status='succeeded'
                     ))
                   UNION ALL
                   SELECT ('stage:' || upstream.idempotency_key) AS identity,
                          NULL AS source_stage, NULL AS relative_path
                   FROM stages upstream
                   WHERE upstream.batch_id=? AND upstream.project_id=? AND upstream.task_id=?
                     AND upstream.sort_index<? AND upstream.state='succeeded'
                   ORDER BY identity""",
                (
                    stage["batch_id"],
                    stage["project_id"],
                    stage["task_id"],
                    stage["sort_index"],
                    stage["batch_id"],
                    stage["project_id"],
                    stage["task_id"],
                    stage["sort_index"],
                ),
            ).fetchall()
        digests: list[str] = []
        selected = Path(str(accepted_path)).resolve() if accepted_path else None
        for row in artifact_rows:
            if row["source_stage"] == "image_edit" and selected is not None:
                if root is None or (root / str(row["relative_path"])).resolve() != selected:
                    continue
            digests.append(str(row["identity"]))
        if accepted_path:
            # Include the selected identity as well as its bytes.  Two
            # candidates may be byte-identical in a fake run but are still
            # distinct user choices and must not reuse downstream outputs.
            digests.append("accepted:" + str(Path(str(accepted_path)).resolve()))
        idem = _idempotency_key(
            stage["batch_id"],
            stage["project_id"],
            stage["task_id"],
            stage["name"],
            stage["contract_version"],
            digests,
            config,
        )
        connection.execute("UPDATE stages SET idempotency_key=? WHERE id=?", (idem, stage["id"]))

    def _allocate_resources(
        self,
        connection: Any,
        stage: Any,
        capacities: Mapping[str, int],
        manifest: Mapping[str, Any],
    ) -> dict[str, str] | None:
        spec = STAGE_BY_NAME[stage["name"]]
        active = [
            json.loads(row["resources_json"])
            for row in connection.execute("SELECT resources_json FROM leases")
        ]
        allocation: dict[str, str] = {}
        for resource in spec.resources:
            if resource == "project_lock":
                key = f"project:{stage['project_id']}"
                if any(item.get("project_lock") == key for item in active):
                    return None
                allocation[resource] = key
                continue
            capacity = int(capacities.get(resource, 0))
            if capacity <= 0:
                return None
            if resource == "remote_gpu":
                pool = [str(item) for item in manifest["resources"]["remoteGpuPool"]]
                pool = pool[:capacity]
                used = {item.get(resource) for item in active}
                available = next((f"gpu:{gpu}" for gpu in pool if f"gpu:{gpu}" not in used), None)
                if available is None:
                    return None
                allocation[resource] = available
            else:
                used_count = sum(resource in item for item in active)
                if used_count >= capacity:
                    return None
                allocation[resource] = f"slot:{used_count}"
        return allocation

    def _require_lease(self, connection: Any, stage_id: int, lease_token: str) -> Any:
        lease = connection.execute(
            "SELECT * FROM leases WHERE stage_id=? AND token=? AND revoked_at IS NULL AND expires_at>?",
            (stage_id, lease_token, self.clock()),
        ).fetchone()
        if lease is None:
            raise LeaseFencedError("lease 已过期、失效或 token 不匹配，拒绝旧 worker 写入")
        return lease

    def _check_artifacts(
        self,
        staging_dir: Path,
        artifacts: list[Mapping[str, Any]],
    ) -> list[tuple[Mapping[str, Any], Path, int, str]]:
        result = []
        root = staging_dir.resolve()
        for index, artifact in enumerate(artifacts):
            artifact_id = artifact.get("artifactId")
            relative = Path(str(artifact.get("path", "")))
            if not artifact_id or relative.is_absolute() or ".." in relative.parts:
                raise ContractError(f"$.artifacts[{index}]", "artifactId 和安全相对 path 必填")
            path = (root / relative).resolve()
            if not _within(path, root) or not path.is_file():
                raise ContractError(f"$.artifacts[{index}].path", "文件不存在或越出 staging 目录")
            size = path.stat().st_size
            if size <= 0:
                raise ContractError(f"$.artifacts[{index}].path", "artifact 不能为空")
            digest = _sha256_file(path)
            expected = artifact.get("sha256")
            if expected is not None and expected != digest:
                raise ContractError(f"$.artifacts[{index}].sha256", "与实际文件不一致")
            expected_size = artifact.get("size")
            if expected_size is not None and expected_size != size:
                raise ContractError(f"$.artifacts[{index}].size", "与实际文件大小不一致")
            result.append((artifact, relative, size, digest))
        return result

    def _publish_commit_files(self, commit: Any) -> None:
        staging_dir = Path(commit["staging_dir"])
        output_dir = Path(commit["output_dir"])
        artifacts = json.loads(commit["artifacts_json"])
        staging_exists = staging_dir.is_dir()
        output_exists = output_dir.is_dir()
        if staging_exists and output_exists:
            raise StoreError("staging 与正式输出同时存在，拒绝猜测哪份是权威")
        if not staging_exists and not output_exists:
            raise StoreError("staging 与正式输出都不存在，无法恢复发布")
        if staging_exists:
            self._check_artifacts(staging_dir, artifacts)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_dir, output_dir)
        self._check_artifacts(output_dir, artifacts)
        with self.store.transaction() as connection:
            current = connection.execute(
                "SELECT state, lease_token FROM artifact_commits WHERE id=?", (commit["id"],)
            ).fetchone()
            if current is None or current["lease_token"] != commit["lease_token"]:
                raise LeaseFencedError("artifact 发布日志已被替换")
            if current["state"] != "committed":
                connection.execute(
                    "UPDATE artifact_commits SET state='published' WHERE id=?",
                    (commit["id"],),
                )

    def _finalize_artifact_commit(self, stage_id: int, lease_token: str, *, allow_expired: bool) -> None:
        now = self.clock()
        with self.store.transaction() as connection:
            commit = connection.execute(
                "SELECT * FROM artifact_commits WHERE stage_id=? AND lease_token=? ORDER BY id DESC LIMIT 1",
                (stage_id, lease_token),
            ).fetchone()
            if commit is None or commit["lease_token"] != lease_token:
                raise LeaseFencedError("artifact 发布日志或 token 不匹配")
            if commit["state"] == "committed":
                return
            lease = connection.execute(
                "SELECT * FROM leases WHERE stage_id=? AND token=? AND revoked_at IS NULL",
                (stage_id, lease_token),
            ).fetchone()
            if lease is None or (not allow_expired and float(lease["expires_at"]) <= now):
                raise LeaseFencedError("artifact 发布时 lease 已失效")
            stage = connection.execute("SELECT * FROM stages WHERE id=?", (stage_id,)).fetchone()
            if stage is None or stage["state"] != "committing" or lease["attempt_id"] != commit["attempt_id"]:
                raise LeaseFencedError("stage/attempt 不再属于该发布事务")
            batch = _require_batch(connection, stage["batch_id"])
            output_dir = Path(commit["output_dir"]).resolve()
            root = Path(batch["root_path"]).resolve()
            if not _within(output_dir, root):
                raise StoreError("artifact 输出越出 batch 根目录")
            output_relative = output_dir.relative_to(root)
            artifacts = json.loads(commit["artifacts_json"])
            for artifact in artifacts:
                connection.execute(
                    """INSERT INTO artifacts(
                           artifact_id, stage_id, attempt_id, type, relative_path, sha256, size, created_at
                       ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        artifact["artifactId"],
                        stage_id,
                        commit["attempt_id"],
                        artifact["type"],
                        str(output_relative / artifact["path"]),
                        artifact["sha256"],
                        artifact["size"],
                        now,
                    ),
                )
            connection.execute(
                "UPDATE attempts SET status='succeeded', finished_at=?, cleanup_status='clean' WHERE id=?",
                (now, commit["attempt_id"]),
            )
            generation_complete = True
            generation_attempt = connection.execute(
                "SELECT generation_group_id, generation_index FROM attempts WHERE id=?",
                (commit["attempt_id"],),
            ).fetchone()
            if stage["name"] == "image_edit" and generation_attempt and generation_attempt["generation_index"] is not None:
                generation_group_id = str(generation_attempt["generation_group_id"])
                generation_index = int(generation_attempt["generation_index"])
                attempt_number = int(connection.execute("SELECT attempt_number FROM attempts WHERE id=?", (commit["attempt_id"],)).fetchone()[0])
                output_payload = {"artifacts": artifacts, "attempt": attempt_number}
                connection.execute(
                    "UPDATE image_edit_generations SET status='succeeded', output_json=?, finished_at=? WHERE stage_id=? AND group_id=? AND generation_index=?",
                    (_json(output_payload), now, stage_id, generation_group_id, generation_index),
                )
                config = json.loads(stage["effective_config_json"])
                required = generation_count(config.get("task", config))
                generations = connection.execute(
                    "SELECT status FROM image_edit_generations WHERE stage_id=? AND group_id=?",
                    (stage_id, generation_group_id),
                ).fetchall()
                completed = sum(row["status"] == "succeeded" for row in generations)
                unfinished = {"pending", "running", "failed_retryable"}
                generation_complete = completed >= required or not any(row["status"] in unfinished for row in generations)
            if stage["name"] == "image_edit" and not generation_complete:
                connection.execute(
                    "UPDATE stages SET state='ready', last_error_code=NULL, last_message=NULL, updated_at=? WHERE id=?",
                    (now, stage_id),
                )
            else:
                connection.execute(
                    "UPDATE stages SET state='succeeded', aggregate_finished_at=COALESCE(aggregate_started_at, ?), last_error_code=NULL, last_message=NULL, updated_at=? WHERE id=?",
                    (now, now, stage_id),
                )
            connection.execute("UPDATE artifact_commits SET state='committed', finished_at=? WHERE id=?", (now, commit["id"]))
            connection.execute("DELETE FROM leases WHERE stage_id=? AND token=?", (stage_id, lease_token))
            attempt = connection.execute("SELECT attempt_number FROM attempts WHERE id=?", (commit["attempt_id"],)).fetchone()
            self.store.event(
                connection,
                stage["batch_id"],
                "stage_succeeded",
                {"attempt": attempt["attempt_number"], "artifactCount": len(artifacts), "reconciled": allow_expired},
                now,
                stage_id=stage_id,
            )
            if stage["name"] == "image_edit" and generation_complete:
                self._create_edit_review(connection, batch, stage, int(attempt["attempt_number"]), now)
            self._refresh_ready(connection, batch, now)
            self._update_aggregate_status(connection, stage["batch_id"], now)

    def _fail_artifact_commit(self, commit: Any, error: BaseException) -> None:
        now = self.clock()
        with self.store.transaction() as connection:
            current = connection.execute("SELECT * FROM artifact_commits WHERE id=?", (commit["id"],)).fetchone()
            if current is None or current["state"] == "committed":
                return
            connection.execute(
                "UPDATE artifact_commits SET state='failed', error=?, finished_at=? WHERE id=?",
                (f"{type(error).__name__}: {error}", now, commit["id"]),
            )
            connection.execute(
                """UPDATE attempts SET status='failed_terminal', finished_at=?, error_code='artifact_commit_recovery',
                   message=?, cleanup_status='manual_review' WHERE id=?""",
                (now, str(error), commit["attempt_id"]),
            )
            connection.execute(
                """UPDATE stages SET state='failed_terminal', last_error_code='artifact_commit_recovery',
                   last_message=?, updated_at=? WHERE id=?""",
                (str(error), now, commit["stage_id"]),
            )
            connection.execute("DELETE FROM leases WHERE stage_id=?", (commit["stage_id"],))
            stage = connection.execute("SELECT batch_id FROM stages WHERE id=?", (commit["stage_id"],)).fetchone()
            self.store.event(
                connection,
                stage["batch_id"],
                "artifact_commit_failed",
                {"error": f"{type(error).__name__}: {error}"},
                now,
                stage_id=commit["stage_id"],
            )

    def _route_debug_bundle(self, connection: Any, failed_stage: Any, error_code: str, now: float) -> None:
        index = int(failed_stage["sort_index"])
        if not (REMOTE_EVIDENCE_START <= index <= REMOTE_EVIDENCE_END):
            return
        connection.execute(
            """UPDATE stages SET state='canceled', last_error_code='upstream_failed', updated_at=?
               WHERE batch_id=? AND project_id=? AND task_id=? AND sort_index>? AND sort_index<?
                 AND state IN ('pending','ready')""",
            (
                now,
                failed_stage["batch_id"],
                failed_stage["project_id"],
                failed_stage["task_id"],
                index,
                STAGE_INDEX["debug_bundle"],
            ),
        )
        connection.execute(
            """UPDATE stages SET state='ready', last_error_code=?, last_message='best-effort debug after upstream failure', updated_at=?
               WHERE batch_id=? AND project_id=? AND task_id=? AND name='debug_bundle' AND state='pending'""",
            (error_code, now, failed_stage["batch_id"], failed_stage["project_id"], failed_stage["task_id"]),
        )
        connection.execute(
            """UPDATE stages SET state='canceled', last_error_code='upstream_failed', updated_at=?
               WHERE batch_id=? AND project_id=? AND task_id=? AND sort_index>?
                 AND state IN ('pending','ready')""",
            (
                now,
                failed_stage["batch_id"],
                failed_stage["project_id"],
                failed_stage["task_id"],
                STAGE_INDEX["download_results"],
            ),
        )

    def _update_aggregate_status(self, connection: Any, batch_id: str, now: float) -> None:
        self._skip_incomplete_evaluation(connection, batch_id, now)
        for task in connection.execute("SELECT * FROM tasks WHERE batch_id=?", (batch_id,)).fetchall():
            stages = connection.execute(
                """SELECT name, state, last_error_code FROM stages
                   WHERE batch_id=? AND project_id=? AND task_id=?""",
                (batch_id, task["project_id"], task["task_id"]),
            ).fetchall()
            states = {row["state"] for row in stages}
            evaluation_skipped = all(
                row["state"] == "succeeded"
                or (
                    row["name"] == "evaluate_absolute"
                    and row["state"] == "canceled"
                    and row["last_error_code"] == EVALUATION_SKIPPED_ERROR
                )
                for row in stages
            )
            if states == {"succeeded"} or evaluation_skipped:
                status = "succeeded"
            elif "failed_terminal" in states:
                status = "failed"
            elif "rejected" in states:
                status = "rejected"
            elif states <= {"succeeded", "canceled"} and "canceled" in states:
                status = "canceled"
            elif "waiting_review" in states or "waiting_manual" in states:
                status = "waiting"
            else:
                status = "running"
            connection.execute(
                "UPDATE tasks SET status=? WHERE batch_id=? AND project_id=? AND task_id=?",
                (status, batch_id, task["project_id"], task["task_id"]),
            )
        statuses = {row["status"] for row in connection.execute("SELECT status FROM tasks WHERE batch_id=?", (batch_id,))}
        if statuses == {"succeeded"}:
            batch_status = "succeeded"
        elif statuses <= {"succeeded", "canceled"} and "canceled" in statuses:
            batch_status = "canceled"
        elif statuses <= {"succeeded", "failed", "rejected", "canceled"} and statuses & {"failed", "rejected"}:
            batch_status = "failed"
        else:
            batch_status = "running"
        connection.execute("UPDATE batches SET status=?, updated_at=? WHERE batch_id=?", (batch_status, now, batch_id))

    def _skip_incomplete_evaluation(self, connection: Any, batch_id: str, now: float) -> None:
        """Close an unrecoverable batch once every task reached the eval boundary.

        ``evaluate_absolute`` is a controller-only finalizer.  A task rejected
        before ``unity_eval6`` cancels its own finalizer, while successful peer
        tasks leave theirs ready.  Do not cancel those peers while another task
        is still executing; once every finalizer is ready/succeeded/canceled,
        no complete evaluation set can exist and the remaining finalizers must
        be canceled instead of leaving the batch permanently runnable.
        """
        evaluation_rows = connection.execute(
            """SELECT state FROM stages
               WHERE batch_id=? AND name='evaluate_absolute'""",
            (batch_id,),
        ).fetchall()
        if not evaluation_rows or any(
            row["state"] not in {"ready", "succeeded", "canceled"}
            for row in evaluation_rows
        ):
            return
        failed = connection.execute(
            """SELECT 1 FROM stages
               WHERE batch_id=? AND state IN ('failed_terminal', 'rejected')
               LIMIT 1""",
            (batch_id,),
        ).fetchone()
        if failed is None:
            return
        cursor = connection.execute(
            """UPDATE stages
               SET state='canceled', last_error_code=?,
                   last_message='其他任务未完成，未执行整批 GPTEval', updated_at=?
               WHERE batch_id=? AND name='evaluate_absolute' AND state='ready'""",
            (EVALUATION_SKIPPED_ERROR, now, batch_id),
        )
        if cursor.rowcount:
            self.store.event(
                connection,
                batch_id,
                "evaluation_skipped_incomplete_batch",
                {"canceledFinalizers": cursor.rowcount},
                now,
            )

    @staticmethod
    def _effective_stage_config(
        manifest: Mapping[str, Any],
        project: Mapping[str, Any],
        task: Mapping[str, Any] | str,
        stage_name: str,
    ) -> dict[str, Any]:
        task_config = {} if isinstance(task, str) else {key: value for key, value in task.items() if key != "taskId"}
        value = {
            "stage": stage_name,
            "defaultsRef": manifest["defaultsRef"],
            "renderProtocol": manifest["renderProtocol"],
            "remoteProfile": manifest["remoteProfile"],
            "project": {"projectPath": project["projectPath"], "scenePath": project["scenePath"]},
            "task": task_config,
        }
        if stage_name == "image_edit":
            value["editGeneration"] = 1
            task_identity = str(task.get("taskId")) if isinstance(task, Mapping) else "task"
            value["editGenerationGroup"] = f"{task_identity}-group-1"
        return value


def default_capacities(manifest: Mapping[str, Any]) -> dict[str, int]:
    resources = manifest["resources"]
    _mode, _initial, maximum = image_api_limits(resources)
    return {
        "unity_gpu": int(resources["unitySlots"]),
        "image_api": maximum,
        "remote_gpu": len(resources["remoteGpuPool"]),
        "ssh_io": 2,
        "remote_cpu": 4,
        "remote_io": 2,
        "evaluation_api": 2,
    }


def _require_batch(connection: Any, batch_id: str) -> Any:
    row = connection.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
    if row is None:
        raise StoreError(f"batch 不存在: {batch_id}")
    return row


def _idempotency_key(
    batch_id: str,
    project_id: str,
    task_id: str,
    stage: str,
    contract_version: str,
    input_hashes: list[str],
    config: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "batchId": batch_id,
            "projectId": project_id,
            "taskId": task_id,
            "stage": stage,
            "contractVersion": contract_version,
            "inputArtifactHashes": input_hashes,
            "effectiveConfigSha256": canonical_sha256(config),
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
