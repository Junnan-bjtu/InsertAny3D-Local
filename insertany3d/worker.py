"""Small persistent worker loop for local batch execution.

The worker owns orchestration only: it leases one ready stage, delegates the
stage to an injected executor, and commits or classifies the result.  The
default executor is deterministic and local, so ``batch worker`` is safe to
use while wiring the real Unity/remote adapters.  Real workers can implement
the same ``StageExecutor`` protocol without changing queue semantics.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .dag import STAGE_INDEX
from .executors import ExecutionResult, FakeExecutor
from .scheduler import BatchController, WorkItem, default_capacities
from .store import StoreError


class StageExecutor(Protocol):
    """Executor contract used by the queue loop.

    The executor must not update stage state itself.  The worker applies the
    result through the controller so lease fencing and artifact publication
    stay in one place.
    """

    def execute(self, controller: BatchController, item: WorkItem) -> ExecutionResult:
        ...


@dataclass
class WorkerReport:
    """Machine-readable outcome of one worker invocation."""

    batch_id: str
    worker_id: str
    executor: str
    processed_stages: int = 0
    succeeded_stages: int = 0
    failed_stages: int = 0
    blocked_reason: str | None = None
    stage_names: list[str] = field(default_factory=list)
    submission_errors: list[dict[str, Any]] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "batchId": self.batch_id,
            "workerId": self.worker_id,
            "executor": self.executor,
            "processedStages": self.processed_stages,
            "succeededStages": self.succeeded_stages,
            "failedStages": self.failed_stages,
            "blockedReason": self.blocked_reason,
            "stageNames": list(self.stage_names),
            "submissionErrors": list(self.submission_errors),
            "status": self.status,
        }


class BatchWorker:
    """Lease and execute ready stages until the batch reaches a stop point.

    ``idle_polls`` is intentionally bounded.  A manual edit review is a valid
    stop point, not a reason to spin forever.  Re-invoking the worker after a
    reviewer accepts/regenerates a task resumes from the durable queue.
    """

    def __init__(
        self,
        controller: BatchController,
        capacities: Mapping[str, int],
        executor: StageExecutor | None = None,
        *,
        worker_id: str = "batch-worker",
        idle_sleep_seconds: float = 0.0,
        max_idle_polls: int = 1,
        max_parallel: int = 1,
        stage_names: Sequence[str] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        if not worker_id.strip():
            raise ValueError("worker_id 不能为空")
        if idle_sleep_seconds < 0:
            raise ValueError("idle_sleep_seconds 不能为负数")
        if max_idle_polls < 1:
            raise ValueError("max_idle_polls 必须大于 0")
        if max_parallel < 1:
            raise ValueError("max_parallel 必须大于 0")
        self.controller = controller
        self.capacities = dict(capacities)
        self.executor = executor or FakeExecutor()
        self.worker_id = worker_id
        self.idle_sleep_seconds = idle_sleep_seconds
        self.max_idle_polls = max_idle_polls
        self.max_parallel = max_parallel
        declared_stages = stage_names
        if declared_stages is None:
            declared_stages = getattr(self.executor, "supported_stages", None)
        self.stage_names = tuple(declared_stages) if declared_stages is not None else None
        if self.stage_names is not None:
            unknown = sorted(set(self.stage_names) - set(STAGE_INDEX))
            if unknown:
                raise ValueError("stage_names 包含未知步骤: " + ", ".join(unknown))
            if not self.stage_names:
                raise ValueError("stage_names 不能为空")
        self.clock = clock

    def run(
        self,
        batch_id: str,
        *,
        max_steps: int = 10000,
        once: bool = False,
    ) -> WorkerReport:
        """Run up to ``max_steps`` stages and return a durable status snapshot."""

        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        if once:
            max_steps = 1
        current = self.controller.status(batch_id)
        if current["status"] != "running":
            raise StoreError(
                f"batch {batch_id} 当前状态为 {current['status']}；请先执行 batch start 或 batch resume"
            )

        # Reconcile a controller restart before taking a new lease.  This is
        # cheap for a fresh batch and makes repeated invocations safe.
        self.controller.resume(batch_id)
        report = WorkerReport(
            batch_id=batch_id,
            worker_id=self.worker_id,
            executor=self._executor_name,
        )
        idle_polls = 0
        dispatch_index = 0
        while report.processed_stages < max_steps:
            wave: list[WorkItem] = []
            wave_size = min(self.max_parallel, max_steps - report.processed_stages)
            for _slot in range(wave_size):
                item = self._lease_next(batch_id, dispatch_index)
                if item is None:
                    break
                dispatch_index += 1
                wave.append(item)
                report.processed_stages += 1
                report.stage_names.append(item.stage)

            if not wave:
                idle_polls += 1
                report.blocked_reason = self._blocked_reason(batch_id)
                if once or idle_polls >= self.max_idle_polls:
                    break
                if self.idle_sleep_seconds:
                    time.sleep(self.idle_sleep_seconds)
                self.controller.refresh(batch_id)
                continue

            idle_polls = 0
            report.blocked_reason = None
            with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="insertany3d-stage") as pool:
                outcomes = list(pool.map(self._execute_safely, wave))
            for item, outcome in zip(wave, outcomes):
                if outcome.succeeded:
                    try:
                        self.controller.commit_success(item, outcome.artifacts)
                    except StoreError as exc:
                        report.submission_errors.append(
                            self._record_submission_error(item, "commit_success", exc)
                        )
                        report.failed_stages += 1
                        report.blocked_reason = "submission_error"
                    else:
                        report.succeeded_stages += 1
                else:
                    try:
                        self.controller.fail(
                            item,
                            outcome.error_code or "worker_crash",
                            outcome.message,
                            retry_after=outcome.retry_after_seconds,
                            cleanup_completed=outcome.cleanup_completed,
                            stage_status=outcome.stage_status,
                        )
                    except StoreError as exc:
                        report.submission_errors.append(
                            self._record_submission_error(item, "fail", exc)
                        )
                        report.failed_stages += 1
                        report.blocked_reason = "submission_error"
                    else:
                        report.failed_stages += 1

        report.status = self.controller.status(batch_id)
        if report.submission_errors:
            report.status["workerErrors"] = list(report.submission_errors)
        if (
            report.blocked_reason is None
            and report.processed_stages >= max_steps
            and report.status.get("status") == "running"
        ):
            report.blocked_reason = "max_steps_reached"
        return report

    def _record_submission_error(
        self,
        item: WorkItem,
        operation: str,
        error: StoreError,
    ) -> dict[str, Any]:
        """Persist a fenced submission failure without claiming stage success.

        A stale worker must not overwrite a newer attempt.  The controller's
        resume path first reconciles/recoveries the lease, after which this
        audit update only annotates the same stage and attempt for operators.
        """
        message = f"{operation} 提交失败: {error}"
        error_code = (
            "lease_fenced"
            if any(token in str(error).lower() for token in ("lease", "token", "租约", "失效"))
            else "store_error"
        )
        recovery_error: str | None = None
        try:
            self.controller.resume(item.batch_id)
        except StoreError as exc:
            recovery_error = f"恢复租约失败: {exc}"
            message = f"{message}; {recovery_error}"
        now = self.clock()
        try:
            with self.controller.store.transaction() as connection:
                stage = connection.execute(
                    "SELECT state FROM stages WHERE id=? AND batch_id=?",
                    (item.stage_id, item.batch_id),
                ).fetchone()
                attempt = connection.execute(
                    "SELECT id FROM attempts WHERE id=? AND stage_id=?",
                    (item.attempt_id, item.stage_id),
                ).fetchone()
                if stage is not None and attempt is not None:
                    connection.execute(
                        "UPDATE attempts SET error_code=?, message=? WHERE id=?",
                        (error_code, message, item.attempt_id),
                    )
                    # A newer attempt may have completed while this worker was
                    # returning.  Preserve that terminal stage state and only
                    # retain the stale attempt's audit row/event.
                    if stage["state"] not in {
                        "succeeded",
                        "failed_terminal",
                        "rejected",
                        "canceled",
                        "recovering",
                    }:
                        connection.execute(
                            "UPDATE stages SET last_error_code=?, last_message=?, updated_at=? WHERE id=?",
                            (error_code, message, now, item.stage_id),
                        )
                    self.controller.store.event(
                        connection,
                        item.batch_id,
                        "worker_submission_error",
                        {
                            "stage": item.stage,
                            "attempt": item.attempt,
                            "operation": operation,
                            "errorCode": error_code,
                            "message": message,
                        },
                        now,
                        stage_id=item.stage_id,
                    )
        except StoreError as exc:
            recovery_error = recovery_error or f"错误摘要写入失败: {exc}"
            message = f"{message}; {recovery_error}"
        return {
            "stage": item.stage,
            "projectId": item.project_id,
            "taskId": item.task_id,
            "attempt": item.attempt,
            "operation": operation,
            "errorCode": error_code,
            "message": message,
            "recoveryError": recovery_error,
        }

    @property
    def _executor_name(self) -> str:
        return "fake" if isinstance(self.executor, FakeExecutor) else type(self.executor).__name__

    def _lease_next(self, batch_id: str, dispatch_index: int) -> WorkItem | None:
        worker_id = f"{self.worker_id}-{dispatch_index}"
        if self.stage_names is None:
            return self.controller.lease_next(batch_id, worker_id, self.capacities)
        for stage_name in self.stage_names:
            item = self.controller.lease_next(
                batch_id,
                worker_id,
                self.capacities,
                stage_name=stage_name,
            )
            if item is not None:
                return item
        return None

    def _execute_safely(self, item: WorkItem) -> ExecutionResult:
        try:
            return self.executor.execute(self.controller, item)
        except Exception as exc:
            return ExecutionResult(
                False,
                error_code="worker_crash",
                message=f"executor 异常: {type(exc).__name__}: {exc}",
            )

    def _blocked_reason(self, batch_id: str) -> str:
        status = self.controller.status(batch_id)
        states = status.get("stageCounts", {})
        if status.get("status") in {"succeeded", "failed", "canceled"}:
            return f"batch_{status['status']}"
        if states.get("waiting_review", 0) or states.get("waiting_manual", 0):
            return "waiting_manual_review"
        if states.get("recovering", 0):
            return "recovering_cleanup"
        if states.get("leased", 0) or states.get("running", 0) or states.get("committing", 0):
            return "active_worker_or_resource_capacity"
        if states.get("ready", 0):
            if self.stage_names is not None:
                placeholders = ",".join("?" for _ in self.stage_names)
                row = self.controller.store.row(
                    f"SELECT COUNT(*) AS count FROM stages WHERE batch_id=? AND state='ready' AND name IN ({placeholders})",
                    (batch_id, *self.stage_names),
                )
                if row is not None and int(row["count"]) == 0:
                    return "no_supported_stage"
            return "resource_capacity_or_not_before"
        return "no_ready_stage"


def run_fake_batch(
    controller: BatchController,
    manifest: Mapping[str, Any],
    batch_id: str,
    *,
    worker_id: str = "fake-worker",
    max_steps: int = 10000,
    once: bool = False,
) -> WorkerReport:
    """Convenience wrapper used by the CLI and low-cost integration tests."""

    return BatchWorker(
        controller,
        default_capacities(manifest),
        FakeExecutor(),
        worker_id=worker_id,
    ).run(batch_id, max_steps=max_steps, once=once)
