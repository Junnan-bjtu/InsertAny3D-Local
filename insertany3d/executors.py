"""Cheap fake execution plus a conservative subprocess adapter extension point."""

from __future__ import annotations

import json
import os
import subprocess
import time
import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import ContractError, validate_stage_result
from .processes import (
    ProcessIdentity,
    ProcessSupervisor,
    WindowsJobCommandBuilder,
    WindowsJobRunner,
    current_boot_id,
    process_start_ticks,
)
from .scheduler import BatchController, WorkItem
from .contracts.models import EVAL6_VIEW_LAYOUT, canonical_sha256


@dataclass(frozen=True)
class ExecutionResult:
    succeeded: bool
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    stage_status: str | None = None
    error_code: str | None = None
    message: str = ""
    retry_after_seconds: float | None = None
    cleanup_completed: bool = True


class FakeExecutor:
    """Deterministic stage executor used by tests and queue dry-runs."""

    def __init__(self, outcomes: Mapping[tuple[str, str, str, int], str] | None = None):
        self.outcomes = dict(outcomes or {})
        self.executed: list[tuple[str, str, str, int]] = []

    def execute(self, controller: BatchController, item: WorkItem) -> ExecutionResult:
        identity = (item.project_id, item.task_id, item.stage, item.attempt)
        self.executed.append(identity)
        controller.mark_running(
            item,
            pid=os.getpid(),
            pgid=os.getpgrp(),
            host_boot_id=_boot_id(),
            process_start_ticks=_process_start_ticks(os.getpid()),
        )
        controller.heartbeat(item.stage_id, item.lease_token, progress={"completed": 1, "total": 2, "unit": "fake"})
        outcome = self.outcomes.get(identity, "success")
        if outcome != "success":
            code, retry_after = {
                "429": ("http_429", 0.0),
                "503": ("http_503", 0.0),
                "timeout": ("delivery_unknown", None),
                "oom": ("resource_oom", 0.0),
                "crash": ("worker_crash", 0.0),
                "stall": ("stalled", 0.0),
                "invalid": ("invalid_input", None),
                "rejected": ("quality_rejected", None),
            }.get(outcome, (outcome, None))
            return ExecutionResult(False, error_code=code, message=f"fake outcome: {outcome}", retry_after_seconds=retry_after)
        if item.stage == "unity_eval6":
            return self._fake_eval6(item)
        output = item.staging_dir / ("edited.png" if item.stage == "image_edit" else "result.json")
        output.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "projectId": item.project_id,
                    "taskId": item.task_id,
                    "stage": item.stage,
                    "attempt": item.attempt,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return ExecutionResult(True, artifacts=[{"artifactId": "result", "type": "fake_stage_result", "path": output.name}])

    def _fake_eval6(self, item: WorkItem) -> ExecutionResult:
        """Write a tiny, structurally valid eval6 fixture for offline GPTEval."""
        # The parser only needs a verifiable PNG header; append task identity
        # so separate fake tasks cannot collide on the GPTEval request key.
        png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", 1024, 1024) + item.task_id.encode("ascii")
        artifacts: list[dict[str, Any]] = []
        views = []
        for index, (view_id, pitch, direction) in enumerate(EVAL6_VIEW_LAYOUT):
            original = item.staging_dir / "eval6" / "original" / f"{view_id}.png"
            inserted = item.staging_dir / "eval6" / "inserted" / f"{view_id}.png"
            camera = item.staging_dir / "eval6" / "cameras" / f"{view_id}.camera.json"
            for path in (original, inserted):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png)
                artifacts.append({"artifactId": path.stem + path.parent.name, "type": "image/png", "path": str(path.relative_to(item.staging_dir))})
            camera.parent.mkdir(parents=True, exist_ok=True)
            camera.write_text(json.dumps({"viewId": view_id, "pitchDegrees": pitch, "yawOffsetDegrees": direction * 24, "width": 1024, "height": 1024, "cameraToWorldMatrix": [float(index + 1), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "projectionMatrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}, sort_keys=True), encoding="utf-8")
            artifacts.append({"artifactId": camera.stem, "type": "application/json", "path": str(camera.relative_to(item.staging_dir))})
            views.append({"viewId": view_id, "pitchDegrees": pitch, "yawOffsetDegrees": direction * 24, "original": {"path": str(original.relative_to(item.staging_dir)), "sha256": hashlib.sha256(png).hexdigest()}, "inserted": {"path": str(inserted.relative_to(item.staging_dir)), "sha256": hashlib.sha256(png).hexdigest()}, "camera": {"path": str(camera.relative_to(item.staging_dir)), "sha256": hashlib.sha256(camera.read_bytes()).hexdigest()}})
        config = {"pitchDegrees": [10, 40], "yawOffsetDegrees": 24}
        manifest = {"schemaVersion": 1, "kind": "insertany3d.evaluation", "protocol": "eval6-v1", "batchId": item.batch_id, "projectId": item.project_id, "scenePath": "Assets/Farm_Test_001.unity", "taskId": item.task_id, "runId": f"fake-{item.attempt}", "methodId": "insertany3d-main", "taskPrompt": "offline fake", "viewConfig": {**config, "sha256": canonical_sha256(config)}, "render": {"width": 1024, "height": 1024, "cameraConvention": "unity-c2w-v1"}, "views": views}
        manifest_path = item.staging_dir / "evaluation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        artifacts.append({"artifactId": "evaluation_manifest", "type": "application/json", "path": manifest_path.name})
        return ExecutionResult(True, artifacts=artifacts)


class FakeRunner:
    """Lease work in capacity-sized waves so resource limits are exercised."""

    def __init__(self, controller: BatchController, capacities: Mapping[str, int], executor: FakeExecutor | None = None):
        self.controller = controller
        self.capacities = dict(capacities)
        self.executor = executor or FakeExecutor()
        self.peak_resources: dict[str, int] = {}

    def run_until_blocked(self, batch_id: str, *, max_waves: int = 10000) -> dict[str, Any]:
        for wave_number in range(max_waves):
            wave: list[WorkItem] = []
            while True:
                item = self.controller.lease_next(batch_id, f"fake-{wave_number}-{len(wave)}", self.capacities)
                if item is None:
                    break
                wave.append(item)
            if not wave:
                return self.controller.status(batch_id)
            counts: dict[str, int] = {}
            for item in wave:
                for resource in item.resources:
                    counts[resource] = counts.get(resource, 0) + 1
            for resource, count in counts.items():
                self.peak_resources[resource] = max(self.peak_resources.get(resource, 0), count)
            for item in wave:
                result = self.executor.execute(self.controller, item)
                if result.succeeded:
                    self.controller.commit_success(item, result.artifacts)
                else:
                    self.controller.fail(
                        item,
                        result.error_code or "worker_crash",
                        result.message,
                        retry_after=result.retry_after_seconds,
                        cleanup_completed=result.cleanup_completed,
                        stage_status=result.stage_status,
                    )
        raise RuntimeError(f"fake runner 超过 {max_waves} 轮，可能存在重试风暴")


class CommandExecutor:
    """Run a declared command in its own process group.

    Real Unity/SSH workers can adapt this interface later.  No heavy command is
    registered by default, so normal tests cannot accidentally start a model.
    """

    def __init__(
        self,
        heartbeat_seconds: float = 15.0,
        terminate_grace_seconds: float = 10.0,
        process_supervisor: ProcessSupervisor | None = None,
        windows_job_runner: WindowsJobCommandBuilder | None = None,
    ):
        self.heartbeat_seconds = heartbeat_seconds
        self.terminate_grace_seconds = terminate_grace_seconds
        self.process_supervisor = process_supervisor or ProcessSupervisor(grace_seconds=terminate_grace_seconds)
        self.windows_job_runner = windows_job_runner or WindowsJobRunner()

    def execute(
        self,
        controller: BatchController,
        item: WorkItem,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        canceled: Callable[[], bool] = lambda: False,
        timeout_seconds: float | None = None,
    ) -> ExecutionResult:
        if not command:
            raise ValueError("command 不能为空")
        stdout_path = item.staging_dir.parent / "stdout.log"
        stderr_path = item.staging_dir.parent / "stderr.log"
        launch_command = self.windows_job_runner.wrap(list(command), cwd=item.staging_dir.parent)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                launch_command,
                cwd=item.staging_dir.parent,
                env=dict(env) if env is not None else None,
                stdout=stdout,
                stderr=stderr,
                start_new_session=os.name != "nt",
            )
            try:
                self.process_supervisor.bind(process.pid)
            except BaseException:
                process.kill()
                process.wait()
                raise
            pgid = os.getpgid(process.pid) if os.name != "nt" else process.pid
            try:
                controller.mark_running(
                    item,
                    pid=process.pid,
                    pgid=pgid,
                    host_boot_id=_boot_id(),
                    process_start_ticks=_process_start_ticks(process.pid),
                )
                started = time.monotonic()
                while process.poll() is None:
                    cancel_requested = canceled()
                    if cancel_requested or (
                        timeout_seconds is not None
                        and time.monotonic() - started >= timeout_seconds
                    ):
                        identity = ProcessIdentity(
                            process.pid,
                            pgid,
                            _boot_id(),
                            _process_start_ticks(process.pid),
                        )
                        cleanup = self.process_supervisor.terminate(
                            identity,
                            diagnostics_dir=item.staging_dir.parent / "diagnostics" / "timeout",
                        )
                        try:
                            process.wait(timeout=max(1.0, self.terminate_grace_seconds))
                        except subprocess.TimeoutExpired:
                            pass
                        code = "canceled" if cancel_requested else "stalled"
                        return ExecutionResult(
                            False,
                            error_code=code,
                            message=code,
                            cleanup_completed=cleanup.completed,
                        )
                    controller.heartbeat(item.stage_id, item.lease_token)
                    time.sleep(self.heartbeat_seconds)
            finally:
                self.process_supervisor.release(process.pid)
        manifest_path = item.staging_dir / "stage_result.json"
        if not manifest_path.is_file():
            if process.returncode != 0:
                return ExecutionResult(False, error_code="worker_crash", message=f"exit code {process.returncode}")
            return ExecutionResult(False, error_code="compile_or_contract", message="缺少 stage_result.json")
        try:
            value = validate_stage_result(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (ContractError, json.JSONDecodeError, OSError) as exc:
            return ExecutionResult(False, error_code="compile_or_contract", message=f"stage_result 无效: {exc}")
        expected = {
            "batchId": item.batch_id,
            "projectId": item.project_id,
            "taskId": item.task_id,
            "stage": item.stage,
            "contractVersion": item.contract_version,
            "attempt": item.attempt,
            "leaseToken": item.lease_token,
        }
        mismatches = [key for key, expected_value in expected.items() if value.get(key) != expected_value]
        if mismatches:
            return ExecutionResult(
                False,
                error_code="compile_or_contract",
                message="stage_result 身份不匹配: " + ", ".join(mismatches),
            )
        cleanup_completed = bool(value["cleanup"]["completed"])
        if value["status"] == "succeeded":
            if process.returncode != 0:
                return ExecutionResult(
                    False,
                    error_code="worker_crash",
                    message=f"stage_result 声称成功但进程退出码为 {process.returncode}",
                    cleanup_completed=cleanup_completed,
                )
            if not cleanup_completed:
                return ExecutionResult(
                    False,
                    error_code="worker_crash",
                    message="stage_result 成功但进程清理未完成",
                    cleanup_completed=False,
                )
            return ExecutionResult(True, artifacts=list(value["artifacts"]), cleanup_completed=cleanup_completed)
        return ExecutionResult(
            False,
            stage_status=str(value["status"]),
            error_code=str(value.get("errorCode") or "worker_crash"),
            message=str(value.get("message") or value["status"]),
            cleanup_completed=cleanup_completed,
        )


def _boot_id() -> str:
    return current_boot_id()


def _process_start_ticks(pid: int) -> int:
    return process_start_ticks(pid) or 0
