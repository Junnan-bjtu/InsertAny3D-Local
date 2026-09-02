"""Explicit recovery for fenced SSH stage attempts.

This module never launches remote work.  A read-only probe identifies the
existing lease-derived control directory.  Only a verified RESULT can
reactivate and submit the original attempt; EXITED/MISSING require an explicit
retry or terminal decision before the resource lease is released.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, validate_stage_request, validate_stage_result
from .remote_worker import (
    CommandRunner,
    RemoteProfile,
    RemoteRunReport,
    RemoteStageRunner,
    SubprocessCommandRunner,
    build_remote_attempt_plan,
)
from .scheduler import BatchController, WorkItem
from .store import StoreError


class RemoteRecoveryError(StoreError):
    """The requested recovery action is unsafe or no longer current."""


@dataclass(frozen=True)
class RemoteRecoveryReport:
    batch_id: str
    project_id: str
    task_id: str
    stage: str
    attempt: int
    remote_state: str | None
    action: str
    scheduler_state: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "batchId": self.batch_id,
            "projectId": self.project_id,
            "taskId": self.task_id,
            "stage": self.stage,
            "attempt": self.attempt,
            "remoteState": self.remote_state,
            "action": self.action,
            "schedulerState": self.scheduler_state,
            "message": self.message,
        }


@dataclass(frozen=True)
class _RecoveryTarget:
    item: WorkItem
    root: Path
    request_path: Path


class RemoteRecoveryManager:
    """Probe and explicitly settle one delivery-unknown remote attempt."""

    def __init__(
        self,
        controller: BatchController,
        profile: RemoteProfile,
        *,
        command_runner: CommandRunner | None = None,
    ):
        self.controller = controller
        self.profile = profile
        self.runner = RemoteStageRunner(
            profile,
            command_runner=command_runner or SubprocessCommandRunner(),
        )

    def probe(
        self,
        batch_id: str,
        project_id: str,
        task_id: str,
        stage: str,
        attempt: int,
        lease_token: str,
    ) -> RemoteRecoveryReport:
        """Read remote state without changing files, leases, or scheduler state."""

        target = self._target(batch_id, project_id, task_id, stage, attempt, lease_token)
        remote = self.runner.probe_existing(target.request_path, target.root)
        message = _probe_message(remote)
        return self._report(target.item, remote.remote_state, "probe", "recovering", message)

    def recover_result(
        self,
        batch_id: str,
        project_id: str,
        task_id: str,
        stage: str,
        attempt: int,
        lease_token: str,
    ) -> RemoteRecoveryReport:
        """Download and submit RESULT using the original attempt and lease."""

        target = self._target(batch_id, project_id, task_id, stage, attempt, lease_token)
        first_probe = self.runner.probe_existing(target.request_path, target.root)
        if first_probe.remote_state != "RESULT":
            return self._report(
                target.item,
                first_probe.remote_state,
                "recover_result",
                "recovering",
                "远端尚无可提交 RESULT；租约和资源保持占用",
            )

        self._archive_transport_failure(target)
        downloaded = self.runner.download_existing_result(target.request_path, target.root)
        if downloaded.remote_state != "RESULT" or downloaded.classification != "result_downloaded":
            return self._report(
                target.item,
                downloaded.remote_state,
                "recover_result",
                "recovering",
                "RESULT 下载未完成；原租约仍保持撤销和占用",
            )

        result = self._validated_downloaded_result(target)
        if result["status"] == "succeeded" and not bool(result["cleanup"]["completed"]):
            return self._report(
                target.item,
                "RESULT",
                "recover_result",
                "recovering",
                "远端结果结构和产物已校验，但 cleanup=false；租约和 GPU 资源继续占用",
            )
        active_item = self.controller.reactivate_recovering_item(target.item)
        if result["status"] == "succeeded":
            self.controller.commit_success(active_item, list(result["artifacts"]))
            scheduler_state = "succeeded"
        else:
            scheduler_state = self.controller.fail(
                active_item,
                str(result.get("errorCode") or "worker_crash"),
                str(result.get("message") or result["status"]),
                cleanup_completed=bool(result["cleanup"]["completed"]),
                stage_status=str(result["status"]),
            )
        return self._report(
            active_item,
            "RESULT",
            "recover_result",
            scheduler_state,
            "已校验并提交原远端 attempt 的结构化结果",
        )

    def resolve_stopped(
        self,
        batch_id: str,
        project_id: str,
        task_id: str,
        stage: str,
        attempt: int,
        lease_token: str,
        *,
        action: str,
        message: str | None = None,
    ) -> RemoteRecoveryReport:
        """Explicitly retry or terminate only after observing EXITED/MISSING."""

        target = self._target(batch_id, project_id, task_id, stage, attempt, lease_token)
        remote = self.runner.probe_existing(target.request_path, target.root)
        if remote.remote_state not in {"EXITED", "MISSING"}:
            raise RemoteRecoveryError(
                f"远端状态是 {remote.remote_state or 'UNKNOWN'}；只有 EXITED/MISSING 才能选择 retry/terminal"
            )
        state = self.controller.resolve_recovering_item(
            target.item,
            observed_remote_state=remote.remote_state,
            action=action,
            message=message,
        )
        return self._report(
            target.item,
            remote.remote_state,
            action,
            state,
            "已按显式决定释放原远端资源租约",
        )

    def cancel_running(
        self,
        batch_id: str,
        project_id: str,
        task_id: str,
        stage: str,
        attempt: int,
        lease_token: str,
        *,
        action: str,
        message: str | None = None,
    ) -> RemoteRecoveryReport:
        """Cancel only a verified leader, then release only after its group is empty."""

        target = self._target(batch_id, project_id, task_id, stage, attempt, lease_token)
        before = self.runner.probe_existing(target.request_path, target.root)
        if before.remote_state != "RUNNING":
            raise RemoteRecoveryError(
                f"远端状态是 {before.remote_state or 'UNKNOWN'}；只有顶层身份仍匹配的 RUNNING 才能安全取消"
            )
        cleanup = self.runner.cancel_existing(target.request_path, target.root)
        if cleanup.remote_state != "CLEANED":
            return self._report(
                target.item,
                cleanup.remote_state,
                "cancel_running",
                "recovering",
                "远端进程组未证明为空；租约和 GPU 资源继续占用",
            )
        state = self.controller.resolve_recovering_item(
            target.item,
            observed_remote_state="EXITED",
            action=action,
            message=message or "显式取消已确认远端进程组为空",
        )
        return self._report(
            target.item,
            "CLEANED",
            "cancel_running",
            state,
            "已终止完整远端进程组并释放资源租约",
        )

    def _target(
        self,
        batch_id: str,
        project_id: str,
        task_id: str,
        stage: str,
        attempt: int,
        lease_token: str,
    ) -> _RecoveryTarget:
        if not lease_token:
            raise RemoteRecoveryError("lease token 不能为空")
        item = self.controller.get_recovering_item(
            batch_id,
            project_id,
            task_id,
            stage,
            attempt,
            lease_token,
        )
        batch = self.controller.store.row(
            "SELECT root_path FROM batches WHERE batch_id=?",
            (batch_id,),
        )
        stage_row = self.controller.store.row(
            "SELECT idempotency_key, last_error_code FROM stages WHERE id=?",
            (item.stage_id,),
        )
        if batch is None or stage_row is None:
            raise RemoteRecoveryError("恢复 attempt 的 batch/stage 已不存在")
        if stage_row["last_error_code"] != "delivery_unknown":
            raise RemoteRecoveryError("只有 delivery_unknown 的远端 attempt 可走此恢复入口")

        root = _strict_root(Path(batch["root_path"]))
        attempt_root = _strict_child(
            root,
            root / item.project_id / item.task_id / "stages" / item.stage / f"attempt-{item.attempt:04d}",
            "attempt root",
        )
        expected_staging = attempt_root / "output.staging"
        if _strict_child(root, item.staging_dir, "staging_dir") != expected_staging.resolve():
            raise RemoteRecoveryError("数据库 staging_dir 不符合该 attempt 的固定目录")
        expected_output = root / item.project_id / item.task_id / "artifacts" / item.stage / stage_row["idempotency_key"]
        if _strict_child(root, item.output_dir, "output_dir") != expected_output.resolve():
            raise RemoteRecoveryError("数据库 output_dir 不符合该 stage 的固定目录")

        request_path = root / "requests" / item.project_id / item.task_id / item.stage / f"attempt-{item.attempt:04d}.json"
        request_path = _strict_existing_file(root, request_path, "原 stage request")
        try:
            request = validate_stage_request(json.loads(request_path.read_text(encoding="utf-8")))
        except (ContractError, json.JSONDecodeError, OSError) as exc:
            raise RemoteRecoveryError(f"原 stage request 无效: {exc}") from exc
        expected_request = self.controller.build_stage_request(item)
        if request != expected_request:
            raise RemoteRecoveryError("原 stage request 与当前数据库契约不完全一致，拒绝恢复")
        build_remote_attempt_plan(self.profile, request_path, root)
        return _RecoveryTarget(item, root, request_path)

    def _archive_transport_failure(self, target: _RecoveryTarget) -> None:
        staging = target.item.staging_dir
        if not staging.is_dir():
            raise RemoteRecoveryError("恢复下载前本地 staging_dir 不存在")
        children = list(staging.iterdir())
        if not children:
            return
        for path in staging.rglob("*"):
            if path.is_symlink():
                raise RemoteRecoveryError("本地恢复 staging 不能包含符号链接")
        previous_result = staging / "stage_result.json"
        if not previous_result.is_file():
            raise RemoteRecoveryError("非空恢复 staging 缺少旧 stage_result.json，拒绝覆盖")
        try:
            value = validate_stage_result(json.loads(previous_result.read_text(encoding="utf-8")))
        except (ContractError, json.JSONDecodeError, OSError) as exc:
            raise RemoteRecoveryError(f"旧 transport 结果无效，拒绝覆盖: {exc}") from exc
        _require_result_identity(value, target.item)
        if value.get("errorCode") != "delivery_unknown" or bool(value["cleanup"]["completed"]):
            raise RemoteRecoveryError("旧 staging 不是 cleanup=false 的 delivery_unknown 证据")

        evidence_root = staging.parent / "recovery-evidence"
        evidence_root.mkdir(parents=True, exist_ok=True)
        if evidence_root.is_symlink() or not evidence_root.is_dir():
            raise RemoteRecoveryError("恢复证据目录必须是 attempt 内的普通目录")
        for index in range(1, 1000):
            destination = evidence_root / f"transport-{index:04d}"
            if not destination.exists():
                staging.replace(destination)
                staging.mkdir(parents=True, exist_ok=False)
                return
        raise RemoteRecoveryError("恢复证据目录数量异常，拒绝覆盖旧证据")

    def _validated_downloaded_result(self, target: _RecoveryTarget) -> Mapping[str, Any]:
        path = _strict_existing_file(target.root, target.item.staging_dir / "stage_result.json", "下载结果")
        try:
            value = validate_stage_result(json.loads(path.read_text(encoding="utf-8")))
        except (ContractError, json.JSONDecodeError, OSError) as exc:
            raise RemoteRecoveryError(f"下载结果无效: {exc}") from exc
        _require_result_identity(value, target.item)
        return value

    @staticmethod
    def _report(
        item: WorkItem,
        remote_state: str | None,
        action: str,
        scheduler_state: str,
        message: str,
    ) -> RemoteRecoveryReport:
        return RemoteRecoveryReport(
            item.batch_id,
            item.project_id,
            item.task_id,
            item.stage,
            item.attempt,
            remote_state,
            action,
            scheduler_state,
            message,
        )


def _require_result_identity(value: Mapping[str, Any], item: WorkItem) -> None:
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
        raise RemoteRecoveryError("stage result 身份不匹配: " + ", ".join(mismatches))


def _strict_root(path: Path) -> Path:
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise RemoteRecoveryError("batch root 必须是存在且非符号链接的绝对目录")
    return path.resolve()


def _strict_child(root: Path, path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise RemoteRecoveryError(f"{label} 必须是绝对路径")
    current = root
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RemoteRecoveryError(f"{label} 越出 batch root") from exc
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RemoteRecoveryError(f"{label} 不能经过符号链接")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RemoteRecoveryError(f"{label} 解析后越出 batch root") from exc
    return resolved


def _strict_existing_file(root: Path, path: Path, label: str) -> Path:
    resolved = _strict_child(root, path, label)
    if not resolved.is_file():
        raise RemoteRecoveryError(f"{label} 不存在或不是普通文件")
    return resolved


def _probe_message(report: RemoteRunReport) -> str:
    if report.remote_state == "RUNNING":
        return "远端原进程仍在运行；资源继续占用，不允许重试"
    if report.remote_state == "GROUP_RUNNING":
        return "远端顶层已退出但同组仍有子进程；身份不足以自动终止，资源继续占用"
    if report.remote_state == "RESULT":
        return "远端已有结构化结果；可显式执行 recover-result 下载并提交"
    if report.remote_state in {"EXITED", "MISSING"}:
        return "远端已停止或控制目录缺失；需显式选择 retry 或 terminal"
    if report.remote_state == "IDENTITY_INVALID":
        return "远端进程身份记录不完整或不匹配；资源继续占用，需先人工排查"
    return "SSH 未能确认远端状态；资源继续占用"
