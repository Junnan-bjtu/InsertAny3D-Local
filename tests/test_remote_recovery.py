from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from insertany3d.executors import FakeExecutor
from insertany3d.remote_recovery import RemoteRecoveryError, RemoteRecoveryManager
from insertany3d.remote_worker import CommandOutcome, RemoteProfile
from insertany3d.scheduler import BatchController, WorkItem, default_capacities
from insertany3d.store import SchedulerStore
from tests.fixtures import batch_manifest


class Clock:
    def __init__(self, value: float = 1000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeSsh:
    def __init__(self, states: Sequence[str], *, result_factory=None):
        self.states = list(states)
        self.result_factory = result_factory
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: Sequence[str], *, timeout_seconds: float | None = None) -> CommandOutcome:
        del timeout_seconds
        values = tuple(str(value) for value in command)
        self.commands.append(values)
        if values[0] == "scp":
            destination = Path(values[-1])
            destination.mkdir(parents=True, exist_ok=False)
            if self.result_factory is None:
                raise AssertionError("unexpected fake SCP")
            self.result_factory(destination)
            return CommandOutcome(0, "", "")
        if not self.states:
            raise AssertionError("unexpected fake SSH probe")
        return CommandOutcome(0, self.states.pop(0) + "\n", "")


class RemoteRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.clock = Clock()
        self.store = SchedulerStore(self.root / "state.sqlite3")
        self.controller = BatchController(self.store, clock=self.clock, lease_seconds=30)
        self.manifest = batch_manifest(mode="automatic", project_count=1)
        # This fixture exercises remote lease recovery, not candidate-group
        # scheduling; keep its image stage single-candidate and let the
        # multi-generation tests cover the default group size of three.
        self.manifest["projects"][0]["tasks"][0]["num_gen_image_per_task"] = 1
        self.run_root = self.root / "runs"
        self.controller.plan(self.manifest, self.run_root, formal=False)
        self.controller.start(self.manifest["batchId"], formal=False)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _recovering_model(self) -> WorkItem:
        capacities = default_capacities(self.manifest)
        for stage_name in ("unity_anchor", "image_edit", "upload_inputs"):
            self.controller.refresh(self.manifest["batchId"])
            item = self.controller.lease_next(
                self.manifest["batchId"],
                "setup",
                capacities,
                project_id="Scene_01",
                task_id="Task_001",
                stage_name=stage_name,
            )
            self.assertIsNotNone(item, stage_name)
            outcome = FakeExecutor().execute(self.controller, item)
            self.controller.commit_success(item, outcome.artifacts)
        self.controller.refresh(self.manifest["batchId"])
        item = self.controller.lease_next(
            self.manifest["batchId"],
            "remote-worker",
            capacities,
            project_id="Scene_01",
            task_id="Task_001",
            stage_name="model_generation",
        )
        self.assertIsNotNone(item)
        self.controller.write_stage_request(item)
        self._write_unknown_result(item)
        state = self.controller.fail(
            item,
            "delivery_unknown",
            "fake SSH disconnected",
            cleanup_completed=False,
            stage_status="failed_retryable",
        )
        self.assertEqual(state, "recovering")
        return item

    @staticmethod
    def _write_unknown_result(item: WorkItem) -> None:
        value = {
            "schemaVersion": 1,
            "kind": "insertany3d.stage-result",
            "batchId": item.batch_id,
            "projectId": item.project_id,
            "taskId": item.task_id,
            "stage": item.stage,
            "contractVersion": item.contract_version,
            "attempt": item.attempt,
            "leaseToken": item.lease_token,
            "status": "failed_retryable",
            "artifacts": [],
            "errorCode": "delivery_unknown",
            "message": "fake disconnect",
            "diagnosticPaths": [],
            "cleanup": {"completed": False},
            "finishedAtUtc": "2026-08-29T12:00:00Z",
        }
        item.staging_dir.joinpath("stage_result.json").write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def _profile() -> RemoteProfile:
        return RemoteProfile(
            target="worker@example.test",
            project_root="/srv/InsertAny3D",
            artifact_root="/srv/InsertRuns",
        )

    @staticmethod
    def _identity(item: WorkItem) -> tuple[str, str, str, str, int, str]:
        return item.batch_id, item.project_id, item.task_id, item.stage, item.attempt, item.lease_token

    @staticmethod
    def _success_result(item: WorkItem, *, cleanup_completed: bool):
        def write_result(destination: Path) -> None:
            artifact = destination / "sample.ply"
            artifact.write_bytes(b"ply\nremote result\n")
            value = {
                "schemaVersion": 1,
                "kind": "insertany3d.stage-result",
                "batchId": item.batch_id,
                "projectId": item.project_id,
                "taskId": item.task_id,
                "stage": item.stage,
                "contractVersion": item.contract_version,
                "attempt": item.attempt,
                "leaseToken": item.lease_token,
                "status": "succeeded",
                "artifacts": [
                    {
                        "artifactId": "model",
                        "type": "ply",
                        "path": "sample.ply",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "size": artifact.stat().st_size,
                    }
                ],
                "errorCode": None,
                "message": "",
                "diagnosticPaths": [],
                "cleanup": {"completed": cleanup_completed},
                "finishedAtUtc": "2026-08-29T12:00:00Z",
            }
            destination.joinpath("stage_result.json").write_text(json.dumps(value), encoding="utf-8")

        return write_result

    def test_resume_never_releases_expired_revoked_recovery_lease(self) -> None:
        item = self._recovering_model()
        self.clock.advance(3600)
        self.assertEqual(self.controller.resume(item.batch_id), 0)
        stage = self.store.row("SELECT state FROM stages WHERE id=?", (item.stage_id,))
        lease = self.store.row("SELECT revoked_at FROM leases WHERE stage_id=?", (item.stage_id,))
        self.assertEqual(stage["state"], "recovering")
        self.assertIsNotNone(lease["revoked_at"])

    def test_probe_running_is_read_only_and_keeps_resource_fenced(self) -> None:
        item = self._recovering_model()
        ssh = FakeSsh(["RUNNING"])
        manager = RemoteRecoveryManager(self.controller, self._profile(), command_runner=ssh)
        before = self.store.row("SELECT * FROM leases WHERE stage_id=?", (item.stage_id,))
        report = manager.probe(*self._identity(item))
        after = self.store.row("SELECT * FROM leases WHERE stage_id=?", (item.stage_id,))
        self.assertEqual(report.remote_state, "RUNNING")
        self.assertEqual(dict(before), dict(after))
        self.assertEqual(self.controller.status(item.batch_id)["stageCounts"]["recovering"], 1)
        self.assertIn("process.identity", ssh.commands[0][-1])
        self.assertIn("saved_ticks", ssh.commands[0][-1])
        self.assertIn("saved_pgid", ssh.commands[0][-1])

    def test_result_download_reactivates_same_attempt_and_commits(self) -> None:
        item = self._recovering_model()
        ssh = FakeSsh(["RESULT", "RESULT"], result_factory=self._success_result(item, cleanup_completed=True))
        manager = RemoteRecoveryManager(self.controller, self._profile(), command_runner=ssh)
        report = manager.recover_result(*self._identity(item))
        self.assertEqual(report.scheduler_state, "succeeded")
        self.assertIsNone(self.store.row("SELECT * FROM leases WHERE stage_id=?", (item.stage_id,)))
        attempt = self.store.row("SELECT status FROM attempts WHERE id=?", (item.attempt_id,))
        self.assertEqual(attempt["status"], "succeeded")
        artifact = self.store.row("SELECT * FROM artifacts WHERE stage_id=?", (item.stage_id,))
        self.assertEqual(artifact["sha256"], hashlib.sha256(b"ply\nremote result\n").hexdigest())
        evidence = item.staging_dir.parent / "recovery-evidence" / "transport-0001" / "stage_result.json"
        self.assertTrue(evidence.is_file())

    def test_success_result_with_incomplete_cleanup_keeps_remote_resource_fenced(self) -> None:
        item = self._recovering_model()
        ssh = FakeSsh(["RESULT", "RESULT"], result_factory=self._success_result(item, cleanup_completed=False))
        manager = RemoteRecoveryManager(self.controller, self._profile(), command_runner=ssh)

        report = manager.recover_result(*self._identity(item))

        self.assertEqual(report.scheduler_state, "recovering")
        self.assertIn("cleanup=false", report.message)
        stage = self.store.row("SELECT state FROM stages WHERE id=?", (item.stage_id,))
        lease = self.store.row("SELECT revoked_at FROM leases WHERE stage_id=?", (item.stage_id,))
        attempt = self.store.row("SELECT status FROM attempts WHERE id=?", (item.attempt_id,))
        self.assertEqual(stage["state"], "recovering")
        self.assertIsNotNone(lease["revoked_at"])
        self.assertNotEqual(attempt["status"], "succeeded")
        self.assertIsNone(self.store.row("SELECT * FROM artifacts WHERE stage_id=?", (item.stage_id,)))

    def test_exited_requires_explicit_choice_before_release(self) -> None:
        item = self._recovering_model()
        manager = RemoteRecoveryManager(self.controller, self._profile(), command_runner=FakeSsh(["EXITED"]))
        report = manager.resolve_stopped(*self._identity(item), action="retry")
        self.assertEqual(report.scheduler_state, "ready")
        self.assertIsNone(self.store.row("SELECT * FROM leases WHERE stage_id=?", (item.stage_id,)))
        replacement = self.controller.lease_next(
            item.batch_id,
            "retry-worker",
            default_capacities(self.manifest),
            project_id=item.project_id,
            task_id=item.task_id,
            stage_name=item.stage,
        )
        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.attempt, item.attempt + 1)
        self.assertNotEqual(replacement.lease_token, item.lease_token)

    def test_missing_can_only_be_explicitly_marked_terminal(self) -> None:
        item = self._recovering_model()
        manager = RemoteRecoveryManager(self.controller, self._profile(), command_runner=FakeSsh(["MISSING"]))
        report = manager.resolve_stopped(*self._identity(item), action="terminal")
        self.assertEqual(report.scheduler_state, "failed_terminal")
        self.assertIsNone(self.store.row("SELECT * FROM leases WHERE stage_id=?", (item.stage_id,)))
        stage = self.store.row("SELECT state, last_error_code FROM stages WHERE id=?", (item.stage_id,))
        self.assertEqual(stage["state"], "failed_terminal")
        self.assertEqual(stage["last_error_code"], "remote_stopped_terminal")

    def test_running_and_invalid_identity_cannot_release_resource(self) -> None:
        item = self._recovering_model()
        for remote_state in ("RUNNING", "GROUP_RUNNING", "IDENTITY_INVALID"):
            with self.subTest(remote_state=remote_state):
                manager = RemoteRecoveryManager(
                    self.controller,
                    self._profile(),
                    command_runner=FakeSsh([remote_state]),
                )
                with self.assertRaises(RemoteRecoveryError):
                    manager.resolve_stopped(*self._identity(item), action="terminal")
                lease = self.store.row("SELECT revoked_at FROM leases WHERE stage_id=?", (item.stage_id,))
                self.assertIsNotNone(lease["revoked_at"])

    def test_explicit_cancel_releases_only_after_group_is_confirmed_empty(self) -> None:
        item = self._recovering_model()
        manager = RemoteRecoveryManager(
            self.controller,
            self._profile(),
            command_runner=FakeSsh(["RUNNING", "CLEANED"]),
        )
        report = manager.cancel_running(*self._identity(item), action="terminal")
        self.assertEqual(report.scheduler_state, "failed_terminal")
        self.assertEqual(report.remote_state, "CLEANED")
        self.assertIsNone(self.store.row("SELECT * FROM leases WHERE stage_id=?", (item.stage_id,)))

    def test_incomplete_cancel_keeps_remote_resource_fenced(self) -> None:
        item = self._recovering_model()
        manager = RemoteRecoveryManager(
            self.controller,
            self._profile(),
            command_runner=FakeSsh(["RUNNING", "GROUP_REMAINS"]),
        )
        report = manager.cancel_running(*self._identity(item), action="terminal")
        self.assertEqual(report.scheduler_state, "recovering")
        self.assertEqual(report.remote_state, "GROUP_REMAINS")
        lease = self.store.row("SELECT revoked_at FROM leases WHERE stage_id=?", (item.stage_id,))
        self.assertIsNotNone(lease["revoked_at"])

    def test_unknown_descendants_after_cancel_keep_remote_resource_fenced(self) -> None:
        item = self._recovering_model()
        manager = RemoteRecoveryManager(
            self.controller,
            self._profile(),
            command_runner=FakeSsh(["RUNNING", "DESCENDANTS_UNKNOWN"]),
        )
        report = manager.cancel_running(*self._identity(item), action="terminal")
        self.assertEqual(report.scheduler_state, "recovering")
        self.assertEqual(report.remote_state, "DESCENDANTS_UNKNOWN")
        self.assertIsNotNone(self.store.row("SELECT * FROM leases WHERE stage_id=?", (item.stage_id,)))

    def test_tampered_original_request_is_rejected_before_ssh(self) -> None:
        item = self._recovering_model()
        request_path = self.run_root / "requests" / item.project_id / item.task_id / item.stage / "attempt-0001.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["effectiveConfig"]["stageOptions"]["gpuDevice"] = "99"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        ssh = FakeSsh([])
        manager = RemoteRecoveryManager(self.controller, self._profile(), command_runner=ssh)
        with self.assertRaises(RemoteRecoveryError):
            manager.probe(*self._identity(item))
        self.assertEqual(ssh.commands, [])
        self.assertIsNotNone(self.store.row("SELECT * FROM leases WHERE stage_id=?", (item.stage_id,)))


if __name__ == "__main__":
    unittest.main()
