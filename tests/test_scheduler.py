from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from insertany3d.dag import STAGES
from insertany3d.executors import CommandExecutor, FakeExecutor, FakeRunner
from insertany3d.scheduler import BatchController, LeaseFencedError, default_capacities
from insertany3d.stage_wiring import (
    StageWiringError,
    _workspace_result_root,
    ensure_unity_project_not_running,
    unity_command_path,
)
from insertany3d.store import SchedulerStore, StoreError
from insertany3d.worker import BatchWorker
from tests.fixtures import batch_manifest


class Clock:
    def __init__(self, value: float = 1000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def create_controller(tmp_path: Path, *, mode: str = "automatic", clock=None, num_generations: int | None = None):
    store = SchedulerStore(tmp_path / "state.sqlite3")
    controller = BatchController(store, clock=clock or Clock(), lease_seconds=30)
    manifest = batch_manifest(mode=mode)
    if num_generations is not None:
        for project in manifest["projects"]:
            for task in project["tasks"]:
                if isinstance(task, dict):
                    task["num_gen_image_per_task"] = num_generations
    controller.plan(manifest, tmp_path / "runs")
    controller.start(manifest["batchId"])
    return store, controller, manifest


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="insertany3d_scheduler_")
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_unity_command_path_converts_wsl_drive(self) -> None:
        self.assertEqual(
            unity_command_path("/mnt/z/synthetic/UnityProject"),
            r"Z:\synthetic\UnityProject",
        )

    @patch("insertany3d.stage_wiring.shutil.which", return_value="powershell.exe")
    @patch("insertany3d.stage_wiring.subprocess.run")
    def test_unity_project_guard_reports_existing_pid(self, run, _which) -> None:
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout='{"ProcessId":108504,"CommandLine":"Unity.exe -projectpath F:\\\\Farm"}',
            stderr="",
        )
        with self.assertRaisesRegex(StageWiringError, "108504"):
            ensure_unity_project_not_running("/mnt/z/synthetic/Farm")

    def test_status_exposes_resource_queue_position_and_pending_predecessor(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path)
        try:
            initial = controller.status(manifest["batchId"])
            first = next(task for task in initial["tasks"] if task["task_id"] == "Task_001")
            self.assertEqual(first["current_stage"], "unity_anchor")
            self.assertEqual(first["queueResource"], "unity")
            self.assertEqual(first["resourceQueuePosition"], 1)

            stages = store.rows(
                "SELECT id, name, sort_index FROM stages WHERE batch_id=? AND project_id=? AND task_id=? ORDER BY sort_index",
                (manifest["batchId"], "Scene_01", "Task_001"),
            )
            upload_index = next(row["sort_index"] for row in stages if row["name"] == "upload_inputs")
            upload_stage = next(row for row in stages if row["name"] == "upload_inputs")
            model_stage = next(row for row in stages if row["name"] == "model_generation")
            with store.transaction() as connection:
                connection.execute(
                    "UPDATE stages SET state='succeeded' WHERE batch_id=? AND project_id=? AND task_id=? AND sort_index<?",
                    (manifest["batchId"], "Scene_01", "Task_001", upload_index),
                )
                connection.execute("UPDATE stages SET state='running' WHERE id=?", (upload_stage["id"],))
                connection.execute("UPDATE stages SET state='pending' WHERE id=?", (model_stage["id"],))

            # The normal status query selects the earliest non-successful
            # stage (upload_inputs here).  Exercise the resolver directly to
            # verify the downstream pending stage still identifies that
            # predecessor when a caller needs its blocker details.
            blocker = controller._status_blocker(
                int(model_stage["id"]),
                ready_positions={},
                resource_ready_positions={},
                now=1000.0,
            )
            self.assertIsNotNone(blocker)
            self.assertEqual(blocker["stage"], "upload_inputs")
            self.assertEqual(blocker["stageState"], "running")
        finally:
            store.close()

    def test_sixty_fake_tasks_finish_full_dag_with_resource_caps(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path)
        try:
            capacities = default_capacities(manifest)
            runner = FakeRunner(controller, capacities)
            status = runner.run_until_blocked(manifest["batchId"])
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(status["stageCounts"], {"succeeded": 60 * len(STAGES)})
            for resource, peak in runner.peak_resources.items():
                self.assertLessEqual(peak, capacities.get(resource, 60))
            self.assertEqual(runner.peak_resources["unity_gpu"], 1)
            self.assertLessEqual(runner.peak_resources["image_api"], 4)
            self.assertLessEqual(runner.peak_resources["remote_gpu"], 2)
            first_stage_by_task = {}
            for project_id, task_id, stage, _attempt in runner.executor.executed:
                first_stage_by_task.setdefault((project_id, task_id), stage)
            self.assertEqual(len(first_stage_by_task), 60)
            self.assertEqual(set(first_stage_by_task.values()), {"unity_anchor"})
            self.assertFalse(list((self.tmp_path / "runs").glob("**/output.staging/result.json")))
        finally:
            store.close()

    def test_manual_mode_pages_five_and_decisions_are_task_local(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path, mode="manual")
        try:
            runner = FakeRunner(controller, default_capacities(manifest))
            status = runner.run_until_blocked(manifest["batchId"])
            self.assertEqual(status["status"], "running")
            page = controller.review_page(manifest["batchId"])
            self.assertEqual(len(page), 5)
            self.assertTrue(all(len(item["editArtifacts"]) == 3 for item in page))
            self.assertTrue(all(item["editArtifacts"][0]["sha256"] for item in page))
            first = page[0]
            controller.decide_edit(
                manifest["batchId"],
                first["project_id"],
                first["task_id"],
                first["edit_attempt"],
                "accepted",
                decided_by="tester",
            )
            runner.run_until_blocked(manifest["batchId"])
            accepted = store.row(
                "SELECT status FROM tasks WHERE batch_id=? AND project_id=? AND task_id=?",
                (manifest["batchId"], first["project_id"], first["task_id"]),
            )
            self.assertEqual(accepted["status"], "succeeded")
            self.assertEqual(len(controller.review_page(manifest["batchId"])), 5)
            pending = store.row("SELECT COUNT(*) AS count FROM edit_reviews WHERE status='pending_review'")
            self.assertEqual(int(pending["count"]), 59)
        finally:
            store.close()

    def test_expired_lease_is_recovered_and_old_token_is_fenced(self) -> None:
        clock = Clock()
        store, controller, manifest = create_controller(self.tmp_path, clock=clock)
        try:
            capacities = default_capacities(manifest)
            old = controller.lease_next(manifest["batchId"], "worker-old", capacities)
            self.assertIsNotNone(old)
            # This test isolates lease recovery from unity_anchor's one-attempt
            # production policy so a replacement lease can be observed.
            with store.transaction() as connection:
                connection.execute("UPDATE stages SET max_attempts=2 WHERE id=?", (old.stage_id,))
            controller.mark_running(old, pid=10, pgid=10, host_boot_id="old-boot", process_start_ticks=1)
            clock.advance(31)
            with self.assertRaises(LeaseFencedError):
                controller.heartbeat(old.stage_id, old.lease_token)
            self.assertEqual(controller.recover_expired(manifest["batchId"], current_boot_id="new-boot"), 1)
            with self.assertRaises(LeaseFencedError):
                controller.heartbeat(old.stage_id, old.lease_token)
            recovered = store.row("SELECT state FROM stages WHERE id=?", (old.stage_id,))
            self.assertEqual(recovered["state"], "ready")

            # Fair scheduling may dispatch tasks that have never run before this
            # recovered task. Lease without assuming a particular queue order.
            new = None
            while new is None:
                item = controller.lease_next(manifest["batchId"], "worker-new", capacities)
                self.assertIsNotNone(item)
                if item.stage_id == old.stage_id:
                    new = item
                else:
                    controller.fail(item, "invalid_input", "advance fair queue for recovery test")
            self.assertEqual(new.attempt, 2)
            self.assertNotEqual(new.lease_token, old.lease_token)
        finally:
            store.close()

    def test_terminal_stage_elapsed_is_frozen_at_attempt_finish(self) -> None:
        clock = Clock()
        store, controller, manifest = create_controller(self.tmp_path, clock=clock)
        try:
            item = controller.lease_next(manifest["batchId"], "worker", default_capacities(manifest))
            self.assertIsNotNone(item)
            controller.mark_running(
                item,
                pid=os.getpid(),
                pgid=os.getpgrp(),
                host_boot_id="test-boot",
                process_start_ticks=1,
            )
            clock.advance(20)
            controller.fail(item, "invalid_input", "terminal fixture")
            clock.advance(500)

            task = controller.status(manifest["batchId"])["tasks"][0]
            self.assertEqual(task["stage_state"], "failed_terminal")
            self.assertEqual(task["stageElapsedSeconds"], 20.0)
            self.assertEqual(task["taskElapsedSeconds"], 20.0)
        finally:
            store.close()

    def test_rejected_task_skips_evaluation_only_after_all_tasks_reach_boundary(self) -> None:
        manifest = batch_manifest(mode="automatic", project_count=1)
        manifest["projects"][0]["tasks"] = [
            {**manifest["projects"][0]["tasks"][0], "taskId": f"Task_{index:03d}"}
            for index in range(1, 4)
        ]
        store = SchedulerStore(self.tmp_path / "incomplete-evaluation.sqlite3")
        controller = BatchController(store, clock=Clock(), lease_seconds=30)
        controller.plan(manifest, self.tmp_path / "incomplete-evaluation-runs", formal=False)
        controller.start(manifest["batchId"], formal=False)
        try:
            executor = FakeExecutor({("Scene_01", "Task_003", "estimate_pose", 1): "rejected"})
            worker = BatchWorker(
                controller,
                default_capacities(manifest),
                executor,
                worker_id="evaluation-boundary-test",
                stage_names=tuple(stage.name for stage in STAGES if stage.name != "evaluate_absolute"),
            )
            report = worker.run(manifest["batchId"], max_steps=1000)

            self.assertEqual(report.status["status"], "failed")
            evaluation_rows = store.rows(
                """SELECT task_id, state, last_error_code FROM stages
                   WHERE batch_id=? AND name='evaluate_absolute' ORDER BY task_id""",
                (manifest["batchId"],),
            )
            self.assertEqual(
                [(row["task_id"], row["state"], row["last_error_code"]) for row in evaluation_rows],
                [
                    ("Task_001", "canceled", "evaluation_skipped_incomplete_batch"),
                    ("Task_002", "canceled", "evaluation_skipped_incomplete_batch"),
                    ("Task_003", "canceled", "upstream_failed"),
                ],
            )
            tasks = store.rows(
                "SELECT task_id, status FROM tasks WHERE batch_id=? ORDER BY task_id",
                (manifest["batchId"],),
            )
            self.assertEqual(
                [(row["task_id"], row["status"]) for row in tasks],
                [("Task_001", "succeeded"), ("Task_002", "succeeded"), ("Task_003", "rejected")],
            )
            self.assertFalse(any(stage == "evaluate_absolute" for *_unused, stage, _attempt in executor.executed))
            self.assertEqual(
                store.row(
                    """SELECT COUNT(*) AS count FROM events
                       WHERE batch_id=? AND kind='evaluation_skipped_incomplete_batch'""",
                    (manifest["batchId"],),
                )["count"],
                1,
            )
        finally:
            store.close()

    def test_expired_remote_lease_stays_fenced_until_explicit_probe(self) -> None:
        clock = Clock()
        store, controller, manifest = create_controller(self.tmp_path, clock=clock, num_generations=1)
        try:
            capacities = default_capacities(manifest)
            executor = FakeExecutor()
            for stage_name in ("unity_anchor", "image_edit", "upload_inputs"):
                item = controller.lease_next(
                    manifest["batchId"],
                    "setup-worker",
                    capacities,
                    project_id="Scene_01",
                    task_id="Task_001",
                    stage_name=stage_name,
                )
                self.assertIsNotNone(item)
                result = executor.execute(controller, item)
                self.assertTrue(result.succeeded)
                controller.commit_success(item, result.artifacts)

            remote = controller.lease_next(
                manifest["batchId"],
                "remote-worker",
                capacities,
                project_id="Scene_01",
                task_id="Task_001",
                stage_name="model_generation",
            )
            self.assertIsNotNone(remote)
            controller.mark_running(
                remote,
                pid=10,
                pgid=10,
                host_boot_id="old-boot",
                process_start_ticks=1,
            )
            clock.advance(31)
            self.assertEqual(
                controller.recover_expired(manifest["batchId"], current_boot_id="new-boot"),
                1,
            )
            stage = store.row("SELECT state, last_error_code FROM stages WHERE id=?", (remote.stage_id,))
            lease = store.row("SELECT revoked_at FROM leases WHERE stage_id=?", (remote.stage_id,))
            self.assertEqual((stage["state"], stage["last_error_code"]), ("recovering", "delivery_unknown"))
            self.assertIsNotNone(lease)
            self.assertIsNotNone(lease["revoked_at"])
            self.assertIsNone(
                controller.lease_next(
                    manifest["batchId"],
                    "duplicate-remote-worker",
                    capacities,
                    project_id="Scene_01",
                    task_id="Task_001",
                    stage_name="model_generation",
                )
            )
        finally:
            store.close()

    def test_cancel_refuses_to_release_an_active_remote_attempt(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path, num_generations=1)
        try:
            capacities = default_capacities(manifest)
            executor = FakeExecutor()
            for stage_name in ("unity_anchor", "image_edit", "upload_inputs"):
                item = controller.lease_next(
                    manifest["batchId"],
                    "setup-worker",
                    capacities,
                    project_id="Scene_01",
                    task_id="Task_001",
                    stage_name=stage_name,
                )
                self.assertIsNotNone(item)
                result = executor.execute(controller, item)
                controller.commit_success(item, result.artifacts)
            remote = controller.lease_next(
                manifest["batchId"],
                "remote-worker",
                capacities,
                project_id="Scene_01",
                task_id="Task_001",
                stage_name="model_generation",
            )
            self.assertIsNotNone(remote)

            with self.assertRaisesRegex(StoreError, "recover-remote"):
                controller.cancel(
                    manifest["batchId"],
                    project_id="Scene_01",
                    task_id="Task_001",
                )
            stage = store.row("SELECT state FROM stages WHERE id=?", (remote.stage_id,))
            lease = store.row("SELECT token, revoked_at FROM leases WHERE stage_id=?", (remote.stage_id,))
            self.assertEqual(stage["state"], "leased")
            self.assertEqual(lease["token"], remote.lease_token)
            self.assertIsNone(lease["revoked_at"])
        finally:
            store.close()

    def test_retryable_and_delivery_unknown_have_distinct_states(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path, num_generations=1)
        try:
            project = "Scene_01"
            task = "Task_001"
            outcomes = {
                (project, task, "image_edit", 1): "429",
                (project, "Task_002", "image_edit", 1): "timeout",
            }
            runner = FakeRunner(controller, default_capacities(manifest), FakeExecutor(outcomes))
            runner.run_until_blocked(manifest["batchId"])
            attempts = store.rows(
                """SELECT s.task_id, s.name, a.attempt_number, a.error_code FROM attempts a
                   JOIN stages s ON s.id=a.stage_id WHERE s.project_id=? AND s.name='image_edit'
                   ORDER BY s.task_id, a.attempt_number""",
                (project,),
            )
            task1 = [row for row in attempts if row["task_id"] == task]
            task2 = [row for row in attempts if row["task_id"] == "Task_002"]
            self.assertEqual([(row["attempt_number"], row["error_code"]) for row in task1], [(1, "http_429"), (2, None)])
            self.assertEqual([(row["attempt_number"], row["error_code"]) for row in task2], [(1, "delivery_unknown")])
            stage2 = store.row(
                "SELECT state FROM stages WHERE project_id=? AND task_id='Task_002' AND name='image_edit'",
                (project,),
            )
            self.assertEqual(stage2["state"], "waiting_manual")
        finally:
            store.close()

    def test_cancel_and_journaled_gc_do_not_escape_batch_root(self) -> None:
        clock = Clock()
        store, controller, manifest = create_controller(self.tmp_path, clock=clock)
        try:
            item = controller.lease_next(manifest["batchId"], "worker", default_capacities(manifest))
            self.assertIsNotNone(item)
            item.staging_dir.joinpath("partial.bin").write_bytes(b"partial")
            controller.fail(item, "worker_crash", "fake crash")
            clock.advance(90000)
            targets = controller.gc(manifest["batchId"], owner_token="test-owner", dry_run=True)
            self.assertIn(str(item.staging_dir), targets)
            self.assertTrue(item.staging_dir.exists())
            deleted = controller.gc(manifest["batchId"], owner_token="test-owner", dry_run=False)
            self.assertEqual(deleted, targets)
            self.assertFalse(item.staging_dir.exists())
            canceled = controller.cancel(manifest["batchId"], project_id="Scene_02", task_id="Task_001")
            self.assertGreater(canceled, 0)
            self.assertFalse(
                store.rows(
                    """SELECT l.stage_id FROM leases l JOIN stages s ON s.id=l.stage_id
                       WHERE s.project_id='Scene_02' AND s.task_id='Task_001'"""
                )
            )
        finally:
            store.close()

    def test_failed_cleanup_keeps_revoked_lease_and_resource_blocked(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path)
        try:
            capacities = default_capacities(manifest)
            item = controller.lease_next(manifest["batchId"], "worker", capacities)
            self.assertIsNotNone(item)
            controller.fail(item, "worker_crash", "residual process", cleanup_completed=False)
            lease = store.row("SELECT revoked_at FROM leases WHERE stage_id=?", (item.stage_id,))
            self.assertIsNotNone(lease)
            self.assertIsNotNone(lease["revoked_at"])
            stage = store.row("SELECT state FROM stages WHERE id=?", (item.stage_id,))
            self.assertEqual(stage["state"], "recovering")
            self.assertIsNone(controller.lease_next(manifest["batchId"], "worker-2", capacities))
        finally:
            store.close()

    def test_command_executor_rejects_stage_result_with_wrong_lease(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path)
        try:
            item = controller.lease_next(manifest["batchId"], "worker", default_capacities(manifest))
            self.assertIsNotNone(item)
            result = {
                "schemaVersion": 1,
                "kind": "insertany3d.stage-result",
                "batchId": item.batch_id,
                "projectId": item.project_id,
                "taskId": item.task_id,
                "stage": item.stage,
                "contractVersion": item.contract_version,
                "attempt": item.attempt,
                "leaseToken": "wrong-token",
                "status": "failed_terminal",
                "artifacts": [],
                "errorCode": "invalid_input",
                "message": "fake",
                "diagnosticPaths": [],
                "cleanup": {"completed": True},
                "finishedAtUtc": "2026-08-29T12:00:00Z",
            }
            script = "from pathlib import Path; Path('output.staging/stage_result.json').write_text(" + repr(json.dumps(result)) + ")"
            outcome = CommandExecutor(heartbeat_seconds=0.01).execute(controller, item, [sys.executable, "-c", script])
            self.assertFalse(outcome.succeeded)
            self.assertEqual(outcome.error_code, "compile_or_contract")
            self.assertIn("leaseToken", outcome.message)
        finally:
            store.close()

    def test_stage_request_wiring_uses_batch_root_for_unity_result(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path)
        try:
            item = controller.lease_next(
                manifest["batchId"], "cli", default_capacities(manifest),
                project_id="Scene_01", task_id="Task_001", stage_name="unity_anchor",
            )
            self.assertIsNotNone(item)
            request, request_path = controller.write_stage_request(item)
            root = (self.tmp_path / "runs").resolve()
            self.assertEqual(request["outputStagingDir"], item.staging_dir.resolve().relative_to(root).as_posix())
            self.assertEqual(request["inputs"], [])
            command = controller.build_stage_command(item, request_path, unity_executable="Unity.exe")
            self.assertIn("-insertAny3DArtifactRoot", command)
            self.assertEqual(command[command.index("-insertAny3DArtifactRoot") + 1], str(root))
            self.assertEqual(command[command.index("-insertAny3DResult") + 1], str(item.staging_dir / "stage_result.json"))
            self.assertEqual(command[command.index("-projectPath") + 1], manifest["projects"][0]["projectPath"])
        finally:
            store.close()

    def test_unity_apply_uses_canonical_workspace_pose_and_ply(self) -> None:
        root = self.tmp_path / "runs"
        task_root = root / "Scene_01" / "Task_001"
        pose = task_root / "stages" / "pose" / "output" / "pose.json"
        ply = task_root / "stages" / "sags" / "output" / "inserted_object.ply"
        pose.parent.mkdir(parents=True)
        ply.parent.mkdir(parents=True)
        pose.write_text('{"status":"ready","position":{"x":0}}', encoding="utf-8")
        ply.write_bytes(b"ply\nformat ascii 1.0\n")
        (task_root / "task_manifest.json").write_text(
            '{"projectId":"Scene_01","taskId":"Task_001"}', encoding="utf-8"
        )
        (task_root / "stages" / "pose" / "manifest.json").write_text(
            '{"status":"succeeded"}', encoding="utf-8"
        )
        (task_root / "stages" / "sags" / "manifest.json").write_text(
            '{"status":"succeeded"}', encoding="utf-8"
        )
        item = SimpleNamespace(project_id="Scene_01", task_id="Task_001")
        records = {
            "pose": {
                "sourceStage": "estimate_pose",
                "relative": "stages/pose/output/pose.json",
                "path": str(pose.relative_to(root)),
            },
            "ply": {
                "sourceStage": "sags_segment_vote",
                "relative": "stages/sags/output/inserted_object.ply",
                "path": str(ply.relative_to(root)),
            },
        }
        result_root = _workspace_result_root(root, item, records)
        self.assertEqual(result_root, "Scene_01")
        self.assertFalse((root / "unity_inputs").exists())

    def test_unity_apply_materializes_committed_hashed_outputs(self) -> None:
        root = self.tmp_path / "runs"
        task_root = root / "Scene_01" / "Task_001"
        canonical_pose = task_root / "stages" / "pose" / "output" / "pose.json"
        canonical_ply = task_root / "stages" / "sags" / "output" / "inserted_object.ply"
        canonical_pose.parent.mkdir(parents=True)
        canonical_ply.parent.mkdir(parents=True)
        canonical_pose.write_text('{"status":"ready"}', encoding="utf-8")
        canonical_ply.write_bytes(b"ply\n")
        (task_root / "task_manifest.json").write_text(
            '{"projectId":"Scene_01","taskId":"Task_001"}', encoding="utf-8"
        )
        (task_root / "stages" / "pose" / "manifest.json").write_text(
            '{"status":"succeeded"}', encoding="utf-8"
        )
        (task_root / "stages" / "sags" / "manifest.json").write_text(
            '{"status":"succeeded"}', encoding="utf-8"
        )
        records = {
            "manifest": {"sourceStage": "unity_anchor", "relative": "Task_001/task_manifest.json", "path": "legacy/task_manifest.json", "sha256": hashlib.sha256(b'{"projectId":"Scene_01","taskId":"Task_001"}').hexdigest()},
            "pose": {"sourceStage": "estimate_pose", "relative": "pose.json", "path": "legacy/pose.json", "sha256": hashlib.sha256(b'{"status":"ready"}').hexdigest()},
            "ply": {
                "sourceStage": "sags_segment_vote",
                "relative": "inserted_object.ply",
                "path": "legacy/inserted_object.ply",
                "sha256": hashlib.sha256(b"ply\n").hexdigest(),
            },
        }
        (root / "legacy").mkdir()
        (root / "legacy/task_manifest.json").write_text('{"projectId":"Scene_01","taskId":"Task_001"}', encoding="utf-8")
        (root / "legacy/pose.json").write_text('{}', encoding="utf-8")
        (root / "legacy/pose.json").write_text('{"status":"ready"}', encoding="utf-8")
        (root / "legacy/inserted_object.ply").write_bytes(b"ply\n")
        result_root = _workspace_result_root(root, SimpleNamespace(project_id="Scene_01", task_id="Task_001"), records)
        self.assertEqual(result_root, "Scene_01")
        self.assertTrue((task_root / "task_manifest.json").is_file())
        self.assertTrue((task_root / "stages/pose/output/pose.json").is_file())
        self.assertTrue((task_root / "stages/sags/output/inserted_object.ply").is_file())

    def test_windows_unity_executable_is_runnable_from_wsl(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path)
        try:
            item = controller.lease_next(
                manifest["batchId"], "cli", default_capacities(manifest),
                project_id="Scene_01", task_id="Task_001", stage_name="unity_anchor",
            )
            self.assertIsNotNone(item)
            _, request_path = controller.write_stage_request(item)
            command = controller.build_stage_command(
                item,
                request_path,
                unity_executable=r"F:\Programs\Unity\Editor\Unity.exe",
            )
            self.assertEqual(command[0], "/mnt/f/Programs/Unity/Editor/Unity.exe")
            project_argument = command[command.index("-projectPath") + 1]
            self.assertTrue(project_argument.startswith(r"\\"))
            self.assertIn(r"private\Scene_01", project_argument)
        finally:
            store.close()

    def test_stage_request_maps_committed_edit_image_to_remote_input(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path, num_generations=1)
        try:
            capacities = default_capacities(manifest)
            anchor = controller.lease_next(
                manifest["batchId"], "worker", capacities,
                project_id="Scene_01", task_id="Task_001", stage_name="unity_anchor",
            )
            self.assertIsNotNone(anchor)
            result = FakeExecutor().execute(controller, anchor)
            controller.commit_success(anchor, result.artifacts)
            controller.refresh(manifest["batchId"])
            edit = controller.lease_next(
                manifest["batchId"], "worker", capacities,
                project_id="Scene_01", task_id="Task_001", stage_name="image_edit",
            )
            self.assertIsNotNone(edit)
            result = FakeExecutor().execute(controller, edit)
            controller.commit_success(edit, result.artifacts)
            controller.refresh(manifest["batchId"])
            upload = controller.lease_next(
                manifest["batchId"], "worker", capacities,
                project_id="Scene_01", task_id="Task_001", stage_name="upload_inputs",
            )
            self.assertIsNotNone(upload)
            result = FakeExecutor().execute(controller, upload)
            controller.commit_success(upload, result.artifacts)
            controller.refresh(manifest["batchId"])
            model = controller.lease_next(
                manifest["batchId"], "worker", capacities,
                project_id="Scene_01", task_id="Task_001", stage_name="model_generation",
            )
            self.assertIsNotNone(model)
            request = controller.build_stage_request(model)
            self.assertEqual(request["effectiveConfig"]["stageOptions"]["gpuDevice"], model.resources["remote_gpu"][4:])
            self.assertIn("input_image", {item["artifactId"] for item in request["inputs"]})
            self.assertTrue(any(item["path"].endswith("edited.png") for item in request["inputs"]))
            command = controller.build_stage_command(model, self.tmp_path / "request.json", python_executable="python3")
            self.assertEqual(command[0], "python3")
            self.assertIn("tools/stage_adapter.py", "/".join(command[1:2]))
        finally:
            store.close()

    def test_multi_candidate_review_selection_filters_downstream_input(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path, mode="manual")
        try:
            capacities = default_capacities(manifest)
            anchor = controller.lease_next(
                manifest["batchId"], "anchor", capacities,
                project_id="Scene_01", task_id="Task_001", stage_name="unity_anchor",
            )
            self.assertIsNotNone(anchor)
            center = anchor.staging_dir / "step1" / "center" / "image.png"
            center.parent.mkdir(parents=True)
            center.write_bytes(b"center")
            controller.commit_success(anchor, [{"artifactId": "center", "type": "scene_rgb", "path": "step1/center/image.png"}])
            controller.refresh(manifest["batchId"])

            for expected_index in (1, 2, 3):
                edit = controller.lease_next(
                    manifest["batchId"], f"edit-{expected_index}", capacities,
                    project_id="Scene_01", task_id="Task_001", stage_name="image_edit",
                )
                self.assertIsNotNone(edit)
                self.assertEqual(edit.generation_index, expected_index)
                request = controller.build_stage_request(edit)
                self.assertEqual(request["effectiveConfig"]["task"]["generationIndex"], expected_index)
                output = edit.staging_dir / f"edited-{expected_index}.png"
                output.write_bytes(f"candidate-{expected_index}".encode())
                metadata = edit.staging_dir / "image_edit.json"
                metadata.write_text(json.dumps({
                    "schemaVersion": 1,
                    "status": "ready",
                    "generationIndex": expected_index,
                    "generationCount": 3,
                    "output": {"path": output.name},
                }), encoding="utf-8")
                controller.commit_success(edit, [
                    {"artifactId": "edited_image", "type": "edited_image", "path": output.name},
                    {"artifactId": "image_edit_manifest", "type": "image_edit_manifest", "path": metadata.name},
                ])
                controller.refresh(manifest["batchId"])

            review = controller.review_page(manifest["batchId"])[0]
            self.assertEqual([candidate["index"] for candidate in review["reviewManifest"]["candidates"]], [1, 2, 3])
            controller.decide_edit(
                manifest["batchId"], "Scene_01", "Task_001", review["edit_attempt"], "2", decided_by="tester",
            )
            controller.refresh(manifest["batchId"])
            upload = controller.lease_next(
                manifest["batchId"], "upload", capacities,
                project_id="Scene_01", task_id="Task_001", stage_name="upload_inputs",
            )
            self.assertIsNotNone(upload)
            upload_output = upload.staging_dir / "receipt.json"
            upload_output.write_text("{}", encoding="utf-8")
            controller.commit_success(upload, [{"artifactId": "receipt", "type": "receipt", "path": upload_output.name}])
            controller.refresh(manifest["batchId"])
            model = controller.lease_next(
                manifest["batchId"], "model", capacities,
                project_id="Scene_01", task_id="Task_001", stage_name="model_generation",
            )
            self.assertIsNotNone(model)
            request = controller.build_stage_request(model)
            image_inputs = [item for item in request["inputs"] if item["type"] == "edited_image"]
            self.assertEqual(len(image_inputs), 1)
            self.assertIn("generation-002", image_inputs[0]["path"])
            self.assertNotIn("generation-001", image_inputs[0]["path"])
            self.assertNotIn("generation-003", image_inputs[0]["path"])
            self.assertEqual(request["effectiveConfig"]["acceptedGenerationPath"], image_inputs[0]["path"] if Path(image_inputs[0]["path"]).is_absolute() else str((self.tmp_path / "runs" / image_inputs[0]["path"]).resolve()))
        finally:
            store.close()

    def test_command_executor_preserves_valid_failure_from_nonzero_worker(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path)
        try:
            item = controller.lease_next(manifest["batchId"], "worker", default_capacities(manifest))
            self.assertIsNotNone(item)
            result = {
                "schemaVersion": 1,
                "kind": "insertany3d.stage-result",
                "batchId": item.batch_id,
                "projectId": item.project_id,
                "taskId": item.task_id,
                "stage": item.stage,
                "contractVersion": item.contract_version,
                "attempt": item.attempt,
                "leaseToken": item.lease_token,
                "status": "failed_terminal",
                "artifacts": [],
                "errorCode": "invalid_input",
                "message": "fixture rejected",
                "diagnosticPaths": [],
                "cleanup": {"completed": True},
                "finishedAtUtc": "2026-08-29T12:00:00Z",
            }
            script = (
                "import json,sys; from pathlib import Path; "
                "Path('output.staging/stage_result.json').write_text(json.dumps(" + repr(result) + ")); "
                "sys.exit(7)"
            )
            outcome = CommandExecutor(heartbeat_seconds=0.01).execute(
                controller, item, [sys.executable, "-c", script]
            )
            self.assertFalse(outcome.succeeded)
            self.assertEqual(outcome.stage_status, "failed_terminal")
            self.assertEqual(outcome.error_code, "invalid_input")
            self.assertEqual(outcome.message, "fixture rejected")
            state = controller.fail(
                item,
                outcome.error_code,
                outcome.message,
                cleanup_completed=outcome.cleanup_completed,
                stage_status=outcome.stage_status,
            )
            self.assertEqual(state, "failed_terminal")
        finally:
            store.close()

    def test_manifest_and_db_survive_controller_reopen(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path)
        store.close()
        with SchedulerStore(self.tmp_path / "state.sqlite3") as reopened:
            resumed = BatchController(reopened, clock=Clock())
            self.assertEqual(resumed.resume(manifest["batchId"]), 0)
            status = resumed.status(manifest["batchId"])
            self.assertEqual(len(status["tasks"]), 60)
            self.assertEqual(status["status"], "running")

    def test_artifact_commit_reconciles_crash_after_atomic_rename(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path)
        item = controller.lease_next(manifest["batchId"], "worker", default_capacities(manifest))
        self.assertIsNotNone(item)
        result = FakeExecutor().execute(controller, item)
        controller.prepare_artifact_commit(item, result.artifacts)
        item.output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(item.staging_dir, item.output_dir)
        store.close()

        with SchedulerStore(self.tmp_path / "state.sqlite3") as reopened:
            resumed = BatchController(reopened, clock=Clock())
            resumed.resume(manifest["batchId"])
            stage = reopened.row("SELECT state FROM stages WHERE id=?", (item.stage_id,))
            commit = reopened.row("SELECT state FROM artifact_commits WHERE attempt_id=?", (item.attempt_id,))
            self.assertEqual(stage["state"], "succeeded")
            self.assertEqual(commit["state"], "committed")
            self.assertTrue(item.output_dir.joinpath("result.json").is_file())

    def test_regenerate_creates_new_edit_identity_and_approval(self) -> None:
        store, controller, manifest = create_controller(self.tmp_path, mode="manual")
        try:
            runner = FakeRunner(controller, default_capacities(manifest))
            runner.run_until_blocked(manifest["batchId"])
            first = controller.review_page(manifest["batchId"])[0]
            first_attempt = first["edit_attempt"]
            image_before = store.row(
                "SELECT * FROM stages WHERE batch_id=? AND project_id=? AND task_id=? AND name='image_edit'",
                (manifest["batchId"], first["project_id"], first["task_id"]),
            )
            controller.decide_edit(
                manifest["batchId"], first["project_id"], first["task_id"], first_attempt,
                "regenerate", decided_by="tester",
            )
            image_after = store.row("SELECT * FROM stages WHERE id=?", (image_before["id"],))
            self.assertNotEqual(image_before["idempotency_key"], image_after["idempotency_key"])
            self.assertEqual(json.loads(image_after["effective_config_json"])["editGeneration"], 2)
            runner.run_until_blocked(manifest["batchId"])
            second = store.row(
                """SELECT * FROM edit_reviews WHERE batch_id=? AND project_id=? AND task_id=?
                   AND status='pending_review' ORDER BY edit_attempt DESC LIMIT 1""",
                (manifest["batchId"], first["project_id"], first["task_id"]),
            )
            self.assertEqual(second["status"], "pending_review")
            controller.decide_edit(
                manifest["batchId"], first["project_id"], first["task_id"], int(second["edit_attempt"]),
                "accepted", decided_by="tester",
            )
            gate = store.row(
                "SELECT effective_config_json FROM stages WHERE batch_id=? AND project_id=? AND task_id=? AND name='edit_gate'",
                (manifest["batchId"], first["project_id"], first["task_id"]),
            )
            self.assertEqual(json.loads(gate["effective_config_json"])["approvedEditAttempt"], second["edit_attempt"])
            commits = store.rows("SELECT output_dir FROM artifact_commits WHERE stage_id=?", (image_before["id"],))
            self.assertEqual(len(commits), 6)
            self.assertEqual(len({row["output_dir"] for row in commits}), 6)
        finally:
            store.close()

    def test_batch_worker_pauses_for_review_and_resumes_one_task(self) -> None:
        """The CLI worker must stop at manual review, then resume independently."""
        manifest = batch_manifest(mode="manual", project_count=1)
        store = SchedulerStore(self.tmp_path / "worker.sqlite3")
        controller = BatchController(store, clock=Clock(), lease_seconds=30)
        controller.plan(manifest, self.tmp_path / "worker-runs", formal=False)
        controller.start(manifest["batchId"], formal=False)
        try:
            worker = BatchWorker(controller, default_capacities(manifest), worker_id="test-worker")
            first = worker.run(manifest["batchId"])
            self.assertEqual(first.executor, "fake")
            self.assertGreater(first.processed_stages, 0)
            self.assertEqual(first.blocked_reason, "waiting_manual_review")
            page = controller.review_page(manifest["batchId"])
            self.assertEqual(len(page), 5)

            accepted = page[0]
            controller.decide_edit(
                manifest["batchId"],
                accepted["project_id"],
                accepted["task_id"],
                accepted["edit_attempt"],
                "accepted",
                decided_by="test",
            )
            second = worker.run(manifest["batchId"])
            task = store.row(
                "SELECT status FROM tasks WHERE batch_id=? AND project_id=? AND task_id=?",
                (manifest["batchId"], accepted["project_id"], accepted["task_id"]),
            )
            self.assertEqual(task["status"], "succeeded")
            self.assertGreater(second.succeeded_stages, 0)
            self.assertEqual(second.status["status"], "running")
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
