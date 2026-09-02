from __future__ import annotations

import hashlib
import json
import io
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from insertany3d.cli import (
    _close_review_preview,
    _is_review_image_path,
    _open_review_preview,
    _review_preview_path,
    _review_display_path,
    _format_status_table,
    _format_run_summary,
    _watch_batch_status,
    _run_worker_with_monitor,
    _run_interactive_reviews,
    _emit_stage_completion_logs,
    _git_snapshot,
    _remote_git_snapshot,
    _record_run_provenance,
    build_parser,
    main,
)
from insertany3d.evaluation import fixed_fake_response
from insertany3d.executors import FakeRunner
from insertany3d.remote_worker import CommandOutcome, RemoteProfile
from insertany3d.scheduler import BatchController, default_capacities
from insertany3d.store import SchedulerStore
from tests.fixtures import batch_manifest
from tests.test_evaluation import _write_manifest


class CliTests(unittest.TestCase):
    def test_run_all_emits_concise_stage_completion_logs_from_new_events(self) -> None:
        class EventStore:
            def rows(self, _query, parameters=()):
                after = int(parameters[1])
                events = [
                    {
                        "id": 1,
                        "stage_id": 10,
                        "kind": "stage_heartbeat",
                        "payload_json": '{"attempt": 1}',
                        "created_at": 100.0,
                        "project_id": "Farm_Test_001",
                        "task_id": "Task_001",
                        "name": "unity_anchor",
                    },
                    {
                        "id": 2,
                        "stage_id": 10,
                        "kind": "stage_succeeded",
                        "payload_json": '{"attempt": 1, "artifactCount": 1}',
                        "created_at": 102.0,
                        "project_id": "Farm_Test_001",
                        "task_id": "Task_001",
                        "name": "unity_anchor",
                    },
                    {
                        "id": 3,
                        "stage_id": 11,
                        "kind": "stage_failed",
                        "payload_json": '{"attempt": 2, "errorCode": "http_429", "nextState": "ready"}',
                        "created_at": 104.0,
                        "project_id": "Farm_Test_001",
                        "task_id": "Task_001",
                        "name": "image_edit",
                    },
                ]
                return [event for event in events if event["id"] > after]

            def row(self, query, parameters=()):
                if "MAX(id)" in query:
                    return {"event_id": 0}
                stage_id, attempt = parameters
                if (stage_id, attempt) == (10, 1):
                    return {"started_at": 100.0, "finished_at": 102.0}
                return {"started_at": 103.0, "finished_at": 104.0}

        output = io.StringIO()
        cursor = _emit_stage_completion_logs(EventStore(), "farm", 0, stderr=output)
        self.assertEqual(cursor, 3)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("| INFO | stage.completed |", lines[0])
        self.assertIn("project=Farm_Test_001 task=Task_001 stage=unity_anchor", lines[0])
        self.assertIn("status=succeeded attempt=1 duration=2.0s", lines[0])
        self.assertIn("| WARNING | stage.completed |", lines[1])
        self.assertIn("stage=image_edit status=ready attempt=2 duration=1.0s error=http_429", lines[1])

    def test_git_snapshot_records_clean_or_dirty_status_without_raising(self) -> None:
        class Runner:
            def __init__(self, status: str):
                self.status = status

            def __call__(self, command, **kwargs):
                del kwargs
                if command[1] == "rev-parse":
                    return type("Result", (), {"returncode": 0, "stdout": "abc123\n", "stderr": ""})()
                return type("Result", (), {"returncode": 0, "stdout": self.status, "stderr": ""})()

        with tempfile.TemporaryDirectory(prefix="insertany3d_provenance_git_") as directory:
            clean = _git_snapshot(directory, runner=Runner(""))
            self.assertEqual(clean["head"], "abc123")
            self.assertEqual(clean["status"], "clean")
            dirty = _git_snapshot(directory, runner=Runner(" M file.py\n?? output.bin\n"))
            self.assertEqual(dirty["status"], "dirty")
            self.assertEqual(dirty["statusOutput"], [" M file.py", "?? output.bin"])

    def test_remote_git_snapshot_parses_status_and_redacts_local_secrets(self) -> None:
        profile = RemoteProfile(
            target="worker@example.test",
            project_root="/srv/insertany3d",
            artifact_root="/srv/insertany3d-runs",
            connect_timeout_seconds=3,
        )
        observed = {}

        def runner(command, **kwargs):
            observed["command"] = command
            observed["env"] = kwargs["env"]
            return CommandOutcome(0, "HEAD\tdeadbeef\nSTATUS_BEGIN\n M tools/run.py\nSTATUS_CODE\t0\nSTATUS_END\n")

        with patch.dict(os.environ, {"APIYI_API_KEY": "secret"}, clear=False):
            snapshot = _remote_git_snapshot(profile, runner=runner)
        self.assertEqual(snapshot["head"], "deadbeef")
        self.assertEqual(snapshot["status"], "dirty")
        self.assertEqual(snapshot["statusOutput"], [" M tools/run.py"])
        self.assertNotIn("APIYI_API_KEY", observed["env"])
        self.assertIn("git status", observed["command"][-1])

    def test_run_provenance_is_atomic_append_and_unavailable_server_is_non_blocking(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_provenance_manifest_") as directory:
            root = Path(directory)
            existing = {"batchId": "old", "custom": {"keep": True}}
            (root / "run_manifest.json").write_text(json.dumps(existing), encoding="utf-8")
            local = {"repository": "/local", "head": "abc", "status": "dirty", "statusOutput": [" M x"], "error": None}
            with patch("insertany3d.cli._git_snapshot", return_value=local):
                first = _record_run_provenance("batch_test", root)
                second = _record_run_provenance("batch_test", root)
            self.assertEqual(first["local"]["head"], "abc")
            self.assertEqual(second["server"]["status"], "unavailable")
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["batchId"], "batch_test")
            self.assertEqual(manifest["custom"], {"keep": True})
            self.assertEqual(len(manifest["provenanceHistory"]), 2)
            self.assertEqual(manifest["provenance"]["local"]["status"], "dirty")

    def test_worker_and_run_all_default_to_serial_execution(self) -> None:
        worker_args = build_parser().parse_args(["batch", "worker", "batch_test", "--fake"])
        run_all_args = build_parser().parse_args(["batch", "run-all", "batch_test", "--fake"])
        self.assertEqual(worker_args.max_parallel, 1)
        self.assertEqual(run_all_args.max_parallel, 1)
        self.assertFalse(run_all_args.json)
        self.assertFalse(run_all_args.no_monitor)
        self.assertFalse(run_all_args.no_open_review_images)
        self.assertEqual(run_all_args.monitor_interval, 2.0)

    def test_run_all_monitor_can_be_disabled_and_preview_can_be_disabled(self) -> None:
        args = build_parser().parse_args([
            "batch", "run-all", "batch_test", "--fake", "--no-monitor",
            "--monitor-interval", "0.5", "--no-open-review-images",
        ])
        self.assertTrue(args.no_monitor)
        self.assertTrue(args.no_open_review_images)
        self.assertEqual(args.monitor_interval, 0.5)

    def test_main_reports_recoverable_worker_interrupt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_interrupt_") as directory:
            error = io.StringIO()
            with patch("insertany3d.cli._run", side_effect=KeyboardInterrupt), redirect_stderr(error):
                code = main([
                    "--db", str(Path(directory) / "state.sqlite3"),
                    "batch", "worker", "batch_test", "--fake",
                ])
        self.assertEqual(code, 130)
        self.assertIn("活动 lease 未被强制释放", error.getvalue())
        self.assertIn("recover-remote", error.getvalue())

    def test_run_all_worker_monitor_reads_status_while_worker_runs(self) -> None:
        class Controller:
            batch_id = None

            def status(self, batch_id):
                self.batch_id = batch_id
                return {
                    "batchId": batch_id,
                    "status": "running",
                    "stageCounts": {"running": 1},
                    "tasks": [],
                }

        controller = Controller()
        args = type(
            "Args",
            (),
            {"batch_id": "batch_monitor", "monitor_interval": 0.01, "no_monitor": False},
        )()
        output = io.StringIO()
        with patch("insertany3d.cli._run", return_value={"status": "running"}), patch(
            "insertany3d.cli.sys.stderr", output
        ):
            result = _run_worker_with_monitor(controller, None, args)  # type: ignore[arg-type]
        self.assertEqual(result["status"], "running")
        self.assertEqual(controller.batch_id, "batch_monitor")
        self.assertIn("InsertAny3D", output.getvalue())

    def test_interactive_review_opens_one_contact_sheet_and_selects_candidate(self) -> None:
        class Store:
            def row(self, query, parameters):
                return {"root_path": "/tmp/insertany3d-review"}

        class Controller:
            decisions = []

            def decide_edit(self, *values, **kwargs):
                self.decisions.append((values, kwargs))

        args = type(
            "Args",
            (),
            {
                "batch_id": "batch_review",
                "no_open_review_images": False,
            },
        )()
        reviews = [{
            "project_id": "Scene_01",
            "task_id": "Task_001",
            "edit_attempt": 1,
            "editArtifacts": [],
            "reviewManifest": {"candidates": [
                {"index": 1, "path": "/tmp/insertany3d-review/Scene_01/Task_001/edited-a.png"},
                {"index": 2, "path": "/tmp/insertany3d-review/Scene_01/Task_001/edited-b.png"},
                {"index": 3, "path": "/tmp/insertany3d-review/Scene_01/Task_001/edited-c.png"},
            ]},
        }]
        opened = []
        with patch("insertany3d.cli._build_review_contact_sheet", return_value=Path("/tmp/insertany3d-review/contact-sheet.png")), patch(
            "insertany3d.cli._open_review_preview", side_effect=lambda path, **_: (opened.append(path) or (None, None))), patch(
            "insertany3d.cli.input", return_value="2"
        ):
            controller = Controller()
            _run_interactive_reviews(controller, Store(), args, reviews)
        self.assertEqual(opened, [Path("/tmp/insertany3d-review/contact-sheet.png")])
        self.assertEqual(controller.decisions[0][0][-1], "2")
        self.assertEqual(controller.decisions[0][1]["decided_by"], "manual")

    def test_run_summary_shows_result_paths_and_failure_reason(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_summary_") as directory:
            root = Path(directory)
            (root / "Scene_01" / "Task_001" / "artifacts" / "unity_eval6").mkdir(parents=True)
            (root / "Scene_01" / "Task_003" / "artifacts" / "estimate_pose").mkdir(parents=True)
            rendered = _format_run_summary({
                "status": "failed",
                "runRoot": str(root),
                "evaluation": "skipped",
                "statusSnapshot": {
                    "batchId": "batch_summary",
                    "status": "failed",
                    "tasks": [
                        {"project_id": "Scene_01", "task_id": "Task_001", "status": "succeeded", "taskElapsedSeconds": 12},
                        {"project_id": "Scene_01", "task_id": "Task_003", "status": "rejected", "current_stage": "estimate_pose", "stage_state": "rejected", "taskElapsedSeconds": 7, "last_error_code": "pose_quality_rejected", "last_message": "位姿未通过多视角质量门禁"},
                    ],
                },
            })
            self.assertIn("完成 1/2", rendered)
            self.assertIn("[完成] Scene_01/Task_001", rendered)
            self.assertIn("unity_eval6", rendered)
            self.assertIn("[未完成] Scene_01/Task_003", rendered)
            self.assertIn("卡点: estimate_pose（rejected）", rendered)
            self.assertIn("pose_quality_rejected", rendered)
            self.assertIn("运行目录:", rendered)

    def test_run_all_defaults_to_human_summary_and_json_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_summary_main_") as directory:
            output = io.StringIO()
            with patch("insertany3d.cli._run_all", return_value={
                "status": "failed",
                "runRoot": directory,
                "statusSnapshot": {"batchId": "batch_summary", "status": "failed", "tasks": []},
            }), redirect_stdout(output):
                self.assertEqual(main(["--db", str(Path(directory) / "state.sqlite3"), "batch", "run-all", "batch_summary", "--fake"]), 0)
            self.assertIn("=== InsertAny3D 运行摘要 ===", output.getvalue())
            self.assertNotIn('"status"', output.getvalue())

    def test_review_display_path_shortens_only_image_edit_uid(self) -> None:
        path = "/test/InsertRuns/run/Farm_Test_001/Task_001/artifacts/image_edit/0123456789abcdef/edited.png"
        shown = _review_display_path(path)
        self.assertIn("image_edit/0123456789/edited.png", shown)
        self.assertNotIn("0123456789abcdef", shown)

    def test_review_preview_path_resolves_relative_artifact_against_batch_root(self) -> None:
        resolved = _review_preview_path(
            "Farm_Test_001/Task_003/artifacts/image_edit/0123456789abcdef/edited.png",
            "/test/InsertRuns/farm-test-001-phase2",
        )
        self.assertEqual(
            str(resolved),
            "/test/InsertRuns/farm-test-001-phase2/Farm_Test_001/Task_003/artifacts/image_edit/0123456789abcdef/edited.png",
        )

    def test_review_preview_accepts_images_and_rejects_json(self) -> None:
        self.assertTrue(_is_review_image_path("attempt/edited.png"))
        self.assertTrue(_is_review_image_path("attempt/EDITED.JPEG"))
        self.assertFalse(_is_review_image_path("attempt/image_edit.json"))

    def disabled_legacy_wsl_preview_uses_wslpath_and_closes_started_process(self) -> None:
        class FakeProcess:
            def __init__(self):
                self.pid = 321
                self.terminated = False
            def poll(self): return None
            def terminate(self): self.terminated = True
            def wait(self, timeout=None): return 0

        calls = []
        fake = FakeProcess()
        def run(command, **kwargs):
            calls.append(command)
            return type("Result", (), {"stdout": "C:\\tmp\\edited.png\n"})()
        with patch("insertany3d.cli._is_wsl", return_value=True), patch("insertany3d.cli.shutil.which", return_value="tool"), patch("insertany3d.cli.subprocess.run", side_effect=run), patch("insertany3d.cli.subprocess.Popen", return_value=fake):
            process, before = _open_review_preview("/mnt/f/tmp/edited.png", stderr=io.StringIO())
            _close_review_preview(process, before, stderr=io.StringIO())
        self.assertIs(process, fake)
        self.assertTrue(fake.terminated)
        self.assertEqual(calls[0], ["wslpath", "-w", "/mnt/f/tmp/edited.png"])

    def test_run_all_fake_auto_review_runs_evaluation_after_eval6(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_run_all_eval_") as directory:
            root = Path(directory)
            manifest = batch_manifest(mode="automatic", project_count=1)
            manifest["projects"][0]["tasks"] = [
                {**manifest["projects"][0]["tasks"][0], "taskId": f"Task_{index:03d}"}
                for index in range(1, 4)
            ]
            manifest_path = root / "batch.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main([
                        "--db", str(root / "state.sqlite3"), "batch", "run-all", "batch_test",
                        "--manifest", str(manifest_path), "--root", str(root / "runs"),
                        "--fake", "--fake-score", "8", "--max-steps", "1000",
                        "--json",
                        "--expected-tasks", "3", "--expected-scenes", "1", "--tasks-per-scene", "3",
                    ]),
                    0,
                )
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["evaluation"]["status"], "ready")
            self.assertTrue(Path(result["evaluation"]["outputs"]["xlsx"]).is_file())

    def test_run_all_fake_three_tasks_stops_before_evaluation_worker_stage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_run_all_") as directory:
            root = Path(directory)
            manifest = batch_manifest(mode="manual", project_count=1)
            manifest["projects"][0]["tasks"] = [
                {**manifest["projects"][0]["tasks"][0], "taskId": f"Task_{index:03d}"}
                for index in range(1, 4)
            ]
            manifest_path = root / "batch.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output, error = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                self.assertEqual(
                    main([
                        "--db", str(root / "state.sqlite3"), "batch", "run-all", "batch_test",
                        "--manifest", str(manifest_path), "--root", str(root / "runs"),
                        "--fake", "--non-interactive", "--max-steps", "100",
                        "--json",
                    ]),
                    0,
                )
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "waiting_manual_review")
            self.assertNotIn("no_supported_stage", error.getvalue())

    def disabled_legacy_run_all_prioritizes_regenerated_review_before_ready_work(self) -> None:
        """A newly regenerated edit must be surfaced before leasing other work."""
        with tempfile.TemporaryDirectory(prefix="insertany3d_run_all_review_priority_") as directory:
            root = Path(directory)
            manifest = batch_manifest(mode="manual", project_count=1)
            manifest["projects"][0]["tasks"] = manifest["projects"][0]["tasks"][:2]
            database = root / "state.sqlite3"
            batch_root = root / "runs"
            store = SchedulerStore(database)
            controller = BatchController(store)
            controller.plan(manifest, batch_root, formal=False)
            controller.start(manifest["batchId"], formal=False)
            try:
                runner = FakeRunner(controller, default_capacities(manifest))
                runner.run_until_blocked(manifest["batchId"])
                initial = controller.review_page(manifest["batchId"], page=1, size=None)
                self.assertEqual(len(initial), 2)
                for item in initial:
                    controller.decide_edit(
                        manifest["batchId"], item["project_id"], item["task_id"],
                        item["edit_attempt"],
                        "regenerate" if item["task_id"] == "Task_001" else "accepted",
                        decided_by="cli-test",
                    )
                runner.run_until_blocked(manifest["batchId"])
                reviews = controller.review_page(manifest["batchId"], page=1, size=None)
                self.assertTrue(any(item["task_id"] == "Task_001" and item["edit_attempt"] == 2 for item in reviews))

                args = build_parser().parse_args([
                    "--db", str(database), "batch", "run-all", manifest["batchId"],
                    "--fake", "--non-interactive",
                ])
                with patch("insertany3d.cli._run", side_effect=AssertionError("worker must not run before review")):
                    result = __import__("insertany3d.cli", fromlist=["_run_all"])._run_all(
                        controller, store, args,
                    )
                self.assertEqual(result["status"], "waiting_manual_review")
                self.assertTrue(any(item["edit_attempt"] == 2 for item in result["reviews"]))
            finally:
                store.close()

    def test_run_all_rejects_reusing_batch_id_with_different_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_run_all_identity_") as directory:
            root = Path(directory)
            manifest = batch_manifest(mode="manual", project_count=1)
            first = root / "first.json"
            first.write_text(json.dumps(manifest), encoding="utf-8")
            common = ["--db", str(root / "state.sqlite3"), "batch"]
            self.assertEqual(main([*common, "plan", str(first), "--root", str(root / "runs"), "--draft"]), 0)
            changed = json.loads(first.read_text(encoding="utf-8"))
            changed["projects"][0]["tasks"][0]["taskId"] = "Task_999"
            second = root / "second.json"
            second.write_text(json.dumps(changed), encoding="utf-8")
            error = io.StringIO()
            with redirect_stderr(error):
                self.assertEqual(
                    main([*common, "run-all", manifest["batchId"], "--manifest", str(second),
                          "--root", str(root / "runs"), "--fake", "--non-interactive"]),
                    2,
                )
            self.assertIn("任务集合与 --manifest 不一致", error.getvalue())
    def test_recover_remote_can_cancel_verified_running_group(self) -> None:
        args = build_parser().parse_args(
            [
                "batch",
                "recover-remote",
                "batch",
                "Project",
                "Task_001",
                "model_generation",
                "1",
                "--lease-token",
                "token",
                "--cancel-running",
                "retry",
            ]
        )
        self.assertEqual(args.cancel_running, "retry")
        self.assertFalse(args.probe)

    def test_plan_start_status_core_cli(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_cli_") as directory:
            tmp_path = Path(directory)
            manifest_path = tmp_path / "batch.json"
            manifest_path.write_text(json.dumps(batch_manifest()), encoding="utf-8")
            database = tmp_path / "state.sqlite3"
            common = ["--db", str(database), "batch"]
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([*common, "plan", str(manifest_path), "--root", str(tmp_path / "runs")]), 0)
                self.assertEqual(main([*common, "start", "batch_test"]), 0)
                self.assertEqual(main([*common, "status", "batch_test"]), 0)
            self.assertIn('"batchId": "batch_test"', output.getvalue())
            self.assertIn('"status": "running"', output.getvalue())

    def test_status_watch_emits_json_lines_and_stops_at_terminal_state(self) -> None:
        snapshots = [
            {
                "schemaVersion": 1,
                "batchId": "batch_watch",
                "status": "running",
                "stageCounts": {"running": 1},
                "tasks": [],
            },
            {
                "schemaVersion": 1,
                "batchId": "batch_watch",
                "status": "succeeded",
                "stageCounts": {"succeeded": 15},
                "tasks": [],
            },
        ]

        class Controller:
            def status(self, batch_id):
                self.assert_batch = batch_id
                return snapshots.pop(0)

        sleeps = []
        output = io.StringIO()
        _watch_batch_status(
            Controller(),
            "batch_watch",
            interval_seconds=0.25,
            json_lines=True,
            stream=output,
            sleep=sleeps.append,
        )
        lines = output.getvalue().splitlines()
        self.assertEqual([json.loads(line)["status"] for line in lines], ["running", "succeeded"])
        self.assertEqual(sleeps, [0.25])

    def test_status_table_has_stable_columns_and_truncates_long_values(self) -> None:
        rendered = _format_status_table(
            {
                "batchId": "batch_watch",
                "status": "running",
                "stageCounts": {"pending": 12, "succeeded": 3},
                "tasks": [
                    {
                        "project_id": "project-name-that-is-longer-than-the-column",
                        "task_id": "Task_001",
                        "status": "running",
                        "current_stage": "render_alignment_views",
                        "last_error_code": "worker_exit_nonzero",
                        "last_message": "子进程退出码 1（命令 1/7）",
                    }
                ],
            }
        )
        self.assertIn("PROJECT", rendered)
        self.assertIn("CURRENT STAGE", rendered)
        self.assertIn("succeeded=3 pending=12", rendered)
        self.assertIn("project-name-that-is-~", rendered)
        self.assertIn("render_alignment_views", rendered)
        self.assertIn("ERROR SUMMARY", rendered)
        self.assertIn("worker_exit_nonzero", rendered)

    def test_status_table_uses_resource_labels_and_resolved_blockers(self) -> None:
        rendered = _format_status_table(
            {
                "batchId": "batch_labels",
                "status": "running",
                "observedAt": 100.0,
                "stageCounts": {"ready": 1, "running": 3, "pending": 2},
                "tasks": [
                    {
                        "project_id": "Scene_01", "task_id": "Task_001", "status": "succeeded",
                        "current_stage": None, "stage_state": None,
                    },
                    {
                        "project_id": "Scene_01", "task_id": "Task_002", "status": "running",
                        "current_stage": "model_generation", "stage_state": "ready",
                        "queueResource": "remote_gpu", "resourceQueuePosition": 2,
                    },
                    {
                        "project_id": "Scene_01", "task_id": "Task_003", "status": "running",
                        "current_stage": "image_edit", "stage_state": "running",
                        "queueResource": "image_api", "resources": {"image_api": "slot:0"},
                    },
                    {
                        "project_id": "Scene_01", "task_id": "Task_004", "status": "running",
                        "current_stage": "model_generation", "stage_state": "running",
                        "queueResource": "remote_gpu", "resources": {"remote_gpu": "gpu:1"},
                    },
                    {
                        "project_id": "Scene_01", "task_id": "Task_005", "status": "running",
                        "current_stage": "model_generation", "stage_state": "pending",
                        "blockedBy": {
                            "stage": "upload_inputs", "stageState": "running", "queueResource": "upload",
                            "resources": {},
                        },
                    },
                    {
                        "project_id": "Scene_01", "task_id": "Task_006", "status": "running",
                        "current_stage": "model_generation", "stage_state": "pending",
                        "blockedBy": {
                            "stage": "upload_inputs", "stageState": "failed_terminal",
                            "errorCode": "invalid_input", "resources": {},
                        },
                    },
                ],
            }
        )
        self.assertIn("CURRENT STAGE", rendered)
        self.assertIn("STATUS", rendered)
        self.assertNotIn("QUEUE/RESOURCE", rendered)
        self.assertIn("完成", rendered)
        self.assertIn("排队-#2 GPU", rendered)
        self.assertIn("API调用", rendered)
        self.assertIn("服务器运行 gpu #1", rendered)
        self.assertIn("上传", rendered)
        self.assertIn("前置失败: invalid_input", rendered)

    def test_status_table_resolves_edit_gate_blocker_to_review_label(self) -> None:
        rendered = _format_status_table(
            {
                "batchId": "batch_review_blocker",
                "status": "running",
                "observedAt": 100.0,
                "stageCounts": {"pending": 1},
                "tasks": [
                    {
                        "project_id": "Scene_01", "task_id": "Task_001", "status": "running",
                        "current_stage": "upload_inputs", "stage_state": "pending",
                        "blockedBy": {
                            "stage": "edit_gate", "stageState": "pending", "resources": {},
                        },
                    },
                ],
            }
        )
        self.assertIn("等待图片审批", rendered)
        self.assertNotIn("等待调度", rendered)

    def disabled_legacy_batch_worker_cli_reports_manual_review_block_and_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_cli_worker_") as directory:
            root = Path(directory)
            manifest = batch_manifest(mode="manual", project_count=1)
            manifest_path = root / "batch.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            database = root / "state.sqlite3"
            batch_root = root / "runs"
            common = ["--db", str(database), "batch"]

            self.assertEqual(
                main([*common, "plan", str(manifest_path), "--root", str(batch_root), "--draft"]),
                0,
            )
            self.assertEqual(main([*common, "start", manifest["batchId"], "--canary"]), 0)

            error = io.StringIO()
            with redirect_stderr(error):
                self.assertEqual(main([*common, "worker", manifest["batchId"]]), 2)
            self.assertIn("必须显式选择 --fake 或 --real", error.getvalue())

            clean_environment = {
                key: value
                for key, value in os.environ.items()
                if key
                not in {
                    "UNITY_EXECUTABLE",
                    "GEMINI_IMAGE_URL",
                    "APIYI_API_KEY",
                    "APIYI_API_KEY_FILE",
                    "GEMINI_API_KEY",
                    "GEMINI_API_KEY_FILE",
                    "BEE_API_KEY",
                    "INSERTANY3D_REMOTE_TARGET",
                    "INSERTANY3D_REMOTE_PROJECT_ROOT",
                    "INSERTANY3D_REMOTE_ARTIFACT_ROOT",
                }
            }
            missing_key_file = root / "missing-apiyi-key"
            clean_environment["APIYI_API_KEY_FILE"] = str(missing_key_file)
            error = io.StringIO()
            with patch.dict(os.environ, clean_environment, clear=True), redirect_stderr(error):
                self.assertEqual(
                    main([*common, "worker", manifest["batchId"], "--real"]),
                    2,
                )
            self.assertIn("真实 worker 缺少配置", error.getvalue())
            self.assertIn(str(missing_key_file), error.getvalue())
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([*common, "status", manifest["batchId"]]), 0)
            self.assertEqual(json.loads(output.getvalue())["stageCounts"]["ready"], 5)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([*common, "worker", manifest["batchId"], "--fake"]), 0)
            worker_result = json.loads(output.getvalue())
            self.assertEqual(worker_result["executor"], "fake")
            self.assertEqual(worker_result["blockedReason"], "waiting_manual_review")
            self.assertEqual(worker_result["processedStages"], 10)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main([*common, "review", "list", manifest["batchId"]]),
                    0,
                )
            review = json.loads(output.getvalue())["items"][0]
            self.assertEqual(
                main(
                    [
                        *common,
                        "review",
                        "decide",
                        manifest["batchId"],
                        review["project_id"],
                        review["task_id"],
                        str(review["edit_attempt"]),
                        "accepted",
                    ]
                ),
                0,
            )

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([*common, "worker", manifest["batchId"], "--fake"]), 0)
            resumed = json.loads(output.getvalue())
            self.assertGreater(resumed["succeededStages"], 0)
            self.assertEqual(resumed["status"]["tasks"][0]["status"], "succeeded")

    def test_batch_evaluate_discovers_manifests_from_batch_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_cli_batch_eval_") as directory:
            root = Path(directory)
            manifest = batch_manifest(mode="automatic", project_count=1)
            manifest_path = root / "batch.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            database = root / "state.sqlite3"
            batch_root = root / "runs"
            common = ["--db", str(database), "batch"]
            self.assertEqual(
                main([*common, "plan", str(manifest_path), "--root", str(batch_root), "--draft"]),
                0,
            )
            # The wrapper reads this same layout used by top-level evaluate.
            from tests.test_evaluation import _write_manifest

            evaluation_manifest_path = _write_manifest(
                batch_root / "Scene_01" / "Task_001", project_index=1, task_index=1
            )
            evaluation_manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
            evaluation_manifest["batchId"] = manifest["batchId"]
            evaluation_manifest["projectId"] = "Scene_01"
            evaluation_manifest["scenePath"] = "Assets/Scene_01.unity"
            evaluation_manifest["taskId"] = "Task_001"
            evaluation_manifest_path.write_text(json.dumps(evaluation_manifest), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        *common,
                        "evaluate",
                        manifest["batchId"],
                        "--output",
                        str(root / "evaluation"),
                        "--fake-score",
                        "8",
                        "--expected-tasks",
                        "1",
                        "--expected-scenes",
                        "1",
                        "--tasks-per-scene",
                        "1",
                    ]
                )
            self.assertEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["batchId"], manifest["batchId"])
            self.assertEqual(result["evaluation"]["status"], "ready")
            self.assertTrue((root / "evaluation" / "batch_summary.json").is_file())
            self.assertTrue((root / "evaluation" / "task_scores.jsonl").is_file())
            self.assertTrue((root / "evaluation" / "scene_scores.csv").is_file())
            self.assertTrue((root / "evaluation" / "gpteval_summary.xlsx").is_file())

    def disabled_legacy_batch_evaluate_finalizes_ready_queue_without_recomputing_stages(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_cli_batch_finalize_") as directory:
            root = Path(directory)
            manifest = batch_manifest(mode="automatic", project_count=1)
            manifest["projects"][0]["tasks"] = manifest["projects"][0]["tasks"][:1]
            manifest_path = root / "batch.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            database = root / "state.sqlite3"
            batch_root = root / "runs"
            common = ["--db", str(database), "batch"]
            self.assertEqual(
                main([*common, "plan", str(manifest_path), "--root", str(batch_root), "--draft"]),
                0,
            )
            self.assertEqual(main([*common, "start", manifest["batchId"], "--canary"]), 0)
            self.assertEqual(
                main(
                    [
                        *common,
                        "worker",
                        manifest["batchId"],
                        "--fake",
                        "--max-steps",
                        "13",
                    ]
                ),
                0,
            )
            with sqlite3.connect(database) as connection:
                connection.row_factory = sqlite3.Row
                committed_eval6 = connection.execute(
                    """SELECT s.id AS stage_id, a.id AS attempt_id, a.output_dir
                         FROM stages s JOIN attempts a ON a.stage_id=s.id
                        WHERE s.batch_id=? AND s.project_id='Scene_01'
                          AND s.task_id='Task_001' AND s.name='unity_eval6'
                          AND s.state='succeeded' AND a.status='succeeded'""",
                    (manifest["batchId"],),
                ).fetchone()
            self.assertIsNotNone(committed_eval6)
            evaluation_manifest_path = _write_manifest(
                Path(committed_eval6["output_dir"]), project_index=1, task_index=1
            )
            evaluation_manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
            evaluation_manifest["batchId"] = manifest["batchId"]
            evaluation_manifest["projectId"] = "Scene_01"
            evaluation_manifest["scenePath"] = "Assets/Scene_01.unity"
            evaluation_manifest["taskId"] = "Task_001"
            evaluation_manifest_path.write_text(json.dumps(evaluation_manifest), encoding="utf-8")
            payload = evaluation_manifest_path.read_bytes()
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """DELETE FROM artifacts
                       WHERE stage_id=? AND attempt_id=? AND relative_path=?""",
                    (
                        committed_eval6["stage_id"],
                        committed_eval6["attempt_id"],
                        evaluation_manifest_path.relative_to(batch_root).as_posix(),
                    ),
                )
                connection.execute(
                    """INSERT INTO artifacts(
                           artifact_id, stage_id, attempt_id, type, relative_path,
                           sha256, size, created_at
                       ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        "evaluation_manifest",
                        committed_eval6["stage_id"],
                        committed_eval6["attempt_id"],
                        "evaluation_manifest",
                        evaluation_manifest_path.relative_to(batch_root).as_posix(),
                        hashlib.sha256(payload).hexdigest(),
                        len(payload),
                        0.0,
                    ),
                )
            stale_manifest_path = _write_manifest(
                batch_root / "stale-attempt" / "output.staging",
                project_index=9,
                task_index=9,
            )
            stale_manifest = json.loads(stale_manifest_path.read_text(encoding="utf-8"))
            stale_manifest["batchId"] = "uncommitted_wrong_batch"
            stale_manifest_path.write_text(json.dumps(stale_manifest), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            *common,
                            "evaluate",
                            manifest["batchId"],
                            "--output",
                            str(root / "evaluation"),
                            "--fake-score",
                            "8",
                            "--expected-tasks",
                            "1",
                            "--expected-scenes",
                            "1",
                            "--tasks-per-scene",
                            "1",
                        ]
                    ),
                    0,
                )
            result = json.loads(output.getvalue())
            self.assertEqual(result["queue"]["gate"], "ready")
            self.assertEqual(result["queue"]["finalizedStages"], 1)
            self.assertEqual(result["queue"]["status"]["status"], "succeeded")

    def test_batch_evaluate_rejects_wrong_batch_before_writing_responses(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_cli_batch_eval_guard_") as directory:
            root = Path(directory)
            manifest = batch_manifest(mode="automatic", project_count=1)
            manifest_path = root / "batch.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            database = root / "state.sqlite3"
            batch_root = root / "runs"
            common = ["--db", str(database), "batch"]
            self.assertEqual(
                main([*common, "plan", str(manifest_path), "--root", str(batch_root), "--draft"]),
                0,
            )
            wrong_input = root / "wrong-batch"
            evaluation_manifest_path = _write_manifest(
                wrong_input / "Scene_01" / "Task_001", project_index=1, task_index=1
            )
            evaluation_manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
            evaluation_manifest["batchId"] = "some_other_batch"
            evaluation_manifest_path.write_text(json.dumps(evaluation_manifest), encoding="utf-8")
            output_root = root / "evaluation"
            error = io.StringIO()
            with redirect_stderr(error):
                code = main(
                    [
                        *common,
                        "evaluate",
                        manifest["batchId"],
                        "--input",
                        str(wrong_input),
                        "--output",
                        str(output_root),
                        "--fake-score",
                        "8",
                        "--expected-tasks",
                        "1",
                        "--expected-scenes",
                        "1",
                        "--tasks-per-scene",
                        "1",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("与目标批次", error.getvalue())
            self.assertFalse((output_root / "responses").exists())

    def test_evaluate_plan_fake_run_status_and_summarize_are_offline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_cli_eval_") as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            output_dir = root / "evaluation"
            _write_manifest(input_dir / "task", project_index=1, task_index=1)
            common = [
                "evaluate",
                "COMMAND",
                str(input_dir),
                "--output",
                str(output_dir),
                "--model",
                "offline-model",
                "--expected-tasks",
                "1",
                "--expected-scenes",
                "1",
                "--tasks-per-scene",
                "1",
            ]
            transport_calls = []

            def forbidden_transport(*args):
                transport_calls.append(args)
                raise AssertionError("offline command attempted network transport")

            code, planned, error = _invoke_eval(
                _replace_command(common, "plan"), transport=forbidden_transport
            )
            self.assertEqual(code, 0, error)
            self.assertEqual(planned["status"], "ready")
            self.assertEqual(planned["requests"]["planned"], 1)
            self.assertEqual(planned["requests"]["pending"], 1)
            self.assertEqual(
                planned["dimensions"],
                ["visual_quality", "geometric_accuracy"],
            )
            self.assertEqual(planned["network"], {"allowed": False, "requestsSent": 0})
            self.assertEqual(transport_calls, [])

            code, _result, error = _invoke_eval(
                _replace_command(common, "run"), transport=forbidden_transport
            )
            self.assertEqual(code, 2)
            self.assertIn("默认禁止付费请求", error)
            self.assertEqual(transport_calls, [])

            fake_args = [*_replace_command(common, "run"), "--fake-score", "8", "--retries", "0"]
            code, run_result, error = _invoke_eval(fake_args, transport=forbidden_transport)
            self.assertEqual(code, 0, error)
            self.assertEqual(run_result["executionMode"], "fixed_fake_response")
            self.assertEqual(run_result["progress"]["completed"], 1)
            self.assertEqual(run_result["status"], "ready")
            self.assertIn("xlsx", run_result["outputs"])
            self.assertEqual(transport_calls, [])

            code, status, error = _invoke_eval(
                _replace_command(common, "status"), transport=forbidden_transport
            )
            self.assertEqual(code, 0, error)
            self.assertEqual(status["status"], "ready")
            self.assertEqual(status["completion"]["readyTaskResults"], 1)

            code, summarized, error = _invoke_eval(
                _replace_command(common, "summarize"), transport=forbidden_transport
            )
            self.assertEqual(code, 0, error)
            self.assertEqual(summarized["status"], "ready")
            self.assertTrue((output_dir / "batch_summary.json").is_file())
            self.assertTrue((output_dir / "task_scores.jsonl").is_file())
            self.assertTrue((output_dir / "scene_scores.csv").is_file())
            self.assertTrue((output_dir / "gpteval_summary.xlsx").is_file())
            self.assertEqual(transport_calls, [])

    def test_paid_evaluate_requires_flag_and_environment_key_and_uses_injected_transport(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_cli_paid_") as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            output_dir = root / "evaluation"
            _write_manifest(input_dir / "task", project_index=1, task_index=1)
            arguments = [
                "evaluate",
                "run",
                str(input_dir),
                "--output",
                str(output_dir),
                "--model",
                "fake-provider-model",
                "--expected-tasks",
                "1",
                "--expected-scenes",
                "1",
                "--tasks-per-scene",
                "1",
                "--allow-paid-api",
                "--base-url",
                "https://fake.invalid/v1",
                "--retries",
                "0",
            ]
            calls = []

            def fake_transport(endpoint, headers, body, timeout):
                calls.append((endpoint, dict(headers), body, timeout))
                response = fixed_fake_response(9)
                return {
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(response)}]}}
                    ]
                }

            missing_key_file = root / "missing-apiyi-key"
            with patch.dict(
                os.environ,
                {"APIYI_API_KEY_FILE": str(missing_key_file)},
                clear=True,
            ):
                code, _result, error = _invoke_eval(arguments, transport=fake_transport)
            self.assertEqual(code, 2)
            self.assertIn(str(missing_key_file), error)
            self.assertEqual(calls, [])

            key_file = root / "apiyi-key"
            key_file.write_text("fake-secret\n", encoding="utf-8")
            if os.name == "posix":
                key_file.chmod(0o600)
            with patch.dict(
                os.environ,
                {
                    "APIYI_API_KEY_FILE": str(key_file),
                    "BEE_API_KEY": "invalid-legacy-key",
                },
                clear=True,
            ):
                code, result, error = _invoke_eval(arguments, transport=fake_transport)
            self.assertEqual(code, 0, error)
            self.assertEqual(result["executionMode"], f"paid_api:{key_file}")
            self.assertEqual(len(calls), 1)
            endpoint, headers, body, timeout = calls[0]
            self.assertEqual(
                endpoint,
                "https://fake.invalid/v1beta/models/fake-provider-model:generateContent",
            )
            self.assertEqual(headers["x-goog-api-key"], "fake-secret")
            self.assertEqual(timeout, 300.0)
            parts = body["contents"][0]["parts"]
            self.assertEqual(len([part for part in parts if "inlineData" in part]), 12)
            serialized_outputs = "".join(
                path.read_text(encoding="utf-8")
                for path in output_dir.rglob("*.json")
            )
            self.assertNotIn("fake-secret", serialized_outputs)

    def test_paid_evaluate_failure_reports_mode_and_absolute_error_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_cli_paid_failure_") as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            output_dir = root / "evaluation"
            _write_manifest(input_dir / "task_1", project_index=1, task_index=1)
            _write_manifest(input_dir / "task_2", project_index=1, task_index=2)
            arguments = [
                "evaluate",
                "run",
                str(input_dir),
                "--output",
                str(output_dir),
                "--model",
                "fake-provider-model",
                "--expected-tasks",
                "2",
                "--expected-scenes",
                "1",
                "--tasks-per-scene",
                "2",
                "--allow-paid-api",
                "--base-url",
                "https://fake.invalid/v1",
                "--retries",
                "0",
            ]

            def failing_transport(endpoint, headers, body, timeout):
                self.assertEqual(
                    endpoint,
                    "https://fake.invalid/v1beta/models/"
                    "fake-provider-model:generateContent",
                )
                self.assertEqual(headers["x-goog-api-key"], "secret-that-must-not-leak")
                raise RuntimeError("fixed offline transport failure")

            key_file = root / "apiyi-key"
            key_file.write_text("secret-that-must-not-leak\n", encoding="utf-8")
            if os.name == "posix":
                key_file.chmod(0o600)
            with patch.dict(
                os.environ,
                {
                    "APIYI_API_KEY_FILE": str(key_file),
                    "BEE_API_KEY": "invalid-legacy-key",
                },
                clear=True,
            ):
                code, result, error = _invoke_eval(
                    arguments,
                    transport=failing_transport,
                )

            self.assertEqual(code, 2)
            self.assertIsNone(result)
            error_paths = sorted(
                path.resolve()
                for path in (output_dir / "responses" / "errors").glob("*.json")
            )
            self.assertEqual(len(error_paths), 2)
            self.assertIn(f"执行模式: paid_api:{key_file}", error)
            error_lines = error.splitlines()
            for error_path in error_paths:
                self.assertTrue(error_path.is_file())
                self.assertIn(str(error_path), error_lines)
            self.assertNotIn("secret-that-must-not-leak", error)
            for error_path in error_paths:
                record = json.loads(error_path.read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["message"], "fixed offline transport failure")
                self.assertNotIn(
                    "secret-that-must-not-leak",
                    error_path.read_text(encoding="utf-8"),
                )

    def test_evaluate_rejects_non_finite_timing_options_without_transport(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_cli_eval_timing_") as directory:
            root = Path(directory)
            input_dir = root / "inputs"
            output_dir = root / "evaluation"
            _write_manifest(input_dir / "task", project_index=1, task_index=1)
            calls = []

            def forbidden_transport(*args):
                calls.append(args)
                raise AssertionError("invalid timing option reached transport")

            common = [
                "evaluate",
                "run",
                str(input_dir),
                "--output",
                str(output_dir),
                "--expected-tasks",
                "1",
                "--expected-scenes",
                "1",
                "--tasks-per-scene",
                "1",
            ]
            code, _result, error = _invoke_eval(
                [*common, "--fake-score", "7", "--retry-delay-seconds", "nan"],
                transport=forbidden_transport,
            )
            self.assertEqual(code, 2)
            self.assertIn("有限的非负数", error)

            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-secret"}, clear=False):
                code, _result, error = _invoke_eval(
                    [*common, "--allow-paid-api", "--timeout", "inf"],
                    transport=forbidden_transport,
                )
            self.assertEqual(code, 2)
            self.assertIn("有限的正数", error)
            self.assertEqual(calls, [])


def _replace_command(arguments: list[str], command: str) -> list[str]:
    return [command if value == "COMMAND" else value for value in arguments]


def _invoke_eval(arguments, *, transport):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(arguments, evaluation_transport=transport)
    value = json.loads(stdout.getvalue()) if stdout.getvalue().strip() else None
    return code, value, stderr.getvalue()


if __name__ == "__main__":
    unittest.main()
