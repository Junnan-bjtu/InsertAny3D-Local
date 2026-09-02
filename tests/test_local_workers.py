from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from insertany3d.executors import CommandExecutor, ExecutionResult
from insertany3d.local_workers import (
    ImageWorkerConfig,
    LocalStageExecutor,
    LocalWorkerConfigurationError,
    local_worker_capacities,
)
from insertany3d.scheduler import BatchController, WorkItem, default_capacities
from insertany3d.store import SchedulerStore, StoreError
from insertany3d.worker import BatchWorker
from tests.fixtures import batch_manifest


class ImageHandler(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length))
        type(self).received.append(value)
        if self.path == "/429":
            self._send(429, {"Retry-After": "2", "X-Request-Id": "limit-test"})
            return
        if self.path == "/503":
            self._send(503, {"Retry-After": "1"})
            return
        if self.path == "/timeout":
            time.sleep(0.12)
        image = base64.b64encode(b"edited-image-bytes").decode("ascii")
        self._send(
            200,
            body={"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": image}}]}}]},
        )

    def _send(self, status: int, headers=None, body=None):
        encoded = json.dumps(body or {"status": status}).encode("utf-8")
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except BrokenPipeError:
            pass

    def log_message(self, *_args):
        return


class SlowStageExecutor:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def execute(self, controller: BatchController, item: WorkItem) -> ExecutionResult:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.04)
            output = item.staging_dir / "result.json"
            output.write_text(json.dumps({"stage": item.stage}), encoding="utf-8")
            return ExecutionResult(
                True,
                artifacts=[{"artifactId": "result", "type": "test", "path": output.name}],
            )
        finally:
            with self._lock:
                self.active -= 1


class LocalWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="insertany3d_local_worker_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _controller(self, *, mode: str = "manual", project_count: int = 1):
        manifest = batch_manifest(mode=mode, project_count=project_count)
        manifest["projects"][0]["tasks"][0]["anchorPrompt"] = "the tractor hood"
        store = SchedulerStore(self.root / "state.sqlite3")
        controller = BatchController(store, lease_seconds=2)
        controller.plan(manifest, self.root / "runs", formal=False)
        controller.start(manifest["batchId"], formal=False)
        return store, controller, manifest

    @staticmethod
    def _commit_center(controller: BatchController, manifest: dict) -> None:
        item = controller.lease_next(
            manifest["batchId"],
            "anchor-fixture",
            default_capacities(manifest),
            project_id="Scene_01",
            task_id="Task_001",
            stage_name="unity_anchor",
        )
        assert item is not None
        center = item.staging_dir / "step1" / "center" / "image.png"
        center.parent.mkdir(parents=True)
        center.write_bytes(b"source-image-bytes")
        controller.commit_success(
            item,
            [{"artifactId": "center", "type": "scene_rgb", "path": "step1/center/image.png"}],
        )

    def test_real_executor_requires_explicit_opt_in(self) -> None:
        with self.assertRaisesRegex(LocalWorkerConfigurationError, "默认关闭"):
            LocalStageExecutor(unity_executable="Unity")
        manifest = batch_manifest(project_count=1)
        self.assertEqual(local_worker_capacities(manifest)["image_api"], 24)

    def test_expired_commit_is_reported_and_recovered_without_worker_crash(self) -> None:
        class Clock:
            now = 0.0

            def __call__(self):
                return self.now

        class ExpiringExecutor:
            def execute(self, _controller, item):
                clock.now = 3.0
                output = item.staging_dir / "result.json"
                output.write_text("expired", encoding="utf-8")
                return ExecutionResult(
                    True,
                    artifacts=[{"artifactId": "result", "type": "test", "path": output.name}],
                )

        clock = Clock()
        store, controller, manifest = self._controller()
        controller.clock = clock
        try:
            report = BatchWorker(
                controller,
                default_capacities(manifest),
                ExpiringExecutor(),
                clock=clock,
            ).run(manifest["batchId"], once=True)
            self.assertEqual(report.succeeded_stages, 0)
            self.assertEqual(report.failed_stages, 1)
            self.assertEqual(len(report.submission_errors), 1)
            self.assertEqual(report.submission_errors[0]["errorCode"], "lease_fenced")
            self.assertEqual(report.status["workerErrors"][0]["operation"], "commit_success")
            stage = store.row(
                "SELECT state, last_error_code, last_message FROM stages "
                "WHERE batch_id=? AND name='unity_anchor'",
                (manifest["batchId"],),
            )
            self.assertIsNotNone(stage)
            self.assertIn(stage["state"], {"ready", "failed_terminal"})
            self.assertIn(stage["last_error_code"], {"worker_lost", "lease_fenced"})
            if stage["last_error_code"] == "lease_fenced":
                self.assertIn("commit_success 提交失败", stage["last_message"])
        finally:
            store.close()

    def test_expired_failure_submission_is_reported_without_worker_crash(self) -> None:
        class Clock:
            now = 0.0

            def __call__(self):
                return self.now

        class FailingExecutor:
            def execute(self, _controller, _item):
                clock.now = 3.0
                return ExecutionResult(False, error_code="worker_crash", message="child failed")

        clock = Clock()
        store, controller, manifest = self._controller()
        controller.clock = clock
        original_fail = controller.fail

        def fenced_fail(*_args, **_kwargs):
            raise StoreError("lease 已过期")

        controller.fail = fenced_fail
        try:
            report = BatchWorker(
                controller,
                default_capacities(manifest),
                FailingExecutor(),
                clock=clock,
            ).run(manifest["batchId"], once=True)
            self.assertEqual(report.failed_stages, 1)
            self.assertEqual(len(report.submission_errors), 1)
            self.assertEqual(report.submission_errors[0]["operation"], "fail")
            self.assertEqual(report.submission_errors[0]["errorCode"], "lease_fenced")
        finally:
            controller.fail = original_fail
            store.close()

    def test_unity_stage_uses_request_preflight_and_command_executor(self) -> None:
        store, controller, manifest = self._controller()
        try:
            project = self.root / "UnityProject"
            (project / "Packages").mkdir(parents=True)
            (project / "Assets").mkdir()
            (project / "Packages" / "manifest.json").write_text(
                json.dumps({"dependencies": {"com.junnan.insertany3d": "file:package"}}),
                encoding="utf-8",
            )
            (project / "Assets" / "Scene_01.unity").write_text("fixture", encoding="utf-8")
            with store.transaction() as connection:
                row = connection.execute("SELECT manifest_json FROM batches WHERE batch_id=?", (manifest["batchId"],)).fetchone()
                stored = json.loads(row["manifest_json"])
                stored["projects"][0]["projectPath"] = str(project)
                stored["projects"][0]["scenePath"] = "Assets/Scene_01.unity"
                connection.execute(
                    "UPDATE batches SET manifest_json=? WHERE batch_id=?",
                    (json.dumps(stored), manifest["batchId"]),
                )
            fake_unity = self.root / "fake-unity"
            fake_unity.write_text(
                """#!/usr/bin/env python3
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
def arg(name): return sys.argv[sys.argv.index(name) + 1]
request = json.loads(Path(arg('-insertAny3DRequest')).read_text())
result_path = Path(arg('-insertAny3DResult'))
output = result_path.parent / 'step1' / 'center' / 'image.png'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(b'fake-unity-center')
artifact = {'artifactId':'center','type':'scene_rgb','path':'step1/center/image.png','sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'size':output.stat().st_size}
result = {'schemaVersion':1,'kind':'insertany3d.stage-result','batchId':request['batchId'],'projectId':request['projectId'],'taskId':request['taskId'],'stage':request['stage'],'contractVersion':request['contractVersion'],'attempt':request['attempt'],'leaseToken':request['leaseToken'],'status':'succeeded','artifacts':[artifact],'errorCode':None,'message':None,'diagnosticPaths':[],'cleanup':{'completed':True},'finishedAtUtc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
result_path.write_text(json.dumps(result))
""",
                encoding="utf-8",
            )
            fake_unity.chmod(0o755)
            guarded = []
            executor = LocalStageExecutor(
                allow_real=True,
                unity_executable=str(fake_unity),
                command_executor=CommandExecutor(heartbeat_seconds=0.01),
                unity_process_guard=lambda path: guarded.append(Path(path)),
            )
            report = BatchWorker(
                controller,
                default_capacities(manifest),
                executor,
                max_parallel=1,
            ).run(manifest["batchId"], once=True)
            self.assertEqual(report.succeeded_stages, 1)
            self.assertEqual(report.stage_names, ["unity_anchor"])
            self.assertEqual(guarded, [project.resolve()])
            request_path = self.root / "runs" / "requests" / "Scene_01" / "Task_001" / "unity_anchor" / "attempt-0001.json"
            self.assertTrue(request_path.is_file())
        finally:
            store.close()

    def test_image_edit_uses_center_artifact_and_task_prompts(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ImageHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        ImageHandler.received = []
        thread.start()
        store, controller, manifest = self._controller()
        try:
            self._commit_center(controller, manifest)
            token = "fake-secret-token"
            executor = LocalStageExecutor(
                allow_real=True,
                image_config=ImageWorkerConfig(
                    endpoint=f"http://127.0.0.1:{server.server_port}/ok?secret=must-not-persist",
                    token=token,
                    timeout_seconds=1,
                    num_gen_image_per_task=1,
                ),
                image_heartbeat_seconds=0.01,
            )
            report = BatchWorker(
                controller,
                default_capacities(manifest),
                executor,
                max_parallel=4,
            ).run(manifest["batchId"], once=True)
            self.assertEqual(report.succeeded_stages, 1)
            self.assertEqual(report.stage_names, ["image_edit"])
            parts = ImageHandler.received[0]["contents"][0]["parts"]
            self.assertIn("LOCKED ANCHOR", parts[0]["text"])
            self.assertIn("the tractor hood", parts[0]["text"])
            self.assertIn("add chair", parts[0]["text"])
            self.assertEqual(base64.b64decode(parts[1]["inlineData"]["data"]), b"source-image-bytes")
            outputs = list((self.root / "runs").rglob("edited-*.png"))
            self.assertEqual(len(outputs), 1)
            output = outputs[0]
            self.assertRegex(output.name, r"^edited-[0-9a-f]{32}\.png$")
            self.assertEqual(output.read_bytes(), b"edited-image-bytes")
            metadata = json.loads(output.with_name("image_edit.json").read_text(encoding="utf-8"))
            self.assertNotIn("secret=", metadata["request"]["endpoint"])
            self.assertNotIn(token, json.dumps(metadata))
            timing = metadata["timing"]
            self.assertGreaterEqual(timing["queueWaitSeconds"], 0)
            self.assertGreaterEqual(timing["apiRequestSeconds"], 0)
            self.assertGreaterEqual(timing["stageElapsedSeconds"], timing["apiRequestSeconds"])
            self.assertLessEqual(timing["requestStartedAtUtc"], timing["requestEndedAtUtc"])
        finally:
            store.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_image_edit_request_and_metadata_keep_three_generation_indices(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ImageHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        ImageHandler.received = []
        thread.start()
        store, controller, manifest = self._controller()
        try:
            self._commit_center(controller, manifest)
            executor = LocalStageExecutor(
                allow_real=True,
                image_config=ImageWorkerConfig(
                    endpoint=f"http://127.0.0.1:{server.server_port}/ok",
                    token="fake-token",
                    timeout_seconds=1,
                    num_gen_image_per_task=3,
                ),
                image_heartbeat_seconds=0.01,
            )
            report = BatchWorker(
                controller,
                default_capacities(manifest),
                executor,
                max_parallel=3,
            ).run(manifest["batchId"], once=False, max_steps=10)
            self.assertEqual(report.blocked_reason, "waiting_manual_review")
            outputs = sorted((self.root / "runs").rglob("image_edit.json"))
            self.assertEqual(len(outputs), 3)
            self.assertEqual(
                [json.loads(path.read_text(encoding="utf-8"))["generationIndex"] for path in outputs],
                [1, 2, 3],
            )
            requests = sorted(
                self.root.joinpath("runs", "requests", "Scene_01", "Task_001", "image_edit").glob("attempt-*.json")
            )
            self.assertEqual(len(requests), 3)
            self.assertEqual(
                [json.loads(path.read_text(encoding="utf-8"))["effectiveConfig"]["task"]["generationIndex"] for path in requests],
                [1, 2, 3],
            )
        finally:
            store.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_image_api_429_503_and_timeout_keep_distinct_scheduler_states(self) -> None:
        for route, expected_code, expected_state in (
            ("429", "http_429", "ready"),
            ("503", "http_503", "ready"),
            ("timeout", "delivery_unknown", "waiting_manual"),
        ):
            with self.subTest(route=route):
                case_root = self.root / route
                case_root.mkdir()
                previous = self.root
                self.root = case_root
                server = ThreadingHTTPServer(("127.0.0.1", 0), ImageHandler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                store, controller, manifest = self._controller()
                try:
                    self._commit_center(controller, manifest)
                    executor = LocalStageExecutor(
                        allow_real=True,
                        image_config=ImageWorkerConfig(
                            endpoint=f"http://127.0.0.1:{server.server_port}/{route}",
                            token=f"token-{route}",
                            timeout_seconds=0.03 if route == "timeout" else 1,
                        ),
                        image_heartbeat_seconds=0.01,
                    )
                    report = BatchWorker(
                        controller,
                        default_capacities(manifest),
                        executor,
                    ).run(manifest["batchId"], once=True)
                    self.assertEqual(report.failed_stages, 1)
                    row = store.row(
                        "SELECT state, last_error_code FROM stages WHERE batch_id=? AND project_id='Scene_01' AND task_id='Task_001' AND name='image_edit'",
                        (manifest["batchId"],),
                    )
                    self.assertEqual(row["last_error_code"], expected_code)
                    self.assertEqual(row["state"], expected_state)
                    attempt = store.row(
                        "SELECT staging_dir FROM attempts a JOIN stages s ON s.id=a.stage_id WHERE s.name='image_edit' ORDER BY a.id DESC LIMIT 1"
                    )
                    diagnostic = json.loads(Path(attempt["staging_dir"]).joinpath("image_api_outcome.json").read_text())
                    self.assertEqual(diagnostic["errorCode"], expected_code)
                    timing = diagnostic["timing"]
                    self.assertGreaterEqual(timing["stageElapsedSeconds"], 0)
                    if route != "429" and route != "503":
                        self.assertIsNotNone(timing["requestStartedAtUtc"])
                finally:
                    store.close()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
                    self.root = previous

    def test_parallel_worker_obeys_scheduler_capacity(self) -> None:
        store, controller, manifest = self._controller(mode="automatic", project_count=2)
        try:
            manifest["resources"]["unitySlots"] = 2
            executor = SlowStageExecutor()
            worker = BatchWorker(
                controller,
                {**default_capacities(manifest), "unity_gpu": 2},
                executor,
                stage_names=["unity_anchor"],
                max_parallel=8,
            )
            report = worker.run(manifest["batchId"], max_steps=10)
            self.assertEqual(report.succeeded_stages, 10)
            self.assertEqual(executor.peak, 2)
            self.assertEqual(set(report.stage_names), {"unity_anchor"})
            self.assertEqual(report.blocked_reason, "max_steps_reached")
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
