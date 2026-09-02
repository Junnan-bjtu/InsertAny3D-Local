from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import stage_adapter


CONTRACTS = stage_adapter.SUPPORTED_CONTRACTS


class StageAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="insertany3d_stage_adapter_")
        self.root = Path(self.temporary.name)
        self.fake = self.root / "fake_worker.py"
        self.fake.write_text(
            """from __future__ import annotations
import json, os, pathlib, sys, time
mode = sys.argv[1]
if mode == 'sleep':
    root = pathlib.Path(os.environ['INSERTANY3D_STAGE_OUTPUT'])
    root.mkdir(parents=True, exist_ok=True)
    (root / '_worker.pid').write_text(str(os.getpid()), encoding='ascii')
    time.sleep(10)
if mode == 'fail':
    raise SystemExit(7)
if mode == 'missing':
    raise SystemExit(0)
plan = json.loads(pathlib.Path(os.environ['INSERTANY3D_STAGE_PLAN']).read_text(encoding='utf-8'))
root = pathlib.Path(os.environ['INSERTANY3D_STAGE_OUTPUT'])
for relative in plan['requiredOutputs']:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.json':
        value = {'status': 'ready'} if relative == 'pose.json' else {'ok': True}
        path.write_text(json.dumps(value), encoding='utf-8')
    else:
        path.write_bytes(b'fixture')
if mode == 'mutate':
    pathlib.Path(sys.argv[2]).write_bytes(b'changed-during-run')
""",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def add_input(self, artifact_id: str, relative: str, content: bytes = b"fixture") -> dict:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"artifactId": artifact_id, "path": relative, "sha256": stage_adapter.sha256_file(path)}

    def request(self, stage: str, inputs: list[dict], options: dict | None = None) -> dict:
        config = {"stage": stage, "stageOptions": options or {}}
        return {
            "schemaVersion": 1,
            "kind": "insertany3d.stage-request",
            "batchId": "batch_test",
            "projectId": "Scene_01",
            "taskId": "Task_001",
            "stage": stage,
            "contractVersion": CONTRACTS[stage],
            "attempt": 1,
            "leaseToken": "lease-test",
            "inputs": inputs,
            "effectiveConfig": config,
            "effectiveConfigSha256": stage_adapter.canonical_sha256(config),
            "outputStagingDir": f"stages/{stage}/attempt-0001/output.staging",
        }

    def plan(self, request: dict) -> stage_adapter.StagePlan:
        checked = stage_adapter.validate_request(request)
        inputs = stage_adapter.resolve_inputs(checked, self.root.resolve())
        output = self.root / checked["outputStagingDir"]
        return stage_adapter.build_plan(checked, inputs, output)

    def test_all_stage_command_mappings_use_only_existing_runtime_clis(self) -> None:
        model = self.request("model_generation", [self.add_input("input_image", "inputs/edited.png")])
        render = self.request(
            "render_alignment_views", [self.add_input("sample_ply", "inputs/sample.ply")],
            {"viewNames": ["left", "center", "right"], "yawOffsets": [-24, 0, 24]},
        )
        segment = self.request(
            "segment_inputs", [self.add_input("input_image", "inputs/segment.png")],
            {"mode": "target", "prompt": "red chair"},
        )
        gim = self.request(
            "gim_match",
            [self.add_input("image0", "inputs/scene.png"), self.add_input("image1", "inputs/generated.png")],
        )
        pose_inputs = [
            self.add_input("generated_cameras", "inputs/render/cameras.txt"),
            self.add_input("generated_images", "inputs/render/images.txt"),
        ]
        pose_views = []
        for name in ("left", "center", "right"):
            matches_id = f"matches_{name}"
            scene_depth_id = f"scene_depth_{name}"
            scene_camera_id = f"scene_camera_{name}"
            generated_depth_id = f"generated_depth_{name}"
            pose_inputs.extend((
                self.add_input(matches_id, f"inputs/gim/{name}/matches.json", b"{}"),
                self.add_input(scene_depth_id, f"inputs/scene/{name}.raw"),
                self.add_input(scene_camera_id, f"inputs/scene/{name}.camera.json", b"{}"),
                self.add_input(generated_depth_id, f"inputs/generated/{name}.npy"),
            ))
            pose_views.append({
                "name": name, "matchesArtifactId": matches_id, "sceneDepthArtifactId": scene_depth_id,
                "sceneCameraArtifactId": scene_camera_id, "generatedDepthArtifactId": generated_depth_id,
            })
        pose = self.request("estimate_pose", pose_inputs, {"views": pose_views})
        sags_inputs = [self.add_input("model_cfg_args", "inputs/model/cfg_args")]
        annotations = []
        for index, name in enumerate(("center", "ring_060", "ring_120", "ring_180", "ring_240", "ring_300")):
            mask_id, points_id = f"mask_{index}", f"points_{index}"
            sags_inputs.extend((
                self.add_input(mask_id, f"inputs/annotations/{name}/mask.png"),
                self.add_input(points_id, f"inputs/annotations/{name}/points.json", b"[]"),
            ))
            annotations.append({"name": name, "maskArtifactId": mask_id, "pointsArtifactId": points_id})
        sags = self.request("sags_segment_vote", sags_inputs, {"annotations": annotations})
        debug = self.request(
            "debug_bundle", [self.add_input("batch_manifest", "inputs/run/batch_manifest.json", b"{}")],
            {"allowCompatibilityFacade": True},
        )

        plans = {request["stage"]: self.plan(request) for request in (model, render, segment, gim, pose, sags, debug)}
        scripts = {
            stage: [Path(command[1]).name for command in plan.commands]
            for stage, plan in plans.items()
        }
        self.assertEqual(scripts["model_generation"], ["generate_trellis_asset.py"])
        self.assertEqual(scripts["render_alignment_views"], ["render_trellis_views.py"])
        self.assertEqual(scripts["segment_inputs"], ["auto_segment.py"])
        self.assertEqual(scripts["gim_match"], ["run_gim_match.py"])
        self.assertEqual(scripts["estimate_pose"], ["estimate_similarity_pose.py"])
        self.assertEqual(scripts["sags_segment_vote"], ["run_sags_text.py"])
        self.assertEqual(scripts["debug_bundle"], ["build_debug_bundle.py"])
        self.assertIn("--min-votes", plans["sags_segment_vote"].commands[0])
        min_vote_index = plans["sags_segment_vote"].commands[0].index("--min-votes")
        self.assertEqual(plans["sags_segment_vote"].commands[0][min_vote_index + 1], "3")
        self.assertTrue(plans["debug_bundle"].compatibility_facade)
        self.assertIn("--skip-depth-gim", plans["debug_bundle"].commands[0])

    def test_ring6_is_wired_inside_render_and_segment_stages(self) -> None:
        ring_names = ["center", "ring_060", "ring_120", "ring_180", "ring_240", "ring_300"]
        render = self.request(
            "render_alignment_views",
            [self.add_input("sample_ply", "inputs/ring/sample.ply")],
            {
                "viewNames": ["left", "center", "right"],
                "yawOffsets": [-24, 0, 24],
                "ringViewNames": ring_names,
                "ringYawOffsets": [0, 60, 120, 180, 240, 300],
            },
        )
        render_plan = self.plan(render)
        self.assertEqual(len(render_plan.commands), 2)
        self.assertEqual(Path(render_plan.commands[1][1]).name, "render_trellis_views.py")
        self.assertIn("ring6/source/images/ring_300.png", render_plan.required_outputs)

        segment_inputs = [
            self.add_input(f"ring_image_{index}", f"inputs/ring/{name}.png")
            for index, name in enumerate(ring_names)
        ]
        segment = self.request(
            "segment_inputs",
            segment_inputs,
            {
                "mode": "target",
                "inputImageArtifactId": "ring_image_0",
                "taskPrompt": "insert a red chair",
                "sagsTaskPrompt": "chair",
                "sagsViews": [
                    {"name": name, "imageArtifactId": f"ring_image_{index}"}
                    for index, name in enumerate(ring_names)
                ],
            },
        )
        segment_plan = self.plan(segment)
        self.assertEqual(len(segment_plan.commands), 7)
        self.assertTrue(all(Path(command[1]).name == "auto_segment.py" for command in segment_plan.commands))
        self.assertIn("sags_annotations/ring_300/points.json", segment_plan.required_outputs)

    def test_atomic_debug_bundle_copies_only_hash_verified_inputs(self) -> None:
        source = self.add_input("pose", "inputs/atomic/pose.json", b'{"status":"ready"}')
        request = self.request("debug_bundle", [source], {"mode": "atomic"})
        checked = stage_adapter.validate_request(request)
        result, plan = stage_adapter.execute_request(checked, self.root)
        self.assertTrue(plan.atomic_bundle)
        self.assertEqual(plan.commands, ())
        self.assertEqual(result["status"], "succeeded")
        task_root = self.root / request["outputStagingDir"] / "Task_001"
        manifest = json.loads((task_root / "artifact_index.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["artifactCount"], 1)
        bundled = task_root / manifest["artifacts"][0]["bundledPath"]
        self.assertEqual(stage_adapter.sha256_file(bundled), source["sha256"])

    def test_fake_command_success_writes_hashed_artifacts(self) -> None:
        request = self.request("model_generation", [self.add_input("input_image", "inputs/edited.png")])
        checked = stage_adapter.validate_request(request)
        result, _ = stage_adapter.execute_request(
            checked, self.root, fake_command=[sys.executable, str(self.fake), "success"], poll_seconds=0.01,
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["leaseToken"], "lease-test")
        self.assertEqual(result["contractVersion"], "model-generation-v1")
        self.assertGreaterEqual(len(result["artifacts"]), 2)
        for artifact in result["artifacts"]:
            path = self.root / request["outputStagingDir"] / artifact["path"]
            self.assertEqual(stage_adapter.sha256_file(path), artifact["sha256"])

    def test_cli_dry_run_prints_plan_without_starting_or_publishing(self) -> None:
        request = self.request("model_generation", [self.add_input("input_image", "inputs/dry-run.png")])
        request_path = self.root / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = stage_adapter.main([
                "--request", str(request_path), "--artifact-root", str(self.root), "--dry-run",
            ])
        self.assertEqual(code, 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(Path(plan["commands"][0][1]).name, "generate_trellis_asset.py")
        self.assertFalse((self.root / request["outputStagingDir"]).exists())

    def test_cli_contract_failure_still_writes_identity_bound_result(self) -> None:
        request = self.request("model_generation", [self.add_input("input_image", "inputs/bad-config.png")])
        request["effectiveConfigSha256"] = "0" * 64
        request_path = self.root / "bad-request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = stage_adapter.main(["--request", str(request_path), "--artifact-root", str(self.root)])
        self.assertEqual(code, 0)
        result_path = self.root / request["outputStagingDir"] / "stage_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "failed_terminal")
        self.assertEqual(result["errorCode"], "config_hash_mismatch")
        for field in ("batchId", "projectId", "taskId", "stage", "contractVersion", "attempt", "leaseToken"):
            self.assertEqual(result[field], request[field])

    def test_identity_path_config_and_input_hash_are_rejected(self) -> None:
        request = self.request("model_generation", [self.add_input("input_image", "inputs/edited.png")])
        request["taskId"] = "Task_006"
        with self.assertRaisesRegex(stage_adapter.AdapterError, "Task_001"):
            stage_adapter.validate_request(request)
        request["taskId"] = "Task_001"
        request["outputStagingDir"] = "../escape"
        with self.assertRaisesRegex(stage_adapter.AdapterError, "相对路径"):
            stage_adapter.validate_request(request)
        request["outputStagingDir"] = "stages/model/output.staging"
        request["effectiveConfigSha256"] = "0" * 64
        with self.assertRaisesRegex(stage_adapter.AdapterError, "不一致"):
            stage_adapter.validate_request(request)
        request["effectiveConfigSha256"] = stage_adapter.canonical_sha256(request["effectiveConfig"])
        request["inputs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(stage_adapter.AdapterError, "哈希不一致"):
            stage_adapter.resolve_inputs(stage_adapter.validate_request(request), self.root)

        symlink = self.root / "inputs" / "linked.png"
        symlink.symlink_to(self.root / "inputs" / "edited.png")
        request = self.request(
            "model_generation",
            [{"artifactId": "input_image", "path": "inputs/linked.png", "sha256": stage_adapter.sha256_file(symlink)}],
        )
        with self.assertRaisesRegex(stage_adapter.AdapterError, "符号链接"):
            stage_adapter.resolve_inputs(stage_adapter.validate_request(request), self.root)

    def test_algorithm_contracts_reject_drift_before_launch(self) -> None:
        request = self.request(
            "model_generation", [self.add_input("input_image", "inputs/model.png")],
            {"noGlb": True, "requireGlb": True},
        )
        with self.assertRaisesRegex(stage_adapter.AdapterError, "不能同时"):
            self.plan(request)

        request = self.request(
            "render_alignment_views", [self.add_input("sample_ply", "inputs/render.ply")],
            {"near": 2.0, "far": 1.0},
        )
        with self.assertRaisesRegex(stage_adapter.AdapterError, "far"):
            self.plan(request)

        request = self.request(
            "model_generation", [self.add_input("input_image", "inputs/timeout.png")],
            {"timeoutSeconds": 0},
        )
        with self.assertRaisesRegex(stage_adapter.AdapterError, "timeoutSeconds"):
            self.plan(request)

        sags_inputs = [self.add_input("model_cfg_args", "inputs/drift-model/cfg_args")]
        annotations = []
        for index, name in enumerate(("center", "ring_060", "ring_120", "ring_180", "ring_240", "wrong")):
            mask_id, points_id = f"drift_mask_{index}", f"drift_points_{index}"
            sags_inputs.extend((
                self.add_input(mask_id, f"inputs/drift-annotations/{name}/mask.png"),
                self.add_input(points_id, f"inputs/drift-annotations/{name}/points.json", b"[]"),
            ))
            annotations.append({"name": name, "maskArtifactId": mask_id, "pointsArtifactId": points_id})
        request = self.request("sags_segment_vote", sags_inputs, {"annotations": annotations})
        with self.assertRaisesRegex(stage_adapter.AdapterError, "SAGS 六视角"):
            self.plan(request)

    def test_nonzero_timeout_cancel_and_missing_artifact_results(self) -> None:
        scenarios = (
            ("fail", {}, None, "failed_retryable", "worker_exit_nonzero"),
            ("sleep", {"timeoutSeconds": 0.05}, None, "failed_retryable", "worker_timeout"),
            ("missing", {}, None, "failed_terminal", "missing_artifact"),
        )
        for index, (mode, options, cancel_file, status, code) in enumerate(scenarios):
            with self.subTest(mode=mode):
                request = self.request(
                    "model_generation", [self.add_input("input_image", f"inputs/edited-{index}.png")], options,
                )
                request["outputStagingDir"] = f"stages/scenario-{index}/output.staging"
                result, _ = stage_adapter.execute_request(
                    stage_adapter.validate_request(request), self.root,
                    fake_command=[sys.executable, str(self.fake), mode], cancel_file=cancel_file, poll_seconds=0.01,
                )
                self.assertEqual(result["status"], status)
                self.assertEqual(result["errorCode"], code)
                self.assertTrue(result["cleanup"]["completed"])

        cancel = self.root / "cancel.requested"
        request = self.request(
            "model_generation", [self.add_input("input_image", "inputs/edited-cancel.png")],
        )
        request["outputStagingDir"] = "stages/cancel/output.staging"
        thread = threading.Thread(target=lambda: (time.sleep(0.05), cancel.write_text("cancel", encoding="utf-8")))
        thread.start()
        try:
            result, _ = stage_adapter.execute_request(
                stage_adapter.validate_request(request), self.root,
                fake_command=[sys.executable, str(self.fake), "sleep"], cancel_file=cancel, poll_seconds=0.01,
            )
        finally:
            thread.join()
        self.assertEqual(result["status"], "canceled")
        self.assertEqual(result["errorCode"], "canceled")
        self.assertTrue(result["cleanup"]["completed"])

    def test_existing_staging_files_are_never_reused_as_success(self) -> None:
        request = self.request("model_generation", [self.add_input("input_image", "inputs/stale.png")])
        output = self.root / request["outputStagingDir"]
        output.mkdir(parents=True)
        (output / "sample.ply").write_bytes(b"stale")
        with self.assertRaisesRegex(stage_adapter.AdapterError, "必须为空"):
            stage_adapter.execute_request(
                stage_adapter.validate_request(request), self.root,
                fake_command=[sys.executable, str(self.fake), "missing"], poll_seconds=0.01,
            )

    def test_input_replacement_during_execution_fails_hash_recheck(self) -> None:
        input_record = self.add_input("input_image", "inputs/mutable.png")
        request = self.request("model_generation", [input_record])
        result, _ = stage_adapter.execute_request(
            stage_adapter.validate_request(request), self.root,
            fake_command=[sys.executable, str(self.fake), "mutate", str(self.root / input_record["path"])],
            poll_seconds=0.01,
        )
        self.assertEqual(result["status"], "failed_terminal")
        self.assertEqual(result["errorCode"], "input_hash_mismatch")

    @unittest.skipIf(os.name == "nt", "远端适配器的进程组测试只适用于 Linux")
    def test_sigterm_cleans_worker_process_group_and_writes_canceled_result(self) -> None:
        request = self.request("model_generation", [self.add_input("input_image", "inputs/signal.png")])
        request["outputStagingDir"] = "stages/signal/output.staging"
        request_path = self.root / "signal-request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        command = [
            sys.executable, str(Path(stage_adapter.__file__)),
            "--request", str(request_path), "--artifact-root", str(self.root),
            "--fake-command-json", json.dumps([sys.executable, str(self.fake), "sleep"]),
            "--poll-seconds", "0.01",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        worker_pid_path = self.root / request["outputStagingDir"] / "_worker.pid"
        deadline = time.monotonic() + 3
        while not worker_pid_path.is_file() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(worker_pid_path.is_file())
        worker_pid = int(worker_pid_path.read_text(encoding="ascii"))
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=6)
        self.assertEqual(process.returncode, 0, stderr)
        result = json.loads((self.root / request["outputStagingDir"] / "stage_result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "canceled")
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)

    def test_pose_rejection_and_debug_facade_gate_are_explicit(self) -> None:
        debug = self.request(
            "debug_bundle", [self.add_input("batch_manifest", "inputs/debug/batch_manifest.json", b"{}")],
        )
        with self.assertRaisesRegex(stage_adapter.AdapterError, "compatibility facade"):
            self.plan(debug)

        pose_inputs = [
            self.add_input("generated_cameras", "inputs/reject/cameras.txt"),
            self.add_input("generated_images", "inputs/reject/images.txt"),
        ]
        pose_views = []
        for name in ("left", "center", "right"):
            ids = {
                "matchesArtifactId": f"reject_matches_{name}",
                "sceneDepthArtifactId": f"reject_scene_depth_{name}",
                "sceneCameraArtifactId": f"reject_scene_camera_{name}",
                "generatedDepthArtifactId": f"reject_generated_depth_{name}",
            }
            pose_inputs.extend((
                self.add_input(ids["matchesArtifactId"], f"inputs/reject/{name}/matches.json", b"{}"),
                self.add_input(ids["sceneDepthArtifactId"], f"inputs/reject/{name}.raw"),
                self.add_input(ids["sceneCameraArtifactId"], f"inputs/reject/{name}.camera.json", b"{}"),
                self.add_input(ids["generatedDepthArtifactId"], f"inputs/reject/generated/{name}.npy"),
            ))
            pose_views.append({"name": name, **ids})
        pose = self.request("estimate_pose", pose_inputs, {"views": pose_views})
        checked = stage_adapter.validate_request(pose)
        output = self.root / pose["outputStagingDir"]
        inputs = stage_adapter.resolve_inputs(checked, self.root)
        plan = stage_adapter.build_plan(checked, inputs, output)
        output.mkdir(parents=True, exist_ok=True)
        for relative in plan.required_outputs:
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            value = {"status": "rejected"} if relative == "pose.json" else {"ok": True}
            path.write_text(json.dumps(value), encoding="utf-8")
        status, code, _ = stage_adapter._inspect_outputs(plan, output)
        self.assertEqual((status, code), ("rejected", "pose_quality_rejected"))


if __name__ == "__main__":
    unittest.main()
