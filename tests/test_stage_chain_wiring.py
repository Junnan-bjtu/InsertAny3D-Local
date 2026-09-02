from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from insertany3d.scheduler import BatchController, WorkItem, default_capacities
from insertany3d.store import SchedulerStore
from tests.fixtures import batch_manifest
from tools import stage_adapter


RING_VIEWS = ("center", "ring_060", "ring_120", "ring_180", "ring_240", "ring_300")


def _payload(relative: str, *, pose_ready: bool = False) -> bytes:
    if relative.endswith(".json"):
        value = {"status": "ready"} if pose_ready and relative == "pose.json" else {"fixture": True}
        return json.dumps(value).encode("utf-8")
    return f"fixture:{relative}".encode("utf-8")


def _commit_files(
    controller: BatchController,
    item: WorkItem,
    files: Mapping[str, bytes],
) -> None:
    artifacts = []
    for index, (relative, content) in enumerate(files.items()):
        path = item.staging_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts.append(
            {
                "artifactId": f"fixture_{index:03d}",
                "type": "stage_output",
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    controller.commit_success(item, artifacts)


class StageChainWiringTests(unittest.TestCase):
    def test_manifest_results_feed_ring6_sags_and_atomic_debug_requests(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_stage_chain_") as temporary:
            root = Path(temporary)
            store = SchedulerStore(root / "state.sqlite3")
            controller = BatchController(store, lease_seconds=30)
            manifest = batch_manifest(mode="automatic", project_count=1)
            manifest["projects"][0]["tasks"] = manifest["projects"][0]["tasks"][:1]
            manifest["projects"][0]["tasks"][0]["num_gen_image_per_task"] = 1
            run_root = root / "runs"
            controller.plan(manifest, run_root, formal=False)
            controller.start(manifest["batchId"], formal=False)
            capacities = default_capacities(manifest)

            def lease(stage: str) -> WorkItem:
                item = controller.lease_next(
                    manifest["batchId"], f"fixture-{stage}", capacities,
                    project_id="Scene_01", task_id="Task_001", stage_name=stage,
                )
                self.assertIsNotNone(item, stage)
                return item  # type: ignore[return-value]

            try:
                anchor = lease("unity_anchor")
                anchor_files = {
                    "run_manifest.json": _payload("run_manifest.json"),
                    "Task_001/task_manifest.json": _payload("Task_001/task_manifest.json"),
                }
                for view in ("left", "center", "right"):
                    for name in ("image.png", "image.raw", "image.camera.json"):
                        relative = f"Task_001/inputs/unity/{view}/{name}"
                        anchor_files[relative] = _payload(relative)
                _commit_files(controller, anchor, anchor_files)

                edit = lease("image_edit")
                _commit_files(controller, edit, {"edited.png": b"edited-image"})
                upload = lease("upload_inputs")
                _commit_files(controller, upload, {"transfer_receipt.json": _payload("transfer_receipt.json")})

                model = lease("model_generation")
                model_request = controller.build_stage_request(model)
                self.assertIn("input_image", {item["artifactId"] for item in model_request["inputs"]})
                _commit_files(
                    controller,
                    model,
                    {"sample.ply": b"ply", "manifest.json": _payload("manifest.json")},
                )

                render = lease("render_alignment_views")
                render_request = controller.build_stage_request(render)
                render_options = render_request["effectiveConfig"]["stageOptions"]
                self.assertEqual(render_options["yawOffsets"], [-24, 0, 24])
                self.assertEqual(render_options["ringViewNames"], list(RING_VIEWS))
                render_inputs = stage_adapter.resolve_inputs(render_request, run_root.resolve())
                render_plan = stage_adapter.build_plan(render_request, render_inputs, render.staging_dir)
                self.assertEqual(len(render_plan.commands), 2)
                render_files = {relative: _payload(relative) for relative in render_plan.required_outputs}
                for view in ("left", "center", "right"):
                    relative = f"source/depths/absdepth/{view}.raw"
                    render_files[relative] = _payload(relative)
                render_files["ring6/model/point_cloud/iteration_0/point_cloud.ply"] = b"gaussians"
                _commit_files(controller, render, render_files)

                segment = lease("segment_inputs")
                segment_request = controller.build_stage_request(segment)
                segment_options = segment_request["effectiveConfig"]["stageOptions"]
                self.assertEqual([item["name"] for item in segment_options["sagsViews"]], list(RING_VIEWS))
                self.assertEqual(segment_options["sagsTaskPrompt"], "chair")
                segment_inputs = stage_adapter.resolve_inputs(segment_request, run_root.resolve())
                segment_plan = stage_adapter.build_plan(segment_request, segment_inputs, segment.staging_dir)
                self.assertEqual(len(segment_plan.commands), 7)
                self.assertEqual(Path(segment_plan.commands[0][1]).name, "segment_anchor_views.py")
                self.assertTrue(
                    all(Path(command[1]).name == "auto_segment.py" for command in segment_plan.commands[1:])
                )
                _commit_files(
                    controller,
                    segment,
                    {relative: _payload(relative) for relative in segment_plan.required_outputs},
                )

                gim = lease("gim_match")
                gim_request = controller.build_stage_request(gim)
                gim_inputs = stage_adapter.resolve_inputs(gim_request, run_root.resolve())
                gim_plan = stage_adapter.build_plan(gim_request, gim_inputs, gim.staging_dir)
                self.assertEqual(len(gim_plan.commands), 3)
                _commit_files(controller, gim, {relative: _payload(relative) for relative in gim_plan.required_outputs})

                pose = lease("estimate_pose")
                pose_request = controller.build_stage_request(pose)
                self.assertEqual(
                    {item["name"] for item in pose_request["effectiveConfig"]["stageOptions"]["views"]},
                    {"left", "center", "right"},
                )
                pose_inputs = stage_adapter.resolve_inputs(pose_request, run_root.resolve())
                pose_plan = stage_adapter.build_plan(pose_request, pose_inputs, pose.staging_dir)
                _commit_files(
                    controller,
                    pose,
                    {
                        relative: _payload(relative, pose_ready=True)
                        for relative in pose_plan.required_outputs
                    },
                )

                sags = lease("sags_segment_vote")
                sags_request = controller.build_stage_request(sags)
                sags_options = sags_request["effectiveConfig"]["stageOptions"]
                self.assertEqual(sags_options["modelMarkerArtifactId"], "sags_model_cfg_args")
                self.assertEqual([item["name"] for item in sags_options["annotations"]], list(RING_VIEWS))
                sags_inputs = stage_adapter.resolve_inputs(sags_request, run_root.resolve())
                sags_plan = stage_adapter.build_plan(sags_request, sags_inputs, sags.staging_dir)
                model_dir_index = sags_plan.commands[0].index("--model-dir") + 1
                self.assertIn("ring6/model", sags_plan.commands[0][model_dir_index].replace("\\", "/"))
                _commit_files(controller, sags, {relative: _payload(relative) for relative in sags_plan.required_outputs})

                debug = lease("debug_bundle")
                debug_request = controller.build_stage_request(debug)
                self.assertEqual(debug_request["effectiveConfig"]["stageOptions"]["mode"], "atomic")
                self.assertNotIn("batchManifestArtifactId", debug_request["effectiveConfig"]["stageOptions"])
                result, debug_plan = stage_adapter.execute_request(debug_request, run_root.resolve())
                self.assertTrue(debug_plan.atomic_bundle)
                self.assertEqual(result["status"], "succeeded")
                controller.commit_success(debug, result["artifacts"])

                index_path = debug.output_dir / "Task_001" / "artifact_index.json"
                index = json.loads(index_path.read_text(encoding="utf-8"))
                self.assertEqual(index["artifactCount"], len(debug_request["inputs"]))
                self.assertEqual(
                    {item["sourcePath"] for item in index["artifacts"]},
                    {item["path"] for item in debug_request["inputs"]},
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
