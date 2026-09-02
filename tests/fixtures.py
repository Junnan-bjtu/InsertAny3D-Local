from __future__ import annotations

from typing import Any

from insertany3d.contracts.models import EVAL6_VIEW_LAYOUT, canonical_sha256


SHA = "a" * 64


def batch_manifest(*, mode: str = "automatic", project_count: int = 12) -> dict[str, Any]:
    projects = []
    for project_index in range(project_count):
        projects.append(
            {
                "projectId": f"Scene_{project_index + 1:02d}",
                "projectPath": f"/private/Scene_{project_index + 1:02d}",
                "scenePath": f"Assets/Scene_{project_index + 1:02d}.unity",
                "unityVersion": "2022.3.55f1c1",
                "manifestSha256": SHA,
                "packagesLockSha256": SHA,
                "tasks": [
                    {"taskId": f"Task_{task_index:03d}", "objectPrompt": "chair", "editPrompt": "add chair"}
                    for task_index in range(1, 6)
                ],
            }
        )
    return {
        "schemaVersion": 1,
        "kind": "insertany3d.batch",
        "batchId": "batch_test",
        "defaultsRef": "codex_ops/insert_workflow.defaults.json",
        "renderProfile": "eval6",
        "renderProtocol": {"protocol": "eval6-v1", "viewYawOffsetDegrees": 24},
        "resources": {"unitySlots": 1, "editSlotsInitial": 4, "editSlotsMax": 24, "remoteGpuPool": [1, 3]},
        "editPolicy": {"mode": mode, "reviewBatchSize": 5, "policyVersion": "mechanical-v1"},
        "remoteProfile": "server-default",
        "projects": projects,
        "pins": {
            "insertAny3dCommit": SHA,
            "unityPackageCommit": SHA,
            "defaultsSha256": SHA,
            "evaluatorCommit": SHA,
            "evaluatorModel": "fake-gpteval",
            "evaluatorRubricSha256": SHA,
            "submodules": {"TRELLIS": SHA},
        },
    }


def evaluation_manifest() -> dict[str, Any]:
    yaw = 24
    config = {"pitchDegrees": [10, 40], "yawOffsetDegrees": yaw}
    views = []
    for view_id, pitch, direction in EVAL6_VIEW_LAYOUT:
        views.append(
            {
                "viewId": view_id,
                "pitchDegrees": pitch,
                "yawOffsetDegrees": direction * yaw,
                "original": {"path": f"original/{view_id}.png", "sha256": SHA},
                "inserted": {"path": f"inserted/{view_id}.png", "sha256": SHA},
                "camera": {"path": f"cameras/{view_id}.camera.json", "sha256": SHA},
            }
        )
    return {
        "schemaVersion": 1,
        "kind": "insertany3d.evaluation",
        "protocol": "eval6-v1",
        "batchId": "batch_test",
        "projectId": "Scene_01",
        "scenePath": "Assets/Scene_01.unity",
        "taskId": "Task_001",
        "runId": "run_001",
        "methodId": "insertany3d-main",
        "taskPrompt": "add chair",
        "viewConfig": {**config, "sha256": canonical_sha256(config)},
        "render": {"width": 1024, "height": 1024, "cameraConvention": "unity-c2w-v1"},
        "views": views,
    }
