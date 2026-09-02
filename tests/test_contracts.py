from __future__ import annotations

import copy
import json
import unittest

from insertany3d.contracts import (
    ContractError,
    schema_path,
    validate_batch_manifest,
    validate_evaluation_manifest,
    validate_heartbeat,
    validate_stage_request,
    validate_stage_result,
)
from insertany3d.contracts.models import canonical_sha256
from tests.fixtures import SHA, batch_manifest, evaluation_manifest


class ContractTests(unittest.TestCase):
    def test_schema_files_are_installed_json(self) -> None:
        for name in ("batch", "stage-request", "stage-result", "heartbeat", "evaluation", "edit-review"):
            value = json.loads(schema_path(name).read_text(encoding="utf-8"))
            self.assertTrue(value["$schema"].endswith("2020-12/schema"))

    def test_formal_batch_requires_explicit_12_by_5_and_pins(self) -> None:
        value = validate_batch_manifest(batch_manifest(), formal=True)
        self.assertEqual(len(value["projects"]), 12)
        self.assertEqual(sum(len(project["tasks"]) for project in value["projects"]), 60)
        invalid = batch_manifest(project_count=11)
        with self.assertRaisesRegex(ContractError, "12 个 Project"):
            validate_batch_manifest(invalid, formal=True)
        validate_batch_manifest(invalid, formal=False)

    def test_manual_review_defaults_are_applied(self) -> None:
        value = batch_manifest(mode="manual")
        value["editPolicy"] = {}
        value["renderProtocol"].pop("viewYawOffsetDegrees")
        normalized = validate_batch_manifest(value, formal=True)
        self.assertEqual(normalized["editPolicy"], {"mode": "manual", "reviewBatchSize": 5})
        self.assertEqual(normalized["renderProtocol"]["viewYawOffsetDegrees"], 24)

    def test_image_api_policy_is_explicit_and_normalized(self) -> None:
        value = batch_manifest()
        value["resources"].pop("editSlotsInitial")
        value["resources"].pop("editSlotsMax")
        value["resources"]["imageApi"] = {"mode": "adaptive"}
        normalized = validate_batch_manifest(value, formal=False)
        self.assertEqual(normalized["resources"]["imageApi"]["initialLimit"], 3)
        self.assertEqual(normalized["resources"]["imageApi"]["maximumLimit"], 5)
        self.assertEqual(normalized["resources"]["editSlotsInitial"], 3)

    def test_formal_batch_rejects_empty_gpu_pool(self) -> None:
        value = batch_manifest()
        value["resources"]["remoteGpuPool"] = []
        with self.assertRaisesRegex(ContractError, "至少需要一个远端 GPU"):
            validate_batch_manifest(value, formal=True)

    def test_draft_batch_still_rejects_unknown_task_ids(self) -> None:
        value = batch_manifest(project_count=1)
        value["projects"][0]["tasks"] = ["Task_006"]
        with self.assertRaisesRegex(ContractError, "Task_001 至 Task_005"):
            validate_batch_manifest(value, formal=False)

    def test_stage_request_result_and_heartbeat_contracts(self) -> None:
        config = {"threshold": 0.25}
        request = {
        "schemaVersion": 1,
        "kind": "insertany3d.stage-request",
        "batchId": "batch_test",
        "projectId": "Scene_01",
        "taskId": "Task_001",
        "stage": "gim_match",
        "contractVersion": "gim-match-v1",
        "attempt": 1,
        "leaseToken": "lease",
        "inputs": [{"artifactId": "rgb", "path": "inputs/rgb.png", "sha256": SHA}],
        "effectiveConfig": config,
        "effectiveConfigSha256": canonical_sha256(config),
        "outputStagingDir": "stages/gim/attempt-0001/output.staging",
        }
        self.assertEqual(validate_stage_request(request)["stage"], "gim_match")
        request["effectiveConfigSha256"] = "b" * 64
        with self.assertRaisesRegex(ContractError, "不一致"):
            validate_stage_request(request)
        request["effectiveConfigSha256"] = canonical_sha256(config)
        request["outputStagingDir"] = "C:/outside/output.staging"
        with self.assertRaisesRegex(ContractError, "相对路径"):
            validate_stage_request(request)

        result = {
        "schemaVersion": 1,
        "kind": "insertany3d.stage-result",
        "batchId": "batch_test",
        "projectId": "Scene_01",
        "taskId": "Task_001",
        "stage": "gim_match",
        "contractVersion": "gim-match-v1",
        "attempt": 1,
        "leaseToken": "lease",
        "status": "succeeded",
        "artifacts": [{"artifactId": "matches", "type": "gim", "path": "matches.json", "sha256": SHA, "size": 1}],
        "diagnosticPaths": [],
        "cleanup": {"completed": True},
        "finishedAtUtc": "2026-08-29T12:00:00Z",
        }
        self.assertEqual(validate_stage_result(result)["status"], "succeeded")
        result.pop("contractVersion")
        with self.assertRaisesRegex(ContractError, "contractVersion"):
            validate_stage_result(result)

        heartbeat = {
        "schemaVersion": 1,
        "kind": "insertany3d.heartbeat",
        "leaseToken": "lease",
        "pid": 100,
        "pgid": 100,
        "hostBootId": "boot",
        "processStartTicks": 99,
        "progress": {"completed": 1, "total": 2},
        "logOffset": 0,
        "observedAtUtc": "2026-08-29T12:00:00Z",
        }
        self.assertEqual(validate_heartbeat(heartbeat)["pid"], 100)

    def test_eval6_contract_rejects_mixed_yaw(self) -> None:
        value = evaluation_manifest()
        self.assertEqual(len(validate_evaluation_manifest(value)["views"]), 6)
        invalid = copy.deepcopy(value)
        invalid["views"][0]["yawOffsetDegrees"] = -12
        with self.assertRaisesRegex(ContractError, "全局 yawOffsetDegrees"):
            validate_evaluation_manifest(invalid)


if __name__ == "__main__":
    unittest.main()
