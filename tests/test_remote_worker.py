from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from insertany3d.contracts import validate_stage_result
from insertany3d.contracts.models import canonical_sha256
from insertany3d.remote_worker import (
    CommandOutcome,
    RemoteCommandBuilder,
    RemoteProfile,
    RemoteStageRunner,
    RemoteWorkerError,
    build_remote_attempt_plan,
    build_remote_worker_command,
    _parse_remote_state,
    verify_remote_runtime,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _request(root: Path, *, stage: str = "model_generation") -> tuple[Path, dict]:
    data = b"edited-image"
    relative = "artifacts/image_edit/key/edited.png"
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    config = {"stageOptions": {"provider": "trellis"}}
    value = {
        "schemaVersion": 1,
        "kind": "insertany3d.stage-request",
        "batchId": "batch_001",
        "projectId": "Farm_Test_001",
        "taskId": "Task_001",
        "stage": stage,
        "contractVersion": {
            "upload_inputs": "upload-inputs-v1",
            "download_results": "download-results-v1",
        }.get(stage, "model-generation-v1"),
        "attempt": 1,
        "leaseToken": "lease-token-001",
        "inputs": [{"artifactId": "input_image", "type": "edited_image", "path": relative, "sha256": _sha(data), "size": len(data)}],
        "effectiveConfig": config,
        "effectiveConfigSha256": canonical_sha256(config),
        "outputStagingDir": f"Farm_Test_001/Task_001/stages/{stage}/attempt-0001/output.staging",
    }
    path = root / "request.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def _profile(**overrides) -> RemoteProfile:
    values = {
        "target": "worker@example.test",
        "port": 25367,
        "project_root": "/srv/insertany3d",
        "artifact_root": "/srv/insertany3d-runs/batch_001",
        "connect_timeout_seconds": 3,
        "poll_interval_seconds": 0.001,
    }
    values.update(overrides)
    return RemoteProfile(**values)


class ScriptedRunner:
    def __init__(self, states: Sequence[CommandOutcome], *, download_result: dict | None = None, artifact: bytes = b"ply"):
        self.states = list(states)
        self.download_result = download_result
        self.artifact = artifact
        self.commands: list[list[str]] = []
        self.timeouts: list[float | None] = []

    def run(self, command: Sequence[str], *, timeout_seconds: float | None = None) -> CommandOutcome:
        command = list(command)
        self.commands.append(command)
        self.timeouts.append(timeout_seconds)
        if command[0] == "scp" and "-r" in command:
            destination = Path(command[-1])
            destination.mkdir(parents=True)
            if self.download_result is not None:
                artifact_path = destination / "sample.ply"
                artifact_path.write_bytes(self.artifact)
                (destination / "stage_result.json").write_text(json.dumps(self.download_result), encoding="utf-8")
            return CommandOutcome(0)
        # Prepare, each upload, and each hash+move operation succeed.  Only
        # start/probe commands contain the stable RESULT/RUNNING/etc marker.
        if command[0] == "ssh" and "nohup" in command[-1]:
            return self.states.pop(0)
        if command[0] == "ssh" and "kill -0" in command[-1] and "nohup" not in command[-1]:
            return self.states.pop(0)
        return CommandOutcome(0)


def _remote_result(request: dict, artifact: bytes = b"ply", **changes) -> dict:
    value = {
        "schemaVersion": 1,
        "kind": "insertany3d.stage-result",
        "batchId": request["batchId"],
        "projectId": request["projectId"],
        "taskId": request["taskId"],
        "stage": request["stage"],
        "contractVersion": request["contractVersion"],
        "attempt": request["attempt"],
        "leaseToken": request["leaseToken"],
        "status": "succeeded",
        "artifacts": [{"artifactId": "sample_ply", "type": "gaussian_ply", "path": "sample.ply", "sha256": _sha(artifact), "size": len(artifact)}],
        "diagnosticPaths": [],
        "cleanup": {"completed": True},
        "finishedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    value.update(changes)
    return value


class RemoteWorkerTests(unittest.TestCase):
    def test_runtime_preflight_rejects_missing_or_stale_server_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            tools = repository / "tools"
            tools.mkdir(parents=True)
            adapter = tools / "stage_adapter.py"
            adapter.write_text("value = 1\n", encoding="utf-8")
            lock = tools / "remote_runtime.lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "insertany3d.remote-runtime-lock",
                        "sourceAuthority": "codex_remote_tools",
                        "files": [
                            {
                                "path": "stage_adapter.py",
                                "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                                "size": adapter.stat().st_size,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            class RuntimeProbe:
                def __init__(self, stdout: str):
                    self.stdout = stdout
                    self.commands: list[tuple[str, ...]] = []

                def run(self, command: Sequence[str], *, timeout_seconds: float | None = None) -> CommandOutcome:
                    del timeout_seconds
                    self.commands.append(tuple(command))
                    return CommandOutcome(0, self.stdout)

            digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
            probe = RuntimeProbe(
                "CONFIG\tOK\n"
                f"FILE\t{digest}\t{adapter.stat().st_size}\tstage_adapter.py\n"
            )
            verified = verify_remote_runtime(
                _profile(),
                lock_path=lock,
                command_runner=probe,
            )
            self.assertEqual(verified["files"], 1)
            self.assertTrue(probe.commands)
            self.assertIn("sha256sum", probe.commands[0][-1])

            noisy_probe = RuntimeProbe(
                "BASH=/usr/bin/bash\n"
                "CONFIG\tOK\n"
                f"FILE\t{digest}\t{adapter.stat().st_size}\tstage_adapter.py\n"
            )
            noisy_verified = verify_remote_runtime(
                _profile(),
                lock_path=lock,
                command_runner=noisy_probe,
            )
            self.assertEqual(noisy_verified["files"], 1)

            invalid_config = RuntimeProbe(
                "CONFIG\tINVALID\n"
                f"FILE\t{digest}\t{adapter.stat().st_size}\tstage_adapter.py\n"
            )
            with self.assertRaisesRegex(RemoteWorkerError, "私有运行配置"):
                verify_remote_runtime(
                    _profile(),
                    lock_path=lock,
                    command_runner=invalid_config,
                )

            self.assertEqual(_parse_remote_state("BASH=/usr/bin/bash\nSTARTED 123 123\n"), "STARTED")
            self.assertEqual(_parse_remote_state("BASH=/usr/bin/bash\nRUNNING\n"), "RUNNING")

            stale = RuntimeProbe(
                "CONFIG\tOK\n"
                f"FILE\t{'0' * 64}\t{adapter.stat().st_size}\tstage_adapter.py\n"
            )
            with self.assertRaisesRegex(RemoteWorkerError, "缺失或与锁文件不一致"):
                verify_remote_runtime(
                    _profile(),
                    lock_path=lock,
                    command_runner=stale,
                )

            missing = RuntimeProbe(
                "CONFIG\tOK\nMISSING\t-\t-\tstage_adapter.py\n"
            )
            with self.assertRaisesRegex(RemoteWorkerError, "stage_adapter.py"):
                verify_remote_runtime(
                    _profile(),
                    lock_path=lock,
                    command_runner=missing,
                )

    def test_upload_and_download_boundaries_publish_verified_receipts(self) -> None:
        for stage, transfer_mode in (
            ("upload_inputs", "atomic_scp_upload"),
            ("download_results", "per_stage_eager_download"),
        ):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                request_path, _ = _request(root, stage=stage)
                command_runner = ScriptedRunner([])
                report = RemoteStageRunner(
                    _profile(), command_runner=command_runner, sleep=lambda _: None
                ).run(request_path, root)
                result = validate_stage_result(
                    json.loads(report.result_path.read_text(encoding="utf-8"))
                )
                self.assertEqual(result["status"], "succeeded")
                receipt = json.loads(
                    report.result_path.with_name("transfer_receipt.json").read_text(encoding="utf-8")
                )
                self.assertEqual(receipt["transferMode"], transfer_mode)
                self.assertEqual(receipt["verifiedInputs"][0]["artifactId"], "input_image")
                if stage == "upload_inputs":
                    self.assertTrue(any(command[0] == "scp" for command in command_runner.commands))
                else:
                    self.assertEqual(command_runner.commands, [])

    def test_remote_roots_are_absolute_specific_and_shell_safe(self) -> None:
        for invalid in ("relative/path", "/", "/root/../tmp", "/path with space"):
            with self.subTest(invalid=invalid), self.assertRaises(RemoteWorkerError):
                _profile(artifact_root=invalid)

    def test_commands_preserve_input_relative_path_and_use_detached_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, request = _request(root)
            profile = _profile()
            plan = build_remote_attempt_plan(profile, request_path, root)
            builder = RemoteCommandBuilder(profile)
            remote_input = f"{profile.artifact_root}/{request['inputs'][0]['path']}"
            upload = builder.scp_upload(root / request["inputs"][0]["path"], remote_input + ".tmp")
            start = builder.start_or_probe(plan)
            self.assertEqual(upload[-1], f"{profile.target}:{remote_input}.tmp")
            self.assertIn("nohup", start[-1])
            self.assertIn("setsid", start[-1])
            self.assertIn("stage_adapter.py", start[-1])
            self.assertIn(profile.environment_file, start[-1])
            self.assertLess(start[-1].index("set -a; ."), start[-1].index("nohup"))
            self.assertIn(plan.remote_pid_path, start[-1])
            self.assertIn("saved_pgid", builder.probe(plan)[-1])
            self.assertIn("group_live", builder.probe(plan)[-1])
            self.assertIn(plan.owner_id, plan.remote_control_dir)
            wrapper = build_remote_worker_command(request_path, root, profile, local_python="python-test")
            self.assertEqual(wrapper[:3], ["python-test", "-m", "insertany3d.remote_worker"])
            self.assertIn(profile.artifact_root, wrapper)

    def test_success_downloads_and_revalidates_result_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, request = _request(root)
            result = _remote_result(request)
            runner = ScriptedRunner(
                [CommandOutcome(0, "STARTED 123\n"), CommandOutcome(0, "RUNNING\n"), CommandOutcome(0, "RESULT\n")],
                download_result=result,
            )
            report = RemoteStageRunner(_profile(), command_runner=runner, sleep=lambda _: None).run(request_path, root)
            self.assertEqual(report.classification, "result_downloaded")
            downloaded = validate_stage_result(json.loads(report.result_path.read_text(encoding="utf-8")))
            self.assertEqual(downloaded["status"], "succeeded")
            self.assertEqual((report.result_path.parent / "sample.ply").read_bytes(), b"ply")
            scp_timeouts = [
                timeout
                for command, timeout in zip(runner.commands, runner.timeouts)
                if command[0] == "scp"
            ]
            self.assertTrue(scp_timeouts)
            self.assertTrue(all(timeout is None for timeout in scp_timeouts))

    def test_start_disconnect_is_delivery_unknown_with_recovery_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, _ = _request(root)
            runner = ScriptedRunner([CommandOutcome(255, "", "connection reset")])
            report = RemoteStageRunner(_profile(), command_runner=runner).run(request_path, root)
            result = validate_stage_result(json.loads(report.result_path.read_text(encoding="utf-8")))
            self.assertEqual(report.classification, "delivery_unknown")
            self.assertEqual(result["errorCode"], "delivery_unknown")
            self.assertFalse(result["cleanup"]["completed"])
            self.assertIn("不得自动重启", result["message"])
            self.assertIn("kill -0", " ".join(report.recovery_command))

    def test_poll_disconnect_records_remote_status_unknown_without_duplicate_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, _ = _request(root)
            runner = ScriptedRunner([CommandOutcome(0, "STARTED 123\n"), CommandOutcome(255, "", "network down")])
            report = RemoteStageRunner(_profile(), command_runner=runner, sleep=lambda _: None).run(request_path, root)
            result = validate_stage_result(json.loads(report.result_path.read_text(encoding="utf-8")))
            self.assertEqual(report.classification, "remote_status_unknown")
            self.assertEqual(result["errorCode"], "delivery_unknown")
            self.assertFalse(result["cleanup"]["completed"])
            diagnostic = json.loads((report.result_path.parent / "_remote" / "transport.json").read_text(encoding="utf-8"))
            self.assertEqual(diagnostic["classification"], "remote_status_unknown")
            self.assertEqual(sum("nohup" in command[-1] for command in runner.commands), 1)

    def test_exited_top_process_without_result_remains_delivery_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, _ = _request(root)
            runner = ScriptedRunner([CommandOutcome(0, "EXITED\n")])
            report = RemoteStageRunner(_profile(), command_runner=runner).run(request_path, root)
            result = validate_stage_result(json.loads(report.result_path.read_text(encoding="utf-8")))
            self.assertEqual(report.classification, "remote_status_unknown")
            self.assertEqual(result["errorCode"], "delivery_unknown")
            self.assertFalse(result["cleanup"]["completed"])

    def test_group_running_never_becomes_retryable_exited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, _ = _request(root)
            runner = ScriptedRunner([CommandOutcome(0, "GROUP_RUNNING\n")])
            report = RemoteStageRunner(_profile(), command_runner=runner).run(request_path, root)
            result = validate_stage_result(json.loads(report.result_path.read_text(encoding="utf-8")))
            self.assertEqual(report.classification, "remote_status_unknown")
            self.assertEqual(result["errorCode"], "delivery_unknown")
            self.assertFalse(result["cleanup"]["completed"])

    def test_explicit_remote_cleanup_command_requires_verified_leader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, _ = _request(root)
            runner = ScriptedRunner([CommandOutcome(0, "DESCENDANTS_UNKNOWN\n")])
            report = RemoteStageRunner(_profile(), command_runner=runner).cancel_existing(request_path, root)
            self.assertEqual(report.classification, "remote_cleanup_incomplete")
            self.assertEqual(report.remote_state, "DESCENDANTS_UNKNOWN")
            script = next(command[-1] for command in runner.commands if command[0] == "ssh")
            self.assertIn("kill -TERM -- \"-$saved_pgid\"", script)
            self.assertIn("GROUP_REMAINS", script)
            self.assertIn("DESCENDANTS_UNKNOWN", script)
            self.assertNotIn("printf 'CLEANED", script)

    def test_invalid_remote_identity_is_quarantined_and_becomes_terminal_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, request = _request(root)
            invalid = _remote_result(request, leaseToken="wrong-lease")
            runner = ScriptedRunner([CommandOutcome(0, "RESULT\n")], download_result=invalid)
            report = RemoteStageRunner(_profile(), command_runner=runner).run(request_path, root)
            result = validate_stage_result(json.loads(report.result_path.read_text(encoding="utf-8")))
            self.assertEqual(report.classification, "remote_contract_invalid")
            self.assertEqual(result["status"], "failed_terminal")
            self.assertEqual(result["errorCode"], "remote_contract_invalid")
            self.assertTrue(any(root.glob("**/*.invalid")))

    def test_upload_failure_is_retryable_before_remote_start(self) -> None:
        class FailingUpload:
            def __init__(self):
                self.commands: list[list[str]] = []

            def run(self, command: Sequence[str], *, timeout_seconds: float | None = None) -> CommandOutcome:
                command = list(command)
                self.commands.append(command)
                if command[0] == "scp":
                    return CommandOutcome(1, "", "upload failed")
                return CommandOutcome(0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, _ = _request(root)
            runner = FailingUpload()
            report = RemoteStageRunner(_profile(), command_runner=runner).run(request_path, root)
            result = validate_stage_result(json.loads(report.result_path.read_text(encoding="utf-8")))
            self.assertEqual(result["status"], "failed_retryable")
            self.assertEqual(result["errorCode"], "transient_network")
            self.assertFalse(any("nohup" in command[-1] for command in runner.commands))


if __name__ == "__main__":
    unittest.main()
