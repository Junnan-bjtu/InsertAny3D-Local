from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from insertany3d.executors import ExecutionResult
from insertany3d.remote_worker import RemoteProfile
from insertany3d.runtime_workers import (
    CompositeStageExecutor,
    RemoteStageExecutor,
    RuntimeWorkerConfigurationError,
    load_environment_file,
    load_local_environment,
)
from insertany3d.scheduler import WorkItem


class StubExecutor:
    def __init__(self, *stages: str):
        self.supported_stages = stages
        self.items: list[str] = []

    def execute(self, _controller, item):
        self.items.append(item.stage)
        return ExecutionResult(True, artifacts=[{"artifactId": "x", "path": "x"}])


class StubStore:
    def __init__(self, root: Path):
        self.root = root

    def row(self, _query, _parameters):
        return {"root_path": str(self.root)}


class StubCommandExecutor:
    def __init__(self, result: ExecutionResult):
        self.result = result
        self.commands: list[list[str]] = []
        self.kwargs = []

    def execute(self, _controller, _item, command, **kwargs):
        self.commands.append(list(command))
        self.kwargs.append(kwargs)
        return self.result


class RuntimeWorkerTests(unittest.TestCase):
    def test_local_environment_is_additive_and_does_not_execute_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / ".env"
            env_file.write_text(
                "REMOTE_VALUE=from-file\n"
                "export QUOTED=\"two words\"\n"
                "COMMAND=\"$(touch should-not-exist)\"\n",
                encoding="utf-8",
            )
            values = {"REMOTE_VALUE": "from-shell"}
            loaded = load_environment_file(env_file, environ=values)
            self.assertEqual(loaded["COMMAND"], "$(touch should-not-exist)")
            self.assertEqual(values["REMOTE_VALUE"], "from-shell")
            self.assertEqual(values["QUOTED"], "two words")
            self.assertFalse((root / "should-not-exist").exists())

    def test_local_environment_missing_and_invalid_files_are_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values: dict[str, str] = {}
            self.assertIsNone(load_local_environment(repository_root=root, environ=values))
            invalid = root / "invalid.env"
            invalid.write_text("not-an-assignment\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeWorkerConfigurationError, "缺少"):
                load_environment_file(invalid, environ=values)
            values["INSERTANY3D_LOCAL_ENV_FILE"] = "missing.env"
            with self.assertRaisesRegex(RuntimeWorkerConfigurationError, "不存在"):
                load_local_environment(repository_root=root, environ=values)

    def test_local_environment_explicit_path_is_relative_to_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config").mkdir()
            (root / "config" / "local.env").write_text("REMOTE_TARGET=host\n", encoding="utf-8")
            values = {"INSERTANY3D_LOCAL_ENV_FILE": "config/local.env"}
            loaded = load_local_environment(repository_root=root, environ=values)
            self.assertEqual(loaded, (root / "config" / "local.env").resolve())
            self.assertEqual(values["REMOTE_TARGET"], "host")

    def test_composite_routes_in_dag_order_and_rejects_duplicates(self) -> None:
        local = StubExecutor("unity_anchor", "image_edit")
        remote = StubExecutor("upload_inputs", "model_generation")
        composite = CompositeStageExecutor([remote, local])
        self.assertEqual(
            composite.supported_stages,
            ("unity_anchor", "image_edit", "upload_inputs", "model_generation"),
        )
        item = SimpleNamespace(stage="model_generation")
        self.assertTrue(composite.execute(None, item).succeeded)
        self.assertEqual(remote.items, ["model_generation"])
        with self.assertRaisesRegex(RuntimeWorkerConfigurationError, "重复注册"):
            CompositeStageExecutor([local, StubExecutor("image_edit")])

    def test_remote_wrapper_crash_fences_resource_as_delivery_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            request_path.write_text("{}", encoding="utf-8")
            command = StubCommandExecutor(
                ExecutionResult(False, error_code="worker_crash", message="ssh wrapper exited")
            )
            executor = RemoteStageExecutor(
                RemoteProfile(
                    target="worker@example.test",
                    project_root="/srv/InsertAny3D",
                    artifact_root="/srv/insert-runs/batch",
                ),
                command_executor=command,
                local_python="python-test",
            )
            item = WorkItem(
                1,
                "batch_001",
                "Farm_Test_001",
                "Task_001",
                "model_generation",
                "model-generation-v1",
                1,
                1,
                "lease-token",
                {"remote_gpu": "gpu:1"},
                root / "output.staging",
                root / "output",
                100.0,
            )
            controller = SimpleNamespace(store=StubStore(root))
            credential_environment = {
                "APIYI_API_KEY": "apiyi-secret",
                "APIYI_API_KEY_FILE": "/private/apiyi-key",
                "GEMINI_API_KEY": "gemini-secret",
                "GEMINI_API_KEY_FILE": "/private/gemini-key",
                "BEE_API_KEY": "bee-secret",
            }
            with (
                patch(
                    "insertany3d.runtime_workers.write_stage_request",
                    return_value=({}, request_path),
                ),
                patch.dict(os.environ, credential_environment, clear=False),
            ):
                result = executor.execute(controller, item)
            self.assertFalse(result.succeeded)
            self.assertEqual(result.error_code, "delivery_unknown")
            self.assertFalse(result.cleanup_completed)
            self.assertEqual(command.commands[0][:3], ["python-test", "-m", "insertany3d.remote_worker"])
            self.assertIn("PYTHONPATH", command.kwargs[0]["env"])
            for name in credential_environment:
                self.assertNotIn(name, command.kwargs[0]["env"])


if __name__ == "__main__":
    unittest.main()
