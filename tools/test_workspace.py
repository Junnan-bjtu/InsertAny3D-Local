from __future__ import annotations

import json
from unittest import mock
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workspace import TaskWorkspace, atomic_write_json, safe_name


class WorkspaceTests(unittest.TestCase):
    def test_initialize(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value) / "task"
            workspace = TaskWorkspace(root)
            manifest = workspace.initialize(run_id="run-1", project_id="Farm", task_id="Task_001")
            self.assertEqual(manifest["schemaVersion"], 1)
            self.assertEqual(json.loads((root / "task_manifest.json").read_text())["taskId"], "Task_001")

    def test_attempts_preserve_evidence(self) -> None:
        with TemporaryDirectory() as value:
            workspace = TaskWorkspace(Path(value))
            workspace.initialize(run_id="r", project_id="p", task_id="t")
            first, first_dir, first_manifest = workspace.begin_stage("sags")
            workspace.finish_stage(first_manifest, status="ready")
            second, second_dir, second_manifest = workspace.begin_stage("sags")
            workspace.finish_stage(second_manifest, status="failed", error="boom")
            self.assertEqual((first, second), ("0001", "0002"))
            self.assertNotEqual(first_dir, second_dir)
            self.assertEqual(json.loads(first_manifest.read_text())["status"], "ready")
            self.assertEqual(json.loads(second_manifest.read_text())["status"], "failed")

    def test_safe_name_and_atomic_json(self) -> None:
        with self.assertRaises(ValueError):
            safe_name("../outside", "task_id")
        with TemporaryDirectory() as value:
            path = Path(value) / "manifest.json"
            atomic_write_json(path, {"version": 1})
            atomic_write_json(path, {"version": 2})
            self.assertEqual(json.loads(path.read_text())["version"], 2)

    def test_publish_file(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            workspace = TaskWorkspace(root)
            workspace.initialize(run_id="r", project_id="p", task_id="t")
            source = root / "stages" / "sags" / "attempts" / "0001" / "x.ply"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"ply")
            published = workspace.publish_file("sags", source, "inserted_object.ply")
            self.assertEqual(published.read_bytes(), b"ply")

    def test_publish_tree_restores_previous_output_when_replace_fails(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            workspace = TaskWorkspace(root)
            workspace.initialize(run_id="r", project_id="p", task_id="t")
            output = root / "stages" / "sags" / "output"
            output.mkdir(parents=True)
            (output / "result.txt").write_text("ready", encoding="utf-8")
            attempt = root / "attempt"
            attempt.mkdir()
            (attempt / "result.txt").write_text("failed-attempt", encoding="utf-8")

            real_replace = __import__("os").replace
            calls = 0

            def fail_final_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated rename failure")
                return real_replace(source, destination)

            with mock.patch("workspace.os.replace", side_effect=fail_final_replace):
                with self.assertRaises(OSError):
                    workspace.publish_tree("sags", attempt)
            self.assertEqual((output / "result.txt").read_text(encoding="utf-8"), "ready")


if __name__ == "__main__":
    unittest.main()
