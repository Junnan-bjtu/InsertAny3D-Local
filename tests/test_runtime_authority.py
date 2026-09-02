from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "tools" / "sync_remote_runtime.py"
SPEC = importlib.util.spec_from_file_location("sync_remote_runtime", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


class RuntimeAuthorityTests(unittest.TestCase):
    def test_server_source_is_preferred_over_legacy_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = root / "InsertAny3D-Server"
            server_tools = server / "tools"
            server_tools.mkdir(parents=True)
            (server_tools / "stage_adapter.py").write_text("# server\n", encoding="utf-8")
            legacy = root / "codex_remote_tools"
            legacy.mkdir()
            source, authority = runtime_sync.resolve_runtime_source(
                root / "integration",
                server_root=server,
                source=legacy,
            )
            self.assertEqual(source, server_tools.resolve())
            self.assertEqual(authority, "server_checkout")

    def test_server_source_accepts_checkout_or_tools_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            (tools / "stage_adapter.py").write_text("# server\n", encoding="utf-8")
            for candidate in (root, tools):
                source, authority = runtime_sync.resolve_runtime_source(
                    root / "integration", server_source=candidate
                )
                self.assertEqual(source, tools.resolve())
                self.assertEqual(authority, "server_checkout")

    def test_server_authority_is_valid_in_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            sample = tools / "sample.py"
            sample.write_text("value = 1\n", encoding="utf-8")
            lock = {
                "schemaVersion": 1,
                "kind": "insertany3d.remote-runtime-lock",
                "sourceAuthority": "server_checkout",
                "files": [
                    {
                        "path": "sample.py",
                        "sha256": runtime_sync._sha256(sample),
                        "size": sample.stat().st_size,
                    }
                ],
            }
            runtime_sync._write_json_atomic(root / runtime_sync.LOCK_PATH, lock)
            self.assertEqual(runtime_sync.verify_lock(root, {"sample.py"}), [])

    def test_committed_runtime_matches_lock(self) -> None:
        self.assertEqual(runtime_sync.verify_lock(REPOSITORY_ROOT), [])

    def test_lock_detects_modified_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            sample = tools / "sample.py"
            sample.write_text("value = 1\n", encoding="utf-8")
            lock = {
                "schemaVersion": 1,
                "kind": "insertany3d.remote-runtime-lock",
                "sourceAuthority": "codex_remote_tools",
                "files": [
                    {
                        "path": "sample.py",
                        "sha256": runtime_sync._sha256(sample),
                        "size": sample.stat().st_size,
                    }
                ],
            }
            runtime_sync._write_json_atomic(root / runtime_sync.LOCK_PATH, lock)
            self.assertEqual(runtime_sync.verify_lock(root, {"sample.py"}), [])
            sample.write_text("value = 2\n", encoding="utf-8")
            self.assertEqual(
                runtime_sync.verify_lock(root, {"sample.py"}),
                ["公开副本与锁文件不一致: sample.py"],
            )


if __name__ == "__main__":
    unittest.main()
