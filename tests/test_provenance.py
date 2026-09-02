from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from insertany3d.cli import _git_snapshot, _record_run_provenance


class ProvenanceTests(unittest.TestCase):
    def test_git_snapshot_records_head_and_dirty_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "InsertAny3D test"], check=True)
            tracked = root / "tracked.txt"
            tracked.write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
            (root / "untracked.txt").write_text("after\n", encoding="utf-8")

            snapshot = _git_snapshot(root)

            self.assertRegex(snapshot["head"], r"^[0-9a-f]{40}$")
            self.assertEqual(snapshot["status"], "dirty")
            self.assertIn("?? untracked.txt", snapshot["statusOutput"])
            self.assertIsNone(snapshot["error"])

    def test_unavailable_server_snapshot_is_written_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            with patch.dict(os.environ, {}, clear=True):
                entry = _record_run_provenance("batch-test", root, None)

            self.assertIsNotNone(entry)
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "insertany3d.run-manifest")
            self.assertEqual(manifest["batchId"], "batch-test")
            self.assertEqual(manifest["provenance"]["server"]["head"], "unavailable")
            self.assertEqual(manifest["provenance"]["server"]["status"], "unavailable")
            self.assertEqual(
                manifest["provenance"]["configurationSource"]["server"]["source"],
                ".insertany3d/runtime.env",
            )
            self.assertEqual(
                manifest["provenance"]["configurationSource"]["local"]["status"],
                "not_configured",
            )
            self.assertEqual(len(manifest["provenanceHistory"]), 1)

    def test_provenance_write_failure_returns_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run-file"
            root.write_text("not a directory", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                result = _record_run_provenance("batch-test", root, None)
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
