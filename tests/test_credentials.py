from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from insertany3d.credentials import ApiYiCredentialError, load_apiyi_api_key


class ApiYiCredentialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="insertany3d_credentials_")
        self.root = Path(self.temporary.name)
        self.default_file = self.root / "default_key"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_key(path: Path, value: str = "file-secret", mode: int = 0o600) -> None:
        path.write_text(value, encoding="utf-8")
        if os.name == "posix":
            path.chmod(mode)

    def test_priority_prefers_apiyi_environment_then_file_then_legacy(self) -> None:
        self._write_key(self.default_file)
        key, source = load_apiyi_api_key(
            {
                "APIYI_API_KEY": "apiyi-secret",
                "GEMINI_API_KEY": "gemini-secret",
                "BEE_API_KEY": "bee-secret",
            },
            default_key_file=self.default_file,
        )
        self.assertEqual((key, source), ("apiyi-secret", "APIYI_API_KEY"))

        key, source = load_apiyi_api_key(
            {"GEMINI_API_KEY": "invalid-legacy", "BEE_API_KEY": "bee-secret"},
            default_key_file=self.default_file,
        )
        self.assertEqual(key, "file-secret")
        self.assertNotIn(key, source)
        self.assertEqual(source, str(self.default_file))

        self.default_file.unlink()
        self.assertEqual(
            load_apiyi_api_key(
                {"GEMINI_API_KEY": "gemini-secret", "BEE_API_KEY": "bee-secret"},
                default_key_file=self.default_file,
            ),
            ("gemini-secret", "GEMINI_API_KEY"),
        )

    def test_explicit_file_overrides_default_and_missing_file_is_an_error(self) -> None:
        explicit = self.root / "explicit_key"
        self._write_key(self.default_file, "default-secret")
        self._write_key(explicit, "explicit-secret")
        self.assertEqual(
            load_apiyi_api_key(
                {
                    "APIYI_API_KEY_FILE": str(explicit),
                    "GEMINI_API_KEY_FILE": str(self.default_file),
                    "BEE_API_KEY": "legacy-secret",
                },
                default_key_file=self.default_file,
            ),
            ("explicit-secret", str(explicit)),
        )
        with self.assertRaisesRegex(ApiYiCredentialError, "文件不存在"):
            load_apiyi_api_key(
                {
                    "APIYI_API_KEY_FILE": str(self.root / "missing"),
                    "BEE_API_KEY": "must-not-fallback",
                },
                default_key_file=self.default_file,
            )

    def test_existing_empty_or_unsafe_default_file_does_not_fall_back(self) -> None:
        self._write_key(self.default_file, "")
        with self.assertRaisesRegex(ApiYiCredentialError, "文件为空"):
            load_apiyi_api_key(
                {"BEE_API_KEY": "must-not-fallback"},
                default_key_file=self.default_file,
            )

        self._write_key(self.default_file, mode=0o644)
        if os.name == "posix":
            with self.assertRaisesRegex(ApiYiCredentialError, "权限过宽"):
                load_apiyi_api_key({}, default_key_file=self.default_file)

        self._write_key(self.default_file, mode=0o000)
        if os.name == "posix":
            with self.assertRaisesRegex(ApiYiCredentialError, "不可读"):
                load_apiyi_api_key({}, default_key_file=self.default_file)

    def test_missing_default_reports_setup_without_secret(self) -> None:
        with self.assertRaisesRegex(ApiYiCredentialError, "未找到 APIYi key") as caught:
            load_apiyi_api_key({}, default_key_file=self.default_file)
        self.assertIn(str(self.default_file), str(caught.exception))


if __name__ == "__main__":
    unittest.main()
