from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from insertany3d.cli import _build_review_contact_sheet
from insertany3d.image_edit import (
    decide_generation,
    generation_count,
    missing_generations,
    new_generation_group,
    record_generation,
    validate_generation_group,
    write_group,
)


UUID_OUTPUT = re.compile(r"^edited-[0-9a-f]{32}\.png$")


class ImageEditGenerationTests(unittest.TestCase):
    def test_default_group_has_three_independent_generations(self) -> None:
        group = new_generation_group("Task_001")
        self.assertEqual(generation_count(), 3)
        self.assertEqual(group["requestedCount"], 3)
        self.assertEqual([item["index"] for item in group["generations"]], [1, 2, 3])
        self.assertEqual([item["attempt"] for item in group["generations"]], [0, 0, 0])
        self.assertEqual(missing_generations(group), [1, 2, 3])

    def test_dynamic_candidate_count_scales_indexes_and_manifest(self) -> None:
        for count in (1, 2, 4, 7):
            with self.subTest(count=count):
                group = new_generation_group("Task_dynamic", count=count)
                self.assertEqual(generation_count({"num_gen_image_per_task": count}), count)
                self.assertEqual(
                    [item["index"] for item in group["generations"]],
                    list(range(1, count + 1)),
                )

    def test_manifest_maps_each_display_index_to_one_uuid_output(self) -> None:
        group = new_generation_group("Task_001")
        paths = {
            index: f"/runs/Task_001/generation-{index}/edited-{index:032x}.png"
            for index in range(1, 4)
        }
        for index, path in paths.items():
            group = record_generation(
                group, index, status="succeeded", attempt=1,
                output={"path": path, "fullPath": path},
            )
        manifest = validate_generation_group(group)["reviewManifest"]
        self.assertEqual(
            manifest["candidates"],
            [{"index": index, "path": path} for index, path in paths.items()],
        )
        self.assertEqual(len({item["path"] for item in manifest["candidates"]}), 3)
        self.assertTrue(all(UUID_OUTPUT.match(Path(path).name) for path in paths.values()))

    def test_retry_updates_only_failed_generation_and_retains_success(self) -> None:
        group = new_generation_group("Task_001")
        first_path = "/runs/edited-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.png"
        group = record_generation(group, 1, status="succeeded", attempt=1, output={"fullPath": first_path})
        group = record_generation(group, 2, status="failed_retryable", attempt=1, error={"code": "http_503"})
        self.assertEqual(missing_generations(group), [2, 3])
        retried_path = "/runs/edited-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.png"
        group = record_generation(group, 2, status="succeeded", attempt=2, output={"fullPath": retried_path})
        self.assertEqual(group["generations"][0]["output"]["fullPath"], first_path)
        self.assertEqual(group["generations"][1]["attempt"], 2)
        self.assertEqual(missing_generations(group), [3])

    def test_partial_group_can_select_success_after_retry_exhaustion(self) -> None:
        group = new_generation_group("Task_partial")
        group = record_generation(
            group, 1, status="succeeded", attempt=1,
            output={"fullPath": "/runs/edited-11111111111111111111111111111111.png"},
        )
        for index, code in ((2, "compile_or_contract"), (3, "http_400")):
            group = record_generation(group, index, status="failed_terminal", attempt=3, error={"code": code})
        accepted = decide_generation(group, "1")
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["acceptedGeneration"]["index"], 1)
        self.assertEqual(accepted["acceptedGeneration"]["attempt"], 1)

    def test_regenerate_starts_complete_new_group_and_cancel_is_terminal(self) -> None:
        group = new_generation_group("Task_001", count=4)
        regenerated = decide_generation(group, "R")
        self.assertEqual(regenerated["status"], "generating")
        self.assertNotEqual(regenerated["groupId"], group["groupId"])
        self.assertEqual(regenerated["requestedCount"], 4)
        self.assertEqual(missing_generations(regenerated), [1, 2, 3, 4])
        self.assertEqual(decide_generation(group, "N")["status"], "canceled")

    def test_group_manifest_is_json(self) -> None:
        with tempfile.TemporaryDirectory(prefix="insertany3d_image_edit_manifest_") as directory:
            path = Path(directory) / "generations" / "manifest.json"
            write_group(path, new_generation_group("Task_001"))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["requestedCount"], 3)

    def test_contact_sheet_contains_one_panel_per_available_candidate(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        with tempfile.TemporaryDirectory(prefix="insertany3d_contact_sheet_") as directory:
            root = Path(directory)
            candidates = []
            for index, color in ((1, "red"), (3, "blue")):
                path = root / f"edited-{index:032x}.png"
                Image.new("RGB", (20, 10), color).save(path)
                candidates.append({"index": index, "path": str(path)})
            sheet = _build_review_contact_sheet(candidates, root / "contact-sheet.png")
            self.assertIsNotNone(sheet)
            with Image.open(sheet) as rendered:
                self.assertEqual(rendered.size, (40, 46))


if __name__ == "__main__":
    unittest.main()
