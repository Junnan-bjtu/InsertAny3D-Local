#!/usr/bin/env python3
"""Select a canonical TRELLIS yaw with a two-pass CLIP-I search.

The search compares the exact TRELLIS generation input with full canonical
renders.  RGBA inputs are composited over black so the cutout alpha channel is
preserved in CLIP's RGB input and matches the renderer background.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from clip_image_similarity import image_similarity_scores, prepare_rgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_CHECKPOINT = (
    PROJECT_ROOT
    / "metrics"
    / "checkpoints"
    / "models"
    / "AI-ModelScope--CLIP-ViT-H-14-laion2B-s32B-b79K"
    / "snapshots"
    / "master"
    / "open_clip_pytorch_model.bin"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 CLIP-I 两阶段搜索 TRELLIS canonical yaw")
    parser.add_argument("--input-image", required=True, type=Path, help="TRELLIS 实际生成输入（通常为 cutout.png）")
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trellis-python", required=True, type=Path)
    parser.add_argument("--render-script", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--fov", type=float, default=53.1301023542)
    parser.add_argument("--pitch", type=float, required=True)
    parser.add_argument("--distance", type=float, default=1.5)
    parser.add_argument("--near", type=float, default=0.8)
    parser.add_argument("--far", type=float, default=1.6)
    parser.add_argument("--side-angle", type=float, default=24.0)
    parser.add_argument("--clip-checkpoint", type=Path, default=DEFAULT_CLIP_CHECKPOINT)
    return parser.parse_args()


def _render(args: argparse.Namespace, output: Path, yaw_offsets: list[float], names: list[str]) -> None:
    command = [
        str(args.trellis_python),
        str(args.render_script),
        "--input-ply", str(args.input_ply),
        "--output-dir", str(output),
        "--resolution", str(args.resolution),
        "--fov", str(args.fov),
        "--yaw-degrees", "0",
        "--pitch-degrees", str(args.pitch),
        "--distance", str(args.distance),
        "--near", str(args.near),
        "--far", str(args.far),
        "--yaw-offsets", ",".join(str(value) for value in yaw_offsets),
        "--view-names", ",".join(names),
    ]
    subprocess.run(command, check=True)


def _clip_scores(reference: Path, candidates: list[Path], checkpoint: Path) -> list[float]:
    return image_similarity_scores(reference, candidates, checkpoint)


def _candidate_record(yaws: list[float], scores: list[float], images: list[Path]) -> list[dict[str, object]]:
    return [
        {"yaw": float(yaw % 360.0), "score": float(score), "image": str(image)}
        for yaw, score, image in zip(yaws, scores, images)
    ]


def main() -> int:
    args = parse_args()
    for path in (args.input_image, args.input_ply, args.trellis_python, args.render_script):
        if not path.exists():
            raise SystemExit(f"输入路径不存在: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coarse_dir = args.output_dir / "coarse"
    fine_dir = args.output_dir / "fine"
    selected_dir = args.output_dir / "selected"
    for path in (coarse_dir, fine_dir, selected_dir):
        if path.exists():
            shutil.rmtree(path)
    clip_reference = args.output_dir / "reference_clip_input.png"
    prepare_rgb(args.input_image).save(clip_reference)

    coarse_yaws = [float(value) for value in range(0, 360, 60)]
    coarse_names = [f"yaw_{int(yaw):03d}" for yaw in coarse_yaws]
    _render(args, coarse_dir, coarse_yaws, coarse_names)
    coarse_images = [coarse_dir / "source" / "images" / f"{name}.png" for name in coarse_names]
    coarse_scores = _clip_scores(args.input_image, coarse_images, args.clip_checkpoint)
    coarse_best_index = max(range(len(coarse_scores)), key=coarse_scores.__getitem__)
    coarse_best = coarse_yaws[coarse_best_index]

    # Search the circular 60-degree neighborhood around the coarse winner.
    fine_yaws = [coarse_best - 30.0 + 10.0 * index for index in range(7)]
    fine_names = [f"yaw_{int(round(yaw)) % 360:03d}" for yaw in fine_yaws]
    _render(args, fine_dir, fine_yaws, fine_names)
    fine_images = [fine_dir / "source" / "images" / f"{name}.png" for name in fine_names]
    fine_scores = _clip_scores(args.input_image, fine_images, args.clip_checkpoint)
    fine_best_index = max(range(len(fine_scores)), key=fine_scores.__getitem__)
    selected_yaw = fine_yaws[fine_best_index] % 360.0

    # Keep the original renderer contract: this directory contains exactly
    # left/center/right for the unchanged second-stage GIM code.
    _render(
        args,
        selected_dir,
        [selected_yaw - args.side_angle, selected_yaw, selected_yaw + args.side_angle],
        ["left", "center", "right"],
    )
    result = {
        "status": "ready",
        "method": "clip_image_similarity_cascade",
        "referenceImage": str(args.input_image.resolve()),
        "referenceClipInput": str(clip_reference.resolve()),
        "pitchDegrees": float(args.pitch),
        "coarseYaws": coarse_yaws,
        "coarseBestYaw": float(coarse_best % 360.0),
        "coarseCandidates": _candidate_record(coarse_yaws, coarse_scores, coarse_images),
        "fineYaws": [float(yaw % 360.0) for yaw in fine_yaws],
        "fineCandidates": _candidate_record(fine_yaws, fine_scores, fine_images),
        "selectedYaw": float(selected_yaw),
        "selectedRenderDir": str(selected_dir),
        "clipCheckpoint": str(args.clip_checkpoint),
    }
    (args.output_dir / "yaw_search.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TRELLIS_YAW_SEARCH_READY", json.dumps({"selectedYaw": selected_yaw, "coarseBest": coarse_best}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
