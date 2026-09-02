#!/usr/bin/env python3
"""Small OpenCLIP image-encoder boundary used by TRELLIS yaw search.

Heavy dependencies are imported only when an encoder is actually loaded, so
runtime preflight and unit tests never initialize Torch or a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


DEFAULT_MODEL_NAME = "ViT-H-14"


def prepare_rgb(path: str | Path, *, image_module: Any | None = None):
    """Load an image as RGB, compositing alpha over the renderer's black background."""

    if image_module is None:
        from PIL import Image as image_module

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"CLIP 图片不存在: {source_path}")
    with image_module.open(source_path) as source:
        source.load()
        if "A" in source.getbands():
            rgba = source.convert("RGBA")
            result = image_module.new("RGB", rgba.size, (0, 0, 0))
            result.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
            return result
        return source.convert("RGB")


def load_image_encoder(
    checkpoint: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    open_clip_module: Any | None = None,
    torch_module: Any | None = None,
):
    """Load a local OpenCLIP checkpoint without any historical metric package."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise FileNotFoundError(f"CLIP 权重不存在或为空: {checkpoint_path}")
    if open_clip_module is None:
        try:
            import open_clip as open_clip_module
        except ImportError as exc:
            raise RuntimeError(
                "缺少 open-clip-torch；请按 environment/requirements-main.txt 安装服务器主环境"
            ) from exc
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("缺少 PyTorch，无法运行 CLIP 朝向搜索") from exc

    resolved_device = device or ("cuda" if torch_module.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip_module.create_model_and_transforms(
        model_name,
        pretrained=str(checkpoint_path),
        device=resolved_device,
        precision="fp32",
    )
    return model.eval(), preprocess, resolved_device


def image_similarity_scores(
    reference: str | Path,
    candidates: Sequence[str | Path],
    checkpoint: str | Path,
) -> list[float]:
    """Return normalized CLIP image-to-image similarities in candidate order."""

    candidate_paths = [Path(path) for path in candidates]
    if not candidate_paths:
        raise ValueError("CLIP 候选图片不能为空")
    for path in (Path(reference), *candidate_paths):
        if not path.is_file():
            raise FileNotFoundError(f"CLIP 图片不存在: {path}")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("缺少 PyTorch，无法运行 CLIP 朝向搜索") from exc
    model, preprocess, device = load_image_encoder(checkpoint, torch_module=torch)
    reference_tensor = preprocess(prepare_rgb(reference)).unsqueeze(0).to(device)
    candidate_tensor = torch.stack(
        [preprocess(prepare_rgb(path)) for path in candidate_paths]
    ).to(device)
    with torch.inference_mode():
        reference_features = model.encode_image(reference_tensor, normalize=True)
        candidate_features = model.encode_image(candidate_tensor, normalize=True)
        return (
            (candidate_features @ reference_features.T)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
