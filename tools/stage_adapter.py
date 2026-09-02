#!/usr/bin/env python3
"""Execute one versioned InsertAny3D remote stage request.

This module is intentionally dependency-free.  It validates the scheduler's
stage-request v1 contract, translates supported stages to the existing runtime
CLIs, supervises the child process group, and writes stage-result v1.  The
model, matching, pose, and SAGS algorithms remain in their existing scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_ROOT.parent
TRELLIS_PYTHON = PROJECT_ROOT / "third_party" / "TRELLIS" / ".venv" / "bin" / "python"
GIM_PYTHON = PROJECT_ROOT / "third_party" / "gim" / ".venv" / "bin" / "python"

SUPPORTED_CONTRACTS = {
    "model_generation": "model-generation-v1",
    "render_alignment_views": "render-alignment-v1",
    "segment_inputs": "segment-inputs-v1",
    "gim_match": "gim-match-v1",
    "estimate_pose": "estimate-pose-v1",
    "sags_segment_vote": "sags-vote-v1",
    "debug_bundle": "debug-bundle-v1",
}
ALL_STAGE_NAMES = {
    "unity_anchor", "image_edit", "edit_gate", "upload_inputs", "model_generation",
    "render_alignment_views", "segment_inputs", "gim_match", "estimate_pose",
    "sags_segment_vote", "debug_bundle", "download_results", "unity_apply",
    "unity_eval6", "evaluate_absolute",
}
FORMAL_TASK_IDS = {f"Task_{index:03d}" for index in range(1, 6)}
RESULT_STATUSES = {"succeeded", "failed_retryable", "failed_terminal", "rejected", "canceled"}
REQUEST_FIELDS = {
    "schemaVersion",
    "kind",
    "batchId",
    "projectId",
    "taskId",
    "stage",
    "contractVersion",
    "attempt",
    "leaseToken",
    "inputs",
    "effectiveConfig",
    "effectiveConfigSha256",
    "outputStagingDir",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GPU_DEVICE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")
_TERMINATION_SIGNAL: int | None = None
_POSE_VIEW_NAMES = {"left", "center", "right"}
_SAGS_VIEW_NAMES = {"center", "ring_060", "ring_120", "ring_180", "ring_240", "ring_300"}


class AdapterError(ValueError):
    """A stable stage error that can be serialized into stage-result v1."""

    def __init__(self, code: str, message: str, *, status: str = "failed_terminal"):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class InputArtifact:
    artifact_id: str
    relative_path: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class StagePlan:
    stage: str
    contract_version: str
    commands: tuple[tuple[str, ...], ...]
    environment: Mapping[str, str]
    required_outputs: tuple[str, ...]
    compatibility_facade: bool = False
    atomic_bundle: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": "insertany3d.stage-plan",
            "stage": self.stage,
            "contractVersion": self.contract_version,
            "commands": [list(command) for command in self.commands],
            "environment": dict(self.environment),
            "requiredOutputs": list(self.required_outputs),
            "compatibilityFacade": self.compatibility_facade,
            "atomicBundle": self.atomic_bundle,
        }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError("invalid_stage_request", f"{field} 必须是非空相对路径")
    text = value.replace("\\", "/")
    candidate = Path(text)
    windows = PureWindowsPath(text)
    if candidate.is_absolute() or windows.is_absolute() or windows.drive or ".." in candidate.parts:
        raise AdapterError("unsafe_path", f"{field} 必须是不会越出 artifact 根目录的相对路径")
    if text in {".", "./"}:
        raise AdapterError("unsafe_path", f"{field} 不能指向 artifact 根目录本身")
    return text


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_symlink_components(path: Path, root: Path, field: str) -> None:
    """Reject mutable path indirection below the already-resolved artifact root."""

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise AdapterError("unsafe_path", f"{field} 越出 artifact 根目录") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AdapterError("unsafe_path", f"{field} 不能包含符号链接: {current}")


def _resolve_relative(root: Path, value: Any, field: str) -> tuple[str, Path]:
    root = root.resolve()
    relative = _safe_relative(value, field)
    unresolved = root / relative
    _reject_symlink_components(unresolved, root, field)
    path = unresolved.resolve()
    if not _within(path, root):
        raise AdapterError("unsafe_path", f"{field} 解析后越出 artifact 根目录")
    return relative, path


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise AdapterError("invalid_stage_request", f"{field} 不是安全 ID")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError("invalid_stage_request", f"{field} 必须是非空字符串")
    return value


def validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdapterError("invalid_stage_request", "stage request 顶层必须是对象")
    missing = REQUEST_FIELDS - set(value)
    extra = set(value) - REQUEST_FIELDS
    if missing:
        raise AdapterError("invalid_stage_request", "stage request 缺少字段: " + ", ".join(sorted(missing)))
    if extra:
        raise AdapterError("invalid_stage_request", "stage request 包含未知字段: " + ", ".join(sorted(extra)))
    if value.get("schemaVersion") != 1 or value.get("kind") != "insertany3d.stage-request":
        raise AdapterError("invalid_stage_request", "只支持 insertany3d.stage-request schemaVersion 1")
    _safe_id(value.get("batchId"), "batchId")
    _safe_id(value.get("projectId"), "projectId")
    if value.get("taskId") not in FORMAL_TASK_IDS:
        raise AdapterError("invalid_stage_request", "taskId 必须是 Task_001 至 Task_005")
    stage = value.get("stage")
    if stage not in ALL_STAGE_NAMES:
        raise AdapterError("invalid_stage_request", "stage 不是 v1 契约中的已知阶段")
    _nonempty(value.get("contractVersion"), "contractVersion")
    if not isinstance(value.get("attempt"), int) or isinstance(value.get("attempt"), bool) or value["attempt"] < 1:
        raise AdapterError("invalid_stage_request", "attempt 必须是正整数")
    _nonempty(value.get("leaseToken"), "leaseToken")
    if not isinstance(value.get("inputs"), list):
        raise AdapterError("invalid_stage_request", "inputs 必须是数组")
    for index, item in enumerate(value["inputs"]):
        if not isinstance(item, dict):
            raise AdapterError("invalid_stage_request", f"inputs[{index}] 必须是对象")
        for field in ("artifactId", "path", "sha256"):
            if field not in item:
                raise AdapterError("invalid_stage_request", f"inputs[{index}] 缺少 {field}")
        _safe_id(item.get("artifactId"), f"inputs[{index}].artifactId")
        _safe_relative(item.get("path"), f"inputs[{index}].path")
        if not isinstance(item.get("sha256"), str) or not _SHA256.fullmatch(item["sha256"]):
            raise AdapterError("invalid_stage_request", f"inputs[{index}].sha256 格式错误")
    if not isinstance(value.get("effectiveConfig"), dict):
        raise AdapterError("invalid_stage_request", "effectiveConfig 必须是对象")
    digest = value.get("effectiveConfigSha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise AdapterError("invalid_stage_request", "effectiveConfigSha256 格式错误")
    if canonical_sha256(value["effectiveConfig"]) != digest:
        raise AdapterError("config_hash_mismatch", "effectiveConfigSha256 与 effectiveConfig 不一致")
    _safe_relative(value.get("outputStagingDir"), "outputStagingDir")
    return dict(value)


def resolve_inputs(request: Mapping[str, Any], artifact_root: Path) -> dict[str, InputArtifact]:
    result: dict[str, InputArtifact] = {}
    for index, item in enumerate(request["inputs"]):
        artifact_id = str(item["artifactId"])
        if artifact_id in result:
            raise AdapterError("invalid_stage_request", f"inputs 的 artifactId 重复: {artifact_id}")
        relative, path = _resolve_relative(artifact_root, item["path"], f"inputs[{index}].path")
        if not path.is_file():
            raise AdapterError("missing_input", f"输入 artifact 不存在或不是文件: {relative}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise AdapterError("input_hash_mismatch", f"输入 artifact 哈希不一致: {artifact_id}")
        result[artifact_id] = InputArtifact(artifact_id, relative, path, actual)
    return result


def _stage_options(request: Mapping[str, Any], allowed: Iterable[str]) -> dict[str, Any]:
    value = request["effectiveConfig"].get("stageOptions", {})
    if not isinstance(value, dict):
        raise AdapterError("invalid_stage_config", "effectiveConfig.stageOptions 必须是对象")
    common = {"timeoutSeconds", "gpuDevice"}
    unknown = set(value) - set(allowed) - common
    if unknown:
        raise AdapterError("invalid_stage_config", "stageOptions 包含未知字段: " + ", ".join(sorted(unknown)))
    if "timeoutSeconds" in value:
        _number(value["timeoutSeconds"], "stageOptions.timeoutSeconds", minimum=0.001)
    return dict(value)


def _artifact(inputs: Mapping[str, InputArtifact], artifact_id: str) -> Path:
    try:
        return inputs[artifact_id].path
    except KeyError as exc:
        raise AdapterError("missing_input", f"缺少输入 artifact: {artifact_id}") from exc


def _artifact_id(value: Any, field: str) -> str:
    return _safe_id(value, field)


def _number(
    value: Any, field: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise AdapterError("invalid_stage_config", f"{field} 必须是有限数字")
    result = float(value)
    if minimum is not None and result < minimum:
        raise AdapterError("invalid_stage_config", f"{field} 必须大于等于 {minimum}")
    if maximum is not None and result > maximum:
        raise AdapterError("invalid_stage_config", f"{field} 必须小于等于 {maximum}")
    return result


def _integer(
    value: Any, field: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AdapterError("invalid_stage_config", f"{field} 必须是整数")
    if minimum is not None and value < minimum:
        raise AdapterError("invalid_stage_config", f"{field} 必须大于等于 {minimum}")
    if maximum is not None and value > maximum:
        raise AdapterError("invalid_stage_config", f"{field} 必须小于等于 {maximum}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise AdapterError("invalid_stage_config", f"{field} 必须是布尔值")
    return value


def _string(value: Any, field: str) -> str:
    return _nonempty(value, field)


def _enum(value: Any, field: str, allowed: set[str]) -> str:
    text = _string(value, field)
    if text not in allowed:
        raise AdapterError("invalid_stage_config", f"{field} 必须是: {', '.join(sorted(allowed))}")
    return text


def _option(command: list[str], options: Mapping[str, Any], key: str, flag: str, parser) -> None:
    if key in options:
        command.extend((flag, str(parser(options[key], f"stageOptions.{key}"))))


def _environment(options: Mapping[str, Any]) -> dict[str, str]:
    result = {"PYTHONUNBUFFERED": "1", "SPCONV_ALGO": "native", "MPLBACKEND": "Agg"}
    if "gpuDevice" in options:
        device = _string(options["gpuDevice"], "stageOptions.gpuDevice")
        if not _GPU_DEVICE.fullmatch(device):
            raise AdapterError("invalid_stage_config", "gpuDevice 必须是逗号分隔的非负 GPU 编号")
        result["CUDA_VISIBLE_DEVICES"] = device
    return result


def _view_names(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise AdapterError("invalid_stage_config", "stageOptions.viewNames 必须是非空数组")
    result = []
    for index, item in enumerate(value):
        name = _safe_id(item, f"stageOptions.viewNames[{index}]")
        if name in result:
            raise AdapterError("invalid_stage_config", f"viewNames 重复: {name}")
        result.append(name)
    return result


def _model_generation_plan(
    request: Mapping[str, Any], inputs: Mapping[str, InputArtifact], output: Path
) -> StagePlan:
    allowed = {
        "provider", "model", "seed", "sparseSteps", "slatSteps", "sparseCfg", "slatCfg",
        "multiMode", "noGlb", "requireGlb", "debug", "debugDumpInterval",
        "trellisMaskPrompts", "inputMaskEngine", "inputMaskPointsPerMask",
    }
    options = _stage_options(request, allowed)
    provider = str(options.get("provider", "trellis"))
    if provider != "trellis":
        raise AdapterError("unsupported_stage_variant", "model_generation v1 目前只安全拆分 TRELLIS provider")
    image_ids = sorted(
        artifact_id for artifact_id in inputs
        if artifact_id == "input_image" or artifact_id.startswith("input_image_")
    )
    if not image_ids:
        raise AdapterError("missing_input", "model_generation 至少需要 input_image artifact")
    prompts = options.get("trellisMaskPrompts")
    if prompts is not None:
        if not isinstance(prompts, list) or not prompts or any(not isinstance(value, str) or not value.strip() for value in prompts):
            raise AdapterError("invalid_stage_config", "stageOptions.trellisMaskPrompts 必须是非空字符串数组")
        if len(image_ids) != 1:
            raise AdapterError("invalid_stage_config", "组合输入蒙版目前只支持一张编辑图")
        engine = _enum(options.get("inputMaskEngine", "legacy"), "stageOptions.inputMaskEngine", {"auto", "legacy", "langsam"})
        mask_output = output / "model_input"
        mask_command = [
            str(TRELLIS_PYTHON), str(TOOLS_ROOT / "auto_segment.py"),
            "--input", str(_artifact(inputs, image_ids[0])),
            "--output-dir", str(mask_output), "--engine", engine,
        ]
        if "inputMaskPointsPerMask" in options:
            mask_command.extend(("--points-per-mask", str(_integer(options["inputMaskPointsPerMask"], "stageOptions.inputMaskPointsPerMask", minimum=1))))
        for index, value in enumerate(prompts):
            mask_command.extend(("--prompt", _string(value, f"stageOptions.trellisMaskPrompts[{index}]")))
        generation_inputs = [mask_output / "cutout.png"]
    else:
        mask_command = None
        generation_inputs = [Path(_artifact(inputs, artifact_id)) for artifact_id in image_ids]
    command = [str(TRELLIS_PYTHON), str(TOOLS_ROOT / "generate_trellis_asset.py"), "--input-image"]
    command.extend(str(path) for path in generation_inputs)
    command.extend(("--output-dir", str(output)))
    _option(command, options, "model", "--model", _string)
    _option(command, options, "seed", "--seed", lambda value, field: _integer(value, field, minimum=0))
    _option(command, options, "sparseSteps", "--sparse-steps", lambda value, field: _integer(value, field, minimum=1))
    _option(command, options, "slatSteps", "--slat-steps", lambda value, field: _integer(value, field, minimum=1))
    _option(command, options, "sparseCfg", "--sparse-cfg", _number)
    _option(command, options, "slatCfg", "--slat-cfg", _number)
    if "multiMode" in options:
        command.extend(("--multi-mode", _enum(options["multiMode"], "stageOptions.multiMode", {"stochastic", "multidiffusion"})))
    for key, flag in (("noGlb", "--no-glb"), ("requireGlb", "--require-glb"), ("debug", "--debug")):
        if key in options and _boolean(options[key], f"stageOptions.{key}"):
            command.append(flag)
    if options.get("noGlb") is True and options.get("requireGlb") is True:
        raise AdapterError("invalid_stage_config", "noGlb 与 requireGlb 不能同时为 true")
    _option(command, options, "debugDumpInterval", "--debug-dump-interval", lambda value, field: _number(value, field, minimum=0.1))
    required = ["sample.ply", "manifest.json"]
    if mask_command is not None:
        required.insert(0, "model_input/cutout.png")
    if options.get("requireGlb") is True:
        required.append("sample.glb")
    commands = (tuple(mask_command), tuple(command)) if mask_command is not None else (tuple(command),)
    return StagePlan(request["stage"], request["contractVersion"], commands, _environment(options), tuple(required))


def _render_plan(request: Mapping[str, Any], inputs: Mapping[str, InputArtifact], output: Path) -> StagePlan:
    allowed = {
        "inputPlyArtifactId", "yawDegrees", "pitchDegrees", "distance", "sideAngleDegrees",
        "yawOffsets", "viewNames", "resolution", "fov", "near", "far", "ssaa",
        "unityCameraArtifactIds", "coarsePoseArtifactId", "unityManifestArtifactId",
        "ringViewNames", "ringYawOffsets",
    }
    options = _stage_options(request, allowed)
    ply_id = _artifact_id(options.get("inputPlyArtifactId", "sample_ply"), "stageOptions.inputPlyArtifactId")
    names = _view_names(options.get("viewNames", ["left", "center", "right"]))

    def render_command(target: Path, view_names: Sequence[str], offsets: Sequence[Any] | None) -> list[str]:
        command = [
            str(TRELLIS_PYTHON), str(TOOLS_ROOT / "render_trellis_views.py"),
            "--input-ply", str(_artifact(inputs, ply_id)), "--output-dir", str(target),
            "--view-names", ",".join(view_names),
        ]
        for key, flag, parser in (
            ("yawDegrees", "--yaw-degrees", _number),
            ("pitchDegrees", "--pitch-degrees", _number),
            ("distance", "--distance", lambda value, field: _number(value, field, minimum=0.000001)),
            ("sideAngleDegrees", "--side-angle-degrees", lambda value, field: _number(value, field, minimum=0)),
            ("resolution", "--resolution", lambda value, field: _integer(value, field, minimum=1)),
            ("fov", "--fov", lambda value, field: _number(value, field, minimum=0.000001)),
            ("near", "--near", lambda value, field: _number(value, field, minimum=0.000001)),
            ("far", "--far", lambda value, field: _number(value, field, minimum=0.000001)),
            ("ssaa", "--ssaa", lambda value, field: _integer(value, field, minimum=1)),
        ):
            _option(command, options, key, flag, parser)
        if offsets is not None:
            if not isinstance(offsets, list) or len(offsets) != len(view_names):
                raise AdapterError("invalid_stage_config", "yaw offset 必须与 view name 等长")
            parsed = [_number(item, "stageOptions.yawOffsets") for item in offsets]
            # argparse treats a value beginning with ``-`` as another option;
            # use the equals form for the standard [-24, 0, 24] alignment yaw.
            command.append("--yaw-offsets=" + ",".join(str(item) for item in parsed))
        return command

    alignment_offsets = options.get("yawOffsets")
    command = render_command(output, names, alignment_offsets)
    effective_near = _number(options.get("near", 0.8), "stageOptions.near", minimum=0.000001)
    effective_far = _number(options.get("far", 1.6), "stageOptions.far", minimum=0.000001)
    if effective_far <= effective_near:
        raise AdapterError("invalid_stage_config", "stageOptions.far 必须大于 near")
    camera_ids = options.get("unityCameraArtifactIds", [])
    if not isinstance(camera_ids, list):
        raise AdapterError("invalid_stage_config", "unityCameraArtifactIds 必须是数组")
    if camera_ids:
        if len(camera_ids) != len(names):
            raise AdapterError("invalid_stage_config", "unityCameraArtifactIds 必须与 viewNames 等长")
        coarse_id = _artifact_id(options.get("coarsePoseArtifactId"), "stageOptions.coarsePoseArtifactId")
        command.extend(("--coarse-pose", str(_artifact(inputs, coarse_id))))
        for index, artifact_id in enumerate(camera_ids):
            command.extend(("--unity-camera", str(_artifact(inputs, _artifact_id(artifact_id, f"unityCameraArtifactIds[{index}]")))))
        if "unityManifestArtifactId" in options:
            manifest_id = _artifact_id(options["unityManifestArtifactId"], "stageOptions.unityManifestArtifactId")
            command.extend(("--unity-manifest", str(_artifact(inputs, manifest_id))))
    elif "coarsePoseArtifactId" in options or "unityManifestArtifactId" in options:
        raise AdapterError("invalid_stage_config", "coarsePoseArtifactId/unityManifestArtifactId 只能与 Unity 相机列表一起使用")
    required = ["views.json", "source/sparse/0/cameras.txt", "source/sparse/0/images.txt", "model/cfg_args"]
    required.extend(f"source/images/{name}.png" for name in names)
    commands = [tuple(command)]

    ring_names_value = options.get("ringViewNames")
    ring_offsets = options.get("ringYawOffsets")
    if (ring_names_value is None) != (ring_offsets is None):
        raise AdapterError("invalid_stage_config", "ringViewNames 与 ringYawOffsets 必须同时提供")
    if ring_names_value is not None:
        ring_names = _view_names(ring_names_value)
        expected = ("center", "ring_060", "ring_120", "ring_180", "ring_240", "ring_300")
        if tuple(ring_names) != expected:
            raise AdapterError("invalid_stage_config", "SAGS ringViewNames 必须按固定顺序包含六个环拍视角")
        ring_command = render_command(output / "ring6", ring_names, ring_offsets)
        commands.append(tuple(ring_command))
        required.extend((
            "ring6/views.json",
            "ring6/source/sparse/0/cameras.txt",
            "ring6/source/sparse/0/images.txt",
            "ring6/model/cfg_args",
        ))
        required.extend(f"ring6/source/images/{name}.png" for name in ring_names)
    return StagePlan(request["stage"], request["contractVersion"], tuple(commands), _environment(options), tuple(required))


def _segment_plan(request: Mapping[str, Any], inputs: Mapping[str, InputArtifact], output: Path) -> StagePlan:
    allowed = {
        "mode", "inputImageArtifactId", "prompt", "taskPrompt", "engine", "samType",
        "boxThreshold", "textThreshold", "pointsPerMask", "device", "views",
        "sagsViews", "sagsPrompt", "sagsTaskPrompt",
    }
    options = _stage_options(request, allowed)
    mode = _enum(options.get("mode", "target"), "stageOptions.mode", {"target", "anchor"})
    if mode == "target":
        image_id = _artifact_id(options.get("inputImageArtifactId", "input_image"), "stageOptions.inputImageArtifactId")
        command = [
            str(TRELLIS_PYTHON), str(TOOLS_ROOT / "auto_segment.py"),
            "--input", str(_artifact(inputs, image_id)), "--output-dir", str(output),
        ]
        prompt = options.get("prompt")
        task_prompt = options.get("taskPrompt")
        if prompt is not None and task_prompt is not None:
            raise AdapterError("invalid_stage_config", "prompt 与 taskPrompt 只能提供一个")
        if prompt is not None:
            prompts = prompt if isinstance(prompt, list) else [prompt]
            if not prompts:
                raise AdapterError("invalid_stage_config", "prompt 不能为空")
            for index, item in enumerate(prompts):
                command.extend(("--prompt", _string(item, f"stageOptions.prompt[{index}]")))
        elif task_prompt is not None:
            command.extend(("--task-prompt", _string(task_prompt, "stageOptions.taskPrompt")))
        else:
            task = request["effectiveConfig"].get("task", {})
            fallback = task.get("objectPrompt") if isinstance(task, dict) else None
            command.extend(("--task-prompt", _string(fallback, "effectiveConfig.task.objectPrompt")))
        if "engine" in options:
            command.extend(("--engine", _enum(options["engine"], "stageOptions.engine", {"langsam", "legacy", "auto"})))
        _option(command, options, "samType", "--sam-type", _string)
        _option(command, options, "boxThreshold", "--box-threshold", lambda value, field: _number(value, field, minimum=0, maximum=1))
        _option(command, options, "textThreshold", "--text-threshold", lambda value, field: _number(value, field, minimum=0, maximum=1))
        _option(command, options, "pointsPerMask", "--points-per-mask", lambda value, field: _integer(value, field, minimum=1))
        _option(command, options, "device", "--device", _string)
        required = ("mask.png", "cutout.png", "points.json", "manifest.json")
    else:
        prompt = _string(options.get("prompt"), "stageOptions.prompt")
        views = options.get("views")
        if not isinstance(views, list) or not views:
            raise AdapterError("invalid_stage_config", "anchor 模式需要非空 views")
        command = [
            str(TRELLIS_PYTHON), str(TOOLS_ROOT / "segment_anchor_views.py"),
            "--prompt", prompt, "--output-dir", str(output),
        ]
        names = []
        for index, view in enumerate(views):
            if not isinstance(view, dict) or set(view) != {"name", "sceneArtifactId", "generatedArtifactId"}:
                raise AdapterError("invalid_stage_config", f"views[{index}] 字段必须是 name/sceneArtifactId/generatedArtifactId")
            name = _safe_id(view["name"], f"views[{index}].name")
            names.append(name)
            scene_id = _artifact_id(view["sceneArtifactId"], f"views[{index}].sceneArtifactId")
            generated_id = _artifact_id(view["generatedArtifactId"], f"views[{index}].generatedArtifactId")
            command.extend(("--view", name, str(_artifact(inputs, scene_id)), str(_artifact(inputs, generated_id))))
        if len(names) != 3 or set(names) != _POSE_VIEW_NAMES:
            raise AdapterError("invalid_stage_config", "anchor 分割必须恰好包含 left/center/right 三视图")
        _option(command, options, "boxThreshold", "--box-threshold", lambda value, field: _number(value, field, minimum=0, maximum=1))
        _option(command, options, "textThreshold", "--text-threshold", lambda value, field: _number(value, field, minimum=0, maximum=1))
        _option(command, options, "device", "--device", _string)
        required = tuple(["manifest.json"] + [f"{name}/{side}/mask.png" for name in names for side in ("scene", "generated")])
    commands = [tuple(command)]
    required_outputs = list(required)
    sags_views = options.get("sagsViews")
    if sags_views is not None:
        if not isinstance(sags_views, list) or len(sags_views) != 6:
            raise AdapterError("invalid_stage_config", "sagsViews 必须恰好包含六个独立环拍视角")
        sags_prompt = options.get("sagsPrompt")
        sags_task_prompt = options.get("sagsTaskPrompt")
        if sags_prompt is not None and sags_task_prompt is not None:
            raise AdapterError("invalid_stage_config", "sagsPrompt 与 sagsTaskPrompt 只能提供一个")
        if sags_prompt is None and sags_task_prompt is None:
            task = request["effectiveConfig"].get("task", {})
            sags_task_prompt = task.get("objectPrompt") if isinstance(task, dict) else None
        ring_names = []
        for index, view in enumerate(sags_views):
            if not isinstance(view, dict) or set(view) != {"name", "imageArtifactId"}:
                raise AdapterError("invalid_stage_config", f"sagsViews[{index}] 必须包含 name/imageArtifactId")
            name = _safe_id(view["name"], f"sagsViews[{index}].name")
            ring_names.append(name)
            image_id = _artifact_id(view["imageArtifactId"], f"sagsViews[{index}].imageArtifactId")
            ring_output = output / "sags_annotations" / name
            ring_command = [
                str(TRELLIS_PYTHON), str(TOOLS_ROOT / "auto_segment.py"),
                "--input", str(_artifact(inputs, image_id)), "--output-dir", str(ring_output),
            ]
            if sags_prompt is not None:
                prompts = sags_prompt if isinstance(sags_prompt, list) else [sags_prompt]
                if not prompts:
                    raise AdapterError("invalid_stage_config", "sagsPrompt 不能为空")
                for prompt_index, item in enumerate(prompts):
                    ring_command.extend(("--prompt", _string(item, f"stageOptions.sagsPrompt[{prompt_index}]")))
            else:
                ring_command.extend(("--task-prompt", _string(sags_task_prompt, "stageOptions.sagsTaskPrompt")))
            if "engine" in options:
                ring_command.extend(("--engine", _enum(options["engine"], "stageOptions.engine", {"langsam", "legacy", "auto"})))
            _option(ring_command, options, "samType", "--sam-type", _string)
            _option(ring_command, options, "boxThreshold", "--box-threshold", lambda value, field: _number(value, field, minimum=0, maximum=1))
            _option(ring_command, options, "textThreshold", "--text-threshold", lambda value, field: _number(value, field, minimum=0, maximum=1))
            _option(ring_command, options, "pointsPerMask", "--points-per-mask", lambda value, field: _integer(value, field, minimum=1))
            _option(ring_command, options, "device", "--device", _string)
            commands.append(tuple(ring_command))
            required_outputs.extend((
                f"sags_annotations/{name}/mask.png",
                f"sags_annotations/{name}/points.json",
                f"sags_annotations/{name}/manifest.json",
            ))
        expected = ("center", "ring_060", "ring_120", "ring_180", "ring_240", "ring_300")
        if tuple(ring_names) != expected:
            raise AdapterError("invalid_stage_config", "sagsViews 必须按固定顺序包含六个环拍视角")
    return StagePlan(
        request["stage"], request["contractVersion"], tuple(commands),
        _environment(options), tuple(required_outputs),
    )


def _gim_plan(request: Mapping[str, Any], inputs: Mapping[str, InputArtifact], output: Path) -> StagePlan:
    allowed = {
        "pairs", "model", "checkpointArtifactId", "maxMatches", "resizeMax", "ransacThreshold",
        "seed", "foregroundThreshold", "maxAlignedDisplacement", "autoMask1Nonblack", "allowEmpty",
    }
    options = _stage_options(request, allowed)
    pairs = options.get("pairs")
    if pairs is None:
        pairs = [{"name": "pair_00", "image0ArtifactId": "image0", "image1ArtifactId": "image1"}]
    if not isinstance(pairs, list) or not pairs:
        raise AdapterError("invalid_stage_config", "gim_match 需要至少一个 pair")
    commands = []
    required = []
    pair_names: set[str] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise AdapterError("invalid_stage_config", f"pairs[{index}] 必须是对象")
        allowed_pair = {"name", "image0ArtifactId", "image1ArtifactId", "mask0ArtifactId", "mask1ArtifactId", "roi0", "roi1"}
        if set(pair) - allowed_pair or not {"name", "image0ArtifactId", "image1ArtifactId"} <= set(pair):
            raise AdapterError("invalid_stage_config", f"pairs[{index}] 字段不完整或包含未知字段")
        name = _safe_id(pair["name"], f"pairs[{index}].name")
        if name in pair_names:
            raise AdapterError("invalid_stage_config", f"pairs 的 name 重复: {name}")
        pair_names.add(name)
        pair_output = output / name
        image0_id = _artifact_id(pair["image0ArtifactId"], f"pairs[{index}].image0ArtifactId")
        image1_id = _artifact_id(pair["image1ArtifactId"], f"pairs[{index}].image1ArtifactId")
        command = [
            str(GIM_PYTHON), str(TOOLS_ROOT / "run_gim_match.py"),
            "--image0", str(_artifact(inputs, image0_id)), "--image1", str(_artifact(inputs, image1_id)),
            "--output-dir", str(pair_output),
        ]
        for key, flag, parser in (
            ("model", "--model", lambda value, field: _enum(value, field, {"gim_dkm", "gim_roma", "gim_loftr", "gim_lightglue"})),
            ("maxMatches", "--max-matches", lambda value, field: _integer(value, field, minimum=1)),
            ("resizeMax", "--resize-max", lambda value, field: _integer(value, field, minimum=1)),
            ("ransacThreshold", "--ransac-threshold", lambda value, field: _number(value, field, minimum=0)),
            ("seed", "--seed", lambda value, field: _integer(value, field, minimum=0)),
            ("foregroundThreshold", "--foreground-threshold", lambda value, field: _integer(value, field, minimum=0, maximum=255)),
            ("maxAlignedDisplacement", "--max-aligned-displacement", lambda value, field: _number(value, field, minimum=0)),
        ):
            _option(command, options, key, flag, parser)
        if "checkpointArtifactId" in options:
            checkpoint_id = _artifact_id(options["checkpointArtifactId"], "stageOptions.checkpointArtifactId")
            command.extend(("--checkpoint", str(_artifact(inputs, checkpoint_id))))
        for side in (0, 1):
            key = f"mask{side}ArtifactId"
            if key in pair:
                artifact_id = _artifact_id(pair[key], f"pairs[{index}].{key}")
                command.extend((f"--mask{side}", str(_artifact(inputs, artifact_id))))
            roi_key = f"roi{side}"
            if roi_key in pair:
                roi = pair[roi_key]
                if not isinstance(roi, list) or len(roi) != 3:
                    raise AdapterError("invalid_stage_config", f"pairs[{index}].{roi_key} 必须是三个数字")
                parsed_roi = [_number(item, f"pairs[{index}].{roi_key}[{item_index}]") for item_index, item in enumerate(roi)]
                if parsed_roi[2] < 0:
                    raise AdapterError("invalid_stage_config", f"pairs[{index}].{roi_key} 半径不能为负")
                command.extend((f"--roi{side}", *(str(item) for item in parsed_roi)))
        if options.get("autoMask1Nonblack", True):
            command.append("--auto-mask1-nonblack")
        if options.get("allowEmpty", True):
            command.append("--allow-empty")
        for key in ("autoMask1Nonblack", "allowEmpty"):
            if key in options:
                _boolean(options[key], f"stageOptions.{key}")
        commands.append(tuple(command))
        required.extend((f"{name}/matches.json", f"{name}/match.png", f"{name}/warp.png"))
    return StagePlan(request["stage"], request["contractVersion"], tuple(commands), _environment(options), tuple(required))


def _pose_plan(request: Mapping[str, Any], inputs: Mapping[str, InputArtifact], output: Path) -> StagePlan:
    allowed = {
        "generatedCamerasArtifactId", "generatedImagesArtifactId", "views", "anchorMasksManifestArtifactId",
        "generatedAxis", "ransacThreshold", "ransacIterations", "minInliers", "minViewInliers",
        "minViewInlierRatio", "maxMatchesPerView", "maxDepthRelativeSpread", "spatialGridSize",
        "anchorMaskDilation", "crossViewNeighbors", "crossViewMinSupport", "crossViewFallbackSupport",
        "minConsistentPoints", "minConsistentViewPoints", "seed", "withoutScale", "pixelCenterOffset",
    }
    options = _stage_options(request, allowed)
    cameras_id = _artifact_id(options.get("generatedCamerasArtifactId", "generated_cameras"), "generatedCamerasArtifactId")
    images_id = _artifact_id(options.get("generatedImagesArtifactId", "generated_images"), "generatedImagesArtifactId")
    views = options.get("views")
    if not isinstance(views, list) or not views:
        raise AdapterError("invalid_stage_config", "estimate_pose 需要非空 views")
    pose_output = output / "pose.json"
    diagnostics = output / "diagnostics"
    command = [
        str(GIM_PYTHON), str(TOOLS_ROOT / "estimate_similarity_pose.py"),
        "--generated-cameras", str(_artifact(inputs, cameras_id)),
        "--generated-images", str(_artifact(inputs, images_id)),
        "--output", str(pose_output), "--diagnostics-dir", str(diagnostics),
        "--generated-axis", _enum(options.get("generatedAxis", "legacy-flip-z"), "generatedAxis", {"identity", "legacy-flip-z"}),
        "--run-id", request["batchId"], "--candidate-id", f"{request['projectId']}-{request['taskId']}",
        "--exit-zero-on-rejected",
    ]
    numeric = (
        ("ransacThreshold", "--ransac-threshold", lambda value, field: _number(value, field, minimum=0)),
        ("ransacIterations", "--ransac-iterations", lambda value, field: _integer(value, field, minimum=1)),
        ("minInliers", "--min-inliers", lambda value, field: _integer(value, field, minimum=3)),
        ("minViewInliers", "--min-view-inliers", lambda value, field: _integer(value, field, minimum=0)),
        ("minViewInlierRatio", "--min-view-inlier-ratio", lambda value, field: _number(value, field, minimum=0, maximum=1)),
        ("maxMatchesPerView", "--max-matches-per-view", lambda value, field: _integer(value, field, minimum=0)),
        ("maxDepthRelativeSpread", "--max-depth-relative-spread", lambda value, field: _number(value, field, minimum=0)),
        ("spatialGridSize", "--spatial-grid-size", lambda value, field: _integer(value, field, minimum=1)),
        ("anchorMaskDilation", "--anchor-mask-dilation", lambda value, field: _integer(value, field, minimum=0)),
        ("crossViewNeighbors", "--cross-view-neighbors", lambda value, field: _integer(value, field, minimum=1)),
        ("crossViewMinSupport", "--cross-view-min-support", lambda value, field: _integer(value, field, minimum=1)),
        ("crossViewFallbackSupport", "--cross-view-fallback-support", lambda value, field: _integer(value, field, minimum=1)),
        ("minConsistentPoints", "--min-consistent-points", lambda value, field: _integer(value, field, minimum=3)),
        ("minConsistentViewPoints", "--min-consistent-view-points", lambda value, field: _integer(value, field, minimum=1)),
        ("seed", "--seed", lambda value, field: _integer(value, field, minimum=0)),
        ("pixelCenterOffset", "--pixel-center-offset", _number),
    )
    for key, flag, parser in numeric:
        _option(command, options, key, flag, parser)
    if "withoutScale" in options and _boolean(options["withoutScale"], "stageOptions.withoutScale"):
        command.append("--without-scale")
    if "anchorMasksManifestArtifactId" in options:
        marker_id = _artifact_id(options["anchorMasksManifestArtifactId"], "anchorMasksManifestArtifactId")
        marker = _artifact(inputs, marker_id)
        _require_directory_covered(marker.parent, inputs)
        command.extend(("--anchor-masks-dir", str(marker.parent)))
    view_names: list[str] = []
    for index, view in enumerate(views):
        fields = {"name", "matchesArtifactId", "sceneDepthArtifactId", "sceneCameraArtifactId", "generatedDepthArtifactId"}
        if not isinstance(view, dict) or set(view) != fields:
            raise AdapterError("invalid_stage_config", f"views[{index}] 必须恰好包含 {', '.join(sorted(fields))}")
        view_name = _safe_id(view["name"], f"views[{index}].name")
        view_names.append(view_name)
        paths = []
        for field in ("matchesArtifactId", "sceneDepthArtifactId", "sceneCameraArtifactId", "generatedDepthArtifactId"):
            paths.append(_artifact(inputs, _artifact_id(view[field], f"views[{index}].{field}")))
        if paths[-1].stem != view_name:
            raise AdapterError(
                "invalid_stage_config",
                f"views[{index}].name 必须与 generated depth 文件名一致: {view_name} != {paths[-1].stem}",
            )
        command.extend(("--view", *(str(path) for path in paths)))
    if len(view_names) != 3 or set(view_names) != _POSE_VIEW_NAMES:
        raise AdapterError("invalid_stage_config", "estimate_pose 必须恰好联合 left/center/right 三视图")
    return StagePlan(
        request["stage"], request["contractVersion"], (tuple(command),), _environment(options),
        ("pose.json", "diagnostics/multiview_summary.json"),
    )


def _require_directory_covered(
    directory: Path,
    inputs: Mapping[str, InputArtifact],
    *,
    allow_runtime_mutations: bool = False,
) -> None:
    resolved = directory.resolve()
    covered = {artifact.path.resolve() for artifact in inputs.values()}
    missing = []
    for path in resolved.rglob("*"):
        if path.is_symlink():
            raise AdapterError("unsafe_path", f"目录输入不能包含符号链接: {path}")
        if allow_runtime_mutations:
            relative = path.relative_to(resolved)
            # SAGS lazily creates these caches while loading a Gaussian model.
            # They are runtime-owned and are not part of the immutable input.
            if relative.parts and (
                relative.parts[0] == "sam_pt"
                or relative.parts[:2] == ("objects", "tmp")
            ):
                continue
        if path.is_file() and path.resolve() not in covered:
            missing.append(path)
    if missing:
        sample = ", ".join(str(path) for path in missing[:3])
        raise AdapterError("unhashed_directory_input", f"目录输入包含未列入 inputs 的文件: {sample}")


def _sags_plan(
    request: Mapping[str, Any],
    inputs: Mapping[str, InputArtifact],
    output: Path,
    *,
    allow_runtime_mutations: bool = False,
) -> StagePlan:
    allowed = {
        "modelMarkerArtifactId", "annotations", "viewName", "samCheckpointArtifactId", "samArch",
        "maskId", "threshold", "minVotes", "voteMode", "gdInterval", "forceSeedRadius",
        "noForceSeed", "visibilityDepthTolerance", "independentMinPriorCoverage", "centerMaskHard",
    }
    options = _stage_options(request, allowed)
    marker_id = _artifact_id(options.get("modelMarkerArtifactId", "model_cfg_args"), "modelMarkerArtifactId")
    marker = _artifact(inputs, marker_id)
    model_dir = marker.parent
    _require_directory_covered(model_dir, inputs, allow_runtime_mutations=allow_runtime_mutations)
    annotations = options.get("annotations")
    if not isinstance(annotations, list) or len(annotations) != 6:
        raise AdapterError("invalid_stage_config", "sags_segment_vote v1 要求恰好六个独立视角 annotation")
    output_ply = output / "inserted_object.ply"
    command = [
        str(TRELLIS_PYTHON), str(TOOLS_ROOT / "run_sags_text.py"),
        "--model-dir", str(model_dir), "--output-ply", str(output_ply),
        "--annotation-mode", "independent", "--diagnostics-dir", str(output / "diagnostics"),
        "--min-votes", str(_integer(options.get("minVotes", 3), "stageOptions.minVotes", minimum=1, maximum=6)),
        "--independent-min-prior-coverage", str(_number(options.get("independentMinPriorCoverage", 0.25), "stageOptions.independentMinPriorCoverage", minimum=0, maximum=1)),
    ]
    names = []
    for index, annotation in enumerate(annotations):
        fields = {"name", "maskArtifactId", "pointsArtifactId"}
        if not isinstance(annotation, dict) or set(annotation) != fields:
            raise AdapterError("invalid_stage_config", f"annotations[{index}] 必须恰好包含 name/maskArtifactId/pointsArtifactId")
        name = _safe_id(annotation["name"], f"annotations[{index}].name")
        if name in names:
            raise AdapterError("invalid_stage_config", f"annotation 视角重复: {name}")
        names.append(name)
        mask = _artifact(inputs, _artifact_id(annotation["maskArtifactId"], f"annotations[{index}].maskArtifactId"))
        points = _artifact(inputs, _artifact_id(annotation["pointsArtifactId"], f"annotations[{index}].pointsArtifactId"))
        command.extend(("--view-annotation", name, str(mask), str(points)))
    if set(names) != _SAGS_VIEW_NAMES:
        raise AdapterError(
            "invalid_stage_config",
            "SAGS 六视角必须是 center/ring_060/ring_120/ring_180/ring_240/ring_300",
        )
    source_view = _safe_id(options.get("viewName", "center"), "stageOptions.viewName")
    if source_view not in names:
        raise AdapterError("invalid_stage_config", "viewName 必须存在于 annotations")
    command.extend(("--view-name", source_view))
    for key, flag, parser in (
        ("samArch", "--sam-arch", _string),
        ("maskId", "--mask-id", lambda value, field: _integer(value, field, minimum=-1, maximum=2)),
        ("threshold", "--threshold", lambda value, field: _number(value, field, minimum=0, maximum=1)),
        ("voteMode", "--vote-mode", lambda value, field: _enum(value, field, {"majority", "union"})),
        ("gdInterval", "--gd-interval", _integer),
        ("forceSeedRadius", "--force-seed-radius", lambda value, field: _integer(value, field, minimum=0)),
        ("visibilityDepthTolerance", "--visibility-depth-tolerance", lambda value, field: _number(value, field, minimum=0)),
    ):
        _option(command, options, key, flag, parser)
    if "gdInterval" in options and options["gdInterval"] not in {-1} and options["gdInterval"] < 1:
        raise AdapterError("invalid_stage_config", "stageOptions.gdInterval 必须是 -1 或正整数")
    if "samCheckpointArtifactId" in options:
        checkpoint_id = _artifact_id(options["samCheckpointArtifactId"], "samCheckpointArtifactId")
        command.extend(("--sam-checkpoint", str(_artifact(inputs, checkpoint_id))))
    if "noForceSeed" in options and _boolean(options["noForceSeed"], "stageOptions.noForceSeed"):
        command.append("--no-force-seed")
    if "centerMaskHard" in options:
        command.append("--center-mask-hard" if _boolean(options["centerMaskHard"], "stageOptions.centerMaskHard") else "--no-center-mask-hard")
    return StagePlan(
        request["stage"], request["contractVersion"], (tuple(command),), _environment(options),
        ("inserted_object.ply", "inserted_object.json", "diagnostics/sags_diagnostics.json"),
    )


def _debug_bundle_plan(request: Mapping[str, Any], inputs: Mapping[str, InputArtifact], output: Path) -> StagePlan:
    allowed = {
        "mode", "batchManifestArtifactId", "allowCompatibilityFacade", "includeDepthGim", "reuseExistingDepth",
        "includeRaw", "gridCellSize", "maxGridMatches", "gimModel",
    }
    options = _stage_options(request, allowed)
    # Existing direct callers without a mode remain on the guarded legacy
    # path.  The scheduler always emits mode=atomic for the 15-stage DAG.
    mode = _enum(options.get("mode", "legacy"), "stageOptions.mode", {"atomic", "legacy"})
    if mode == "atomic":
        if not inputs:
            raise AdapterError("missing_input", "atomic debug bundle 至少需要一个已提交上游 artifact")
        return StagePlan(
            request["stage"], request["contractVersion"], (), _environment(options),
            ("bundle_manifest.json", f"{request['taskId']}/bundle_manifest.json", f"{request['taskId']}/artifact_index.json"),
            atomic_bundle=True,
        )
    if not _boolean(options.get("allowCompatibilityFacade", False), "stageOptions.allowCompatibilityFacade"):
        raise AdapterError(
            "unsupported_atomic_stage",
            "build_debug_bundle.py 会读取完整任务目录；只有显式允许 compatibility facade 且 inputs 覆盖目录全部文件时才能运行",
        )
    manifest_id = _artifact_id(options.get("batchManifestArtifactId", "batch_manifest"), "batchManifestArtifactId")
    manifest = _artifact(inputs, manifest_id)
    run_root = manifest.parent
    _require_directory_covered(run_root, inputs)
    if _within(output, run_root):
        raise AdapterError("unsafe_path", "debug bundle outputStagingDir 必须位于输入 run 目录之外")
    command = [
        str(TRELLIS_PYTHON), str(TOOLS_ROOT / "build_debug_bundle.py"),
        "--run-root", str(run_root), "--output-root", str(output), "--task", request["taskId"],
    ]
    include_depth = _boolean(options.get("includeDepthGim", False), "stageOptions.includeDepthGim")
    reuse_depth = _boolean(options.get("reuseExistingDepth", False), "stageOptions.reuseExistingDepth")
    if not include_depth:
        if reuse_depth:
            raise AdapterError("invalid_stage_config", "reuseExistingDepth 需要同时启用 includeDepthGim")
        command.append("--skip-depth-gim")
    elif reuse_depth:
        command.append("--reuse-existing-depth")
    if not _boolean(options.get("includeRaw", True), "stageOptions.includeRaw"):
        command.append("--no-raw")
    _option(command, options, "gridCellSize", "--grid-cell-size", lambda value, field: _integer(value, field, minimum=128))
    _option(command, options, "maxGridMatches", "--max-grid-matches", lambda value, field: _integer(value, field, minimum=0))
    if "gimModel" in options:
        command.extend(("--gim-model", _enum(options["gimModel"], "stageOptions.gimModel", {"gim_dkm", "gim_roma", "gim_loftr", "gim_lightglue"})))
    return StagePlan(
        request["stage"], request["contractVersion"], (tuple(command),), _environment(options),
        ("bundle_manifest.json", f"{request['taskId']}/bundle_manifest.json"), True,
    )


def build_plan(
    request: Mapping[str, Any],
    inputs: Mapping[str, InputArtifact],
    output_staging: Path,
    *,
    allow_runtime_mutations: bool = False,
) -> StagePlan:
    stage = request["stage"]
    expected_contract = SUPPORTED_CONTRACTS.get(stage)
    if expected_contract is None:
        raise AdapterError("unsupported_stage", f"远端 stage adapter 不执行本机阶段: {stage}")
    if request["contractVersion"] != expected_contract:
        raise AdapterError(
            "unsupported_contract_version",
            f"{stage} 只支持 {expected_contract}，收到 {request['contractVersion']}",
        )
    builders = {
        "model_generation": _model_generation_plan,
        "render_alignment_views": _render_plan,
        "segment_inputs": _segment_plan,
        "gim_match": _gim_plan,
        "estimate_pose": _pose_plan,
        "sags_segment_vote": _sags_plan,
        "debug_bundle": _debug_bundle_plan,
    }
    if stage == "sags_segment_vote":
        return _sags_plan(
            request,
            inputs,
            output_staging,
            allow_runtime_mutations=allow_runtime_mutations,
        )
    return builders[stage](request, inputs, output_staging)


def _timeout_seconds(request: Mapping[str, Any]) -> float | None:
    options = request["effectiveConfig"].get("stageOptions", {})
    value = options.get("timeoutSeconds") if isinstance(options, dict) else None
    if value is None:
        return None
    result = _number(value, "stageOptions.timeoutSeconds", minimum=0.001)
    return result


def _diagnostic_paths(output: Path) -> list[str]:
    diagnostics = output / "_diagnostics"
    if not diagnostics.is_dir():
        return []
    result = []
    for path in diagnostics.rglob("*"):
        if path.is_symlink():
            raise AdapterError("unsafe_path", f"诊断输出不能包含符号链接: {path}")
        if path.is_file():
            result.append(path.relative_to(output).as_posix())
    return sorted(result)


def _artifact_id_for(relative: str) -> str:
    known = {
        "sample.ply": "sample_ply",
        "sample.glb": "sample_glb",
        "manifest.json": "stage_manifest",
        "views.json": "views_manifest",
        "mask.png": "mask",
        "cutout.png": "cutout",
        "points.json": "points",
        "pose.json": "pose",
        "inserted_object.ply": "inserted_object_ply",
        "inserted_object.json": "sags_manifest",
        "bundle_manifest.json": "bundle_manifest",
    }
    return known.get(relative, "file_" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20])


def _collect_artifacts(output: Path) -> list[dict[str, Any]]:
    result = []
    ids = set()
    for path in sorted(output.rglob("*")):
        relative_path = path.relative_to(output)
        if path.is_symlink():
            raise AdapterError("unsafe_path", f"输出 artifact 不能包含符号链接: {relative_path.as_posix()}")
        if not path.is_file() or "_diagnostics" in relative_path.parts or path.name == "stage_result.json":
            continue
        resolved = path.resolve()
        if not _within(resolved, output):
            raise AdapterError("unsafe_path", f"输出 artifact 越出 staging: {relative_path.as_posix()}")
        relative = relative_path.as_posix()
        size = path.stat().st_size
        if size <= 0:
            raise AdapterError("invalid_artifact", f"输出 artifact 为空: {relative}")
        artifact_id = _artifact_id_for(relative)
        if artifact_id in ids:
            artifact_id = "file_" + hashlib.sha256((relative + "#").encode("utf-8")).hexdigest()[:20]
        ids.add(artifact_id)
        result.append({
            "artifactId": artifact_id,
            "type": "stage_output",
            "path": relative,
            "sha256": sha256_file(path),
            "size": size,
        })
    if not result:
        raise AdapterError("missing_artifact", "stage 没有生成任何可发布 artifact")
    return result


def _inspect_outputs(plan: StagePlan, output: Path) -> tuple[str, str | None, str | None]:
    for relative in plan.required_outputs:
        _, path = _resolve_relative(output, relative, f"required output {relative}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise AdapterError("missing_artifact", f"stage 成功退出但缺少必要产物: {relative}")
        if path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AdapterError("invalid_artifact", f"JSON 产物无法解析: {relative}: {exc}") from exc
            if not isinstance(value, (dict, list)):
                raise AdapterError("invalid_artifact", f"JSON 产物顶层格式错误: {relative}")
    if plan.stage == "estimate_pose":
        pose = json.loads((output / "pose.json").read_text(encoding="utf-8"))
        if pose.get("status") == "rejected":
            return "rejected", "pose_quality_rejected", "位姿未通过多视角质量门禁"
        if pose.get("status") != "ready":
            raise AdapterError("invalid_artifact", "pose.json status 必须是 ready 或 rejected")
    return "succeeded", None, None


def _finished_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_result(
    request: Mapping[str, Any], status: str, *, artifacts: Sequence[Mapping[str, Any]] = (),
    error_code: str | None = None, message: str | None = None,
    diagnostics: Sequence[str] = (), cleanup_completed: bool = True,
    compatibility_facade: bool = False,
) -> dict[str, Any]:
    if status not in RESULT_STATUSES:
        raise ValueError(status)
    if status != "succeeded" and (not error_code or not message):
        raise ValueError("非成功结果必须包含 error_code 和 message")
    return {
        "schemaVersion": 1,
        "kind": "insertany3d.stage-result",
        "batchId": request["batchId"],
        "projectId": request["projectId"],
        "taskId": request["taskId"],
        "stage": request["stage"],
        "contractVersion": request["contractVersion"],
        "attempt": request["attempt"],
        "leaseToken": request["leaseToken"],
        "status": status,
        "artifacts": [dict(item) for item in artifacts],
        "errorCode": error_code,
        "message": message,
        "diagnosticPaths": list(diagnostics),
        "cleanup": {"completed": cleanup_completed, "compatibilityFacade": compatibility_facade},
        "finishedAtUtc": _finished_at(),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _process_group_exists(pgid: int) -> bool:
    if os.name == "nt":
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_process_group(pgid: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.02)
    return not _process_group_exists(pgid)


def _terminate(process: subprocess.Popen[Any], grace_seconds: float = 2.0) -> bool:
    if os.name == "nt":
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    return False
        return process.poll() is not None

    pgid = process.pid
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
    group_gone = _wait_process_group(pgid, grace_seconds)
    if not group_gone:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        group_gone = _wait_process_group(pgid, grace_seconds)
    if process.poll() is None:
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            return False
    return process.poll() is not None and group_gone


def _run_commands(
    commands: Sequence[Sequence[str]], output: Path, env_overrides: Mapping[str, str],
    *, timeout_seconds: float | None, cancel_file: Path | None, poll_seconds: float,
) -> tuple[str | None, str | None, bool]:
    diagnostics = output / "_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, command in enumerate(commands):
        if _TERMINATION_SIGNAL is not None:
            return "canceled", f"stage adapter 收到终止信号 {_TERMINATION_SIGNAL}", True
        if cancel_file is not None and cancel_file.exists():
            return "canceled", "stage cancellation requested", True
        stdout_path = diagnostics / f"stdout-{index:02d}.log"
        stderr_path = diagnostics / f"stderr-{index:02d}.log"
        env = os.environ.copy()
        env.update(env_overrides)
        env["INSERTANY3D_STAGE_OUTPUT"] = str(output)
        env["INSERTANY3D_STAGE_PLAN"] = str(diagnostics / "stage_plan.json")
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    list(command), cwd=PROJECT_ROOT, env=env, stdout=stdout, stderr=stderr,
                    start_new_session=os.name != "nt",
                )
                while process.poll() is None:
                    if _TERMINATION_SIGNAL is not None:
                        cleanup = _terminate(process)
                        return "canceled", f"stage adapter 收到终止信号 {_TERMINATION_SIGNAL}", cleanup
                    if cancel_file is not None and cancel_file.exists():
                        cleanup = _terminate(process)
                        return "canceled", "stage cancellation requested", cleanup
                    if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                        cleanup = _terminate(process)
                        return "worker_timeout", f"stage 超过 {timeout_seconds:g} 秒", cleanup
                    time.sleep(poll_seconds)
        except OSError as exc:
            return "worker_launch_failed", f"子进程无法启动: {type(exc).__name__}: {exc}", True
        cleanup = _terminate(process)
        if process.returncode != 0:
            return "worker_exit_nonzero", f"子进程退出码 {process.returncode}（命令 {index + 1}/{len(commands)}）", cleanup
        if not cleanup:
            return "worker_cleanup_failed", f"子进程 {index + 1}/{len(commands)} 退出后仍有同组进程", False
    return None, None, True


def _materialize_atomic_debug_bundle(
    request: Mapping[str, Any], inputs: Mapping[str, InputArtifact], output: Path
) -> None:
    """Create a self-contained bundle from only hash-verified stage inputs."""

    task_root = output / request["taskId"]
    bundled_root = task_root / "artifacts"
    records = []
    for artifact_id in sorted(inputs):
        artifact = inputs[artifact_id]
        # Keep the scheduler-relative layout so a downloaded bundle can be
        # inspected as the same stage tree, while artifact_index.json records
        # the request-local alias used by downstream contracts.
        destination = bundled_root / artifact.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact.path, destination)
        copied_sha = sha256_file(destination)
        if copied_sha != artifact.sha256:
            raise AdapterError("input_hash_mismatch", f"调试包复制后 hash 不一致: {artifact_id}")
        records.append(
            {
                "artifactId": artifact_id,
                "sourcePath": artifact.relative_path,
                "bundledPath": destination.relative_to(task_root).as_posix(),
                "sha256": copied_sha,
                "size": destination.stat().st_size,
            }
        )
    task_manifest = {
        "schemaVersion": 1,
        "kind": "insertany3d.atomic-debug-task",
        "batchId": request["batchId"],
        "projectId": request["projectId"],
        "taskId": request["taskId"],
        "sourceStage": request["stage"],
        "sourceAttempt": request["attempt"],
        "effectiveConfigSha256": request["effectiveConfigSha256"],
        "artifactCount": len(records),
        "artifacts": records,
    }
    _atomic_json(task_root / "artifact_index.json", task_manifest)
    _atomic_json(task_root / "bundle_manifest.json", task_manifest)
    _atomic_json(
        output / "bundle_manifest.json",
        {
            "schemaVersion": 1,
            "kind": "insertany3d.atomic-debug-bundle",
            "batchId": request["batchId"],
            "projectId": request["projectId"],
            "taskId": request["taskId"],
            "taskManifest": f"{request['taskId']}/bundle_manifest.json",
            "artifactCount": len(records),
        },
    )


def execute_request(
    request: Mapping[str, Any], artifact_root: Path, *, fake_command: Sequence[str] | None = None,
    cancel_file: Path | None = None, poll_seconds: float = 0.1,
) -> tuple[dict[str, Any], StagePlan]:
    root = artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _, output = _resolve_relative(root, request["outputStagingDir"], "outputStagingDir")
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.iterdir())
    if existing:
        raise AdapterError("staging_not_empty", f"outputStagingDir 必须为空: {existing[0]}")
    inputs = resolve_inputs(request, root)
    plan = build_plan(request, inputs, output)
    diagnostics_dir = output / "_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(diagnostics_dir / "stage_plan.json", plan.to_dict())
    commands: Sequence[Sequence[str]] = (tuple(fake_command),) if fake_command is not None else plan.commands
    if plan.atomic_bundle and fake_command is None:
        try:
            _materialize_atomic_debug_bundle(request, inputs, output)
        except (AdapterError, OSError) as exc:
            error = exc if isinstance(exc, AdapterError) else AdapterError(
                "debug_bundle_failed", f"调试包复制失败: {type(exc).__name__}: {exc}"
            )
            return make_result(
                request, error.status, error_code=error.code, message=str(error),
                diagnostics=_diagnostic_paths(output), cleanup_completed=True,
            ), plan
    error_code, message, cleanup = _run_commands(
        commands, output, plan.environment, timeout_seconds=_timeout_seconds(request),
        cancel_file=cancel_file, poll_seconds=poll_seconds,
    )
    diagnostics = _diagnostic_paths(output)
    if error_code == "canceled":
        return make_result(
            request, "canceled", error_code="canceled", message=message,
            diagnostics=diagnostics, cleanup_completed=cleanup,
            compatibility_facade=plan.compatibility_facade,
        ), plan
    if error_code is not None:
        status = "failed_terminal" if error_code == "worker_launch_failed" else "failed_retryable"
        return make_result(
            request, status, error_code=error_code, message=message,
            diagnostics=diagnostics, cleanup_completed=cleanup,
            compatibility_facade=plan.compatibility_facade,
        ), plan
    try:
        # Artifact publication assumes immutable inputs.  Re-hash after the
        # child exits so an in-place replacement cannot be committed silently.
        verified_inputs = resolve_inputs(request, root)
        build_plan(request, verified_inputs, output, allow_runtime_mutations=True)
        status, inspection_code, inspection_message = _inspect_outputs(plan, output)
        artifacts = _collect_artifacts(output)
        return make_result(
            request, status, artifacts=artifacts, error_code=inspection_code, message=inspection_message,
            diagnostics=diagnostics, cleanup_completed=True,
            compatibility_facade=plan.compatibility_facade,
        ), plan
    except AdapterError as exc:
        return make_result(
            request, exc.status, error_code=exc.code, message=str(exc),
            diagnostics=diagnostics, cleanup_completed=True,
            compatibility_facade=plan.compatibility_facade,
        ), plan


def _identity_for_failure(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        _safe_id(value.get("batchId"), "batchId")
        _safe_id(value.get("projectId"), "projectId")
        if value.get("taskId") not in FORMAL_TASK_IDS:
            return None
        if value.get("stage") not in ALL_STAGE_NAMES:
            return None
        _nonempty(value.get("contractVersion"), "contractVersion")
        if not isinstance(value.get("attempt"), int) or isinstance(value["attempt"], bool) or value["attempt"] < 1:
            return None
        _nonempty(value.get("leaseToken"), "leaseToken")
    except AdapterError:
        return None
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行一个 InsertAny3D stage-request v1")
    parser.add_argument("--request", required=True, type=Path, help="stage-request v1 JSON")
    parser.add_argument("--artifact-root", required=True, type=Path, help="所有相对输入和 staging 的允许根目录")
    parser.add_argument("--result", type=Path, help="stage-result 输出；正常情况默认写入 outputStagingDir/stage_result.json")
    parser.add_argument("--dry-run", action="store_true", help="严格校验后只输出白名单命令，不启动子进程")
    parser.add_argument("--fake-command-json", help="测试专用：用 JSON 字符串数组替换整个 stage 命令")
    parser.add_argument("--cancel-file", type=Path, help="文件出现时取消当前子进程组")
    parser.add_argument("--poll-seconds", type=float, default=0.1, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _main(args: argparse.Namespace) -> int:
    root = args.artifact_root.resolve()
    raw: Any = None
    result_path = args.result.resolve() if args.result else None
    try:
        if args.poll_seconds <= 0:
            raise AdapterError("invalid_stage_config", "poll-seconds 必须为正数")
        raw = json.loads(args.request.read_text(encoding="utf-8"))
        request = validate_request(raw)
        _, output = _resolve_relative(root, request["outputStagingDir"], "outputStagingDir")
        expected_result = output / "stage_result.json"
        if result_path is not None and result_path != expected_result.resolve():
            raise AdapterError("unsafe_path", "有效请求的 --result 必须是 outputStagingDir/stage_result.json")
        result_path = expected_result
        inputs = resolve_inputs(request, root)
        plan = build_plan(request, inputs, output)
        if args.dry_run:
            print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
            return 0
        fake_command = None
        if args.fake_command_json is not None:
            value = json.loads(args.fake_command_json)
            if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
                raise AdapterError("invalid_stage_config", "fake-command-json 必须是非空字符串数组")
            fake_command = value
        result, _ = execute_request(
            request, root, fake_command=fake_command, cancel_file=args.cancel_file,
            poll_seconds=args.poll_seconds,
        )
        _atomic_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (AdapterError, json.JSONDecodeError, OSError) as exc:
        error = exc if isinstance(exc, AdapterError) else AdapterError("invalid_stage_request", f"{type(exc).__name__}: {exc}")
        identity = _identity_for_failure(raw)
        if identity is not None:
            try:
                if result_path is None:
                    _, output = _resolve_relative(root, identity.get("outputStagingDir"), "outputStagingDir")
                    result_path = output / "stage_result.json"
                elif not _within(result_path, root):
                    raise AdapterError("unsafe_path", "--result 必须位于 artifact-root 内")
                result = make_result(identity, error.status, error_code=error.code, message=str(error))
                _atomic_json(result_path, result)
                print(json.dumps(result, ensure_ascii=False))
                return 0
            except (AdapterError, OSError):
                pass
        print(f"STAGE_ADAPTER_ERROR {error.code}: {error}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    global _TERMINATION_SIGNAL
    _TERMINATION_SIGNAL = None
    previous_handlers: dict[signal.Signals, Any] = {}

    def record_termination(signum: int, _frame: Any) -> None:
        global _TERMINATION_SIGNAL
        _TERMINATION_SIGNAL = signum

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, record_termination)
    try:
        return _main(args)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
