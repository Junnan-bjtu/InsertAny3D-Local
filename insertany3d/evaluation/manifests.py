"""Discovery and preflight validation for ``eval6-v1`` manifests."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from insertany3d.contracts import (
    ContractError,
    load_manifest,
    validate_evaluation_manifest,
)
from insertany3d.contracts.models import canonical_sha256


MANIFEST_NAME = "evaluation_manifest.json"


class EvaluationError(RuntimeError):
    """Evaluation inputs are incomplete, inconsistent, or unsafe to use."""


@dataclass(frozen=True)
class EvaluationManifest:
    """A validated manifest and its stable identities."""

    path: Path
    data: dict[str, Any]
    manifest_sha256: str
    comparison_config_sha256: str

    @property
    def batch_id(self) -> str:
        return str(self.data["batchId"])

    @property
    def project_id(self) -> str:
        return str(self.data["projectId"])

    @property
    def task_id(self) -> str:
        return str(self.data["taskId"])

    @property
    def method_id(self) -> str:
        return str(self.data["methodId"])

    @property
    def run_id(self) -> str:
        return str(self.data["runId"])

    @property
    def scene_path(self) -> str:
        return str(self.data["scenePath"])

    @property
    def task_key(self) -> str:
        return f"{self.project_id}/{self.task_id}/{self.method_id}"

    @property
    def scene_key(self) -> str:
        return f"{self.project_id}:{self.scene_path}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_evaluation_manifests(
    root: str | Path,
    *,
    manifest_name: str = MANIFEST_NAME,
) -> list[EvaluationManifest]:
    """Find and fully validate manifests before any paid request is planned."""

    source = Path(root)
    _reject_symlink_path(source, label="评测输入")
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.rglob(manifest_name))
    else:
        raise EvaluationError(f"评测输入不存在: {source}")
    if not paths:
        raise EvaluationError(f"没有找到 {manifest_name}: {source}")

    records = [load_evaluation_manifest(path) for path in paths]
    validate_manifest_collection(records)
    return records


def load_evaluation_manifest(path: str | Path) -> EvaluationManifest:
    """Load one manifest and verify every referenced image and camera file."""

    source_path = Path(path)
    _reject_symlink_path(source_path, label="评测清单")
    manifest_path = source_path.resolve()
    try:
        data = validate_evaluation_manifest(load_manifest(manifest_path))
    except (OSError, ContractError, ValueError) as exc:
        raise EvaluationError(f"评测清单无效 {manifest_path}: {exc}") from exc

    root = manifest_path.parent
    render = data["render"]
    expected_size = (int(render["width"]), int(render["height"]))
    referenced_paths: set[Path] = set()
    camera_documents: list[dict[str, Any]] = []
    camera_fingerprints: set[str] = set()

    for view in data["views"]:
        for field in ("original", "inserted", "camera"):
            reference = view[field]
            target = _resolve_reference(root, reference["path"])
            if target in referenced_paths:
                raise EvaluationError(
                    f"{manifest_path}: 多个视角或角色重复引用同一文件: {reference['path']}"
                )
            referenced_paths.add(target)
            _verify_file(target, reference["sha256"], manifest_path)
            if field == "camera":
                document, fingerprint = _load_camera_document(target, view, expected_size)
                if fingerprint in camera_fingerprints:
                    raise EvaluationError(f"{manifest_path}: 六个视角包含重复的相机记录")
                camera_fingerprints.add(fingerprint)
                camera_documents.append(document)
            else:
                actual_size = read_image_size(target)
                if actual_size != expected_size:
                    raise EvaluationError(
                        f"{manifest_path}: {reference['path']} 尺寸为 {actual_size[0]}x{actual_size[1]}，"
                        f"清单要求 {expected_size[0]}x{expected_size[1]}"
                    )

    if len(camera_documents) != 6:
        raise EvaluationError(f"{manifest_path}: 必须恰好验证 6 个相机记录")

    comparison_config = {
        "protocol": data["protocol"],
        "viewConfig": data["viewConfig"],
        "render": data["render"],
    }
    return EvaluationManifest(
        path=manifest_path,
        data=data,
        manifest_sha256=canonical_sha256(data),
        comparison_config_sha256=canonical_sha256(comparison_config),
    )


def validate_manifest_collection(records: Iterable[EvaluationManifest]) -> dict[str, Any]:
    """Reject ambiguous tasks and camera settings that cannot be compared."""

    items = list(records)
    if not items:
        raise EvaluationError("评测集合为空")

    batches = {item.batch_id for item in items}
    if len(batches) != 1:
        raise EvaluationError(f"一次评测只能包含一个 batchId，实际为: {sorted(batches)}")
    configs = {item.comparison_config_sha256 for item in items}
    if len(configs) != 1:
        raise EvaluationError("评测集合的 viewConfig 或 render 配置不一致，禁止混合汇总")

    task_keys: set[str] = set()
    project_scenes: dict[str, str] = {}
    for item in items:
        if item.task_key in task_keys:
            raise EvaluationError(f"任务/方法重复，无法确定应评测哪个 run: {item.task_key}")
        task_keys.add(item.task_key)
        previous_scene = project_scenes.setdefault(item.project_id, item.scene_path)
        if previous_scene != item.scene_path:
            raise EvaluationError(
                f"同一 Project 的 scenePath 不一致: {item.project_id}: "
                f"{previous_scene!r} / {item.scene_path!r}"
            )

    return {
        "batchId": next(iter(batches)),
        "manifests": len(items),
        "projects": len(project_scenes),
        "methods": sorted({item.method_id for item in items}),
        "comparisonConfigSha256": next(iter(configs)),
    }


def read_image_size(path: str | Path) -> tuple[int, int]:
    """Read PNG/JPEG dimensions without decoding or adding a heavy dependency."""

    image_path = Path(path)
    with image_path.open("rb") as stream:
        header = stream.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            if len(header) < 24 or header[12:16] != b"IHDR":
                raise EvaluationError(f"PNG 头损坏: {image_path}")
            width, height = struct.unpack(">II", header[16:24])
            if width < 1 or height < 1:
                raise EvaluationError(f"PNG 尺寸无效: {image_path}")
            return width, height
        if header[:2] == b"\xff\xd8":
            stream.seek(2)
            return _read_jpeg_size(stream, image_path)
    raise EvaluationError(f"eval6 RGB 只支持可验证的 PNG/JPEG 文件: {image_path}")


def _read_jpeg_size(stream: Any, path: Path) -> tuple[int, int]:
    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while True:
        marker_start = stream.read(1)
        if not marker_start:
            break
        if marker_start != b"\xff":
            continue
        marker_bytes = stream.read(1)
        while marker_bytes == b"\xff":
            marker_bytes = stream.read(1)
        if not marker_bytes:
            break
        marker = marker_bytes[0]
        if marker in {0x01, *range(0xD0, 0xDA)}:
            continue
        length_bytes = stream.read(2)
        if len(length_bytes) != 2:
            break
        length = struct.unpack(">H", length_bytes)[0]
        if length < 2:
            break
        if marker in start_of_frame:
            payload = stream.read(5)
            if len(payload) != 5:
                break
            height, width = struct.unpack(">HH", payload[1:5])
            if width > 0 and height > 0:
                return width, height
            break
        stream.seek(length - 2, 1)
    raise EvaluationError(f"JPEG 尺寸头损坏: {path}")


def _resolve_reference(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    _reject_symlink_path(candidate, root=root, label="评测文件")
    target = candidate.resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise EvaluationError(f"评测文件越出清单目录: {relative_path}") from exc
    return target


def _verify_file(path: Path, expected_sha256: str, manifest_path: Path) -> None:
    if not path.is_file():
        raise EvaluationError(f"{manifest_path}: 缺少评测文件: {path}")
    if path.stat().st_size <= 0:
        raise EvaluationError(f"{manifest_path}: 评测文件为空: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise EvaluationError(
            f"{manifest_path}: 文件 SHA-256 不匹配: {path.name}，"
            f"清单 {expected_sha256}，实际 {actual}"
        )


def _load_camera_document(
    path: Path,
    view: Mapping[str, Any],
    expected_size: tuple[int, int],
) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"相机 JSON 无法解析: {path}: {exc}") from exc
    if not isinstance(value, Mapping) or not value:
        raise EvaluationError(f"相机 JSON 必须是非空对象: {path}")
    document = dict(value)

    _match_required(document, ("viewId", "view_id"), view["viewId"], path)
    _match_required_number(
        document,
        ("pitchDegrees", "pitch_degrees"),
        float(view["pitchDegrees"]),
        path,
    )
    _match_required_number(
        document,
        ("yawOffsetDegrees", "yaw_offset_degrees"),
        float(view["yawOffsetDegrees"]),
        path,
    )
    _match_required_number(document, ("width", "pixelWidth"), expected_size[0], path)
    _match_required_number(document, ("height", "pixelHeight"), expected_size[1], path)
    camera_to_world = _required_camera_matrix(
        document,
        ("cameraToWorldMatrix", "camera_to_world_matrix"),
        path,
    )
    projection = _required_camera_matrix(
        document,
        ("projectionMatrix", "projection_matrix"),
        path,
    )
    fingerprint = canonical_sha256(
        {
            "cameraToWorldMatrix": camera_to_world,
            "projectionMatrix": projection,
        }
    )
    return document, fingerprint


def _match_required(
    document: Mapping[str, Any], keys: tuple[str, ...], expected: Any, path: Path
) -> None:
    found = False
    for key in keys:
        if key not in document:
            continue
        found = True
        if document[key] != expected:
            raise EvaluationError(
                f"相机 JSON {path} 的 {key}={document[key]!r}，应为 {expected!r}"
            )
    if not found:
        raise EvaluationError(f"相机 JSON {path} 缺少字段: {keys[0]}")


def _match_required_number(
    document: Mapping[str, Any], keys: tuple[str, ...], expected: float, path: Path
) -> None:
    found = False
    for key in keys:
        if key not in document:
            continue
        found = True
        actual = document[key]
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), float(expected), abs_tol=1e-6)
        ):
            raise EvaluationError(
                f"相机 JSON {path} 的 {key}={actual!r}，应为 {expected:g}"
            )
    if not found:
        raise EvaluationError(f"相机 JSON {path} 缺少字段: {keys[0]}")


def _required_camera_matrix(
    document: Mapping[str, Any], keys: tuple[str, ...], path: Path
) -> list[int | float]:
    found = [(key, document[key]) for key in keys if key in document]
    if not found:
        raise EvaluationError(f"相机 JSON {path} 缺少字段: {keys[0]}")
    normalized = [_validate_numeric_tree(value, f"{path}:{key}") for key, value in found]
    first = normalized[0]
    if any(value != first for value in normalized[1:]):
        raise EvaluationError(f"相机 JSON {path} 的别名字段内容不一致: {keys}")
    if not isinstance(first, list) or len(first) != 16 or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in first
    ):
        raise EvaluationError(f"相机 JSON {path} 的 {keys[0]} 必须是 16 个有限数值组成的 4x4 矩阵")
    return first


def _validate_numeric_tree(value: Any, label: str) -> Any:
    if isinstance(value, bool):
        raise EvaluationError(f"{label} 必须只包含有限数值")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise EvaluationError(f"{label} 必须只包含有限数值")
        return value
    if isinstance(value, Mapping):
        if not value:
            raise EvaluationError(f"{label} 不能为空")
        return {
            str(key): _validate_numeric_tree(item, f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if not value:
            raise EvaluationError(f"{label} 不能为空")
        return [_validate_numeric_tree(item, f"{label}[{index}]") for index, item in enumerate(value)]
    raise EvaluationError(f"{label} 必须只包含有限数值")


def _reject_symlink_path(
    path: Path, *, root: Path | None = None, label: str
) -> None:
    """Reject symlinks so validated inputs cannot silently point outside their root."""

    if root is None:
        absolute = path.absolute()
        current = Path(absolute.anchor)
        parts = absolute.parts[1:]
    else:
        current = root
        try:
            parts = path.relative_to(root).parts
        except ValueError as exc:
            raise EvaluationError(f"{label} 越出允许目录: {path}") from exc
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise EvaluationError(f"{label} 不允许使用符号链接: {current}")
