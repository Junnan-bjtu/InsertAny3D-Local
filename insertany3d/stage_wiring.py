"""Build versioned worker requests from scheduler state.

The scheduler stores stage configuration and committed artifacts separately.
This module is the small boundary that joins them into the JSON request used
by the Unity runner or the remote ``stage_adapter.py``.  Building a command is
side-effect free; the CLI decides whether a generated Unity command may run.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractError, validate_stage_request
from .contracts.models import canonical_sha256
from .dag import STAGE_INDEX
from .image_edit import generation_count


class StageWiringError(ValueError):
    """The scheduler cannot construct a safe worker request."""


def unity_command_path(value: str | Path) -> str:
    """Convert a WSL path to a path a Windows Unity process can open."""
    text = str(value)
    drive_match = re.match(r"^([A-Za-z]):[\\/]*(.*)$", text)
    if drive_match:
        rest = drive_match.group(2).replace("/", "\\")
        return drive_match.group(1).upper() + ":\\" + rest if rest else drive_match.group(1).upper() + ":\\"
    mnt_match = re.match(r"^/mnt/([A-Za-z])(?:/(.*))?$", text)
    if mnt_match:
        rest = (mnt_match.group(2) or "").replace("/", "\\")
        return mnt_match.group(1).upper() + ":\\" + rest if rest else mnt_match.group(1).upper() + ":\\"
    wslpath = shutil.which("wslpath")
    if wslpath and text.startswith("/"):
        try:
            converted = subprocess.run(
                [wslpath, "-w", text],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout.strip()
            if converted:
                return converted
        except (OSError, subprocess.CalledProcessError):
            pass
    return text


def _is_windows_executable(executable: str) -> bool:
    value = str(executable).replace("/", "\\")
    # A bare ``Unity.exe`` may be resolved by PATH on either host; only a
    # path-bearing executable gives us enough evidence to rewrite arguments.
    return bool(re.match(r"^[A-Za-z]:\\", value)) or ("\\" in value and value.lower().endswith(".exe"))


def _wsl_command_executable(executable: str) -> str:
    """Make a Windows drive path runnable by WSL's subprocess launcher."""
    if os.name == "nt":
        return executable
    if re.match(r"^[A-Za-z]:[\\/]", str(executable)):
        return str(_host_path(executable))
    return executable


def _project_entry(store: Any, batch_id: str, project_id: str) -> Mapping[str, Any]:
    batch = store.row("SELECT manifest_json FROM batches WHERE batch_id=?", (batch_id,))
    if batch is None:
        raise StageWiringError(f"batch 不存在: {batch_id}")
    manifest = json.loads(batch["manifest_json"])
    project = next((p for p in manifest["projects"] if p["projectId"] == project_id), None)
    if project is None:
        raise StageWiringError(f"找不到 Project: {project_id}")
    return project


def validate_unity_project_for_batch(store: Any, batch_id: str, project_id: str) -> Path:
    """Validate project files before taking a queue lease for Unity."""
    project = _project_entry(store, batch_id, project_id)
    project_root = _host_path(project["projectPath"]).resolve()
    if not project_root.is_dir():
        raise StageWiringError(f"Unity Project 路径不存在: {project_root}")
    package_manifest = project_root / "Packages" / "manifest.json"
    if not package_manifest.is_file():
        raise StageWiringError(f"Unity Project 缺少 Packages/manifest.json: {project_root}")
    try:
        packages = json.loads(package_manifest.read_text(encoding="utf-8")).get("dependencies", {})
    except (OSError, json.JSONDecodeError) as exc:
        raise StageWiringError(f"Unity 包清单无法读取: {package_manifest}") from exc
    if "com.junnan.insertany3d" not in packages:
        raise StageWiringError("Unity Project 尚未安装 com.junnan.insertany3d 公共包")
    scene = str(project.get("scenePath") or "").replace("\\", "/")
    scene_file = (project_root / scene).resolve()
    if not scene_file.is_file() or not scene_file.is_relative_to(project_root):
        raise StageWiringError(f"Unity 场景不存在或越出 Project: {scene_file}")
    return project_root


def _host_path(value: str | Path) -> Path:
    text = str(value)
    match = re.match(r"^[A-Za-z]:[\\/]", text)
    if match:
        drive = text[0].lower()
        rest = text[2:].lstrip("\\/").replace("\\", "/")
        return Path("/mnt") / drive / rest
    return Path(text)


def validate_unity_project(store: Any, item: Any) -> Path:
    """Check the declared Unity project, scene and package before launching."""
    project = _project_entry(store, item.batch_id, item.project_id)
    project_root = validate_unity_project_for_batch(store, item.batch_id, item.project_id)
    scene = str(project["scenePath"]).replace("\\", "/")
    scene_file = (project_root / scene).resolve()
    if not scene_file.is_file() or not scene_file.is_relative_to(project_root):
        raise StageWiringError(f"Unity 场景不存在或越出 Project: {scene_file}")
    return project_root


def ensure_unity_project_not_running(project_root: str | Path) -> None:
    """Refuse a launch when another Unity process already owns this project."""
    target = unity_command_path(project_root).rstrip("\\/").lower()
    script = (
        "$target = '" + target.replace("'", "''") + "'; "
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'Unity.exe' } | "
        "ForEach-Object { $line = [string]$_.CommandLine; "
        "if ($line.IndexOf($target, [StringComparison]::OrdinalIgnoreCase) -ge 0) { "
        "[PSCustomObject]@{ ProcessId = $_.ProcessId; CommandLine = $line } } } | "
        "ConvertTo-Json -Compress"
    )
    candidates = ["powershell.exe", "pwsh", "powershell"]
    executable = next((shutil.which(name) for name in candidates if shutil.which(name)), None)
    if executable is None:
        raise StageWiringError("无法确认 Unity 项目是否已打开（找不到 PowerShell）；为避免启动第二个实例，已拒绝执行")
    try:
        completed = subprocess.run(
            [executable, "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise StageWiringError(f"无法检查 Unity 项目占用状态：{exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "PowerShell 查询失败"
        raise StageWiringError(f"无法检查 Unity 项目占用状态：{detail}")
    output = completed.stdout.strip()
    if not output or output == "null":
        return
    try:
        values = json.loads(output)
    except json.JSONDecodeError as exc:
        raise StageWiringError("无法解析 Unity 项目占用状态；为避免启动第二个实例，已拒绝执行") from exc
    entries = values if isinstance(values, list) else [values]
    pids = ", ".join(str(entry.get("ProcessId")) for entry in entries if isinstance(entry, dict))
    raise StageWiringError(f"Unity 项目已被打开（PID {pids or '未知'}）；请关闭现有实例后再执行")


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_UNITY_STAGES = {"unity_anchor", "unity_apply", "unity_eval6"}
_REMOTE_STAGES = {
    "model_generation",
    "render_alignment_views",
    "segment_inputs",
    "gim_match",
    "estimate_pose",
    "sags_segment_vote",
    "debug_bundle",
}
_VIEWS = ("left", "center", "right")
_RING_VIEWS = ("center", "ring_060", "ring_120", "ring_180", "ring_240", "ring_300")
_RING_YAW_OFFSETS = (0, 60, 120, 180, 240, 300)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_alias(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value[:120] or "artifact"


def _stage_relative(path: str, stage: str, idempotency_key: str) -> str:
    """Return a path relative to one committed stage output directory."""
    parts = Path(path).parts
    marker = ("artifacts", stage, idempotency_key)
    for index in range(len(parts) - len(marker) + 1):
        if parts[index : index + len(marker)] == marker:
            suffix = Path(*parts[index + len(marker) :])
            if suffix == Path("."):
                break
            return suffix.as_posix()
    # Older/manual artifacts may not contain the normal marker.  Keep the
    # original path as a deterministic fallback rather than guessing.
    return Path(path).as_posix()


def _records(
    store: Any,
    item: Any,
    root: Path,
    *,
    accepted_generation_path: str | None = None,
) -> list[dict[str, Any]]:
    rows = store.rows(
        """
        SELECT a.artifact_id AS sourceArtifactId, a.type, a.relative_path AS path,
               a.sha256, a.size, s.name AS sourceStage, s.idempotency_key AS sourceKey,
               s.sort_index
          FROM artifacts a
          JOIN stages s ON s.id=a.stage_id
          JOIN attempts attempt ON attempt.id=a.attempt_id
         WHERE s.batch_id=? AND s.project_id=? AND s.task_id=?
           AND s.sort_index<? AND attempt.status='succeeded'
           AND (
               s.name='image_edit'
               OR attempt.attempt_number=(
                   SELECT MAX(latest.attempt_number) FROM attempts latest
                    WHERE latest.stage_id=s.id AND latest.status='succeeded'
               )
           )
         ORDER BY s.sort_index, a.id
        """,
        (item.batch_id, item.project_id, item.task_id, STAGE_INDEX[item.stage]),
    )
    # ``WorkItem`` intentionally contains only lease data.  The caller adds
    # ``sort_index`` when it loads the item, while this fallback keeps the
    # helper useful for older WorkItem-like test doubles.
    result: list[dict[str, Any]] = []
    selected = Path(str(accepted_generation_path)).resolve() if accepted_generation_path else None
    for row in rows:
        path = Path(str(row["path"]))
        full = root / path
        if not full.is_file():
            raise StageWiringError(f"上游 artifact 不存在: {path}")
        if selected is not None and row["sourceStage"] == "image_edit" and full.resolve() != selected:
            continue
        value = dict(row)
        value["path"] = path.as_posix()
        value["relative"] = _stage_relative(value["path"], str(row["sourceStage"]), str(row["sourceKey"]))
        result.append(value)
    return result


def _assign_aliases(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    used: set[str] = set()
    by_alias: dict[str, dict[str, Any]] = {}

    def add(record: dict[str, Any], preferred: str) -> None:
        alias = _safe_alias(preferred)
        if alias in used:
            suffix = 2
            while f"{alias}_{suffix}" in used:
                suffix += 1
            alias = f"{alias}_{suffix}"
        used.add(alias)
        record["artifactId"] = alias
        by_alias[alias] = record

    for record in records:
        stage = record["sourceStage"]
        relative = str(record["relative"]).replace("\\", "/")
        lower = relative.lower()
        preferred = str(record["sourceArtifactId"])
        if stage == "image_edit" and Path(relative).suffix.lower() in _IMAGE_SUFFIXES:
            preferred = "input_image"
        elif stage == "unity_anchor":
            # Unity outputs may be rooted at the legacy step1 directory or at
            # the canonical task workspace. Match the view-relative suffix so
            # both layouts produce stable scene_* artifact aliases.
            match = re.search(r"(?:^|/)(left|center|right)/image(?:\.raw|\.camera\.json|\.png)?$", lower)
            if match:
                view = match.group(1)
                if lower.endswith("image.png"):
                    preferred = f"scene_{view}_image"
                elif lower.endswith("image.raw"):
                    preferred = f"scene_{view}_depth"
                else:
                    preferred = f"scene_{view}_camera"
        elif stage == "render_alignment_views":
            if lower == "ring6/model/cfg_args":
                preferred = "sags_model_cfg_args"
            elif lower.startswith("ring6/source/images/"):
                match = re.fullmatch(r"ring6/source/images/(center|ring_\d{3})\.(?:png|jpg|jpeg)", lower)
                if match:
                    preferred = f"sags_{match.group(1)}_image"
            elif lower.endswith("source/sparse/0/cameras.txt") and not lower.startswith("ring6/"):
                preferred = "generated_cameras"
            elif lower.endswith("source/sparse/0/images.txt") and not lower.startswith("ring6/"):
                preferred = "generated_images"
            else:
                match = re.fullmatch(r"source/images/(left|center|right)\.(?:png|jpg|jpeg)", lower)
                if match:
                    preferred = f"generated_{match.group(1)}_image"
                else:
                    match = re.fullmatch(
                        r"(?:source/)?(?:depth|depths)/(?:absdepth/)?(left|center|right)\.[^/]+",
                        lower,
                    )
                    if match:
                        preferred = f"generated_{match.group(1)}_depth"
        elif stage == "segment_inputs":
            match = re.search(r"(left|center|right)/(scene|generated)/(mask|points)\.[^/]+$", lower)
            if match:
                preferred = f"anchor_{match.group(1)}_{match.group(2)}_{match.group(3)}"
            else:
                match = re.fullmatch(
                    r"sags_annotations/(center|ring_\d{3})/(mask|points)\.[^/]+",
                    lower,
                )
                if match:
                    preferred = f"sags_{match.group(1)}_{match.group(2)}"
        elif stage == "gim_match":
            match = re.search(r"(left|center|right)/(?:matches|match|warp)\.[^/]+$", lower)
            if match:
                preferred = f"gim_{match.group(1)}_{Path(relative).stem}"
        elif stage == "estimate_pose" and lower.endswith("pose.json"):
            preferred = "pose"
        elif stage == "sags_segment_vote":
            if lower.endswith("model/cfg_args") or lower.endswith("cfg_args"):
                preferred = "model_cfg_args"
            else:
                match = re.search(r"(?:^|/)(center|ring_\d{3})/(?:mask|points)\.[^/]+$", lower)
                if match:
                    preferred = f"sags_{match.group(1)}_{Path(relative).stem}"
        elif stage == "debug_bundle" and lower.endswith("bundle_manifest.json"):
            preferred = "batch_manifest"
        add(record, preferred)
    return by_alias


def _first(records: Mapping[str, dict[str, Any]], predicate) -> str | None:
    for alias, record in records.items():
        if predicate(alias, record):
            return alias
    return None


def _refs(records: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "artifactId": alias,
            "type": record["type"],
            "path": record["path"],
            "sha256": record["sha256"],
            "size": int(record["size"]),
        }
        for alias, record in records.items()
    ]


def _stage_options(stage: str, options: dict[str, Any], records: Mapping[str, dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill only deterministic defaults; explicit task options remain authoritative."""
    if stage == "model_generation":
        # Composite TRELLIS input must contain both the inserted object and
        # the registration anchor.  The older monolithic pipeline performed
        # this union mask immediately before generation; keep the same rule
        # in the atomic stage when an anchor prompt is available.
        task = config.get("task") or {}
        prompts = task.get("trellisMaskPrompts")
        if prompts is None and task.get("anchorMaskPrompt"):
            prompts = [task.get("objectPrompt"), task.get("anchorMaskPrompt")]
        if isinstance(prompts, list):
            prompts = [str(value).strip() for value in prompts if str(value).strip()]
            if prompts:
                options.setdefault("trellisMaskPrompts", prompts)
                options.setdefault("inputMaskEngine", "legacy")
        return options
    elif stage == "render_alignment_views":
        ply = _first(records, lambda alias, row: alias == "sample_ply" or row["relative"].lower().endswith("sample.ply"))
        if ply:
            options.setdefault("inputPlyArtifactId", ply)
        options.setdefault("viewNames", list(_VIEWS))
        render_protocol = config.get("renderProtocol") or {}
        yaw = render_protocol.get("viewYawOffsetDegrees", 24)
        if options.get("viewNames") == list(_VIEWS):
            options.setdefault("yawOffsets", [-yaw, 0, yaw])
        options.setdefault("ringViewNames", list(_RING_VIEWS))
        options.setdefault("ringYawOffsets", list(_RING_YAW_OFFSETS))
    elif stage == "segment_inputs":
        scene_images = {view: _first(records, lambda alias, row, view=view: alias == f"scene_{view}_image") for view in _VIEWS}
        generated_images = {view: _first(records, lambda alias, row, view=view: alias == f"generated_{view}_image") for view in _VIEWS}
        if all(scene_images.values()) and all(generated_images.values()):
            prompt = (config.get("task") or {}).get("anchorMaskPrompt") or (config.get("task") or {}).get("objectPrompt")
            if prompt:
                options.setdefault("mode", "anchor")
                options.setdefault("prompt", prompt)
                options.setdefault("views", [
                    {"name": view, "sceneArtifactId": scene_images[view], "generatedArtifactId": generated_images[view]}
                    for view in _VIEWS
                ])
        else:
            image = _first(records, lambda alias, row: alias == "input_image" or Path(row["relative"]).suffix.lower() in _IMAGE_SUFFIXES)
            if image:
                options.setdefault("mode", "target")
                options.setdefault("inputImageArtifactId", image)
                task = config.get("task") or {}
                if task.get("objectPrompt"):
                    options.setdefault("taskPrompt", task["objectPrompt"])
        ring_images = {
            view: _first(records, lambda alias, row, view=view: alias == f"sags_{view}_image")
            for view in _RING_VIEWS
        }
        if all(ring_images.values()):
            options.setdefault(
                "sagsViews",
                [{"name": view, "imageArtifactId": ring_images[view]} for view in _RING_VIEWS],
            )
            task = config.get("task") or {}
            if task.get("objectPrompt"):
                options.setdefault("sagsTaskPrompt", task["objectPrompt"])
    elif stage == "gim_match":
        pairs = []
        for view in _VIEWS:
            scene = _first(records, lambda alias, row, view=view: alias == f"scene_{view}_image")
            generated = _first(records, lambda alias, row, view=view: alias == f"generated_{view}_image")
            scene_mask = _first(records, lambda alias, row, view=view: alias == f"anchor_{view}_scene_mask")
            generated_mask = _first(records, lambda alias, row, view=view: alias == f"anchor_{view}_generated_mask")
            if scene and generated:
                pair: dict[str, Any] = {"name": view, "image0ArtifactId": scene, "image1ArtifactId": generated}
                if scene_mask:
                    pair["mask0ArtifactId"] = scene_mask
                if generated_mask:
                    pair["mask1ArtifactId"] = generated_mask
                pairs.append(pair)
        if pairs:
            options.setdefault("pairs", pairs)
    elif stage == "estimate_pose":
        cameras = _first(records, lambda alias, row: alias == "generated_cameras")
        images = _first(records, lambda alias, row: alias == "generated_images")
        if cameras and images:
            options.setdefault("generatedCamerasArtifactId", cameras)
            options.setdefault("generatedImagesArtifactId", images)
        views = []
        for view in _VIEWS:
            matches = _first(records, lambda alias, row, view=view: alias.startswith(f"gim_{view}_") and "matches" in row["relative"].lower())
            scene_depth = _first(records, lambda alias, row, view=view: alias == f"scene_{view}_depth")
            scene_camera = _first(records, lambda alias, row, view=view: alias == f"scene_{view}_camera")
            generated_depth = _first(records, lambda alias, row, view=view: alias == f"generated_{view}_depth")
            if all((matches, scene_depth, scene_camera, generated_depth)):
                views.append({"name": view, "matchesArtifactId": matches, "sceneDepthArtifactId": scene_depth, "sceneCameraArtifactId": scene_camera, "generatedDepthArtifactId": generated_depth})
        if len(views) == 3:
            options.setdefault("views", views)
    elif stage == "sags_segment_vote":
        marker = _first(records, lambda alias, row: alias == "sags_model_cfg_args")
        if marker is None:
            marker = _first(records, lambda alias, row: alias == "model_cfg_args")
        if marker:
            options.setdefault("modelMarkerArtifactId", marker)
        annotations = []
        for view in _RING_VIEWS:
            mask = _first(records, lambda alias, row, view=view: alias == f"sags_{view}_mask")
            points = _first(records, lambda alias, row, view=view: alias == f"sags_{view}_points")
            if mask and points:
                annotations.append({"name": view, "maskArtifactId": mask, "pointsArtifactId": points})
        if len(annotations) == 6:
            options.setdefault("annotations", annotations)
            options.setdefault("minVotes", 3)
            options.setdefault("independentMinPriorCoverage", 0.25)
    elif stage == "debug_bundle":
        # The 15-stage DAG does not produce the legacy monolithic pipeline's
        # batch_manifest.json.  Atomic mode indexes and copies every hashed,
        # committed upstream artifact instead of pretending that file exists.
        options.setdefault("mode", "atomic")
    return options


def _workspace_result_root(
    root: Path,
    item: Any,
    records: Mapping[str, dict[str, Any]],
) -> str:
    """Validate canonical task outputs and return the project workspace root.

    Unity resolves ``<resultRoot>/<taskId>/stages/...`` itself.  Keep this
    boundary manifest-driven and do not materialize a second legacy result
    tree.  The scheduler's artifact rows are still checked so an apply stage
    cannot accidentally consume an unrelated task's output.
    """

    def find_record(predicate: Any, label: str) -> dict[str, Any]:
        for record in records.values():
            if predicate(record):
                return record
        raise StageWiringError(f"unity_apply 缺少已提交的 {label} artifact")

    pose = find_record(
        lambda record: record.get("sourceStage") == "estimate_pose"
        and str(record["relative"]).replace("\\", "/").lower().endswith("pose.json"),
        "pose.json",
    )
    ply = find_record(
        lambda record: record.get("sourceStage") == "sags_segment_vote"
        and str(record["relative"]).replace("\\", "/").lower().endswith("inserted_object.ply"),
        "inserted_object.ply",
    )
    project_root = (root / _safe_alias(str(item.project_id))).resolve()
    task_root = (project_root / _safe_alias(str(item.task_id))).resolve()
    if not task_root.is_relative_to(root):
        raise StageWiringError("Unity workspace 越出 batch root")

    def verified_source(record: dict[str, Any], label: str) -> Path:
        source = (root / str(record["path"])).resolve()
        if not source.is_file() or not source.is_relative_to(root):
            raise StageWiringError(f"Unity workspace 输入不存在或越出 batch root: {source}")
        expected_sha = str(record.get("sha256") or "")
        if expected_sha:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != expected_sha:
                raise StageWiringError(f"{label} artifact SHA-256 不匹配: {source}")
        return source

    def publish(source: Path, destination: Path, label: str) -> None:
        """Materialize an immutable committed artifact at Unity's workspace path."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(source.read_bytes()).hexdigest():
                raise StageWiringError(f"Unity workspace 已有不同的 {label}: {destination}")
            return
        temporary = destination.with_name(f".{destination.name}.insertany3d-tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)

    task_manifest_path = task_root / "task_manifest.json"
    manifest_record = next(
        (record for record in records.values()
         if record.get("sourceStage") == "unity_anchor"
         and str(record.get("relative", "")).replace("\\", "/").lower().endswith("/task_manifest.json")),
        None,
    )
    if manifest_record is not None:
        manifest_source = verified_source(manifest_record, "task_manifest.json")
        publish(manifest_source, task_manifest_path, "task_manifest.json")
    elif not task_manifest_path.is_file():
        raise StageWiringError(f"Unity workspace 缺少 task manifest: {task_manifest_path}")
    try:
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageWiringError(f"无法读取 task manifest: {task_manifest_path}") from exc
    if task_manifest.get("projectId") != item.project_id or task_manifest.get("taskId") != item.task_id:
        raise StageWiringError(f"task manifest 与 unity_apply 任务不匹配: {task_manifest_path}")

    # The canonical paths are part of the server workspace contract.  Check
    # both files and their current stage manifests before launching Unity.
    expected = (
        ("estimate_pose", task_root / "stages" / "pose" / "output" / "pose.json", pose),
        ("sags_segment_vote", task_root / "stages" / "sags" / "output" / "inserted_object.ply", ply),
    )
    for stage, canonical, record in expected:
        source = verified_source(record, stage)
        publish(source, canonical, stage)
        manifest_path = task_root / "stages" / ("pose" if stage == "estimate_pose" else "sags") / "manifest.json"
        if not manifest_path.is_file():
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps({"status": "succeeded", "stage": stage, "sourceArtifact": str(record["path"])}, ensure_ascii=False),
                encoding="utf-8",
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StageWiringError(f"无法读取 {stage} manifest: {manifest_path}") from exc
        if manifest.get("status") not in {"succeeded", "ready"}:
            raise StageWiringError(f"{stage} manifest 尚未成功: {manifest_path}")
    return project_root.relative_to(root).as_posix()


def build_stage_request(store: Any, item: Any) -> dict[str, Any]:
    batch = store.row("SELECT manifest_json, root_path FROM batches WHERE batch_id=?", (item.batch_id,))
    if batch is None:
        raise StageWiringError(f"batch 不存在: {item.batch_id}")
    root = Path(batch["root_path"]).resolve()
    try:
        output_relative = item.staging_dir.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise StageWiringError("stage staging 目录必须位于 batch root 内") from exc
    config_row = store.row("SELECT effective_config_json FROM stages WHERE id=?", (item.stage_id,))
    if config_row is None:
        raise StageWiringError(f"stage 不存在: {item.stage_id}")
    config = json.loads(config_row["effective_config_json"])
    accepted_path = config.get("acceptedGenerationPath")
    if not accepted_path and item.stage != "image_edit":
        # The edit gate owns the decision.  Downstream stage configs are
        # intentionally immutable, so inherit the gate's exact path here.
        gate_row = store.row(
            """SELECT effective_config_json FROM stages
               WHERE batch_id=? AND project_id=? AND task_id=? AND name='edit_gate'""",
            (item.batch_id, item.project_id, item.task_id),
        )
        if gate_row is not None:
            gate_config = json.loads(gate_row["effective_config_json"])
            accepted_path = gate_config.get("acceptedGenerationPath")
    raw_records = _records(
        store,
        item,
        root,
        accepted_generation_path=accepted_path,
    )
    # Once edit review records an accepted path, carry exactly that file into
    # downstream requests.  This avoids relying on artifact insertion order.
    if accepted_path and not any(record.get("sourceStage") == "image_edit" for record in raw_records):
        raise StageWiringError(f"acceptedGenerationPath 不存在或不是 image_edit artifact: {accepted_path}")
    records = _assign_aliases(raw_records)
    options = dict(config.get("stageOptions") or {})
    options = _stage_options(item.stage, options, records, config)
    unity_result_root: str | None = None
    if item.stage == "unity_apply":
        unity_result_root = _workspace_result_root(root, item, records)
    if "remote_gpu" in item.resources:
        device = str(item.resources["remote_gpu"])
        if device.startswith("gpu:"):
            options.setdefault("gpuDevice", device[4:])
    effective = dict(config)
    if accepted_path and item.stage != "image_edit":
        effective["acceptedGenerationPath"] = str(Path(str(accepted_path)).resolve())
    if item.stage == "image_edit":
        # SQLite allocates generation identity so interleaved/retried workers
        # cannot accidentally publish duplicate candidate-1 metadata.
        task = dict(effective.get("task") or {})
        generation_index = getattr(item, "generation_index", None)
        generation_group_id = getattr(item, "generation_group_id", None)
        if generation_index is None or generation_group_id is None:
            attempt_row = store.row(
                "SELECT generation_group_id, generation_index FROM attempts WHERE id=?",
                (item.attempt_id,),
            )
            if attempt_row is not None:
                generation_index = attempt_row["generation_index"]
                generation_group_id = attempt_row["generation_group_id"]
        if generation_index is not None:
            task["generationIndex"] = int(generation_index)
        if generation_group_id:
            task["generationGroupId"] = str(generation_group_id)
        task["generationCount"] = generation_count(task)
        effective["task"] = task
    effective["stageOptions"] = options
    if unity_result_root is not None:
        # UnityStageRunner reads this as a top-level effectiveConfig field;
        # keeping it out of stageOptions also avoids confusing remote adapters.
        effective["resultRoot"] = unity_result_root
    request = {
        "schemaVersion": 1,
        "kind": "insertany3d.stage-request",
        "batchId": item.batch_id,
        "projectId": item.project_id,
        "taskId": item.task_id,
        "stage": item.stage,
        "contractVersion": item.contract_version,
        "attempt": item.attempt,
        "leaseToken": item.lease_token,
        "inputs": _refs(records),
        "effectiveConfig": effective,
        "effectiveConfigSha256": canonical_sha256(effective),
        "outputStagingDir": output_relative,
    }
    try:
        return validate_stage_request(request)
    except ContractError as exc:
        raise StageWiringError(f"无法构造 {item.stage} 请求: {exc}") from exc


def write_stage_request(store: Any, item: Any, path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    request = build_stage_request(store, item)
    batch = store.row("SELECT root_path FROM batches WHERE batch_id=?", (item.batch_id,))
    root = Path(batch["root_path"]).resolve()
    target = Path(path) if path is not None else root / "requests" / item.project_id / item.task_id / item.stage / f"attempt-{item.attempt:04d}.json"
    target = target.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise StageWiringError("stage request 必须写入 batch root 内") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return request, target


def build_stage_command(
    store: Any,
    item: Any,
    request_path: str | Path,
    *,
    unity_executable: str | None = None,
    adapter_path: str | Path | None = None,
    python_executable: str | None = None,
) -> list[str]:
    batch = store.row("SELECT root_path, manifest_json FROM batches WHERE batch_id=?", (item.batch_id,))
    if batch is None:
        raise StageWiringError(f"batch 不存在: {item.batch_id}")
    root = Path(batch["root_path"]).resolve()
    result_path = item.staging_dir.resolve() / "stage_result.json"
    request = Path(request_path).resolve()
    if item.stage in _UNITY_STAGES:
        manifest = json.loads(batch["manifest_json"])
        project = next((p for p in manifest["projects"] if p["projectId"] == item.project_id), None)
        if project is None:
            raise StageWiringError(f"找不到 Project: {item.project_id}")
        executable = unity_executable or os.environ.get("UNITY_EXECUTABLE") or "Unity"
        convert = _is_windows_executable(executable)
        path_value = lambda value: unity_command_path(value) if convert else str(value)
        return [
            _wsl_command_executable(executable),
            "-batchmode", "-quit",
            "-projectPath", path_value(project["projectPath"]),
            "-executeMethod", "InsertAny3D.Editor.InsertAny3DStageRunner.Run",
            "-insertAny3DRequest", path_value(request),
            "-insertAny3DResult", path_value(result_path),
            "-insertAny3DArtifactRoot", path_value(root),
        ]
    if item.stage in _REMOTE_STAGES:
        executable = python_executable or sys.executable
        adapter = Path(adapter_path) if adapter_path is not None else Path(__file__).resolve().parents[1] / "tools" / "stage_adapter.py"
        return [
            executable, str(adapter),
            "--request", str(request),
            "--artifact-root", str(root),
            "--result", str(result_path),
        ]
    raise StageWiringError(f"stage {item.stage} 当前没有可执行 worker 命令")
