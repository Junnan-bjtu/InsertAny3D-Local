"""Dependency-free validation for manifests shared by Python and Unity.

The JSON Schema files are the portable contract.  These validators add the
cross-field rules that JSON Schema cannot express clearly, such as the formal
12 x 5 batch shape and the exact eval6 camera layout.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
CONTRACT_KIND_BATCH = "insertany3d.batch"
CONTRACT_KIND_STAGE_REQUEST = "insertany3d.stage-request"
CONTRACT_KIND_STAGE_RESULT = "insertany3d.stage-result"
CONTRACT_KIND_HEARTBEAT = "insertany3d.heartbeat"
CONTRACT_KIND_EVALUATION = "insertany3d.evaluation"
CONTRACT_KIND_EDIT_REVIEW = "insertany3d.edit-review"

FORMAL_TASK_IDS = tuple(f"Task_{index:03d}" for index in range(1, 6))
STAGE_NAMES = (
    "unity_anchor",
    "image_edit",
    "edit_gate",
    "upload_inputs",
    "model_generation",
    "render_alignment_views",
    "segment_inputs",
    "gim_match",
    "estimate_pose",
    "sags_segment_vote",
    "debug_bundle",
    "download_results",
    "unity_apply",
    "unity_eval6",
    "evaluate_absolute",
)
STAGE_RESULT_STATUSES = frozenset(
    {"succeeded", "failed_retryable", "failed_terminal", "rejected", "canceled"}
)
EDIT_DECISIONS = frozenset({"accepted", "rejected", "regenerate", "accepted_by_policy"})
EVAL6_VIEW_LAYOUT = (
    ("low_left", 10.0, -1),
    ("low_center", 10.0, 0),
    ("low_right", 10.0, 1),
    ("high_left", 40.0, -1),
    ("high_center", 40.0, 0),
    ("high_right", 40.0, 1),
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


class ContractError(ValueError):
    """A manifest did not satisfy a stable InsertAny3D contract."""

    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def schema_path(name: str) -> Path:
    """Return an installed schema path by stable base name."""
    path = Path(__file__).with_name("schemas") / f"{name}-v1.schema.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON manifest, with optional YAML support when PyYAML exists."""
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ContractError("$", "YAML 文件需要安装可选依赖 PyYAML；也可直接使用 JSON") from exc
        value = yaml.safe_load(text)
    else:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ContractError("$", f"JSON 无法解析: {exc.msg}") from exc
    return _object(value, "$")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_batch_manifest(value: Mapping[str, Any], *, formal: bool = True) -> dict[str, Any]:
    data = _object(value, "$")
    _version_kind(data, CONTRACT_KIND_BATCH)
    _safe_id(data.get("batchId"), "$.batchId")
    _nonempty(data.get("defaultsRef"), "$.defaultsRef")
    if data.get("renderProfile") != "eval6":
        raise ContractError("$.renderProfile", "当前正式协议必须是 eval6")

    render = _object(data.get("renderProtocol"), "$.renderProtocol")
    render.setdefault("viewYawOffsetDegrees", 24)
    data["renderProtocol"] = render
    if render.get("protocol") != "eval6-v1":
        raise ContractError("$.renderProtocol.protocol", "必须是 eval6-v1")
    yaw = _positive_number(render.get("viewYawOffsetDegrees"), "$.renderProtocol.viewYawOffsetDegrees")
    if yaw >= 180:
        raise ContractError("$.renderProtocol.viewYawOffsetDegrees", "必须小于 180 度")

    resources = _object(data.get("resources"), "$.resources")
    _positive_int(resources.get("unitySlots"), "$.resources.unitySlots")
    # imageApi is the explicit API-only bucket contract. Legacy editSlots* are
    # accepted and normalized so older manifests remain reproducible.
    api = resources.get("imageApi")
    if api is None:
        initial = _positive_int(resources.get("editSlotsInitial", 3), "$.resources.editSlotsInitial")
        maximum = _positive_int(resources.get("editSlotsMax", 5), "$.resources.editSlotsMax")
        api = {"mode": "adaptive", "initialLimit": initial, "maximumLimit": maximum}
    else:
        api = _object(api, "$.resources.imageApi")
        mode = api.setdefault("mode", "adaptive")
        if mode not in {"fixed", "adaptive"}:
            raise ContractError("$.resources.imageApi.mode", "必须是 fixed 或 adaptive")
        initial = _positive_int(api.get("initialLimit", api.get("limit", 3)), "$.resources.imageApi.initialLimit")
        maximum = _positive_int(api.get("maximumLimit", initial if mode == "fixed" else 5), "$.resources.imageApi.maximumLimit")
        if mode == "fixed":
            fixed = _positive_int(api.get("limit", initial), "$.resources.imageApi.limit")
            initial = maximum = fixed
        elif initial > maximum:
            raise ContractError("$.resources.imageApi.initialLimit", "不能大于 maximumLimit")
        api.update({"initialLimit": initial, "maximumLimit": maximum})
    resources["imageApi"] = api
    resources["editSlotsInitial"] = initial
    resources["editSlotsMax"] = maximum
    data["resources"] = resources
    if initial > maximum:
        raise ContractError("$.resources.editSlotsInitial", "不能大于 editSlotsMax")
    gpu_pool = _array(resources.get("remoteGpuPool"), "$.resources.remoteGpuPool")
    if not gpu_pool:
        raise ContractError("$.resources.remoteGpuPool", "至少需要一个远端 GPU")
    if len(gpu_pool) != len(set(gpu_pool)) or any(not isinstance(item, int) or item < 0 for item in gpu_pool):
        raise ContractError("$.resources.remoteGpuPool", "GPU 编号必须是互不重复的非负整数")

    policy = _object(data.get("editPolicy"), "$.editPolicy")
    policy.setdefault("mode", "manual")
    policy.setdefault("reviewBatchSize", 5)
    data["editPolicy"] = policy
    if policy.get("mode") not in {"manual", "automatic"}:
        raise ContractError("$.editPolicy.mode", "必须是 manual 或 automatic")
    _positive_int(policy.get("reviewBatchSize"), "$.editPolicy.reviewBatchSize")
    if policy.get("mode") == "automatic":
        _nonempty(policy.get("policyVersion"), "$.editPolicy.policyVersion")

    _nonempty(data.get("remoteProfile"), "$.remoteProfile")
    projects = _array(data.get("projects"), "$.projects")
    if not projects:
        raise ContractError("$.projects", "至少需要一个显式 Project；禁止目录自动发现")
    seen_projects: set[str] = set()
    total_tasks = 0
    for index, project_value in enumerate(projects):
        path = f"$.projects[{index}]"
        project = _object(project_value, path)
        project_id = _safe_id(project.get("projectId"), f"{path}.projectId")
        if project_id in seen_projects:
            raise ContractError(f"{path}.projectId", "Project ID 重复")
        seen_projects.add(project_id)
        _nonempty(project.get("projectPath"), f"{path}.projectPath")
        scene = _nonempty(project.get("scenePath"), f"{path}.scenePath")
        if not scene.replace("\\", "/").startswith("Assets/") or not scene.endswith(".unity"):
            raise ContractError(f"{path}.scenePath", "必须是 Assets/ 下的 .unity 场景")
        tasks = _array(project.get("tasks"), f"{path}.tasks")
        task_ids: list[str] = []
        for task_index, task_value in enumerate(tasks):
            task_path = f"{path}.tasks[{task_index}]"
            if isinstance(task_value, str):
                task_id = task_value
            else:
                task = _object(task_value, task_path)
                task_id = task.get("taskId")
                if formal:
                    _nonempty(task.get("objectPrompt"), f"{task_path}.objectPrompt")
                    _nonempty(task.get("editPrompt"), f"{task_path}.editPrompt")
            checked_task_id = _nonempty(task_id, f"{task_path}.taskId")
            if checked_task_id not in FORMAL_TASK_IDS:
                raise ContractError(f"{task_path}.taskId", "必须是 Task_001 至 Task_005")
            task_ids.append(checked_task_id)
        if len(task_ids) != len(set(task_ids)):
            raise ContractError(f"{path}.tasks", "taskId 重复")
        if formal and tuple(task_ids) != FORMAL_TASK_IDS:
            raise ContractError(f"{path}.tasks", f"正式 Project 必须按顺序包含 {', '.join(FORMAL_TASK_IDS)}")
        total_tasks += len(task_ids)
        if formal:
            _nonempty(project.get("unityVersion"), f"{path}.unityVersion")
            _sha(project.get("manifestSha256"), f"{path}.manifestSha256")
            _sha(project.get("packagesLockSha256"), f"{path}.packagesLockSha256")

    if formal and (len(projects) != 12 or total_tasks != 60):
        raise ContractError("$.projects", "正式批次必须恰好是 12 个 Project、每个 5 个 task，共 60 个 task")
    if formal:
        pins = _object(data.get("pins"), "$.pins")
        for key in (
            "insertAny3dCommit",
            "unityPackageCommit",
            "defaultsSha256",
            "evaluatorCommit",
            "evaluatorRubricSha256",
        ):
            _sha(pins.get(key), f"$.pins.{key}")
        _nonempty(pins.get("evaluatorModel"), "$.pins.evaluatorModel")
        submodules = _object(pins.get("submodules"), "$.pins.submodules")
        if not submodules:
            raise ContractError("$.pins.submodules", "必须固定使用到的 submodule gitlink")
        for key, digest in submodules.items():
            _nonempty(key, "$.pins.submodules key")
            _sha(digest, f"$.pins.submodules.{key}")
    return dict(data)


def validate_stage_request(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _object(value, "$")
    _version_kind(data, CONTRACT_KIND_STAGE_REQUEST)
    _safe_id(data.get("batchId"), "$.batchId")
    _safe_id(data.get("projectId"), "$.projectId")
    task_id = _nonempty(data.get("taskId"), "$.taskId")
    if task_id not in FORMAL_TASK_IDS:
        raise ContractError("$.taskId", "必须是 Task_001 至 Task_005")
    if data.get("stage") not in STAGE_NAMES:
        raise ContractError("$.stage", "未知 stage")
    _nonempty(data.get("contractVersion"), "$.contractVersion")
    _positive_int(data.get("attempt"), "$.attempt")
    _nonempty(data.get("leaseToken"), "$.leaseToken")
    for index, artifact_value in enumerate(_array(data.get("inputs"), "$.inputs")):
        _validate_artifact(_object(artifact_value, f"$.inputs[{index}]"), f"$.inputs[{index}]", require_size=False)
    config = _object(data.get("effectiveConfig"), "$.effectiveConfig")
    digest = _sha(data.get("effectiveConfigSha256"), "$.effectiveConfigSha256")
    if canonical_sha256(config) != digest:
        raise ContractError("$.effectiveConfigSha256", "与 effectiveConfig 的规范化 SHA-256 不一致")
    _relative_path(data.get("outputStagingDir"), "$.outputStagingDir")
    return dict(data)


def validate_stage_result(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _object(value, "$")
    _version_kind(data, CONTRACT_KIND_STAGE_RESULT)
    for key in ("batchId", "projectId"):
        _safe_id(data.get(key), f"$.{key}")
    if data.get("taskId") not in FORMAL_TASK_IDS:
        raise ContractError("$.taskId", "必须是 Task_001 至 Task_005")
    if data.get("stage") not in STAGE_NAMES:
        raise ContractError("$.stage", "未知 stage")
    _nonempty(data.get("contractVersion"), "$.contractVersion")
    _positive_int(data.get("attempt"), "$.attempt")
    _nonempty(data.get("leaseToken"), "$.leaseToken")
    status = data.get("status")
    if status not in STAGE_RESULT_STATUSES:
        raise ContractError("$.status", "不是允许的 stage 结果状态")
    artifacts = _array(data.get("artifacts"), "$.artifacts")
    for index, artifact_value in enumerate(artifacts):
        _validate_artifact(_object(artifact_value, f"$.artifacts[{index}]"), f"$.artifacts[{index}]", require_size=True)
    if status == "succeeded" and not artifacts:
        raise ContractError("$.artifacts", "成功结果必须至少发布一个 artifact")
    if status != "succeeded":
        _nonempty(data.get("errorCode"), "$.errorCode")
        _nonempty(data.get("message"), "$.message")
    diagnostics = _array(data.get("diagnosticPaths", []), "$.diagnosticPaths")
    for index, path in enumerate(diagnostics):
        _relative_path(path, f"$.diagnosticPaths[{index}]")
    cleanup = _object(data.get("cleanup"), "$.cleanup")
    if not isinstance(cleanup.get("completed"), bool):
        raise ContractError("$.cleanup.completed", "必须是布尔值")
    _utc(data.get("finishedAtUtc"), "$.finishedAtUtc")
    return dict(data)


def validate_heartbeat(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _object(value, "$")
    _version_kind(data, CONTRACT_KIND_HEARTBEAT)
    _nonempty(data.get("leaseToken"), "$.leaseToken")
    _positive_int(data.get("pid"), "$.pid")
    _positive_int(data.get("pgid"), "$.pgid")
    _nonempty(data.get("hostBootId"), "$.hostBootId")
    if not isinstance(data.get("processStartTicks"), int) or data["processStartTicks"] < 0:
        raise ContractError("$.processStartTicks", "必须是非负整数")
    progress = _object(data.get("progress"), "$.progress")
    completed = _nonnegative_number(progress.get("completed"), "$.progress.completed")
    total = _positive_number(progress.get("total"), "$.progress.total")
    if completed > total:
        raise ContractError("$.progress.completed", "不能大于 total")
    if not isinstance(data.get("logOffset"), int) or data["logOffset"] < 0:
        raise ContractError("$.logOffset", "必须是非负整数")
    _utc(data.get("observedAtUtc"), "$.observedAtUtc")
    return dict(data)


def validate_evaluation_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _object(value, "$")
    _version_kind(data, CONTRACT_KIND_EVALUATION)
    if data.get("protocol") != "eval6-v1":
        raise ContractError("$.protocol", "必须是 eval6-v1")
    for key in ("batchId", "projectId", "runId", "methodId"):
        _safe_id(data.get(key), f"$.{key}")
    if data.get("taskId") not in FORMAL_TASK_IDS:
        raise ContractError("$.taskId", "必须是 Task_001 至 Task_005")
    _nonempty(data.get("scenePath"), "$.scenePath")
    _nonempty(data.get("taskPrompt"), "$.taskPrompt")
    view_config = _object(data.get("viewConfig"), "$.viewConfig")
    pitches = _array(view_config.get("pitchDegrees"), "$.viewConfig.pitchDegrees")
    if [float(item) for item in pitches] != [10.0, 40.0]:
        raise ContractError("$.viewConfig.pitchDegrees", "eval6-v1 当前必须是 [10, 40]")
    yaw = _positive_number(view_config.get("yawOffsetDegrees"), "$.viewConfig.yawOffsetDegrees")
    config_without_hash = {key: item for key, item in view_config.items() if key != "sha256"}
    if canonical_sha256(config_without_hash) != _sha(view_config.get("sha256"), "$.viewConfig.sha256"):
        raise ContractError("$.viewConfig.sha256", "与完整 viewConfig 不一致")
    render = _object(data.get("render"), "$.render")
    _positive_int(render.get("width"), "$.render.width")
    _positive_int(render.get("height"), "$.render.height")
    _nonempty(render.get("cameraConvention"), "$.render.cameraConvention")
    views = _array(data.get("views"), "$.views")
    if len(views) != 6:
        raise ContractError("$.views", "必须恰好包含 6 个视角")
    for index, (view_value, expected) in enumerate(zip(views, EVAL6_VIEW_LAYOUT)):
        path = f"$.views[{index}]"
        view = _object(view_value, path)
        view_id, pitch, direction = expected
        if view.get("viewId") != view_id:
            raise ContractError(f"{path}.viewId", f"必须是 {view_id}")
        if not _same_number(view.get("pitchDegrees"), pitch):
            raise ContractError(f"{path}.pitchDegrees", f"必须是 {pitch:g}")
        if not _same_number(view.get("yawOffsetDegrees"), direction * yaw):
            raise ContractError(f"{path}.yawOffsetDegrees", "与全局 yawOffsetDegrees 不一致")
        for field in ("original", "inserted", "camera"):
            _validate_file_ref(_object(view.get(field), f"{path}.{field}"), f"{path}.{field}")
    return dict(data)


def validate_edit_review(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _object(value, "$")
    _version_kind(data, CONTRACT_KIND_EDIT_REVIEW)
    if data.get("taskId") not in FORMAL_TASK_IDS:
        raise ContractError("$.taskId", "必须是 Task_001 至 Task_005")
    _positive_int(data.get("editAttempt"), "$.editAttempt")
    status = data.get("status")
    if status not in {"pending_review", "decided"}:
        raise ContractError("$.status", "必须是 pending_review 或 decided")
    decision = data.get("decision")
    if status == "pending_review":
        if decision is not None:
            raise ContractError("$.decision", "等待审核时必须为空")
    elif decision not in EDIT_DECISIONS:
        raise ContractError("$.decision", "不是允许的审核决定")
    if decision == "accepted_by_policy":
        _nonempty(data.get("policyVersion"), "$.policyVersion")
    if decision is not None:
        _nonempty(data.get("decidedBy"), "$.decidedBy")
        _utc(data.get("decidedAtUtc"), "$.decidedAtUtc")
    return dict(data)


def _version_kind(data: Mapping[str, Any], kind: str) -> None:
    if data.get("schemaVersion") != SCHEMA_VERSION:
        raise ContractError("$.schemaVersion", f"只支持版本 {SCHEMA_VERSION}")
    if data.get("kind") != kind:
        raise ContractError("$.kind", f"必须是 {kind}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(path, "必须是对象")
    return dict(value)


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractError(path, "必须是数组")
    return list(value)


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(path, "必须是非空字符串")
    return value


def _safe_id(value: Any, path: str) -> str:
    text = _nonempty(value, path)
    if not _SAFE_ID.fullmatch(text):
        raise ContractError(path, "只能包含安全文件名字符 A-Z、a-z、0-9、点、下划线和短横线")
    return text


def _sha(value: Any, path: str) -> str:
    text = _nonempty(value, path)
    if not _SHA256.fullmatch(text):
        raise ContractError(path, "必须是小写 64 位 SHA-256")
    return text


def _utc(value: Any, path: str) -> str:
    text = _nonempty(value, path)
    if not _UTC.fullmatch(text):
        raise ContractError(path, "必须是以 Z 结尾的 UTC ISO 8601 时间")
    return text


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(path, "必须是正整数")
    return value


def _positive_number(value: Any, path: str) -> float:
    number = _number(value, path)
    if number <= 0:
        raise ContractError(path, "必须是正数")
    return number


def _nonnegative_number(value: Any, path: str) -> float:
    number = _number(value, path)
    if number < 0:
        raise ContractError(path, "必须是非负数")
    return number


def _number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ContractError(path, "必须是有限数字")
    return float(value)


def _same_number(value: Any, expected: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isclose(float(value), expected)


def _relative_path(value: Any, path: str) -> str:
    text = _nonempty(value, path).replace("\\", "/")
    candidate = Path(text)
    windows_candidate = PureWindowsPath(text)
    if candidate.is_absolute() or windows_candidate.is_absolute() or windows_candidate.drive or ".." in candidate.parts:
        raise ContractError(path, "必须是不会越出 artifact 根目录的相对路径")
    return text


def _validate_file_ref(data: Mapping[str, Any], path: str) -> None:
    _relative_path(data.get("path"), f"{path}.path")
    _sha(data.get("sha256"), f"{path}.sha256")


def _validate_artifact(data: Mapping[str, Any], path: str, *, require_size: bool) -> None:
    _safe_id(data.get("artifactId"), f"{path}.artifactId")
    if "type" in data:
        _nonempty(data.get("type"), f"{path}.type")
    _relative_path(data.get("path"), f"{path}.path")
    _sha(data.get("sha256"), f"{path}.sha256")
    if require_size and (not isinstance(data.get("size"), int) or data["size"] < 0):
        raise ContractError(f"{path}.size", "必须是非负整数")
