"""Provider-neutral planning and resumable caching for GPTEval.

This module deliberately contains no HTTP client.  Production code must inject
the evaluator callable, while tests can use :func:`fixed_fake_response` without
any possibility of contacting a paid API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from insertany3d.contracts.models import canonical_sha256

from .manifests import (
    EvaluationError,
    EvaluationManifest,
    load_evaluation_manifest,
    validate_manifest_collection,
)


SUPPORTED_EVALUATORS = ("gpteval",)
SUPPORTED_DIMENSIONS = (
    "visual_quality",
    "insertion_rationality",
    "geometric_accuracy",
)
DEFAULT_DIMENSIONS = (
    "visual_quality",
    "geometric_accuracy",
)
# Kept as the public name used by callers that want the default score set.
DIMENSIONS = DEFAULT_DIMENSIONS
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GPTEvalRequest:
    request_key: str
    evaluator_version: str
    model: str
    rubric_sha256: str
    sheet_sha256: str
    repeat_index: int
    batch_id: str
    project_id: str
    scene_path: str
    task_id: str
    run_id: str
    method_id: str
    task_prompt: str
    manifest_sha256: str
    manifest_path: Path
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS

    @property
    def task_key(self) -> str:
        return f"{self.project_id}/{self.task_id}/{self.method_id}"

    @property
    def scene_key(self) -> str:
        return f"{self.project_id}:{self.scene_path}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": "insertany3d.gpteval-request",
            "requestKey": self.request_key,
            "evaluator": "gpteval",
            "evaluatorVersion": self.evaluator_version,
            "model": self.model,
            "rubricSha256": self.rubric_sha256,
            "sheetSha256": self.sheet_sha256,
            "repeatIndex": self.repeat_index,
            "batchId": self.batch_id,
            "projectId": self.project_id,
            "scenePath": self.scene_path,
            "taskId": self.task_id,
            "runId": self.run_id,
            "methodId": self.method_id,
            "taskPrompt": self.task_prompt,
            "manifestSha256": self.manifest_sha256,
            "dimensions": list(self.dimensions),
        }


def require_supported_evaluator(name: str) -> str:
    normalized = name.strip().casefold()
    if normalized not in SUPPORTED_EVALUATORS:
        raise EvaluationError(
            f"当前只支持 GPTEval；未注册评测器 {name!r}。GPTEval3D_v2 本阶段明确不加载"
        )
    return normalized


def make_request_key(
    *,
    evaluator_version: str,
    model: str,
    rubric_sha256: str,
    sheet_sha256: str,
    repeat_index: int,
    dimensions: Iterable[str] | str | None = None,
) -> str:
    """Build the exact cache key defined by the evaluation protocol."""

    if (
        not isinstance(evaluator_version, str)
        or not evaluator_version.strip()
        or not isinstance(model, str)
        or not model.strip()
    ):
        raise EvaluationError("evaluator_version 和 model 不能为空")
    _require_sha256(rubric_sha256, "rubric_sha256")
    _require_sha256(sheet_sha256, "sheet_sha256")
    if not isinstance(repeat_index, int) or isinstance(repeat_index, bool) or repeat_index < 0:
        raise EvaluationError("repeat_index 必须是非负整数")
    normalized_dimensions = normalize_dimensions(dimensions)
    return canonical_sha256(
        {
            "evaluator": "gpteval",
            "evaluatorVersion": evaluator_version,
            "model": model,
            "rubricSha256": rubric_sha256,
            "sheetSha256": sheet_sha256,
            "repeatIndex": repeat_index,
            "dimensions": list(normalized_dimensions),
        }
    )


def normalize_dimensions(
    dimensions: Iterable[str] | str | None = None,
) -> tuple[str, ...]:
    """Validate and canonicalize the configured GPTEval dimensions."""

    if dimensions is None:
        return DEFAULT_DIMENSIONS
    raw_items = [dimensions] if isinstance(dimensions, str) else list(dimensions)
    values: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, str):
            raise EvaluationError("GPTEval dimensions 必须是字符串列表")
        values.extend(item.strip() for item in raw_item.split(",") if item.strip())
    if not values:
        raise EvaluationError("GPTEval dimensions 不能为空")
    if len(values) != len(set(values)):
        raise EvaluationError("GPTEval dimensions 不能重复")
    unknown = sorted(set(values) - set(SUPPORTED_DIMENSIONS))
    if unknown:
        raise EvaluationError(
            f"GPTEval dimensions 包含不支持的维度: {unknown}；"
            f"可选值为 {list(SUPPORTED_DIMENSIONS)}"
        )
    return tuple(dimension for dimension in SUPPORTED_DIMENSIONS if dimension in values)


def evaluation_config_sha256(
    manifest_comparison_config_sha256: str,
    dimensions: Iterable[str] | str | None = None,
) -> str:
    """Hash manifest comparison settings together with scoring dimensions."""

    _require_sha256(
        manifest_comparison_config_sha256,
        "manifest_comparison_config_sha256",
    )
    return canonical_sha256(
        {
            "manifestComparisonConfigSha256": manifest_comparison_config_sha256,
            "dimensions": list(normalize_dimensions(dimensions)),
        }
    )


def rubric_sha256(rubric: str | bytes) -> str:
    value = rubric.encode("utf-8") if isinstance(rubric, str) else bytes(rubric)
    if not value:
        raise EvaluationError("评分规则不能为空")
    return hashlib.sha256(value).hexdigest()


def sheet_input_sha256(manifest: EvaluationManifest) -> str:
    """Hash all content that affects an anonymous before/after sheet request.

    The prompt is intentionally included.  This prevents two tasks with the
    same pixels but different requested insertions from sharing a paid result.
    """

    views = []
    for view in manifest.data["views"]:
        views.append(
            {
                "viewId": view["viewId"],
                "pitchDegrees": view["pitchDegrees"],
                "yawOffsetDegrees": view["yawOffsetDegrees"],
                "originalSha256": view["original"]["sha256"],
                "insertedSha256": view["inserted"]["sha256"],
                "cameraSha256": view["camera"]["sha256"],
            }
        )
    return canonical_sha256(
        {
            "sheetProtocol": "gpteval-eval6-sheet-v1",
            "taskPrompt": manifest.data["taskPrompt"],
            "viewConfigSha256": manifest.data["viewConfig"]["sha256"],
            "render": manifest.data["render"],
            "views": views,
        }
    )


def plan_gpteval_requests(
    manifests: Iterable[EvaluationManifest],
    *,
    evaluator_version: str,
    model: str,
    rubric_sha256_value: str,
    repeats: int = 1,
    dimensions: Iterable[str] | str | None = None,
) -> list[GPTEvalRequest]:
    """Validate a comparison set, then produce deterministic requests."""

    records = list(manifests)
    validate_manifest_collection(records)
    require_supported_evaluator("gpteval")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise EvaluationError("repeats 必须是正整数")
    _require_sha256(rubric_sha256_value, "rubric_sha256_value")
    normalized_dimensions = normalize_dimensions(dimensions)

    requests: list[GPTEvalRequest] = []
    for manifest in sorted(
        records,
        key=lambda item: (item.project_id, item.task_id, item.method_id, item.run_id),
    ):
        sheet_sha = sheet_input_sha256(manifest)
        for repeat_index in range(repeats):
            key = make_request_key(
                evaluator_version=evaluator_version,
                model=model,
                rubric_sha256=rubric_sha256_value,
                sheet_sha256=sheet_sha,
                repeat_index=repeat_index,
                dimensions=normalized_dimensions,
            )
            requests.append(
                GPTEvalRequest(
                    request_key=key,
                    evaluator_version=evaluator_version,
                    model=model,
                    rubric_sha256=rubric_sha256_value,
                    sheet_sha256=sheet_sha,
                    repeat_index=repeat_index,
                    batch_id=manifest.batch_id,
                    project_id=manifest.project_id,
                    scene_path=manifest.scene_path,
                    task_id=manifest.task_id,
                    run_id=manifest.run_id,
                    method_id=manifest.method_id,
                    task_prompt=str(manifest.data["taskPrompt"]),
                    manifest_sha256=manifest.manifest_sha256,
                    manifest_path=manifest.path,
                    dimensions=normalized_dimensions,
                )
            )
    keys = [request.request_key for request in requests]
    if len(keys) != len(set(keys)):
        raise EvaluationError(
            "GPTEval request key 冲突；请检查任务图片、prompt 和 manifest 是否被错误复用"
        )
    for request in requests:
        validate_gpteval_request(request)
    return requests


def validate_gpteval_request(
    request: GPTEvalRequest, manifest: EvaluationManifest | None = None
) -> None:
    """Validate an executable request and, when supplied, its source manifest."""

    for name, value in (
        ("batch_id", request.batch_id),
        ("project_id", request.project_id),
        ("scene_path", request.scene_path),
        ("task_id", request.task_id),
        ("run_id", request.run_id),
        ("method_id", request.method_id),
        ("task_prompt", request.task_prompt),
    ):
        if not isinstance(value, str) or not value.strip():
            raise EvaluationError(f"GPTEval request {name} 不能为空")
    _require_sha256(request.manifest_sha256, "manifest_sha256")
    normalized_dimensions = normalize_dimensions(request.dimensions)
    if request.dimensions != normalized_dimensions:
        raise EvaluationError("GPTEval request dimensions 未按标准顺序保存")
    expected_key = make_request_key(
        evaluator_version=request.evaluator_version,
        model=request.model,
        rubric_sha256=request.rubric_sha256,
        sheet_sha256=request.sheet_sha256,
        repeat_index=request.repeat_index,
        dimensions=request.dimensions,
    )
    if request.request_key != expected_key:
        raise EvaluationError("GPTEval requestKey 与 evaluator/model/rubric/sheet/repeat 不一致")
    if manifest is None:
        return
    expected = {
        "batch_id": manifest.batch_id,
        "project_id": manifest.project_id,
        "scene_path": manifest.scene_path,
        "task_id": manifest.task_id,
        "run_id": manifest.run_id,
        "method_id": manifest.method_id,
        "task_prompt": str(manifest.data["taskPrompt"]),
        "manifest_sha256": manifest.manifest_sha256,
        "sheet_sha256": sheet_input_sha256(manifest),
    }
    for name, value in expected.items():
        if getattr(request, name) != value:
            raise EvaluationError(f"GPTEval 请求与评测清单的 {name} 不一致: {manifest.task_key}")


def adapt_gpteval_response(
    payload: Any,
    dimensions: Iterable[str] | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Normalize direct fake scores or a Gemini-style response payload."""

    value = payload
    if isinstance(value, Mapping) and "scores" in value:
        value = value["scores"]
    elif isinstance(value, Mapping) and "candidates" in value:
        value = _extract_candidate_json(value)
    if isinstance(value, str):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.I)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"GPTEval 返回的文本不是 JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise EvaluationError("GPTEval 响应必须是评分对象")

    normalized_dimensions = normalize_dimensions(dimensions)
    returned_dimensions = set(value)
    expected_dimensions = set(normalized_dimensions)
    if returned_dimensions != expected_dimensions:
        missing = sorted(expected_dimensions - returned_dimensions)
        extra = sorted(str(item) for item in returned_dimensions - expected_dimensions)
        raise EvaluationError(
            f"GPTEval 响应维度必须严格匹配；缺少 {missing}，多出 {extra}"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for dimension in normalized_dimensions:
        item = value.get(dimension)
        if not isinstance(item, Mapping):
            raise EvaluationError(f"GPTEval 响应缺少维度: {dimension}")
        if set(item) != {"score", "reason"}:
            raise EvaluationError(f"{dimension} 只能包含 score 和 reason")
        score = item.get("score")
        reason = item.get("reason")
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 10:
            raise EvaluationError(f"{dimension}.score 必须是 1 到 10 的整数")
        if not isinstance(reason, str) or not reason.strip():
            raise EvaluationError(f"{dimension}.reason 不能为空")
        normalized[dimension] = {"score": score, "reason": reason.strip()}
    return normalized


def fixed_fake_response(
    score: int = 7,
    dimensions: Iterable[str] | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a deterministic local response for tests and dry runs."""

    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 10:
        raise EvaluationError("固定假分数必须是 1 到 10 的整数")
    return {
        dimension: {"score": score, "reason": "fixed offline test response"}
        for dimension in normalize_dimensions(dimensions)
    }


class ResponseCache:
    """Atomic, content-addressed GPTEval response storage."""

    _request_locks_guard = threading.Lock()
    _request_locks: dict[Path, threading.Lock] = {}

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def response_path(self, request: GPTEvalRequest) -> Path:
        validate_gpteval_request(request)
        return self.root / request.request_key[:2] / f"{request.request_key}.json"

    def error_path(self, request: GPTEvalRequest) -> Path:
        validate_gpteval_request(request)
        return self.root / "errors" / f"{request.request_key}.json"

    @contextmanager
    def request_lock(self, request: GPTEvalRequest) -> Iterable[None]:
        """Serialize one request across threads and processes.

        The operating-system lock is released automatically if a process dies.
        The thread lock is still needed because POSIX advisory locks are owned
        by the process rather than by an individual Python thread.
        """

        path = self.response_path(request).absolute()
        with self._request_locks_guard:
            lock = self._request_locks.setdefault(path, threading.Lock())
        with lock:
            lock_path = self.root / "locks" / f"{request.request_key}.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+b") as lock_stream:
                _lock_stream(lock_stream)
                try:
                    yield
                finally:
                    _unlock_stream(lock_stream)

    def get(self, request: GPTEvalRequest) -> dict[str, Any] | None:
        path = self.response_path(request)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"GPTEval 缓存损坏: {path}: {exc}") from exc
        if not isinstance(record, Mapping):
            raise EvaluationError(f"GPTEval 缓存不是对象: {path}")
        if record.get("schemaVersion") != 1 or record.get("status") != "ready":
            raise EvaluationError(f"GPTEval 缓存状态无效: {path}")
        if record.get("request") != request.to_dict():
            raise EvaluationError(f"GPTEval 缓存键与请求内容不一致: {path}")
        scores = adapt_gpteval_response(record.get("scores"), request.dimensions)
        return {**dict(record), "scores": scores}

    def store(self, request: GPTEvalRequest, payload: Any) -> dict[str, Any]:
        scores = adapt_gpteval_response(payload, request.dimensions)
        record = {
            "schemaVersion": 1,
            "kind": "insertany3d.gpteval-response",
            "status": "ready",
            "createdAtUtc": _utc_now(),
            "request": request.to_dict(),
            "scores": scores,
        }
        _atomic_write_json(self.response_path(request), record)
        error_path = self.error_path(request)
        if error_path.exists():
            error_path.unlink()
        return record

    def store_error(self, request: GPTEvalRequest, error: BaseException, attempts: int) -> None:
        _atomic_write_json(
            self.error_path(request),
            {
                "schemaVersion": 1,
                "kind": "insertany3d.gpteval-error",
                "status": "failed",
                "createdAtUtc": _utc_now(),
                "request": request.to_dict(),
                "attempts": attempts,
                "errorType": type(error).__name__,
                "message": str(error),
            },
        )


def pending_gpteval_requests(
    requests: Iterable[GPTEvalRequest], cache: ResponseCache
) -> list[GPTEvalRequest]:
    return [request for request in requests if cache.get(request) is None]


def execute_gpteval_requests(
    requests: Iterable[GPTEvalRequest],
    cache: ResponseCache,
    evaluator: Callable[[GPTEvalRequest], Any],
    *,
    retries: int = 0,
    retry_delay_seconds: float = 0.0,
    limit: int | None = None,
) -> dict[str, int]:
    """Evaluate only cache misses through an explicitly supplied callable."""

    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise EvaluationError("retries 必须是非负整数")
    if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
        raise EvaluationError("retry_delay_seconds 必须是有限的非负数")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
        raise EvaluationError("limit 必须是非负整数或 None")

    skipped = 0
    pending: list[GPTEvalRequest] = []
    for request in requests:
        if cache.get(request) is not None:
            skipped += 1
        else:
            pending.append(request)
    if limit is not None:
        pending = pending[:limit]

    planned = completed = failed = 0
    for request in pending:
        validate_gpteval_request(request)
        with cache.request_lock(request):
            if cache.get(request) is not None:
                skipped += 1
                continue
            planned += 1
            last_error: BaseException | None = None
            attempts_made = 0
            for attempt in range(retries + 1):
                attempts_made = attempt + 1
                # Local input drift cannot be repaired by repeating a paid call,
                # so validation failures propagate immediately.
                current_manifest = load_evaluation_manifest(request.manifest_path)
                validate_gpteval_request(request, current_manifest)
                try:
                    cache.store(request, evaluator(request))
                    completed += 1
                    last_error = None
                    break
                except Exception as exc:  # The provider adapter owns concrete error types.
                    last_error = exc
                    if not getattr(exc, "retryable", True):
                        break
                    if attempt < retries and retry_delay_seconds:
                        time.sleep(retry_delay_seconds)
            if last_error is not None:
                failed += 1
                cache.store_error(request, last_error, attempts_made)
    return {
        "planned": planned,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
    }


def _extract_candidate_json(payload: Mapping[str, Any]) -> str:
    if payload.get("error"):
        raise EvaluationError(f"GPTEval provider 返回错误: {payload['error']}")
    for candidate in payload.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        content = candidate.get("content", {})
        if not isinstance(content, Mapping):
            continue
        parts = content.get("parts", [])
        texts = [
            part.get("text", "")
            for part in parts
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        text = "\n".join(item for item in texts if item).strip()
        if text:
            return text
    raise EvaluationError("GPTEval provider 响应中没有文本")


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise EvaluationError(f"{name} 必须是小写 64 位 SHA-256")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _lock_stream(stream: Any) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - exercised on Windows hosts.
        import msvcrt

        stream.seek(0)
        if stream.read(1) == b"":
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)


def _unlock_stream(stream: Any) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - exercised on Windows hosts.
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{threading.get_ident()}.tmp"
    temporary = path.with_name(path.name + suffix)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
