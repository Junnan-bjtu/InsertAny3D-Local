"""Explicitly enabled local stage executors for Unity and image editing.

Importing this module or constructing a normal :class:`BatchWorker` never
starts Unity or calls an API.  The caller must construct ``LocalStageExecutor``
with ``allow_real=True`` and provide the required executable/credentials.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit, urlunsplit

from .apiyi import AdaptiveImageQueue, ApiOutcome, ImageApiClient
from .contracts import validate_stage_result
from .image_edit import DEFAULT_NUM_GENERATIONS, generation_count
from .dag import STAGE_BY_NAME, image_api_limits
from .executors import CommandExecutor, ExecutionResult
from .scheduler import BatchController, WorkItem, default_capacities
from .stage_wiring import (
    StageWiringError,
    build_stage_command,
    ensure_unity_project_not_running,
    validate_unity_project,
    write_stage_request,
)


UNITY_STAGES = frozenset({"unity_anchor", "unity_apply", "unity_eval6"})
LOCAL_STAGES = frozenset({*UNITY_STAGES, "image_edit"})


class LocalWorkerConfigurationError(ValueError):
    """A real local stage was requested without an explicit safe setup."""


def local_worker_capacities(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Let the adaptive API gate, rather than the scheduler, own edit ramp-up.

    The scheduler may lease up to the configured maximum.  The token/model
    gate still starts at ``editSlotsInitial`` and admits only its current
    adaptive limit, so a caller can increase throughput after clean responses
    without bypassing the rate-limit controller.
    """

    capacities = default_capacities(manifest)
    capacities["image_api"] = image_api_limits(manifest["resources"])[2]
    return capacities


@dataclass(frozen=True)
class ImageEditResponse:
    outcome: ApiOutcome
    image_bytes: bytes | None = None
    mime_type: str | None = None


class ImageEditAdapter(Protocol):
    """Replaceable boundary used by tests and alternate APIYi clients."""

    def edit(
        self,
        *,
        endpoint: str,
        token: str,
        model: str,
        prompt: str,
        input_bytes: bytes,
        input_mime_type: str,
        aspect_ratio: str,
        image_size: str,
    ) -> ImageEditResponse:
        ...


@dataclass(frozen=True)
class ImageWorkerConfig:
    endpoint: str
    token: str
    model: str = "gemini-3.1-flash-image-preview"
    timeout_seconds: float = 360.0
    aspect_ratio: str = "1:1"
    image_size: str = "1K"
    num_gen_image_per_task: int = DEFAULT_NUM_GENERATIONS

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise LocalWorkerConfigurationError("APIYi 图片编辑 endpoint 不能为空")
        endpoint = urlsplit(self.endpoint)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
            raise LocalWorkerConfigurationError("APIYi 图片编辑 endpoint 必须是完整的 http/https URL")
        if not self.token.strip():
            raise LocalWorkerConfigurationError("APIYi 图片编辑 token 不能为空")
        if not self.model.strip():
            raise LocalWorkerConfigurationError("APIYi 图片编辑 model 不能为空")
        if self.timeout_seconds <= 0:
            raise LocalWorkerConfigurationError("图片编辑 timeout_seconds 必须大于 0")
        if self.timeout_seconds > STAGE_BY_NAME["image_edit"].timeout_seconds:
            raise LocalWorkerConfigurationError("图片编辑 timeout_seconds 不能超过 stage 的 420 秒上限")
        try:
            generation_count({"num_gen_image_per_task": self.num_gen_image_per_task})
        except ValueError as exc:
            raise LocalWorkerConfigurationError(str(exc)) from exc


class ApiYiImageEditAdapter:
    """Native Gemini-compatible APIYi adapter using the shared classifier."""

    def __init__(self, *, timeout_seconds: float = 360.0, client: ImageApiClient | None = None):
        self.client = client or ImageApiClient(timeout_seconds=timeout_seconds)

    def edit(
        self,
        *,
        endpoint: str,
        token: str,
        model: str,
        prompt: str,
        input_bytes: bytes,
        input_mime_type: str,
        aspect_ratio: str,
        image_size: str,
    ) -> ImageEditResponse:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": input_mime_type,
                                "data": base64.b64encode(input_bytes).decode("ascii"),
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": aspect_ratio, "imageSize": image_size},
            },
        }
        outcome = self.client.post_json(endpoint, token, payload)
        if outcome.status != "succeeded":
            return ImageEditResponse(outcome)
        extracted = _extract_image(outcome.body)
        if extracted is None:
            invalid = ApiOutcome(
                "failed_terminal",
                "compile_or_contract",
                False,
                False,
                outcome.http_status,
                None,
                outcome.response_headers,
                None,
            )
            return ImageEditResponse(invalid)
        image_bytes, mime_type = extracted
        return ImageEditResponse(outcome, image_bytes, mime_type)


class LocalStageExecutor:
    """Route the four local stages without registering any remote stage.

    ``supported_stages`` is intentionally exposed so a queue worker can lease
    only work this executor owns.  This prevents a local-only worker from
    consuming and failing an upload or remote GPU stage.
    """

    def __init__(
        self,
        *,
        allow_real: bool = False,
        unity_executable: str | None = None,
        image_config: ImageWorkerConfig | None = None,
        command_executor: CommandExecutor | None = None,
        image_adapter: ImageEditAdapter | None = None,
        unity_validator: Callable[[Any, Any], Path] = validate_unity_project,
        unity_process_guard: Callable[[str | Path], None] = ensure_unity_project_not_running,
        image_heartbeat_seconds: float = 15.0,
    ):
        if not allow_real:
            raise LocalWorkerConfigurationError(
                "真实本地 worker 默认关闭；只有命令行显式选择真实模式后才能传入 allow_real=True"
            )
        if unity_executable is None and image_config is None:
            raise LocalWorkerConfigurationError("至少配置 Unity 或图片编辑 worker")
        if image_heartbeat_seconds <= 0:
            raise LocalWorkerConfigurationError("image_heartbeat_seconds 必须大于 0")
        self.unity_executable = unity_executable
        self.image_config = image_config
        self.command_executor = command_executor or CommandExecutor()
        self.image_adapter = image_adapter or (
            ApiYiImageEditAdapter(timeout_seconds=image_config.timeout_seconds)
            if image_config is not None
            else None
        )
        self.unity_validator = unity_validator
        self.unity_process_guard = unity_process_guard
        self.image_heartbeat_seconds = image_heartbeat_seconds
        self._image_queues: dict[Path, AdaptiveImageQueue] = {}
        self._image_queues_lock = threading.Lock()

    @property
    def supported_stages(self) -> tuple[str, ...]:
        stages: list[str] = []
        if self.unity_executable is not None:
            stages.extend(("unity_eval6", "unity_apply"))
        if self.image_config is not None:
            stages.append("image_edit")
        if self.unity_executable is not None:
            stages.append("unity_anchor")
        return tuple(stages)

    def execute(self, controller: BatchController, item: WorkItem) -> ExecutionResult:
        if item.stage in UNITY_STAGES and self.unity_executable is not None:
            return self._execute_unity(controller, item)
        if item.stage == "image_edit" and self.image_config is not None:
            return self._execute_image(controller, item)
        return ExecutionResult(
            False,
            stage_status="failed_terminal",
            error_code="invalid_input",
            message=f"本地 worker 未配置 stage: {item.stage}",
        )

    def _execute_unity(self, controller: BatchController, item: WorkItem) -> ExecutionResult:
        try:
            project_root = self.unity_validator(controller.store, item)
            self.unity_process_guard(project_root)
            _request, request_path = write_stage_request(controller.store, item)
            command = build_stage_command(
                controller.store,
                item,
                request_path,
                unity_executable=self.unity_executable,
            )
        except (OSError, StageWiringError, ValueError) as exc:
            return ExecutionResult(
                False,
                stage_status="failed_terminal",
                error_code="invalid_input",
                message=f"Unity 启动前检查失败: {exc}",
            )
        return self.command_executor.execute(
            controller,
            item,
            command,
            timeout_seconds=float(STAGE_BY_NAME[item.stage].timeout_seconds),
        )

    def _execute_image(self, controller: BatchController, item: WorkItem) -> ExecutionResult:
        assert self.image_config is not None
        assert self.image_adapter is not None
        timing: dict[str, Any] = {"queuedAtUtc": _utc_now()}
        timing_started = time.monotonic()
        try:
            request, _request_path = write_stage_request(controller.store, item)
            prompt = _task_edit_prompt(request)
            source_ref = _center_image_ref(request)
            source_path = _artifact_path(controller, item, source_ref)
            source_bytes = source_path.read_bytes()
            if not source_bytes:
                raise StageWiringError("unity_anchor 的 center 图片为空")
            expected_sha = source_ref.get("sha256")
            actual_sha = _sha256(source_bytes)
            if expected_sha and expected_sha != actual_sha:
                raise StageWiringError("unity_anchor 的 center 图片 SHA-256 与 artifact 记录不一致")
        except (OSError, StageWiringError, ValueError) as exc:
            return self._image_failure(
                item,
                ApiOutcome("failed_terminal", "invalid_input", False, False, None, None, {}),
                f"图片编辑输入无效: {exc}", timing=timing, timing_started=timing_started,
            )

        queue = self._image_queue(controller, item.batch_id)
        queue_started = time.monotonic()
        if not self._acquire_image_slot(controller, item, queue, timing, queue_started):
            return self._image_failure(
                item,
                ApiOutcome("failed_retryable", "stalled", True, False, None, None, {}),
                "等待 APIYi 图片编辑并发槽超过 stage 时限", timing=timing, timing_started=timing_started,
            )

        controller.heartbeat(
            item.stage_id,
            item.lease_token,
            progress={"completed": 0, "total": 1, "unit": "image_edit"},
        )
        timing["requestStartedAtUtc"] = _utc_now()
        request_started = time.monotonic()
        try:
            response = self._call_image_adapter_with_heartbeats(
                controller,
                item,
                prompt=prompt,
                source_bytes=source_bytes,
                source_path=source_path,
            )
        except Exception as exc:
            timing["requestEndedAtUtc"] = _utc_now()
            timing["apiRequestSeconds"] = max(0.0, time.monotonic() - request_started)
            outcome = ApiOutcome("failed_retryable", "worker_crash", True, False, None, None, {})
            queue.release(outcome, attempt_index=item.attempt - 1)
            return self._image_failure(item, outcome, f"图片编辑 adapter 异常: {type(exc).__name__}: {exc}", timing=timing, timing_started=timing_started)
        except BaseException:
            timing["requestEndedAtUtc"] = _utc_now()
            timing["apiRequestSeconds"] = max(0.0, time.monotonic() - request_started)
            queue.release(
                ApiOutcome("failed_retryable", "worker_crash", True, False, None, None, {}),
                attempt_index=item.attempt - 1,
            )
            raise
        timing["requestEndedAtUtc"] = _utc_now()
        timing["apiRequestSeconds"] = max(0.0, time.monotonic() - request_started)
        queue.release(response.outcome, attempt_index=item.attempt - 1)

        if response.outcome.status != "succeeded":
            message = _api_failure_message(response.outcome)
            return self._image_failure(item, response.outcome, message, timing=timing, timing_started=timing_started)
        if not response.image_bytes:
            invalid = ApiOutcome("failed_terminal", "compile_or_contract", False, False, None, None, {})
            return self._image_failure(item, invalid, "APIYi 返回成功但没有可发布的图片", timing=timing, timing_started=timing_started)

        generation_index = _generation_index(request, item)
        request_task = request.get("effectiveConfig", {}).get("task")
        generation_count_value = self.image_config.num_gen_image_per_task
        if isinstance(request_task, Mapping):
            configured_count = request_task.get("generationCount", request_task.get("num_gen_image_per_task"))
            if configured_count is not None:
                try:
                    generation_count_value = generation_count({"num_gen_image_per_task": configured_count})
                except ValueError:
                    # The scheduler validates the effective config. Keep the
                    # worker-level fallback for legacy hand-built requests.
                    generation_count_value = self.image_config.num_gen_image_per_task
        output_path = item.staging_dir / f"edited-{uuid.uuid4().hex}.png"
        output_path.write_bytes(response.image_bytes)
        metadata = {
            "schemaVersion": 1,
            "status": "ready",
            "provenanceType": "model_image_edit",
            "generator": "apiyi-gemini-generateContent",
            "batchId": item.batch_id,
            "projectId": item.project_id,
            "taskId": item.task_id,
            "attempt": item.attempt,
            "generationIndex": generation_index,
            "generationCount": generation_count_value,
            "input": {
                "artifactId": source_ref["artifactId"],
                "path": source_ref["path"],
                "bytes": len(source_bytes),
                "sha256": _sha256(source_bytes),
            },
            "prompt": {"text": prompt, "sha256": _sha256(prompt.encode("utf-8"))},
            "request": {
                "endpoint": _public_endpoint(self.image_config.endpoint),
                "model": self.image_config.model,
                "generationConfig": {
                    "aspectRatio": self.image_config.aspect_ratio,
                    "imageSize": self.image_config.image_size,
                },
            },
            "response": {
                "httpStatus": response.outcome.http_status,
                "headers": response.outcome.response_headers,
                "imageMimeType": response.mime_type,
            },
            "output": {
                # Keep both the durable relative name and the resolved path so
                # review consumers can pass the selected file downstream
                # without reconstructing an attempt/group identifier.
                "path": output_path.name,
                "fullPath": str(output_path.resolve()),
                "bytes": len(response.image_bytes),
                "sha256": _sha256(response.image_bytes),
            },
            "createdAtUtc": _utc_now(),
        }
        timing["finishedAtUtc"] = _utc_now()
        timing["stageElapsedSeconds"] = max(0.0, time.monotonic() - timing_started)
        metadata["timing"] = dict(timing)
        metadata_path = item.staging_dir / "image_edit.json"
        _write_json(metadata_path, metadata)
        artifacts = [
            _artifact("edited_image" if generation_index == 1 else f"edited_image_g{generation_index:03d}", "edited_image", output_path),
            _artifact("image_edit_manifest", "image_edit_manifest", metadata_path),
        ]
        _write_stage_result(item, "succeeded", artifacts=artifacts, timing=timing)
        controller.heartbeat(
            item.stage_id,
            item.lease_token,
            progress={"completed": 1, "total": 1, "unit": "image_edit"},
        )
        return ExecutionResult(True, artifacts=artifacts)

    def _call_image_adapter_with_heartbeats(
        self,
        controller: BatchController,
        item: WorkItem,
        *,
        prompt: str,
        source_bytes: bytes,
        source_path: Path,
    ) -> ImageEditResponse:
        config = self.image_config
        assert config is not None
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="insertany3d-image")
        future: Future[ImageEditResponse] = pool.submit(
            self.image_adapter.edit,
            endpoint=config.endpoint,
            token=config.token,
            model=config.model,
            prompt=prompt,
            input_bytes=source_bytes,
            input_mime_type=mimetypes.guess_type(source_path.name)[0] or "image/png",
            aspect_ratio=config.aspect_ratio,
            image_size=config.image_size,
        )
        try:
            heartbeat_seconds = min(self.image_heartbeat_seconds, controller.lease_seconds / 3.0)
            while True:
                try:
                    return future.result(timeout=heartbeat_seconds)
                except FutureTimeout:
                    controller.heartbeat(
                        item.stage_id,
                        item.lease_token,
                        progress={"completed": 0, "total": 1, "unit": "image_edit"},
                    )
        finally:
            pool.shutdown(wait=True, cancel_futures=False)

    def _acquire_image_slot(
        self,
        controller: BatchController,
        item: WorkItem,
        queue: AdaptiveImageQueue,
        timing: dict[str, Any],
        queue_started: float,
    ) -> bool:
        heartbeat_seconds = min(self.image_heartbeat_seconds, controller.lease_seconds / 3.0)
        deadline = time.monotonic() + STAGE_BY_NAME["image_edit"].timeout_seconds
        while time.monotonic() < deadline:
            timeout = min(heartbeat_seconds, max(0.0, deadline - time.monotonic()))
            if queue.acquire(timeout=timeout):
                timing["slotAcquiredAtUtc"] = _utc_now()
                timing["queueWaitSeconds"] = max(0.0, time.monotonic() - queue_started)
                return True
            controller.heartbeat(
                item.stage_id,
                item.lease_token,
                progress={"completed": 0, "total": 1, "unit": "image_api_queue"},
            )
        return False

    def _image_queue(self, controller: BatchController, batch_id: str) -> AdaptiveImageQueue:
        key = controller.store.path.resolve()
        with self._image_queues_lock:
            queue = self._image_queues.get(key)
            if queue is None:
                assert self.image_config is not None
                batch = controller.store.row("SELECT manifest_json FROM batches WHERE batch_id=?", (batch_id,))
                if batch is None:
                    raise LocalWorkerConfigurationError(f"batch 不存在: {batch_id}")
                resources = json.loads(batch["manifest_json"])["resources"]
                mode, initial, maximum = image_api_limits(resources)
                queue = AdaptiveImageQueue(
                    controller.store,
                    self.image_config.token,
                    self.image_config.model,
                    initial_limit=initial,
                    maximum_limit=maximum,
                    mode=mode,
                    fixed_limit=initial if mode == "fixed" else None,
                )
                self._image_queues[key] = queue
            return queue

    @staticmethod
    def _image_failure(item: WorkItem, outcome: ApiOutcome, message: str, *, timing: dict[str, Any] | None = None, timing_started: float | None = None) -> ExecutionResult:
        if timing is None:
            timing = {"queuedAtUtc": _utc_now()}
        if timing_started is None:
            timing_started = time.monotonic()
        timing.setdefault("finishedAtUtc", _utc_now())
        timing.setdefault("stageElapsedSeconds", max(0.0, time.monotonic() - timing_started))
        timing.setdefault("queueWaitSeconds", None)
        timing.setdefault("apiRequestSeconds", None)
        diagnostic = {
            "schemaVersion": 1,
            "status": outcome.status,
            "errorCode": outcome.error_code,
            "deliveryUnknown": outcome.delivery_unknown,
            "httpStatus": outcome.http_status,
            "retryAfterSeconds": outcome.retry_after_seconds,
            "responseHeaders": outcome.response_headers,
            "message": message,
            "observedAtUtc": _utc_now(),
            "timing": dict(timing),
        }
        diagnostic_path = item.staging_dir / "image_api_outcome.json"
        _write_json(diagnostic_path, diagnostic)
        contract_status = "failed_retryable" if outcome.retryable else "failed_terminal"
        _write_stage_result(
            item,
            contract_status,
            error_code=outcome.error_code or "worker_crash",
            message=message,
            diagnostics=[diagnostic_path.name],
            timing=timing,
        )
        stage_status = None if outcome.delivery_unknown else contract_status
        return ExecutionResult(
            False,
            stage_status=stage_status,
            error_code=outcome.error_code or "worker_crash",
            message=message,
            retry_after_seconds=outcome.retry_after_seconds,
        )


def _task_edit_prompt(request: Mapping[str, Any]) -> str:
    task = request.get("effectiveConfig", {}).get("task")
    if not isinstance(task, Mapping):
        raise StageWiringError("effectiveConfig.task 不是对象")
    prompt = task.get("effectiveEditPrompt") or task.get("editPrompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise StageWiringError(
            "effectiveConfig.task.editPrompt 为空；请在 batch manifest 的对应任务中显式填写"
        )
    anchor = task.get("effectiveAnchorPrompt") or task.get("anchorPrompt") or task.get("anchorMaskPrompt")
    sections: list[str] = []
    if isinstance(anchor, str) and anchor.strip():
        sections.append("LOCKED ANCHOR (must remain unchanged):\n" + anchor.strip())
    sections.append("REQUESTED NEW-OBJECT INSERTION:\n" + prompt.strip())
    return "\n\n".join(sections)


def _generation_index(request: Mapping[str, Any], item: WorkItem) -> int:
    task = request.get("effectiveConfig", {}).get("task")
    if isinstance(task, Mapping):
        value = task.get("generationIndex")
        if value is not None:
            try:
                index = int(value)
                if index > 0:
                    return index
            except (TypeError, ValueError):
                pass
    item_index = getattr(item, "generation_index", None)
    if item_index is not None:
        try:
            index = int(item_index)
            if index > 0:
                return index
        except (TypeError, ValueError):
            pass
    # Legacy single-candidate requests remain candidate 1.
    return 1


def _center_image_ref(request: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        value
        for value in request.get("inputs", [])
        if isinstance(value, Mapping) and value.get("artifactId") == "scene_center_image"
    ]
    if len(matches) != 1:
        raise StageWiringError(
            f"image_edit 必须恰好收到一个 unity_anchor center 图片，实际为 {len(matches)} 个"
        )
    return matches[0]


def _artifact_path(
    controller: BatchController,
    item: WorkItem,
    artifact: Mapping[str, Any],
) -> Path:
    row = controller.store.row("SELECT root_path FROM batches WHERE batch_id=?", (item.batch_id,))
    if row is None:
        raise StageWiringError(f"batch 不存在: {item.batch_id}")
    root = Path(row["root_path"]).resolve()
    relative = Path(str(artifact.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise StageWiringError("center 图片 artifact path 不是安全相对路径")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageWiringError("center 图片 artifact path 越出 batch root") from exc
    if not path.is_file():
        raise StageWiringError(f"center 图片 artifact 不存在: {relative.as_posix()}")
    return path


def _extract_image(body: bytes | None) -> tuple[bytes, str] | None:
    if not body:
        return None
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    for item in _iter_mappings(value):
        for key in ("inlineData", "inline_data"):
            inline = item.get(key)
            if isinstance(inline, Mapping) and isinstance(inline.get("data"), str):
                decoded = _decode_base64(inline["data"])
                if decoded:
                    return decoded, str(inline.get("mimeType") or inline.get("mime_type") or "image/png")
        encoded = item.get("b64_json")
        if isinstance(encoded, str):
            decoded = _decode_base64(encoded)
            if decoded:
                return decoded, str(item.get("mime_type") or "image/png")
        image_url = item.get("image_url")
        if isinstance(image_url, Mapping):
            image_url = image_url.get("url")
        if isinstance(image_url, str) and image_url.startswith("data:"):
            header, separator, encoded = image_url.partition(",")
            decoded = _decode_base64(encoded) if separator else None
            if decoded:
                mime = header[5:].split(";", 1)[0] or "image/png"
                return decoded, mime
    return None


def _iter_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _iter_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child)


def _decode_base64(value: str) -> bytes | None:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error):
        return None
    return decoded or None


def _api_failure_message(outcome: ApiOutcome) -> str:
    if outcome.error_code == "http_429":
        return "APIYi 返回 429：当前令牌与模型的并发或速率额度已满"
    if outcome.error_code == "http_503":
        return "APIYi 返回 503：上游图片模型暂时过载"
    if outcome.delivery_unknown:
        return "图片请求超时或连接中断；服务端可能已收到请求，为避免重复计费已暂停人工确认"
    if outcome.error_code in {"http_400", "http_403"}:
        return f"APIYi 拒绝图片请求: {outcome.error_code}"
    return f"图片编辑失败: {outcome.error_code or outcome.status}"


def _public_endpoint(endpoint: str) -> str:
    """Persist the route without accidentally recording query-string secrets."""
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _artifact(artifact_id: str, artifact_type: str, path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "artifactId": artifact_id,
        "type": artifact_type,
        "path": path.name,
        "sha256": _sha256(content),
        "size": len(content),
    }


def _write_stage_result(
    item: WorkItem,
    status: str,
    *,
    artifacts: list[dict[str, Any]] | None = None,
    error_code: str | None = None,
    message: str | None = None,
    diagnostics: list[str] | None = None,
    timing: Mapping[str, Any] | None = None,
) -> Path:
    value = {
        "schemaVersion": 1,
        "kind": "insertany3d.stage-result",
        "batchId": item.batch_id,
        "projectId": item.project_id,
        "taskId": item.task_id,
        "stage": item.stage,
        "contractVersion": item.contract_version,
        "attempt": item.attempt,
        "leaseToken": item.lease_token,
        "status": status,
        "artifacts": list(artifacts or []),
        "errorCode": error_code,
        "message": message,
        "diagnosticPaths": list(diagnostics or []),
        "cleanup": {"completed": True},
        "finishedAtUtc": _utc_now(),
    }
    if timing is not None:
        value["timing"] = dict(timing)
    validate_stage_result(value)
    target = item.staging_dir / "stage_result.json"
    _write_json(target, value)
    return target


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
