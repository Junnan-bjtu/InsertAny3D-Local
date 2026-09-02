"""Explicitly gated GPTEval transport for Gemini-compatible API endpoints."""

from __future__ import annotations

import base64
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

from ..credentials import ApiYiCredentialError, load_apiyi_api_key
from .gpteval import (
    DEFAULT_DIMENSIONS,
    GPTEvalRequest,
    normalize_dimensions,
    rubric_sha256,
    validate_gpteval_request,
)
from .manifests import EvaluationError, load_evaluation_manifest


DEFAULT_EVALUATOR_VERSION = "gpteval-eval6-v1"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_BASE_URL = "https://api.apiyi.com"
DEFAULT_RUBRIC = """You are an impartial evaluator of one anonymous 3D scene insertion result.
The request supplies six REFERENCE views before insertion and the corresponding six CANDIDATE views after insertion.
Judge only the requested insertion. Account for normal viewpoint parallax, occlusion, reflections, exposure, and defects already present in the reference scene. Penalize background damage only when the candidate clearly introduced it.

Use the same absolute scale for every requested dimension: 10 means no meaningful visible defect, 8 means only localized mild imperfections, 5 means strengths and substantial defects coexist, 3 means severe repeated defects dominate, and 1 means complete or near-complete failure."""

DIMENSION_RUBRICS = {
    "visual_quality": (
        "rendering quality, visible artifacts, boundaries, material and lighting "
        "integration, and multi-view consistency"
    ),
    "insertion_rationality": (
        "semantic correctness, scale, pose, placement, contact, occlusion, and "
        "whether the requested relationship is plausible"
    ),
    "geometric_accuracy": (
        "3D shape, perspective, depth, support, penetration, floating, and "
        "cross-view geometric consistency"
    ),
}


def response_schema_for(dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS) -> dict[str, Any]:
    normalized = normalize_dimensions(dimensions)
    return {
        "type": "OBJECT",
        "properties": {
            dimension: {
                "type": "OBJECT",
                "properties": {
                    "score": {"type": "INTEGER"},
                    "reason": {"type": "STRING"},
                },
                "required": ["score", "reason"],
            }
            for dimension in normalized
        },
        "required": list(normalized),
    }


RESPONSE_SCHEMA = response_schema_for(DEFAULT_DIMENSIONS)

Transport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]
]


class GPTEvalProviderError(EvaluationError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


class GPTEvalAPIClient:
    """A callable evaluator whose construction requires an environment key."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        rubric: str = DEFAULT_RUBRIC,
        timeout_seconds: float = 300.0,
        transport: Transport | None = None,
    ):
        if not api_key.strip():
            raise EvaluationError("GPTEval API key 不能为空")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise EvaluationError("GPTEval timeout 必须是有限的正数")
        self._api_key = api_key
        self.base_url = base_url
        self.rubric = rubric
        self.timeout_seconds = timeout_seconds
        self.transport = transport or urllib_transport

    def __call__(self, request: GPTEvalRequest) -> Mapping[str, Any]:
        expected_rubric = rubric_sha256(self.rubric)
        if request.rubric_sha256 != expected_rubric:
            raise EvaluationError("GPTEval 请求的 rubric hash 与运行时评分规则不一致")
        endpoint = endpoint_for(self.base_url, request.model)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        return self.transport(
            endpoint,
            headers,
            build_request_body(request, self.rubric),
            self.timeout_seconds,
        )


def read_api_key_from_environment() -> tuple[str, str]:
    """Compatibility wrapper for the shared APIYi credential loader."""

    try:
        return load_apiyi_api_key()
    except ApiYiCredentialError as exc:
        raise EvaluationError(str(exc)) from exc


def endpoint_for(base_url: str, model: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise EvaluationError(f"GPTEval base URL 无效: {base_url}")
    path = parsed.path.rstrip("/")
    for suffix in ("/v1beta", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    root = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")
    encoded_model = urllib.parse.quote(model, safe="-_.")
    return f"{root}/v1beta/models/{encoded_model}:generateContent"


def build_request_body(request: GPTEvalRequest, rubric: str) -> dict[str, Any]:
    """Build a multi-image request without creating persistent sheet images."""

    manifest = load_evaluation_manifest(request.manifest_path)
    validate_gpteval_request(request, manifest)
    parts: list[dict[str, Any]] = [
        {
            "text": (
                f"{_scoring_prompt(rubric, request.dimensions)}\n\n"
                f"Requested insertion:\n{request.task_prompt}\n\n"
                "The following images are ordered by the stable eval6 view IDs."
            )
        }
    ]
    root = manifest.path.parent
    for label, field in (("REFERENCE", "original"), ("CANDIDATE", "inserted")):
        for view in manifest.data["views"]:
            image_path = root / view[field]["path"]
            mime_type = _image_mime_type(image_path)
            parts.append({"text": f"{label} {view['viewId']}"})
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                    }
                }
            )
    generation_seed = int(request.request_key[:8], 16) % (2**31 - 1)
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "seed": generation_seed,
            "responseModalities": ["TEXT"],
            "responseMimeType": "application/json",
            "responseSchema": response_schema_for(request.dimensions),
        },
    }


def _scoring_prompt(rubric: str, dimensions: tuple[str, ...]) -> str:
    lines = [
        rubric.strip(),
        "",
        "Return an integer score from 1 to 10 and a concise reason only for "
        "these dimensions:",
    ]
    lines.extend(
        f"- {dimension}: {DIMENSION_RUBRICS[dimension]}."
        for dimension in dimensions
    )
    lines.append(
        "Return only the requested JSON object and do not add disabled dimensions."
    )
    return "\n".join(lines)


def urllib_transport(
    endpoint: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    encoded = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=encoded, headers=dict(headers), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        for secret in headers.values():
            if secret:
                detail = detail.replace(secret, "[REDACTED]")
        authorization = headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            detail = detail.replace(authorization[7:], "[REDACTED]")
        retryable = exc.code in {408, 425, 429} or exc.code >= 500
        raise GPTEvalProviderError(
            f"GPTEval HTTP {exc.code}: {detail}", retryable=retryable
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise GPTEvalProviderError(
            f"GPTEval 网络请求失败: {exc}", retryable=True
        ) from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GPTEvalProviderError(
            f"GPTEval 响应不是有效 JSON: {exc}", retryable=False
        ) from exc
    if not isinstance(value, Mapping):
        raise GPTEvalProviderError("GPTEval 响应必须是 JSON 对象", retryable=False)
    return value


def _image_mime_type(path: Path) -> str:
    prefix = path.read_bytes()[:12]
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith(b"\xff\xd8"):
        return "image/jpeg"
    raise EvaluationError(f"GPTEval 不支持图片格式: {path}")
