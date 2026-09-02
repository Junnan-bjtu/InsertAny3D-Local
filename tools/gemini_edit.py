#!/usr/bin/env python3
"""Call an APIYi/Gemini image-edit endpoint from the command line.

The API key is deliberately read from GEMINI_API_KEY (or --api-key) and is
never included in output files or status messages.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import requests


DEFAULT_BASE_URL = "https://api.apiyi.com/v1"
DEFAULT_IMAGE_URL = (
    "https://api.apiyi.com/v1beta/models/"
    "gemini-3.1-flash-image-preview:generateContent"
)
DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_KEY_FILE = Path.home() / ".config" / "insertany3d" / "apiyi_key"


def _mime_type(path: Path) -> str:
    value, _ = mimetypes.guess_type(path.name)
    return value or "image/png"


def _endpoint_from_args(args: argparse.Namespace) -> str:
    endpoint = args.endpoint or os.environ.get("GEMINI_IMAGE_URL")
    if not endpoint:
        # APIYi exposes the native Gemini route beside its OpenAI-compatible
        # /v1 base.  Derive /v1beta from the configured base when callers do
        # not provide the more specific GEMINI_IMAGE_URL.
        parts = urlsplit(args.base_url or DEFAULT_BASE_URL)
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        root = urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")
        endpoint = f"{root}/v1beta/models/{args.model}:generateContent"
    # APIYi deployments can expose a model-specific image route while using a
    # different model name for metadata.  Preserve an explicitly configured
    # image URL exactly; callers who need another route can pass --endpoint.
    return endpoint


def _redact(value: str, secret: str | None) -> str:
    return value.replace(secret, "[REDACTED]") if secret else value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key:
        return args.api_key.strip()
    environment_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    key_file = args.api_key_file
    if key_file and key_file.is_file():
        if os.name != "nt" and stat.S_IMODE(key_file.stat().st_mode) & 0o077:
            raise SystemExit(f"API key 文件权限过宽；请执行 chmod 600 {key_file}")
        return key_file.read_text(encoding="utf-8").strip()
    return None


def _response_request_ids(response: requests.Response) -> dict[str, str]:
    allowed = ("x-request-id", "x-goog-request-id", "request-id", "x-correlation-id")
    return {name: response.headers[name] for name in allowed if response.headers.get(name)}


def _normalize_image_for_output(image_bytes: bytes, mime: str, output_path: Path) -> tuple[bytes, str]:
    expected_mime = _mime_type(output_path)
    normalized_mime = mime.split(";", 1)[0].strip().lower()
    if normalized_mime == expected_mime:
        return image_bytes, expected_mime
    if expected_mime != "image/png":
        raise SystemExit(f"API 返回 {mime}，与输出扩展名 {output_path.suffix} 不一致")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(f"API 返回 {mime}；写入 .png 前需要本机安装 ffmpeg")
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-f", "image2pipe",
            "-i", "pipe:0",
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "pipe:1",
        ],
        input=image_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or not completed.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
        detail = completed.stderr.decode("utf-8", errors="replace")[:1000]
        raise SystemExit(f"ffmpeg 无法把 {mime} 转为 PNG: {detail}")
    return completed.stdout, "image/png"


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _decode_data_uri(value: str) -> tuple[bytes, str] | None:
    match = re.match(r"^data:([^;,]+)(?:;base64)?,(.*)$", value, flags=re.S)
    if not match:
        return None
    mime = match.group(1)
    encoded = match.group(2)
    try:
        return base64.b64decode(encoded), mime
    except (ValueError, base64.binascii.Error):
        return None


def _extract_image(response: Any, session: requests.Session, timeout: float) -> tuple[bytes, str] | None:
    """Accept native Gemini inlineData and common OpenAI-compatible variants."""
    for item in _iter_dicts(response):
        for key in ("inlineData", "inline_data"):
            candidate = item.get(key)
            if isinstance(candidate, dict) and candidate.get("data"):
                try:
                    return base64.b64decode(candidate["data"]), candidate.get("mimeType", "image/png")
                except (ValueError, base64.binascii.Error):
                    pass
        if item.get("b64_json"):
            try:
                return base64.b64decode(item["b64_json"]), item.get("mime_type", "image/png")
            except (ValueError, base64.binascii.Error):
                pass
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if isinstance(image_url, str):
            decoded = _decode_data_uri(image_url)
            if decoded:
                return decoded
            if image_url.startswith(("http://", "https://")):
                downloaded = session.get(image_url, timeout=timeout)
                downloaded.raise_for_status()
                return downloaded.content, downloaded.headers.get("Content-Type", "image/png").split(";", 1)[0]
    return None


def _response_text(response: Any) -> str:
    parts: list[str] = []
    for item in _iter_dicts(response):
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(dict.fromkeys(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 APIYi/Gemini 根据提示词编辑图片")
    parser.add_argument("--input-image", required=True, type=Path)
    parser.add_argument("--output-image", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--endpoint", help="完整 generateContent URL，默认使用 GEMINI_IMAGE_URL")
    parser.add_argument("--base-url", default=os.environ.get("GEMINI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=None, help="仅兼容旧调用；会暴露在进程参数中，不建议使用")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path(os.environ.get("GEMINI_API_KEY_FILE", DEFAULT_KEY_FILE)),
        help="密钥文件；默认 ~/.config/insertany3d/apiyi_key，环境变量优先",
    )
    parser.add_argument("--aspect-ratio", default="1:1")
    parser.add_argument("--image-size", default="1K", choices=("1K", "2K", "4K"))
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--response-json", type=Path, help="可选：保存不含请求头的原始 JSON 响应")
    parser.add_argument("--manifest", type=Path, help="保存输入、prompt、模型和输出哈希等来源信息")
    parser.add_argument("--overwrite", action="store_true", help="显式允许替换已有输出图片")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入和请求配置，不发送请求")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_image.is_file():
        raise SystemExit(f"输入图片不存在: {args.input_image}")
    prompt_text = args.prompt
    if args.prompt_file:
        prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise SystemExit("提示词为空")
    if args.output_image.exists() and not args.overwrite:
        raise SystemExit(f"输出图片已存在；使用 --overwrite 才会替换: {args.output_image}")

    endpoint = _endpoint_from_args(args)
    api_key = _read_api_key(args)
    if args.dry_run:
        print(
            "GEMINI_EDIT_CONFIG",
            _redact(endpoint, api_key),
            args.model,
            args.image_size,
            args.aspect_ratio,
            args.input_image,
            args.output_image,
        )
        return 0
    if not api_key:
        raise SystemExit(
            "未找到 API key；请设置 GEMINI_API_KEY，或写入 "
            f"{args.api_key_file}（文件只包含 key 本身）"
        )

    input_image_bytes = args.input_image.read_bytes()
    encoded = base64.b64encode(input_image_bytes).decode("ascii")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {"inlineData": {"mimeType": _mime_type(args.input_image), "data": encoded}},
                ],
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": args.aspect_ratio,
                "imageSize": args.image_size,
            },
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    session = requests.Session()
    response = None
    last_error = None
    started = time.monotonic()
    attempts = 0
    for attempt in range(max(0, args.retries) + 1):
        attempts = attempt + 1
        try:
            response = session.post(endpoint, headers=headers, json=payload, timeout=args.timeout)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 8))
                    continue
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= args.retries:
                detail = ""
                if response is not None:
                    detail = _redact(response.text[:2000], api_key)
                raise SystemExit(f"Gemini 请求失败: {_redact(str(exc), api_key)}\n{detail}") from exc
    if response is None:
        raise SystemExit(f"Gemini 请求失败: {last_error}")

    try:
        response_json = response.json()
    except ValueError as exc:
        raise SystemExit(f"Gemini 返回不是 JSON: {_redact(response.text[:1000], api_key)}") from exc
    if args.response_json:
        _write_json(args.response_json, response_json)

    image = _extract_image(response_json, session, args.timeout)
    if image is None:
        text = _response_text(response_json)
        raise SystemExit("Gemini 响应中没有找到图片数据" + (f"；文本响应: {text[:500]}" if text else ""))
    image_bytes, response_mime = image
    image_bytes, mime = _normalize_image_for_output(image_bytes, response_mime, args.output_image)
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    temporary_image = args.output_image.with_name(args.output_image.name + ".tmp")
    temporary_image.write_bytes(image_bytes)
    temporary_image.replace(args.output_image)
    if args.manifest:
        usage = response_json.get("usageMetadata") if isinstance(response_json, dict) else None
        candidates = response_json.get("candidates", []) if isinstance(response_json, dict) else []
        finish_reasons = [
            item.get("finishReason")
            for item in candidates
            if isinstance(item, dict) and item.get("finishReason")
        ]
        manifest = {
            "schemaVersion": 1,
            "status": "ready",
            "provenanceType": "model_image_edit",
            "generator": "apiyi-gemini-generateContent",
            "createdAtUtc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(args.input_image.resolve()),
                "mimeType": _mime_type(args.input_image),
                "bytes": len(input_image_bytes),
                "sha256": _sha256_bytes(input_image_bytes),
            },
            "prompt": {
                "text": prompt_text,
                "sha256": _sha256_bytes(prompt_text.encode("utf-8")),
            },
            "request": {
                "endpoint": _redact(endpoint, api_key),
                "model": args.model,
                "generationConfig": payload["generationConfig"],
                "attempts": attempts,
                "durationSeconds": round(time.monotonic() - started, 3),
            },
            "response": {
                "requestIds": _response_request_ids(response),
                "usageMetadata": usage,
                "finishReasons": finish_reasons,
                "imageMimeType": response_mime,
                "jsonPath": str(args.response_json.resolve()) if args.response_json else None,
            },
            "output": {
                "path": str(args.output_image.resolve()),
                "mimeType": mime,
                "bytes": len(image_bytes),
                "sha256": _sha256_bytes(image_bytes),
            },
        }
        _write_json(args.manifest, manifest)
    print("GEMINI_EDIT_READY", args.output_image, mime)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("已取消")
