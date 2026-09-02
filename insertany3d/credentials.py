"""Shared local credential loading for APIYi-backed operations."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Mapping


DEFAULT_APIYI_KEY_FILE = Path.home() / ".config" / "insertany3d" / "apiyi_key"


class ApiYiCredentialError(ValueError):
    """The configured APIYi credential is missing or unsafe to read."""


def load_apiyi_api_key(
    environ: Mapping[str, str] | None = None,
    *,
    default_key_file: Path | None = None,
) -> tuple[str, str]:
    """Return the APIYi key and a non-secret label describing its source."""

    values = os.environ if environ is None else environ
    api_key = values.get("APIYI_API_KEY", "").strip()
    if api_key:
        return api_key, "APIYI_API_KEY"

    explicit_path: Path | None = None
    explicit_name: str | None = None
    for name in ("APIYI_API_KEY_FILE", "GEMINI_API_KEY_FILE"):
        if name in values:
            explicit_name = name
            raw_path = values.get(name, "").strip()
            if not raw_path:
                raise ApiYiCredentialError(f"{name} 已设置但路径为空")
            explicit_path = Path(raw_path).expanduser()
            break

    key_file = explicit_path or default_key_file or DEFAULT_APIYI_KEY_FILE
    label = _display_path(key_file)
    try:
        metadata = key_file.stat()
    except FileNotFoundError:
        if explicit_name is not None:
            raise ApiYiCredentialError(
                f"{explicit_name} 指定的 APIYi key 文件不存在: {label}"
            )
    except OSError as exc:
        raise ApiYiCredentialError(f"无法读取 APIYi key 文件 {label}: {exc}") from exc
    else:
        return _read_key_file(key_file, metadata)

    for name in ("GEMINI_API_KEY", "BEE_API_KEY"):
        value = values.get(name, "").strip()
        if value:
            return value, name

    raise ApiYiCredentialError(
        "未找到 APIYi key；请设置 APIYI_API_KEY，或将 key 写入 "
        f"{_display_path(key_file)} 并执行 chmod 600 {_display_path(key_file)}"
    )


def _read_key_file(path: Path, metadata: os.stat_result) -> tuple[str, str]:
    label = _display_path(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ApiYiCredentialError(f"APIYi key 路径不是普通文件: {label}")
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ApiYiCredentialError(
            f"APIYi key 文件权限过宽: {label}；请执行 chmod 600 {label}"
        )
    if os.name == "posix" and not stat.S_IMODE(metadata.st_mode) & stat.S_IRUSR:
        raise ApiYiCredentialError(
            f"APIYi key 文件不可读: {label}；请执行 chmod 600 {label}"
        )
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ApiYiCredentialError(f"无法读取 APIYi key 文件 {label}: {exc}") from exc
    if not value:
        raise ApiYiCredentialError(f"APIYi key 文件为空: {label}")
    return value, label


def _display_path(path: Path) -> str:
    expanded = path.expanduser()
    try:
        relative = expanded.relative_to(Path.home())
    except ValueError:
        return str(expanded)
    return str(Path("~") / relative)
