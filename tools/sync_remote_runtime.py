#!/usr/bin/env python3
"""Audit and publish the server runtime snapshot.

The server checkout is the preferred source during the repository split.  The
old sibling ``codex_remote_tools`` directory remains an explicit and automatic
migration fallback until the first server canary has passed.  This script only
handles the public runtime snapshot; it never edits a server checkout or its
private environment file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path("tools/remote_runtime.lock.json")
SOURCE_AUTHORITIES = frozenset({"server_checkout", "codex_remote_tools"})
RUNTIME_FILES = (
    "auto_segment.py",
    "build_debug_bundle.py",
    "clip_image_similarity.py",
    "estimate_similarity_pose.py",
    "gaussian_model.py",
    "gemini_edit.py",
    "generate_trellis_asset.py",
    "insertany3d_render_utils.py",
    "render_trellis_3dgs.py",
    "render_trellis_views.py",
    "render_utils.py",
    "run_gim_match.py",
    "run_insert_batch.py",
    "run_insert_pipeline.py",
    "run_sags_text.py",
    "sags_gaussian_renderer.py",
    "segment_anchor_views.py",
    "segment_image.py",
    "select_trellis_yaw.py",
    "stage_adapter.py",
    "test_estimate_similarity_pose.py",
    "test_gemini_edit.py",
    "test_insert_batch.py",
    "test_stage_adapter.py",
    "workspace.py",
    "test_workspace.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_center_files(root: Path) -> tuple[str, ...]:
    directory = root / "model_center"
    if not directory.is_dir():
        return ()
    return tuple(
        path.relative_to(root).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )


def runtime_paths(source_root: Path) -> tuple[str, ...]:
    return tuple(RUNTIME_FILES) + _model_center_files(source_root)


def compare_trees(source_root: Path, public_tools: Path) -> list[str]:
    problems: list[str] = []
    expected = set(runtime_paths(source_root))
    public_model_files = set(_model_center_files(public_tools))
    for relative in sorted(expected):
        source = source_root / relative
        target = public_tools / relative
        if not source.is_file():
            problems.append(f"权威副本缺文件: {relative}")
        elif not target.is_file():
            problems.append(f"公开副本缺文件: {relative}")
        elif _sha256(source) != _sha256(target):
            problems.append(f"内容不同: {relative}")
    for relative in sorted(public_model_files - expected):
        problems.append(f"公开 model_center 存在未纳入权威的文件: {relative}")
    return problems


def _lock_value(
    source_root: Path,
    source_authority: str = "server_checkout",
) -> dict[str, object]:
    if source_authority not in SOURCE_AUTHORITIES:
        raise ValueError(f"unsupported runtime source authority: {source_authority}")
    files = []
    for relative in sorted(runtime_paths(source_root)):
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"权威副本缺文件: {path}")
        files.append({"path": relative, "sha256": _sha256(path), "size": path.stat().st_size})
    return {
        "schemaVersion": 1,
        "kind": "insertany3d.remote-runtime-lock",
        "sourceAuthority": source_authority,
        "files": files,
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sync_tree(source_root: Path, public_tools: Path, delete_stale: bool) -> None:
    expected = set(runtime_paths(source_root))
    for relative in sorted(expected):
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"权威副本缺文件: {source}")
        target = public_tools / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if delete_stale:
        for relative in sorted(set(_model_center_files(public_tools)) - expected):
            (public_tools / relative).unlink()


def verify_lock(repository_root: Path, expected_paths: set[str] | None = None) -> list[str]:
    lock_path = repository_root / LOCK_PATH
    if not lock_path.is_file():
        return [f"缺少运行时锁文件: {lock_path}"]
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"运行时锁文件无法读取: {exc}"]
    if (
        value.get("schemaVersion") != 1
        or value.get("kind") != "insertany3d.remote-runtime-lock"
        or value.get("sourceAuthority") not in SOURCE_AUTHORITIES
    ):
        return ["运行时锁文件版本或类型不受支持"]
    problems: list[str] = []
    records = value.get("files")
    if not isinstance(records, list):
        return ["运行时锁文件 files 必须是数组"]
    locked_paths: set[str] = set()
    for record in records:
        relative = record.get("path") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            problems.append(f"锁文件包含非法路径: {relative!r}")
            continue
        if relative in locked_paths:
            problems.append(f"锁文件包含重复路径: {relative}")
            continue
        locked_paths.add(relative)
        target = repository_root / "tools" / relative
        if not target.is_file():
            problems.append(f"公开副本缺文件: {relative}")
        elif target.stat().st_size != record.get("size") or _sha256(target) != record.get("sha256"):
            problems.append(f"公开副本与锁文件不一致: {relative}")
    public_tools = repository_root / "tools"
    expected_paths = expected_paths or (set(RUNTIME_FILES) | set(_model_center_files(public_tools)))
    for relative in sorted(expected_paths - locked_paths):
        problems.append(f"锁文件缺少运行文件: {relative}")
    for relative in sorted(locked_paths - expected_paths):
        problems.append(f"锁文件包含未声明运行文件: {relative}")
    return problems


def _runtime_directory(path: Path) -> Path:
    """Accept either a checkout root or its ``tools`` directory.

    A server checkout normally contains ``tools/stage_adapter.py``.  Taking
    both forms makes the command convenient for a local clone and for the
    server path supplied by deployment tooling while keeping ``--source``
    backwards compatible with a direct runtime directory.
    """

    candidate = path.expanduser()
    if (candidate / "stage_adapter.py").is_file():
        return candidate
    tools = candidate / "tools"
    if tools.is_dir():
        return tools
    return candidate


def resolve_runtime_source(
    repository_root: Path,
    *,
    source: Path | None = None,
    server_source: Path | None = None,
    server_root: Path | None = None,
    server_checkout: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, str]:
    """Resolve the runtime source and its non-secret authority label.

    Explicit server arguments win over the old ``--source`` option.  The
    latter remains the explicit migration-mirror override and is always
    labelled ``codex_remote_tools``; use a ``--server-*`` option for the
    Server checkout authority.  No existence check is performed here so
    callers can produce a precise error for an explicitly requested path.
    """

    env = os.environ if environ is None else environ
    if server_checkout is not None:
        if server_root is not None:
            raise ValueError("server_root 与 server_checkout 不能同时使用")
        server_root = server_checkout
    if sum(value is not None for value in (server_source, server_root)) > 1:
        raise ValueError("--server-source 与 --server-root 不能同时使用")

    if server_source is not None:
        return _runtime_directory(Path(server_source).resolve()), "server_checkout"
    if server_root is not None:
        return _runtime_directory(Path(server_root).resolve()), "server_checkout"

    # An explicitly supplied legacy option remains an intentional override.
    # This matters for rollback when a ``server/`` submodule is present but a
    # maintainer needs to compare it with the old mirror.
    if source is not None:
        # ``--source`` is the backwards-compatible migration-mirror option.
        # Even when a caller gives it a custom path, do not label that path as
        # the Server checkout; use ``--server-source`` for that authority.
        return _runtime_directory(Path(source).resolve()), "codex_remote_tools"

    env_server_source = env.get("INSERTANY3D_SERVER_RUNTIME_SOURCE") or env.get("INSERTANY3D_SERVER_SOURCE")
    env_server_root = env.get("INSERTANY3D_SERVER_ROOT") or env.get("INSERTANY3D_SERVER_CHECKOUT")
    if env_server_source:
        return _runtime_directory(Path(env_server_source).expanduser().resolve()), "server_checkout"
    if env_server_root:
        return _runtime_directory(Path(env_server_root).expanduser().resolve()), "server_checkout"

    # Preserve an explicitly configured legacy environment override as well;
    # this is useful while rolling back a canary even if a server submodule is
    # already checked out next to the integration repository.
    legacy = env.get("INSERTANY3D_REMOTE_RUNTIME_SOURCE")
    if legacy:
        return _runtime_directory(Path(legacy).expanduser().resolve()), "codex_remote_tools"

    # A checked-out integration repository may expose the server as either a
    # ``server`` submodule or a sibling ``InsertAny3D-Server`` checkout.
    for candidate in (
        repository_root / "server",
        repository_root.parent / "InsertAny3D-Server",
    ):
        runtime = _runtime_directory(candidate)
        if runtime.is_dir() and (runtime / "stage_adapter.py").is_file():
            return runtime.resolve(), "server_checkout"

    # The sibling mirror is the final controlled migration fallback.
    return (repository_root.parent / "codex_remote_tools").resolve(), "codex_remote_tools"


def _print_problems(problems: Iterable[str]) -> int:
    values = list(problems)
    if not values:
        return 0
    for problem in values:
        print(f"REMOTE_RUNTIME_DRIFT {problem}", file=sys.stderr)
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "核对或同步 Server checkout 与公开 tools；迁移期可回退到 "
            "codex_remote_tools，不会连接服务器"
        )
    )
    parser.add_argument("command", choices=("check", "sync", "verify-lock"))
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "兼容参数：直接指定运行时目录；显式 Server 参数优先。"
            "未指定时自动寻找 Server checkout，再回退 codex_remote_tools"
        ),
    )
    parser.add_argument(
        "--server-source",
        "--server-runtime-source",
        type=Path,
        default=None,
        help="Server 运行时来源目录（可传 checkout 根目录或其 tools 目录）",
    )
    parser.add_argument(
        "--server-root",
        "--server-checkout",
        dest="server_root",
        type=Path,
        default=None,
        help="Server checkout 根目录；运行时来源为其 tools/ 子目录",
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--delete-stale",
        action="store_true",
        help="sync 时删除公开 model_center 中不属于权威副本的旧文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    public_tools = repository_root / "tools"
    if args.command == "verify-lock":
        problems = verify_lock(repository_root)
        if problems:
            return _print_problems(problems)
        print("REMOTE_RUNTIME_LOCK_OK", repository_root)
        return 0

    try:
        source_root, source_authority = resolve_runtime_source(
            repository_root,
            source=args.source,
            server_source=args.server_source,
            server_root=args.server_root,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if source_root == public_tools.resolve():
        raise SystemExit("权威目录和公开 tools 不能是同一目录")
    if not source_root.is_dir():
        raise SystemExit(f"找不到运行时来源目录: {source_root}")
    print(f"REMOTE_RUNTIME_SOURCE {source_authority} {source_root}")
    if args.command == "sync":
        sync_tree(source_root, public_tools, args.delete_stale)
        _write_json_atomic(
            repository_root / LOCK_PATH,
            _lock_value(source_root, source_authority),
        )
    problems = compare_trees(source_root, public_tools)
    if problems:
        return _print_problems(problems)
    lock_problems = verify_lock(repository_root)
    if lock_problems:
        return _print_problems(lock_problems)
    print("REMOTE_RUNTIME_MIRROR_OK", len(runtime_paths(source_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
