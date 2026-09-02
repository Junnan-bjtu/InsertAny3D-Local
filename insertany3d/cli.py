"""Command-line interface for the local InsertAny3D batch control plane."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, TextIO

from .contracts import ContractError, load_manifest
from .credentials import ApiYiCredentialError, load_apiyi_api_key
from .evaluation import (
    DEFAULT_BASE_URL,
    DEFAULT_EVALUATOR_VERSION,
    DEFAULT_MODEL,
    DEFAULT_RUBRIC,
    EvaluationError,
    EvaluationManifest,
    GPTEvalAPIClient,
    GPTEvalRequest,
    ResponseCache,
    Transport,
    aggregate_gpteval,
    discover_evaluation_manifests,
    execute_gpteval_requests,
    evaluation_config_sha256,
    fixed_fake_response,
    load_evaluation_manifest,
    normalize_dimensions,
    pending_gpteval_requests,
    plan_gpteval_requests,
    require_supported_evaluator,
    rubric_sha256,
    validate_manifest_collection,
    write_gpteval_summary,
)
from .executors import CommandExecutor, FakeRunner
from .local_workers import (
    ImageWorkerConfig,
    LocalStageExecutor,
    local_worker_capacities,
)
from .remote_worker import (
    RemoteCommandBuilder,
    RemoteProfile,
    RemoteWorkerError,
    verify_remote_runtime,
)
from .remote_recovery import RemoteRecoveryManager
from .runtime_workers import (
    CompositeStageExecutor,
    DEFAULT_LOCAL_ENVIRONMENT_FILE,
    LOCAL_ENVIRONMENT_FILE_VARIABLE,
    RemoteStageExecutor,
    load_local_environment,
)
from .scheduler import (
    EVALUATION_SKIPPED_ERROR,
    BatchController,
    default_capacities,
    status_resource_for_stage,
)
from .stage_wiring import (
    StageWiringError,
    ensure_unity_project_not_running,
    validate_unity_project_for_batch,
)
from .store import SchedulerStore, StoreError
from .worker import BatchWorker


_REVIEW_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
_RUN_MANIFEST_NAME = "run_manifest.json"
_GIT_ENV_SECRET_NAMES = frozenset(
    {
        "APIYI_API_KEY",
        "APIYI_API_KEY_FILE",
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_FILE",
        "BEE_API_KEY",
    }
)


def _unavailable_git_snapshot(repository: str | os.PathLike[str], reason: str) -> dict[str, Any]:
    """Return the stable shape used when provenance collection cannot run."""

    message = str(reason).strip() or "unknown error"
    return {
        "repository": str(repository),
        "head": "unavailable",
        "status": "unavailable",
        "statusOutput": [],
        "error": message[:1000],
    }


def _git_snapshot(
    repository: str | os.PathLike[str],
    *,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Capture one local Git HEAD/status snapshot without raising.

    HEAD and status are queried independently so a repository with a readable
    commit but a broken worktree still records the information that is known.
    This function deliberately has no policy gate: provenance is evidence for
    later review, never a reason to reject a development run.
    """

    root = Path(repository).expanduser().resolve()
    run = runner or subprocess.run
    base = {
        "repository": str(root),
        "head": "unavailable",
        "status": "unavailable",
        "statusOutput": [],
        "error": None,
    }
    if not root.is_dir():
        return _unavailable_git_snapshot(root, "repository directory is unavailable")
    errors: list[str] = []
    try:
        head = run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"HEAD: {exc}")
    else:
        value = str(getattr(head, "stdout", "") or "").strip()
        if getattr(head, "returncode", 1) == 0 and value:
            base["head"] = value.splitlines()[0].strip()
        else:
            detail = str(getattr(head, "stderr", "") or value).strip()
            return_code = getattr(head, "returncode", 1)
            errors.append(f"HEAD: {detail or f'exit code {return_code}'}")
    try:
        status = run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"status: {exc}")
    else:
        status_text = str(getattr(status, "stdout", "") or "")
        if getattr(status, "returncode", 1) == 0:
            base["statusOutput"] = status_text.splitlines()
            base["status"] = "dirty" if base["statusOutput"] else "clean"
        else:
            detail = str(getattr(status, "stderr", "") or status_text).strip()
            return_code = getattr(status, "returncode", 1)
            errors.append(f"status: {detail or f'exit code {return_code}'}")
    if errors:
        base["error"] = "; ".join(errors)[:1000]
        if base["head"] == "unavailable" and base["status"] == "unavailable":
            base["statusOutput"] = []
    return base


def _sanitized_environment() -> dict[str, str]:
    """Build an environment safe to pass to a remote SSH subprocess."""

    return {
        key: value
        for key, value in os.environ.items()
        if key not in _GIT_ENV_SECRET_NAMES
    }


def _provenance_profile(args: argparse.Namespace | None) -> tuple[RemoteProfile | None, str | None]:
    """Build a best-effort profile from CLI values or the Local env file."""

    def value(name: str, env_name: str, default: Any = None) -> Any:
        candidate = getattr(args, name, None) if args is not None else None
        return candidate if candidate not in (None, "") else os.environ.get(env_name, default)

    target = value("remote_target", "INSERTANY3D_REMOTE_TARGET")
    project_root = value("remote_project_root", "INSERTANY3D_REMOTE_PROJECT_ROOT")
    if not target or not project_root:
        missing = []
        if not target:
            missing.append("INSERTANY3D_REMOTE_TARGET")
        if not project_root:
            missing.append("INSERTANY3D_REMOTE_PROJECT_ROOT")
        return None, "未配置服务器 Git provenance: " + ", ".join(missing)
    artifact_root = value(
        "remote_artifact_root",
        "INSERTANY3D_REMOTE_ARTIFACT_ROOT",
        str(PurePosixPath(str(project_root), "runs")),
    )
    try:
        profile = RemoteProfile(
            target=str(target),
            port=int(value("remote_port", "INSERTANY3D_REMOTE_PORT", 22)),
            project_root=str(project_root),
            artifact_root=str(artifact_root),
            python_executable=str(value("remote_python", "INSERTANY3D_REMOTE_PYTHON", "third_party/TRELLIS/.venv/bin/python")),
            connect_timeout_seconds=float(value("remote_connect_timeout", "INSERTANY3D_REMOTE_CONNECT_TIMEOUT", 30.0)),
            control_timeout_seconds=float(value("remote_control_timeout", "INSERTANY3D_REMOTE_CONTROL_TIMEOUT", 60.0)),
        )
    except (TypeError, ValueError, RemoteWorkerError) as exc:
        return None, f"服务器 Git provenance 配置无效: {exc}"
    return profile, None


def _remote_git_snapshot(
    profile: RemoteProfile,
    *,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Capture Git HEAD/status in the Server checkout over one SSH command."""

    run = runner or subprocess.run
    repository = profile.project_root
    script = (
        f"if ! cd -- {shlex.quote(profile.project_root)}; then "
        "printf 'ERROR\\tserver project root is unavailable\\n'; exit 1; fi; "
        "head=$(git rev-parse --verify HEAD 2>&1); head_code=$?; "
        "printf 'HEAD\\t%s\\n' \"$head\"; printf 'STATUS_BEGIN\\n'; "
        "git status --porcelain=v1 --untracked-files=all 2>&1; status_code=$?; "
        "printf 'STATUS_CODE\\t%s\\nSTATUS_END\\n' \"$status_code\""
    )
    try:
        outcome = run(
            RemoteCommandBuilder(profile).ssh(script),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=profile.control_timeout_seconds,
            env=_sanitized_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _unavailable_git_snapshot(repository, f"SSH: {exc}")
    stdout = str(getattr(outcome, "stdout", "") or "")
    stderr = str(getattr(outcome, "stderr", "") or "").strip()
    if getattr(outcome, "returncode", 1) != 0 and not stdout:
        return _unavailable_git_snapshot(repository, f"SSH exit code {getattr(outcome, 'returncode', 1)}: {stderr or 'no output'}")
    head = "unavailable"
    status_output: list[str] = []
    status_code: int | None = None
    in_status = False
    errors: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("HEAD\t"):
            candidate = line.split("\t", 1)[1].strip()
            if candidate:
                head = candidate.splitlines()[0]
        elif line == "STATUS_BEGIN":
            in_status = True
        elif line.startswith("STATUS_CODE\t"):
            try:
                status_code = int(line.split("\t", 1)[1])
            except ValueError:
                errors.append("远端 status 返回码无法解析")
            in_status = False
        elif line == "STATUS_END":
            in_status = False
        elif line.startswith("ERROR\t"):
            errors.append(line.split("\t", 1)[1].strip())
        elif in_status:
            status_output.append(line)
    if status_code == 0:
        status = "dirty" if status_output else "clean"
    else:
        status = "unavailable"
        errors.append("远端 git status 执行失败" if status_code is None else f"远端 git status exit code {status_code}")
    if head == "unavailable":
        errors.append("远端 git HEAD 执行失败")
    if stderr:
        errors.append(f"SSH: {stderr[:500]}")
    return {
        "repository": repository,
        "head": head,
        "status": status,
        "statusOutput": status_output if status != "unavailable" else [],
        "error": "; ".join(dict.fromkeys(errors))[:1000] or None,
    }


def _local_environment_source() -> dict[str, str]:
    """Describe the Local configuration source without including values."""

    local_root = Path(__file__).resolve().parents[1]
    configured = os.environ.get(LOCAL_ENVIRONMENT_FILE_VARIABLE)
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = local_root / candidate
        source = str(candidate.resolve())
        status = "loaded" if candidate.is_file() and not candidate.is_symlink() else "unavailable"
        return {"source": source, "status": status}
    candidate = local_root / DEFAULT_LOCAL_ENVIRONMENT_FILE
    if candidate.is_file() and not candidate.is_symlink():
        return {"source": str(candidate.resolve()), "status": "loaded"}
    return {"source": "process environment", "status": "not_configured"}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_run_provenance(
    batch_id: str,
    run_root: str | os.PathLike[str],
    args: argparse.Namespace | None = None,
) -> dict[str, Any] | None:
    """Append one non-blocking local/server Git snapshot to run_manifest.json."""

    root = Path(run_root).expanduser().resolve()
    local_root = Path(__file__).resolve().parents[1]
    local = _git_snapshot(local_root)
    profile, profile_error = _provenance_profile(args)
    if profile is None:
        server = _unavailable_git_snapshot(
            os.environ.get("INSERTANY3D_REMOTE_PROJECT_ROOT", "server checkout"),
            profile_error or "server profile unavailable",
        )
    else:
        server = _remote_git_snapshot(profile)
    configuration = {
        "local": _local_environment_source(),
        "server": {
            "source": profile.environment_file if profile is not None else ".insertany3d/runtime.env",
            "status": "configured" if profile is not None else "unavailable",
        },
    }
    captured_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    entry = {
        "capturedAtUtc": captured_at,
        "configurationSource": configuration,
        "local": local,
        "server": server,
    }
    path = root / _RUN_MANIFEST_NAME
    try:
        previous: dict[str, Any] = {}
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                previous = dict(loaded)
        history = previous.get("provenanceHistory")
        if not isinstance(history, list):
            history = []
        history.append(entry)
        previous.update(
            {
                "schemaVersion": 1,
                "kind": "insertany3d.run-manifest",
                "batchId": str(batch_id),
                "provenance": entry,
                "provenanceHistory": history,
            }
        )
        _write_json_atomic(path, previous)
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        # A provenance write failure must never block a batch. Keep the reason
        # in stderr so an operator can repair the run directory later.
        print(f"WARNING: 无法写入 run provenance（不影响运行）: {path}: {exc}", file=sys.stderr)
        return None
    return entry


def _review_display_path(path: str | os.PathLike[str]) -> str:
    """Shorten only the opaque image-edit attempt id shown to reviewers."""
    value = str(path)
    parts = Path(value).parts
    try:
        index = parts.index("image_edit")
    except ValueError:
        return value
    if index + 1 >= len(parts):
        return value
    token = parts[index + 1]
    if len(token) <= 10:
        return value
    shortened = parts[: index + 1] + (token[:10],) + parts[index + 2 :]
    return str(Path(*shortened))


def _is_review_image_path(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in _REVIEW_IMAGE_SUFFIXES


def _review_preview_path(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(root) / candidate
    return candidate.resolve()


def _is_wsl() -> bool:
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except (OSError, UnicodeError):
        return False


def _explorer_pids() -> set[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-x", "explorer.exe"], capture_output=True, text=True, check=False,
        )
    except OSError:
        return set()
    return {int(line) for line in result.stdout.splitlines() if line.strip().isdigit()}


def _open_review_preview(
    path: str | os.PathLike[str],
    *,
    stderr: TextIO = sys.stderr,
) -> tuple[subprocess.Popen[str] | None, set[int] | None]:
    """Open one review image using the host platform's default viewer."""
    candidate = Path(path).resolve()
    if os.name == "nt":
        try:
            os.startfile(str(candidate))  # type: ignore[attr-defined]
            print(f"已打开图片: {candidate}", file=stderr)
        except OSError as exc:
            print(f"图片预览打开失败（不影响审核）: {exc}", file=stderr)
        return None, None
    if _is_wsl() and shutil.which("explorer.exe") and shutil.which("wslpath"):
        try:
            converted = subprocess.run(
                ["wslpath", "-w", str(candidate)], capture_output=True, text=True, check=True,
            ).stdout.strip()
            if not converted:
                raise RuntimeError("wslpath 返回空路径")
            before = _explorer_pids()
            process = subprocess.Popen(["explorer.exe", converted], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            print(f"已打开图片: {converted}", file=stderr)
            return process, before
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            print(f"图片预览打开失败（不影响审核）: {exc}", file=stderr)
            return None, None
    opener = shutil.which("xdg-open") or shutil.which("open")
    if opener is None:
        return None, None
    try:
        process = subprocess.Popen([opener, str(candidate)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        print(f"已打开图片: {candidate}", file=stderr)
        return process, None
    except OSError as exc:
        print(f"图片预览打开失败（不影响审核）: {exc}", file=stderr)
        return None, None


def _build_review_contact_sheet(candidates: list[dict[str, Any]], output: str | os.PathLike[str]) -> Path | None:
    """Build one numbered contact sheet; return None when imaging is unavailable."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print(
            "审核候选拼图需要 Pillow；请使用项目的 `uv run --locked` 环境后重试。",
            file=sys.stderr,
        )
        return None
    loaded = []
    for candidate in candidates:
        path = Path(str(candidate.get("path", "")))
        if not path.is_file():
            continue
        try:
            image = Image.open(path).convert("RGB")
            loaded.append((int(candidate["index"]), image))
        except (OSError, ValueError):
            continue
    if not loaded:
        return None
    width = max(image.width for _, image in loaded)
    height = max(image.height for _, image in loaded) + 36
    sheet = Image.new("RGB", (width * len(loaded), height), "white")
    draw = ImageDraw.Draw(sheet)
    for position, (index, image) in enumerate(loaded):
        sheet.paste(image, (position * width, 36))
        draw.text((position * width + 8, 8), str(index), fill="black")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG")
    return target
def _close_review_preview(
    process: subprocess.Popen[str] | None,
    before: set[int] | None,
    *,
    stderr: TextIO = sys.stderr,
) -> None:
    """Close only the Explorer process started for this preview when possible."""
    if process is not None:
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            pass
    # explorer.exe may hand the request to a persistent process and exit. In that
    # case only terminate PIDs that appeared after our snapshot; never kill the
    # user's pre-existing Explorer processes.
    if before is None:
        return
    after = _explorer_pids()
    for pid in sorted(after - before):
        try:
            os.kill(pid, 15)
        except OSError:
            continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="insertany3d", description="InsertAny3D 本机批处理控制器")
    parser.add_argument("--db", type=Path, default=Path(".insertany3d/state.sqlite3"), help="SQLite 状态文件")
    subparsers = parser.add_subparsers(dest="group", required=True)
    batch = subparsers.add_parser("batch", help="批次规划、调度与恢复")
    commands = batch.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="校验显式清单并创建持久 DAG")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("--root", type=Path, required=True, help="该批次的运行产物根目录")
    plan.add_argument("--draft", action="store_true", help="只做结构检查；不能 start")

    for name in ("start", "resume", "doctor", "fake-run"):
        command = commands.add_parser(name)
        command.add_argument("batch_id")
        if name == "start":
            command.add_argument(
                "--canary",
                action="store_true",
                help="允许只启动一个本地小样本批次；正式批次仍要求 12 个工程/60 个任务",
            )

    status = commands.add_parser("status", help="查看一次状态，或持续刷新任务进度")
    status.add_argument("batch_id")
    status.add_argument("--watch", action="store_true", help="持续刷新，批次进入终态后自动退出")
    status.add_argument("--table", action="store_true", help="单次输出可读的逐任务状态表")
    status.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="持续刷新间隔秒数，默认 2 秒",
    )
    status.add_argument(
        "--json",
        action="store_true",
        help="持续刷新时强制每次输出一行 JSON；单次状态仍保持原 JSON 格式",
    )

    recovery = commands.add_parser(
        "recover-remote",
        help="探测并显式处理 delivery_unknown；不会启动新的远端任务",
    )
    recovery.add_argument("batch_id")
    recovery.add_argument("project_id")
    recovery.add_argument("task_id")
    recovery.add_argument("stage")
    recovery.add_argument("attempt", type=int)
    recovery.add_argument("--lease-token", required=True)
    recovery_action = recovery.add_mutually_exclusive_group(required=True)
    recovery_action.add_argument("--probe", action="store_true", help="只读查看原远端 attempt")
    recovery_action.add_argument("--recover-result", action="store_true", help="下载并提交已有 RESULT")
    recovery_action.add_argument("--retry", action="store_true", help="仅在确认 EXITED/MISSING 后释放并重试")
    recovery_action.add_argument("--terminal", action="store_true", help="仅在确认 EXITED/MISSING 后标记终止")
    recovery_action.add_argument(
        "--cancel-running",
        choices=("retry", "terminal"),
        metavar="{retry,terminal}",
        help="终止已确认身份的远端进程组；只有整组清理完成后才释放并重试/终止",
    )
    recovery.add_argument("--message")
    _add_remote_profile_arguments(recovery)

    worker = commands.add_parser(
        "worker",
        help="持续处理 ready 步骤；fake 和真实执行都必须显式选择",
    )
    worker.add_argument("batch_id")
    worker_mode = worker.add_mutually_exclusive_group()
    worker_mode.add_argument(
        "--fake",
        action="store_true",
        help="显式允许用占位产物验证队列；禁止用于真实实验数据库",
    )
    worker_mode.add_argument(
        "--real",
        action="store_true",
        help="显式运行 Unity、APIYi 和 SSH 远端步骤；缺少任一配置会在领取任务前失败",
    )
    worker.add_argument(
        "--worker-id",
        default="batch-worker",
        help="写入任务占用记录的 worker 名称",
    )
    worker.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="本次最多处理多少个步骤，防止异常情况下无限循环",
    )
    worker.add_argument(
        "--once",
        action="store_true",
        help="只处理一个步骤后返回",
    )
    worker.add_argument(
        "--idle-sleep-seconds",
        type=float,
        default=0.0,
        help="暂时没有可用步骤时的等待秒数；人工审核场景通常直接返回",
    )
    worker.add_argument(
        "--max-idle-polls",
        type=int,
        default=1,
        help="连续没有可租用步骤多少次后返回",
    )
    worker.add_argument("--max-parallel", type=int, default=1, help="单个 worker 同时执行的步骤上限；默认串行")
    worker.add_argument(
        "--unity-executable",
        default=os.environ.get("UNITY_EXECUTABLE"),
        help="Unity 可执行文件；也可设置 UNITY_EXECUTABLE",
    )
    worker.add_argument(
        "--image-endpoint",
        default=os.environ.get("GEMINI_IMAGE_URL"),
        help="APIYi generateContent 完整 URL；也可设置 GEMINI_IMAGE_URL",
    )
    worker.add_argument(
        "--image-model",
        default=os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview"),
    )
    worker.add_argument("--image-timeout", type=float, default=360.0)
    _add_remote_profile_arguments(worker)

    run_all = commands.add_parser(
        "run-all", help="续跑整个批次；人工审核后继续，并在全部 eval6 完成后运行 GPTEval"
    )
    run_all.add_argument("batch_id")
    run_all.add_argument("--manifest", type=Path, help="批次不存在时用于自动规划的清单")
    run_all.add_argument("--root", type=Path, help="批次不存在时的运行产物根目录")
    mode = run_all.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true", help="使用离线假 worker")
    mode.add_argument("--real", action="store_true", help="运行真实 Unity/APIYi/远端 worker")
    run_all.add_argument("--max-steps", type=int, default=10000)
    run_all.add_argument("--max-parallel", type=int, default=1, help="单个 worker 同时执行的步骤上限；默认串行")
    run_all.add_argument("--non-interactive", action="store_true", help="遇到人工审核时安全返回")
    run_all.add_argument(
        "--no-monitor",
        action="store_true",
        help="关闭 run-all 内置的持续状态监控；默认在 worker 执行时显示状态表",
    )
    run_all.add_argument(
        "--monitor-interval",
        type=float,
        default=2.0,
        help="run-all 状态监控刷新间隔秒数，默认 2 秒",
    )
    run_all.add_argument(
        "--no-open-review-images",
        action="store_true",
        help="审核时不自动打开编辑结果图片",
    )
    run_all.add_argument(
        "--json",
        action="store_true",
        help="输出完整机器可读 JSON；默认在流程结束时输出人类可读摘要",
    )
    run_all.add_argument("--evaluation-output", type=Path)
    eval_mode = run_all.add_mutually_exclusive_group()
    eval_mode.add_argument("--fake-score", type=int, metavar="1-10", help="用固定假分数完成离线评测")
    eval_mode.add_argument("--allow-paid-api", action="store_true", help="明确允许真实 GPTEval API")
    run_all.add_argument("--expected-tasks", type=int, default=3)
    run_all.add_argument("--expected-scenes", type=int, default=1)
    run_all.add_argument("--tasks-per-scene", type=int, default=3)
    _add_dimensions_argument(run_all)
    run_all.add_argument("--unity-executable", default=os.environ.get("UNITY_EXECUTABLE"))
    run_all.add_argument("--image-endpoint", default=os.environ.get("GEMINI_IMAGE_URL"))
    run_all.add_argument("--image-model", default=os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview"))
    run_all.add_argument("--image-timeout", type=float, default=360.0)
    _add_remote_profile_arguments(run_all)

    batch_evaluate = commands.add_parser(
        "evaluate",
        help="从该批次运行目录发现 eval6 清单并复用 GPTEval 入口",
    )
    batch_evaluate.add_argument("batch_id")
    batch_evaluate.add_argument(
        "--input",
        type=Path,
        help="评测清单目录；省略时使用批次运行根目录",
    )
    batch_evaluate.add_argument("--output", type=Path, required=True, help="评测缓存和汇总输出目录")
    batch_evaluate.add_argument("--metric", default="gpteval")
    batch_evaluate.add_argument("--model", default=os.environ.get("GEMINI_VLM_MODEL", DEFAULT_MODEL))
    batch_evaluate.add_argument("--evaluator-version", default=DEFAULT_EVALUATOR_VERSION)
    batch_evaluate.add_argument("--rubric-file", type=Path)
    batch_evaluate.add_argument("--repeats", type=int, default=1)
    batch_evaluate.add_argument("--expected-tasks", type=int, default=60)
    batch_evaluate.add_argument("--expected-scenes", type=int, default=12)
    batch_evaluate.add_argument("--tasks-per-scene", type=int, default=5)
    _add_dimensions_argument(batch_evaluate)
    evaluate_mode = batch_evaluate.add_mutually_exclusive_group()
    evaluate_mode.add_argument("--fake-score", type=int, metavar="1-10", help="使用本地固定假分数")
    evaluate_mode.add_argument("--allow-paid-api", action="store_true", help="明确允许调用 GPTEval API")
    batch_evaluate.add_argument("--base-url", default=os.environ.get("GEMINI_BASE_URL", DEFAULT_BASE_URL))
    batch_evaluate.add_argument("--timeout", type=float, default=300.0)
    batch_evaluate.add_argument("--retries", type=int, default=2)
    batch_evaluate.add_argument("--retry-delay-seconds", type=float, default=1.0)
    batch_evaluate.add_argument("--limit", type=int)

    stage_command = commands.add_parser(
        "stage-command",
        help="为一个已排队的步骤生成请求文件和命令；默认不启动，Unity 可用 --execute",
    )
    stage_command.add_argument("batch_id")
    stage_command.add_argument("project_id")
    stage_command.add_argument("task_id")
    stage_command.add_argument("stage", choices=(
        "unity_anchor", "unity_apply", "unity_eval6", "model_generation",
        "render_alignment_views", "segment_inputs", "gim_match", "estimate_pose",
        "sags_segment_vote", "debug_bundle",
    ))
    stage_command.add_argument("--worker-id", default="cli-stage-command")
    stage_command.add_argument("--request-path", type=Path)
    stage_command.add_argument("--unity-executable", default=os.environ.get("UNITY_EXECUTABLE", "Unity"))
    stage_command.add_argument("--adapter-path", type=Path)
    stage_command.add_argument("--python-executable", default=sys.executable)
    stage_command.add_argument(
        "--execute",
        action="store_true",
        help="直接执行 Unity 阶段并提交结果；远端重模型阶段禁止此选项",
    )
    stage_command.add_argument("--timeout", type=float, default=3600.0)

    retry = commands.add_parser("retry")
    retry.add_argument("batch_id")
    retry.add_argument("--project")
    retry.add_argument("--task")
    retry.add_argument("--stage")

    cancel = commands.add_parser("cancel")
    cancel.add_argument("batch_id")
    cancel.add_argument("--project")
    cancel.add_argument("--task")

    gc = commands.add_parser("gc")
    gc.add_argument("batch_id")
    gc.add_argument("--execute", action="store_true", help="实际删除；默认只列出")
    gc.add_argument("--owner-token", default=None)
    gc.add_argument("--older-than-seconds", type=float, default=86400)

    review = commands.add_parser("review")
    review_sub = review.add_subparsers(dest="review_command", required=True)
    review_list = review_sub.add_parser("list")
    review_list.add_argument("batch_id")
    review_list.add_argument("--page", type=int, default=1)
    review_list.add_argument("--size", type=int)
    decide = review_sub.add_parser("decide")
    decide.add_argument("batch_id")
    decide.add_argument("project_id")
    decide.add_argument("task_id")
    decide.add_argument("edit_attempt", type=int)
    decide.add_argument("decision", choices=("accepted", "rejected", "regenerate"))
    decide.add_argument("--note")

    evaluate = subparsers.add_parser("evaluate", help="验证 eval6 并运行 GPTEval")
    evaluate_commands = evaluate.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("plan", "完整预检并显示请求计划；不访问网络"),
        ("run", "执行缓存中缺失的 GPTEval 请求"),
        ("status", "重新预检并显示当前完成度"),
        ("summarize", "重新预检并写出 task/scene/batch 汇总"),
    ):
        command = evaluate_commands.add_parser(name, help=help_text)
        _add_evaluation_arguments(command)
        if name == "run":
            mode = command.add_mutually_exclusive_group()
            mode.add_argument(
                "--fake-score",
                type=int,
                metavar="1-10",
                help="使用固定本地假响应；不会访问网络",
            )
            mode.add_argument(
                "--allow-paid-api",
                action="store_true",
                help="明确允许调用可能产生费用的 GPTEval API",
            )
            command.add_argument("--base-url", default=os.environ.get("GEMINI_BASE_URL", DEFAULT_BASE_URL))
            command.add_argument("--timeout", type=float, default=300.0)
            command.add_argument("--retries", type=int, default=2)
            command.add_argument("--retry-delay-seconds", type=float, default=1.0)
            command.add_argument("--limit", type=int)
    return parser


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="包含 evaluation_manifest.json 的目录或单个清单")
    parser.add_argument("--output", type=Path, required=True, help="评测缓存和汇总输出目录")
    parser.add_argument("--metric", default="gpteval", help="当前只支持 gpteval")
    parser.add_argument("--model", default=os.environ.get("GEMINI_VLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--evaluator-version", default=DEFAULT_EVALUATOR_VERSION)
    parser.add_argument("--rubric-file", type=Path, help="可选的 UTF-8 评分规则文件")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--expected-tasks", type=int, default=60)
    parser.add_argument("--expected-scenes", type=int, default=12)
    parser.add_argument("--tasks-per-scene", type=int, default=5)
    _add_dimensions_argument(parser)


def _add_dimensions_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=None,
        metavar="DIMENSION",
        help=(
            "启用的评分维度，可用空格或逗号分隔；默认 visual_quality "
            "geometric_accuracy。insertion_rationality 仅在显式加入时运行"
        ),
    )


def _add_remote_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--remote-target",
        default=os.environ.get("INSERTANY3D_REMOTE_TARGET"),
        help="SSH host 或 user@host；也可设置 INSERTANY3D_REMOTE_TARGET",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=int(os.environ.get("INSERTANY3D_REMOTE_PORT", "22")),
    )
    parser.add_argument(
        "--remote-project-root",
        default=os.environ.get("INSERTANY3D_REMOTE_PROJECT_ROOT"),
        help="服务器 InsertAny3D 绝对路径",
    )
    parser.add_argument(
        "--remote-artifact-root",
        default=os.environ.get("INSERTANY3D_REMOTE_ARTIFACT_ROOT"),
        help="服务器批次产物绝对路径",
    )
    parser.add_argument(
        "--remote-python",
        default=os.environ.get(
            "INSERTANY3D_REMOTE_PYTHON", "third_party/TRELLIS/.venv/bin/python"
        ),
    )
    parser.add_argument("--remote-connect-timeout", type=float, default=30.0)
    parser.add_argument("--remote-control-timeout", type=float, default=60.0)
    parser.add_argument("--remote-transfer-timeout", type=float)
    parser.add_argument("--remote-poll-interval", type=float, default=5.0)
    parser.add_argument("--remote-status-timeout", type=float)


@dataclass(frozen=True)
class _EvaluationContext:
    manifests: list[EvaluationManifest]
    requests: list[GPTEvalRequest]
    cache: ResponseCache
    rubric: str
    rubric_source: str
    collection: Mapping[str, Any]
    dimensions: tuple[str, ...]
    comparison_config_sha256: str


class _StatusWatchInterrupted(Exception):
    pass


def _run_interactive_reviews(
    controller: BatchController,
    store: SchedulerStore,
    args: argparse.Namespace,
    reviews: list[dict[str, Any]],
) -> None:
    """Process the current review queue without leasing unrelated work."""
    for item in reviews:
        print(
            f"待审核 {item['project_id']}/{item['task_id']} attempt={item['edit_attempt']}",
            file=sys.stderr,
        )
        preview_handles: list[
            tuple[subprocess.Popen[str] | None, set[int] | None]
        ] = []
        review_candidates = item.get("reviewManifest", {}).get("candidates", [])
        artifacts = item.get("editArtifacts", [])
        if review_candidates:
            review_candidates = sorted(review_candidates, key=lambda candidate: int(candidate.get("index", 0)))
            print(f"  候选数量: {len(review_candidates)}", file=sys.stderr)
            if len(review_candidates) == 1:
                if not args.no_open_review_images:
                    preview_handles.append(_open_review_preview(review_candidates[0]["path"], stderr=sys.stderr))
            else:
                sheet = _build_review_contact_sheet(review_candidates, Path(str(review_candidates[0]["path"])).parent / "contact-sheet.png")
                if sheet is not None and not args.no_open_review_images:
                    preview_handles.append(_open_review_preview(sheet, stderr=sys.stderr))
        for artifact in artifacts if not review_candidates else []:
            artifact_path = artifact.get("path") or artifact.get("relativePath")
            if not artifact_path or not _is_review_image_path(artifact_path):
                continue
            print(f"  图片: {_review_display_path(artifact_path)}", file=sys.stderr)
            batch_row = store.row("SELECT root_path FROM batches WHERE batch_id=?", (args.batch_id,))
            preview_path = _review_preview_path(artifact_path, batch_row["root_path"])
            if args.no_open_review_images:
                handle, before = None, None
            else:
                handle, before = _open_review_preview(preview_path, stderr=sys.stderr)
            preview_handles.append((handle, before))
        try:
            while True:
                prompt = "审核 [Y=接受/R=重生成/N=取消]: " if len(review_candidates) <= 1 else f"审核 [1..{len(review_candidates)}=选择/R=重生成/N=取消]: "
                # ``run-all`` callers may capture stdout as machine-readable
                # JSON. Keep the human prompt on stderr and read stdin
                # without passing a prompt to ``input`` so it cannot
                # contaminate that JSON stream.
                print(prompt, end="", file=sys.stderr, flush=True)
                answer = input().strip().lower()
                answer = {"y": "accepted", "r": "regenerate", "n": "rejected"}.get(answer, answer)
                if answer in {"accepted", "regenerate", "rejected"}:
                    break
                if answer.isdigit() and any(int(candidate.get("index", -1)) == int(answer) for candidate in review_candidates):
                    selected = next(candidate for candidate in review_candidates if int(candidate["index"]) == int(answer))
                    print(f"已选择图片: {selected['path']}", file=sys.stderr)
                    break
        finally:
            for handle, before in preview_handles:
                _close_review_preview(handle, before, stderr=sys.stderr)
        controller.decide_edit(
            args.batch_id, item["project_id"], item["task_id"],
            int(item["edit_attempt"]), answer, decided_by="manual",
        )


def _run_worker_with_monitor(
    controller: BatchController,
    store: SchedulerStore,
    worker_args: argparse.Namespace,
    *,
    evaluation_transport: Transport | None = None,
    progress_cursor: list[int] | None = None,
) -> dict[str, Any]:
    """Run one durable worker pass while rendering live status snapshots.

    Worker execution is synchronous so exceptions and lease handling retain the
    existing semantics.  The monitor only reads the same WAL-backed store and
    stops before the store is closed, making it an observation layer rather
    than a second scheduler.
    """
    interval = float(getattr(worker_args, "monitor_interval", 2.0))
    enabled = not bool(getattr(worker_args, "no_monitor", False))
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("--monitor-interval 必须是有限的正数")
    stop = threading.Event()
    monitor_thread: threading.Thread | None = None
    progress_thread: threading.Thread | None = None
    progress_cursor = progress_cursor if progress_cursor is not None else [
        _latest_batch_event_id(store, worker_args.batch_id)
    ] if store is not None else None

    def monitor() -> None:
        first = True
        while not stop.is_set():
            try:
                snapshot = controller.status(worker_args.batch_id)
                print(_format_status_table(snapshot), file=sys.stderr)
                if snapshot.get("status") in _TERMINAL_BATCH_STATUSES:
                    return
            except (OSError, StoreError):
                return
            if not first:
                stop.wait(interval)
            else:
                first = False
                stop.wait(interval)

    if enabled:
        monitor_thread = threading.Thread(
            target=monitor,
            name="insertany3d-run-all-monitor",
            daemon=True,
        )
        monitor_thread.start()

    def progress() -> None:
        if store is None or progress_cursor is None:
            return
        while not stop.is_set():
            try:
                progress_cursor[0] = _emit_stage_completion_logs(
                    store,
                    worker_args.batch_id,
                    progress_cursor[0],
                    stderr=sys.stderr,
                )
            except (OSError, StoreError):
                return
            stop.wait(min(interval, 2.0))

    if store is not None and progress_cursor is not None:
        progress_thread = threading.Thread(
            target=progress,
            name="insertany3d-stage-progress",
            daemon=True,
        )
        progress_thread.start()
    try:
        return _run(controller, store, worker_args, evaluation_transport=evaluation_transport)
    finally:
        stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=max(1.0, min(interval + 1.0, 5.0)))
        if progress_thread is not None:
            progress_thread.join(timeout=max(1.0, min(interval + 1.0, 5.0)))
            progress_cursor[0] = _emit_stage_completion_logs(
                store,
                worker_args.batch_id,
                progress_cursor[0],
                stderr=sys.stderr,
            )


_STAGE_COMPLETION_EVENT_KINDS = frozenset({"stage_succeeded", "stage_failed"})


def _latest_batch_event_id(store: SchedulerStore, batch_id: str) -> int:
    """Return the event cursor used to avoid duplicate run-all progress logs."""

    row = store.row(
        "SELECT COALESCE(MAX(id), 0) AS event_id FROM events WHERE batch_id=?",
        (batch_id,),
    )
    return int(row["event_id"]) if row is not None else 0


def _stage_completion_log(
    event: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    started_at: float | None,
    finished_at: float | None,
) -> str:
    """Format one concise, loguru-like stage completion line."""

    kind = str(event.get("kind", ""))
    succeeded = kind == "stage_succeeded"
    level = "INFO" if succeeded else "WARNING"
    status = "succeeded" if succeeded else str(payload.get("nextState") or "failed")
    created_at = float(event.get("created_at") or time.time())
    timestamp = datetime.fromtimestamp(created_at).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    duration: str
    if started_at is not None and finished_at is not None:
        elapsed = finished_at - started_at
        duration = f"{elapsed:.1f}s" if math.isfinite(elapsed) and elapsed >= 0 else "-"
    else:
        duration = "-"
    fields = (
        f"project={event.get('project_id', '-')}",
        f"task={event.get('task_id', '-')}",
        f"stage={event.get('name', '-')}",
        f"status={status}",
        f"attempt={payload.get('attempt', '-')}",
        f"duration={duration}",
    )
    error_code = payload.get("errorCode")
    if error_code:
        fields += (f"error={error_code}",)
    return f"{timestamp} | {level} | stage.completed | " + " ".join(fields)


def _emit_stage_completion_logs(
    store: SchedulerStore,
    batch_id: str,
    after_event_id: int,
    *,
    stderr: TextIO = sys.stderr,
) -> int:
    """Print newly committed stage outcomes and return the advanced cursor.

    The cursor advances over every event, not only completion events. Manual
    review and heartbeat events may be interleaved with completions; consuming
    them here prevents a later run-all pass from printing an old completion a
    second time.
    """

    rows = store.rows(
        """SELECT e.id, e.stage_id, e.kind, e.payload_json, e.created_at,
                  s.project_id, s.task_id, s.name
             FROM events e
             LEFT JOIN stages s ON s.id=e.stage_id
            WHERE e.batch_id=? AND e.id>?
            ORDER BY e.id""",
        (batch_id, int(after_event_id)),
    )
    cursor = int(after_event_id)
    for row in rows:
        cursor = max(cursor, int(row["id"]))
        if row["kind"] not in _STAGE_COMPLETION_EVENT_KINDS:
            continue
        event = dict(row)
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, Mapping):
            payload = {}
        started_at = finished_at = None
        attempt = payload.get("attempt")
        if row["stage_id"] is not None and attempt is not None:
            try:
                attempt_row = store.row(
                    """SELECT started_at, finished_at FROM attempts
                       WHERE stage_id=? AND attempt_number=?""",
                    (int(row["stage_id"]), int(attempt)),
                )
            except (OSError, StoreError, ValueError, TypeError):
                attempt_row = None
            if attempt_row is not None:
                started_at = attempt_row["started_at"]
                finished_at = attempt_row["finished_at"]
        print(
            _stage_completion_log(
                event,
                payload=payload,
                started_at=float(started_at) if started_at is not None else None,
                finished_at=float(finished_at) if finished_at is not None else None,
            ),
            file=stderr,
            flush=True,
        )
    return cursor


def _run_all(
    controller: BatchController,
    store: SchedulerStore,
    args: argparse.Namespace,
    *,
    evaluation_transport: Transport | None = None,
) -> dict[str, Any]:
    """Drive planning, workers, review gates, and optional evaluation durably.

    The command is intentionally a loop over the existing durable worker.  A
    second invocation resumes the same database and therefore never repeats a
    succeeded stage or an accepted edit attempt.
    """
    row = store.row("SELECT root_path, status FROM batches WHERE batch_id=?", (args.batch_id,))
    if row is None:
        if args.manifest is None or args.root is None:
            raise StoreError("批次不存在；首次运行必须同时提供 --manifest 和 --root")
        manifest = load_manifest(args.manifest)
        planned = controller.plan(manifest, args.root, formal=False)
        if planned != args.batch_id:
            raise StoreError(f"清单 batchId={planned} 与命令 batch_id={args.batch_id} 不一致")
        controller.start(args.batch_id, formal=False)
        row = store.row("SELECT root_path, status FROM batches WHERE batch_id=?", (args.batch_id,))
    else:
        if args.manifest is not None:
            requested = load_manifest(args.manifest)
            stored_row = store.row("SELECT manifest_json FROM batches WHERE batch_id=?", (args.batch_id,))
            stored = json.loads(stored_row["manifest_json"])
            requested_tasks = {
                (str(project["projectId"]), str(task["taskId"]))
                for project in requested["projects"] for task in project["tasks"]
            }
            stored_tasks = {
                (str(project["projectId"]), str(task["taskId"]))
                for project in stored["projects"] for task in project["tasks"]
            }
            if requested_tasks != stored_tasks:
                raise StoreError("已有 batch_id 的任务集合与 --manifest 不一致；请使用新的 batch_id")
            if args.root is not None and Path(row["root_path"]).resolve() != args.root.resolve():
                raise StoreError("已有 batch_id 的运行目录与 --root 不一致；请使用新的 batch_id")
        if row["status"] == "planned":
            controller.start(args.batch_id, formal=False)

    run_root = str(Path(row["root_path"]).resolve())
    run_provenance = _record_run_provenance(args.batch_id, run_root, args)
    event_cursor = _latest_batch_event_id(store, args.batch_id)

    def with_run_context(payload: dict[str, Any]) -> dict[str, Any]:
        payload.setdefault("runRoot", run_root)
        if run_provenance is not None:
            payload.setdefault("provenance", run_provenance)
        queue = payload.get("queue")
        if isinstance(queue, Mapping) and queue.get("tasks") is not None:
            payload.setdefault("statusSnapshot", queue)
        return payload

    worker_argv = argparse.Namespace(**vars(args))
    worker_argv.command = "worker"
    worker_argv.fake = bool(args.fake)
    worker_argv.real = bool(args.real)
    worker_argv.worker_id = "run-all-worker"
    worker_argv.once = False
    # The run-all controller captured provenance before entering its worker
    # loop.  Avoid repeating an SSH Git query on every worker pass while
    # keeping standalone ``batch worker`` calls independently auditable.
    worker_argv._provenance_recorded = True
    worker_argv.idle_sleep_seconds = 0.0
    worker_argv.max_idle_polls = 1
    eval6_complete = False
    while True:
        # Review gates are operator work and must take precedence over leasing
        # more stages.  In particular, a regenerate decision can create a new
        # pending review while other tasks still have ready remote stages.
        reviews = controller.review_page(args.batch_id, page=1, size=None)
        if reviews:
            if args.non_interactive or not sys.stdin.isatty():
                return with_run_context({
                    "status": "waiting_manual_review",
                    "queue": controller.status(args.batch_id),
                    "reviews": reviews,
                })
            _run_interactive_reviews(controller, store, args, reviews)
            # Re-enter at the review gate.  This lets a regenerate-produced
            # review be surfaced before any unrelated ready stage is leased.
            continue

        # evaluate_absolute is a controller finalizer, not a worker stage.
        # Stop leasing before it becomes "ready" so _run below can submit the
        # complete GPTEval result atomically after validating all eval6 inputs.
        eval6_states = store.rows(
            "SELECT state FROM stages WHERE batch_id=? AND name='unity_eval6'",
            (args.batch_id,),
        )
        if eval6_states and all(str(item["state"]) == "succeeded" for item in eval6_states):
            eval6_complete = True
            break
        progress_cursor = [event_cursor]
        report = _run_worker_with_monitor(
            controller,
            store,
            worker_argv,
            evaluation_transport=evaluation_transport,
            progress_cursor=progress_cursor,
        )
        # ``_run_worker_with_monitor`` flushes the shared cursor on exit.  A
        # final pass here covers implementations that return before starting
        # the progress thread (for example a mocked worker in tests).
        event_cursor = progress_cursor[0]
        event_cursor = _emit_stage_completion_logs(store, args.batch_id, event_cursor, stderr=sys.stderr)
        snapshot = report.get("status", controller.status(args.batch_id))
        print(_format_status_table(snapshot), file=sys.stderr)
        for error in report.get("submissionErrors", []):
            print(
                "WORKER ERROR "
                f"{error.get('projectId')}/{error.get('taskId')} "
                f"{error.get('stage')} attempt={error.get('attempt')}: "
                f"{error.get('errorCode')} {error.get('message')}",
                file=sys.stderr,
            )
        # A worker may have leased the controller-only evaluate_absolute stage
        # in the same pass (notably the fake executor).  Inspect eval6 before
        # honoring the aggregate terminal status so run-all still invokes the
        # configured evaluator and returns its output.
        eval6_states = store.rows(
            "SELECT state FROM stages WHERE batch_id=? AND name='unity_eval6'",
            (args.batch_id,),
        )
        if eval6_states and all(str(item["state"]) == "succeeded" for item in eval6_states):
            eval6_complete = True
            break
        if snapshot.get("status") in _TERMINAL_BATCH_STATUSES:
            break
        # A worker may have produced a review during this pass.  Return to the
        # top so it is handled before considering terminal/blocked outcomes.
        if controller.review_page(args.batch_id, page=1, size=None):
            continue
        if report.get("blockedReason") not in {None, "batch_succeeded", "batch_failed", "batch_canceled"}:
            return with_run_context({"status": "blocked", "blockedReason": report.get("blockedReason"), "queue": snapshot})
        break

    snapshot = controller.status(args.batch_id)
    # The durable DAG includes evaluate_absolute as its final stage.  Once all
    # eval6 evidence is committed, the batch remains "running" until this
    # controller-side finalizer submits the evaluation result.  Do not require
    # the aggregate batch status to be "succeeded" here or run-all would skip
    # evaluation on every normal pre-evaluation batch.
    if eval6_complete and snapshot.get("status") not in {"failed", "rejected", "canceled"}:
        if args.fake_score is None and not args.allow_paid_api:
            return with_run_context({"status": "succeeded", "queue": snapshot, "evaluation": "skipped (use --fake-score or --allow-paid-api)"})
        batch = store.row("SELECT root_path FROM batches WHERE batch_id=?", (args.batch_id,))
        output = args.evaluation_output or (Path(batch["root_path"]) / "evaluation")
        eval_args = argparse.Namespace(
            command="evaluate", batch_id=args.batch_id, input=None, output=output,
            metric="gpteval", model=os.environ.get("GEMINI_VLM_MODEL", DEFAULT_MODEL),
            evaluator_version=DEFAULT_EVALUATOR_VERSION, rubric_file=None, repeats=1,
            expected_tasks=args.expected_tasks, expected_scenes=args.expected_scenes,
            tasks_per_scene=args.tasks_per_scene, dimensions=args.dimensions,
            fake_score=args.fake_score, allow_paid_api=args.allow_paid_api,
            base_url=os.environ.get("GEMINI_BASE_URL", DEFAULT_BASE_URL), timeout=300.0,
            retries=2, retry_delay_seconds=1.0, limit=None,
        )
        evaluation = _run(controller, store, eval_args, evaluation_transport=evaluation_transport)
        return with_run_context({
            "status": "succeeded",
            "queue": evaluation["queue"]["status"],
            "statusSnapshot": controller.status(args.batch_id),
            "evaluation": evaluation["evaluation"],
        })
    if snapshot.get("status") == "failed":
        skipped = store.row(
            """SELECT COUNT(*) AS count FROM stages
               WHERE batch_id=? AND name='evaluate_absolute' AND state='canceled'
                 AND last_error_code=?""",
            (args.batch_id, EVALUATION_SKIPPED_ERROR),
        )
        if skipped is not None and int(skipped["count"]):
            return with_run_context({
                "status": "failed",
                "blockedReason": EVALUATION_SKIPPED_ERROR,
                "message": "至少一个任务未完成，未执行整批 GPTEval",
                "queue": snapshot,
                "evaluation": "skipped",
            })
    return with_run_context({"status": snapshot.get("status"), "queue": snapshot})


def main(
    argv: list[str] | None = None,
    *,
    evaluation_transport: Transport | None = None,
) -> int:
    try:
        # Load Local-only configuration before argparse evaluates its env
        # backed defaults.  The loader is additive, so shell exports remain
        # authoritative and the Server runtime.env is never searched here.
        load_local_environment()
        args = build_parser().parse_args(argv)
        if args.group == "evaluate":
            result = _run_evaluate(args, evaluation_transport=evaluation_transport)
        else:
            with SchedulerStore(args.db) as store:
                controller = BatchController(store)
                result = _run(
                    controller,
                    store,
                    args,
                    evaluation_transport=evaluation_transport,
                )
        if result is not None:
            if getattr(args, "command", None) == "status" and getattr(args, "table", False):
                print(_format_status_table(result))
            elif getattr(args, "command", None) == "run-all" and not getattr(args, "json", False):
                print(_format_run_summary(result))
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            xlsx_path = result.get("outputs", {}).get("xlsx") if isinstance(result, dict) else None
            if xlsx_path:
                print(f"XLSX: {xlsx_path}", file=sys.stderr)
        return 0
    except _StatusWatchInterrupted:
        print("已停止状态监视", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        # Interrupting a worker is deliberately recoverable rather than an
        # implicit cancel: an in-flight API/remote request may have completed
        # after the local process stopped.  The next resume pass reconciles
        # expired local leases; remote leases remain fenced for probe/recovery.
        print(
            "已中断；活动 lease 未被强制释放。请等待本地 lease 到期后执行 resume --run-id <run_id>；"
            "若出现 recovering，请先使用 recover-remote 探测原 attempt。",
            file=sys.stderr,
        )
        return 130
    except (ContractError, EvaluationError, StageWiringError, StoreError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _run(
    controller: BatchController,
    store: SchedulerStore,
    args: argparse.Namespace,
    *,
    evaluation_transport: Transport | None = None,
) -> Any:
    if args.command == "plan":
        manifest = load_manifest(args.manifest)
        batch_id = controller.plan(manifest, args.root, formal=not args.draft)
        return {"batchId": batch_id, "status": "planned", "formal": not args.draft}
    if args.command == "start":
        controller.start(args.batch_id, formal=not args.canary)
        batch = store.row("SELECT root_path FROM batches WHERE batch_id=?", (args.batch_id,))
        provenance = _record_run_provenance(
            args.batch_id,
            batch["root_path"] if batch is not None else Path("."),
            args,
        )
        result = controller.status(args.batch_id)
        if provenance is not None:
            result["provenance"] = provenance
        return result
    if args.command == "stage-command":
        batch = store.row("SELECT root_path FROM batches WHERE batch_id=?", (args.batch_id,))
        if batch is None:
            raise StoreError(f"batch 不存在: {args.batch_id}")
        # stage-command is the explicit prepare entry point. Capture the
        # source state before leasing so a prepare failure still has evidence.
        _record_run_provenance(args.batch_id, batch["root_path"], args)
        if args.execute:
            if not args.stage.startswith("unity_"):
                raise StageWiringError("--execute 目前只允许 unity_anchor/unity_apply/unity_eval6，不会启动远端重模型")
            project_root = validate_unity_project_for_batch(store, args.batch_id, args.project_id)
            ensure_unity_project_not_running(project_root)
        item = controller.lease_next(
            args.batch_id,
            args.worker_id,
            default_capacities(json.loads(store.row(
                "SELECT manifest_json FROM batches WHERE batch_id=?", (args.batch_id,)
            )["manifest_json"])),
            project_id=args.project_id,
            task_id=args.task_id,
            stage_name=args.stage,
        )
        if item is None:
            raise StageWiringError("目标步骤当前不是 ready，或资源槽位暂时不可用")
        request, request_path = controller.write_stage_request(item, args.request_path)
        command = controller.build_stage_command(
            item,
            request_path,
            unity_executable=args.unity_executable,
            adapter_path=args.adapter_path,
            python_executable=args.python_executable,
        )
        outcome = None
        queue_state = None
        if args.execute:
            outcome = CommandExecutor(heartbeat_seconds=15.0).execute(
                controller,
                item,
                command,
                timeout_seconds=args.timeout,
            )
            if outcome.succeeded:
                controller.commit_success(item, outcome.artifacts)
                queue_state = "succeeded"
            else:
                queue_state = controller.fail(
                    item,
                    outcome.error_code or "worker_crash",
                    outcome.message,
                    cleanup_completed=outcome.cleanup_completed,
                    stage_status=outcome.stage_status,
                )
        result = {
            "batchId": item.batch_id,
            "projectId": item.project_id,
            "taskId": item.task_id,
            "stage": item.stage,
            "attempt": item.attempt,
            "requestPath": str(request_path),
            "resultPath": str(item.staging_dir / "stage_result.json"),
            "resources": item.resources,
            "command": command,
            "executed": bool(args.execute),
            "request": request,
        }
        if outcome is not None:
            result.update(
                {
                    # This is the durable scheduler state after result handling,
                    # not merely an indication that the child process started.
                    "stageStatus": queue_state,
                    "stageResultStatus": (
                        "succeeded"
                        if outcome.succeeded
                        else (outcome.stage_status or "failed_without_stage_result")
                    ),
                    "succeeded": outcome.succeeded,
                    "errorCode": outcome.error_code,
                    "message": outcome.message,
                }
            )
        return result
    if args.command == "resume":
        recovered = controller.resume(args.batch_id)
        batch = store.row("SELECT root_path FROM batches WHERE batch_id=?", (args.batch_id,))
        provenance = _record_run_provenance(
            args.batch_id,
            batch["root_path"] if batch is not None else Path("."),
            args,
        )
        result = {"recoveredLeases": recovered, **controller.status(args.batch_id)}
        if provenance is not None:
            result["provenance"] = provenance
        return result
    if args.command == "status":
        if not math.isfinite(args.interval) or args.interval <= 0:
            raise ValueError("--interval 必须是有限的正数")
        if args.watch:
            try:
                _watch_batch_status(
                    controller,
                    args.batch_id,
                    interval_seconds=args.interval,
                    json_lines=args.json or not sys.stdout.isatty(),
                )
            except KeyboardInterrupt as exc:
                raise _StatusWatchInterrupted from exc
            return None
        return controller.status(args.batch_id)
    if args.command == "recover-remote":
        manager = RemoteRecoveryManager(controller, _remote_profile_from_args(args))
        identity = (
            args.batch_id,
            args.project_id,
            args.task_id,
            args.stage,
            args.attempt,
            args.lease_token,
        )
        if args.probe:
            return manager.probe(*identity).as_dict()
        if args.recover_result:
            return manager.recover_result(*identity).as_dict()
        if args.cancel_running:
            return manager.cancel_running(
                *identity,
                action=args.cancel_running,
                message=args.message,
            ).as_dict()
        return manager.resolve_stopped(
            *identity,
            action="retry" if args.retry else "terminal",
            message=args.message,
        ).as_dict()
    if args.command == "retry":
        count = controller.retry(args.batch_id, project_id=args.project, task_id=args.task, stage_name=args.stage)
        return {"retried": count, **controller.status(args.batch_id)}
    if args.command == "cancel":
        count = controller.cancel(args.batch_id, project_id=args.project, task_id=args.task)
        return {"canceledStages": count, **controller.status(args.batch_id)}
    if args.command == "doctor":
        return controller.doctor(args.batch_id)
    if args.command == "gc":
        token = args.owner_token or secrets.token_hex(16)
        targets = controller.gc(
            args.batch_id,
            owner_token=token,
            dry_run=not args.execute,
            older_than_seconds=args.older_than_seconds,
        )
        return {"dryRun": not args.execute, "ownerToken": token, "targets": targets}
    if args.command == "fake-run":
        row = store.row("SELECT manifest_json FROM batches WHERE batch_id=?", (args.batch_id,))
        if row is None:
            raise StoreError(f"batch 不存在: {args.batch_id}")
        batch = store.row("SELECT root_path FROM batches WHERE batch_id=?", (args.batch_id,))
        if not getattr(args, "_provenance_recorded", False):
            _record_run_provenance(args.batch_id, batch["root_path"], args)
        manifest = json.loads(row["manifest_json"])
        runner = FakeRunner(controller, default_capacities(manifest))
        status = runner.run_until_blocked(args.batch_id)
        return {**status, "peakResources": runner.peak_resources}
    if args.command == "worker":
        if not args.fake and not args.real:
            raise StoreError("必须显式选择 --fake 或 --real；两种模式都不会默认启用")
        row = store.row("SELECT manifest_json FROM batches WHERE batch_id=?", (args.batch_id,))
        if row is None:
            raise StoreError(f"batch 不存在: {args.batch_id}")
        batch = store.row("SELECT root_path FROM batches WHERE batch_id=?", (args.batch_id,))
        if not getattr(args, "_provenance_recorded", False):
            _record_run_provenance(args.batch_id, batch["root_path"], args)
        manifest = json.loads(row["manifest_json"])
        capacities = default_capacities(manifest)
        executor = None
        if args.real:
            if manifest["editPolicy"]["mode"] != "manual":
                raise StoreError(
                    "真实 worker 当前只允许 manual 图片审核；"
                    "automatic 仍缺少已批准的图片完整性验收，不能直接放行付费结果"
                )
            missing = []
            for name, value in (
                ("--unity-executable / UNITY_EXECUTABLE", args.unity_executable),
                ("--image-endpoint / GEMINI_IMAGE_URL", args.image_endpoint),
                ("--remote-target / INSERTANY3D_REMOTE_TARGET", args.remote_target),
                ("--remote-project-root / INSERTANY3D_REMOTE_PROJECT_ROOT", args.remote_project_root),
                ("--remote-artifact-root / INSERTANY3D_REMOTE_ARTIFACT_ROOT", args.remote_artifact_root),
            ):
                if not value:
                    missing.append(name)
            try:
                api_key, _key_source = load_apiyi_api_key()
            except ApiYiCredentialError as exc:
                missing.append(str(exc))
            if missing:
                raise StoreError("真实 worker 缺少配置: " + ", ".join(missing))
            remote_profile = _remote_profile_from_args(args)
            try:
                verify_remote_runtime(remote_profile)
            except ValueError as exc:
                raise StoreError(f"真实 worker 的服务器运行时预检失败: {exc}") from exc
            # Refuse before leasing anything when a Unity project is invalid or
            # already open in the GUI.  A project-lock error must not consume a
            # one-attempt Unity stage.
            for project in manifest["projects"]:
                project_root = validate_unity_project_for_batch(
                    store,
                    args.batch_id,
                    str(project["projectId"]),
                )
                ensure_unity_project_not_running(project_root)
            image = ImageWorkerConfig(
                endpoint=args.image_endpoint,
                token=api_key,
                model=args.image_model,
                timeout_seconds=args.image_timeout,
            )
            local = LocalStageExecutor(
                allow_real=True,
                unity_executable=args.unity_executable,
                image_config=image,
            )
            remote = RemoteStageExecutor(
                remote_profile
            )
            executor = CompositeStageExecutor([local, remote])
            capacities = local_worker_capacities(manifest)
        worker = BatchWorker(
            controller,
            capacities,
            executor,
            worker_id=args.worker_id,
            idle_sleep_seconds=args.idle_sleep_seconds,
            max_idle_polls=args.max_idle_polls,
            max_parallel=args.max_parallel,
        )
        return worker.run(
            args.batch_id,
            max_steps=args.max_steps,
            once=args.once,
        ).as_dict()
    if args.command == "run-all":
        return _run_all(controller, store, args, evaluation_transport=evaluation_transport)
    if args.command == "evaluate":
        batch = store.row("SELECT root_path, status FROM batches WHERE batch_id=?", (args.batch_id,))
        if batch is None:
            raise StoreError(f"batch 不存在: {args.batch_id}")
        queue_gate = _evaluation_queue_gate(controller, args.batch_id)
        committed_manifest_paths = None
        if queue_gate in {"ready", "already_finalized"}:
            if args.input is not None:
                raise StoreError(
                    "运行中或已完成批次的 evaluate 固定读取数据库已提交的 unity_eval6 清单；"
                    "--input 只用于 planned/draft 的离线清单检查"
                )
            committed_manifest_paths = _committed_evaluation_manifest_paths(
                controller,
                args.batch_id,
            )
        evaluation_args = argparse.Namespace(
            command="run",
            input=args.input or Path(batch["root_path"]),
            output=args.output,
            metric=args.metric,
            model=args.model,
            evaluator_version=args.evaluator_version,
            rubric_file=args.rubric_file,
            repeats=args.repeats,
            expected_tasks=args.expected_tasks,
            expected_scenes=args.expected_scenes,
            tasks_per_scene=args.tasks_per_scene,
            dimensions=args.dimensions,
            fake_score=args.fake_score,
            allow_paid_api=args.allow_paid_api,
            base_url=args.base_url,
            timeout=args.timeout,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
            limit=args.limit,
            expected_batch_id=args.batch_id,
            manifest_paths=committed_manifest_paths,
        )
        result = _run_evaluate(
            evaluation_args,
            evaluation_transport=evaluation_transport,
        )
        finalized = 0
        if result.get("status") == "ready" and queue_gate == "ready":
            finalized = _commit_evaluation_stages(
                controller,
                args.batch_id,
                args.output,
                default_capacities(json.loads(store.row(
                    "SELECT manifest_json FROM batches WHERE batch_id=?", (args.batch_id,)
                )["manifest_json"])),
            )
        return {
            "batchId": args.batch_id,
            "evaluation": result,
            "queue": {
                "gate": queue_gate,
                "finalizedStages": finalized,
                "status": controller.status(args.batch_id),
            },
        }
    if args.command == "review":
        if args.review_command == "list":
            return {"items": controller.review_page(args.batch_id, page=args.page, size=args.size)}
        controller.decide_edit(
            args.batch_id,
            args.project_id,
            args.task_id,
            args.edit_attempt,
            args.decision,
            decided_by="manual",
            note=args.note,
        )
        return controller.status(args.batch_id)
    raise ValueError(f"未知命令: {args.command}")


_TERMINAL_BATCH_STATUSES = frozenset({"succeeded", "failed", "canceled"})
_STAGE_STATE_ORDER = (
    "succeeded",
    "running",
    "leased",
    "ready",
    "waiting_review",
    "waiting_manual",
    "recovering",
    "failed_retryable",
    "failed_terminal",
    "rejected",
    "canceled",
    "pending",
)


def _watch_batch_status(
    controller: BatchController,
    batch_id: str,
    *,
    interval_seconds: float,
    json_lines: bool,
    stream: TextIO | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_updates: int | None = None,
) -> None:
    """Render snapshots until the batch finishes or the caller interrupts."""
    output = stream or sys.stdout
    updates = 0
    while True:
        snapshot = controller.status(batch_id)
        if json_lines:
            output.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            if updates:
                output.write("\x1b[2J\x1b[H")
            output.write(_format_status_table(snapshot) + "\n")
        output.flush()
        updates += 1
        if snapshot.get("status") in _TERMINAL_BATCH_STATUSES:
            return
        if max_updates is not None and updates >= max_updates:
            return
        sleep(interval_seconds)


_STATUS_RESOURCE_LABELS = {
    "remote_gpu": "GPU",
    "image_api": "API",
    "upload": "上传",
    "download": "下载",
    "unity": "Unity",
    "evaluation_api": "评测",
    "remote_cpu": "服务器",
    "remote_io": "服务器",
    "ssh_io": "服务器",
}


def _status_resource(task: Mapping[str, Any]) -> str | None:
    resource = task.get("queueResource")
    if resource:
        return str(resource)
    return status_resource_for_stage(task.get("current_stage"))


def _gpu_status_suffix(resources: Mapping[str, Any]) -> str:
    value = resources.get("remote_gpu")
    if value is None:
        return ""
    text = str(value)
    if text.startswith("gpu:"):
        text = text[4:]
    return f" gpu #{text}" if text else " gpu"


def _format_stage_status(task: Mapping[str, Any], observed: float) -> str:
    """Map raw stage/lease state to the concise operator-facing label."""

    state = str(task.get("stage_state") or "")
    resource = _status_resource(task)
    resources = task.get("resources") or {}
    resource_label = _STATUS_RESOURCE_LABELS.get(resource or "", "服务器")
    if state == "ready":
        not_before = float(task.get("notBefore") or 0.0)
        if not_before > observed:
            return "重试等待"
        position = task.get("resourceQueuePosition")
        if position is None:
            position = task.get("readyQueuePosition")
        suffix = f"#{int(position)} " if position is not None else ""
        return f"排队-{suffix}{resource_label}"
    if state == "leased":
        if resource == "remote_gpu":
            return f"服务器准备{_gpu_status_suffix(resources)}"
        return f"{resource_label}准备" if resource else "服务器准备"
    if state == "running":
        if resource == "remote_gpu":
            return f"服务器运行{_gpu_status_suffix(resources)}"
        return {
            "image_api": "API调用",
            "upload": "上传",
            "download": "下载",
            "unity": "Unity操作",
            "evaluation_api": "评测调用",
            "remote_cpu": "服务器运行",
            "remote_io": "服务器运行",
            "ssh_io": "服务器运行",
        }.get(resource or "", "服务器运行")
    return {
        "committing": "提交结果",
        "waiting_review": "等待图片审批",
        "waiting_manual": "等待人工处理",
        "recovering": "等待恢复确认",
        "suspect": "状态待确认",
        "failed_retryable": "失败，准备重试",
        "failed_terminal": "失败，已终止",
        "rejected": "已拒绝",
        "canceled": "已取消",
        "pending": "等待调度",
    }.get(state, state or "等待调度")


def _format_blocker_status(blocker: Mapping[str, Any], observed: float) -> str:
    """Describe a pending stage using the predecessor that actually blocks it."""

    state = str(blocker.get("stageState") or "")
    if blocker.get("stage") == "edit_gate" and state in {"pending", "waiting_review"}:
        return "等待图片审批"
    blocker_task = {
        "stage_state": state,
        "current_stage": blocker.get("stage"),
        "queueResource": blocker.get("queueResource"),
        "resourceQueuePosition": blocker.get("resourceQueuePosition"),
        "readyQueuePosition": blocker.get("readyQueuePosition"),
        "notBefore": blocker.get("notBefore"),
        "resources": blocker.get("resources") or {},
    }
    label = _format_stage_status(blocker_task, observed)
    if state == "failed_terminal":
        reason = blocker.get("errorCode") or blocker.get("message")
        return f"前置失败: {reason}" if reason else "前置失败"
    if state == "rejected":
        return "前置已拒绝"
    if state == "canceled":
        return "前置已取消"
    return label


def _format_task_status(task: Mapping[str, Any], observed: float) -> str:
    if task.get("status") == "succeeded":
        return "完成"
    if task.get("stage_state") == "pending":
        if task.get("current_stage") == "edit_gate":
            return "等待图片审批"
        blocker = task.get("blockedBy")
        if isinstance(blocker, Mapping):
            return _format_blocker_status(blocker, observed)
        return "等待调度"
    return _format_stage_status(task, observed)


def _format_status_table(snapshot: Mapping[str, Any]) -> str:
    counts = snapshot.get("stageCounts", {})
    ordered_states = [state for state in _STAGE_STATE_ORDER if state in counts]
    ordered_states.extend(sorted(set(counts) - set(ordered_states)))
    count_text = " ".join(f"{state}={counts[state]}" for state in ordered_states) or "无步骤"
    observed = float(snapshot.get("observedAt") or time.time())
    active = snapshot.get("activeLeases", 0)
    lines = [
        f"InsertAny3D  批次: {snapshot.get('batchId', '-')}  状态: {snapshot.get('status', '-')}",
        f"步骤: {count_text}    本地并行: {active}",
        "",
        f"{'PROJECT':<22} {'TASK':<10} {'CURRENT STAGE':<28} {'STAGE TIME':<12} {'TASK TIME':<12} {'STATUS':<24} {'HEARTBEAT':<12} ERROR SUMMARY",
        f"{'-' * 22} {'-' * 10} {'-' * 28} {'-' * 12} {'-' * 12} {'-' * 24} {'-' * 12} {'-' * 48}",
    ]
    for task in snapshot.get("tasks", []):
        stage_elapsed = _format_elapsed(task.get("stageElapsedSeconds"))
        task_elapsed = _format_elapsed(task.get("taskElapsedSeconds"))
        status_label = _format_task_status(task, observed)
        heartbeat = task.get("heartbeatAt")
        heartbeat_text = f"{max(0, observed - float(heartbeat)):.0f}s ago" if heartbeat else "-"
        error_code = task.get("last_error_code")
        error_message = task.get("last_message")
        error_summary = ""
        if error_code or error_message:
            error_summary = ": ".join(str(value) for value in (error_code, error_message) if value)
        lines.append(
            " ".join(
                (
                    _fixed_cell(task.get("project_id"), 22),
                    _fixed_cell(task.get("task_id"), 10),
                    _fixed_cell(task.get("current_stage") or "-", 28),
                    _fixed_cell(stage_elapsed, 12),
                    _fixed_cell(task_elapsed, 12),
                    _fixed_cell(status_label, 24),
                    _fixed_cell(heartbeat_text, 12),
                    _fixed_cell(error_summary, 48),
                )
            )
        )
    return "\n".join(lines)


_SUMMARY_ARTIFACT_STAGES = (
    "unity_anchor", "image_edit", "upload_inputs", "model_generation",
    "render_alignment_views", "segment_inputs", "gim_match", "estimate_pose",
    "sags_segment_vote", "debug_bundle", "download_results", "unity_apply", "unity_eval6",
)


def _format_run_summary(result: Mapping[str, Any]) -> str:
    """Render the end-of-run information an operator needs, without internals."""
    snapshot = result.get("statusSnapshot")
    if not isinstance(snapshot, Mapping):
        queue = result.get("queue")
        snapshot = queue if isinstance(queue, Mapping) else {}
    root = Path(str(result.get("runRoot") or "-")).resolve() if result.get("runRoot") else None
    tasks = list(snapshot.get("tasks") or [])
    completed = [task for task in tasks if task.get("status") == "succeeded"]
    lines = [
        "\n=== InsertAny3D 运行摘要 ===",
        f"批次: {snapshot.get('batchId', result.get('batchId', '-'))}",
        f"批次结果: {result.get('status', snapshot.get('status', '-'))}（完成 {len(completed)}/{len(tasks)}）",
    ]
    if root is not None:
        lines.append(f"运行目录: {root}")
    evaluation = result.get("evaluation")
    if evaluation:
        if isinstance(evaluation, str):
            lines.append(f"整批评价: {evaluation}")
        elif isinstance(evaluation, Mapping):
            lines.append(f"整批评价: {evaluation.get('status', '-')}" )
    lines.append("")
    for task in tasks:
        project_id = str(task.get("project_id", "-"))
        task_id = str(task.get("task_id", "-"))
        status = str(task.get("status", "-"))
        elapsed = _format_elapsed(task.get("taskElapsedSeconds"))
        if status == "succeeded":
            label = "完成"
        else:
            label = "未完成"
        lines.append(f"[{label}] {project_id}/{task_id}  耗时: {elapsed}")
        if status != "succeeded":
            stage = task.get("current_stage") or "未知阶段"
            state = task.get("stage_state") or status
            reason = ": ".join(str(value) for value in (task.get("last_error_code"), task.get("last_message")) if value)
            lines.append(f"  卡点: {stage}（{state}）")
            if reason:
                lines.append(f"  原因: {reason}")
        if root is not None:
            task_root = root / project_id / task_id
            lines.append(f"  任务目录: {task_root}")
            existing = [task_root / "artifacts" / stage for stage in _SUMMARY_ARTIFACT_STAGES
                        if (task_root / "artifacts" / stage).is_dir()]
            if existing:
                lines.append("  阶段产物:")
                lines.extend(f"    {path}" for path in existing)
    if not tasks:
        lines.append("没有任务状态可供显示。")
    return "\n".join(lines)


def _format_elapsed(value: Any) -> str:
    if value is None:
        return "-"
    seconds = max(0, int(float(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _evaluation_queue_gate(controller: BatchController, batch_id: str) -> str:
    """Prevent paid evaluation before every task has committed eval6 evidence."""

    batch = controller.store.row("SELECT status FROM batches WHERE batch_id=?", (batch_id,))
    if batch is None:
        raise StoreError(f"batch 不存在: {batch_id}")
    if batch["status"] == "planned":
        # Path-based/draft evaluation remains useful for offline fixtures and
        # imported historical manifests; it does not mutate the queue.
        return "not_running"
    rows = controller.store.rows(
        "SELECT state FROM stages WHERE batch_id=? AND name='evaluate_absolute'",
        (batch_id,),
    )
    states = {str(row["state"]) for row in rows}
    if not rows:
        raise StoreError("批次没有 evaluate_absolute 步骤")
    if states <= {"succeeded"}:
        return "already_finalized"
    if not states <= {"ready", "succeeded"}:
        counts: dict[str, int] = {}
        for row in rows:
            state = str(row["state"])
            counts[state] = counts.get(state, 0) + 1
        raise StoreError(
            "GPTEval 只能在全部任务完成 unity_eval6 后运行；当前评测步骤状态: "
            + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )
    return "ready"


def _remote_profile_from_args(args: argparse.Namespace) -> RemoteProfile:
    missing = [
        label
        for label, value in (
            ("--remote-target / INSERTANY3D_REMOTE_TARGET", args.remote_target),
            ("--remote-project-root / INSERTANY3D_REMOTE_PROJECT_ROOT", args.remote_project_root),
            ("--remote-artifact-root / INSERTANY3D_REMOTE_ARTIFACT_ROOT", args.remote_artifact_root),
        )
        if not value
    ]
    if missing:
        raise StoreError("远端配置不完整: " + ", ".join(missing))
    return RemoteProfile(
        target=args.remote_target,
        port=args.remote_port,
        project_root=args.remote_project_root,
        artifact_root=args.remote_artifact_root,
        python_executable=args.remote_python,
        connect_timeout_seconds=args.remote_connect_timeout,
        control_timeout_seconds=args.remote_control_timeout,
        transfer_timeout_seconds=args.remote_transfer_timeout,
        poll_interval_seconds=args.remote_poll_interval,
        status_timeout_seconds=args.remote_status_timeout,
    )


def _commit_evaluation_stages(
    controller: BatchController,
    batch_id: str,
    output: Path,
    capacities: Mapping[str, int],
) -> int:
    """Publish one cached score artifact per task and close the durable DAG."""

    score_path = output / "task_scores.jsonl"
    try:
        rows = [
            json.loads(line)
            for line in score_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"无法读取 GPTEval task 汇总: {score_path}: {exc}") from exc
    by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("projectId") or ""), str(row.get("taskId") or ""))
        by_task.setdefault(key, []).append(row)
    batch_summary_path = output / "batch_summary.json"
    try:
        batch_summary_sha256 = hashlib.sha256(batch_summary_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvaluationError(f"无法读取 GPTEval batch 汇总: {batch_summary_path}: {exc}") from exc
    pending = controller.store.rows(
        """SELECT project_id, task_id FROM stages
             WHERE batch_id=? AND name='evaluate_absolute' AND state='ready'
             ORDER BY project_id, task_id""",
        (batch_id,),
    )
    committed = 0
    for pending_stage in pending:
        project_id = str(pending_stage["project_id"])
        task_id = str(pending_stage["task_id"])
        task_scores = by_task.get((project_id, task_id), [])
        if not task_scores or any(score.get("status") != "ready" for score in task_scores):
            raise EvaluationError(f"{project_id}/{task_id} 缺少 ready 的 GPTEval 结果")
        item = controller.lease_next(
            batch_id,
            f"evaluation-finalizer-{project_id}-{task_id}",
            capacities,
            project_id=project_id,
            task_id=task_id,
            stage_name="evaluate_absolute",
        )
        if item is None:
            raise StoreError(f"无法领取 {project_id}/{task_id} 的评测完成步骤")
        task_result = item.staging_dir / "gpteval_task_result.json"
        value = {
            "schemaVersion": 1,
            "kind": "insertany3d.gpteval-task-result",
            "batchId": batch_id,
            "projectId": project_id,
            "taskId": task_id,
            "scores": task_scores,
            "batchSummarySha256": batch_summary_sha256,
            "evaluationOutputLabel": output.name,
        }
        task_result.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        payload = task_result.read_bytes()
        controller.commit_success(
            item,
            [
                {
                    "artifactId": "gpteval_task_result",
                    "type": "gpteval_task_result",
                    "path": task_result.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            ],
        )
        committed += 1
    return committed


def _committed_evaluation_manifest_paths(
    controller: BatchController,
    batch_id: str,
) -> list[Path]:
    """Resolve only manifests published by successful unity_eval6 attempts."""

    batch = controller.store.row(
        "SELECT root_path FROM batches WHERE batch_id=?",
        (batch_id,),
    )
    if batch is None:
        raise StoreError(f"batch 不存在: {batch_id}")
    root = Path(str(batch["root_path"])).resolve()
    rows = controller.store.rows(
        """SELECT s.project_id, s.task_id, a.relative_path, a.sha256, a.size
             FROM artifacts a
             JOIN stages s ON s.id=a.stage_id
             JOIN attempts attempt ON attempt.id=a.attempt_id
            WHERE s.batch_id=? AND s.name='unity_eval6' AND s.state='succeeded'
              AND attempt.status='succeeded'
              AND replace(a.relative_path, '\\', '/') LIKE '%evaluation_manifest.json'
            ORDER BY s.project_id, s.task_id, a.id""",
        (batch_id,),
    )
    task_count = int(
        controller.store.row(
            "SELECT COUNT(*) AS count FROM tasks WHERE batch_id=?",
            (batch_id,),
        )["count"]
    )
    by_task: dict[tuple[str, str], Path] = {}
    for row in rows:
        identity = (str(row["project_id"]), str(row["task_id"]))
        if identity in by_task:
            raise EvaluationError(f"{identity[0]}/{identity[1]} 有多个已提交 evaluation_manifest.json")
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise EvaluationError(f"数据库中的评测清单路径不安全: {relative}")
        candidate = root / relative
        if candidate.is_symlink():
            raise EvaluationError(f"已提交的评测清单不能是符号链接: {relative}")
        path = candidate.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise EvaluationError(f"数据库中的评测清单越出批次目录: {relative}") from exc
        if not path.is_file():
            raise EvaluationError(f"已提交的评测清单不存在: {relative}")
        payload = path.read_bytes()
        if len(payload) != int(row["size"]) or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise EvaluationError(f"已提交的评测清单与数据库哈希不一致: {relative}")
        by_task[identity] = path
    if len(by_task) != task_count:
        raise EvaluationError(
            f"批次应有 {task_count} 个已提交 unity_eval6 清单，数据库实际发现 {len(by_task)} 个"
        )
    return [by_task[key] for key in sorted(by_task)]


def _fixed_cell(value: Any, width: int) -> str:
    text = str(value if value is not None else "-")
    if len(text) > width:
        text = text[: width - 1] + "~"
    return text.ljust(width)


def _run_evaluate(
    args: argparse.Namespace,
    *,
    evaluation_transport: Transport | None,
) -> Mapping[str, Any]:
    context = _prepare_evaluation(args)
    plan = _evaluation_plan(args, context)
    if args.command == "plan":
        return plan

    if args.command == "run":
        pending = pending_gpteval_requests(context.requests, context.cache)
        if pending and args.fake_score is not None:
            evaluator = _fixed_score_evaluator(args.fake_score)
            execution_mode = "fixed_fake_response"
        elif pending and args.allow_paid_api:
            try:
                api_key, key_source = load_apiyi_api_key()
            except ApiYiCredentialError as exc:
                raise EvaluationError(str(exc)) from exc
            evaluator = GPTEvalAPIClient(
                api_key=api_key,
                base_url=args.base_url,
                rubric=context.rubric,
                timeout_seconds=args.timeout,
                transport=evaluation_transport,
            )
            execution_mode = f"paid_api:{key_source}"
        elif pending:
            raise EvaluationError(
                "run 默认禁止付费请求；测试请显式使用 --fake-score，"
                "正式调用请显式使用 --allow-paid-api 并配置 APIYi key"
            )
        else:
            evaluator = _fixed_score_evaluator(7)
            execution_mode = "cache_only"
        progress = execute_gpteval_requests(
            context.requests,
            context.cache,
            evaluator,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
            limit=args.limit,
        )
        if progress["failed"]:
            error_paths = sorted(
                {
                    context.cache.error_path(request).resolve()
                    for request in context.requests
                    if context.cache.error_path(request).is_file()
                },
                key=str,
            )
            details = [
                f"{progress['failed']} 个 GPTEval 请求失败（执行模式: {execution_mode}）；"
                "重跑只会补缺失响应",
                "错误文件:",
            ]
            details.extend(str(path) for path in error_paths)
            if not error_paths:
                details.append("未找到已写入的错误 JSON，请检查输出目录是否可写")
            raise EvaluationError("\n".join(details))
        summary = _aggregate_evaluation(args, context)
        write_gpteval_summary(args.output, summary)
        return {
            "command": "run",
            "batchId": summary["batchId"],
            "executionMode": execution_mode,
            "progress": progress,
            "status": summary["status"],
            "dimensions": summary["dimensions"],
            "completion": summary["completion"],
            "outputs": {
                "batch": str((args.output / "batch_summary.json").resolve()),
                "tasks": str((args.output / "task_scores.jsonl").resolve()),
                "scenes": str((args.output / "scene_scores.csv").resolve()),
                "xlsx": str((args.output / "gpteval_summary.xlsx").resolve()),
            },
        }

    summary = _aggregate_evaluation(args, context)
    if args.command == "status":
        return {
            "command": "status",
            "status": summary["status"],
            "batchId": summary["batchId"],
            "metric": summary["metric"],
            "model": summary["model"],
            "dimensions": summary["dimensions"],
            "rubricSha256": summary["rubricSha256"],
            "completion": summary["completion"],
            "methods": {
                method_id: {
                    "status": method["status"],
                    "completion": method["completion"],
                }
                for method_id, method in summary["methods"].items()
            },
        }
    if args.command == "summarize":
        write_gpteval_summary(args.output, summary)
        return {
            "command": "summarize",
            "status": summary["status"],
            "dimensions": summary["dimensions"],
            "completion": summary["completion"],
            "outputs": {
                "batch": str((args.output / "batch_summary.json").resolve()),
                "tasks": str((args.output / "task_scores.jsonl").resolve()),
                "scenes": str((args.output / "scene_scores.csv").resolve()),
                "xlsx": str((args.output / "gpteval_summary.xlsx").resolve()),
            },
        }
    raise EvaluationError(f"未知 evaluate 命令: {args.command}")


def _fixed_score_evaluator(
    score: int,
) -> Callable[[GPTEvalRequest], Mapping[str, Any]]:
    def evaluate(request: GPTEvalRequest) -> Mapping[str, Any]:
        return fixed_fake_response(score, request.dimensions)

    return evaluate


def _prepare_evaluation(args: argparse.Namespace) -> _EvaluationContext:
    require_supported_evaluator(args.metric)
    for name in ("expected_tasks", "expected_scenes", "tasks_per_scene"):
        value = getattr(args, name)
        if value < 1:
            raise EvaluationError(f"--{name.replace('_', '-')} 必须是正整数")
    if args.command == "run" and args.fake_score is not None and not 1 <= args.fake_score <= 10:
        raise EvaluationError("--fake-score 必须是 1 到 10 的整数")
    manifest_paths = getattr(args, "manifest_paths", None)
    if manifest_paths is None:
        manifests = discover_evaluation_manifests(args.input)
    else:
        manifests = [load_evaluation_manifest(path) for path in manifest_paths]
        validate_manifest_collection(manifests)
    rubric, rubric_source = _load_rubric(args.rubric_file)
    dimensions = normalize_dimensions(args.dimensions)
    requests = plan_gpteval_requests(
        manifests,
        evaluator_version=args.evaluator_version,
        model=args.model,
        rubric_sha256_value=rubric_sha256(rubric),
        repeats=args.repeats,
        dimensions=dimensions,
    )
    collection = validate_manifest_collection(manifests)
    expected_batch_id = getattr(args, "expected_batch_id", None)
    if expected_batch_id is not None and collection["batchId"] != expected_batch_id:
        raise EvaluationError(
            f"评测清单 batchId={collection['batchId']!r} 与目标批次 {expected_batch_id!r} 不一致"
        )
    return _EvaluationContext(
        manifests=manifests,
        requests=requests,
        cache=ResponseCache(args.output / "responses"),
        rubric=rubric,
        rubric_source=rubric_source,
        collection=collection,
        dimensions=dimensions,
        comparison_config_sha256=evaluation_config_sha256(
            collection["comparisonConfigSha256"], dimensions
        ),
    )


def _evaluation_plan(
    args: argparse.Namespace, context: _EvaluationContext
) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for method_id in sorted({item.method_id for item in context.manifests}):
        records = [item for item in context.manifests if item.method_id == method_id]
        scenes = {(item.project_id, item.scene_path) for item in records}
        methods[method_id] = {
            "expectedTasks": args.expected_tasks,
            "discoveredTasks": len(records),
            "expectedScenes": args.expected_scenes,
            "discoveredScenes": len(scenes),
            "status": (
                "ready"
                if len(records) == args.expected_tasks and len(scenes) == args.expected_scenes
                else "partial"
            ),
        }
    cached = len(context.requests) - len(
        pending_gpteval_requests(context.requests, context.cache)
    )
    return {
        "command": "plan",
        "status": (
            "ready" if methods and all(item["status"] == "ready" for item in methods.values())
            else "partial"
        ),
        "batchId": context.collection["batchId"],
        "metric": "gpteval",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "comparisonConfigSha256": context.comparison_config_sha256,
        "dimensions": list(context.dimensions),
        "model": args.model,
        "evaluatorVersion": args.evaluator_version,
        "rubric": {
            "source": context.rubric_source,
            "sha256": rubric_sha256(context.rubric),
        },
        "repeats": args.repeats,
        "methods": methods,
        "requests": {
            "expected": args.expected_tasks * len(methods) * args.repeats,
            "planned": len(context.requests),
            "cached": cached,
            "pending": len(context.requests) - cached,
        },
        "network": {"allowed": False, "requestsSent": 0},
    }


def _aggregate_evaluation(
    args: argparse.Namespace, context: _EvaluationContext
) -> dict[str, Any]:
    return aggregate_gpteval(
        context.manifests,
        context.requests,
        context.cache,
        expected_tasks_per_method=args.expected_tasks,
        expected_scenes_per_method=args.expected_scenes,
        expected_tasks_per_scene=args.tasks_per_scene,
    )


def _load_rubric(path: Path | None) -> tuple[str, str]:
    if path is None:
        return DEFAULT_RUBRIC, "built-in:gpteval-eval6-v1"
    rubric = path.read_text(encoding="utf-8")
    if not rubric.strip():
        raise EvaluationError(f"评分规则文件为空: {path}")
    return rubric, str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
