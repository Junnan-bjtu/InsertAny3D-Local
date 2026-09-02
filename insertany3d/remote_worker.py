"""Durable SSH transport for one remote stage request.

The local :class:`~insertany3d.executors.CommandExecutor` can supervise this
module as an ordinary subprocess.  The expensive stage is detached on the
remote host and identified by a lease-derived attempt directory, so losing the
SSH connection does not imply that the remote process died.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import ContractError, validate_stage_request, validate_stage_result


REMOTE_STAGES = frozenset(
    {
        "upload_inputs",
        "model_generation",
        "render_alignment_views",
        "segment_inputs",
        "gim_match",
        "estimate_pose",
        "sags_segment_vote",
        "debug_bundle",
        "download_results",
    }
)
DEFAULT_REMOTE_ENVIRONMENT_FILE = ".insertany3d/runtime.env"
ALLOWED_RUNTIME_AUTHORITIES = frozenset({"codex_remote_tools", "server_checkout"})
_SAFE_TARGET = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_.-]*@)?(?:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?|\[[0-9A-Fa-f:]+\])$"
)
_SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
_KNOWN_REMOTE_STATES = frozenset(
    {"RESULT", "RUNNING", "GROUP_RUNNING", "STARTED", "EXITED", "MISSING", "IDENTITY_INVALID"}
)
_KNOWN_CLEANUP_STATES = frozenset(
    {"CLEANED", "GROUP_REMAINS", "DESCENDANTS_UNKNOWN", *_KNOWN_REMOTE_STATES}
)


class RemoteWorkerError(ValueError):
    """The remote profile, request, or downloaded output is unsafe/invalid."""


def verify_remote_runtime(
    profile: "RemoteProfile",
    *,
    lock_path: str | Path | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Read-only proof that the server runtime matches the committed lock.

    This preflight intentionally runs before the scheduler leases any work.  A
    missing adapter or stale server snapshot must be discovered before APIYi or
    a GPU-backed stage can incur cost.
    """

    default_repository_root = Path(__file__).resolve().parents[1]
    source = (
        Path(lock_path).resolve()
        if lock_path is not None
        else default_repository_root / "tools" / "remote_runtime.lock.json"
    )
    repository_root = source.parent.parent if lock_path is not None else default_repository_root
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RemoteWorkerError(f"远端运行时锁文件无法读取: {source}: {exc}") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schemaVersion") != 1
        or value.get("kind") != "insertany3d.remote-runtime-lock"
        or value.get("sourceAuthority") not in ALLOWED_RUNTIME_AUTHORITIES
        or not isinstance(value.get("files"), list)
    ):
        raise RemoteWorkerError("远端运行时锁文件版本或类型不受支持")

    expected: dict[str, tuple[str, int]] = {}
    for index, record in enumerate(value["files"]):
        if not isinstance(record, Mapping):
            raise RemoteWorkerError(f"远端运行时锁 files[{index}] 必须是对象")
        relative = record.get("path")
        digest = record.get("sha256")
        size = record.get("size")
        posix = PurePosixPath(str(relative))
        if (
            not isinstance(relative, str)
            or not relative
            or posix.is_absolute()
            or ".." in posix.parts
            or str(posix) != relative
            or not _SAFE_REMOTE_PATH.fullmatch("/" + relative)
        ):
            raise RemoteWorkerError(f"远端运行时锁包含非法路径: {relative!r}")
        if relative in expected:
            raise RemoteWorkerError(f"远端运行时锁包含重复路径: {relative}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RemoteWorkerError(f"远端运行时锁哈希无效: {relative}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RemoteWorkerError(f"远端运行时锁大小无效: {relative}")
        local_file = repository_root / "tools" / relative
        if not local_file.is_file() or local_file.is_symlink():
            raise RemoteWorkerError(f"公开运行时快照缺少普通文件: {relative}")
        if local_file.stat().st_size != size or _sha256_file(local_file) != digest:
            raise RemoteWorkerError(f"公开运行时快照与锁文件不一致: {relative}")
        expected[relative] = (digest, size)
    if "stage_adapter.py" not in expected:
        raise RemoteWorkerError("远端运行时锁缺少 stage_adapter.py")

    commands = RemoteCommandBuilder(profile)
    environment_file = str(
        PurePosixPath(profile.project_root, profile.environment_file)
    )
    quoted_environment_file = shlex.quote(environment_file)
    script_parts = [
        "set -u; ",
        f"config={quoted_environment_file}; ",
        "config_state=INVALID; ",
        "if test -f \"$config\" && ! test -L \"$config\"; then ",
        "set -a; . \"$config\"; set +a; ",
        "if case ${HF_HOME:-} in /*) test -d \"$HF_HOME\";; *) false;; esac ",
        "&& case ${TORCH_HOME:-} in /*) test -d \"$TORCH_HOME\";; *) false;; esac ",
        "&& case ${MODELSCOPE_CACHE:-} in /*) test -d \"$MODELSCOPE_CACHE\";; *) false;; esac ",
        "&& find -L \"$HF_HOME/hub/models--microsoft--TRELLIS-image-large/snapshots\" ",
        "-mindepth 2 -maxdepth 2 -type f -name pipeline.json -print -quit 2>/dev/null ",
        "| grep -q . ",
        "&& test -f \"$TORCH_HOME/hub/facebookresearch_dinov2_main/hubconf.py\"; ",
        "then config_state=OK; fi; fi; ",
        "printf 'CONFIG\\t%s\\n' \"$config_state\"; ",
    ]
    for relative in sorted(expected):
        remote = str(PurePosixPath(profile.project_root, "tools", relative))
        quoted_path = shlex.quote(remote)
        quoted_label = shlex.quote(relative)
        script_parts.append(
            f"if test -f {quoted_path}; then "
            f"digest=$(sha256sum -- {quoted_path} | awk '{{print $1}}'); "
            f"size=$(wc -c < {quoted_path}); "
            f"printf 'FILE\\t%s\\t%s\\t%s\\n' \"$digest\" \"$size\" {quoted_label}; "
            f"else printf 'MISSING\\t-\\t-\\t%s\\n' {quoted_label}; fi; "
        )
    runner = command_runner or SubprocessCommandRunner()
    outcome = runner.run(
        commands.ssh("".join(script_parts)),
        timeout_seconds=profile.control_timeout_seconds,
    )
    if outcome.returncode != 0 or outcome.timed_out:
        detail = outcome.stderr.strip() or outcome.stdout.strip() or f"exit code {outcome.returncode}"
        raise RemoteWorkerError(f"无法核对服务器运行时: {detail[:1000]}")

    config_states = [
        line.split("\t", 1)[1]
        for line in outcome.stdout.splitlines()
        if line.startswith("CONFIG\t") and "\t" in line
    ]
    if config_states != ["OK"]:
        raise RemoteWorkerError(
            "服务器私有运行配置缺失或无效；请检查 "
            f"{environment_file} 中的 HF_HOME、TORCH_HOME 和 MODELSCOPE_CACHE"
        )

    actual: dict[str, tuple[str, int] | None] = {}
    # SSH may start an interactive/login shell whose profile prints harmless
    # diagnostics (for example ``BASH=/usr/bin/bash``).  Only the deliberately
    # tab-separated FILE/MISSING records are part of this machine protocol;
    # ignore other stdout lines while keeping malformed protocol records
    # rejected below.
    protocol_lines = [
        line
        for line in outcome.stdout.splitlines()
        if line.startswith(("FILE\t", "MISSING\t"))
    ]
    for line in protocol_lines:
        fields = line.split("\t")
        if len(fields) != 4 or fields[0] not in {"FILE", "MISSING"}:
            raise RemoteWorkerError(f"服务器运行时核对输出无法解析: {line[:200]}")
        _kind, digest, size_text, relative = fields
        if relative not in expected or relative in actual:
            raise RemoteWorkerError(f"服务器运行时核对返回未知或重复路径: {relative}")
        if fields[0] == "MISSING":
            actual[relative] = None
            continue
        try:
            actual[relative] = (digest, int(size_text.strip()))
        except ValueError as exc:
            raise RemoteWorkerError(f"服务器运行时核对返回非法大小: {relative}") from exc
    problems = [
        relative
        for relative, identity in expected.items()
        if actual.get(relative) != identity
    ]
    if problems:
        sample = ", ".join(problems[:8])
        suffix = " ..." if len(problems) > 8 else ""
        raise RemoteWorkerError(f"服务器运行时缺失或与锁文件不一致: {sample}{suffix}")
    return {
        "schemaVersion": 1,
        "kind": "insertany3d.remote-runtime-verification",
        "files": len(expected),
        "environmentFile": profile.environment_file,
        "environment": "ready",
        "lockSha256": _sha256_file(source),
    }


@dataclass(frozen=True)
class RemoteProfile:
    """Connection and allowed-root settings for one remote host."""

    target: str
    project_root: str
    artifact_root: str
    port: int = 22
    ssh_executable: str = "ssh"
    scp_executable: str = "scp"
    python_executable: str = "third_party/TRELLIS/.venv/bin/python"
    environment_file: str = DEFAULT_REMOTE_ENVIRONMENT_FILE
    connect_timeout_seconds: float = 30.0
    control_timeout_seconds: float = 60.0
    transfer_timeout_seconds: float | None = None
    poll_interval_seconds: float = 5.0
    status_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not _SAFE_TARGET.fullmatch(self.target):
            raise RemoteWorkerError("remote target 必须是安全的 host 或 user@host")
        _remote_root(self.project_root, "project_root")
        _remote_root(self.artifact_root, "artifact_root")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise RemoteWorkerError("SSH 端口必须在 1..65535")
        for field, value in (
            ("ssh_executable", self.ssh_executable),
            ("scp_executable", self.scp_executable),
            ("python_executable", self.python_executable),
        ):
            if not isinstance(value, str) or not value.strip() or any(char in value for char in "\r\n\0"):
                raise RemoteWorkerError(f"{field} 必须是非空且不含控制字符的路径")
        environment_path = PurePosixPath(self.environment_file)
        if (
            environment_path.is_absolute()
            or ".." in environment_path.parts
            or str(environment_path) != self.environment_file
            or not _SAFE_REMOTE_PATH.fullmatch("/" + self.environment_file)
        ):
            raise RemoteWorkerError("environment_file 必须是项目内安全相对路径")
        for field, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("control_timeout_seconds", self.control_timeout_seconds),
            ("poll_interval_seconds", self.poll_interval_seconds),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise RemoteWorkerError(f"{field} 必须是正的有限数字")
        for field, value in (
            ("transfer_timeout_seconds", self.transfer_timeout_seconds),
            ("status_timeout_seconds", self.status_timeout_seconds),
        ):
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise RemoteWorkerError(f"{field} 必须是正的有限数字或留空")


@dataclass(frozen=True)
class CommandOutcome:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class CommandRunner(Protocol):
    def run(self, command: Sequence[str], *, timeout_seconds: float | None = None) -> CommandOutcome:
        ...


class SubprocessCommandRunner:
    """Default command runner; injectable so tests never need a real server."""

    def run(self, command: Sequence[str], *, timeout_seconds: float | None = None) -> CommandOutcome:
        try:
            completed = subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
            return CommandOutcome(completed.returncode, completed.stdout, completed.stderr)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return CommandOutcome(124, stdout, stderr, timed_out=True)
        except OSError as exc:
            return CommandOutcome(127, "", f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class RemoteAttemptPlan:
    request: Mapping[str, Any]
    local_artifact_root: Path
    local_request_path: Path
    local_output_dir: Path
    remote_request_path: str
    remote_output_dir: str
    remote_result_path: str
    remote_control_dir: str
    remote_pid_path: str
    remote_identity_path: str
    owner_id: str


@dataclass(frozen=True)
class RemoteRunReport:
    classification: str
    result_path: Path
    recovery_command: tuple[str, ...]
    remote_state: str | None = None


def _identity_shell(plan: RemoteAttemptPlan) -> str:
    """Define remote helpers that fence PID reuse and inspect the whole group."""

    identity = shlex.quote(plan.remote_identity_path)
    owner = shlex.quote(plan.owner_id)
    return (
        "load_identity() { "
        f"test -f {identity} || return 2; "
        f"saved_pid=$(sed -n '1p' {identity}) || return 2; "
        f"saved_pgid=$(sed -n '2p' {identity}) || return 2; "
        f"saved_boot=$(sed -n '3p' {identity}) || return 2; "
        f"saved_ticks=$(sed -n '4p' {identity}) || return 2; "
        f"saved_owner=$(sed -n '5p' {identity}) || return 2; "
        "case \"$saved_pid\" in (''|*[!0-9]*) return 2;; esac; "
        "case \"$saved_pgid\" in (''|*[!0-9]*) return 2;; esac; "
        "case \"$saved_ticks\" in (''|*[!0-9]*) return 2;; esac; "
        f"test \"$saved_owner\" = {owner} || return 2; return 0; }}; "
        "group_live() { ps -eo pgid=,stat= 2>/dev/null | "
        "awk -v group=\"$saved_pgid\" '$1 == group && $2 !~ /^Z/ { found=1 } END { exit found ? 0 : 1 }'; }; "
        "leader_identity() { "
        "kill -0 \"$saved_pid\" 2>/dev/null || return 1; "
        "current_ticks=$(awk '{print $22}' \"/proc/$saved_pid/stat\" 2>/dev/null) || return 1; "
        "test \"$current_ticks\" = \"$saved_ticks\" || return 2; "
        "current_pgid=$(ps -o pgid= -p \"$saved_pid\" 2>/dev/null | tr -d ' ') || return 1; "
        "test \"$current_pgid\" = \"$saved_pgid\" || return 2; return 0; }; "
        "probe_state() { "
        "if ! load_identity; then printf 'IDENTITY_INVALID'; return; fi; "
        "current_boot=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null) || { printf 'IDENTITY_INVALID'; return; }; "
        "if test \"$current_boot\" != \"$saved_boot\"; then printf 'EXITED'; return; fi; "
        "if leader_identity; then printf 'RUNNING'; return; else leader_status=$?; fi; "
        "if test \"$leader_status\" -eq 2; then printf 'IDENTITY_INVALID'; "
        "elif group_live; then printf 'GROUP_RUNNING'; else printf 'EXITED'; fi; }; "
    )


class RemoteCommandBuilder:
    """Build argv arrays and quoted remote shell snippets without executing them."""

    def __init__(self, profile: RemoteProfile):
        self.profile = profile

    def ssh(self, script: str) -> list[str]:
        # OpenSSH joins its trailing argv into a remote shell command.  Quote
        # the -c payload so function definitions, command substitutions, and
        # semicolons remain inside one bash invocation instead of being parsed
        # by the login shell first.
        return [
            self.profile.ssh_executable,
            "-p",
            str(self.profile.port),
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.profile.connect_timeout_seconds:g}",
            "--",
            self.profile.target,
            "bash",
            "-lc",
            shlex.quote(script),
        ]

    def scp_upload(self, local_path: Path, remote_path: str) -> list[str]:
        _remote_absolute_path(remote_path, "scp upload path")
        return [
            self.profile.scp_executable,
            "-P",
            str(self.profile.port),
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.profile.connect_timeout_seconds:g}",
            "--",
            str(local_path),
            f"{self.profile.target}:{remote_path}",
        ]

    def scp_download_tree(self, remote_path: str, local_path: Path) -> list[str]:
        _remote_absolute_path(remote_path, "scp download path")
        return [
            self.profile.scp_executable,
            "-r",
            "-P",
            str(self.profile.port),
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.profile.connect_timeout_seconds:g}",
            "--",
            f"{self.profile.target}:{remote_path}",
            str(local_path),
        ]

    def prepare(self, plan: RemoteAttemptPlan) -> list[str]:
        directories = {
            plan.remote_control_dir,
            str(PurePosixPath(plan.remote_request_path).parent),
            plan.remote_output_dir,
            *(
                str(PurePosixPath(self.profile.artifact_root, str(item["path"])).parent)
                for item in plan.request["inputs"]
            ),
        }
        script = "set -eu; mkdir -p -- " + " ".join(shlex.quote(path) for path in sorted(directories))
        return self.ssh(script)

    def finalize_upload(self, temporary: str, final: str, expected_sha256: str) -> list[str]:
        _remote_absolute_path(temporary, "temporary upload path")
        _remote_absolute_path(final, "final upload path")
        script = (
            "set -eu; "
            f"actual=$(sha256sum -- {shlex.quote(temporary)} | awk '{{print $1}}'); "
            f"test \"$actual\" = {shlex.quote(expected_sha256)}; "
            f"mv -f -- {shlex.quote(temporary)} {shlex.quote(final)}"
        )
        return self.ssh(script)

    def probe(self, plan: RemoteAttemptPlan) -> list[str]:
        result = shlex.quote(plan.remote_result_path)
        pid = shlex.quote(plan.remote_pid_path)
        identity = shlex.quote(plan.remote_identity_path)
        lock = shlex.quote(str(PurePosixPath(plan.remote_control_dir, "launch.lock")))
        script = "".join(
            (
                _identity_shell(plan),
                f"if test -f {result}; then printf 'RESULT\n'; "
                f"elif test -f {identity} || test -f {pid}; then probe_state; printf '\n'; "
                f"elif test -d {lock}; then printf 'IDENTITY_INVALID\n'; "
                "else printf 'MISSING\n'; fi",
            )
        )
        return self.ssh(script)

    def start_or_probe(self, plan: RemoteAttemptPlan) -> list[str]:
        result = shlex.quote(plan.remote_result_path)
        pid = shlex.quote(plan.remote_pid_path)
        identity = shlex.quote(plan.remote_identity_path)
        owner = shlex.quote(plan.owner_id)
        lock = shlex.quote(str(PurePosixPath(plan.remote_control_dir, "launch.lock")))
        stdout = shlex.quote(str(PurePosixPath(plan.remote_control_dir, "stdout.log")))
        stderr = shlex.quote(str(PurePosixPath(plan.remote_control_dir, "stderr.log")))
        project = shlex.quote(self.profile.project_root)
        python = self.profile.python_executable
        if not PurePosixPath(python).is_absolute():
            python = str(PurePosixPath(self.profile.project_root, python))
        adapter = str(PurePosixPath(self.profile.project_root, "tools", "stage_adapter.py"))
        environment_file = shlex.quote(
            str(PurePosixPath(self.profile.project_root, self.profile.environment_file))
        )
        invocation = " ".join(
            shlex.quote(value)
            for value in (
                python,
                adapter,
                "--request",
                plan.remote_request_path,
                "--artifact-root",
                self.profile.artifact_root,
                "--result",
                plan.remote_result_path,
            )
        )
        script = "".join(
            (
                "set -u; ",
                _identity_shell(plan),
                f"if test -f {result}; then printf 'RESULT\n'; "
                f"elif test -f {identity} || test -f {pid}; then probe_state; printf '\n'; "
                f"elif mkdir -- {lock} 2>/dev/null; then "
                "if ! command -v setsid >/dev/null 2>&1; then printf 'IDENTITY_INVALID\n'; exit 0; fi; "
                f"if ! test -f {environment_file} || test -L {environment_file}; "
                "then printf 'IDENTITY_INVALID\n'; exit 0; fi; "
                f"set -a; . {environment_file}; set +a; "
                f"cd -- {project}; nohup setsid -- {invocation} >{stdout} 2>{stderr} </dev/null & child=$!; "
                "pgid=$child; boot=$(cat /proc/sys/kernel/random/boot_id); "
                "ticks=$(awk '{print $22}' \"/proc/$child/stat\"); "
                f"printf '%s\n' \"$child\" > {pid}.tmp; mv -f -- {pid}.tmp {pid}; "
                f"printf '%s\n%s\n%s\n%s\n%s\n' \"$child\" \"$pgid\" \"$boot\" \"$ticks\" {owner} > {identity}.tmp; "
                f"mv -f -- {identity}.tmp {identity}; printf 'STARTED %s %s\n' \"$child\" \"$pgid\"; "
                "else printf 'IDENTITY_INVALID\n'; fi",
            )
        )
        return self.ssh(script)

    def terminate_group(self, plan: RemoteAttemptPlan) -> list[str]:
        """Terminate only a group whose leader identity still matches."""

        result = shlex.quote(plan.remote_result_path)
        script = "".join(
            (
                "set -u; ",
                _identity_shell(plan),
                f"if test -f {result}; then printf 'RESULT\n'; exit 0; fi; ",
                "state=$(probe_state); "
                "if test \"$state\" != RUNNING; then printf '%s\n' \"$state\"; exit 0; fi; ",
                "kill -TERM -- \"-$saved_pgid\" 2>/dev/null || true; "
                "count=0; while group_live && test \"$count\" -lt 100; do sleep 0.1; count=$((count + 1)); done; "
                "if group_live; then kill -KILL -- \"-$saved_pgid\" 2>/dev/null || true; fi; "
                "count=0; while group_live && test \"$count\" -lt 100; do sleep 0.1; count=$((count + 1)); done; "
                f"if group_live; then printf 'GROUP_REMAINS\n'; "
                f"elif test -f {result}; then printf 'RESULT\n'; "
                "else printf 'DESCENDANTS_UNKNOWN\n'; fi",
            )
        )
        return self.ssh(script)


class RemoteStageRunner:
    """Upload, start/recover, poll, and download one remote stage attempt."""

    def __init__(
        self,
        profile: RemoteProfile,
        *,
        command_runner: CommandRunner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.profile = profile
        self.commands = RemoteCommandBuilder(profile)
        self.command_runner = command_runner or SubprocessCommandRunner()
        self.sleep = sleep
        self.monotonic = monotonic

    def run(self, request_path: str | Path, artifact_root: str | Path) -> RemoteRunReport:
        plan = build_remote_attempt_plan(self.profile, request_path, artifact_root)
        recovery = tuple(self.commands.probe(plan))
        plan.local_output_dir.mkdir(parents=True, exist_ok=True)
        if any(plan.local_output_dir.iterdir()):
            raise RemoteWorkerError(f"本地 outputStagingDir 必须为空: {plan.local_output_dir}")
        self._verify_local_inputs(plan)

        if plan.request["stage"] == "download_results":
            # Every expensive remote stage is downloaded and hash-checked before
            # it can be committed.  This final boundary records that the complete
            # chain is present locally; it deliberately performs no second copy
            # of the same PLY/debug bundle.
            return self._write_transfer_receipt(plan, "per_stage_eager_download", recovery)

        prepare = self._execute_control(self.commands.prepare(plan))
        if prepare.returncode != 0:
            return self._failure(plan, "transient_network", "failed_retryable", prepare, recovery)

        uploads = [(plan.local_request_path, plan.remote_request_path)]
        uploads.extend(
            (plan.local_artifact_root / str(item["path"]), str(PurePosixPath(self.profile.artifact_root, str(item["path"]))))
            for item in plan.request["inputs"]
        )
        for local, remote in uploads:
            digest = _sha256_file(local)
            temporary = f"{remote}.upload-{plan.owner_id}.tmp"
            uploaded = self._execute_transfer(self.commands.scp_upload(local, temporary))
            if uploaded.returncode != 0:
                return self._failure(plan, "transient_network", "failed_retryable", uploaded, recovery)
            finalized = self._execute_control(self.commands.finalize_upload(temporary, remote, digest))
            if finalized.returncode != 0:
                return self._failure(plan, "transient_network", "failed_retryable", finalized, recovery)

        if plan.request["stage"] == "upload_inputs":
            return self._write_transfer_receipt(plan, "atomic_scp_upload", recovery)

        started = self._execute_control(self.commands.start_or_probe(plan))
        if started.returncode != 0:
            return self._unknown(plan, "delivery_unknown", started, recovery)
        state = _parse_remote_state(started.stdout)
        if state is None:
            return self._unknown(plan, "delivery_unknown", started, recovery)

        began_waiting = self.monotonic()
        while state in {"STARTED", "RUNNING", "MISSING"}:
            if state == "MISSING":
                return self._unknown(plan, "remote_status_unknown", started, recovery)
            if (
                self.profile.status_timeout_seconds is not None
                and self.monotonic() - began_waiting >= self.profile.status_timeout_seconds
            ):
                timeout = CommandOutcome(124, "", "remote status polling timed out", timed_out=True)
                return self._unknown(plan, "remote_status_unknown", timeout, recovery)
            self.sleep(self.profile.poll_interval_seconds)
            probed = self._execute_control(self.commands.probe(plan))
            if probed.returncode != 0:
                return self._unknown(plan, "remote_status_unknown", probed, recovery)
            parsed = _parse_remote_state(probed.stdout)
            if parsed is None:
                return self._unknown(plan, "remote_status_unknown", probed, recovery)
            state = parsed

        if state == "EXITED":
            return self._unknown(
                plan,
                "remote_status_unknown",
                CommandOutcome(
                    1,
                    state,
                    "远端顶层 PID 已退出但没有 stage_result.json；尚未证明其进程组/GPU 子进程已清理",
                ),
                recovery,
            )
        if state != "RESULT":
            return self._unknown(plan, "remote_status_unknown", started, recovery)
        return self._download(plan, recovery)

    def probe_existing(self, request_path: str | Path, artifact_root: str | Path) -> RemoteRunReport:
        """Read the state of an existing attempt without writing local or remote state."""

        plan = build_remote_attempt_plan(self.profile, request_path, artifact_root)
        self._verify_local_inputs(plan)
        recovery = tuple(self.commands.probe(plan))
        outcome = self._execute_control(recovery)
        state = _parse_remote_state(outcome.stdout) if outcome.returncode == 0 else None
        classification = "remote_probe" if state is not None else "remote_status_unknown"
        return RemoteRunReport(classification, plan.local_output_dir / "stage_result.json", recovery, state)

    def download_existing_result(self, request_path: str | Path, artifact_root: str | Path) -> RemoteRunReport:
        """Download a completed existing attempt; never starts or uploads work."""

        plan = build_remote_attempt_plan(self.profile, request_path, artifact_root)
        self._verify_local_inputs(plan)
        recovery = tuple(self.commands.probe(plan))
        probed = self._execute_control(recovery)
        state = _parse_remote_state(probed.stdout) if probed.returncode == 0 else None
        if state != "RESULT":
            return RemoteRunReport(
                "remote_status_unknown" if state is None else "remote_not_complete",
                plan.local_output_dir / "stage_result.json",
                recovery,
                state,
            )
        return self._download(plan, recovery)

    def cancel_existing(self, request_path: str | Path, artifact_root: str | Path) -> RemoteRunReport:
        """Explicitly terminate a verified remote leader and its whole process group."""

        plan = build_remote_attempt_plan(self.profile, request_path, artifact_root)
        self._verify_local_inputs(plan)
        recovery = tuple(self.commands.probe(plan))
        outcome = self._execute_control(self.commands.terminate_group(plan))
        state = _parse_cleanup_state(outcome.stdout) if outcome.returncode == 0 else None
        classification = "remote_cleanup" if state == "CLEANED" else "remote_cleanup_incomplete"
        return RemoteRunReport(
            classification,
            plan.local_output_dir / "stage_result.json",
            recovery,
            state,
        )

    def _write_transfer_receipt(
        self,
        plan: RemoteAttemptPlan,
        transfer_mode: str,
        recovery: tuple[str, ...],
    ) -> RemoteRunReport:
        receipt_path = plan.local_output_dir / "transfer_receipt.json"
        _atomic_json(
            receipt_path,
            {
                "schemaVersion": 1,
                "batchId": plan.request["batchId"],
                "projectId": plan.request["projectId"],
                "taskId": plan.request["taskId"],
                "stage": plan.request["stage"],
                "attempt": plan.request["attempt"],
                "transferMode": transfer_mode,
                "verifiedInputs": [
                    {
                        "artifactId": item["artifactId"],
                        "sha256": item["sha256"],
                        "size": item.get("size"),
                    }
                    for item in plan.request["inputs"]
                ],
                "finishedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
        artifact = {
            "artifactId": "transfer_receipt",
            "type": "transfer_receipt",
            "path": receipt_path.name,
            "sha256": _sha256_file(receipt_path),
            "size": receipt_path.stat().st_size,
        }
        result = {
            "schemaVersion": 1,
            "kind": "insertany3d.stage-result",
            "batchId": plan.request["batchId"],
            "projectId": plan.request["projectId"],
            "taskId": plan.request["taskId"],
            "stage": plan.request["stage"],
            "contractVersion": plan.request["contractVersion"],
            "attempt": plan.request["attempt"],
            "leaseToken": plan.request["leaseToken"],
            "status": "succeeded",
            "artifacts": [artifact],
            "errorCode": None,
            "message": "",
            "diagnosticPaths": [],
            "cleanup": {"completed": True},
            "finishedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        result_path = plan.local_output_dir / "stage_result.json"
        _atomic_json(result_path, validate_stage_result(result))
        return RemoteRunReport("succeeded", result_path, recovery, "RESULT")

    def _execute_control(self, command: Sequence[str]) -> CommandOutcome:
        return self.command_runner.run(command, timeout_seconds=self.profile.control_timeout_seconds)

    def _execute_transfer(self, command: Sequence[str]) -> CommandOutcome:
        # OpenSSH ConnectTimeout remains in argv for connection setup.  Large
        # image/PLY transfers have no total timeout unless explicitly set.
        return self.command_runner.run(command, timeout_seconds=self.profile.transfer_timeout_seconds)

    def _verify_local_inputs(self, plan: RemoteAttemptPlan) -> None:
        for item in plan.request["inputs"]:
            path = _local_relative(plan.local_artifact_root, str(item["path"]), "input path")
            if not path.is_file():
                raise RemoteWorkerError(f"输入 artifact 不存在: {item['path']}")
            actual = _sha256_file(path)
            if actual != item["sha256"]:
                raise RemoteWorkerError(f"输入 artifact 哈希不一致: {item['artifactId']}")

    def _download(self, plan: RemoteAttemptPlan, recovery: tuple[str, ...]) -> RemoteRunReport:
        download = plan.local_output_dir.parent / f"{plan.local_output_dir.name}.remote-download-{plan.owner_id}"
        if download.exists():
            raise RemoteWorkerError(f"远端下载暂存目录已存在，请先人工检查: {download}")
        outcome = self._execute_transfer(self.commands.scp_download_tree(plan.remote_output_dir, download))
        if outcome.returncode != 0:
            return self._unknown(plan, "remote_status_unknown", outcome, recovery)
        try:
            self._validate_download(plan, download)
            children = list(download.iterdir())
            # The result manifest is the completion marker consumed by
            # CommandExecutor, so publish it only after every other file.
            children.sort(key=lambda child: child.name == "stage_result.json")
            for child in children:
                child.replace(plan.local_output_dir / child.name)
            download.rmdir()
        except (ContractError, json.JSONDecodeError, OSError, RemoteWorkerError) as exc:
            quarantine = plan.local_output_dir.parent / f"{download.name}.invalid"
            if download.exists() and not quarantine.exists():
                download.replace(quarantine)
            return self._failure(
                plan,
                "remote_contract_invalid",
                "failed_terminal",
                CommandOutcome(2, "", str(exc)),
                recovery,
            )
        return RemoteRunReport("result_downloaded", plan.local_output_dir / "stage_result.json", recovery, "RESULT")

    def _validate_download(self, plan: RemoteAttemptPlan, download: Path) -> None:
        if not download.is_dir():
            raise RemoteWorkerError("SCP 未生成预期的下载目录")
        for path in download.rglob("*"):
            if path.is_symlink():
                raise RemoteWorkerError(f"远端输出不能包含符号链接: {path.relative_to(download)}")
            if not path.is_file() and not path.is_dir():
                raise RemoteWorkerError(f"远端输出包含不支持的文件类型: {path.relative_to(download)}")
        result_path = download / "stage_result.json"
        if not result_path.is_file():
            raise RemoteWorkerError("远端输出缺少 stage_result.json")
        result = validate_stage_result(json.loads(result_path.read_text(encoding="utf-8")))
        expected = {
            "batchId": plan.request["batchId"],
            "projectId": plan.request["projectId"],
            "taskId": plan.request["taskId"],
            "stage": plan.request["stage"],
            "contractVersion": plan.request["contractVersion"],
            "attempt": plan.request["attempt"],
            "leaseToken": plan.request["leaseToken"],
        }
        mismatches = [key for key, value in expected.items() if result.get(key) != value]
        if mismatches:
            raise RemoteWorkerError("远端 stage_result 身份不匹配: " + ", ".join(mismatches))
        for artifact in result["artifacts"]:
            path = _local_relative(download, str(artifact["path"]), "downloaded artifact path")
            if not path.is_file():
                raise RemoteWorkerError(f"远端结果声明的 artifact 不存在: {artifact['path']}")
            if path.stat().st_size != int(artifact["size"]):
                raise RemoteWorkerError(f"远端结果 artifact 大小不匹配: {artifact['artifactId']}")
            if _sha256_file(path) != artifact["sha256"]:
                raise RemoteWorkerError(f"远端结果 artifact 哈希不匹配: {artifact['artifactId']}")

    def _unknown(
        self,
        plan: RemoteAttemptPlan,
        classification: str,
        outcome: CommandOutcome,
        recovery: tuple[str, ...],
    ) -> RemoteRunReport:
        # cleanup=false keeps the scheduler in recovery with its resource
        # fenced.  The delivery_unknown code plus the finer diagnostic
        # classification tells a recovery command not to launch a duplicate.
        message = (
            f"{classification}: SSH 无法确认远端阶段状态；不得自动重启。"
            f"恢复检查命令: {shlex.join(recovery)}"
        )
        if outcome.stderr.strip():
            message += f"；SSH: {outcome.stderr.strip()[:500]}"
        return self._write_failure_result(
            plan,
            result_status="failed_retryable",
            error_code="delivery_unknown",
            message=message,
            classification=classification,
            recovery=recovery,
            outcome=outcome,
            cleanup_completed=False,
        )

    def _failure(
        self,
        plan: RemoteAttemptPlan,
        error_code: str,
        result_status: str,
        outcome: CommandOutcome,
        recovery: tuple[str, ...],
    ) -> RemoteRunReport:
        detail = outcome.stderr.strip() or outcome.stdout.strip() or f"exit code {outcome.returncode}"
        return self._write_failure_result(
            plan,
            result_status=result_status,
            error_code=error_code,
            message=detail[:1000],
            classification=error_code,
            recovery=recovery,
            outcome=outcome,
        )

    def _write_failure_result(
        self,
        plan: RemoteAttemptPlan,
        *,
        result_status: str,
        error_code: str,
        message: str,
        classification: str,
        recovery: tuple[str, ...],
        outcome: CommandOutcome,
        cleanup_completed: bool = True,
    ) -> RemoteRunReport:
        diagnostics = plan.local_output_dir / "_remote" / "transport.json"
        _atomic_json(
            diagnostics,
            {
                "schemaVersion": 1,
                "classification": classification,
                "remoteStateKnown": False if classification in {"delivery_unknown", "remote_status_unknown"} else None,
                "remoteControlDir": plan.remote_control_dir,
                "returnCode": outcome.returncode,
                "timedOut": outcome.timed_out,
                "stderr": outcome.stderr[-4000:],
                "recoveryCommand": list(recovery),
            },
        )
        result = {
            "schemaVersion": 1,
            "kind": "insertany3d.stage-result",
            "batchId": plan.request["batchId"],
            "projectId": plan.request["projectId"],
            "taskId": plan.request["taskId"],
            "stage": plan.request["stage"],
            "contractVersion": plan.request["contractVersion"],
            "attempt": plan.request["attempt"],
            "leaseToken": plan.request["leaseToken"],
            "status": result_status,
            "artifacts": [],
            "errorCode": error_code,
            "message": message,
            "diagnosticPaths": ["_remote/transport.json"],
            # False keeps the scheduler lease/resource fenced while a remote
            # process might still be alive.  A later explicit probe/recovery
            # must settle it before that GPU slot can be reused.
            "cleanup": {"completed": cleanup_completed},
            "finishedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        validated = validate_stage_result(result)
        result_path = plan.local_output_dir / "stage_result.json"
        _atomic_json(result_path, validated)
        return RemoteRunReport(classification, result_path, recovery)


def build_remote_attempt_plan(
    profile: RemoteProfile,
    request_path: str | Path,
    artifact_root: str | Path,
) -> RemoteAttemptPlan:
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise RemoteWorkerError(f"本地 artifact root 不存在: {root}")
    local_request = Path(request_path).resolve()
    if not local_request.is_file():
        raise RemoteWorkerError(f"stage request 不存在: {local_request}")
    try:
        request = validate_stage_request(json.loads(local_request.read_text(encoding="utf-8")))
    except (ContractError, json.JSONDecodeError, OSError) as exc:
        raise RemoteWorkerError(f"stage request 无效: {exc}") from exc
    if request["stage"] not in REMOTE_STAGES:
        raise RemoteWorkerError(f"{request['stage']} 不是 SSH 远端执行阶段")
    output = _local_relative(root, str(request["outputStagingDir"]), "outputStagingDir")
    output_relative = PurePosixPath(str(request["outputStagingDir"]).replace("\\", "/"))
    remote_output = str(PurePosixPath(profile.artifact_root, output_relative))
    owner_id = hashlib.sha256(str(request["leaseToken"]).encode("utf-8")).hexdigest()[:16]
    control = str(
        PurePosixPath(
            profile.artifact_root,
            ".insertany3d_remote",
            "attempts",
            str(request["batchId"]),
            str(request["projectId"]),
            str(request["taskId"]),
            str(request["stage"]),
            f"attempt-{int(request['attempt']):04d}-{owner_id}",
        )
    )
    remote_request = str(PurePosixPath(control, "request.json"))
    for field, path in (
        ("remote output", remote_output),
        ("remote request", remote_request),
        ("remote control", control),
    ):
        _remote_absolute_path(path, field)
    for item in request["inputs"]:
        _remote_absolute_path(
            str(PurePosixPath(profile.artifact_root, str(item["path"]))),
            f"remote input {item['artifactId']}",
        )
    return RemoteAttemptPlan(
        request=request,
        local_artifact_root=root,
        local_request_path=local_request,
        local_output_dir=output,
        remote_request_path=remote_request,
        remote_output_dir=remote_output,
        remote_result_path=str(PurePosixPath(remote_output, "stage_result.json")),
        remote_control_dir=control,
        remote_pid_path=str(PurePosixPath(control, "pid")),
        remote_identity_path=str(PurePosixPath(control, "process.identity")),
        owner_id=owner_id,
    )


def _remote_root(value: str, field: str) -> str:
    path = _remote_absolute_path(value, field)
    if path == "/":
        raise RemoteWorkerError(f"{field} 禁止使用远端文件系统根目录 /")
    return path


def _remote_absolute_path(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_REMOTE_PATH.fullmatch(value):
        raise RemoteWorkerError(f"{field} 必须是只含安全字符的远端 POSIX 绝对路径")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value.rstrip("/"):
        raise RemoteWorkerError(f"{field} 必须是规范化的远端 POSIX 绝对路径")
    return str(path)


def _local_relative(root: Path, value: str, field: str) -> Path:
    text = value.replace("\\", "/")
    relative = Path(text)
    windows = PureWindowsPath(text)
    if not text or relative.is_absolute() or windows.is_absolute() or windows.drive or ".." in relative.parts:
        raise RemoteWorkerError(f"{field} 必须是安全相对路径")
    unresolved = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RemoteWorkerError(f"{field} 不能包含符号链接: {value}")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RemoteWorkerError(f"{field} 越出 artifact root") from exc
    return resolved


def _parse_remote_state(stdout: str) -> str | None:
    states = []
    for line in stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if fields and fields[0] in _KNOWN_REMOTE_STATES:
            states.append(fields[0])
    return states[0] if len(states) == 1 else None


def _parse_cleanup_state(stdout: str) -> str | None:
    states = []
    for line in stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if fields and fields[0] in _KNOWN_CLEANUP_STATES:
            states.append(fields[0])
    return states[0] if len(states) == 1 else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _profile_from_args(args: argparse.Namespace) -> RemoteProfile:
    return RemoteProfile(
        target=args.remote_target,
        port=args.port,
        project_root=args.remote_project_root,
        artifact_root=args.remote_artifact_root,
        ssh_executable=args.ssh_executable,
        scp_executable=args.scp_executable,
        python_executable=args.remote_python,
        environment_file=args.remote_environment_file,
        connect_timeout_seconds=args.connect_timeout_seconds,
        control_timeout_seconds=args.control_timeout_seconds,
        transfer_timeout_seconds=args.transfer_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        status_timeout_seconds=args.status_timeout_seconds,
    )


def build_remote_worker_command(
    request_path: str | Path,
    artifact_root: str | Path,
    profile: RemoteProfile,
    *,
    local_python: str = sys.executable,
) -> list[str]:
    """Build the local wrapper command accepted by ``CommandExecutor``."""

    command = [
        local_python,
        "-m",
        "insertany3d.remote_worker",
        "--request",
        str(request_path),
        "--artifact-root",
        str(artifact_root),
        "--remote-target",
        profile.target,
        "--port",
        str(profile.port),
        "--remote-project-root",
        profile.project_root,
        "--remote-artifact-root",
        profile.artifact_root,
        "--ssh-executable",
        profile.ssh_executable,
        "--scp-executable",
        profile.scp_executable,
        "--remote-python",
        profile.python_executable,
        "--remote-environment-file",
        profile.environment_file,
        "--connect-timeout-seconds",
        f"{profile.connect_timeout_seconds:g}",
        "--control-timeout-seconds",
        f"{profile.control_timeout_seconds:g}",
        "--poll-interval-seconds",
        f"{profile.poll_interval_seconds:g}",
    ]
    if profile.transfer_timeout_seconds is not None:
        command.extend(["--transfer-timeout-seconds", f"{profile.transfer_timeout_seconds:g}"])
    if profile.status_timeout_seconds is not None:
        command.extend(["--status-timeout-seconds", f"{profile.status_timeout_seconds:g}"])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过 SSH 执行一个可恢复的 InsertAny3D 远端阶段")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--remote-target", required=True, help="host 或 user@host")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--remote-project-root", required=True)
    parser.add_argument("--remote-artifact-root", required=True)
    parser.add_argument("--ssh-executable", default="ssh")
    parser.add_argument("--scp-executable", default="scp")
    parser.add_argument("--remote-python", default="third_party/TRELLIS/.venv/bin/python")
    parser.add_argument(
        "--remote-environment-file",
        default=DEFAULT_REMOTE_ENVIRONMENT_FILE,
        help="相对服务器项目根目录的私有运行环境配置",
    )
    parser.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--control-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--transfer-timeout-seconds", type=float)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--status-timeout-seconds", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = RemoteStageRunner(_profile_from_args(args)).run(args.request, args.artifact_root)
        print(
            json.dumps(
                {
                    "classification": report.classification,
                    "resultPath": str(report.result_path),
                    "remoteState": report.remote_state,
                    "recoveryCommand": list(report.recovery_command),
                },
                ensure_ascii=False,
            )
        )
        # Operational failures are represented by a valid stage_result.  Zero
        # lets CommandExecutor use that contract instead of replacing it with
        # the wrapper process exit code.
        return 0
    except (RemoteWorkerError, OSError) as exc:
        print(f"REMOTE_WORKER_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
