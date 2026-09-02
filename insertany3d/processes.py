"""Process identity, diagnostics, and process-group cleanup primitives."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import signal
import subprocess
import time
from ctypes import Structure, byref, c_size_t, c_ulong, c_ulonglong, c_void_p, sizeof
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgid: int
    host_boot_id: str
    process_start_ticks: int


@dataclass(frozen=True)
class CleanupResult:
    completed: bool
    identity_matched: bool
    remaining_pids: tuple[int, ...]
    diagnostic_status: str


class WindowsJobRunnerError(RuntimeError):
    """A Windows process tree cannot be launched under a kill-on-close job."""


class WindowsJobCommandBuilder(Protocol):
    """Replace a WSL-to-Windows Unity command with a safe wrapper command."""

    def wrap(self, command: list[str], *, cwd: Path) -> list[str]: ...


class WindowsJobRunner:
    """Build a PowerShell Job Object wrapper for WSL-launched ``Unity.exe``.

    WSL cannot create a Windows Job Object through ctypes.  A Windows
    PowerShell process therefore owns the job handle and launches Unity
    suspended before assigning it.  Killing that wrapper closes the handle and
    terminates the complete Windows process tree.
    """

    def __init__(
        self,
        *,
        platform_name: str | None = None,
        powershell_executable: str | None = None,
        script_path: str | Path | None = None,
        which: Callable[[str], str | None] = shutil.which,
    ):
        self.platform_name = platform_name or os.name
        self.powershell_executable = powershell_executable
        self.script_path = (
            Path(script_path)
            if script_path is not None
            else Path(__file__).resolve().parents[1] / "tools" / "windows_job_runner.ps1"
        )
        self.which = which

    def wrap(self, command: list[str], *, cwd: Path) -> list[str]:
        if not command or self.platform_name == "nt" or not _is_windows_unity_executable(command[0]):
            return list(command)
        script = self.script_path.resolve()
        if not script.is_file():
            raise WindowsJobRunnerError(f"Windows Job 包装脚本不存在: {script}")
        powershell = self.powershell_executable or self.which("powershell.exe") or self.which("pwsh.exe")
        if not powershell:
            raise WindowsJobRunnerError("WSL 无法找到 powershell.exe；为避免未托管的 Unity 进程树，已拒绝启动")
        windows_command = [_wsl_to_windows_path(command[0]), *command[1:]]
        command_json = json.dumps(windows_command, ensure_ascii=True, separators=(",", ":"))
        return [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            _wsl_to_windows_path(str(script)),
            "-CommandJsonBase64",
            base64.b64encode(command_json.encode("utf-8")).decode("ascii"),
            "-WorkingDirectory",
            _wsl_to_windows_path(str(cwd.resolve())),
        ]


def current_boot_id() -> str:
    if os.name == "nt":
        # Windows process creation times are absolute FILETIME values, so they
        # already fence PID reuse across reboots.  This value identifies that
        # identity scheme rather than claiming to be a kernel boot UUID.
        return "windows-process-time-v1"
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text(encoding="ascii").strip() if path.is_file() else "unknown-boot"


def process_start_ticks(pid: int) -> int | None:
    if os.name == "nt":
        return _windows_process_start_ticks(pid)
    path = Path(f"/proc/{pid}/stat")
    try:
        fields = _stat_fields(path.read_text(encoding="ascii"))
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return int(fields[20])


def process_group_id(pid: int) -> int | None:
    if os.name == "nt":
        return pid
    try:
        return os.getpgid(pid)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        return None


def identity_matches(identity: ProcessIdentity) -> bool:
    return (
        identity.host_boot_id == current_boot_id()
        and process_start_ticks(identity.pid) == identity.process_start_ticks
        and process_group_id(identity.pid) == identity.pgid
    )


def process_group_members(pgid: int) -> tuple[int, ...]:
    if os.name == "nt" or not Path("/proc").is_dir():
        return ()
    members: list[int] = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = _stat_fields((path / "stat").read_text(encoding="ascii"))
            if int(fields[3]) == pgid and fields[1] != "Z":
                members.append(int(path.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return tuple(sorted(members))


class WindowsProcessTreeBackend(Protocol):
    """Own one Windows Job Object per root process."""

    def bind(self, pid: int) -> None: ...

    def terminate(self, pid: int, *, grace_seconds: float, poll_seconds: float) -> tuple[int, ...]: ...

    def release(self, pid: int) -> None: ...


class _WindowsJobObjectBackend:
    """ctypes-only Job Object implementation, loaded only on native Windows."""

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _ERROR_MORE_DATA = 234

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Object 只能在原生 Windows Python 中创建")
        import ctypes

        self._ctypes = ctypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._jobs: dict[int, int] = {}
        self._configure_functions()

    def bind(self, pid: int) -> None:
        if pid in self._jobs:
            raise RuntimeError(f"PID {pid} 已绑定 Windows Job Object")
        job = self._kernel32.CreateJobObjectW(None, None)
        if not job:
            self._raise_last_error("CreateJobObjectW")
        try:
            limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            limits.BasicLimitInformation.LimitFlags = self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self._kernel32.SetInformationJobObject(
                job,
                self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                byref(limits),
                sizeof(limits),
            ):
                self._raise_last_error("SetInformationJobObject")
            process = self._kernel32.OpenProcess(
                self._PROCESS_TERMINATE
                | self._PROCESS_SET_QUOTA
                | self._PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                pid,
            )
            if not process:
                self._raise_last_error("OpenProcess")
            try:
                if not self._kernel32.AssignProcessToJobObject(job, process):
                    self._raise_last_error("AssignProcessToJobObject")
            finally:
                self._kernel32.CloseHandle(process)
        except BaseException:
            self._kernel32.CloseHandle(job)
            raise
        self._jobs[pid] = int(job)

    def terminate(self, pid: int, *, grace_seconds: float, poll_seconds: float) -> tuple[int, ...]:
        job = self._jobs.get(pid)
        if job is None:
            return (pid,)
        if not self._kernel32.TerminateJobObject(job, 1):
            self._raise_last_error("TerminateJobObject")
        deadline = time.monotonic() + grace_seconds
        remaining = self._members(job)
        while remaining and time.monotonic() < deadline:
            time.sleep(poll_seconds)
            remaining = self._members(job)
        self.release(pid)
        return remaining

    def release(self, pid: int) -> None:
        job = self._jobs.pop(pid, None)
        if job is not None:
            # KILL_ON_JOB_CLOSE also removes descendants left behind after a
            # nominally successful root process exits.
            self._kernel32.CloseHandle(job)

    def _members(self, job: int) -> tuple[int, ...]:
        capacity = 16
        offset = _JOBOBJECT_BASIC_PROCESS_ID_LIST_HEAD.ProcessIdList.offset
        while True:
            buffer = self._ctypes.create_string_buffer(offset + capacity * sizeof(c_size_t))
            returned = c_ulong()
            success = self._kernel32.QueryInformationJobObject(
                job,
                self._JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                buffer,
                len(buffer),
                byref(returned),
            )
            head = self._ctypes.cast(
                buffer, self._ctypes.POINTER(_JOBOBJECT_BASIC_PROCESS_ID_LIST_HEAD)
            ).contents
            if success:
                count = int(head.NumberOfProcessIdsInList)
                values = (c_size_t * count).from_address(self._ctypes.addressof(buffer) + offset)
                return tuple(sorted(int(value) for value in values))
            error = self._ctypes.get_last_error()
            if error != self._ERROR_MORE_DATA:
                raise OSError(error, "QueryInformationJobObject 失败")
            capacity = max(capacity * 2, int(head.NumberOfAssignedProcesses))

    def _configure_functions(self) -> None:
        kernel32 = self._kernel32
        kernel32.CreateJobObjectW.argtypes = [c_void_p, c_void_p]
        kernel32.CreateJobObjectW.restype = c_void_p
        kernel32.SetInformationJobObject.argtypes = [c_void_p, c_ulong, c_void_p, c_ulong]
        kernel32.SetInformationJobObject.restype = c_ulong
        kernel32.OpenProcess.argtypes = [c_ulong, c_ulong, c_ulong]
        kernel32.OpenProcess.restype = c_void_p
        kernel32.AssignProcessToJobObject.argtypes = [c_void_p, c_void_p]
        kernel32.AssignProcessToJobObject.restype = c_ulong
        kernel32.TerminateJobObject.argtypes = [c_void_p, c_ulong]
        kernel32.TerminateJobObject.restype = c_ulong
        kernel32.QueryInformationJobObject.argtypes = [c_void_p, c_ulong, c_void_p, c_ulong, c_void_p]
        kernel32.QueryInformationJobObject.restype = c_ulong
        kernel32.CloseHandle.argtypes = [c_void_p]
        kernel32.CloseHandle.restype = c_ulong

    def _raise_last_error(self, operation: str) -> None:
        error = self._ctypes.get_last_error()
        raise OSError(error, f"{operation} 失败")


class _LARGE_INTEGER(Structure):
    _fields_ = [("QuadPart", c_ulonglong)]


class _IO_COUNTERS(Structure):
    _fields_ = [
        ("ReadOperationCount", c_ulonglong),
        ("WriteOperationCount", c_ulonglong),
        ("OtherOperationCount", c_ulonglong),
        ("ReadTransferCount", c_ulonglong),
        ("WriteTransferCount", c_ulonglong),
        ("OtherTransferCount", c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", _LARGE_INTEGER),
        ("PerJobUserTimeLimit", _LARGE_INTEGER),
        ("LimitFlags", c_ulong),
        ("MinimumWorkingSetSize", c_size_t),
        ("MaximumWorkingSetSize", c_size_t),
        ("ActiveProcessLimit", c_ulong),
        ("Affinity", c_size_t),
        ("PriorityClass", c_ulong),
        ("SchedulingClass", c_ulong),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", c_size_t),
        ("JobMemoryLimit", c_size_t),
        ("PeakProcessMemoryUsed", c_size_t),
        ("PeakJobMemoryUsed", c_size_t),
    ]


class _JOBOBJECT_BASIC_PROCESS_ID_LIST_HEAD(Structure):
    _fields_ = [
        ("NumberOfAssignedProcesses", c_ulong),
        ("NumberOfProcessIdsInList", c_ulong),
        ("ProcessIdList", c_size_t * 1),
    ]


class _FILETIME(Structure):
    _fields_ = [("low", c_ulong), ("high", c_ulong)]


class ProcessSupervisor:
    def __init__(
        self,
        *,
        grace_seconds: float = 10.0,
        poll_seconds: float = 0.05,
        py_spy_path: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        platform_name: str | None = None,
        windows_backend: WindowsProcessTreeBackend | None = None,
        identity_checker: Callable[[ProcessIdentity], bool] = identity_matches,
    ):
        self.grace_seconds = grace_seconds
        self.poll_seconds = poll_seconds
        self.py_spy_path = py_spy_path if py_spy_path is not None else shutil.which("py-spy")
        self.runner = runner
        self.platform_name = platform_name or os.name
        self.identity_checker = identity_checker
        self.windows_backend = windows_backend
        if self.platform_name == "nt" and self.windows_backend is None:
            self.windows_backend = _WindowsJobObjectBackend()

    def bind(self, pid: int) -> None:
        """Attach a new Windows worker to its own kill-on-close Job Object."""
        if self.platform_name == "nt":
            assert self.windows_backend is not None
            self.windows_backend.bind(pid)

    def release(self, pid: int) -> None:
        """Release platform tracking after the root process has exited."""
        if self.platform_name == "nt":
            assert self.windows_backend is not None
            self.windows_backend.release(pid)

    def diagnose(self, identity: ProcessIdentity, directory: Path) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        status_path = directory / "py-spy-status.txt"
        if not self.py_spy_path:
            status_path.write_text("unavailable: py-spy is not installed\n", encoding="utf-8")
            return "unavailable"
        if not self.identity_checker(identity):
            status_path.write_text("skipped: process identity no longer matches\n", encoding="utf-8")
            return "identity_mismatch"
        try:
            completed = self.runner(
                [self.py_spy_path, "dump", "--pid", str(identity.pid), "--subprocesses"],
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            (directory / "py-spy-dump.txt").write_text(
                completed.stdout + ("\nSTDERR:\n" + completed.stderr if completed.stderr else ""),
                encoding="utf-8",
            )
            status = "captured" if completed.returncode == 0 else f"failed:{completed.returncode}"
        except BaseException as exc:
            status = f"failed:{type(exc).__name__}"
        status_path.write_text(status + "\n", encoding="utf-8")
        return status

    def terminate(self, identity: ProcessIdentity, *, diagnostics_dir: Path | None = None) -> CleanupResult:
        diagnostic_status = "not_requested"
        if diagnostics_dir is not None:
            diagnostic_status = self.diagnose(identity, diagnostics_dir)
        if not self.identity_checker(identity):
            return CleanupResult(True, False, (), diagnostic_status)
        if self.platform_name == "nt":
            assert self.windows_backend is not None
            remaining = self.windows_backend.terminate(
                identity.pid,
                grace_seconds=self.grace_seconds,
                poll_seconds=self.poll_seconds,
            )
            return CleanupResult(not remaining, True, remaining, diagnostic_status)
        self._signal_group(identity.pgid, signal.SIGTERM)
        remaining = self._wait_group(identity.pgid, self.grace_seconds)
        if remaining:
            self._signal_group(identity.pgid, signal.SIGKILL)
            remaining = self._wait_group(identity.pgid, self.grace_seconds)
        return CleanupResult(not remaining, True, remaining, diagnostic_status)

    def _wait_group(self, pgid: int, timeout: float) -> tuple[int, ...]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            members = process_group_members(pgid)
            if not members:
                return ()
            time.sleep(self.poll_seconds)
        return process_group_members(pgid)

    @staticmethod
    def _signal_group(pgid: int, value: signal.Signals) -> None:
        try:
            os.killpg(pgid, value)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _stat_fields(text: str) -> list[str]:
    # /proc/<pid>/stat field 2 may contain spaces and parentheses.  Return
    # fields starting at pid, state, ppid, pgrp with comm removed as one unit.
    close = text.rfind(")")
    if close < 0:
        raise ValueError("invalid /proc stat")
    pid = text[: text.find(" ")]
    tail = text[close + 2 :].split()
    return [pid, tail[0], *tail[1:]]


def _is_windows_unity_executable(value: str) -> bool:
    normalized = str(value).replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1].lower() == "unity.exe"


def _wsl_to_windows_path(value: str) -> str:
    match = re.match(r"^/mnt/([A-Za-z])(?:/(.*))?$", str(value))
    if not match:
        return str(value)
    rest = (match.group(2) or "").replace("/", "\\")
    return match.group(1).upper() + ":\\" + rest if rest else match.group(1).upper() + ":\\"


def _windows_process_start_ticks(pid: int) -> int | None:
    if os.name != "nt":
        return None
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [c_ulong, c_ulong, c_ulong]
    kernel32.OpenProcess.restype = c_void_p
    kernel32.GetProcessTimes.argtypes = [
        c_void_p,
        ctypes.POINTER(_FILETIME),
        ctypes.POINTER(_FILETIME),
        ctypes.POINTER(_FILETIME),
        ctypes.POINTER(_FILETIME),
    ]
    kernel32.GetProcessTimes.restype = c_ulong
    kernel32.CloseHandle.argtypes = [c_void_p]
    kernel32.CloseHandle.restype = c_ulong
    process = kernel32.OpenProcess(0x1000, False, pid)
    if not process:
        return None
    creation, exit_time, kernel_time, user_time = (_FILETIME() for _ in range(4))
    try:
        if not kernel32.GetProcessTimes(
            process, byref(creation), byref(exit_time), byref(kernel_time), byref(user_time)
        ):
            return None
        return (int(creation.high) << 32) | int(creation.low)
    finally:
        kernel32.CloseHandle(process)
