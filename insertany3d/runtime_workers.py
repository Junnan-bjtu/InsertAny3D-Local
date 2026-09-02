"""Composition layer for explicitly enabled real batch workers."""

from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path
from .dag import STAGE_BY_NAME, STAGE_INDEX
from .executors import CommandExecutor, ExecutionResult
from .remote_worker import REMOTE_STAGES, RemoteProfile, build_remote_worker_command
from .scheduler import BatchController, WorkItem
from .stage_wiring import StageWiringError, write_stage_request
from .worker import StageExecutor


class RuntimeWorkerConfigurationError(ValueError):
    """A real pipeline worker is missing an executable or remote profile."""


DEFAULT_LOCAL_ENVIRONMENT_FILE = ".env"
LOCAL_ENVIRONMENT_FILE_VARIABLE = "INSERTANY3D_LOCAL_ENV_FILE"
_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_environment_file(
    path: str | os.PathLike[str],
    *,
    environ: dict[str, str] | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Load a small dotenv file without taking ownership of the process env.

    Local configuration is intentionally additive: an explicitly exported
    process value wins over ``.env`` unless callers opt into ``override``.
    The parser accepts the shell-compatible ``export NAME=value`` form but
    never executes the file, so command substitutions and other shell syntax
    cannot run as a side effect of starting the controller.
    """

    target = Path(path).expanduser()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RuntimeWorkerConfigurationError(f"环境配置文件不存在: {target}") from None
    except (OSError, UnicodeError) as exc:
        raise RuntimeWorkerConfigurationError(f"无法读取环境配置文件 {target}: {exc}") from exc

    values = environ if environ is not None else os.environ
    loaded: dict[str, str] = {}
    for line_number, original in enumerate(text.splitlines(), start=1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export") and (len(line) == 6 or line[6].isspace()):
            line = line[6:].lstrip()
        if "=" not in line:
            raise RuntimeWorkerConfigurationError(
                f"环境配置文件 {target} 第 {line_number} 行缺少 '='"
            )
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _ENVIRONMENT_KEY.fullmatch(key):
            raise RuntimeWorkerConfigurationError(
                f"环境配置文件 {target} 第 {line_number} 行变量名无效: {key!r}"
            )
        raw_value = raw_value.strip()
        try:
            tokens = shlex.split(f"{key}={raw_value}", comments=False, posix=True)
        except ValueError as exc:
            raise RuntimeWorkerConfigurationError(
                f"环境配置文件 {target} 第 {line_number} 行引号无效"
            ) from exc
        if len(tokens) != 1 or "=" not in tokens[0]:
            raise RuntimeWorkerConfigurationError(
                f"环境配置文件 {target} 第 {line_number} 行值无效"
            )
        value = tokens[0].split("=", 1)[1]
        loaded[key] = value
        if override or key not in values:
            values[key] = value
    return loaded


def load_local_environment(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: dict[str, str] | None = None,
    repository_root: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Load the Local-only env file and return its path when present.

    The server's ``.insertany3d/runtime.env`` is deliberately not searched by
    this function.  This keeps machine/GPU configuration on the server side
    and prevents a local checkout from silently replacing it.
    """

    values = environ if environ is not None else os.environ
    configured = path or values.get(LOCAL_ENVIRONMENT_FILE_VARIABLE)
    if configured:
        target = Path(configured).expanduser()
        if not target.is_absolute():
            root = Path(repository_root or Path(__file__).resolve().parents[1])
            target = root / target
    else:
        root = Path(repository_root or Path(__file__).resolve().parents[1])
        target = root / DEFAULT_LOCAL_ENVIRONMENT_FILE
    if not target.exists():
        if configured:
            raise RuntimeWorkerConfigurationError(
                f"环境配置文件不存在: {target}"
            )
        return None
    if target.is_symlink() or not target.is_file():
        raise RuntimeWorkerConfigurationError(
            f"环境配置文件必须是普通文件: {target}"
        )
    load_environment_file(target, environ=values)
    return target.resolve()


class CompositeStageExecutor:
    """Route disjoint stage sets through one scheduler worker."""

    def __init__(self, executors: list[StageExecutor]):
        if not executors:
            raise RuntimeWorkerConfigurationError("真实 worker 至少需要一个 stage executor")
        routes: dict[str, StageExecutor] = {}
        for executor in executors:
            stages = getattr(executor, "supported_stages", None)
            if not stages:
                raise RuntimeWorkerConfigurationError(
                    f"{type(executor).__name__} 没有声明 supported_stages"
                )
            for stage in stages:
                if stage not in STAGE_INDEX:
                    raise RuntimeWorkerConfigurationError(f"executor 声明了未知 stage: {stage}")
                if stage in routes:
                    raise RuntimeWorkerConfigurationError(f"stage 被重复注册: {stage}")
                routes[stage] = executor
        self._routes = routes

    @property
    def supported_stages(self) -> tuple[str, ...]:
        return tuple(sorted(self._routes, key=STAGE_INDEX.__getitem__))

    def execute(self, controller: BatchController, item: WorkItem) -> ExecutionResult:
        executor = self._routes.get(item.stage)
        if executor is None:
            return ExecutionResult(
                False,
                stage_status="failed_terminal",
                error_code="invalid_input",
                message=f"没有为 {item.stage} 注册真实 executor",
            )
        return executor.execute(controller, item)


class RemoteStageExecutor:
    """Adapt the durable SSH wrapper to the queue's StageExecutor protocol."""

    supported_stages = tuple(sorted(REMOTE_STAGES, key=STAGE_INDEX.__getitem__))

    def __init__(
        self,
        profile: RemoteProfile,
        *,
        command_executor: CommandExecutor | None = None,
        local_python: str = sys.executable,
    ):
        self.profile = profile
        self.command_executor = command_executor or CommandExecutor()
        self.local_python = local_python

    def execute(self, controller: BatchController, item: WorkItem) -> ExecutionResult:
        try:
            _request, request_path = write_stage_request(controller.store, item)
            batch = controller.store.row(
                "SELECT root_path FROM batches WHERE batch_id=?", (item.batch_id,)
            )
            if batch is None:
                raise StageWiringError(f"batch 不存在: {item.batch_id}")
            root = Path(str(batch["root_path"])).resolve()
            command = build_remote_worker_command(
                request_path,
                root,
                self.profile,
                local_python=self.local_python,
            )
        except (OSError, StageWiringError, ValueError) as exc:
            return ExecutionResult(
                False,
                stage_status="failed_terminal",
                error_code="invalid_input",
                message=f"远端启动前检查失败: {exc}",
            )

        environment = dict(os.environ)
        for secret_name in (
            "APIYI_API_KEY",
            "APIYI_API_KEY_FILE",
            "GEMINI_API_KEY",
            "GEMINI_API_KEY_FILE",
            "BEE_API_KEY",
        ):
            environment.pop(secret_name, None)
        repository_root = str(Path(__file__).resolve().parents[1])
        existing_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            repository_root
            if not existing_python_path
            else repository_root + os.pathsep + existing_python_path
        )
        outcome = self.command_executor.execute(
            controller,
            item,
            command,
            env=environment,
            timeout_seconds=float(STAGE_BY_NAME[item.stage].timeout_seconds),
        )
        if not outcome.succeeded and outcome.error_code in {"stalled", "worker_crash"}:
            # The local SSH wrapper disappearing does not prove that nohup's
            # remote child stopped.  Fence the remote resource until an explicit
            # probe settles the attempt.
            return ExecutionResult(
                False,
                stage_status="failed_retryable",
                error_code="delivery_unknown",
                message=(
                    "本机 SSH worker 中断，无法确认远端进程状态；"
                    "不得自动重启同一远端步骤"
                ),
                cleanup_completed=False,
            )
        return outcome
