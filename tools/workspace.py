"""Task workspace primitives for the InsertAny3D remote pipeline.

The workspace is deliberately task-local: there is no global artifact store or
legacy path resolution.  Stage workers can use ``begin_stage`` to obtain an
attempt directory and publish a small, atomic stage manifest.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
import fcntl
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str, label: str = "name") -> str:
    value = str(value).strip()
    if not value or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{label} 不是安全的目录名: {value!r}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON next to the destination and replace it atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TaskWorkspace:
    """A single task's filesystem workspace."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.task_manifest = self.root / "task_manifest.json"

    def initialize(self, *, run_id: str, project_id: str, task_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        safe_name(project_id, "project_id")
        safe_name(task_id, "task_id")
        for relative in ("inputs", "logs", "stages"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        manifest = {
            "kind": "insertany3d.task",
            "schemaVersion": SCHEMA_VERSION,
            "runId": str(run_id),
            "projectId": str(project_id),
            "taskId": str(task_id),
            "status": "running",
            "config": config or {},
            "stages": {},
            "createdAtUtc": _utc_now(),
            "updatedAtUtc": _utc_now(),
        }
        atomic_write_json(self.task_manifest, manifest)
        return manifest

    def begin_stage(self, stage: str) -> tuple[str, Path, Path]:
        safe_name(stage, "stage")
        stage_root = self.root / "stages" / stage
        attempts = stage_root / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        with (stage_root / ".attempt.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing = [int(p.name) for p in attempts.iterdir() if p.is_dir() and p.name.isdigit()]
            attempt = f"{(max(existing) + 1 if existing else 1):04d}"
            attempt_dir = attempts / attempt
            attempt_dir.mkdir()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        output_dir = stage_root / "output"
        stage_manifest = attempt_dir / "manifest.json"
        current_manifest = stage_root / "manifest.json"
        record = {
            "kind": "insertany3d.stage",
            "schemaVersion": SCHEMA_VERSION,
            "stage": stage,
            "attempt": int(attempt),
            "status": "running",
            "attemptDir": attempt_dir.relative_to(self.root).as_posix(),
            "outputDir": output_dir.relative_to(self.root).as_posix(),
            "startedAtUtc": _utc_now(),
        }
        atomic_write_json(stage_manifest, record)
        atomic_write_json(current_manifest, {
            "kind": "insertany3d.stage.current",
            "schemaVersion": SCHEMA_VERSION,
            "stage": stage,
            "attempt": int(attempt),
            "status": "running",
            "manifest": stage_manifest.relative_to(self.root).as_posix(),
            "updatedAtUtc": _utc_now(),
        })
        return attempt, attempt_dir, stage_manifest

    def finish_stage(self, stage_manifest: Path, *, status: str, outputs: dict[str, str] | None = None, error: str | None = None) -> None:
        record = json.loads(Path(stage_manifest).read_text(encoding="utf-8"))
        record["status"] = status
        if outputs is not None:
            record["outputs"] = outputs
        if error:
            record["error"] = error
        record["finishedAtUtc"] = _utc_now()
        atomic_write_json(Path(stage_manifest), record)
        stage_root = Path(stage_manifest).parent.parent
        current_path = stage_root / "manifest.json"
        try:
            current = json.loads(current_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        if int(current.get("attempt", 0) or 0) > int(record["attempt"]):
            return
        atomic_write_json(current_path, {
            "kind": "insertany3d.stage.current",
            "schemaVersion": SCHEMA_VERSION,
            "stage": record["stage"],
            "attempt": record["attempt"],
            "status": status,
            "manifest": Path(stage_manifest).relative_to(self.root).as_posix(),
            "updatedAtUtc": _utc_now(),
        })

    def publish_file(self, stage: str, attempt_file: Path, relative_output: str) -> Path:
        """Atomically publish one validated attempt file into stage/output."""
        safe_name(stage, "stage")
        source = Path(attempt_file).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = (self.root / "stages" / stage / "output" / relative_output).resolve()
        output_root = (self.root / "stages" / stage / "output").resolve()
        if output_root not in destination.parents:
            raise ValueError("relative_output 不能越过 stage output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def publish_tree(self, stage: str, attempt_tree: Path, relative_output: str = ".") -> Path:
        """Publish a directory via a temporary sibling and atomic rename."""
        safe_name(stage, "stage")
        source = Path(attempt_tree).resolve()
        if not source.is_dir():
            raise NotADirectoryError(source)
        output_root = (self.root / "stages" / stage / "output").resolve()
        destination = (output_root / relative_output).resolve()
        if output_root not in destination.parents and destination != output_root:
            raise ValueError("relative_output 不能越过 stage output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        shutil.copytree(source, temporary)
        backup = None
        published = False
        try:
            if destination.exists():
                backup = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.old"
                os.replace(destination, backup)
            os.replace(temporary, destination)
            published = True
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            if backup and backup.exists():
                if published:
                    shutil.rmtree(backup)
                else:
                    # Preserve the previously published output if the final
                    # rename fails (for example, a transient filesystem
                    # error).  A failed attempt must never erase a ready one.
                    if destination.exists():
                        if destination.is_dir():
                            shutil.rmtree(destination)
                        else:
                            destination.unlink()
                    os.replace(backup, destination)
        return destination


__all__ = ["SCHEMA_VERSION", "TaskWorkspace", "atomic_write_json", "safe_name"]
