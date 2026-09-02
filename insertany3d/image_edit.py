"""Contracts and persistence helpers for multi-candidate image edits.

Each candidate is an independently addressable generation.  The helpers are
pure apart from the explicit atomic JSON writer, which keeps a successful
candidate intact when another candidate fails or is retried.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

DEFAULT_NUM_GENERATIONS = 3


def validate_generation_group(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a persisted generation-group manifest."""
    if value.get("schemaVersion") != 1 or value.get("kind") != "insertany3d.image-edit-generation-group":
        raise ValueError("generation group contract version/kind 无效")
    count = generation_count({"num_gen_image_per_task": value.get("requestedCount")})
    generations = value.get("generations")
    if not isinstance(generations, list) or len(generations) != count:
        raise ValueError("generation group candidates 数量与 requestedCount 不一致")
    indexes = [int(item.get("index", 0)) for item in generations if isinstance(item, Mapping)]
    if indexes != list(range(1, count + 1)):
        raise ValueError("generation index 必须连续从 1 开始")
    # A review manifest is intentionally a simple display-index -> path map.
    # Validate only the mapping shape and path presence; group/attempt identity
    # belongs to the scheduler and is not duplicated here.
    review = value.get("reviewManifest")
    if review is not None:
        if not isinstance(review, Mapping):
            raise ValueError("reviewManifest 必须是对象")
        candidates = review.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("reviewManifest.candidates 必须是数组")
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not candidate.get("path"):
                raise ValueError("reviewManifest 候选必须包含 path")
    return dict(value)


def generation_count(config: Mapping[str, Any] | None = None) -> int:
    value = DEFAULT_NUM_GENERATIONS if config is None else config.get("num_gen_image_per_task", DEFAULT_NUM_GENERATIONS)
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("num_gen_image_per_task 必须是正整数") from exc
    if value <= 0 or value > 32:
        raise ValueError("num_gen_image_per_task 必须在 1..32 范围内")
    return value


def new_generation_group(task_id: str, *, count: int = DEFAULT_NUM_GENERATIONS, group_id: str | None = None) -> dict[str, Any]:
    count = generation_count({"num_gen_image_per_task": count})
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schemaVersion": 1,
        "kind": "insertany3d.image-edit-generation-group",
        "taskId": str(task_id),
        "groupId": group_id or f"{task_id}-group-1",
        "requestedCount": count,
        "status": "generating",
        "acceptedGeneration": None,
        "createdAtUtc": now,
        "updatedAtUtc": now,
        "generations": [
            {"index": index, "status": "pending", "attempt": 0, "output": None, "error": None}
            for index in range(1, count + 1)
        ],
        "reviewManifest": {"candidates": []},
    }


def missing_generations(group: Mapping[str, Any]) -> list[int]:
    return [int(item["index"]) for item in group.get("generations", []) if item.get("status") != "succeeded"]


def record_generation(
    group: Mapping[str, Any],
    index: int,
    *,
    status: str,
    attempt: int,
    output: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"pending", "running", "succeeded", "failed_retryable", "failed_terminal"}:
        raise ValueError(f"未知 generation 状态: {status}")
    result = json.loads(json.dumps(group))
    candidate = next((item for item in result["generations"] if int(item["index"]) == int(index)), None)
    if candidate is None:
        raise ValueError(f"generation index 越界: {index}")
    if int(attempt) < 1:
        raise ValueError("attempt 必须是正整数")
    candidate.update({"status": status, "attempt": int(attempt), "output": dict(output) if output else None, "error": dict(error) if error else None})
    review = result.setdefault("reviewManifest", {"candidates": []})
    if isinstance(review, dict):
        entries = [item for item in review.setdefault("candidates", []) if int(item.get("index", -1)) != int(index)]
        path = (output or {}).get("fullPath") or (output or {}).get("path")
        if status == "succeeded" and path:
            entries.append({"index": int(index), "path": str(path)})
        review["candidates"] = sorted(entries, key=lambda item: int(item["index"]))
    statuses = [item["status"] for item in result["generations"]]
    if result.get("acceptedGeneration") is not None:
        result["status"] = "accepted"
    elif all(value == "succeeded" for value in statuses):
        result["status"] = "ready_for_review"
    elif any(value == "running" for value in statuses):
        result["status"] = "generating"
    else:
        result["status"] = "generating" if missing_generations(result) else "ready_for_review"
    result["updatedAtUtc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return result


def decide_generation(group: Mapping[str, Any], decision: str, *, selected_index: int | None = None) -> dict[str, Any]:
    decision = decision.strip().upper()
    result = json.loads(json.dumps(group))
    if decision == "N":
        result["status"] = "canceled"
        result["acceptedGeneration"] = None
    elif decision == "R":
        result = new_generation_group(str(result["taskId"]), count=int(result["requestedCount"]), group_id=f"{result['taskId']}-group-{int(str(result['groupId']).rsplit('-', 1)[-1]) + 1}")
    else:
        try:
            index = int(decision if selected_index is None else selected_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("审核必须输入 1..x、R 或 N") from exc
        candidate = next((item for item in result["generations"] if int(item["index"]) == index), None)
        if candidate is None or candidate.get("status") != "succeeded":
            raise ValueError("只能选择已成功的 generation")
        output = candidate.get("output") or {}
        path = output.get("fullPath") or output.get("path")
        if not path:
            raise ValueError("所选 generation 缺少输出路径")
        result["acceptedGeneration"] = {"index": index, "attempt": int(candidate["attempt"]), "path": str(path), "output": output}
        result["status"] = "accepted"
    result["updatedAtUtc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return result


def write_group(path: str | Path, group: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(group, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
