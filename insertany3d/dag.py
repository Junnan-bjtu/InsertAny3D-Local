"""The versioned InsertAny3D stage graph and resource declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    name: str
    contract_version: str
    resources: tuple[str, ...]
    max_attempts: int
    timeout_seconds: int


STAGES: tuple[StageSpec, ...] = (
    StageSpec("unity_anchor", "unity-anchor-v1", ("unity_gpu", "project_lock"), 1, 3600),
    StageSpec("image_edit", "image-edit-v1", ("image_api",), 3, 420),
    StageSpec("edit_gate", "edit-gate-v1", (), 1, 0),
    StageSpec("upload_inputs", "upload-inputs-v1", ("ssh_io",), 3, 1800),
    StageSpec("model_generation", "model-generation-v1", ("remote_gpu",), 2, 7200),
    StageSpec("render_alignment_views", "render-alignment-v1", ("remote_gpu",), 2, 3600),
    StageSpec("segment_inputs", "segment-inputs-v1", ("remote_gpu",), 2, 3600),
    StageSpec("gim_match", "gim-match-v1", ("remote_gpu",), 2, 3600),
    StageSpec("estimate_pose", "estimate-pose-v1", ("remote_cpu",), 1, 1800),
    StageSpec("sags_segment_vote", "sags-vote-v1", ("remote_gpu",), 2, 5400),
    StageSpec("debug_bundle", "debug-bundle-v1", ("remote_io",), 2, 1800),
    StageSpec("download_results", "download-results-v1", ("ssh_io",), 3, 1800),
    StageSpec("unity_apply", "unity-apply-v1", ("unity_gpu", "project_lock"), 1, 3600),
    StageSpec("unity_eval6", "unity-eval6-v1", ("unity_gpu", "project_lock"), 2, 3600),
    StageSpec("evaluate_absolute", "evaluate-absolute-v1", ("evaluation_api",), 3, 1800),
)

STAGE_BY_NAME = {stage.name: stage for stage in STAGES}
STAGE_INDEX = {stage.name: index for index, stage in enumerate(STAGES)}
REMOTE_EVIDENCE_START = STAGE_INDEX["model_generation"]
REMOTE_EVIDENCE_END = STAGE_INDEX["sags_segment_vote"]
REMOTE_PROCESS_STAGES = frozenset(
    {
        "model_generation",
        "render_alignment_views",
        "segment_inputs",
        "gim_match",
        "estimate_pose",
        "sags_segment_vote",
        "debug_bundle",
    }
)


def predecessor(name: str) -> str | None:
    index = STAGE_INDEX[name]
    return STAGES[index - 1].name if index else None


def downstream(name: str) -> tuple[str, ...]:
    return tuple(stage.name for stage in STAGES[STAGE_INDEX[name] + 1 :])


def image_api_limits(resources: dict) -> tuple[str, int, int]:
    """Return (mode, initial, maximum) for the API-only concurrency bucket."""
    api = resources.get("imageApi") or {}
    mode = str(api.get("mode", "adaptive"))
    initial = int(api.get("initialLimit", resources.get("editSlotsInitial", 3)))
    maximum = int(api.get("maximumLimit", resources.get("editSlotsMax", 5)))
    if mode == "fixed":
        limit = int(api.get("limit", initial))
        return mode, limit, limit
    return mode, initial, maximum
