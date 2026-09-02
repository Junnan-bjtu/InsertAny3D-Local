#!/usr/bin/env python3
"""Reject private environment details and large assets from the public tree."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
HISTORICAL_SOURCE_PREFIXES = (
    "code/third_party/patches/",
    "code/third_party/overlays/",
)
LARGE_ASSET_SUFFIXES = {
    ".avi",
    ".bin",
    ".blend",
    ".ckpt",
    ".exr",
    ".fbx",
    ".glb",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".obj",
    ".ply",
    ".png",
    ".prefab",
    ".pt",
    ".pth",
    ".raw",
    ".safetensors",
    ".tar",
    ".unity",
    ".unitypackage",
    ".zip",
}

# Historical patch/overlay files preserve an auditable pre-cleanup source diff.
# They may contain old machine paths, but never credentials or committed assets.
PRIVATE_ENVIRONMENT_PATTERNS = (
    ("private IPv4 address", re.compile(r"(?<![=0-9.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![0-9.])")),
    ("Linux user home", re.compile(r"/(?:home|Users)/(?!user(?:name)?/)[A-Za-z0-9._-]+/")),
    ("private server data path", re.compile(r"/opt/data/private/[A-Za-z0-9._-]+/")),
    ("local WSL mount", re.compile(r"/mnt/[a-z]/(?:Shared|Users|InsertRuns|InsertDebug)(?:/|\b)")),
    ("local Windows path", re.compile(r"\b[A-Za-z]:[\\/](?:Users|Shared|InsertRuns|InsertDebug|ServerRuns|Edited|Codes)[\\/]")),
)
SECRET_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("OpenAI-style token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
)


def repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        path = REPOSITORY_ROOT / relative
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    return paths


def check_file(path: Path) -> list[str]:
    relative = path.relative_to(REPOSITORY_ROOT).as_posix()
    findings: list[str] = []
    size = path.stat().st_size
    suffix = PurePosixPath(relative).suffix.lower()
    historical_source = relative.startswith(HISTORICAL_SOURCE_PREFIXES)

    if size > MAX_FILE_BYTES:
        findings.append(f"{relative}: file is larger than {MAX_FILE_BYTES} bytes")
    if suffix in LARGE_ASSET_SUFFIXES and not historical_source:
        findings.append(f"{relative}: generated or binary asset is not allowed")

    data = path.read_bytes()
    if b"\0" in data:
        return findings
    text = data.decode("utf-8", errors="replace")
    if not historical_source:
        for label, pattern in PRIVATE_ENVIRONMENT_PATTERNS:
            if pattern.search(text):
                findings.append(f"{relative}: contains {label}")
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(f"{relative}: contains possible {label}")
    return findings


def main() -> int:
    findings = [finding for path in repository_files() for finding in check_file(path)]
    if findings:
        print("Public boundary check failed:", file=sys.stderr)
        for finding in sorted(findings):
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("PUBLIC_BOUNDARY_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
