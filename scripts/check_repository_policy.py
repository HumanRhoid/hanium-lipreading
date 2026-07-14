"""Reject tracked data, model artifacts, media, and oversized files."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
FORBIDDEN_ENDINGS = (
    ".avi",
    ".ckpt",
    ".lmdb",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".tar",
    ".tar.gz",
    ".task",
    ".wav",
    ".zip",
)
FORBIDDEN_DIRECTORIES = (
    "checkpoints/",
    "models/",
    "runs/",
    "wandb/",
)
ALLOWED_DATA_FILES = {"data/README.md"}


def tracked_files() -> list[Path]:
    """Return paths tracked by Git, preserving platform-specific path decoding."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [
        Path(os.fsdecode(raw_path))
        for raw_path in result.stdout.split(b"\0")
        if raw_path
    ]


def find_violations(paths: list[Path]) -> list[str]:
    """Return human-readable repository policy violations."""
    violations: list[str] = []

    for path in paths:
        if not path.exists():
            continue

        relative_path = path.as_posix()
        lower_path = relative_path.lower()

        if (
            relative_path.startswith("data/")
            and relative_path not in ALLOWED_DATA_FILES
        ):
            violations.append(f"tracked data file: {relative_path}")

        if lower_path.endswith(FORBIDDEN_ENDINGS):
            violations.append(f"prohibited file type: {relative_path}")

        if lower_path.startswith(FORBIDDEN_DIRECTORIES):
            violations.append(f"prohibited artifact directory: {relative_path}")

        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mib = file_size / (1024 * 1024)
            violations.append(
                f"file exceeds 20 MiB ({size_mib:.1f} MiB): {relative_path}"
            )

    return violations


def main() -> int:
    """Validate all tracked files against the repository policy."""
    violations = find_violations(tracked_files())
    if violations:
        print("Repository policy violations found:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("Repository policy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
