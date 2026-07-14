"""Git이 추적하는 데이터, 모델 산출물, 미디어 및 대용량 파일을 차단합니다."""

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
    """운영체제의 경로 인코딩을 보존하면서 Git 추적 파일 경로를 반환합니다."""
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
    """저장소 정책을 위반한 항목의 설명을 반환합니다."""
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
            violations.append(f"추적된 데이터 파일: {relative_path}")

        if lower_path.endswith(FORBIDDEN_ENDINGS):
            violations.append(f"금지된 파일 형식: {relative_path}")

        if lower_path.startswith(FORBIDDEN_DIRECTORIES):
            violations.append(f"금지된 산출물 디렉터리: {relative_path}")

        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mib = file_size / (1024 * 1024)
            violations.append(
                f"파일 크기가 20 MiB를 초과함 ({size_mib:.1f} MiB): {relative_path}"
            )

    return violations


def main() -> int:
    """Git이 추적하는 모든 파일이 저장소 정책을 준수하는지 검사합니다."""
    violations = find_violations(tracked_files())
    if violations:
        print("저장소 정책 위반 항목을 발견했습니다:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("저장소 정책 검사를 통과했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
