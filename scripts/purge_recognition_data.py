"""명시한 범위의 인식 세션과 연결된 발화를 삭제한다."""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

# 설치하지 않은 개발 checkout에서도 파일을 직접 실행할 수 있게 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.core.config import Settings  # noqa: E402
from src.backend.core.database import SQLAlchemyDatabase  # noqa: E402
from src.backend.recognition.adapters.repository import (  # noqa: E402
    SQLAlchemyRecognitionRepository,
)


def positive_session_id(value: str) -> int:
    """양의 세션 ID만 argparse 값으로 허용한다."""

    try:
        session_id = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("세션 ID는 정수여야 합니다.") from error
    if session_id <= 0:
        raise argparse.ArgumentTypeError("세션 ID는 1 이상이어야 합니다.")
    return session_id


def timezone_aware_datetime(value: str) -> datetime:
    """타임존을 포함한 ISO 8601 기준 시각을 UTC로 정규화한다."""

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--before는 타임존을 포함한 ISO 8601 시각이어야 합니다."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--before에는 타임존이 포함되어야 합니다.")
    return parsed.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    """삭제 범위를 하나만 선택하도록 CLI 계약을 구성한다."""

    parser = argparse.ArgumentParser(
        description="인식 세션과 연결된 발화를 명시한 범위만큼 삭제합니다."
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--session-id",
        type=positive_session_id,
        help="정확히 일치하는 세션 ID를 삭제합니다.",
    )
    scope.add_argument(
        "--before",
        type=timezone_aware_datetime,
        help="이 시각보다 먼저 시작한 세션을 삭제합니다(ISO 8601, 타임존 필수).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="실제 삭제를 승인합니다. 이 옵션이 없으면 삭제하지 않습니다.",
    )
    return parser


async def purge_recognition_data(
    *,
    session_id: int | None,
    before: datetime | None,
) -> int:
    """선택한 하나의 조건으로 짧은 transaction에서 데이터를 삭제한다."""

    database = SQLAlchemyDatabase(Settings())
    repository = SQLAlchemyRecognitionRepository(database.session_factory)
    try:
        return await repository.purge(session_id=session_id, before=before)
    finally:
        await database.close()


def main() -> None:
    """확인을 받은 뒤에만 DB 연결과 실제 삭제를 수행한다."""

    parser = build_parser()
    args = parser.parse_args()
    if not args.confirm:
        parser.error("실제 삭제에는 --confirm 옵션이 필요합니다.")

    deleted_count = asyncio.run(
        purge_recognition_data(
            session_id=args.session_id,
            before=args.before,
        )
    )
    print(f"인식 세션 {deleted_count}개와 연결된 발화를 삭제했습니다.")


if __name__ == "__main__":
    main()
