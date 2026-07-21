"""프로세스 강제 종료 후 남은 오래된 열린 인식 세션을 종료 처리한다."""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# 설치하지 않은 개발 checkout에서도 파일을 직접 실행할 수 있게 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.purge_recognition_data import timezone_aware_datetime  # noqa: E402
from src.backend.core.config import Settings  # noqa: E402
from src.backend.core.database import SQLAlchemyDatabase  # noqa: E402
from src.backend.recognition.adapters.repository import (  # noqa: E402
    SQLAlchemyRecognitionRepository,
)


def build_parser() -> argparse.ArgumentParser:
    """운영자가 안전한 기준 시각과 확인을 명시하도록 한다."""

    parser = argparse.ArgumentParser(
        description=(
            "기준 시각보다 먼저 시작했고 아직 열려 있는 인식 세션을 종료 처리합니다."
        )
    )
    parser.add_argument(
        "--before",
        required=True,
        type=timezone_aware_datetime,
        help="이 시각보다 먼저 시작한 열린 세션을 종료합니다(ISO 8601, 타임존 필수).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="실제 변경을 승인합니다. 이 옵션이 없으면 변경하지 않습니다.",
    )
    return parser


async def reconcile_abandoned_sessions(*, before: datetime) -> int:
    """선택한 기준 시각으로 짧은 transaction에서 열린 세션을 정리한다."""

    database = SQLAlchemyDatabase(Settings())
    repository = SQLAlchemyRecognitionRepository(database.session_factory)
    try:
        return await repository.reconcile_abandoned_sessions(before=before)
    finally:
        await database.close()


def main() -> None:
    """확인을 받은 뒤에만 DB의 오래된 열린 세션을 변경한다."""

    parser = build_parser()
    args = parser.parse_args()
    if not args.confirm:
        parser.error("실제 변경에는 --confirm 옵션이 필요합니다.")

    reconciled_count = asyncio.run(reconcile_abandoned_sessions(before=args.before))
    print(f"오래된 열린 세션 {reconciled_count}개를 종료 처리했습니다.")


if __name__ == "__main__":
    main()
