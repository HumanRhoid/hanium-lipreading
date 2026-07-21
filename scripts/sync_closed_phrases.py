"""확정된 폐쇄형 문구를 PostgreSQL에 정확히 동기화한다."""

import asyncio
import sys
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
from src.backend.recognition.domain import PhraseCategory  # noqa: E402

CLOSED_PHRASES: tuple[tuple[str, str, PhraseCategory], ...] = (
    ("PAIN_GENERAL", "아파요", PhraseCategory.PAIN),
    ("REQUEST_WATER", "물 주세요", PhraseCategory.REQUEST),
    ("REQUEST_PAINKILLER", "진통제 주세요", PhraseCategory.REQUEST),
    ("STATE_HUNGRY", "배고파요", PhraseCategory.ETC),
    ("REQUEST_TOILET", "화장실 가고 싶어요", PhraseCategory.REQUEST),
    ("REQUEST_NURSE", "간호사 불러 주세요", PhraseCategory.REQUEST),
    ("REQUEST_GUARDIAN", "보호자 불러 주세요", PhraseCategory.REQUEST),
    ("REQUEST_REPOSITION", "자세 바꿔 주세요", PhraseCategory.REQUEST),
    (
        "SYMPTOM_BREATHING_DIFFICULTY",
        "숨 쉬기 힘들어요",
        PhraseCategory.ETC,
    ),
    ("SYMPTOM_DIZZINESS", "어지러워요", PhraseCategory.ETC),
    ("SYMPTOM_NAUSEA", "토할 것 같아요", PhraseCategory.ETC),
    ("STATE_COLD", "추워요", PhraseCategory.ETC),
    ("STATE_HOT", "더워요", PhraseCategory.ETC),
    ("SYMPTOM_PHLEGM", "가래가 있어요", PhraseCategory.ETC),
    ("REQUEST_HELP", "도와주세요", PhraseCategory.REQUEST),
)


async def sync_closed_phrases() -> None:
    """불변 문구 코드를 기준으로 폐쇄형 문구를 정확히 동기화한다."""

    database = SQLAlchemyDatabase(Settings())
    repository = SQLAlchemyRecognitionRepository(database.session_factory)
    try:
        await repository.sync_phrases(CLOSED_PHRASES)
    finally:
        await database.close()


def main() -> None:
    """폐쇄형 문구 동기화 명령의 진입점."""

    asyncio.run(sync_closed_phrases())
    print(f"폐쇄형 문구 {len(CLOSED_PHRASES)}개를 동기화했습니다.")


if __name__ == "__main__":
    main()
