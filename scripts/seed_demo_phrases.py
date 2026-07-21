"""ERD 명세에 정의된 데모 문구를 PostgreSQL에 입력한다."""

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

DEMO_PHRASES: tuple[tuple[str, str, PhraseCategory], ...] = (
    ("PAIN_GENERAL", "아파요", PhraseCategory.PAIN),
    ("REQUEST_WATER", "물 주세요", PhraseCategory.REQUEST),
    ("REQUEST_TOILET", "화장실", PhraseCategory.REQUEST),
    ("STATE_COLD", "추워요", PhraseCategory.ETC),
    ("STATE_HOT", "더워요", PhraseCategory.ETC),
    ("REQUEST_LIGHTS_OFF", "불 꺼 주세요", PhraseCategory.REQUEST),
)


async def seed_demo_phrases() -> None:
    """불변 문구 코드를 기준으로 데모 문구를 idempotent upsert한다."""

    database = SQLAlchemyDatabase(Settings())
    repository = SQLAlchemyRecognitionRepository(database.session_factory)
    try:
        await repository.seed_phrases(DEMO_PHRASES)
    finally:
        await database.close()


def main() -> None:
    """데모 문구 seed 명령의 진입점."""

    asyncio.run(seed_demo_phrases())
    print(f"데모 문구 {len(DEMO_PHRASES)}개를 입력했습니다.")


if __name__ == "__main__":
    main()
