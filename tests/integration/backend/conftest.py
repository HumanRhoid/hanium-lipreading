"""실제 PostgreSQL을 사용하는 백엔드 통합 테스트 fixture."""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.backend.core.database import Base

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def postgres_url():
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL이 없어 PostgreSQL 통합 테스트를 건너뜁니다")
    parsed = make_url(TEST_DATABASE_URL)
    if os.getenv("ALLOW_DESTRUCTIVE_DB_TESTS") != "1":
        raise pytest.UsageError(
            "파괴적 DB 통합 테스트에는 ALLOW_DESTRUCTIVE_DB_TESTS=1이 필요합니다"
        )
    if not parsed.database or not parsed.database.endswith("_test"):
        raise pytest.UsageError("통합 테스트 DB 이름은 _test로 끝나야 합니다")
    if parsed.host not in {"localhost", "127.0.0.1", "::1"}:
        raise pytest.UsageError("통합 테스트 DB는 loopback host만 허용합니다")
    return TEST_DATABASE_URL


@pytest.fixture
async def postgres_session_factory(postgres_url):
    engine = create_async_engine(postgres_url)
    async with engine.begin() as connection:
        actual_database = await connection.scalar(text("SELECT current_database()"))
        expected_database = make_url(postgres_url).database
        if actual_database != expected_database:
            raise RuntimeError("연결된 DB가 TEST_DATABASE_URL과 다릅니다")
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()
