"""Alembic migration과 SQLAlchemy metadata의 PostgreSQL 계약 테스트."""

import asyncio
import logging
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import command

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANAGED_TABLES = {"session", "phrase", "utterance", "users", "login_session"}


def _alembic_config() -> Config:
    """프로젝트 루트를 기준으로 Alembic 설정을 생성한다."""

    return Config(str(PROJECT_ROOT / "alembic.ini"))


async def _drop_managed_tables(database_url: str) -> None:
    """테스트 DB에서 이 서비스가 소유한 테이블만 명시적으로 제거한다."""

    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            actual_database = await connection.scalar(text("SELECT current_database()"))
            if actual_database != make_url(database_url).database:
                raise RuntimeError("연결된 DB가 TEST_DATABASE_URL과 다릅니다")
            await connection.execute(text("DROP TABLE IF EXISTS login_session CASCADE"))
            await connection.execute(text("DROP TABLE IF EXISTS users CASCADE"))
            await connection.execute(text("DROP TABLE IF EXISTS utterance CASCADE"))
            await connection.execute(text("DROP TABLE IF EXISTS phrase CASCADE"))
            await connection.execute(text('DROP TABLE IF EXISTS "session" CASCADE'))
            await connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
    finally:
        await engine.dispose()


async def _schema_snapshot(database_url: str) -> dict[str, object]:
    """동기 Inspector 결과를 event loop 밖에서도 비교 가능한 값으로 반환한다."""

    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:

            def inspect_schema(sync_connection):
                inspector = inspect(sync_connection)
                return {
                    "tables": set(inspector.get_table_names()),
                    "session_checks": {
                        constraint["name"]
                        for constraint in inspector.get_check_constraints("session")
                    },
                    "phrase_checks": {
                        constraint["name"]
                        for constraint in inspector.get_check_constraints("phrase")
                    },
                    "utterance_checks": {
                        constraint["name"]
                        for constraint in inspector.get_check_constraints("utterance")
                    },
                    "phrase_uniques": {
                        constraint["name"]
                        for constraint in inspector.get_unique_constraints("phrase")
                    },
                    "session_indexes": {
                        index["name"]: tuple(index["column_names"])
                        for index in inspector.get_indexes("session")
                    },
                    "utterance_indexes": {
                        index["name"]: tuple(index["column_names"])
                        for index in inspector.get_indexes("utterance")
                    },
                    "utterance_foreign_keys": {
                        constraint["name"]: (
                            constraint["referred_table"],
                            constraint["options"].get("ondelete"),
                        )
                        for constraint in inspector.get_foreign_keys("utterance")
                    },
                }

            return await connection.run_sync(inspect_schema)
    finally:
        await engine.dispose()


async def _table_names(database_url: str) -> set[str]:
    """현재 PostgreSQL schema의 테이블 이름을 조회한다."""

    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            return await connection.run_sync(
                lambda sync_connection: set(inspect(sync_connection).get_table_names())
            )
    finally:
        await engine.dispose()


def test_empty_database_upgrade_downgrade_upgrade_and_metadata_parity(
    postgres_url,
    monkeypatch,
):
    """빈 DB 왕복과 autogenerate 차이 없음까지 한 흐름으로 검증한다."""

    # env.py가 Settings를 통해 URL을 읽으므로 반드시 전용 테스트 URL로 고정한다.
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    alembic_config = _alembic_config()
    asyncio.run(_drop_managed_tables(postgres_url))

    try:
        assert MANAGED_TABLES.isdisjoint(asyncio.run(_table_names(postgres_url)))

        command.upgrade(alembic_config, "head")
        first_upgrade = asyncio.run(_schema_snapshot(postgres_url))
        assert MANAGED_TABLES <= first_upgrade["tables"]

        command.downgrade(alembic_config, "base")
        assert MANAGED_TABLES.isdisjoint(asyncio.run(_table_names(postgres_url)))

        command.upgrade(alembic_config, "head")
        second_upgrade = asyncio.run(_schema_snapshot(postgres_url))

        assert second_upgrade == first_upgrade
        assert second_upgrade["session_checks"] == {
            "ck_session_ended_after_started",
            "ck_session_mode",
        }
        assert second_upgrade["phrase_checks"] == {
            "ck_phrase_category",
            "ck_phrase_text_not_blank",
        }
        assert second_upgrade["utterance_checks"] == {
            "ck_utterance_confidence_range",
            "ck_utterance_corrected_text_not_blank",
            "ck_utterance_raw_text_not_blank",
        }
        assert second_upgrade["phrase_uniques"] == {"uq_phrase_phrase_code"}
        assert second_upgrade["session_indexes"] == {
            "ix_session_started_at": ("started_at",)
        }
        assert second_upgrade["utterance_indexes"] == {
            "ix_utterance_phrase_id": ("phrase_id",),
            "ix_utterance_session_created_at": ("session_id", "created_at"),
        }
        assert second_upgrade["utterance_foreign_keys"] == {
            "fk_utterance_phrase_id_phrase": ("phrase", "SET NULL"),
            "fk_utterance_session_id_session": ("session", "CASCADE"),
        }

        # CI에서 `alembic check`와 같은 방식으로 migration/metadata drift를 잡는다.
        command.check(alembic_config)
    finally:
        asyncio.run(_drop_managed_tables(postgres_url))


def test_alembic_command_does_not_disable_application_loggers(
    postgres_url,
    monkeypatch,
):
    """프로세스 내 migration이 이후 개인정보 안전 로그를 무력화하지 않는다."""

    monkeypatch.setenv("DATABASE_URL", postgres_url)
    application_logger = logging.getLogger("src.backend.recognition.api")
    original_disabled = application_logger.disabled
    application_logger.disabled = False

    try:
        command.current(_alembic_config())
        assert application_logger.disabled is False
    finally:
        application_logger.disabled = original_disabled
