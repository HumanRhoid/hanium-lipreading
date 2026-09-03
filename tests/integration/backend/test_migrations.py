"""Alembic migration과 SQLAlchemy metadata의 PostgreSQL 계약 테스트."""

import asyncio
import logging
from pathlib import Path
from uuid import UUID

import pytest
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import command

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANAGED_TABLES = {
    "session",
    "phrase",
    "utterance",
    "users",
    "login_session",
    "video_asset",
    "user_consent",
    "phrase_usage_stat",
    "ward",
    "patient_profile",
    "staff_ward_access",
    "communication_request",
    "request_event",
    "request_idempotency",
    "training_candidate",
}

STORAGE_UUID_PREVIOUS_REVISION = "bf490b4f7d1d"


def _alembic_config() -> Config:
    """프로젝트 루트를 기준으로 Alembic 설정을 생성한다."""

    return Config(str(PROJECT_ROOT / "alembic.ini"))


async def _drop_managed_tables(
    database_url: str,
) -> None:
    """??? DB?? ? ???? ??? ???? ????? ????."""

    engine = create_async_engine(database_url)

    try:
        async with engine.begin() as connection:
            actual_database = await connection.scalar(
                text("SELECT current_database()")
            )

            if (
                actual_database
                != make_url(database_url).database
            ):
                raise RuntimeError(
                    "??? DB? TEST_DATABASE_URL? "
                    "???? ????."
                )

            # FK ?? -> ?? ??.
            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "request_idempotency CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "request_event CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "communication_request CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "staff_ward_access CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "patient_profile CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "ward CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "training_candidate CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "phrase_usage_stat CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "user_consent CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "video_asset CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "login_session CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "utterance CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "users CASCADE"
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "phrase CASCADE"
                )
            )

            await connection.execute(
                text(
                    'DROP TABLE IF EXISTS '
                    '"session" CASCADE'
                )
            )

            await connection.execute(
                text(
                    "DROP TABLE IF EXISTS "
                    "alembic_version"
                )
            )

    finally:
        await engine.dispose()


async def _schema_snapshot(
    database_url: str,
) -> dict[str, object]:
    """동기 Inspector 결과를 비교 가능한 값으로 반환한다."""

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
                    "video_asset_checks": {
                        constraint["name"]
                        for constraint in inspector.get_check_constraints("video_asset")
                    },
                    "phrase_usage_stat_checks": {
                        constraint["name"]
                        for constraint in inspector.get_check_constraints(
                            "phrase_usage_stat"
                        )
                    },
                    "users_uniques": {
                        constraint["name"]
                        for constraint in inspector.get_unique_constraints("users")
                    },
                    "phrase_uniques": {
                        constraint["name"]
                        for constraint in inspector.get_unique_constraints("phrase")
                    },
                    "video_asset_uniques": {
                        constraint["name"]
                        for constraint in inspector.get_unique_constraints(
                            "video_asset"
                        )
                    },
                    "session_indexes": {
                        index["name"]: tuple(index["column_names"])
                        for index in inspector.get_indexes("session")
                        if not index.get("duplicates_constraint")
                    },
                    "utterance_indexes": {
                        index["name"]: tuple(index["column_names"])
                        for index in inspector.get_indexes("utterance")
                        if not index.get("duplicates_constraint")
                    },
                    "video_asset_indexes": {
                        index["name"]: tuple(index["column_names"])
                        for index in inspector.get_indexes("video_asset")
                        if not index.get("duplicates_constraint")
                    },
                    "utterance_foreign_keys": {
                        constraint["name"]: (
                            constraint["referred_table"],
                            constraint["options"].get("ondelete"),
                        )
                        for constraint in inspector.get_foreign_keys("utterance")
                    },
                    "video_asset_foreign_keys": {
                        constraint["name"]: (
                            constraint["referred_table"],
                            constraint["options"].get("ondelete"),
                        )
                        for constraint in inspector.get_foreign_keys("video_asset")
                    },
                    "user_consent_foreign_keys": {
                        constraint["name"]: (
                            constraint["referred_table"],
                            constraint["options"].get("ondelete"),
                        )
                        for constraint in inspector.get_foreign_keys("user_consent")
                    },
                    "phrase_usage_stat_foreign_keys": {
                        constraint["name"]: (
                            constraint["referred_table"],
                            constraint["options"].get("ondelete"),
                        )
                        for constraint in inspector.get_foreign_keys(
                            "phrase_usage_stat"
                        )
                    },
                    "users_columns": {
                        column["name"]: {
                            "nullable": column["nullable"],
                            "type": str(column["type"]).upper(),
                        }
                        for column in inspector.get_columns("users")
                    },
                    "utterance_nullable": {
                        column["name"]: column["nullable"]
                        for column in inspector.get_columns("utterance")
                    },
                }

            return await connection.run_sync(inspect_schema)
    finally:
        await engine.dispose()


async def _table_names(
    database_url: str,
) -> set[str]:
    """현재 PostgreSQL schema의 테이블 이름을 조회한다."""

    engine = create_async_engine(database_url)

    try:
        async with engine.connect() as connection:
            return await connection.run_sync(
                lambda sync_connection: set(inspect(sync_connection).get_table_names())
            )
    finally:
        await engine.dispose()


async def _insert_legacy_users(
    database_url: str,
) -> tuple[int, int]:
    """storage_uuid가 없던 revision에 기존 사용자 두 명을 만든다."""

    engine = create_async_engine(database_url)

    try:
        async with engine.begin() as connection:
            first_user_id = await connection.scalar(
                text(
                    """
                    INSERT INTO users (
                        username,
                        password_hash,
                        display_name
                    )
                    VALUES (
                        :username,
                        :password_hash,
                        :display_name
                    )
                    RETURNING user_id
                    """
                ),
                {
                    "username": "legacy-user-1",
                    "password_hash": "hash-1",
                    "display_name": "Legacy 1",
                },
            )

            second_user_id = await connection.scalar(
                text(
                    """
                    INSERT INTO users (
                        username,
                        password_hash,
                        display_name
                    )
                    VALUES (
                        :username,
                        :password_hash,
                        :display_name
                    )
                    RETURNING user_id
                    """
                ),
                {
                    "username": "legacy-user-2",
                    "password_hash": "hash-2",
                    "display_name": "Legacy 2",
                },
            )

            assert first_user_id is not None
            assert second_user_id is not None

            return (
                first_user_id,
                second_user_id,
            )
    finally:
        await engine.dispose()


async def _read_storage_uuids(
    database_url: str,
) -> list[tuple[int, object]]:
    """migration 이후 기존 사용자에게 할당된 storage UUID를 읽는다."""

    engine = create_async_engine(database_url)

    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text(
                    """
                    SELECT
                        user_id,
                        storage_uuid
                    FROM users
                    ORDER BY user_id
                    """
                )
            )

            return list(result.all())
    finally:
        await engine.dispose()


async def _user_column_names(
    database_url: str,
) -> set[str]:
    """users 테이블의 현재 컬럼 이름을 반환한다."""

    engine = create_async_engine(database_url)

    try:
        async with engine.connect() as connection:
            return await connection.run_sync(
                lambda sync_connection: {
                    column["name"]
                    for column in inspect(sync_connection).get_columns("users")
                }
            )
    finally:
        await engine.dispose()


def test_empty_database_upgrade_downgrade_upgrade_and_metadata_parity(
    postgres_url,
    monkeypatch,
):
    """빈 DB 왕복과 metadata/migration 일치를 검증한다."""

    # env.py가 Settings를 통해 URL을 읽으므로
    # 테스트 전용 PostgreSQL URL로 고정한다.
    monkeypatch.setenv(
        "DATABASE_URL",
        postgres_url,
    )

    alembic_config = _alembic_config()

    asyncio.run(_drop_managed_tables(postgres_url))

    try:
        assert MANAGED_TABLES.isdisjoint(asyncio.run(_table_names(postgres_url)))

        command.upgrade(
            alembic_config,
            "head",
        )

        first_upgrade = asyncio.run(_schema_snapshot(postgres_url))

        assert MANAGED_TABLES <= first_upgrade["tables"]

        command.downgrade(
            alembic_config,
            "base",
        )

        assert MANAGED_TABLES.isdisjoint(asyncio.run(_table_names(postgres_url)))

        command.upgrade(
            alembic_config,
            "head",
        )

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

        assert second_upgrade["video_asset_checks"] == {
            "ck_video_asset_storage_purpose",
            "ck_video_asset_storage_status",
        }

        assert second_upgrade["phrase_usage_stat_checks"] == {
            "ck_phrase_usage_stat_accepted_count_nonnegative",
            "ck_phrase_usage_stat_corrected_count_nonnegative",
            "ck_phrase_usage_stat_usage_count_nonnegative",
        }

        assert second_upgrade["users_uniques"] == {
            "uq_users_storage_uuid",
            "uq_users_username",
        }

        assert second_upgrade["phrase_uniques"] == {
            "uq_phrase_phrase_code",
        }

        assert second_upgrade["video_asset_uniques"] == {
            "uq_video_asset_user_id_idempotency_key"
        }

        assert second_upgrade["session_indexes"] == {
            "ix_session_started_at": ("started_at",)
        }

        assert second_upgrade["utterance_indexes"] == {
            "ix_utterance_phrase_id": ("phrase_id",),
            "ix_utterance_session_created_at": (
                "session_id",
                "created_at",
            ),
            "ix_utterance_user_created_at": (
                "user_id",
                "created_at",
            ),
        }

        assert second_upgrade["video_asset_indexes"] == {
            "ix_video_asset_retention_until": ("retention_until",),
            "ix_video_asset_user_created_at": (
                "user_id",
                "created_at",
            ),
        }

        assert second_upgrade["utterance_foreign_keys"] == {
            "fk_utterance_phrase_id_phrase": (
                "phrase",
                "SET NULL",
            ),
            "fk_utterance_session_id_session": (
                "session",
                "CASCADE",
            ),
            "fk_utterance_user_id_users": (
                "users",
                "CASCADE",
            ),
        }

        assert second_upgrade["video_asset_foreign_keys"] == {
            "fk_video_asset_user_id_users": (
                "users",
                "CASCADE",
            ),
            "fk_video_asset_utterance_id_utterance": (
                "utterance",
                "CASCADE",
            ),
        }

        assert second_upgrade["user_consent_foreign_keys"] == {
            "fk_user_consent_user_id_users": (
                "users",
                "CASCADE",
            ),
        }

        assert second_upgrade["phrase_usage_stat_foreign_keys"] == {
            "fk_phrase_usage_stat_phrase_code_phrase": (
                "phrase",
                "CASCADE",
            ),
            "fk_phrase_usage_stat_user_id_users": (
                "users",
                "CASCADE",
            ),
        }

        users_columns = second_upgrade["users_columns"]

        assert users_columns["storage_uuid"]["nullable"] is False

        assert users_columns["storage_uuid"]["type"] == "UUID"

        # 비동기 영상 업로드에서는 추론 전에
        # utterance를 생성할 수 있어야 한다.
        utterance_nullable = second_upgrade["utterance_nullable"]

        assert utterance_nullable["user_id"] is True
        assert utterance_nullable["session_id"] is True
        assert utterance_nullable["raw_text"] is True
        assert utterance_nullable["model_version"] is True

        # CI의 alembic check와 같은 방식으로
        # migration과 SQLAlchemy metadata drift를 검증한다.
        command.check(alembic_config)
    finally:
        asyncio.run(_drop_managed_tables(postgres_url))


def test_storage_uuid_migration_backfills_existing_users(
    postgres_url,
    monkeypatch,
):
    """기존 사용자도 migration 시 서로 다른 storage UUID를 받는다."""

    monkeypatch.setenv(
        "DATABASE_URL",
        postgres_url,
    )

    alembic_config = _alembic_config()

    asyncio.run(_drop_managed_tables(postgres_url))

    try:
        command.upgrade(
            alembic_config,
            STORAGE_UUID_PREVIOUS_REVISION,
        )

        before_columns = asyncio.run(_user_column_names(postgres_url))

        assert "storage_uuid" not in before_columns

        first_user_id, second_user_id = asyncio.run(_insert_legacy_users(postgres_url))

        command.upgrade(
            alembic_config,
            "head",
        )

        rows = asyncio.run(_read_storage_uuids(postgres_url))

        assert len(rows) == 2

        storage_uuids = {user_id: storage_uuid for user_id, storage_uuid in rows}

        assert first_user_id in storage_uuids
        assert second_user_id in storage_uuids

        first_uuid = storage_uuids[first_user_id]
        second_uuid = storage_uuids[second_user_id]

        assert first_uuid is not None
        assert second_uuid is not None

        UUID(str(first_uuid))
        UUID(str(second_uuid))

        assert first_uuid != second_uuid

        command.downgrade(
            alembic_config,
            STORAGE_UUID_PREVIOUS_REVISION,
        )

        after_downgrade_columns = asyncio.run(_user_column_names(postgres_url))

        assert "storage_uuid" not in after_downgrade_columns
    finally:
        asyncio.run(_drop_managed_tables(postgres_url))


def test_alembic_command_does_not_disable_application_loggers(
    postgres_url,
    monkeypatch,
):
    """migration 실행 전후 애플리케이션 logger를 비활성화하지 않는다."""

    monkeypatch.setenv(
        "DATABASE_URL",
        postgres_url,
    )

    application_logger = logging.getLogger("src.backend.recognition.api")

    original_disabled = application_logger.disabled
    application_logger.disabled = False

    try:
        command.current(_alembic_config())

        assert application_logger.disabled is False
    finally:
        application_logger.disabled = original_disabled
