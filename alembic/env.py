"""비동기 PostgreSQL migration 실행 환경."""

from asyncio import run
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from src.backend.core import Base, Settings
from src.backend.recognition.adapters import repository  # noqa: F401

config = context.config
if config.config_file_name is not None:
    # Programmatic Alembic commands share the application/test process. Existing
    # loggers must keep working after migration configuration is loaded.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

settings = Settings()
config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """DB 연결 없이 SQL script를 생성한다."""

    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """동기 Alembic context에서 migration을 실행한다."""

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """asyncpg engine을 열어 동기 migration 함수를 연결한다."""

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """온라인 migration을 별도 event loop에서 실행한다."""

    run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
