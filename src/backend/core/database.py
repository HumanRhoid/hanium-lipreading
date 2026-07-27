"""도메인이 공유하는 SQLAlchemy metadata와 비동기 engine."""

from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.backend.core.config import Settings

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """제약조건 이름을 안정적으로 생성하는 선언형 Base."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class SQLAlchemyDatabase:
    """요청 간 공유하는 engine과 짧은 DB session factory를 소유한다."""

    def __init__(self, settings: Settings):
        self.engine: AsyncEngine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            hide_parameters=True,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout=settings.database_pool_timeout_seconds,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def ping(self) -> None:
        """연결을 짧게 빌려 PostgreSQL 응답을 확인한다."""

        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def close(self) -> None:
        """engine pool의 모든 연결을 반환한다."""

        await self.engine.dispose()
