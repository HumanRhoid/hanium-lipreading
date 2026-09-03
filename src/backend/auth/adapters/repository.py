"""사용자 및 로그인 세션 SQLAlchemy 모델과 PostgreSQL repository."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from src.backend.core.database import Base


class User(Base):
    """사용자 계정 정보를 저장하는 모델."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("username"),
        UniqueConstraint("storage_uuid"),
        CheckConstraint(
            "role IN ('PATIENT', 'STAFF')",
            name="role",
        ),
    )

    user_id: Mapped[int] = mapped_column(
        Integer,
        Identity(),
        primary_key=True,
    )

    storage_uuid: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        default=uuid4,
    )

    username: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    display_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="PATIENT",
        server_default="PATIENT",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class LoginSession(Base):
    """로그인 상태를 저장하는 세션 모델."""

    __tablename__ = "login_session"
    __table_args__ = (UniqueConstraint("session_token"),)

    login_session_id: Mapped[int] = mapped_column(
        Integer,
        Identity(),
        primary_key=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )

    session_token: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class SQLAlchemyAuthRepository:
    """사용자 및 로그인 세션 DB 접근을 담당한다."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def get_user_by_username(
        self,
        username: str,
    ) -> User | None:
        """username으로 사용자를 조회한다."""

        async with self._session_factory() as session:
            result = await session.execute(
                select(User).where(
                    User.username == username,
                )
            )
            return result.scalar_one_or_none()

    async def create_user(
        self,
        *,
        username: str,
        password_hash: str,
        display_name: str,
    ) -> User:
        """새 사용자를 생성한다."""

        user = User(
            username=username,
            password_hash=password_hash,
            display_name=display_name,
        )

        async with self._session_factory.begin() as session:
            session.add(user)
            await session.flush()
            await session.refresh(user)

        return user

    async def create_login_session(
        self,
        *,
        user_id: int,
        session_token: str,
        expires_at: datetime,
    ) -> LoginSession:
        """새 로그인 세션을 생성한다."""

        login_session = LoginSession(
            user_id=user_id,
            session_token=session_token,
            expires_at=expires_at,
        )

        async with self._session_factory.begin() as session:
            session.add(login_session)
            await session.flush()
            await session.refresh(login_session)

        return login_session

    async def get_user_by_active_session_token(
        self,
        session_token: str,
    ) -> User | None:
        """유효한 로그인 세션 토큰으로 현재 사용자를 조회한다."""

        async with self._session_factory() as session:
            result = await session.execute(
                select(User)
                .join(
                    LoginSession,
                    LoginSession.user_id == User.user_id,
                )
                .where(
                    LoginSession.session_token == session_token,
                    LoginSession.revoked_at.is_(None),
                    LoginSession.expires_at > func.now(),
                )
            )
            return result.scalar_one_or_none()

    async def revoke_login_session(
        self,
        session_token: str,
    ) -> bool:
        """로그인 세션을 무효화한다."""

        async with self._session_factory.begin() as session:
            result = await session.execute(
                select(LoginSession).where(
                    LoginSession.session_token == session_token,
                    LoginSession.revoked_at.is_(None),
                )
            )

            login_session = result.scalar_one_or_none()

            if login_session is None:
                return False

            login_session.revoked_at = datetime.now(timezone.utc)
            return True
