"""회원가입, 로그인 및 로그인 세션 비즈니스 로직."""

from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe

from pwdlib import PasswordHash

from src.backend.auth.adapters.repository import (
    LoginSession,
    SQLAlchemyAuthRepository,
    User,
)

SESSION_TOKEN_BYTES = 32
DEFAULT_SESSION_TTL = timedelta(hours=12)


class UsernameAlreadyExistsError(Exception):
    """이미 사용 중인 username으로 회원가입을 요청한 경우."""


class InvalidCredentialsError(Exception):
    """아이디 또는 비밀번호가 올바르지 않은 경우."""


class InvalidSessionError(Exception):
    """로그인 세션이 없거나 만료 또는 로그아웃된 경우."""


class AuthService:
    """회원가입, 로그인, 로그인 상태 확인 및 로그아웃을 처리한다."""

    def __init__(
        self,
        repository: SQLAlchemyAuthRepository,
        session_ttl: timedelta = DEFAULT_SESSION_TTL,
    ) -> None:
        self._repository = repository
        self._session_ttl = session_ttl
        self._password_hash = PasswordHash.recommended()

    async def signup(
        self,
        *,
        username: str,
        password: str,
        name: str,
        hospital: str,
        ward: str | None,
    ) -> User:
        """새 의료진 회원을 생성한다."""

        existing_user = await self._repository.get_user_by_username(username)

        if existing_user is not None:
            raise UsernameAlreadyExistsError

        password_hash = self._password_hash.hash(password)

        return await self._repository.create_user(
            username=username,
            password_hash=password_hash,
            name=name,
            hospital=hospital,
            ward=ward,
        )

    async def login(
        self,
        *,
        username: str,
        password: str,
    ) -> LoginSession:
        """아이디와 비밀번호를 확인하고 로그인 세션을 생성한다."""

        user = await self._repository.get_user_by_username(username)

        if user is None:
            raise InvalidCredentialsError

        if not self._password_hash.verify(
            password,
            user.password_hash,
        ):
            raise InvalidCredentialsError

        session_token = token_urlsafe(SESSION_TOKEN_BYTES)
        expires_at = datetime.now(timezone.utc) + self._session_ttl

        return await self._repository.create_login_session(
            user_id=user.user_id,
            session_token=session_token,
            expires_at=expires_at,
        )

    async def get_current_user(
        self,
        session_token: str,
    ) -> User:
        """유효한 로그인 세션으로 현재 사용자를 조회한다."""

        user = await self._repository.get_user_by_active_session_token(session_token)

        if user is None:
            raise InvalidSessionError

        return user

    async def logout(
        self,
        session_token: str,
    ) -> None:
        """현재 로그인 세션을 무효화한다."""

        revoked = await self._repository.revoke_login_session(session_token)

        if not revoked:
            raise InvalidSessionError
