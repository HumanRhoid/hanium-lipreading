"""회원가입, 로그인 및 로그인 세션 비즈니스 로직을 검증한다."""

from datetime import datetime, timezone

import pytest
from pwdlib import PasswordHash

from src.backend.auth.adapters.repository import LoginSession, User
from src.backend.auth.service import (
    AuthService,
    InvalidCredentialsError,
    InvalidSessionError,
    UsernameAlreadyExistsError,
)


class FakeAuthRepository:
    """DB 없이 AuthService를 테스트하기 위한 메모리 repository."""

    def __init__(self) -> None:
        self.users: dict[str, User] = {}
        self.sessions: dict[str, LoginSession] = {}
        self.next_user_id = 1
        self.next_session_id = 1

    async def get_user_by_username(
        self,
        username: str,
    ) -> User | None:
        return self.users.get(username)

    async def create_user(
        self,
        *,
        username: str,
        password_hash: str,
        display_name: str,
    ) -> User:
        user = User(
            user_id=self.next_user_id,
            username=username,
            password_hash=password_hash,
            display_name=display_name,
        )

        self.next_user_id += 1
        self.users[username] = user

        return user

    async def create_login_session(
        self,
        *,
        user_id: int,
        session_token: str,
        expires_at: datetime,
    ) -> LoginSession:
        login_session = LoginSession(
            login_session_id=self.next_session_id,
            user_id=user_id,
            session_token=session_token,
            expires_at=expires_at,
            revoked_at=None,
        )

        self.next_session_id += 1
        self.sessions[session_token] = login_session

        return login_session

    async def get_user_by_active_session_token(
        self,
        session_token: str,
    ) -> User | None:
        login_session = self.sessions.get(session_token)

        if login_session is None:
            return None

        if login_session.revoked_at is not None:
            return None

        if login_session.expires_at <= datetime.now(timezone.utc):
            return None

        return next(
            (
                user
                for user in self.users.values()
                if user.user_id == login_session.user_id
            ),
            None,
        )

    async def revoke_login_session(
        self,
        session_token: str,
    ) -> bool:
        login_session = self.sessions.get(session_token)

        if login_session is None or login_session.revoked_at is not None:
            return False

        login_session.revoked_at = datetime.now(timezone.utc)

        return True


@pytest.fixture
def repository() -> FakeAuthRepository:
    return FakeAuthRepository()


@pytest.fixture
def auth_service(
    repository: FakeAuthRepository,
) -> AuthService:
    return AuthService(repository=repository)


async def test_signup_creates_user_with_hashed_password(
    auth_service: AuthService,
    repository: FakeAuthRepository,
):
    user = await auth_service.signup(
        username="testuser1",
        password="test12345",
        display_name="테스트사용자",
    )

    assert user.user_id == 1
    assert user.username == "testuser1"
    assert user.display_name == "테스트사용자"

    stored_user = repository.users["testuser1"]

    assert stored_user.password_hash != "test12345"
    assert PasswordHash.recommended().verify(
        "test12345",
        stored_user.password_hash,
    )


async def test_signup_rejects_duplicate_username(
    auth_service: AuthService,
):
    await auth_service.signup(
        username="testuser1",
        password="test12345",
        display_name="테스트사용자",
    )

    with pytest.raises(UsernameAlreadyExistsError):
        await auth_service.signup(
            username="testuser1",
            password="another123",
            display_name="다른사용자",
        )


async def test_login_creates_session(
    auth_service: AuthService,
    repository: FakeAuthRepository,
):
    user = await auth_service.signup(
        username="testuser1",
        password="test12345",
        display_name="테스트사용자",
    )

    login_session = await auth_service.login(
        username="testuser1",
        password="test12345",
    )

    assert login_session.user_id == user.user_id
    assert login_session.session_token
    assert login_session.revoked_at is None
    assert login_session.expires_at > datetime.now(timezone.utc)
    assert login_session.session_token in repository.sessions


async def test_login_rejects_wrong_password(
    auth_service: AuthService,
):
    await auth_service.signup(
        username="testuser1",
        password="test12345",
        display_name="테스트사용자",
    )

    with pytest.raises(InvalidCredentialsError):
        await auth_service.login(
            username="testuser1",
            password="wrongpassword",
        )


async def test_current_user_is_found_by_session_token(
    auth_service: AuthService,
):
    user = await auth_service.signup(
        username="testuser1",
        password="test12345",
        display_name="테스트사용자",
    )

    login_session = await auth_service.login(
        username="testuser1",
        password="test12345",
    )

    current_user = await auth_service.get_current_user(login_session.session_token)

    assert current_user.user_id == user.user_id
    assert current_user.username == "testuser1"
    assert current_user.display_name == "테스트사용자"


async def test_logout_revokes_session(
    auth_service: AuthService,
    repository: FakeAuthRepository,
):
    await auth_service.signup(
        username="testuser1",
        password="test12345",
        display_name="테스트사용자",
    )

    login_session = await auth_service.login(
        username="testuser1",
        password="test12345",
    )

    session_token = login_session.session_token

    await auth_service.logout(session_token)

    assert repository.sessions[session_token].revoked_at is not None

    with pytest.raises(InvalidSessionError):
        await auth_service.get_current_user(session_token)
