"""의료진 대시보드 공통 인증 및 병동 접근 권한을 검증한다."""

from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from src.backend.auth.service import InvalidSessionError
from src.backend.dashboard.api import (
    DashboardAPIError,
    dashboard_api_error_handler,
    require_staff_user,
    require_staff_ward_access,
)
from src.backend.dashboard.service import ResourceNotFoundError

SESSION_TOKEN = "dashboard-test-session"


class FakeAuthService:
    def __init__(
        self,
        *,
        role: str = "STAFF",
        invalid_session: bool = False,
    ) -> None:
        self.role = role
        self.invalid_session = invalid_session
        self.received_tokens: list[str] = []

    async def get_current_user(
        self,
        session_token: str,
    ):
        self.received_tokens.append(session_token)

        if self.invalid_session:
            raise InvalidSessionError

        return SimpleNamespace(
            user_id=7,
            role=self.role,
        )


class FakeDashboardService:
    def __init__(
        self,
        *,
        allowed: bool = True,
    ) -> None:
        self.allowed = allowed
        self.calls: list[dict[str, object]] = []

    async def require_ward_access(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
    ) -> None:
        self.calls.append(
            {
                "staff_user_id": staff_user_id,
                "ward_code": ward_code,
            }
        )

        if not self.allowed:
            raise ResourceNotFoundError


def make_app(
    *,
    auth_service: FakeAuthService | None = None,
    dashboard_service: FakeDashboardService | None = None,
) -> FastAPI:
    app = FastAPI()

    app.add_exception_handler(
        DashboardAPIError,
        dashboard_api_error_handler,
    )

    test_router = APIRouter()

    @test_router.get("/staff-only")
    async def staff_only(
        user=Depends(require_staff_user),
    ):
        return {
            "user_id": user.user_id,
        }

    @test_router.get("/ward-only")
    async def ward_only(
        user=Depends(require_staff_ward_access),
    ):
        return {
            "user_id": user.user_id,
        }

    app.include_router(test_router)

    app.state.auth_service = (
        auth_service
        if auth_service is not None
        else FakeAuthService()
    )

    app.state.dashboard_service = (
        dashboard_service
        if dashboard_service is not None
        else FakeDashboardService()
    )

    return app


async def get(
    app: FastAPI,
    path: str,
    *,
    session_token: str | None = SESSION_TOKEN,
):
    headers = {}

    if session_token is not None:
        headers["X-Session-Token"] = session_token

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get(
            path,
            headers=headers,
        )


async def test_missing_session_returns_invalid_session_401():
    response = await get(
        make_app(),
        "/staff-only",
        session_token=None,
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "유효한 로그인 세션이 필요합니다.",
        "code": "INVALID_SESSION",
    }


async def test_blank_session_returns_invalid_session_401():
    response = await get(
        make_app(),
        "/staff-only",
        session_token="   ",
    )

    assert response.status_code == 401
    assert response.json()["code"] == "INVALID_SESSION"


async def test_invalid_session_returns_invalid_session_401():
    auth_service = FakeAuthService(
        invalid_session=True,
    )

    response = await get(
        make_app(auth_service=auth_service),
        "/staff-only",
    )

    assert response.status_code == 401
    assert response.json()["code"] == "INVALID_SESSION"


async def test_patient_cannot_access_staff_dashboard():
    response = await get(
        make_app(
            auth_service=FakeAuthService(
                role="PATIENT",
            )
        ),
        "/staff-only",
    )

    assert response.status_code == 403
    assert response.json() == {
        "detail": "의료진 권한이 필요합니다.",
        "code": "STAFF_ACCESS_REQUIRED",
    }


async def test_staff_can_pass_staff_authorization():
    response = await get(
        make_app(),
        "/staff-only",
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 7,
    }


async def test_staff_without_ward_access_gets_non_disclosing_404():
    dashboard_service = FakeDashboardService(
        allowed=False,
    )

    response = await get(
        make_app(
            dashboard_service=dashboard_service,
        ),
        "/ward-only?ward_code=WARD-A",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "자원을 찾을 수 없습니다.",
        "code": "RESOURCE_NOT_FOUND",
    }

    assert dashboard_service.calls == [
        {
            "staff_user_id": 7,
            "ward_code": "WARD-A",
        }
    ]


async def test_staff_with_ward_access_can_continue():
    dashboard_service = FakeDashboardService(
        allowed=True,
    )

    response = await get(
        make_app(
            dashboard_service=dashboard_service,
        ),
        "/ward-only?ward_code=WARD-A",
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": 7,
    }
