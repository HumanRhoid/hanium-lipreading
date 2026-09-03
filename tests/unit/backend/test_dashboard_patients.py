"""Patient board service and API contract tests."""

from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.backend.dashboard.adapters.repository import (
    PatientBoardRecord,
    PatientLatestRequestRecord,
)
from src.backend.dashboard.api import (
    DashboardAPIError,
    dashboard_api_error_handler,
    router,
)
from src.backend.dashboard.service import (
    DashboardService,
    ResourceNotFoundError,
)


NOW = datetime(
    2026,
    9,
    2,
    8,
    42,
    tzinfo=UTC,
)


def board_records():
    return [
        PatientBoardRecord(
            patient_id=301,
            patient_code="P-301",
            patient_display_name="Choi",
            room_number="301",
            open_request_count=0,
            unacknowledged_request_count=0,
            critical_open_count=0,
            latest_request=None,
        ),
        PatientBoardRecord(
            patient_id=302,
            patient_code="P-302",
            patient_display_name="Kim",
            room_number="302",
            open_request_count=1,
            unacknowledged_request_count=1,
            critical_open_count=0,
            latest_request=PatientLatestRequestRecord(
                request_id=1042,
                text="Water please",
                status="NEW",
                priority="NORMAL",
                requested_at=datetime(
                    2026,
                    9,
                    2,
                    8,
                    40,
                    tzinfo=UTC,
                ),
            ),
        ),
        PatientBoardRecord(
            patient_id=303,
            patient_code="P-303",
            patient_display_name="Park",
            room_number="303",
            open_request_count=2,
            unacknowledged_request_count=1,
            critical_open_count=1,
            latest_request=PatientLatestRequestRecord(
                request_id=1043,
                text="Help",
                status="NEW",
                priority="CRITICAL",
                requested_at=datetime(
                    2026,
                    9,
                    2,
                    8,
                    41,
                    tzinfo=UTC,
                ),
            ),
        ),
    ]


class FakeRepository:
    def __init__(self, *, access=True):
        self.access = access
        self.calls = []

    async def staff_has_ward_access(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
    ) -> bool:
        return self.access

    async def get_ward(self, *, ward_code: str):
        return SimpleNamespace(
            ward_code=ward_code,
            ward_name="Ward 3",
        )

    async def list_patient_board(
        self,
        *,
        ward_code: str,
    ):
        self.calls.append(ward_code)
        return board_records()


async def test_patient_board_status_precedence():
    service = DashboardService(FakeRepository())

    result = await service.get_patient_board(
        staff_user_id=51,
        ward_code="WARD-3",
        board_status=None,
        now=NOW,
    )

    assert [
        patient.board_status
        for patient in result.patients
    ] == [
        "GREEN",
        "YELLOW",
        "RED",
    ]

    assert result.patients[0].latest_request is None

    assert (
        result.patients[1].latest_request.request_id
        == 1042
    )

    assert result.patients[2].board_status == "RED"


async def test_patient_board_status_filter():
    service = DashboardService(FakeRepository())

    result = await service.get_patient_board(
        staff_user_id=51,
        ward_code="WARD-3",
        board_status="YELLOW",
        now=NOW,
    )

    assert len(result.patients) == 1
    assert result.patients[0].patient_id == 302
    assert result.patients[0].board_status == "YELLOW"


class FakeAuthService:
    async def get_current_user(self, _token: str):
        return SimpleNamespace(
            user_id=51,
            role="STAFF",
        )


class FakeDashboardService:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    async def get_patient_board(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return {
            "ward": {
                "ward_code": "WARD-3",
                "ward_name": "Ward 3",
            },
            "generated_at": NOW,
            "patients": [],
        }


def make_app(*, dashboard_service=None):
    app = FastAPI()

    app.add_exception_handler(
        DashboardAPIError,
        dashboard_api_error_handler,
    )

    app.include_router(router)

    app.state.auth_service = FakeAuthService()
    app.state.dashboard_service = (
        dashboard_service
        if dashboard_service is not None
        else FakeDashboardService()
    )

    return app


async def api_get(query: str):
    app = make_app()

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            f"/api/v1/patients{query}",
            headers={
                "X-Session-Token": "staff-session",
            },
        )

    return response, app


async def test_patient_board_api_default_query():
    response, app = await api_get(
        "?ward_code=WARD-3"
    )

    assert response.status_code == 200

    call = app.state.dashboard_service.calls[0]

    assert call["ward_code"] == "WARD-3"
    assert call["board_status"] is None


async def test_patient_board_api_normalizes_status():
    response, app = await api_get(
        "?ward_code=WARD-3&status=red"
    )

    assert response.status_code == 200

    call = app.state.dashboard_service.calls[0]

    assert call["board_status"] == "RED"


async def test_patient_board_invalid_status_returns_400():
    response, _app = await api_get(
        "?ward_code=WARD-3&status=BLUE"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_patient_board_missing_ward_returns_400():
    response, _app = await api_get("")

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_patient_board_inaccessible_ward_returns_404():
    app = make_app(
        dashboard_service=FakeDashboardService(
            error=ResourceNotFoundError(),
        )
    )

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/api/v1/patients?ward_code=WARD-X",
            headers={
                "X-Session-Token": "staff-session",
            },
        )

    assert response.status_code == 404
    assert response.json()["code"] == "RESOURCE_NOT_FOUND"
