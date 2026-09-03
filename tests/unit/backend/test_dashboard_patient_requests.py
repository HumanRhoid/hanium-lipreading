"""Patient request history service/API contract tests."""

from datetime import UTC, date, datetime
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.backend.dashboard.adapters.repository import (
    PatientProfileDetailRecord,
    PatientRequestHistoryPageRecord,
    PatientRequestHistoryRecord,
)
from src.backend.dashboard.api import (
    DashboardAPIError,
    dashboard_api_error_handler,
    router,
)
from src.backend.dashboard.service import (
    DashboardService,
    InvalidDashboardQueryError,
    ResourceNotFoundError,
)


def history_record(
    *,
    request_id: int = 1042,
    requested_at: datetime | None = None,
):
    return PatientRequestHistoryRecord(
        request_id=request_id,
        utterance_id=8801,
        text="Reposition please",
        phrase_code="REQUEST_REPOSITION",
        category="REQUEST",
        confidence=0.88,
        priority="NORMAL",
        status="COMPLETED",
        requested_at=(
            requested_at
            or datetime(
                2026,
                9,
                2,
                8,
                40,
                tzinfo=UTC,
            )
        ),
        acknowledged_at=datetime(
            2026,
            9,
            2,
            8,
            41,
            tzinfo=UTC,
        ),
        acknowledged_by_user_id=51,
        acknowledged_by_display_name="Nurse Lee",
        completed_at=datetime(
            2026,
            9,
            2,
            8,
            44,
            tzinfo=UTC,
        ),
        completed_by_user_id=51,
        completed_by_display_name="Nurse Lee",
    )


class FakeRepository:
    def __init__(
        self,
        *,
        profile_exists=True,
        access=True,
        has_more=False,
    ):
        self.profile_exists = profile_exists
        self.access = access
        self.has_more = has_more
        self.calls = []

    async def get_patient_profile_detail(
        self,
        *,
        patient_id: int,
    ):
        if not self.profile_exists:
            return None

        return PatientProfileDetailRecord(
            patient_id=patient_id,
            patient_code="P-302",
            patient_display_name="Kim",
            ward_code="WARD-3",
            ward_name="Ward 3",
            room_number="302",
            admitted_on=date(2026, 8, 28),
            communication_status="VOICE_DIFFICULT",
            assistive_method="LIP_READING",
            notes=None,
        )

    async def staff_has_ward_access(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
    ):
        return self.access

    async def list_patient_requests(
        self,
        **kwargs,
    ):
        self.calls.append(kwargs)

        return PatientRequestHistoryPageRecord(
            items=[history_record()],
            has_more=self.has_more,
        )


async def test_patient_history_contract_and_seoul_dates():
    repository = FakeRepository()
    service = DashboardService(repository)

    result = await service.get_patient_requests(
        staff_user_id=51,
        patient_id=302,
        date_from=date(2026, 9, 1),
        date_to=date(2026, 9, 2),
        status_filter="COMPLETED",
        category="REQUEST",
        cursor=None,
        limit=20,
    )

    assert len(result.items) == 1

    item = result.items[0]

    assert item.request_id == 1042
    assert item.utterance_id == 8801
    assert item.status == "COMPLETED"

    assert item.acknowledged_by is not None
    assert item.acknowledged_by.user_id == 51

    assert item.completed_by is not None
    assert item.completed_by.user_id == 51

    call = repository.calls[0]

    assert call["date_from_utc"] == datetime(
        2026,
        8,
        31,
        15,
        0,
        tzinfo=UTC,
    )

    assert call["date_to_utc"] == datetime(
        2026,
        9,
        2,
        15,
        0,
        tzinfo=UTC,
    )


async def test_patient_history_generates_and_reuses_cursor():
    first_repository = FakeRepository(
        has_more=True
    )

    first_service = DashboardService(
        first_repository
    )

    first = await first_service.get_patient_requests(
        staff_user_id=51,
        patient_id=302,
        date_from=None,
        date_to=None,
        status_filter=None,
        category=None,
        cursor=None,
        limit=1,
    )

    assert first.next_cursor is not None

    second_repository = FakeRepository()

    await DashboardService(
        second_repository
    ).get_patient_requests(
        staff_user_id=51,
        patient_id=302,
        date_from=None,
        date_to=None,
        status_filter=None,
        category=None,
        cursor=first.next_cursor,
        limit=1,
    )

    call = second_repository.calls[0]

    assert call["cursor_request_id"] == 1042

    assert call["cursor_requested_at"] == datetime(
        2026,
        9,
        2,
        8,
        40,
        tzinfo=UTC,
    )


async def test_patient_history_invalid_date_range():
    service = DashboardService(
        FakeRepository()
    )

    try:
        await service.get_patient_requests(
            staff_user_id=51,
            patient_id=302,
            date_from=date(2026, 9, 3),
            date_to=date(2026, 9, 2),
            status_filter=None,
            category=None,
            cursor=None,
            limit=20,
        )
    except InvalidDashboardQueryError:
        pass
    else:
        raise AssertionError(
            "InvalidDashboardQueryError expected"
        )


async def test_patient_history_missing_patient_is_404():
    service = DashboardService(
        FakeRepository(
            profile_exists=False,
        )
    )

    try:
        await service.get_patient_requests(
            staff_user_id=51,
            patient_id=999,
            date_from=None,
            date_to=None,
            status_filter=None,
            category=None,
            cursor=None,
            limit=20,
        )
    except ResourceNotFoundError:
        pass
    else:
        raise AssertionError(
            "ResourceNotFoundError expected"
        )


async def test_patient_history_inaccessible_patient_is_404():
    service = DashboardService(
        FakeRepository(
            access=False,
        )
    )

    try:
        await service.get_patient_requests(
            staff_user_id=51,
            patient_id=302,
            date_from=None,
            date_to=None,
            status_filter=None,
            category=None,
            cursor=None,
            limit=20,
        )
    except ResourceNotFoundError:
        pass
    else:
        raise AssertionError(
            "ResourceNotFoundError expected"
        )


class FakeAuthService:
    async def get_current_user(
        self,
        _token: str,
    ):
        return SimpleNamespace(
            user_id=51,
            role="STAFF",
        )


class FakeDashboardService:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    async def get_patient_requests(
        self,
        **kwargs,
    ):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return {
            "items": [],
            "next_cursor": None,
        }


def make_app(
    *,
    dashboard_service=None,
):
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
            (
                "/api/v1/patients/302/requests"
                + query
            ),
            headers={
                "X-Session-Token": "staff-session",
            },
        )

    return response, app


async def test_patient_history_api_defaults():
    response, app = await api_get("")

    assert response.status_code == 200

    call = app.state.dashboard_service.calls[0]

    assert call["date_from"] is None
    assert call["date_to"] is None
    assert call["status_filter"] is None
    assert call["category"] is None
    assert call["limit"] == 20


async def test_patient_history_api_parses_filters():
    response, app = await api_get(
        "?date_from=2026-09-01"
        "&date_to=2026-09-02"
        "&status=completed"
        "&category=request"
        "&limit=50"
    )

    assert response.status_code == 200

    call = app.state.dashboard_service.calls[0]

    assert call["date_from"] == date(
        2026,
        9,
        1,
    )

    assert call["date_to"] == date(
        2026,
        9,
        2,
    )

    assert call["status_filter"] == "COMPLETED"
    assert call["category"] == "REQUEST"
    assert call["limit"] == 50


async def test_patient_history_invalid_date_returns_400():
    response, _app = await api_get(
        "?date_from=not-a-date"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_patient_history_reversed_dates_return_400():
    response, _app = await api_get(
        "?date_from=2026-09-03"
        "&date_to=2026-09-02"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_patient_history_invalid_status_returns_400():
    response, _app = await api_get(
        "?status=INVALID"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_patient_history_invalid_category_returns_400():
    response, _app = await api_get(
        "?category=UNKNOWN"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_patient_history_invalid_limit_returns_400():
    response, _app = await api_get(
        "?limit=abc"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"
