"""Patient detail dashboard contract tests."""

from datetime import UTC, date, datetime
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.backend.dashboard.adapters.repository import (
    FrequentPhraseRecord,
    PatientDetailStatsRecord,
    PatientLatestRequestRecord,
    PatientProfileDetailRecord,
    TodayPhraseCountRecord,
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


def profile_record():
    return PatientProfileDetailRecord(
        patient_id=302,
        patient_code="P-2026-0302",
        patient_display_name="Kim",
        ward_code="WARD-3",
        ward_name="Ward 3",
        room_number="302",
        admitted_on=date(2026, 8, 28),
        communication_status="VOICE_DIFFICULT",
        assistive_method="LIP_READING",
        notes="Night pain",
    )


def stats_record():
    latest = PatientLatestRequestRecord(
        request_id=1042,
        text="Reposition please",
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
    )

    return PatientDetailStatsRecord(
        open_request_count=1,
        unacknowledged_request_count=1,
        latest_request=latest,
        frequent_phrases=[
            FrequentPhraseRecord(
                phrase_code="REQUEST_REPOSITION",
                text="Reposition please",
                count_30d=14,
                last_used_at=datetime(
                    2026,
                    9,
                    2,
                    8,
                    40,
                    tzinfo=UTC,
                ),
            ),
            FrequentPhraseRecord(
                phrase_code="REQUEST_WATER",
                text="Water please",
                count_30d=9,
                last_used_at=datetime(
                    2026,
                    9,
                    2,
                    7,
                    0,
                    tzinfo=UTC,
                ),
            ),
        ],
        today_total_requests=8,
        today_by_category={
            "PAIN": 3,
            "REQUEST": 5,
            "REPLY": 0,
            "ETC": 0,
        },
        today_by_phrase=[
            TodayPhraseCountRecord(
                phrase_code="PAIN_GENERAL",
                text="Pain",
                count=3,
                last_used_at=datetime(
                    2026,
                    9,
                    2,
                    8,
                    30,
                    tzinfo=UTC,
                ),
            ),
            TodayPhraseCountRecord(
                phrase_code="REQUEST_REPOSITION",
                text="Reposition please",
                count=2,
                last_used_at=datetime(
                    2026,
                    9,
                    2,
                    8,
                    40,
                    tzinfo=UTC,
                ),
            ),
        ],
    )


class FakeRepository:
    def __init__(
        self,
        *,
        profile=None,
        access=True,
    ):
        self.profile = (
            profile
            if profile is not None
            else profile_record()
        )
        self.access = access
        self.stats_calls = []

    async def get_patient_profile_detail(
        self,
        *,
        patient_id: int,
    ):
        return self.profile

    async def staff_has_ward_access(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
    ) -> bool:
        return self.access

    async def get_patient_detail_stats(
        self,
        **kwargs,
    ):
        self.stats_calls.append(kwargs)
        return stats_record()


async def test_patient_detail_contract():
    repository = FakeRepository()
    service = DashboardService(repository)

    result = await service.get_patient_detail(
        staff_user_id=51,
        patient_id=302,
        now=NOW,
    )

    assert result.patient_id == 302
    assert result.patient_code == "P-2026-0302"
    assert result.masked_name == "KOO"
    assert result.ward.ward_code == "WARD-3"
    assert result.room_number == "302"

    assert (
        result.communication_status
        == "VOICE_DIFFICULT"
    )

    assert (
        result.communication_status_label
        == "\uc74c\uc131 \uc758\uc0ac\uc18c\ud1b5 \uc5b4\ub824\uc6c0"
    )

    assert result.open_request_count == 1
    assert result.unacknowledged_request_count == 1

    assert result.latest_request is not None
    assert result.latest_request.request_id == 1042

    assert [
        item.count_30d
        for item in result.frequent_phrases
    ] == [14, 9]

    assert result.today_summary.date == date(
        2026,
        9,
        2,
    )

    assert result.today_summary.total_requests == 8

    assert result.today_summary.by_category == {
        "PAIN": 3,
        "REQUEST": 5,
        "REPLY": 0,
        "ETC": 0,
    }


async def test_patient_detail_uses_seoul_day_boundaries():
    repository = FakeRepository()
    service = DashboardService(repository)

    await service.get_patient_detail(
        staff_user_id=51,
        patient_id=302,
        now=NOW,
    )

    call = repository.stats_calls[0]

    assert call["today_start_utc"] == datetime(
        2026,
        9,
        1,
        15,
        0,
        tzinfo=UTC,
    )

    assert call["today_end_utc"] == datetime(
        2026,
        9,
        2,
        15,
        0,
        tzinfo=UTC,
    )


async def test_patient_detail_missing_patient_is_404():
    repository = FakeRepository()
    repository.profile = None

    service = DashboardService(repository)

    try:
        await service.get_patient_detail(
            staff_user_id=51,
            patient_id=999,
            now=NOW,
        )
    except ResourceNotFoundError:
        pass
    else:
        raise AssertionError("ResourceNotFoundError expected")


async def test_patient_detail_inaccessible_patient_is_404():
    service = DashboardService(
        FakeRepository(access=False)
    )

    try:
        await service.get_patient_detail(
            staff_user_id=51,
            patient_id=302,
            now=NOW,
        )
    except ResourceNotFoundError:
        pass
    else:
        raise AssertionError("ResourceNotFoundError expected")


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

    async def get_patient_detail(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        return {
            "patient_id": 302,
            "patient_code": "P-2026-0302",
            "masked_name": "KOO",
            "ward": {
                "ward_code": "WARD-3",
                "ward_name": "Ward 3",
            },
            "room_number": "302",
            "admitted_on": "2026-08-28",
            "communication_status": "VOICE_DIFFICULT",
            "communication_status_label": (
                "\uc74c\uc131 \uc758\uc0ac\uc18c\ud1b5 "
                "\uc5b4\ub824\uc6c0"
            ),
            "assistive_method": "LIP_READING",
            "notes": "Night pain",
            "open_request_count": 1,
            "unacknowledged_request_count": 1,
            "latest_request": None,
            "frequent_phrases": [],
            "today_summary": {
                "date": "2026-09-02",
                "total_requests": 0,
                "by_category": {
                    "PAIN": 0,
                    "REQUEST": 0,
                    "REPLY": 0,
                    "ETC": 0,
                },
                "by_phrase": [],
            },
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


async def api_get(
    *,
    patient_id: int = 302,
    dashboard_service=None,
):
    app = make_app(
        dashboard_service=dashboard_service,
    )

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            f"/api/v1/patients/{patient_id}",
            headers={
                "X-Session-Token": "staff-session",
            },
        )

    return response, app


async def test_patient_detail_api_returns_200():
    response, app = await api_get()

    assert response.status_code == 200
    assert response.json()["patient_id"] == 302

    call = app.state.dashboard_service.calls[0]

    assert call["staff_user_id"] == 51
    assert call["patient_id"] == 302


async def test_patient_detail_api_maps_not_found_to_404():
    response, _app = await api_get(
        patient_id=999,
        dashboard_service=FakeDashboardService(
            error=ResourceNotFoundError(),
        ),
    )

    assert response.status_code == 404
    assert response.json()["code"] == "RESOURCE_NOT_FOUND"
