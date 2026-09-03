"""Dashboard summary 비즈니스 규칙을 검증한다."""

from datetime import UTC, date, datetime
from types import SimpleNamespace

from src.backend.dashboard.adapters.repository import (
    DashboardCountsRecord,
    RequestSummaryRecord,
)
from src.backend.dashboard.service import DashboardService


NOW = datetime(2026, 9, 2, 8, 42, tzinfo=UTC)


class FakeRepository:
    async def staff_has_ward_access(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
    ) -> bool:
        return True

    async def get_ward(
        self,
        *,
        ward_code: str,
    ):
        return SimpleNamespace(
            ward_code=ward_code,
            ward_name="3병동",
        )

    async def get_dashboard_counts(self, **_kwargs):
        return DashboardCountsRecord(
            patients_registered_today=12,
            requests_today=27,
            unacknowledged_requests=3,
            critical_open_requests=1,
        )

    async def list_recent_requests(
        self,
        *,
        ward_code: str,
        limit: int,
    ):
        assert ward_code == "WARD-3"
        assert limit == 5

        return [
            RequestSummaryRecord(
                request_id=1042,
                patient_id=302,
                patient_code="P-2026-0302",
                patient_display_name="김민수",
                ward_code="WARD-3",
                ward_name="3병동",
                room_number="302",
                utterance_id=8801,
                text="자세를 바꿔 주세요",
                phrase_code="REQUEST_REPOSITION",
                category="REQUEST",
                confidence=0.88,
                priority="NORMAL",
                status="NEW",
                requested_at=datetime(
                    2026,
                    9,
                    2,
                    8,
                    40,
                    tzinfo=UTC,
                ),
                acknowledged_at=None,
                acknowledged_by_user_id=None,
                acknowledged_by_display_name=None,
                completed_at=None,
                completed_by_user_id=None,
                completed_by_display_name=None,
            )
        ]


async def test_summary_matches_dashboard_contract():
    service = DashboardService(FakeRepository())

    result = await service.get_summary(
        staff_user_id=51,
        ward_code="WARD-3",
        target_date=date(2026, 9, 2),
        recent_limit=5,
        now=NOW,
    )

    assert result.timezone == "Asia/Seoul"
    assert result.ward.ward_code == "WARD-3"
    assert result.ward.ward_name == "3병동"

    assert result.counts.patients_registered_today == 12
    assert result.counts.requests_today == 27
    assert result.counts.unacknowledged_requests == 3
    assert result.counts.critical_open_requests == 1

    assert len(result.recent_requests) == 1

    item = result.recent_requests[0]

    assert item.patient.masked_name == "김OO"
    assert item.unacknowledged_seconds == 120
    assert item.acknowledged_by is None
    assert item.completed_by is None


async def test_non_new_request_has_no_unacknowledged_seconds():
    record = RequestSummaryRecord(
        request_id=1,
        patient_id=1,
        patient_code="P-1",
        patient_display_name="홍길동",
        ward_code="WARD-3",
        ward_name="3병동",
        room_number="301",
        utterance_id=1,
        text="물 주세요",
        phrase_code="REQUEST_WATER",
        category="REQUEST",
        confidence=0.9,
        priority="NORMAL",
        status="ACKNOWLEDGED",
        requested_at=datetime(2026, 9, 2, 8, 30, tzinfo=UTC),
        acknowledged_at=datetime(2026, 9, 2, 8, 31, tzinfo=UTC),
        acknowledged_by_user_id=51,
        acknowledged_by_display_name="이간호사",
        completed_at=None,
        completed_by_user_id=None,
        completed_by_display_name=None,
    )

    item = DashboardService._to_request_summary(
        record,
        generated_at=NOW,
    )

    assert item.unacknowledged_seconds is None
    assert item.acknowledged_by is not None
    assert item.acknowledged_by.user_id == 51



async def test_summary_non_numeric_recent_limit_returns_400():
    from types import SimpleNamespace

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from src.backend.dashboard.api import (
        DashboardAPIError,
        dashboard_api_error_handler,
        router,
    )

    class FakeAuthService:
        async def get_current_user(self, _token: str):
            return SimpleNamespace(
                user_id=51,
                role="STAFF",
            )

    class FakeDashboardService:
        async def get_summary(self, **_kwargs):
            raise AssertionError(
                "service must not be called for invalid recent_limit"
            )

    app = FastAPI()

    app.add_exception_handler(
        DashboardAPIError,
        dashboard_api_error_handler,
    )

    app.include_router(router)

    app.state.auth_service = FakeAuthService()
    app.state.dashboard_service = FakeDashboardService()

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            (
                "/api/v1/dashboard/summary"
                "?ward_code=WARD-3"
                "&recent_limit=abc"
            ),
            headers={
                "X-Session-Token": "staff-session",
            },
        )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"
