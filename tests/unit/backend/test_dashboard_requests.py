"""의료진 요청 목록 필터·정렬·cursor 계약을 검증한다."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.backend.dashboard.adapters.repository import (
    RequestPageRecord,
    RequestSummaryRecord,
)
from src.backend.dashboard.api import (
    DashboardAPIError,
    dashboard_api_error_handler,
    router,
)
from src.backend.dashboard.service import DashboardService


SESSION_TOKEN = "staff-session"


def make_record(
    *,
    request_id: int,
    status: str = "NEW",
    priority: str = "NORMAL",
    requested_at: datetime | None = None,
) -> RequestSummaryRecord:
    return RequestSummaryRecord(
        request_id=request_id,
        patient_id=302,
        patient_code="P-2026-0302",
        patient_display_name="김민수",
        ward_code="WARD-3",
        ward_name="3병동",
        room_number="302",
        utterance_id=8000 + request_id,
        text="물 주세요",
        phrase_code="REQUEST_WATER",
        category="REQUEST",
        confidence=0.9,
        priority=priority,
        status=status,
        requested_at=(
            requested_at
            or datetime(
                2026,
                9,
                2,
                8,
                request_id,
                tzinfo=UTC,
            )
        ),
        acknowledged_at=None,
        acknowledged_by_user_id=None,
        acknowledged_by_display_name=None,
        completed_at=None,
        completed_by_user_id=None,
        completed_by_display_name=None,
    )


class FakeRepository:
    def __init__(
        self,
        *,
        items=None,
        has_more: bool = False,
    ):
        self.items = (
            items
            if items is not None
            else [make_record(request_id=1)]
        )
        self.has_more = has_more
        self.calls = []

    async def staff_has_ward_access(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
    ) -> bool:
        return True

    async def list_requests(
        self,
        **kwargs,
    ):
        self.calls.append(kwargs)

        return RequestPageRecord(
            items=self.items,
            has_more=self.has_more,
        )


async def test_request_list_defaults_match_spec():
    repository = FakeRepository()
    service = DashboardService(repository)

    result = await service.get_requests(
        staff_user_id=51,
        ward_code="WARD-3",
        statuses=("NEW", "ACKNOWLEDGED"),
        priorities=None,
        category=None,
        patient_id=None,
        sort_mode="attention",
        cursor=None,
        limit=20,
        now=datetime(
            2026,
            9,
            2,
            8,
            5,
            tzinfo=UTC,
        ),
    )

    assert len(result.items) == 1
    assert result.next_cursor is None

    call = repository.calls[0]

    assert call["statuses"] == (
        "NEW",
        "ACKNOWLEDGED",
    )
    assert call["sort_mode"] == "attention"
    assert call["limit"] == 20


async def test_request_list_generates_and_reuses_newest_cursor():
    first_repository = FakeRepository(
        items=[
            make_record(
                request_id=9,
                requested_at=datetime(
                    2026,
                    9,
                    2,
                    8,
                    40,
                    tzinfo=UTC,
                ),
            )
        ],
        has_more=True,
    )

    first_service = DashboardService(
        first_repository
    )

    first = await first_service.get_requests(
        staff_user_id=51,
        ward_code="WARD-3",
        statuses=("NEW",),
        priorities=None,
        category=None,
        patient_id=None,
        sort_mode="newest",
        cursor=None,
        limit=1,
    )

    assert first.next_cursor is not None

    second_repository = FakeRepository()

    second_service = DashboardService(
        second_repository
    )

    await second_service.get_requests(
        staff_user_id=51,
        ward_code="WARD-3",
        statuses=("NEW",),
        priorities=None,
        category=None,
        patient_id=None,
        sort_mode="newest",
        cursor=first.next_cursor,
        limit=1,
    )

    call = second_repository.calls[0]

    assert call["cursor_rank"] is None
    assert call["cursor_request_id"] == 9
    assert (
        call["cursor_requested_at"]
        == datetime(
            2026,
            9,
            2,
            8,
            40,
            tzinfo=UTC,
        )
    )


async def test_attention_cursor_contains_priority_rank():
    repository = FakeRepository(
        items=[
            make_record(
                request_id=5,
                priority="CRITICAL",
            )
        ],
        has_more=True,
    )

    service = DashboardService(repository)

    first = await service.get_requests(
        staff_user_id=51,
        ward_code="WARD-3",
        statuses=("NEW",),
        priorities=None,
        category=None,
        patient_id=None,
        sort_mode="attention",
        cursor=None,
        limit=1,
    )

    second_repository = FakeRepository()

    await DashboardService(
        second_repository
    ).get_requests(
        staff_user_id=51,
        ward_code="WARD-3",
        statuses=("NEW",),
        priorities=None,
        category=None,
        patient_id=None,
        sort_mode="attention",
        cursor=first.next_cursor,
        limit=1,
    )

    assert (
        second_repository.calls[0]["cursor_rank"]
        == 0
    )


async def test_cursor_cannot_be_reused_with_different_sort():
    repository = FakeRepository(
        has_more=True,
    )

    service = DashboardService(repository)

    first = await service.get_requests(
        staff_user_id=51,
        ward_code="WARD-3",
        statuses=("NEW",),
        priorities=None,
        category=None,
        patient_id=None,
        sort_mode="newest",
        cursor=None,
        limit=1,
    )

    with pytest.raises(Exception):
        await service.get_requests(
            staff_user_id=51,
            ward_code="WARD-3",
            statuses=("NEW",),
            priorities=None,
            category=None,
            patient_id=None,
            sort_mode="attention",
            cursor=first.next_cursor,
            limit=1,
        )


class FakeAuthService:
    async def get_current_user(
        self,
        _session_token: str,
    ):
        return SimpleNamespace(
            user_id=51,
            role="STAFF",
        )


class FakeDashboardService:
    def __init__(self):
        self.calls = []

    async def get_requests(
        self,
        **kwargs,
    ):
        self.calls.append(kwargs)

        return {
            "items": [],
            "next_cursor": None,
        }


def make_api_app():
    app = FastAPI()

    app.add_exception_handler(
        DashboardAPIError,
        dashboard_api_error_handler,
    )

    app.include_router(router)

    app.state.auth_service = FakeAuthService()
    app.state.dashboard_service = (
        FakeDashboardService()
    )

    return app


async def api_get(
    query: str,
):
    app = make_api_app()

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            f"/api/v1/requests{query}",
            headers={
                "X-Session-Token": SESSION_TOKEN,
            },
        )

    return response, app


async def test_request_api_default_query_values():
    response, app = await api_get(
        "?ward_code=WARD-3"
    )

    assert response.status_code == 200

    call = app.state.dashboard_service.calls[0]

    assert call["statuses"] == (
        "NEW",
        "ACKNOWLEDGED",
    )
    assert call["priorities"] is None
    assert call["sort_mode"] == "attention"
    assert call["limit"] == 20


async def test_request_api_parses_filters():
    response, app = await api_get(
        "?ward_code=WARD-3"
        "&status=NEW,COMPLETED"
        "&priority=HIGH,CRITICAL"
        "&category=PAIN"
        "&patient_id=302"
        "&sort=newest"
        "&limit=50"
    )

    assert response.status_code == 200

    call = app.state.dashboard_service.calls[0]

    assert call["statuses"] == (
        "NEW",
        "COMPLETED",
    )
    assert call["priorities"] == (
        "HIGH",
        "CRITICAL",
    )
    assert call["category"] == "PAIN"
    assert call["patient_id"] == 302
    assert call["sort_mode"] == "newest"
    assert call["limit"] == 50


async def test_invalid_filter_returns_400_invalid_query():
    response, _app = await api_get(
        "?ward_code=WARD-3&status=INVALID"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_invalid_limit_returns_400_invalid_query():
    response, _app = await api_get(
        "?ward_code=WARD-3&limit=101"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_invalid_priority_returns_400_invalid_query():
    response, _app = await api_get(
        "?ward_code=WARD-3&priority=URGENT"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_invalid_category_returns_400_invalid_query():
    response, _app = await api_get(
        "?ward_code=WARD-3&category=UNKNOWN"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_invalid_sort_returns_400_invalid_query():
    response, _app = await api_get(
        "?ward_code=WARD-3&sort=wrong"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_non_numeric_patient_id_returns_400_invalid_query():
    response, _app = await api_get(
        "?ward_code=WARD-3&patient_id=abc"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_non_positive_patient_id_returns_400_invalid_query():
    response, _app = await api_get(
        "?ward_code=WARD-3&patient_id=0"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"


async def test_non_numeric_limit_returns_400_invalid_query():
    response, _app = await api_get(
        "?ward_code=WARD-3&limit=abc"
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_QUERY"
