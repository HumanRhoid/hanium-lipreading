"""의료진 대시보드 권한 및 비즈니스 로직."""

import base64
import hashlib
import json
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from src.backend.dashboard.adapters.repository import (
    PatientBoardRecord,
    PatientRequestHistoryRecord,
    RepositoryIdempotencyConflictError,
    RepositoryRequestNotFoundError,
    RepositoryTransitionConflictError,
    RequestDetailRecord,
    RequestSummaryRecord,
    SQLAlchemyDashboardRepository,
)
from src.backend.dashboard.schemas import (
    DashboardCountsResponse,
    DashboardSummaryResponse,
    FrequentPhraseResponse,
    PatientBoardItemResponse,
    PatientBoardResponse,
    PatientDetailResponse,
    PatientLatestRequestResponse,
    PatientRequestHistoryItemResponse,
    PatientRequestHistoryResponse,
    PatientTodaySummaryResponse,
    RequestDetailResponse,
    RequestListResponse,
    RequestPatientResponse,
    RequestSummaryResponse,
    RequestTimelineEventResponse,
    StaffActorResponse,
    TodayPhraseCountResponse,
    WardResponse,
)

SEOUL = ZoneInfo("Asia/Seoul")

ALLOWED_REQUEST_STATUSES = {
    "NEW",
    "ACKNOWLEDGED",
    "COMPLETED",
}

ALLOWED_REQUEST_PRIORITIES = {
    "NORMAL",
    "HIGH",
    "CRITICAL",
}

ALLOWED_REQUEST_CATEGORIES = {
    "PAIN",
    "REQUEST",
    "REPLY",
    "ETC",
}

ALLOWED_REQUEST_SORTS = {
    "attention",
    "newest",
}


ALLOWED_BOARD_STATUSES = {
    "GREEN",
    "YELLOW",
    "RED",
}


COMMUNICATION_STATUS_LABELS = {
    "VOICE_DIFFICULT": "\uc74c\uc131 \uc758\uc0ac\uc18c\ud1b5 \uc5b4\ub824\uc6c0",
}


class InvalidRequestTransitionError(Exception):
    """Raised when a communication request state transition is invalid."""


class DashboardIdempotencyConflictError(Exception):
    """Raised when an idempotency key is reused with different request data."""


class ResourceNotFoundError(Exception):
    """자원이 없거나 현재 의료진이 접근할 수 없는 경우."""


class InvalidDashboardQueryError(Exception):
    """대시보드 query parameter가 명세에 맞지 않는 경우."""


def mask_patient_name(display_name: str) -> str:
    """환자 이름의 첫 글자만 남기고 나머지를 O로 마스킹한다."""

    value = display_name.strip()

    if not value:
        return ""

    if len(value) == 1:
        return value

    return value[0] + ("O" * (len(value) - 1))


def _attention_rank_for_record(
    record: RequestSummaryRecord,
) -> int:
    if (
        record.priority == "CRITICAL"
        and record.status != "COMPLETED"
    ):
        return 0

    if (
        record.priority == "HIGH"
        and record.status != "COMPLETED"
    ):
        return 1

    if record.status == "NEW":
        return 2

    if record.status == "ACKNOWLEDGED":
        return 3

    return 4


def _encode_cursor(
    *,
    sort_mode: str,
    record: RequestSummaryRecord,
) -> str:
    payload: dict[str, object] = {
        "sort": sort_mode,
        "requested_at": record.requested_at.astimezone(UTC).isoformat(),
        "request_id": record.request_id,
    }

    if sort_mode == "attention":
        payload["rank"] = _attention_rank_for_record(record)

    raw = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    return (
        base64.urlsafe_b64encode(raw)
        .decode("ascii")
        .rstrip("=")
    )


def _decode_cursor(
    cursor: str,
    *,
    expected_sort: str,
) -> tuple[int | None, datetime, int]:
    try:
        padding = "=" * (-len(cursor) % 4)

        raw = base64.urlsafe_b64decode(
            (cursor + padding).encode("ascii")
        )

        payload = json.loads(raw.decode("utf-8"))

        if payload.get("sort") != expected_sort:
            raise ValueError

        requested_at = datetime.fromisoformat(
            str(payload["requested_at"])
        )

        if requested_at.tzinfo is None:
            raise ValueError

        request_id = int(payload["request_id"])

        if request_id < 1:
            raise ValueError

        rank: int | None = None

        if expected_sort == "attention":
            rank = int(payload["rank"])

            if rank not in {0, 1, 2, 3, 4}:
                raise ValueError

        return (
            rank,
            requested_at.astimezone(UTC),
            request_id,
        )

    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise InvalidDashboardQueryError from exc


class DashboardService:
    """의료진 대시보드 비즈니스 로직."""

    def __init__(
        self,
        repository: SQLAlchemyDashboardRepository,
    ) -> None:
        self._repository = repository

    async def require_ward_access(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
    ) -> None:
        allowed = await self._repository.staff_has_ward_access(
            staff_user_id=staff_user_id,
            ward_code=ward_code,
        )

        if not allowed:
            raise ResourceNotFoundError

    async def get_summary(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
        target_date: date | None,
        recent_limit: int,
        now: datetime | None = None,
    ) -> DashboardSummaryResponse:
        if not ward_code.strip():
            raise InvalidDashboardQueryError

        if recent_limit < 1 or recent_limit > 20:
            raise InvalidDashboardQueryError

        await self.require_ward_access(
            staff_user_id=staff_user_id,
            ward_code=ward_code,
        )

        ward = await self._repository.get_ward(
            ward_code=ward_code,
        )

        if ward is None:
            raise ResourceNotFoundError

        generated_at = now or datetime.now(UTC)

        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)

        generated_at = generated_at.astimezone(UTC)

        seoul_now = generated_at.astimezone(SEOUL)
        summary_date = target_date or seoul_now.date()

        start_seoul = datetime.combine(
            summary_date,
            time.min,
            tzinfo=SEOUL,
        )

        start_utc = start_seoul.astimezone(UTC)
        end_utc = (start_seoul + timedelta(days=1)).astimezone(UTC)

        counts = await self._repository.get_dashboard_counts(
            ward_code=ward_code,
            local_date=summary_date,
            start_utc=start_utc,
            end_utc=end_utc,
        )

        recent_records = await self._repository.list_recent_requests(
            ward_code=ward_code,
            limit=recent_limit,
        )

        return DashboardSummaryResponse(
            generated_at=generated_at,
            timezone="Asia/Seoul",
            ward=WardResponse(
                ward_code=ward.ward_code,
                ward_name=ward.ward_name,
            ),
            counts=DashboardCountsResponse(
                patients_registered_today=(
                    counts.patients_registered_today
                ),
                requests_today=counts.requests_today,
                unacknowledged_requests=(
                    counts.unacknowledged_requests
                ),
                critical_open_requests=(
                    counts.critical_open_requests
                ),
            ),
            recent_requests=[
                self._to_request_summary(
                    record,
                    generated_at=generated_at,
                )
                for record in recent_records
            ],
        )

    async def get_requests(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
        statuses: tuple[str, ...],
        priorities: tuple[str, ...] | None,
        category: str | None,
        patient_id: int | None,
        sort_mode: str,
        cursor: str | None,
        limit: int,
        now: datetime | None = None,
    ) -> RequestListResponse:
        if not ward_code.strip():
            raise InvalidDashboardQueryError

        if limit < 1 or limit > 100:
            raise InvalidDashboardQueryError

        if not statuses:
            raise InvalidDashboardQueryError

        if any(
            value not in ALLOWED_REQUEST_STATUSES
            for value in statuses
        ):
            raise InvalidDashboardQueryError

        if priorities is not None:
            if not priorities:
                raise InvalidDashboardQueryError

            if any(
                value not in ALLOWED_REQUEST_PRIORITIES
                for value in priorities
            ):
                raise InvalidDashboardQueryError

        if (
            category is not None
            and category not in ALLOWED_REQUEST_CATEGORIES
        ):
            raise InvalidDashboardQueryError

        if patient_id is not None and patient_id < 1:
            raise InvalidDashboardQueryError

        if sort_mode not in ALLOWED_REQUEST_SORTS:
            raise InvalidDashboardQueryError

        await self.require_ward_access(
            staff_user_id=staff_user_id,
            ward_code=ward_code,
        )

        cursor_rank = None
        cursor_requested_at = None
        cursor_request_id = None

        if cursor is not None:
            (
                cursor_rank,
                cursor_requested_at,
                cursor_request_id,
            ) = _decode_cursor(
                cursor,
                expected_sort=sort_mode,
            )

        page = await self._repository.list_requests(
            ward_code=ward_code,
            statuses=statuses,
            priorities=priorities,
            category=category,
            patient_id=patient_id,
            sort_mode=sort_mode,
            limit=limit,
            cursor_rank=cursor_rank,
            cursor_requested_at=cursor_requested_at,
            cursor_request_id=cursor_request_id,
        )

        generated_at = now or datetime.now(UTC)

        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)

        generated_at = generated_at.astimezone(UTC)

        next_cursor = None

        if page.has_more and page.items:
            next_cursor = _encode_cursor(
                sort_mode=sort_mode,
                record=page.items[-1],
            )

        return RequestListResponse(
            items=[
                self._to_request_summary(
                    item,
                    generated_at=generated_at,
                )
                for item in page.items
            ],
            next_cursor=next_cursor,
        )

    async def get_patient_board(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
        board_status: str | None,
        now: datetime | None = None,
    ) -> PatientBoardResponse:
        """Return the medical staff patient status board."""

        normalized_ward = ward_code.strip()

        if not normalized_ward:
            raise InvalidDashboardQueryError

        if (
            board_status is not None
            and board_status not in ALLOWED_BOARD_STATUSES
        ):
            raise InvalidDashboardQueryError

        await self.require_ward_access(
            staff_user_id=staff_user_id,
            ward_code=normalized_ward,
        )

        ward = await self._repository.get_ward(
            ward_code=normalized_ward,
        )

        if ward is None:
            raise ResourceNotFoundError

        records = await self._repository.list_patient_board(
            ward_code=normalized_ward,
        )

        generated_at = now or datetime.now(UTC)

        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(
                tzinfo=UTC
            )

        generated_at = generated_at.astimezone(UTC)

        patients: list[PatientBoardItemResponse] = []

        for record in records:
            current_status = self._patient_board_status(
                record
            )

            if (
                board_status is not None
                and current_status != board_status
            ):
                continue

            latest_request = None

            if record.latest_request is not None:
                latest_request = PatientLatestRequestResponse(
                    request_id=(
                        record.latest_request.request_id
                    ),
                    text=record.latest_request.text,
                    status=record.latest_request.status,
                    priority=record.latest_request.priority,
                    requested_at=(
                        record.latest_request.requested_at
                    ),
                )

            patients.append(
                PatientBoardItemResponse(
                    patient_id=record.patient_id,
                    patient_code=record.patient_code,
                    masked_name=mask_patient_name(
                        record.patient_display_name
                    ),
                    room_number=record.room_number,
                    board_status=current_status,
                    open_request_count=(
                        record.open_request_count
                    ),
                    unacknowledged_request_count=(
                        record.unacknowledged_request_count
                    ),
                    critical_open_count=(
                        record.critical_open_count
                    ),
                    latest_request=latest_request,
                )
            )

        return PatientBoardResponse(
            ward=WardResponse(
                ward_code=ward.ward_code,
                ward_name=ward.ward_name,
            ),
            generated_at=generated_at,
            patients=patients,
        )

    @staticmethod
    def _patient_board_status(
        record: PatientBoardRecord,
    ) -> str:
        if record.critical_open_count > 0:
            return "RED"

        if record.unacknowledged_request_count > 0:
            return "YELLOW"

        return "GREEN"

    async def get_patient_detail(
        self,
        *,
        staff_user_id: int,
        patient_id: int,
        now: datetime | None = None,
    ) -> PatientDetailResponse:
        """Return one accessible patient's dashboard detail."""

        if patient_id < 1:
            raise InvalidDashboardQueryError

        patient = (
            await self._repository.get_patient_profile_detail(
                patient_id=patient_id,
            )
        )

        if patient is None:
            raise ResourceNotFoundError

        await self.require_ward_access(
            staff_user_id=staff_user_id,
            ward_code=patient.ward_code,
        )

        generated_at = now or datetime.now(UTC)

        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(
                tzinfo=UTC
            )

        generated_at = generated_at.astimezone(UTC)

        seoul_now = generated_at.astimezone(SEOUL)
        today = seoul_now.date()

        today_start_seoul = datetime.combine(
            today,
            time.min,
            tzinfo=SEOUL,
        )

        today_start_utc = (
            today_start_seoul.astimezone(UTC)
        )

        today_end_utc = (
            today_start_seoul
            + timedelta(days=1)
        ).astimezone(UTC)

        thirty_days_start_utc = (
            generated_at - timedelta(days=30)
        )

        stats = (
            await self._repository.get_patient_detail_stats(
                patient_id=patient_id,
                today_start_utc=today_start_utc,
                today_end_utc=today_end_utc,
                thirty_days_start_utc=(
                    thirty_days_start_utc
                ),
                now_utc=generated_at,
            )
        )

        latest_request = None

        if stats.latest_request is not None:
            latest_request = PatientLatestRequestResponse(
                request_id=stats.latest_request.request_id,
                text=stats.latest_request.text,
                status=stats.latest_request.status,
                priority=stats.latest_request.priority,
                requested_at=(
                    stats.latest_request.requested_at
                ),
            )

        return PatientDetailResponse(
            patient_id=patient.patient_id,
            patient_code=patient.patient_code,
            masked_name=mask_patient_name(
                patient.patient_display_name
            ),
            ward=WardResponse(
                ward_code=patient.ward_code,
                ward_name=patient.ward_name,
            ),
            room_number=patient.room_number,
            admitted_on=patient.admitted_on,
            communication_status=(
                patient.communication_status
            ),
            communication_status_label=(
                COMMUNICATION_STATUS_LABELS.get(
                    patient.communication_status,
                    patient.communication_status,
                )
            ),
            assistive_method=patient.assistive_method,
            notes=patient.notes,
            open_request_count=(
                stats.open_request_count
            ),
            unacknowledged_request_count=(
                stats.unacknowledged_request_count
            ),
            latest_request=latest_request,
            frequent_phrases=[
                FrequentPhraseResponse(
                    phrase_code=item.phrase_code,
                    text=item.text,
                    count_30d=item.count_30d,
                )
                for item in stats.frequent_phrases
            ],
            today_summary=PatientTodaySummaryResponse(
                date=today,
                total_requests=(
                    stats.today_total_requests
                ),
                by_category=stats.today_by_category,
                by_phrase=[
                    TodayPhraseCountResponse(
                        phrase_code=item.phrase_code,
                        text=item.text,
                        count=item.count,
                    )
                    for item in stats.today_by_phrase
                ],
            ),
        )

    async def get_patient_requests(
        self,
        *,
        staff_user_id: int,
        patient_id: int,
        date_from: date | None,
        date_to: date | None,
        status_filter: str | None,
        category: str | None,
        cursor: str | None,
        limit: int,
    ) -> PatientRequestHistoryResponse:
        """Return one accessible patient's request history."""

        if patient_id < 1:
            raise InvalidDashboardQueryError

        if limit < 1 or limit > 100:
            raise InvalidDashboardQueryError

        if (
            status_filter is not None
            and status_filter
            not in ALLOWED_REQUEST_STATUSES
        ):
            raise InvalidDashboardQueryError

        if (
            category is not None
            and category not in ALLOWED_REQUEST_CATEGORIES
        ):
            raise InvalidDashboardQueryError

        if (
            date_from is not None
            and date_to is not None
            and date_from > date_to
        ):
            raise InvalidDashboardQueryError

        patient = (
            await self._repository.get_patient_profile_detail(
                patient_id=patient_id,
            )
        )

        if patient is None:
            raise ResourceNotFoundError

        await self.require_ward_access(
            staff_user_id=staff_user_id,
            ward_code=patient.ward_code,
        )

        date_from_utc = None

        if date_from is not None:
            date_from_utc = datetime.combine(
                date_from,
                time.min,
                tzinfo=SEOUL,
            ).astimezone(UTC)

        date_to_utc = None

        if date_to is not None:
            date_to_utc = (
                datetime.combine(
                    date_to,
                    time.min,
                    tzinfo=SEOUL,
                )
                + timedelta(days=1)
            ).astimezone(UTC)

        cursor_requested_at = None
        cursor_request_id = None

        if cursor is not None:
            (
                cursor_rank,
                cursor_requested_at,
                cursor_request_id,
            ) = _decode_cursor(
                cursor,
                expected_sort="newest",
            )

            if cursor_rank is not None:
                raise InvalidDashboardQueryError

        page = await self._repository.list_patient_requests(
            patient_id=patient_id,
            status_filter=status_filter,
            category=category,
            date_from_utc=date_from_utc,
            date_to_utc=date_to_utc,
            cursor_requested_at=cursor_requested_at,
            cursor_request_id=cursor_request_id,
            limit=limit,
        )

        next_cursor = None

        if page.has_more and page.items:
            next_cursor = _encode_cursor(
                sort_mode="newest",
                record=page.items[-1],
            )

        return PatientRequestHistoryResponse(
            items=[
                self._to_patient_request_history_item(
                    item
                )
                for item in page.items
            ],
            next_cursor=next_cursor,
        )

    @staticmethod
    def _to_patient_request_history_item(
        record: PatientRequestHistoryRecord,
    ) -> PatientRequestHistoryItemResponse:
        acknowledged_by = None

        if (
            record.acknowledged_by_user_id is not None
            and record.acknowledged_by_display_name
            is not None
        ):
            acknowledged_by = StaffActorResponse(
                user_id=(
                    record.acknowledged_by_user_id
                ),
                display_name=(
                    record.acknowledged_by_display_name
                ),
            )

        completed_by = None

        if (
            record.completed_by_user_id is not None
            and record.completed_by_display_name
            is not None
        ):
            completed_by = StaffActorResponse(
                user_id=record.completed_by_user_id,
                display_name=(
                    record.completed_by_display_name
                ),
            )

        return PatientRequestHistoryItemResponse(
            request_id=record.request_id,
            utterance_id=record.utterance_id,
            text=record.text,
            phrase_code=record.phrase_code,
            category=record.category,
            confidence=record.confidence,
            priority=record.priority,
            status=record.status,
            requested_at=record.requested_at,
            acknowledged_at=record.acknowledged_at,
            acknowledged_by=acknowledged_by,
            completed_at=record.completed_at,
            completed_by=completed_by,
        )

    async def acknowledge_request(
        self,
        *,
        staff_user_id: int,
        request_id: int,
        idempotency_key: str,
        note: str | None,
        now: datetime | None = None,
    ) -> RequestDetailResponse:
        """Transition a NEW request to ACKNOWLEDGED."""

        if request_id < 1:
            raise InvalidDashboardQueryError

        normalized_key = idempotency_key.strip()

        if not normalized_key or len(normalized_key) > 128:
            raise InvalidDashboardQueryError

        if note is not None and len(note) > 500:
            raise InvalidDashboardQueryError

        fingerprint_payload = json.dumps(
            {
                "note": note,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        request_fingerprint = hashlib.sha256(
            fingerprint_payload
        ).hexdigest()

        occurred_at = now or datetime.now(UTC)

        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)

        occurred_at = occurred_at.astimezone(UTC)

        try:
            detail = await self._repository.acknowledge_request(
                staff_user_id=staff_user_id,
                request_id=request_id,
                idempotency_key=normalized_key,
                request_fingerprint=request_fingerprint,
                note=note,
                occurred_at=occurred_at,
            )
        except RepositoryRequestNotFoundError as exc:
            raise ResourceNotFoundError from exc
        except RepositoryTransitionConflictError as exc:
            raise InvalidRequestTransitionError from exc
        except RepositoryIdempotencyConflictError as exc:
            raise DashboardIdempotencyConflictError from exc

        return self._to_request_detail(
            detail,
            generated_at=occurred_at,
        )

    async def complete_request(
        self,
        *,
        staff_user_id: int,
        request_id: int,
        idempotency_key: str,
        resolution_note: str | None,
        now: datetime | None = None,
    ) -> RequestDetailResponse:
        """Transition an ACKNOWLEDGED request to COMPLETED."""

        if request_id < 1:
            raise InvalidDashboardQueryError

        normalized_key = idempotency_key.strip()

        if not normalized_key or len(normalized_key) > 128:
            raise InvalidDashboardQueryError

        if (
            resolution_note is not None
            and len(resolution_note) > 500
        ):
            raise InvalidDashboardQueryError

        fingerprint_payload = json.dumps(
            {
                "resolution_note": resolution_note,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        request_fingerprint = hashlib.sha256(
            fingerprint_payload
        ).hexdigest()

        occurred_at = now or datetime.now(UTC)

        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)

        occurred_at = occurred_at.astimezone(UTC)

        try:
            detail = await self._repository.complete_request(
                staff_user_id=staff_user_id,
                request_id=request_id,
                idempotency_key=normalized_key,
                request_fingerprint=request_fingerprint,
                resolution_note=resolution_note,
                occurred_at=occurred_at,
            )
        except RepositoryRequestNotFoundError as exc:
            raise ResourceNotFoundError from exc
        except RepositoryTransitionConflictError as exc:
            raise InvalidRequestTransitionError from exc
        except RepositoryIdempotencyConflictError as exc:
            raise DashboardIdempotencyConflictError from exc

        return self._to_request_detail(
            detail,
            generated_at=occurred_at,
        )

    @classmethod
    def _to_request_detail(
        cls,
        record: RequestDetailRecord,
        *,
        generated_at: datetime,
    ) -> RequestDetailResponse:
        """Convert a repository request-detail record to the API response model."""

        summary = cls._to_request_summary(
            record.summary,
            generated_at=generated_at,
        )

        timeline: list[RequestTimelineEventResponse] = []

        for event in record.timeline:
            actor = None

            if (
                event.actor_user_id is not None
                and event.actor_display_name is not None
            ):
                actor = StaffActorResponse(
                    user_id=event.actor_user_id,
                    display_name=event.actor_display_name,
                )

            timeline.append(
                RequestTimelineEventResponse(
                    event_type=event.event_type,
                    occurred_at=event.occurred_at,
                    actor=actor,
                    note=event.note,
                )
            )

        return RequestDetailResponse(
            **summary.model_dump(),
            resolution_note=record.resolution_note,
            timeline=timeline,
        )

    @staticmethod
    def _to_request_summary(
        record: RequestSummaryRecord,
        *,
        generated_at: datetime,
    ) -> RequestSummaryResponse:
        unacknowledged_seconds: int | None = None

        if record.status == "NEW":
            requested_at = record.requested_at

            if requested_at.tzinfo is None:
                requested_at = requested_at.replace(tzinfo=UTC)

            unacknowledged_seconds = max(
                0,
                int(
                    (
                        generated_at
                        - requested_at.astimezone(UTC)
                    ).total_seconds()
                ),
            )

        acknowledged_by = None

        if (
            record.acknowledged_by_user_id is not None
            and record.acknowledged_by_display_name is not None
        ):
            acknowledged_by = StaffActorResponse(
                user_id=record.acknowledged_by_user_id,
                display_name=(
                    record.acknowledged_by_display_name
                ),
            )

        completed_by = None

        if (
            record.completed_by_user_id is not None
            and record.completed_by_display_name is not None
        ):
            completed_by = StaffActorResponse(
                user_id=record.completed_by_user_id,
                display_name=(
                    record.completed_by_display_name
                ),
            )

        return RequestSummaryResponse(
            request_id=record.request_id,
            patient=RequestPatientResponse(
                patient_id=record.patient_id,
                patient_code=record.patient_code,
                masked_name=mask_patient_name(
                    record.patient_display_name,
                ),
                ward_code=record.ward_code,
                ward_name=record.ward_name,
                room_number=record.room_number,
            ),
            utterance_id=record.utterance_id,
            text=record.text,
            phrase_code=record.phrase_code,
            category=record.category,
            confidence=record.confidence,
            priority=record.priority,
            status=record.status,
            requested_at=record.requested_at,
            unacknowledged_seconds=unacknowledged_seconds,
            acknowledged_at=record.acknowledged_at,
            acknowledged_by=acknowledged_by,
            completed_at=record.completed_at,
            completed_by=completed_by,
        )
