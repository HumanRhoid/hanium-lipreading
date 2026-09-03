"""의료진 대시보드 PostgreSQL repository."""

from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from src.backend.auth.adapters.repository import User
from src.backend.dashboard.models import (
    CommunicationRequest,
    PatientProfile,
    RequestEvent,
    RequestIdempotency,
    StaffWardAccess,
    Ward,
)


@dataclass(frozen=True)
class DashboardCountsRecord:
    patients_registered_today: int
    requests_today: int
    unacknowledged_requests: int
    critical_open_requests: int


@dataclass(frozen=True)
class RequestSummaryRecord:
    request_id: int
    patient_id: int
    patient_code: str
    patient_display_name: str
    ward_code: str
    ward_name: str
    room_number: str
    utterance_id: int
    text: str
    phrase_code: str | None
    category: str
    confidence: float | None
    priority: str
    status: str
    requested_at: datetime
    acknowledged_at: datetime | None
    acknowledged_by_user_id: int | None
    acknowledged_by_display_name: str | None
    completed_at: datetime | None
    completed_by_user_id: int | None
    completed_by_display_name: str | None


@dataclass(frozen=True)
class RequestPageRecord:
    items: list[RequestSummaryRecord]
    has_more: bool


@dataclass(frozen=True)
class RequestEventRecord:
    event_type: str
    occurred_at: datetime
    actor_user_id: int | None
    actor_display_name: str | None
    note: str | None


@dataclass(frozen=True)
class RequestDetailRecord:
    summary: RequestSummaryRecord
    resolution_note: str | None
    timeline: list[RequestEventRecord]


class RepositoryRequestNotFoundError(Exception):
    """Raised when the STAFF user cannot access the requested resource."""


class RepositoryTransitionConflictError(Exception):
    """Raised when the current request state does not permit the operation."""


class RepositoryIdempotencyConflictError(Exception):
    """Raised when an idempotency key is reused with different request data."""


@dataclass(frozen=True)
class PatientLatestRequestRecord:
    request_id: int
    text: str
    status: str
    priority: str
    requested_at: datetime


@dataclass(frozen=True)
class PatientBoardRecord:
    patient_id: int
    patient_code: str
    patient_display_name: str
    room_number: str
    open_request_count: int
    unacknowledged_request_count: int
    critical_open_count: int
    latest_request: PatientLatestRequestRecord | None


@dataclass(frozen=True)
class PatientProfileDetailRecord:
    patient_id: int
    patient_code: str
    patient_display_name: str
    ward_code: str
    ward_name: str
    room_number: str
    admitted_on: date
    communication_status: str
    assistive_method: str | None
    notes: str | None


@dataclass(frozen=True)
class FrequentPhraseRecord:
    phrase_code: str
    text: str
    count_30d: int
    last_used_at: datetime


@dataclass(frozen=True)
class TodayPhraseCountRecord:
    phrase_code: str
    text: str
    count: int
    last_used_at: datetime


@dataclass(frozen=True)
class PatientDetailStatsRecord:
    open_request_count: int
    unacknowledged_request_count: int
    latest_request: PatientLatestRequestRecord | None
    frequent_phrases: list[FrequentPhraseRecord]
    today_total_requests: int
    today_by_category: dict[str, int]
    today_by_phrase: list[TodayPhraseCountRecord]


@dataclass(frozen=True)
class PatientRequestHistoryRecord:
    request_id: int
    utterance_id: int
    text: str
    phrase_code: str | None
    category: str
    confidence: float | None
    priority: str
    status: str
    requested_at: datetime
    acknowledged_at: datetime | None
    acknowledged_by_user_id: int | None
    acknowledged_by_display_name: str | None
    completed_at: datetime | None
    completed_by_user_id: int | None
    completed_by_display_name: str | None


@dataclass(frozen=True)
class PatientRequestHistoryPageRecord:
    items: list[PatientRequestHistoryRecord]
    has_more: bool


def _attention_rank():
    return case(
        (
            and_(
                CommunicationRequest.priority == "CRITICAL",
                CommunicationRequest.status != "COMPLETED",
            ),
            0,
        ),
        (
            and_(
                CommunicationRequest.priority == "HIGH",
                CommunicationRequest.status != "COMPLETED",
            ),
            1,
        ),
        (
            CommunicationRequest.status == "NEW",
            2,
        ),
        (
            CommunicationRequest.status == "ACKNOWLEDGED",
            3,
        ),
        else_=4,
    )


class SQLAlchemyDashboardRepository:
    """대시보드용 PostgreSQL 조회 및 변경을 담당한다."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def staff_has_ward_access(
        self,
        *,
        staff_user_id: int,
        ward_code: str,
    ) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                select(StaffWardAccess.staff_user_id)
                .where(
                    StaffWardAccess.staff_user_id == staff_user_id,
                    StaffWardAccess.ward_code == ward_code,
                )
                .limit(1)
            )
            return result.scalar_one_or_none() is not None

    async def get_ward(
        self,
        *,
        ward_code: str,
    ) -> Ward | None:
        async with self._session_factory() as session:
            return await session.get(Ward, ward_code)

    async def get_dashboard_counts(
        self,
        *,
        ward_code: str,
        local_date: date,
        start_utc: datetime,
        end_utc: datetime,
    ) -> DashboardCountsRecord:
        async with self._session_factory() as session:
            patients_registered_today = await session.scalar(
                select(func.count(PatientProfile.patient_id)).where(
                    PatientProfile.ward_code == ward_code,
                    PatientProfile.admitted_on == local_date,
                )
            )

            requests_today = await session.scalar(
                select(func.count(CommunicationRequest.request_id)).where(
                    CommunicationRequest.ward_code == ward_code,
                    CommunicationRequest.requested_at >= start_utc,
                    CommunicationRequest.requested_at < end_utc,
                )
            )

            unacknowledged_requests = await session.scalar(
                select(func.count(CommunicationRequest.request_id)).where(
                    CommunicationRequest.ward_code == ward_code,
                    CommunicationRequest.status == "NEW",
                )
            )

            critical_open_requests = await session.scalar(
                select(func.count(CommunicationRequest.request_id)).where(
                    CommunicationRequest.ward_code == ward_code,
                    CommunicationRequest.priority == "CRITICAL",
                    CommunicationRequest.status != "COMPLETED",
                )
            )

            return DashboardCountsRecord(
                patients_registered_today=int(patients_registered_today or 0),
                requests_today=int(requests_today or 0),
                unacknowledged_requests=int(unacknowledged_requests or 0),
                critical_open_requests=int(critical_open_requests or 0),
            )

    async def list_recent_requests(
        self,
        *,
        ward_code: str,
        limit: int,
    ) -> list[RequestSummaryRecord]:
        page = await self.list_requests(
            ward_code=ward_code,
            statuses=None,
            priorities=None,
            category=None,
            patient_id=None,
            sort_mode="newest",
            limit=limit,
            cursor_rank=None,
            cursor_requested_at=None,
            cursor_request_id=None,
        )
        return page.items

    async def list_requests(
        self,
        *,
        ward_code: str,
        statuses: tuple[str, ...] | None,
        priorities: tuple[str, ...] | None,
        category: str | None,
        patient_id: int | None,
        sort_mode: str,
        limit: int,
        cursor_rank: int | None,
        cursor_requested_at: datetime | None,
        cursor_request_id: int | None,
    ) -> RequestPageRecord:
        acknowledged_user = aliased(User)
        completed_user = aliased(User)

        rank_expression = _attention_rank()

        statement = (
            select(
                CommunicationRequest,
                PatientProfile,
                Ward,
                acknowledged_user,
                completed_user,
                rank_expression.label("attention_rank"),
            )
            .join(
                PatientProfile,
                PatientProfile.patient_id == CommunicationRequest.patient_id,
            )
            .join(
                Ward,
                Ward.ward_code == CommunicationRequest.ward_code,
            )
            .outerjoin(
                acknowledged_user,
                acknowledged_user.user_id == CommunicationRequest.acknowledged_by,
            )
            .outerjoin(
                completed_user,
                completed_user.user_id == CommunicationRequest.completed_by,
            )
            .where(
                CommunicationRequest.ward_code == ward_code,
            )
        )

        if statuses is not None:
            statement = statement.where(
                CommunicationRequest.status.in_(statuses)
            )

        if priorities is not None:
            statement = statement.where(
                CommunicationRequest.priority.in_(priorities)
            )

        if category is not None:
            statement = statement.where(
                CommunicationRequest.category == category
            )

        if patient_id is not None:
            statement = statement.where(
                CommunicationRequest.patient_id == patient_id
            )

        if cursor_requested_at is not None and cursor_request_id is not None:
            if sort_mode == "newest":
                statement = statement.where(
                    or_(
                        CommunicationRequest.requested_at < cursor_requested_at,
                        and_(
                            CommunicationRequest.requested_at
                            == cursor_requested_at,
                            CommunicationRequest.request_id < cursor_request_id,
                        ),
                    )
                )
            else:
                if cursor_rank is None:
                    raise ValueError(
                        "attention cursor requires rank"
                    )

                statement = statement.where(
                    or_(
                        rank_expression > cursor_rank,
                        and_(
                            rank_expression == cursor_rank,
                            CommunicationRequest.requested_at
                            > cursor_requested_at,
                        ),
                        and_(
                            rank_expression == cursor_rank,
                            CommunicationRequest.requested_at
                            == cursor_requested_at,
                            CommunicationRequest.request_id
                            > cursor_request_id,
                        ),
                    )
                )

        if sort_mode == "newest":
            statement = statement.order_by(
                CommunicationRequest.requested_at.desc(),
                CommunicationRequest.request_id.desc(),
            )
        else:
            statement = statement.order_by(
                rank_expression.asc(),
                CommunicationRequest.requested_at.asc(),
                CommunicationRequest.request_id.asc(),
            )

        statement = statement.limit(limit + 1)

        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()

        has_more = len(rows) > limit
        rows = rows[:limit]

        records: list[RequestSummaryRecord] = []

        for (
            request_row,
            patient,
            ward,
            ack_user,
            done_user,
            _attention_rank_value,
        ) in rows:
            records.append(
                RequestSummaryRecord(
                    request_id=request_row.request_id,
                    patient_id=patient.patient_id,
                    patient_code=patient.patient_code,
                    patient_display_name=patient.display_name,
                    ward_code=request_row.ward_code,
                    ward_name=ward.ward_name,
                    room_number=request_row.room_number,
                    utterance_id=request_row.utterance_id,
                    text=request_row.text,
                    phrase_code=request_row.phrase_code,
                    category=request_row.category,
                    confidence=(
                        float(request_row.confidence)
                        if request_row.confidence is not None
                        else None
                    ),
                    priority=request_row.priority,
                    status=request_row.status,
                    requested_at=request_row.requested_at,
                    acknowledged_at=request_row.acknowledged_at,
                    acknowledged_by_user_id=(
                        ack_user.user_id
                        if ack_user is not None
                        else None
                    ),
                    acknowledged_by_display_name=(
                        ack_user.display_name
                        if ack_user is not None
                        else None
                    ),
                    completed_at=request_row.completed_at,
                    completed_by_user_id=(
                        done_user.user_id
                        if done_user is not None
                        else None
                    ),
                    completed_by_display_name=(
                        done_user.display_name
                        if done_user is not None
                        else None
                    ),
                )
            )

        return RequestPageRecord(
            items=records,
            has_more=has_more,
        )


    async def _load_request_detail(
        self,
        *,
        session: AsyncSession,
        request_id: int,
    ) -> RequestDetailRecord:
        acknowledged_user = aliased(User)
        completed_user = aliased(User)

        row = (
            await session.execute(
                select(
                    CommunicationRequest,
                    PatientProfile,
                    Ward,
                    acknowledged_user,
                    completed_user,
                )
                .join(
                    PatientProfile,
                    PatientProfile.patient_id
                    == CommunicationRequest.patient_id,
                )
                .join(
                    Ward,
                    Ward.ward_code
                    == CommunicationRequest.ward_code,
                )
                .outerjoin(
                    acknowledged_user,
                    acknowledged_user.user_id
                    == CommunicationRequest.acknowledged_by,
                )
                .outerjoin(
                    completed_user,
                    completed_user.user_id
                    == CommunicationRequest.completed_by,
                )
                .where(
                    CommunicationRequest.request_id
                    == request_id,
                )
            )
        ).one_or_none()

        if row is None:
            raise RepositoryRequestNotFoundError

        (
            request_row,
            patient,
            ward,
            ack_user,
            done_user,
        ) = row

        summary = RequestSummaryRecord(
            request_id=request_row.request_id,
            patient_id=patient.patient_id,
            patient_code=patient.patient_code,
            patient_display_name=patient.display_name,
            ward_code=request_row.ward_code,
            ward_name=ward.ward_name,
            room_number=request_row.room_number,
            utterance_id=request_row.utterance_id,
            text=request_row.text,
            phrase_code=request_row.phrase_code,
            category=request_row.category,
            confidence=(
                float(request_row.confidence)
                if request_row.confidence is not None
                else None
            ),
            priority=request_row.priority,
            status=request_row.status,
            requested_at=request_row.requested_at,
            acknowledged_at=request_row.acknowledged_at,
            acknowledged_by_user_id=(
                ack_user.user_id
                if ack_user is not None
                else None
            ),
            acknowledged_by_display_name=(
                ack_user.display_name
                if ack_user is not None
                else None
            ),
            completed_at=request_row.completed_at,
            completed_by_user_id=(
                done_user.user_id
                if done_user is not None
                else None
            ),
            completed_by_display_name=(
                done_user.display_name
                if done_user is not None
                else None
            ),
        )

        actor_user = aliased(User)

        event_rows = (
            await session.execute(
                select(
                    RequestEvent,
                    actor_user,
                )
                .outerjoin(
                    actor_user,
                    actor_user.user_id
                    == RequestEvent.actor_user_id,
                )
                .where(
                    RequestEvent.request_id
                    == request_id,
                )
                .order_by(
                    RequestEvent.occurred_at.asc(),
                    RequestEvent.event_id.asc(),
                )
            )
        ).all()

        timeline = [
            RequestEventRecord(
                event_type=event.event_type,
                occurred_at=event.occurred_at,
                actor_user_id=(
                    actor.user_id
                    if actor is not None
                    else None
                ),
                actor_display_name=(
                    actor.display_name
                    if actor is not None
                    else None
                ),
                note=event.note,
            )
            for event, actor in event_rows
        ]

        return RequestDetailRecord(
            summary=summary,
            resolution_note=request_row.resolution_note,
            timeline=timeline,
        )

    async def acknowledge_request(
        self,
        *,
        staff_user_id: int,
        request_id: int,
        idempotency_key: str,
        request_fingerprint: str,
        note: str | None,
        occurred_at: datetime,
    ) -> RequestDetailRecord:
        """Lock and transition a NEW request to ACKNOWLEDGED."""

        try:
            async with self._session_factory.begin() as session:
                request_row = (
                    await session.execute(
                        select(CommunicationRequest)
                        .join(
                            StaffWardAccess,
                            and_(
                                StaffWardAccess.ward_code
                                == CommunicationRequest.ward_code,
                                StaffWardAccess.staff_user_id
                                == staff_user_id,
                            ),
                        )
                        .where(
                            CommunicationRequest.request_id
                            == request_id,
                        )
                        .with_for_update()
                    )
                ).scalar_one_or_none()

                if request_row is None:
                    raise RepositoryRequestNotFoundError

                # The request row lock serializes competing staff updates.
                # Only the first valid transition succeeds in this transaction.
                existing_idempotency = (
                    await session.execute(
                        select(RequestIdempotency).where(
                            RequestIdempotency.actor_user_id
                            == staff_user_id,
                            RequestIdempotency.idempotency_key
                            == idempotency_key,
                        )
                    )
                ).scalar_one_or_none()

                if existing_idempotency is not None:
                    if (
                        existing_idempotency.request_id
                        != request_id
                        or existing_idempotency.operation
                        != "ACKNOWLEDGE"
                        or existing_idempotency.request_fingerprint
                        != request_fingerprint
                    ):
                        raise RepositoryIdempotencyConflictError

                    return await self._load_request_detail(
                        session=session,
                        request_id=request_id,
                    )

                if request_row.status != "NEW":
                    raise RepositoryTransitionConflictError

                request_row.status = "ACKNOWLEDGED"
                request_row.acknowledged_at = occurred_at
                request_row.acknowledged_by = staff_user_id

                session.add(
                    RequestEvent(
                        request_id=request_id,
                        event_type="ACKNOWLEDGED",
                        actor_user_id=staff_user_id,
                        occurred_at=occurred_at,
                        note=note,
                    )
                )

                session.add(
                    RequestIdempotency(
                        actor_user_id=staff_user_id,
                        request_id=request_id,
                        idempotency_key=idempotency_key,
                        operation="ACKNOWLEDGE",
                        request_fingerprint=request_fingerprint,
                    )
                )

                await session.flush()

                return await self._load_request_detail(
                    session=session,
                    request_id=request_id,
                )

        except IntegrityError as exc:
            # A concurrent retry may race on the same actor/key pair.
            # The unique constraint provides the final idempotency safeguard.
            raise RepositoryIdempotencyConflictError from exc


    async def complete_request(
        self,
        *,
        staff_user_id: int,
        request_id: int,
        idempotency_key: str,
        request_fingerprint: str,
        resolution_note: str | None,
        occurred_at: datetime,
    ) -> RequestDetailRecord:
        """Lock and transition an ACKNOWLEDGED request to COMPLETED."""

        try:
            async with self._session_factory.begin() as session:
                request_row = (
                    await session.execute(
                        select(CommunicationRequest)
                        .join(
                            StaffWardAccess,
                            and_(
                                StaffWardAccess.ward_code
                                == CommunicationRequest.ward_code,
                                StaffWardAccess.staff_user_id
                                == staff_user_id,
                            ),
                        )
                        .where(
                            CommunicationRequest.request_id
                            == request_id,
                        )
                        .with_for_update()
                    )
                ).scalar_one_or_none()

                if request_row is None:
                    raise RepositoryRequestNotFoundError

                existing_idempotency = (
                    await session.execute(
                        select(RequestIdempotency).where(
                            RequestIdempotency.actor_user_id
                            == staff_user_id,
                            RequestIdempotency.idempotency_key
                            == idempotency_key,
                        )
                    )
                ).scalar_one_or_none()

                if existing_idempotency is not None:
                    if (
                        existing_idempotency.request_id
                        != request_id
                        or existing_idempotency.operation
                        != "COMPLETE"
                        or existing_idempotency.request_fingerprint
                        != request_fingerprint
                    ):
                        raise RepositoryIdempotencyConflictError

                    return await self._load_request_detail(
                        session=session,
                        request_id=request_id,
                    )

                if request_row.status != "ACKNOWLEDGED":
                    raise RepositoryTransitionConflictError

                request_row.status = "COMPLETED"
                request_row.completed_at = occurred_at
                request_row.completed_by = staff_user_id
                request_row.resolution_note = resolution_note

                session.add(
                    RequestEvent(
                        request_id=request_id,
                        event_type="COMPLETED",
                        actor_user_id=staff_user_id,
                        occurred_at=occurred_at,
                        note=resolution_note,
                    )
                )

                session.add(
                    RequestIdempotency(
                        actor_user_id=staff_user_id,
                        request_id=request_id,
                        idempotency_key=idempotency_key,
                        operation="COMPLETE",
                        request_fingerprint=request_fingerprint,
                    )
                )

                await session.flush()

                return await self._load_request_detail(
                    session=session,
                    request_id=request_id,
                )

        except IntegrityError as exc:
            raise RepositoryIdempotencyConflictError from exc


    async def list_patient_board(
        self,
        *,
        ward_code: str,
    ) -> list[PatientBoardRecord]:
        """Return patients and request aggregates for one ward."""

        async with self._session_factory() as session:
            patients = (
                await session.execute(
                    select(PatientProfile)
                    .where(
                        PatientProfile.ward_code == ward_code,
                    )
                    .order_by(
                        PatientProfile.room_number.asc(),
                        PatientProfile.patient_id.asc(),
                    )
                )
            ).scalars().all()

            if not patients:
                return []

            count_rows = (
                await session.execute(
                    select(
                        CommunicationRequest.patient_id,
                        func.count(
                            CommunicationRequest.request_id
                        )
                        .filter(
                            CommunicationRequest.status
                            != "COMPLETED"
                        )
                        .label("open_request_count"),
                        func.count(
                            CommunicationRequest.request_id
                        )
                        .filter(
                            CommunicationRequest.status
                            == "NEW"
                        )
                        .label(
                            "unacknowledged_request_count"
                        ),
                        func.count(
                            CommunicationRequest.request_id
                        )
                        .filter(
                            and_(
                                CommunicationRequest.priority
                                == "CRITICAL",
                                CommunicationRequest.status
                                != "COMPLETED",
                            )
                        )
                        .label("critical_open_count"),
                    )
                    .where(
                        CommunicationRequest.ward_code
                        == ward_code,
                    )
                    .group_by(
                        CommunicationRequest.patient_id,
                    )
                )
            ).all()

            counts_by_patient = {
                row.patient_id: (
                    int(row.open_request_count or 0),
                    int(
                        row.unacknowledged_request_count
                        or 0
                    ),
                    int(row.critical_open_count or 0),
                )
                for row in count_rows
            }

            ranked_requests = (
                select(
                    CommunicationRequest.request_id.label(
                        "request_id"
                    ),
                    CommunicationRequest.patient_id.label(
                        "patient_id"
                    ),
                    CommunicationRequest.text.label("text"),
                    CommunicationRequest.status.label(
                        "status"
                    ),
                    CommunicationRequest.priority.label(
                        "priority"
                    ),
                    CommunicationRequest.requested_at.label(
                        "requested_at"
                    ),
                    func.row_number()
                    .over(
                        partition_by=(
                            CommunicationRequest.patient_id
                        ),
                        order_by=(
                            CommunicationRequest.requested_at.desc(),
                            CommunicationRequest.request_id.desc(),
                        ),
                    )
                    .label("row_number"),
                )
                .where(
                    CommunicationRequest.ward_code
                    == ward_code,
                )
                .subquery()
            )

            latest_rows = (
                await session.execute(
                    select(ranked_requests).where(
                        ranked_requests.c.row_number == 1
                    )
                )
            ).all()

            latest_by_patient = {
                row.patient_id: PatientLatestRequestRecord(
                    request_id=row.request_id,
                    text=row.text,
                    status=row.status,
                    priority=row.priority,
                    requested_at=row.requested_at,
                )
                for row in latest_rows
            }

            records: list[PatientBoardRecord] = []

            for patient in patients:
                (
                    open_count,
                    unacknowledged_count,
                    critical_count,
                ) = counts_by_patient.get(
                    patient.patient_id,
                    (0, 0, 0),
                )

                records.append(
                    PatientBoardRecord(
                        patient_id=patient.patient_id,
                        patient_code=patient.patient_code,
                        patient_display_name=(
                            patient.display_name
                        ),
                        room_number=patient.room_number,
                        open_request_count=open_count,
                        unacknowledged_request_count=(
                            unacknowledged_count
                        ),
                        critical_open_count=critical_count,
                        latest_request=(
                            latest_by_patient.get(
                                patient.patient_id
                            )
                        ),
                    )
                )

            return records


    async def get_patient_profile_detail(
        self,
        *,
        patient_id: int,
    ) -> PatientProfileDetailRecord | None:
        """Return patient profile and ward metadata."""

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(
                        PatientProfile,
                        Ward,
                    )
                    .join(
                        Ward,
                        Ward.ward_code
                        == PatientProfile.ward_code,
                    )
                    .where(
                        PatientProfile.patient_id
                        == patient_id,
                    )
                )
            ).one_or_none()

            if row is None:
                return None

            patient, ward = row

            return PatientProfileDetailRecord(
                patient_id=patient.patient_id,
                patient_code=patient.patient_code,
                patient_display_name=patient.display_name,
                ward_code=patient.ward_code,
                ward_name=ward.ward_name,
                room_number=patient.room_number,
                admitted_on=patient.admitted_on,
                communication_status=(
                    patient.communication_status
                ),
                assistive_method=patient.assistive_method,
                notes=patient.notes,
            )

    async def get_patient_detail_stats(
        self,
        *,
        patient_id: int,
        today_start_utc: datetime,
        today_end_utc: datetime,
        thirty_days_start_utc: datetime,
        now_utc: datetime,
    ) -> PatientDetailStatsRecord:
        """Aggregate patient request statistics for detail view."""

        async with self._session_factory() as session:
            count_row = (
                await session.execute(
                    select(
                        func.count(
                            CommunicationRequest.request_id
                        )
                        .filter(
                            CommunicationRequest.status
                            != "COMPLETED"
                        )
                        .label("open_request_count"),
                        func.count(
                            CommunicationRequest.request_id
                        )
                        .filter(
                            CommunicationRequest.status
                            == "NEW"
                        )
                        .label(
                            "unacknowledged_request_count"
                        ),
                    )
                    .where(
                        CommunicationRequest.patient_id
                        == patient_id,
                    )
                )
            ).one()

            latest_row = (
                await session.execute(
                    select(CommunicationRequest)
                    .where(
                        CommunicationRequest.patient_id
                        == patient_id,
                    )
                    .order_by(
                        CommunicationRequest.requested_at.desc(),
                        CommunicationRequest.request_id.desc(),
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()

            latest_request = None

            if latest_row is not None:
                latest_request = PatientLatestRequestRecord(
                    request_id=latest_row.request_id,
                    text=latest_row.text,
                    status=latest_row.status,
                    priority=latest_row.priority,
                    requested_at=latest_row.requested_at,
                )

            recent_rows = (
                await session.execute(
                    select(
                        CommunicationRequest.request_id,
                        CommunicationRequest.phrase_code,
                        CommunicationRequest.text,
                        CommunicationRequest.category,
                        CommunicationRequest.requested_at,
                    )
                    .where(
                        CommunicationRequest.patient_id
                        == patient_id,
                        CommunicationRequest.requested_at
                        >= thirty_days_start_utc,
                        CommunicationRequest.requested_at
                        <= now_utc,
                    )
                    .order_by(
                        CommunicationRequest.requested_at.desc(),
                        CommunicationRequest.request_id.desc(),
                    )
                )
            ).all()

        phrase_30d: dict[
            str,
            dict[str, object],
        ] = {}

        today_by_category = {
            "PAIN": 0,
            "REQUEST": 0,
            "REPLY": 0,
            "ETC": 0,
        }

        today_phrase: dict[
            str,
            dict[str, object],
        ] = {}

        today_total = 0

        for row in recent_rows:
            if row.phrase_code is not None:
                existing = phrase_30d.get(
                    row.phrase_code
                )

                if existing is None:
                    phrase_30d[row.phrase_code] = {
                        "text": row.text,
                        "count": 1,
                        "last_used_at": row.requested_at,
                    }
                else:
                    existing["count"] = (
                        int(existing["count"]) + 1
                    )

            if (
                row.requested_at >= today_start_utc
                and row.requested_at < today_end_utc
            ):
                today_total += 1

                if row.category in today_by_category:
                    today_by_category[row.category] += 1
                else:
                    today_by_category["ETC"] += 1

                if row.phrase_code is not None:
                    existing_today = today_phrase.get(
                        row.phrase_code
                    )

                    if existing_today is None:
                        today_phrase[row.phrase_code] = {
                            "text": row.text,
                            "count": 1,
                            "last_used_at": (
                                row.requested_at
                            ),
                        }
                    else:
                        existing_today["count"] = (
                            int(
                                existing_today["count"]
                            )
                            + 1
                        )

        frequent_phrases = [
            FrequentPhraseRecord(
                phrase_code=phrase_code,
                text=str(data["text"]),
                count_30d=int(data["count"]),
                last_used_at=data["last_used_at"],
            )
            for phrase_code, data
            in phrase_30d.items()
        ]

        frequent_phrases.sort(
            key=lambda item: (
                -item.count_30d,
                -item.last_used_at.timestamp(),
                item.phrase_code,
            )
        )

        frequent_phrases = frequent_phrases[:4]

        today_by_phrase = [
            TodayPhraseCountRecord(
                phrase_code=phrase_code,
                text=str(data["text"]),
                count=int(data["count"]),
                last_used_at=data["last_used_at"],
            )
            for phrase_code, data
            in today_phrase.items()
        ]

        today_by_phrase.sort(
            key=lambda item: (
                -item.count,
                -item.last_used_at.timestamp(),
                item.phrase_code,
            )
        )

        return PatientDetailStatsRecord(
            open_request_count=int(
                count_row.open_request_count or 0
            ),
            unacknowledged_request_count=int(
                count_row.unacknowledged_request_count
                or 0
            ),
            latest_request=latest_request,
            frequent_phrases=frequent_phrases,
            today_total_requests=today_total,
            today_by_category=today_by_category,
            today_by_phrase=today_by_phrase,
        )


    async def list_patient_requests(
        self,
        *,
        patient_id: int,
        status_filter: str | None,
        category: str | None,
        date_from_utc: datetime | None,
        date_to_utc: datetime | None,
        cursor_requested_at: datetime | None,
        cursor_request_id: int | None,
        limit: int,
    ) -> PatientRequestHistoryPageRecord:
        """Return one patient's requests ordered newest first."""

        acknowledged_user = aliased(User)
        completed_user = aliased(User)

        statement = (
            select(
                CommunicationRequest,
                acknowledged_user,
                completed_user,
            )
            .outerjoin(
                acknowledged_user,
                acknowledged_user.user_id
                == CommunicationRequest.acknowledged_by,
            )
            .outerjoin(
                completed_user,
                completed_user.user_id
                == CommunicationRequest.completed_by,
            )
            .where(
                CommunicationRequest.patient_id
                == patient_id,
            )
        )

        if status_filter is not None:
            statement = statement.where(
                CommunicationRequest.status
                == status_filter,
            )

        if category is not None:
            statement = statement.where(
                CommunicationRequest.category
                == category,
            )

        if date_from_utc is not None:
            statement = statement.where(
                CommunicationRequest.requested_at
                >= date_from_utc,
            )

        if date_to_utc is not None:
            statement = statement.where(
                CommunicationRequest.requested_at
                < date_to_utc,
            )

        if (
            cursor_requested_at is not None
            and cursor_request_id is not None
        ):
            statement = statement.where(
                or_(
                    CommunicationRequest.requested_at
                    < cursor_requested_at,
                    and_(
                        CommunicationRequest.requested_at
                        == cursor_requested_at,
                        CommunicationRequest.request_id
                        < cursor_request_id,
                    ),
                )
            )

        statement = statement.order_by(
            CommunicationRequest.requested_at.desc(),
            CommunicationRequest.request_id.desc(),
        ).limit(limit + 1)

        async with self._session_factory() as session:
            rows = (
                await session.execute(statement)
            ).all()

        has_more = len(rows) > limit
        rows = rows[:limit]

        items: list[PatientRequestHistoryRecord] = []

        for request_row, ack_user, done_user in rows:
            items.append(
                PatientRequestHistoryRecord(
                    request_id=request_row.request_id,
                    utterance_id=request_row.utterance_id,
                    text=request_row.text,
                    phrase_code=request_row.phrase_code,
                    category=request_row.category,
                    confidence=(
                        float(request_row.confidence)
                        if request_row.confidence is not None
                        else None
                    ),
                    priority=request_row.priority,
                    status=request_row.status,
                    requested_at=request_row.requested_at,
                    acknowledged_at=(
                        request_row.acknowledged_at
                    ),
                    acknowledged_by_user_id=(
                        ack_user.user_id
                        if ack_user is not None
                        else None
                    ),
                    acknowledged_by_display_name=(
                        ack_user.display_name
                        if ack_user is not None
                        else None
                    ),
                    completed_at=request_row.completed_at,
                    completed_by_user_id=(
                        done_user.user_id
                        if done_user is not None
                        else None
                    ),
                    completed_by_display_name=(
                        done_user.display_name
                        if done_user is not None
                        else None
                    ),
                )
            )

        return PatientRequestHistoryPageRecord(
            items=items,
            has_more=has_more,
        )
