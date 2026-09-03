"""Medical staff dashboard PostgreSQL models."""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text as sql_text,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.backend.core.database import Base


class Ward(Base):
    """Ward master data used by the medical staff dashboard."""

    __tablename__ = "ward"

    ward_code: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
    )

    ward_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class StaffWardAccess(Base):
    """Wards that a STAFF account is allowed to access."""

    __tablename__ = "staff_ward_access"

    staff_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )

    ward_code: Mapped[str] = mapped_column(
        ForeignKey("ward.ward_code", ondelete="CASCADE"),
        primary_key=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class PatientProfile(Base):
    """Dashboard-facing patient profile."""

    __tablename__ = "patient_profile"
    __table_args__ = (
        UniqueConstraint("user_id"),
        UniqueConstraint("patient_code"),
        Index(
            "ix_patient_profile_ward_room",
            "ward_code",
            "room_number",
        ),
        Index(
            "ix_patient_profile_admitted_on",
            "admitted_on",
        ),
    )

    patient_id: Mapped[int] = mapped_column(
        Integer,
        Identity(),
        primary_key=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )

    patient_code: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    display_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    ward_code: Mapped[str] = mapped_column(
        ForeignKey("ward.ward_code", ondelete="RESTRICT"),
        nullable=False,
    )

    room_number: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    admitted_on: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    communication_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    assistive_method: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


class CommunicationRequest(Base):
    """One medical-dashboard request created from one completed utterance."""

    __tablename__ = "communication_request"
    __table_args__ = (
        UniqueConstraint("utterance_id"),
        CheckConstraint(
            "priority IN ('NORMAL', 'HIGH', 'CRITICAL')",
            name="priority",
        ),
        CheckConstraint(
            "status IN ('NEW', 'ACKNOWLEDGED', 'COMPLETED')",
            name="status",
        ),
        CheckConstraint(
            (
                "confidence IS NULL OR "
                "(confidence >= 0 AND confidence <= 1)"
            ),
            name="confidence_range",
        ),
        Index(
            "ix_communication_request_ward_requested_at",
            "ward_code",
            "requested_at",
        ),
        Index(
            "ix_communication_request_patient_requested_at",
            "patient_id",
            "requested_at",
        ),
        Index(
            "ix_communication_request_status_priority_requested_at",
            "status",
            "priority",
            "requested_at",
        ),
    )

    request_id: Mapped[int] = mapped_column(
        Integer,
        Identity(),
        primary_key=True,
    )

    utterance_id: Mapped[int] = mapped_column(
        ForeignKey("utterance.utt_id", ondelete="CASCADE"),
        nullable=False,
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patient_profile.patient_id", ondelete="CASCADE"),
        nullable=False,
    )

    # Snapshot values. They intentionally do not change when the patient moves.
    ward_code: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    room_number: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    phrase_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    text: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(4, 3),
        nullable=True,
    )

    priority: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=sql_text("'NORMAL'"),
    )

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=sql_text("'NEW'"),
    )

    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    acknowledged_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )

    resolution_note: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )


class RequestEvent(Base):
    """Audit trail for every communication request state transition."""

    __tablename__ = "request_event"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('REQUESTED', 'ACKNOWLEDGED', 'COMPLETED')",
            name="event_type",
        ),
        Index(
            "ix_request_event_request_occurred_at",
            "request_id",
            "occurred_at",
        ),
    )

    event_id: Mapped[int] = mapped_column(
        Integer,
        Identity(),
        primary_key=True,
    )

    request_id: Mapped[int] = mapped_column(
        ForeignKey(
            "communication_request.request_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    note: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )


class RequestIdempotency(Base):
    """Stored idempotency result for dashboard state-changing APIs."""

    __tablename__ = "request_idempotency"
    __table_args__ = (
        UniqueConstraint(
            "actor_user_id",
            "idempotency_key",
        ),
        CheckConstraint(
            "operation IN ('ACKNOWLEDGE', 'COMPLETE')",
            name="operation",
        ),
        Index(
            "ix_request_idempotency_request_id",
            "request_id",
        ),
    )

    idempotency_id: Mapped[int] = mapped_column(
        Integer,
        Identity(),
        primary_key=True,
    )

    actor_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )

    request_id: Mapped[int] = mapped_column(
        ForeignKey(
            "communication_request.request_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    idempotency_key: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    operation: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    request_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
