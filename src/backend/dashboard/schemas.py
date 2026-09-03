"""의료진 대시보드 API 요청/응답 스키마."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class StaffActorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int
    display_name: str


class RequestPatientResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: int
    patient_code: str
    masked_name: str
    ward_code: str
    ward_name: str
    room_number: str


class RequestSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: int
    patient: RequestPatientResponse
    utterance_id: int
    text: str
    phrase_code: str | None
    category: str
    confidence: float | None
    priority: str
    status: str
    requested_at: datetime
    unacknowledged_seconds: int | None
    acknowledged_at: datetime | None
    acknowledged_by: StaffActorResponse | None
    completed_at: datetime | None
    completed_by: StaffActorResponse | None


class RequestListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RequestSummaryResponse]
    next_cursor: str | None


class WardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ward_code: str
    ward_name: str


class DashboardCountsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patients_registered_today: int
    requests_today: int
    unacknowledged_requests: int
    critical_open_requests: int


class DashboardSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    timezone: str
    ward: WardResponse
    counts: DashboardCountsResponse
    recent_requests: list[RequestSummaryResponse]



class RequestTimelineEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str
    occurred_at: datetime
    actor: StaffActorResponse | None
    note: str | None


class RequestDetailResponse(RequestSummaryResponse):
    model_config = ConfigDict(extra="forbid")

    resolution_note: str | None
    timeline: list[RequestTimelineEventResponse]


class AcknowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str | None = None



class CompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution_note: str | None = None



class PatientLatestRequestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: int
    text: str
    status: str
    priority: str
    requested_at: datetime


class PatientBoardItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: int
    patient_code: str
    masked_name: str
    room_number: str
    board_status: str
    open_request_count: int
    unacknowledged_request_count: int
    critical_open_count: int
    latest_request: PatientLatestRequestResponse | None


class PatientBoardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ward: WardResponse
    generated_at: datetime
    patients: list[PatientBoardItemResponse]



class FrequentPhraseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase_code: str
    text: str
    count_30d: int


class TodayPhraseCountResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase_code: str
    text: str
    count: int


class PatientTodaySummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: date
    total_requests: int
    by_category: dict[str, int]
    by_phrase: list[TodayPhraseCountResponse]


class PatientDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: int
    patient_code: str
    masked_name: str
    ward: WardResponse
    room_number: str
    admitted_on: date
    communication_status: str
    communication_status_label: str
    assistive_method: str | None
    notes: str | None
    open_request_count: int
    unacknowledged_request_count: int
    latest_request: PatientLatestRequestResponse | None
    frequent_phrases: list[FrequentPhraseResponse]
    today_summary: PatientTodaySummaryResponse



class PatientRequestHistoryItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    acknowledged_by: StaffActorResponse | None
    completed_at: datetime | None
    completed_by: StaffActorResponse | None


class PatientRequestHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PatientRequestHistoryItemResponse]
    next_cursor: str | None
