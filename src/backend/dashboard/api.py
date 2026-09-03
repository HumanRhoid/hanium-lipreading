"""의료진 대시보드 HTTP API."""

from datetime import date
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Header,
    Query,
    Request,
    status,
)
from fastapi.responses import JSONResponse

from src.backend.auth.adapters.repository import User
from src.backend.auth.service import InvalidSessionError
from src.backend.dashboard.schemas import (
    PatientRequestHistoryResponse,
    PatientDetailResponse,
    PatientBoardResponse,
    CompleteRequest,
    RequestDetailResponse,
    AcknowledgeRequest,
    DashboardSummaryResponse,
    RequestListResponse,
)
from src.backend.dashboard.service import (
    ALLOWED_BOARD_STATUSES,
    InvalidRequestTransitionError,
    DashboardIdempotencyConflictError,
    ALLOWED_REQUEST_CATEGORIES,
    ALLOWED_REQUEST_PRIORITIES,
    ALLOWED_REQUEST_SORTS,
    ALLOWED_REQUEST_STATUSES,
    DashboardService,
    InvalidDashboardQueryError,
    ResourceNotFoundError,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["dashboard"],
)


class DashboardAPIError(Exception):
    """대시보드 명세 형식으로 반환할 HTTP 오류."""

    def __init__(
        self,
        *,
        status_code: int,
        detail: str,
        code: str,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code


async def dashboard_api_error_handler(
    _request: Request,
    exc: DashboardAPIError,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "code": exc.code,
        },
    )


def get_dashboard_service(
    request: Request,
) -> DashboardService:
    service = getattr(
        request.app.state,
        "dashboard_service",
        None,
    )

    if service is None:
        raise RuntimeError(
            "dashboard service is unavailable"
        )

    return service


def _invalid_query() -> DashboardAPIError:
    return DashboardAPIError(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="잘못된 조회 조건입니다.",
        code="INVALID_QUERY",
    )


def _resource_not_found() -> DashboardAPIError:
    return DashboardAPIError(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="자원을 찾을 수 없습니다.",
        code="RESOURCE_NOT_FOUND",
    )


def _invalid_transition() -> DashboardAPIError:
    return DashboardAPIError(
        status_code=status.HTTP_409_CONFLICT,
        detail="\uc694\uccad \uc0c1\ud0dc\ub97c \ubcc0\uacbd\ud560 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4.",
        code="INVALID_REQUEST_TRANSITION",
    )


def _idempotency_conflict() -> DashboardAPIError:
    return DashboardAPIError(
        status_code=status.HTTP_409_CONFLICT,
        detail="\uac19\uc740 Idempotency-Key\uac00 \ub2e4\ub978 \uc694\uccad\uc5d0 \uc0ac\uc6a9\ub418\uc5c8\uc2b5\ub2c8\ub2e4.",
        code="IDEMPOTENCY_CONFLICT",
    )


async def require_staff_user(
    request: Request,
    session_token: Annotated[
        str | None,
        Header(alias="X-Session-Token"),
    ] = None,
) -> User:
    if session_token is None or not session_token.strip():
        raise DashboardAPIError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효한 로그인 세션이 필요합니다.",
            code="INVALID_SESSION",
        )

    auth_service = getattr(
        request.app.state,
        "auth_service",
        None,
    )

    if auth_service is None:
        raise RuntimeError(
            "auth service is unavailable"
        )

    try:
        user = await auth_service.get_current_user(
            session_token.strip(),
        )
    except InvalidSessionError as exc:
        raise DashboardAPIError(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효한 로그인 세션이 필요합니다.",
            code="INVALID_SESSION",
        ) from exc

    if user.role != "STAFF":
        raise DashboardAPIError(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="의료진 권한이 필요합니다.",
            code="STAFF_ACCESS_REQUIRED",
        )

    return user


async def require_staff_ward_access(
    request: Request,
    staff_user: Annotated[
        User,
        Depends(require_staff_user),
    ],
    ward_code: str | None = None,
) -> User:
    if ward_code is None or not ward_code.strip():
        raise _invalid_query()

    service = get_dashboard_service(request)

    try:
        await service.require_ward_access(
            staff_user_id=staff_user.user_id,
            ward_code=ward_code.strip(),
        )
    except ResourceNotFoundError as exc:
        raise _resource_not_found() from exc

    return staff_user


def _parse_csv(
    value: str | None,
    *,
    default: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if value is None:
        return default

    parts = tuple(
        item.strip().upper()
        for item in value.split(",")
        if item.strip()
    )

    if not parts:
        raise _invalid_query()

    if len(set(parts)) != len(parts):
        raise _invalid_query()

    return parts


@router.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
)
async def get_dashboard_summary(
    request: Request,
    staff_user: Annotated[
        User,
        Depends(require_staff_user),
    ],
    ward_code: str | None = None,
    date_value: Annotated[
        str | None,
        Query(alias="date"),
    ] = None,
    recent_limit_value: Annotated[
        str,
        Query(alias="recent_limit"),
    ] = "5",
) -> DashboardSummaryResponse:
    if ward_code is None or not ward_code.strip():
        raise _invalid_query()

    try:
        recent_limit = int(recent_limit_value)
    except ValueError as exc:
        raise _invalid_query() from exc

    if recent_limit < 1 or recent_limit > 20:
        raise _invalid_query()

    target_date: date | None = None

    if date_value is not None:
        try:
            target_date = date.fromisoformat(
                date_value
            )
        except ValueError as exc:
            raise _invalid_query() from exc

    service = get_dashboard_service(request)

    try:
        return await service.get_summary(
            staff_user_id=staff_user.user_id,
            ward_code=ward_code.strip(),
            target_date=target_date,
            recent_limit=recent_limit,
        )
    except InvalidDashboardQueryError as exc:
        raise _invalid_query() from exc
    except ResourceNotFoundError as exc:
        raise _resource_not_found() from exc


@router.get(
    "/requests",
    response_model=RequestListResponse,
)
async def get_requests(
    request: Request,
    staff_user: Annotated[
        User,
        Depends(require_staff_user),
    ],
    ward_code: str | None = None,
    status_value: Annotated[
        str | None,
        Query(alias="status"),
    ] = None,
    priority_value: Annotated[
        str | None,
        Query(alias="priority"),
    ] = None,
    category: str | None = None,
    patient_id_value: Annotated[
        str | None,
        Query(alias="patient_id"),
    ] = None,
    sort_mode: Annotated[
        str,
        Query(alias="sort"),
    ] = "attention",
    cursor: str | None = None,
    limit_value: Annotated[
        str,
        Query(alias="limit"),
    ] = "20",
) -> RequestListResponse:
    if ward_code is None or not ward_code.strip():
        raise _invalid_query()

    try:
        statuses = _parse_csv(
            status_value,
            default=(
                "NEW",
                "ACKNOWLEDGED",
            ),
        )

        priorities = _parse_csv(
            priority_value,
            default=None,
        )
    except DashboardAPIError:
        raise

    if statuses is None:
        raise _invalid_query()

    if any(
        value not in ALLOWED_REQUEST_STATUSES
        for value in statuses
    ):
        raise _invalid_query()

    if priorities is not None and any(
        value not in ALLOWED_REQUEST_PRIORITIES
        for value in priorities
    ):
        raise _invalid_query()

    normalized_category = (
        category.strip().upper()
        if category is not None
        else None
    )

    if normalized_category == "":
        raise _invalid_query()

    if (
        normalized_category is not None
        and normalized_category
        not in ALLOWED_REQUEST_CATEGORIES
    ):
        raise _invalid_query()

    normalized_sort = sort_mode.strip().lower()

    if normalized_sort not in ALLOWED_REQUEST_SORTS:
        raise _invalid_query()

    patient_id: int | None = None

    if patient_id_value is not None:
        try:
            patient_id = int(patient_id_value)
        except ValueError as exc:
            raise _invalid_query() from exc

        if patient_id < 1:
            raise _invalid_query()

    try:
        limit = int(limit_value)
    except ValueError as exc:
        raise _invalid_query() from exc

    if limit < 1 or limit > 100:
        raise _invalid_query()

    service = get_dashboard_service(request)

    try:
        return await service.get_requests(
            staff_user_id=staff_user.user_id,
            ward_code=ward_code.strip(),
            statuses=statuses,
            priorities=priorities,
            category=normalized_category,
            patient_id=patient_id,
            sort_mode=normalized_sort,
            cursor=cursor,
            limit=limit,
        )
    except InvalidDashboardQueryError as exc:
        raise _invalid_query() from exc
    except ResourceNotFoundError as exc:
        raise _resource_not_found() from exc



@router.post(
    "/requests/{request_id}/acknowledge",
    response_model=RequestDetailResponse,
)
async def acknowledge_request(
    request_id: int,
    body: AcknowledgeRequest,
    request: Request,
    staff_user: Annotated[
        User,
        Depends(require_staff_user),
    ],
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key"),
    ] = None,
) -> RequestDetailResponse:
    """Record the first staff acknowledgement of a NEW request."""

    if request_id < 1:
        raise _invalid_query()

    if idempotency_key is None:
        raise _invalid_query()

    normalized_key = idempotency_key.strip()

    if not normalized_key or len(normalized_key) > 128:
        raise _invalid_query()

    if body.note is not None and len(body.note) > 500:
        raise _invalid_query()

    service = get_dashboard_service(request)

    try:
        return await service.acknowledge_request(
            staff_user_id=staff_user.user_id,
            request_id=request_id,
            idempotency_key=normalized_key,
            note=body.note,
        )
    except InvalidDashboardQueryError as exc:
        raise _invalid_query() from exc
    except ResourceNotFoundError as exc:
        raise _resource_not_found() from exc
    except InvalidRequestTransitionError as exc:
        raise _invalid_transition() from exc
    except DashboardIdempotencyConflictError as exc:
        raise _idempotency_conflict() from exc



@router.post(
    "/requests/{request_id}/complete",
    response_model=RequestDetailResponse,
)
async def complete_request(
    request_id: int,
    body: CompleteRequest,
    request: Request,
    staff_user: Annotated[
        User,
        Depends(require_staff_user),
    ],
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key"),
    ] = None,
) -> RequestDetailResponse:
    """Complete an acknowledged communication request."""

    if request_id < 1:
        raise _invalid_query()

    if idempotency_key is None:
        raise _invalid_query()

    normalized_key = idempotency_key.strip()

    if not normalized_key or len(normalized_key) > 128:
        raise _invalid_query()

    if (
        body.resolution_note is not None
        and len(body.resolution_note) > 500
    ):
        raise _invalid_query()

    service = get_dashboard_service(request)

    try:
        return await service.complete_request(
            staff_user_id=staff_user.user_id,
            request_id=request_id,
            idempotency_key=normalized_key,
            resolution_note=body.resolution_note,
        )
    except InvalidDashboardQueryError as exc:
        raise _invalid_query() from exc
    except ResourceNotFoundError as exc:
        raise _resource_not_found() from exc
    except InvalidRequestTransitionError as exc:
        raise _invalid_transition() from exc
    except DashboardIdempotencyConflictError as exc:
        raise _idempotency_conflict() from exc



@router.get(
    "/patients",
    response_model=PatientBoardResponse,
)
async def get_patient_board(
    request: Request,
    staff_user: Annotated[
        User,
        Depends(require_staff_user),
    ],
    ward_code: str | None = None,
    status_value: Annotated[
        str | None,
        Query(alias="status"),
    ] = None,
) -> PatientBoardResponse:
    """Return the ward patient status board."""

    if ward_code is None or not ward_code.strip():
        raise _invalid_query()

    normalized_status = None

    if status_value is not None:
        normalized_status = status_value.strip().upper()

        if (
            not normalized_status
            or normalized_status
            not in ALLOWED_BOARD_STATUSES
        ):
            raise _invalid_query()

    service = get_dashboard_service(request)

    try:
        return await service.get_patient_board(
            staff_user_id=staff_user.user_id,
            ward_code=ward_code.strip(),
            board_status=normalized_status,
        )
    except InvalidDashboardQueryError as exc:
        raise _invalid_query() from exc
    except ResourceNotFoundError as exc:
        raise _resource_not_found() from exc



@router.get(
    "/patients/{patient_id}",
    response_model=PatientDetailResponse,
)
async def get_patient_detail(
    patient_id: int,
    request: Request,
    staff_user: Annotated[
        User,
        Depends(require_staff_user),
    ],
) -> PatientDetailResponse:
    """Return patient profile, request summary and phrases."""

    if patient_id < 1:
        raise _invalid_query()

    service = get_dashboard_service(request)

    try:
        return await service.get_patient_detail(
            staff_user_id=staff_user.user_id,
            patient_id=patient_id,
        )
    except InvalidDashboardQueryError as exc:
        raise _invalid_query() from exc
    except ResourceNotFoundError as exc:
        raise _resource_not_found() from exc



@router.get(
    "/patients/{patient_id}/requests",
    response_model=PatientRequestHistoryResponse,
)
async def get_patient_requests(
    patient_id: int,
    request: Request,
    staff_user: Annotated[
        User,
        Depends(require_staff_user),
    ],
    date_from_value: Annotated[
        str | None,
        Query(alias="date_from"),
    ] = None,
    date_to_value: Annotated[
        str | None,
        Query(alias="date_to"),
    ] = None,
    status_value: Annotated[
        str | None,
        Query(alias="status"),
    ] = None,
    category: str | None = None,
    cursor: str | None = None,
    limit_value: Annotated[
        str,
        Query(alias="limit"),
    ] = "20",
) -> PatientRequestHistoryResponse:
    """Return one patient's lip-reading request history."""

    if patient_id < 1:
        raise _invalid_query()

    date_from = None
    date_to = None

    if date_from_value is not None:
        try:
            date_from = date.fromisoformat(
                date_from_value
            )
        except ValueError as exc:
            raise _invalid_query() from exc

    if date_to_value is not None:
        try:
            date_to = date.fromisoformat(
                date_to_value
            )
        except ValueError as exc:
            raise _invalid_query() from exc

    if (
        date_from is not None
        and date_to is not None
        and date_from > date_to
    ):
        raise _invalid_query()

    normalized_status = None

    if status_value is not None:
        normalized_status = (
            status_value.strip().upper()
        )

        if (
            not normalized_status
            or normalized_status
            not in ALLOWED_REQUEST_STATUSES
        ):
            raise _invalid_query()

    normalized_category = None

    if category is not None:
        normalized_category = (
            category.strip().upper()
        )

        if (
            not normalized_category
            or normalized_category
            not in ALLOWED_REQUEST_CATEGORIES
        ):
            raise _invalid_query()

    if cursor is not None and not cursor.strip():
        raise _invalid_query()

    try:
        limit = int(limit_value)
    except ValueError as exc:
        raise _invalid_query() from exc

    if limit < 1 or limit > 100:
        raise _invalid_query()

    service = get_dashboard_service(request)

    try:
        return await service.get_patient_requests(
            staff_user_id=staff_user.user_id,
            patient_id=patient_id,
            date_from=date_from,
            date_to=date_to,
            status_filter=normalized_status,
            category=normalized_category,
            cursor=(
                cursor.strip()
                if cursor is not None
                else None
            ),
            limit=limit,
        )
    except InvalidDashboardQueryError as exc:
        raise _invalid_query() from exc
    except ResourceNotFoundError as exc:
        raise _resource_not_found() from exc
