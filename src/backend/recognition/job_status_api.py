"""로그인 사용자의 비동기 추론 Job 상태 조회 HTTP API."""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Request,
    status,
)
from pydantic import BaseModel, ConfigDict

from src.backend.auth.service import InvalidSessionError
from src.backend.recognition.ports import InferenceJobStatus

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/inference-jobs",
    tags=["inference-jobs"],
)


class InferenceJobStatusResponse(BaseModel):
    """클라이언트에 공개하는 비동기 추론 Job 상태."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    utterance_id: int
    video_id: int
    status: InferenceJobStatus
    error_code: str | None


def _require_session_token(
    value: str | None,
) -> str:
    """필수 로그인 세션 header의 누락과 공백을 거부한다."""

    if value is None or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Session-Token header가 필요합니다.",
        )

    return value.strip()


@router.get(
    "/{job_id}",
    response_model=InferenceJobStatusResponse,
)
async def get_inference_job_status(
    request: Request,
    job_id: UUID,
    session_token: Annotated[
        str | None,
        Header(alias="X-Session-Token"),
    ] = None,
) -> InferenceJobStatusResponse:
    """현재 로그인 사용자가 소유한 추론 Job 상태를 반환한다."""

    session_token = _require_session_token(session_token)

    auth_service = getattr(
        request.app.state,
        "auth_service",
        None,
    )

    job_status_service = getattr(
        request.app.state,
        "inference_job_status_service",
        None,
    )

    if auth_service is None or job_status_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="추론 Job 조회 서비스를 사용할 수 없습니다.",
        )

    try:
        user = await auth_service.get_current_user(session_token)

    except InvalidSessionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 만료된 로그인 세션입니다.",
        ) from exc

    try:
        job = await job_status_service.get_for_user(
            user_id=user.user_id,
            job_id=str(job_id),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            "추론 Job 상태 조회 실패: error_type=%s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="추론 Job 조회 중 서버 오류가 발생했습니다.",
        ) from exc

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="추론 Job을 찾을 수 없습니다.",
        )

    return InferenceJobStatusResponse(
        job_id=job.job_id,
        utterance_id=job.utterance_id,
        video_id=job.video_id,
        status=job.status,
        error_code=job.error_code,
    )
