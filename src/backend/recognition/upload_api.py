"""로그인 사용자의 인식 영상 HTTP 업로드 API."""

import logging
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict

from src.backend.auth.service import InvalidSessionError
from src.backend.recognition.domain import RecognitionMode
from src.backend.recognition.errors import (
    EmptyVideoUploadError,
    IdempotencyConflictError,
    InvalidIdempotencyKeyError,
    UnsupportedVideoMimeTypeError,
    UnsupportedVideoUploadModeError,
    VideoTooLargeError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/recognition",
    tags=["recognition"],
)

UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


class VideoUploadQueuedResponse(BaseModel):
    """Redis 추론 queue 등록까지 완료된 영상 업로드 응답."""

    model_config = ConfigDict(extra="forbid")

    utterance_id: int
    video_id: int
    job_id: str
    status: Literal["QUEUED"] = "QUEUED"
    duplicate: bool


async def _read_upload_file(
    *,
    file: UploadFile,
    max_bytes: int,
) -> bytes:
    """UploadFile을 제한된 chunk로 읽고 최대 크기를 즉시 적용한다."""

    chunks: list[bytes] = []
    total_bytes = 0

    try:
        while True:
            chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)

            if not chunk:
                break

            total_bytes += len(chunk)

            if total_bytes > max_bytes:
                raise VideoTooLargeError("영상 파일 크기가 허용 범위를 초과했습니다.")

            chunks.append(chunk)

        return b"".join(chunks)
    finally:
        await file.close()


def _require_header(
    value: str | None,
    *,
    header_name: str,
) -> str:
    """필수 인증/Idempotency header의 누락과 공백을 거부한다."""

    if value is None or not value.strip():
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
                if header_name == "X-Session-Token"
                else status.HTTP_400_BAD_REQUEST
            ),
            detail=(f"{header_name} header가 필요합니다."),
        )

    return value.strip()


@router.post(
    "/videos",
    response_model=VideoUploadQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_recognition_video(
    request: Request,
    response: Response,
    file: Annotated[
        UploadFile,
        File(),
    ],
    mode: Annotated[
        RecognitionMode,
        Form(),
    ] = RecognitionMode.CLOSED,
    session_token: Annotated[
        str | None,
        Header(alias="X-Session-Token"),
    ] = None,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key"),
    ] = None,
) -> VideoUploadQueuedResponse:
    """영상을 저장하고 Redis 비동기 추론 queue에 Job을 등록한다."""

    session_token = _require_header(
        session_token,
        header_name="X-Session-Token",
    )

    idempotency_key = _require_header(
        idempotency_key,
        header_name="Idempotency-Key",
    )

    auth_service = getattr(
        request.app.state,
        "auth_service",
        None,
    )

    submission_service = getattr(
        request.app.state,
        "submission_service",
        None,
    )

    if auth_service is None or submission_service is None:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=("영상 업로드 서비스를 사용할 수 없습니다."),
        )

    try:
        user = await auth_service.get_current_user(session_token)
    except InvalidSessionError as exc:
        raise HTTPException(
            status_code=(status.HTTP_401_UNAUTHORIZED),
            detail=("유효하지 않거나 만료된 로그인 세션입니다."),
        ) from exc

    try:
        data = await _read_upload_file(
            file=file,
            max_bytes=(request.app.state.settings.max_video_upload_bytes),
        )

        result = await submission_service.submit(
            user_id=user.user_id,
            storage_uuid=user.storage_uuid,
            idempotency_key=idempotency_key,
            data=data,
            content_type=(file.content_type or ""),
            mode=mode,
        )

    except InvalidIdempotencyKeyError as exc:
        raise HTTPException(
            status_code=(status.HTTP_400_BAD_REQUEST),
            detail=str(exc),
        ) from exc

    except EmptyVideoUploadError as exc:
        raise HTTPException(
            status_code=(status.HTTP_400_BAD_REQUEST),
            detail=str(exc),
        ) from exc

    except UnsupportedVideoMimeTypeError as exc:
        raise HTTPException(
            status_code=(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
            detail=str(exc),
        ) from exc

    except UnsupportedVideoUploadModeError as exc:
        raise HTTPException(
            status_code=(status.HTTP_400_BAD_REQUEST),
            detail=str(exc),
        ) from exc

    except VideoTooLargeError as exc:
        raise HTTPException(
            status_code=(status.HTTP_413_CONTENT_TOO_LARGE),
            detail=str(exc),
        ) from exc

    except IdempotencyConflictError as exc:
        raise HTTPException(
            status_code=(status.HTTP_409_CONFLICT),
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            "영상 HTTP 업로드 및 추론 Job 등록 실패: error_type=%s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail=("영상 업로드 처리 중 서버 오류가 발생했습니다."),
        ) from exc

    if result.duplicate:
        response.status_code = status.HTTP_200_OK

    return VideoUploadQueuedResponse(
        utterance_id=(result.asset.utterance_id),
        video_id=(result.asset.video_id),
        job_id=(result.job.job_id),
        duplicate=(result.duplicate),
    )
