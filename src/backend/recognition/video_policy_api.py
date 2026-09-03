"""Current-user model-training consent and video management API."""

import logging
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

from src.backend.auth.service import InvalidSessionError
from src.backend.recognition.video_policy_service import (
    VideoNotFoundError,
    VideoPolicyService,
)
from src.backend.recognition.video_policy_types import (
    UserConsentRecord,
    VideoPolicyAssetRecord,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/users/me",
    tags=["user-video-policy"],
)


class ConsentUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_training_consent: bool
    consent_version: str = Field(
        min_length=1,
        max_length=50,
    )


class ConsentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_training_consent: bool
    consent_version: str | None
    updated_at: datetime | None


class VideoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: int
    user_id: int
    utterance_id: int

    original_mime_type: str
    normalized_mime_type: str | None

    codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    duration_ms: int | None

    size_bytes: int
    checksum: str

    storage_status: str
    storage_purpose: str
    consent_version: str | None

    created_at: datetime
    retention_until: datetime | None
    deleted_at: datetime | None


class VideoListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    videos: list[VideoResponse]


class DeleteVideosResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deleted_count: int


def _require_session_token(
    value: str | None,
) -> str:
    if value is None or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Session-Token header가 필요합니다.",
        )

    return value.strip()


def _get_policy_service(
    request: Request,
) -> VideoPolicyService:
    repository = getattr(
        request.app.state,
        "repository",
        None,
    )

    object_storage = getattr(
        request.app.state,
        "object_storage",
        None,
    )

    if (
        repository is None
        or object_storage is None
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="영상 관리 서비스를 사용할 수 없습니다.",
        )

    return VideoPolicyService(
        repository=repository,
        object_storage=object_storage,
    )


async def _get_current_user(
    request: Request,
    session_token: str | None,
):
    token = _require_session_token(session_token)

    auth_service = getattr(
        request.app.state,
        "auth_service",
        None,
    )

    if auth_service is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="인증 서비스를 사용할 수 없습니다.",
        )

    try:
        return await auth_service.get_current_user(
            token
        )
    except InvalidSessionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 만료된 로그인 세션입니다.",
        ) from exc


def _consent_response(
    consent: UserConsentRecord | None,
) -> ConsentResponse:
    if consent is None:
        return ConsentResponse(
            model_training_consent=False,
            consent_version=None,
            updated_at=None,
        )

    return ConsentResponse(
        model_training_consent=(
            consent.model_training_consent
        ),
        consent_version=consent.consent_version,
        updated_at=consent.updated_at,
    )


def _video_response(
    asset: VideoPolicyAssetRecord,
) -> VideoResponse:
    # object_key is intentionally never exposed.
    return VideoResponse(
        video_id=asset.video_id,
        user_id=asset.user_id,
        utterance_id=asset.utterance_id,
        original_mime_type=(
            asset.original_mime_type
        ),
        normalized_mime_type=(
            asset.normalized_mime_type
        ),
        codec=asset.codec,
        width=asset.width,
        height=asset.height,
        fps=asset.fps,
        duration_ms=asset.duration_ms,
        size_bytes=asset.size_bytes,
        checksum=asset.checksum,
        storage_status=asset.storage_status,
        storage_purpose=asset.storage_purpose,
        consent_version=asset.consent_version,
        created_at=asset.created_at,
        retention_until=asset.retention_until,
        deleted_at=asset.deleted_at,
    )


@router.get(
    "/consents",
    response_model=ConsentResponse,
)
async def get_my_consent(
    request: Request,
    session_token: Annotated[
        str | None,
        Header(alias="X-Session-Token"),
    ] = None,
) -> ConsentResponse:
    user = await _get_current_user(
        request,
        session_token,
    )

    service = _get_policy_service(request)

    consent = await service.get_consent(
        user_id=user.user_id
    )

    return _consent_response(consent)


@router.patch(
    "/consents",
    response_model=ConsentResponse,
)
async def update_my_consent(
    request: Request,
    body: ConsentUpdateRequest,
    session_token: Annotated[
        str | None,
        Header(alias="X-Session-Token"),
    ] = None,
) -> ConsentResponse:
    user = await _get_current_user(
        request,
        session_token,
    )

    version = body.consent_version.strip()

    if not version:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="consent_version은 비어 있을 수 없습니다.",
        )

    service = _get_policy_service(request)

    try:
        consent = await service.update_consent(
            user_id=user.user_id,
            model_training_consent=(
                body.model_training_consent
            ),
            consent_version=version,
        )
    except Exception as exc:
        logger.exception(
            "Failed to update video consent for user_id=%s",
            user.user_id,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="영상 동의 상태 변경 중 서버 오류가 발생했습니다.",
        ) from exc

    return _consent_response(consent)


@router.get(
    "/videos",
    response_model=VideoListResponse,
)
async def get_my_videos(
    request: Request,
    session_token: Annotated[
        str | None,
        Header(alias="X-Session-Token"),
    ] = None,
) -> VideoListResponse:
    user = await _get_current_user(
        request,
        session_token,
    )

    service = _get_policy_service(request)

    assets = await service.list_videos(
        user_id=user.user_id
    )

    return VideoListResponse(
        videos=[
            _video_response(asset)
            for asset in assets
        ]
    )


@router.delete(
    "/videos/{video_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_my_video(
    request: Request,
    video_id: int,
    session_token: Annotated[
        str | None,
        Header(alias="X-Session-Token"),
    ] = None,
) -> Response:
    user = await _get_current_user(
        request,
        session_token,
    )

    if video_id < 1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="영상을 찾을 수 없습니다.",
        )

    service = _get_policy_service(request)

    try:
        await service.delete_video(
            user_id=user.user_id,
            video_id=video_id,
        )
    except VideoNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="영상을 찾을 수 없습니다.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Failed to delete video_id=%s for user_id=%s",
            video_id,
            user.user_id,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="영상 삭제 중 서버 오류가 발생했습니다.",
        ) from exc

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )


@router.delete(
    "/videos",
    response_model=DeleteVideosResponse,
)
async def delete_all_my_videos(
    request: Request,
    session_token: Annotated[
        str | None,
        Header(alias="X-Session-Token"),
    ] = None,
) -> DeleteVideosResponse:
    user = await _get_current_user(
        request,
        session_token,
    )

    service = _get_policy_service(request)

    try:
        deleted_count = await service.delete_all_videos(
            user_id=user.user_id
        )
    except Exception as exc:
        logger.exception(
            "Failed to delete all videos for user_id=%s",
            user.user_id,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="영상 전체 삭제 중 서버 오류가 발생했습니다.",
        ) from exc

    return DeleteVideosResponse(
        deleted_count=deleted_count
    )
