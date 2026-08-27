"""인식 영상 HTTP 업로드 API 계약을 테스트한다."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

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
from src.backend.recognition.ports import (
    InferenceJobRecord,
    VideoAssetRecord,
)
from src.backend.recognition.submission_service import (
    RecognitionSubmissionResult,
)
from src.backend.recognition.upload_api import router as upload_router

SESSION_TOKEN = "test-session-token"

IDEMPOTENCY_KEY = "12345678-1234-4234-8234-123456789abc"

STORAGE_UUID = UUID("11111111-2222-4333-8444-555555555555")

JOB_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

CREATED_AT = datetime(
    2026,
    8,
    27,
    15,
    0,
    tzinfo=UTC,
)


class FakeAuthService:
    def __init__(
        self,
        *,
        invalid_session: bool = False,
    ) -> None:
        self.invalid_session = invalid_session
        self.received_tokens: list[str] = []

    async def get_current_user(
        self,
        session_token: str,
    ):
        self.received_tokens.append(session_token)

        if self.invalid_session:
            raise InvalidSessionError("invalid session")

        return SimpleNamespace(
            user_id=7,
            storage_uuid=STORAGE_UUID,
        )


class FakeSubmissionService:
    def __init__(
        self,
        *,
        result: RecognitionSubmissionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error

        self.calls: list[dict[str, object]] = []

    async def submit(
        self,
        *,
        user_id: int,
        storage_uuid: UUID,
        idempotency_key: str,
        data: bytes,
        content_type: str,
        mode: RecognitionMode,
    ) -> RecognitionSubmissionResult:
        self.calls.append(
            {
                "user_id": user_id,
                "storage_uuid": storage_uuid,
                "idempotency_key": idempotency_key,
                "data": data,
                "content_type": content_type,
                "mode": mode,
            }
        )

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError("result가 설정되지 않았습니다.")

        return self.result


def make_asset() -> VideoAssetRecord:
    return VideoAssetRecord(
        video_id=45,
        utterance_id=123,
        user_id=7,
        idempotency_key=IDEMPOTENCY_KEY,
        object_key=(
            f"{STORAGE_UUID}/2026/08/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee.webm"
        ),
        original_mime_type="video/webm",
        size_bytes=5,
        checksum="a" * 64,
        storage_status="UPLOADED",
        storage_purpose="TEMPORARY_INFERENCE",
        created_at=CREATED_AT,
        retention_until=None,
    )


def make_job() -> InferenceJobRecord:
    asset = make_asset()

    return InferenceJobRecord(
        job_id=JOB_ID,
        utterance_id=asset.utterance_id,
        video_id=asset.video_id,
        object_key=asset.object_key,
        mode=RecognitionMode.CLOSED,
        status="QUEUED",
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
        error_code=None,
    )


def make_submission_result(
    *,
    duplicate: bool = False,
) -> RecognitionSubmissionResult:
    return RecognitionSubmissionResult(
        asset=make_asset(),
        job=make_job(),
        duplicate=duplicate,
    )


def make_app(
    *,
    auth_service: FakeAuthService | None = None,
    submission_service: FakeSubmissionService | None = None,
    max_upload_bytes: int = 1024,
) -> FastAPI:
    app = FastAPI()

    app.include_router(upload_router)

    app.state.settings = SimpleNamespace(max_video_upload_bytes=(max_upload_bytes))

    app.state.auth_service = (
        auth_service if auth_service is not None else FakeAuthService()
    )

    app.state.submission_service = (
        submission_service
        if submission_service is not None
        else FakeSubmissionService(
            result=make_submission_result(),
        )
    )

    return app


async def request_upload(
    app: FastAPI,
    *,
    session_token: str | None = SESSION_TOKEN,
    idempotency_key: str | None = IDEMPOTENCY_KEY,
    data: bytes = b"video",
    content_type: str = "video/webm",
    mode: str = "CLOSED",
):
    headers = {}

    if session_token is not None:
        headers["X-Session-Token"] = session_token

    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/api/v1/recognition/videos",
            headers=headers,
            files={
                "file": (
                    "clip.webm",
                    data,
                    content_type,
                )
            },
            data={
                "mode": mode,
            },
        )


async def test_upload_returns_202_for_new_queued_video():
    auth_service = FakeAuthService()

    submission_service = FakeSubmissionService(
        result=make_submission_result(
            duplicate=False,
        )
    )

    app = make_app(
        auth_service=auth_service,
        submission_service=submission_service,
    )

    response = await request_upload(app)

    assert response.status_code == 202

    assert response.json() == {
        "utterance_id": 123,
        "video_id": 45,
        "job_id": JOB_ID,
        "status": "QUEUED",
        "duplicate": False,
    }

    assert auth_service.received_tokens == [SESSION_TOKEN]

    assert len(submission_service.calls) == 1

    call = submission_service.calls[0]

    assert call["user_id"] == 7
    assert call["storage_uuid"] == STORAGE_UUID
    assert call["idempotency_key"] == IDEMPOTENCY_KEY
    assert call["data"] == b"video"
    assert call["content_type"] == "video/webm"
    assert call["mode"] is RecognitionMode.CLOSED


async def test_upload_returns_200_for_duplicate_queued_request():
    submission_service = FakeSubmissionService(
        result=make_submission_result(
            duplicate=True,
        )
    )

    app = make_app(submission_service=submission_service)

    response = await request_upload(app)

    assert response.status_code == 200

    assert response.json() == {
        "utterance_id": 123,
        "video_id": 45,
        "job_id": JOB_ID,
        "status": "QUEUED",
        "duplicate": True,
    }


async def test_upload_does_not_expose_storage_information():
    app = make_app()

    response = await request_upload(app)

    payload = response.json()

    assert "object_key" not in payload
    assert "storage_uuid" not in payload
    assert "user_id" not in payload


async def test_missing_session_token_returns_401():
    submission_service = FakeSubmissionService(result=make_submission_result())

    app = make_app(submission_service=submission_service)

    response = await request_upload(
        app,
        session_token=None,
    )

    assert response.status_code == 401

    assert submission_service.calls == []


async def test_invalid_session_token_returns_401():
    auth_service = FakeAuthService(invalid_session=True)

    submission_service = FakeSubmissionService(result=make_submission_result())

    app = make_app(
        auth_service=auth_service,
        submission_service=submission_service,
    )

    response = await request_upload(app)

    assert response.status_code == 401

    assert submission_service.calls == []


async def test_missing_idempotency_key_returns_400():
    submission_service = FakeSubmissionService(result=make_submission_result())

    app = make_app(submission_service=submission_service)

    response = await request_upload(
        app,
        idempotency_key=None,
    )

    assert response.status_code == 400

    assert submission_service.calls == []


async def test_invalid_idempotency_key_returns_400():
    submission_service = FakeSubmissionService(
        error=InvalidIdempotencyKeyError("Idempotency-Key는 UUID 형식이어야 합니다.")
    )

    app = make_app(submission_service=submission_service)

    response = await request_upload(
        app,
        idempotency_key="not-a-uuid",
    )

    assert response.status_code == 400


async def test_empty_video_returns_400():
    submission_service = FakeSubmissionService(
        error=EmptyVideoUploadError("빈 영상 파일은 업로드할 수 없습니다.")
    )

    app = make_app(submission_service=submission_service)

    response = await request_upload(
        app,
        data=b"",
    )

    assert response.status_code == 400


async def test_unsupported_mime_type_returns_415():
    submission_service = FakeSubmissionService(
        error=UnsupportedVideoMimeTypeError("지원하지 않는 영상 형식입니다.")
    )

    app = make_app(submission_service=submission_service)

    response = await request_upload(
        app,
        content_type="image/jpeg",
    )

    assert response.status_code == 415


async def test_idempotency_conflict_returns_409():
    submission_service = FakeSubmissionService(
        error=IdempotencyConflictError(
            "같은 Idempotency-Key가 다른 영상에 사용되었습니다."
        )
    )

    app = make_app(submission_service=submission_service)

    response = await request_upload(app)

    assert response.status_code == 409


async def test_oversized_video_returns_413_before_service_call():
    submission_service = FakeSubmissionService(result=make_submission_result())

    app = make_app(
        submission_service=submission_service,
        max_upload_bytes=4,
    )

    response = await request_upload(
        app,
        data=b"12345",
    )

    assert response.status_code == 413

    assert submission_service.calls == []


async def test_service_video_too_large_error_returns_413():
    submission_service = FakeSubmissionService(
        error=VideoTooLargeError("영상 파일 크기가 허용 범위를 초과했습니다.")
    )

    app = make_app(submission_service=submission_service)

    response = await request_upload(app)

    assert response.status_code == 413


async def test_open_mode_returns_400_when_service_rejects_it():
    submission_service = FakeSubmissionService(
        error=UnsupportedVideoUploadModeError(
            "현재 영상 업로드는 CLOSED mode만 지원합니다."
        )
    )

    app = make_app(submission_service=submission_service)

    response = await request_upload(
        app,
        mode="OPEN",
    )

    assert response.status_code == 400

    assert submission_service.calls[0]["mode"] is RecognitionMode.OPEN


async def test_invalid_mode_returns_422_before_service_call():
    submission_service = FakeSubmissionService(result=make_submission_result())

    app = make_app(submission_service=submission_service)

    response = await request_upload(
        app,
        mode="INVALID",
    )

    assert response.status_code == 422

    assert submission_service.calls == []


async def test_unexpected_service_error_returns_safe_500():
    submission_service = FakeSubmissionService(
        error=RuntimeError("redis-password=secret-value")
    )

    app = make_app(submission_service=submission_service)

    response = await request_upload(app)

    assert response.status_code == 500

    body = response.text

    assert "secret-value" not in body

    assert response.json() == {
        "detail": ("영상 업로드 처리 중 서버 오류가 발생했습니다.")
    }


async def test_missing_upload_dependencies_returns_503():
    app = make_app()

    app.state.submission_service = None

    response = await request_upload(app)

    assert response.status_code == 503
