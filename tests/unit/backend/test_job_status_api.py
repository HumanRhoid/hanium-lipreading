"""비동기 추론 Job 상태 조회 HTTP API 계약을 테스트한다."""

from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.backend.auth.service import InvalidSessionError
from src.backend.recognition.domain import RecognitionMode
from src.backend.recognition.job_status_api import (
    router as job_status_router,
)
from src.backend.recognition.ports import InferenceJobRecord

SESSION_TOKEN = "test-session-token"

JOB_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

CREATED_AT = datetime(
    2026,
    8,
    27,
    15,
    0,
    tzinfo=UTC,
)


def make_job(
    *,
    status: str = "QUEUED",
    error_code: str | None = None,
) -> InferenceJobRecord:
    return InferenceJobRecord(
        job_id=JOB_ID,
        utterance_id=123,
        video_id=45,
        object_key=("storage-user/2026/08/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee.webm"),
        mode=RecognitionMode.CLOSED,
        status=status,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
        error_code=error_code,
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
        )


class FakeInferenceJobStatusService:
    def __init__(
        self,
        *,
        job: InferenceJobRecord | None = None,
        error: Exception | None = None,
    ) -> None:
        self.job = job
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def get_for_user(
        self,
        *,
        user_id: int,
        job_id: str,
    ) -> InferenceJobRecord | None:
        self.calls.append(
            {
                "user_id": user_id,
                "job_id": job_id,
            }
        )

        if self.error is not None:
            raise self.error

        return self.job


def make_app(
    *,
    auth_service: FakeAuthService | None = None,
    job_status_service: FakeInferenceJobStatusService | None = None,
) -> FastAPI:
    app = FastAPI()

    app.include_router(job_status_router)

    app.state.auth_service = (
        auth_service if auth_service is not None else FakeAuthService()
    )

    app.state.inference_job_status_service = (
        job_status_service
        if job_status_service is not None
        else FakeInferenceJobStatusService(
            job=make_job(),
        )
    )

    return app


async def request_job_status(
    app: FastAPI,
    *,
    job_id: str = JOB_ID,
    session_token: str | None = SESSION_TOKEN,
):
    headers = {}

    if session_token is not None:
        headers["X-Session-Token"] = session_token

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get(
            f"/api/v1/inference-jobs/{job_id}",
            headers=headers,
        )


async def test_get_job_status_returns_owned_job():
    auth_service = FakeAuthService()

    job_status_service = FakeInferenceJobStatusService(
        job=make_job(),
    )

    app = make_app(
        auth_service=auth_service,
        job_status_service=job_status_service,
    )

    response = await request_job_status(app)

    assert response.status_code == 200

    assert response.json() == {
        "job_id": JOB_ID,
        "utterance_id": 123,
        "video_id": 45,
        "status": "QUEUED",
        "error_code": None,
    }

    assert auth_service.received_tokens == [SESSION_TOKEN]

    assert job_status_service.calls == [
        {
            "user_id": 7,
            "job_id": JOB_ID,
        }
    ]


async def test_get_job_status_returns_processing_status():
    app = make_app(
        job_status_service=(
            FakeInferenceJobStatusService(
                job=make_job(status="PROCESSING"),
            )
        )
    )

    response = await request_job_status(app)

    assert response.status_code == 200

    assert response.json()["status"] == "PROCESSING"


async def test_get_job_status_returns_succeeded_status():
    app = make_app(
        job_status_service=(
            FakeInferenceJobStatusService(
                job=make_job(status="SUCCEEDED"),
            )
        )
    )

    response = await request_job_status(app)

    assert response.status_code == 200

    assert response.json()["status"] == "SUCCEEDED"


async def test_get_job_status_returns_failed_status_and_error_code():
    app = make_app(
        job_status_service=(
            FakeInferenceJobStatusService(
                job=make_job(
                    status="FAILED",
                    error_code="INFERENCE_FAILED",
                ),
            )
        )
    )

    response = await request_job_status(app)

    assert response.status_code == 200

    assert response.json() == {
        "job_id": JOB_ID,
        "utterance_id": 123,
        "video_id": 45,
        "status": "FAILED",
        "error_code": "INFERENCE_FAILED",
    }


async def test_get_job_status_does_not_expose_internal_fields():
    app = make_app()

    response = await request_job_status(app)

    payload = response.json()

    assert "object_key" not in payload
    assert "mode" not in payload
    assert "created_at" not in payload
    assert "updated_at" not in payload


async def test_unknown_or_unowned_job_returns_404():
    app = make_app(
        job_status_service=(
            FakeInferenceJobStatusService(
                job=None,
            )
        )
    )

    response = await request_job_status(app)

    assert response.status_code == 404

    assert response.json() == {"detail": "추론 Job을 찾을 수 없습니다."}


async def test_missing_session_token_returns_401():
    job_status_service = FakeInferenceJobStatusService(
        job=make_job(),
    )

    app = make_app(job_status_service=job_status_service)

    response = await request_job_status(
        app,
        session_token=None,
    )

    assert response.status_code == 401

    assert job_status_service.calls == []


async def test_invalid_session_token_returns_401():
    auth_service = FakeAuthService(invalid_session=True)

    job_status_service = FakeInferenceJobStatusService(
        job=make_job(),
    )

    app = make_app(
        auth_service=auth_service,
        job_status_service=job_status_service,
    )

    response = await request_job_status(app)

    assert response.status_code == 401

    assert job_status_service.calls == []


async def test_invalid_job_id_returns_422_before_service_call():
    job_status_service = FakeInferenceJobStatusService(
        job=make_job(),
    )

    app = make_app(job_status_service=job_status_service)

    response = await request_job_status(
        app,
        job_id="not-a-uuid",
    )

    assert response.status_code == 422

    assert job_status_service.calls == []


async def test_unexpected_service_error_returns_safe_500():
    job_status_service = FakeInferenceJobStatusService(
        error=RuntimeError("redis-password=secret-value"),
    )

    app = make_app(job_status_service=job_status_service)

    response = await request_job_status(app)

    assert response.status_code == 500

    assert "secret-value" not in response.text

    assert response.json() == {"detail": ("추론 Job 조회 중 서버 오류가 발생했습니다.")}


async def test_missing_dependencies_returns_503():
    app = make_app()

    app.state.inference_job_status_service = None

    response = await request_job_status(app)

    assert response.status_code == 503
