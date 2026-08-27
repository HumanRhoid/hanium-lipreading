"""FastAPI 애플리케이션 진입점."""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.auth.adapters.repository import SQLAlchemyAuthRepository
from src.backend.auth.api import router as auth_router
from src.backend.auth.service import AuthService
from src.backend.core import Settings, SQLAlchemyDatabase
from src.backend.health import router as health_router
from src.backend.recognition.adapters.inference import (
    BoundedLocalRecognitionGateway,
    FakeSyncPredictor,
    UnavailableRecognitionGateway,
)
from src.backend.recognition.adapters.media import JpegFrameValidator
from src.backend.recognition.adapters.object_storage import S3ObjectStorage
from src.backend.recognition.adapters.redis_job_queue import (
    RedisInferenceJobQueue,
)
from src.backend.recognition.adapters.repository import (
    SQLAlchemyRecognitionRepository,
)
from src.backend.recognition.api import router as recognition_router
from src.backend.recognition.domain import (
    STREAM_FRAME_HEIGHT,
    STREAM_FRAME_WIDTH,
)
from src.backend.recognition.job_status_api import (
    router as inference_job_status_router,
)
from src.backend.recognition.job_status_service import (
    InferenceJobStatusService,
)
from src.backend.recognition.ports import (
    FrameValidator,
    InferenceJobQueue,
    ObjectStorage,
    RecognitionGateway,
    RecognitionRepository,
)
from src.backend.recognition.service import (
    NoopTextCorrector,
    RecognitionService,
)
from src.backend.recognition.submission_service import (
    RecognitionSubmissionService,
)
from src.backend.recognition.upload_api import router as recognition_upload_router
from src.backend.recognition.video_upload_service import (
    VideoUploadService,
)

CloseCallback = Callable[[], Awaitable[None]]


async def _call_close(
    close: CloseCallback,
) -> None:
    """종료 호출 시점의 예외까지 개별 정리 결과로 격리한다."""

    await close()


async def _collect_close_errors(
    close_callbacks: Sequence[CloseCallback],
) -> tuple[BaseException, ...]:
    """서로 독립적인 자원을 모두 정리하고 발생한 오류를 수집한다."""

    results = await asyncio.gather(
        *(_call_close(close) for close in close_callbacks),
        return_exceptions=True,
    )

    return tuple(
        result
        for result in results
        if isinstance(
            result,
            BaseException,
        )
    )


async def _cleanup_resources(
    dependency_closes: Sequence[CloseCallback],
    database_close: CloseCallback | None,
) -> tuple[BaseException, ...]:
    """실행 자원을 먼저 정리하고 DB engine을 마지막에 닫는다."""

    errors = list(await _collect_close_errors(dependency_closes))

    if database_close is not None:
        errors.extend(await _collect_close_errors((database_close,)))

    return tuple(errors)


async def _wait_for_cleanup(
    cleanup_task: asyncio.Task[
        tuple[
            BaseException,
            ...,
        ]
    ],
) -> tuple[
    tuple[
        BaseException,
        ...,
    ],
    asyncio.CancelledError | None,
]:
    """호출자 취소를 보존하면서 공유 정리 task가 끝날 때까지 보호한다."""

    cancellation: asyncio.CancelledError | None = None

    while True:
        try:
            return (
                await asyncio.shield(cleanup_task),
                cancellation,
            )

        except asyncio.CancelledError as error:
            if cleanup_task.done():
                try:
                    return (
                        cleanup_task.result(),
                        cancellation,
                    )
                except BaseException as cleanup_error:
                    return (
                        (cleanup_error,),
                        cancellation,
                    )

            if cancellation is None:
                cancellation = error

        except BaseException as cleanup_error:
            return (
                (cleanup_error,),
                cancellation,
            )


def _raise_lifecycle_errors(
    errors: Sequence[BaseException],
) -> None:
    """단일 오류는 그대로, 복수 오류는 원인을 잃지 않는 그룹으로 전달한다."""

    if not errors:
        return

    if len(errors) == 1:
        raise errors[0]

    if all(
        isinstance(
            error,
            Exception,
        )
        for error in errors
    ):
        raise ExceptionGroup(
            "애플리케이션 생명주기 오류",
            list(errors),
        )

    raise BaseExceptionGroup(
        "애플리케이션 생명주기 오류",
        list(errors),
    )


def create_gateway(
    settings: Settings,
) -> RecognitionGateway:
    """composition root에서 환경에 맞는 추론 adapter를 선택한다."""

    if settings.inference_backend == "fake":
        return BoundedLocalRecognitionGateway(
            FakeSyncPredictor(),
            max_concurrency=(settings.max_inference_concurrency),
        )

    return UnavailableRecognitionGateway()


def create_app(
    *,
    settings: Settings | None = None,
    database: SQLAlchemyDatabase | None = None,
    repository: RecognitionRepository | None = None,
    gateway: RecognitionGateway | None = None,
    frame_validator: FrameValidator | None = None,
    auth_service: AuthService | None = None,
    object_storage: ObjectStorage | None = None,
    video_upload_service: VideoUploadService | None = None,
    inference_job_queue: InferenceJobQueue | None = None,
    submission_service: RecognitionSubmissionService | None = None,
    job_status_service: InferenceJobStatusService | None = None,
) -> FastAPI:
    """운영 구현과 테스트 대역을 주입할 수 있는 애플리케이션을 생성한다."""

    app_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncGenerator[
        None,
        None,
    ]:
        """애플리케이션 실행 자원을 생성하고 종료 시 정리한다."""

        dependency_closes: list[CloseCallback] = []

        lifecycle_errors: list[BaseException] = []

        database_close: CloseCallback | None = None

        app.state.draining = True

        try:
            app_database = (
                database if database is not None else SQLAlchemyDatabase(app_settings)
            )

            database_close = app_database.close

            app_repository = (
                repository
                if repository is not None
                else SQLAlchemyRecognitionRepository(app_database.session_factory)
            )

            app_auth_service = auth_service

            app_auth_repository = None

            if app_auth_service is None and hasattr(
                app_database,
                "session_factory",
            ):
                app_auth_repository = SQLAlchemyAuthRepository(
                    app_database.session_factory
                )

                app_auth_service = AuthService(repository=(app_auth_repository))

            app_object_storage = (
                object_storage
                if object_storage is not None
                else S3ObjectStorage(app_settings)
            )

            app_video_upload_service = (
                video_upload_service
                if video_upload_service is not None
                else VideoUploadService(
                    repository=app_repository,
                    object_storage=(app_object_storage),
                    max_upload_bytes=(app_settings.max_video_upload_bytes),
                )
            )

            app_gateway = (
                gateway if gateway is not None else create_gateway(app_settings)
            )

            dependency_closes.append(app_gateway.close)

            app_frame_validator = (
                frame_validator
                if frame_validator is not None
                else JpegFrameValidator(
                    width=(STREAM_FRAME_WIDTH),
                    height=(STREAM_FRAME_HEIGHT),
                )
            )

            dependency_closes.append(app_frame_validator.close)

            if inference_job_queue is None:
                app_inference_job_queue = RedisInferenceJobQueue(app_settings)

                dependency_closes.append(app_inference_job_queue.close)
            else:
                app_inference_job_queue = inference_job_queue

            app_submission_service = (
                submission_service
                if submission_service is not None
                else RecognitionSubmissionService(
                    video_upload_service=app_video_upload_service,
                    inference_job_queue=app_inference_job_queue,
                )
            )

            app_job_status_service = (
                job_status_service
                if job_status_service is not None
                else InferenceJobStatusService(
                    repository=app_repository,
                    inference_job_queue=app_inference_job_queue,
                )
            )

            app.state.settings = app_settings

            app.state.database = app_database

            app.state.repository = app_repository

            app.state.gateway = app_gateway

            app.state.frame_validator = app_frame_validator

            app.state.recognition_service = RecognitionService(
                repository=(app_repository),
                gateway=(app_gateway),
                corrector=(NoopTextCorrector()),
                max_active_sessions=(app_settings.max_active_sessions),
            )

            app.state.object_storage = app_object_storage

            app.state.video_upload_service = app_video_upload_service

            app.state.inference_job_queue = app_inference_job_queue

            app.state.submission_service = app_submission_service

            app.state.inference_job_status_service = app_job_status_service

            app.state.auth_repository = app_auth_repository

            app.state.auth_service = app_auth_service

            await app_gateway.start()

            app.state.draining = False

            yield

        except BaseException as error:
            lifecycle_errors.append(error)

        finally:
            app.state.draining = True

        if dependency_closes or database_close is not None:
            cleanup_task = asyncio.create_task(
                _cleanup_resources(
                    tuple(reversed(dependency_closes)),
                    database_close,
                )
            )

            (
                close_errors,
                cancellation,
            ) = await _wait_for_cleanup(cleanup_task)

            if cancellation is not None:
                lifecycle_errors.append(cancellation)

            lifecycle_errors.extend(close_errors)

        _raise_lifecycle_errors(lifecycle_errors)

    app = FastAPI(
        title="한이음 립리딩 API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=(app_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=[
            "GET",
            "POST",
            "OPTIONS",
        ],
        allow_headers=["*"],
    )

    app.include_router(health_router)

    app.include_router(auth_router)

    app.include_router(recognition_router)

    app.include_router(recognition_upload_router)

    app.include_router(inference_job_status_router)

    return app


app = create_app()
