"""FastAPI 애플리케이션 진입점."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.core import Settings, SQLAlchemyDatabase
from src.backend.health import router as health_router
from src.backend.recognition.adapters.inference import (
    BoundedLocalRecognitionGateway,
    FakeSyncPredictor,
    UnavailableRecognitionGateway,
)
from src.backend.recognition.adapters.media import JpegFrameValidator
from src.backend.recognition.adapters.repository import (
    SQLAlchemyRecognitionRepository,
)
from src.backend.recognition.api import router as recognition_router
from src.backend.recognition.domain import INPUT_FRAME_HEIGHT, INPUT_FRAME_WIDTH
from src.backend.recognition.ports import (
    FrameValidator,
    RecognitionGateway,
    RecognitionRepository,
)
from src.backend.recognition.service import NoopTextCorrector, RecognitionService

CloseCallback = Callable[[], Awaitable[None]]
# → 정리해야 할 close() 함수들을 리스트에 모아뒀다가, 종료 시 한꺼번에 실행.


async def _call_close(close: CloseCallback) -> None:
    """동기 호출 시점의 예외까지 개별 정리 결과로 격리한다."""

    await close()


async def _collect_close_errors(
    close_callbacks: Sequence[CloseCallback],
) -> tuple[BaseException, ...]:
    """서로 독립적인 자원을 모두 정리하고 발생한 오류를 수집한다."""

    results = await asyncio.gather(
        *(_call_close(close) for close in close_callbacks),
        return_exceptions=True,
    )
    return tuple(result for result in results if isinstance(result, BaseException))


async def _cleanup_resources(
    dependency_closes: Sequence[CloseCallback],
    database_close: CloseCallback | None,
) -> tuple[BaseException, ...]:
    """실행 자원을 먼저 drain하고 DB engine은 마지막에 닫는다."""

    errors = list(await _collect_close_errors(dependency_closes))
    if database_close is not None:
        errors.extend(await _collect_close_errors((database_close,)))
    return tuple(errors)


async def _wait_for_cleanup(
    cleanup_task: asyncio.Task[tuple[BaseException, ...]],
) -> tuple[tuple[BaseException, ...], asyncio.CancelledError | None]:
    """호출자 취소를 보존하면서 공유 정리 task가 끝날 때까지 보호한다."""

    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return await asyncio.shield(cleanup_task), cancellation
        except asyncio.CancelledError as error:
            # shield 대상이 아니라 현재 lifespan task가 취소된 경우에만 기록한다.
            if cleanup_task.done():
                try:
                    return cleanup_task.result(), cancellation
                except BaseException as cleanup_error:
                    return (cleanup_error,), cancellation
            if cancellation is None:
                cancellation = error
        except BaseException as cleanup_error:
            return (cleanup_error,), cancellation


def _raise_lifecycle_errors(errors: Sequence[BaseException]) -> None:
    """단일 오류는 그대로, 복수 오류는 원인을 잃지 않는 그룹으로 전달한다."""

    if not errors:
        return
    if len(errors) == 1:
        raise errors[0]
    if all(isinstance(error, Exception) for error in errors):
        raise ExceptionGroup("애플리케이션 수명주기 오류", list(errors))
    raise BaseExceptionGroup("애플리케이션 수명주기 오류", list(errors))


def create_gateway(settings: Settings):
    """composition root에서 환경에 맞는 추론 adapter를 선택한다."""

    if settings.inference_backend == "fake":
        return BoundedLocalRecognitionGateway(
            FakeSyncPredictor(),
            max_concurrency=settings.max_inference_concurrency,
        )
    return UnavailableRecognitionGateway()


# create_app에 port로 부터 프레임 조건 추가 및 타입 힌트 추가
def create_app(
    *,
    settings: Settings | None = None,
    database: SQLAlchemyDatabase | None = None,
    repository: RecognitionRepository | None = None,
    gateway: RecognitionGateway | None = None,
    frame_validator: FrameValidator | None = None,
) -> FastAPI:
    """운영 구현과
    테스트 대역을 주입할 수 있는
    애플리케이션을 생성한다.
    """
    app_settings = settings or Settings()

    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        dependency_closes: list[CloseCallback] = []
        lifecycle_errors: list[BaseException] = []
        database_close: CloseCallback | None = None
        app.state.draining = True
        try:
            # 데베 연결 (SQLAlchemyDatabase)과 repository는
            # lifespan에서 생성하고,
            # lifespan 종료 시점에 close()를 호출한다.
            # 삼항연산자로 database와 repository가
            # None이면 기본 구현을 생성하고,
            # None이 아니면 주입된 구현을 사용한다.
            app_database = (
                database if database is not None else SQLAlchemyDatabase(app_settings)
            )
            database_close = app_database.close

            app_repository = (
                repository
                if repository is not None
                else SQLAlchemyRecognitionRepository(app_database.session_factory)
            )
            app_gateway = (
                gateway if gateway is not None else create_gateway(app_settings)
            )
            dependency_closes.append(app_gateway.close)

            app_frame_validator = (
                frame_validator
                if frame_validator is not None
                else JpegFrameValidator(
                    width=INPUT_FRAME_WIDTH,
                    height=INPUT_FRAME_HEIGHT,
                )
            )
            dependency_closes.append(app_frame_validator.close)

            app.state.settings = app_settings
            app.state.database = app_database
            app.state.repository = app_repository
            app.state.gateway = app_gateway
            app.state.frame_validator = app_frame_validator
            app.state.recognition_service = RecognitionService(
                repository=app_repository,
                gateway=app_gateway,
                corrector=NoopTextCorrector(),
                max_active_sessions=app_settings.max_active_sessions,
            )

            await app_gateway.start()
            app.state.draining = False
            yield
        except BaseException as error:
            lifecycle_errors.append(error)
        finally:
            app.state.draining = True

        if dependency_closes or database_close is not None:
            # validator와 gateway를 먼저 drain하고 세션 정리에 필요한 DB는 마지막에 닫는다.
            cleanup_task = asyncio.create_task(
                _cleanup_resources(
                    tuple(reversed(dependency_closes)),
                    database_close,
                )
            )
            close_errors, cancellation = await _wait_for_cleanup(cleanup_task)
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
        allow_origins=app_settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(recognition_router)

    return app


app = create_app()
