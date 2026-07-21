"""FastAPI composition root의 자원 수명주기 계약 테스트."""

import asyncio

import pytest

from src.backend import main as main_module
from src.backend.core.config import Settings
from src.backend.main import create_app


class LifecycleError(RuntimeError):
    """테스트에서 오류 객체의 정체성까지 확인하기 위한 예외."""


class FakeDatabase:
    def __init__(self, *, close_error: BaseException | None = None):
        self.close_error = close_error
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeGateway:
    def __init__(
        self,
        *,
        start_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ):
        self.ready = True
        self.start_error = start_error
        self.close_error = close_error
        self.start_calls = 0
        self.close_calls = 0

    async def start(self):
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    async def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeFrameValidator:
    def __init__(self, *, close_error: BaseException | None = None):
        self.close_error = close_error
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class BlockingResource:
    """종료 호출의 시작과 완료를 외부에서 제어하는 자원 대역."""

    def __init__(self, release: asyncio.Event):
        self.release = release
        self.close_started = asyncio.Event()
        self.close_calls = 0
        self.closed = False

    async def close(self):
        self.close_calls += 1
        self.close_started.set()
        await self.release.wait()
        self.closed = True


class BlockingFailingResource(BlockingResource):
    def __init__(self, release: asyncio.Event, close_error: BaseException):
        super().__init__(release)
        self.close_error = close_error

    async def close(self):
        self.close_calls += 1
        self.close_started.set()
        await self.release.wait()
        raise self.close_error


class BlockingGateway(BlockingResource):
    ready = True

    def __init__(self, release: asyncio.Event):
        super().__init__(release)
        self.start_calls = 0

    async def start(self):
        self.start_calls += 1


def build_settings() -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        database_url=("postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"),
        inference_backend="fake",
    )


def build_app(database, gateway, validator):
    return create_app(
        settings=build_settings(),
        database=database,
        repository=object(),
        gateway=gateway,
        frame_validator=validator,
    )


async def test_start_failure_closes_every_initialized_resource():
    start_error = LifecycleError("gateway start failed")
    database = FakeDatabase()
    gateway = FakeGateway(start_error=start_error)
    validator = FakeFrameValidator()
    app = build_app(database, gateway, validator)

    with pytest.raises(LifecycleError) as raised:
        async with app.router.lifespan_context(app):
            pytest.fail("시작에 실패했으므로 요청 처리 단계에 진입하면 안 된다")

    assert raised.value is start_error
    assert gateway.start_calls == 1
    assert validator.close_calls == 1
    assert gateway.close_calls == 1
    assert database.close_calls == 1


async def test_construction_failure_closes_resources_created_before_it(monkeypatch):
    construction_error = LifecycleError("validator construction failed")
    database = FakeDatabase()
    gateway = FakeGateway()

    def fail_validator_construction(**_kwargs):
        raise construction_error

    monkeypatch.setattr(
        main_module,
        "JpegFrameValidator",
        fail_validator_construction,
    )
    app = create_app(
        settings=build_settings(),
        database=database,
        repository=object(),
        gateway=gateway,
    )

    with pytest.raises(LifecycleError) as raised:
        async with app.router.lifespan_context(app):
            pytest.fail("조립에 실패했으므로 요청 처리 단계에 진입하면 안 된다")

    assert raised.value is construction_error
    assert gateway.start_calls == 0
    assert gateway.close_calls == 1
    assert database.close_calls == 1


async def test_one_close_failure_does_not_block_other_resources():
    close_error = LifecycleError("validator close failed")
    database = FakeDatabase()
    gateway = FakeGateway()
    validator = FakeFrameValidator(close_error=close_error)
    app = build_app(database, gateway, validator)

    with pytest.raises(LifecycleError) as raised:
        async with app.router.lifespan_context(app):
            pass

    assert raised.value is close_error
    assert validator.close_calls == 1
    assert gateway.close_calls == 1
    assert database.close_calls == 1


async def test_body_and_close_errors_are_both_preserved():
    body_error = LifecycleError("request task failed")
    close_error = LifecycleError("database close failed")
    database = FakeDatabase(close_error=close_error)
    gateway = FakeGateway()
    validator = FakeFrameValidator()
    app = build_app(database, gateway, validator)

    with pytest.raises(ExceptionGroup) as raised:
        async with app.router.lifespan_context(app):
            raise body_error

    assert raised.value.exceptions == (body_error, close_error)
    assert validator.close_calls == 1
    assert gateway.close_calls == 1
    assert database.close_calls == 1


async def test_caller_cancellation_waits_for_all_resource_cleanup():
    release = asyncio.Event()
    database = BlockingResource(release)
    gateway = BlockingGateway(release)
    validator = BlockingResource(release)
    app = build_app(database, gateway, validator)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()

    shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
    await asyncio.wait_for(
        asyncio.gather(
            validator.close_started.wait(),
            gateway.close_started.wait(),
        ),
        timeout=0.2,
    )
    assert database.close_started.is_set() is False

    shutdown.cancel()
    await asyncio.sleep(0)
    assert shutdown.done() is False

    # 종료를 기다리는 호출자가 다시 취소되어도 공유 정리 task는 유지되어야 한다.
    shutdown.cancel()
    await asyncio.sleep(0)
    assert shutdown.done() is False

    release.set()
    await asyncio.wait_for(database.close_started.wait(), timeout=0.2)
    with pytest.raises(asyncio.CancelledError):
        await shutdown

    assert validator.closed is True
    assert gateway.closed is True
    assert database.closed is True
    assert validator.close_calls == 1
    assert gateway.close_calls == 1
    assert database.close_calls == 1


async def test_cancellation_and_close_failure_are_both_preserved():
    release = asyncio.Event()
    close_error = LifecycleError("database close failed")
    database = BlockingFailingResource(release, close_error)
    gateway = FakeGateway()
    validator = FakeFrameValidator()
    app = build_app(database, gateway, validator)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()

    shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
    await asyncio.wait_for(database.close_started.wait(), timeout=0.2)
    shutdown.cancel()
    release.set()

    with pytest.raises(BaseExceptionGroup) as raised:
        await shutdown

    assert any(
        isinstance(error, asyncio.CancelledError) for error in raised.value.exceptions
    )
    assert close_error in raised.value.exceptions
    assert gateway.close_calls == 1
    assert validator.close_calls == 1


async def test_database_closes_after_validator_and_gateway_finish():
    dependency_release = asyncio.Event()
    database = FakeDatabase()
    gateway = BlockingGateway(dependency_release)
    validator = BlockingResource(dependency_release)
    app = build_app(database, gateway, validator)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()

    shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
    await asyncio.wait_for(
        asyncio.gather(
            validator.close_started.wait(),
            gateway.close_started.wait(),
        ),
        timeout=0.2,
    )

    assert database.close_calls == 0

    dependency_release.set()
    await shutdown

    assert database.close_calls == 1
