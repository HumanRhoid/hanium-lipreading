"""애플리케이션 health와 lifespan 계약 테스트."""

import asyncio

from httpx2 import ASGITransport, AsyncClient

from src.backend.core.config import Settings
from src.backend.main import create_app


class FakeDatabase:
    def __init__(self, ready=True):
        self.ready = ready
        self.closed = False

    async def ping(self):
        if not self.ready:
            raise ConnectionError("database unavailable")

    async def close(self):
        self.closed = True


class FakeGateway:
    def __init__(self, ready=True):
        self.ready = ready
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True


class FakeFrameValidator:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


def build_settings(**overrides):
    values = {
        "app_env": "test",
        "database_url": (
            "postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"
        ),
        "inference_backend": "fake",
    }
    values.update(overrides)
    return Settings(
        _env_file=None,
        **values,
    )


async def request(app, path):
    """테스트에서 lifespan과 ASGI 요청을 명시적으로 실행한다."""

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)


async def test_liveness_does_not_depend_on_database_or_model():
    app = create_app(
        settings=build_settings(),
        database=FakeDatabase(ready=False),
        repository=object(),
        gateway=FakeGateway(ready=False),
        frame_validator=FakeFrameValidator(),
    )

    response = await request(app, "/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_reports_database_and_inference_state():
    database = FakeDatabase(ready=True)
    gateway = FakeGateway(ready=False)
    app = create_app(
        settings=build_settings(),
        database=database,
        repository=object(),
        gateway=gateway,
        frame_validator=FakeFrameValidator(),
    )

    response = await request(app, "/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "database": "ready",
        "inference": "not_ready",
    }


async def test_readiness_hides_database_exception_details():
    app = create_app(
        settings=build_settings(),
        database=FakeDatabase(ready=False),
        repository=object(),
        gateway=FakeGateway(ready=True),
        frame_validator=FakeFrameValidator(),
    )

    response = await request(app, "/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "database": "not_ready",
        "inference": "ready",
    }
    assert "unavailable" not in response.text


async def test_readiness_limits_slow_database_ping():
    class HangingDatabase(FakeDatabase):
        async def ping(self):
            await asyncio.Event().wait()

    app = create_app(
        settings=build_settings(readiness_timeout_seconds=0.01),
        database=HangingDatabase(),
        repository=object(),
        gateway=FakeGateway(ready=True),
        frame_validator=FakeFrameValidator(),
    )

    response = await asyncio.wait_for(request(app, "/health/ready"), timeout=0.2)

    assert response.status_code == 503
    assert response.json()["database"] == "not_ready"


async def test_readiness_fails_while_application_is_draining():
    app = create_app(
        settings=build_settings(),
        database=FakeDatabase(ready=True),
        repository=object(),
        gateway=FakeGateway(ready=True),
        frame_validator=FakeFrameValidator(),
    )

    async with app.router.lifespan_context(app):
        app.state.draining = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "database": "ready",
        "inference": "ready",
    }


async def test_lifespan_starts_and_closes_owned_resources():
    database = FakeDatabase()
    gateway = FakeGateway()
    validator = FakeFrameValidator()
    app = create_app(
        settings=build_settings(),
        database=database,
        repository=object(),
        gateway=gateway,
        frame_validator=validator,
    )

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 200
        assert gateway.started is True

    assert gateway.closed is True
    assert validator.closed is True
    assert database.closed is True


def test_openapi_describes_both_readiness_status_codes():
    app = create_app(settings=build_settings())

    operation = app.openapi()["paths"]["/health/ready"]["get"]

    assert set(operation["responses"]) >= {"200", "503"}
    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/ReadinessResponse"
    )
    assert (
        operation["responses"]["503"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/ReadinessResponse"
    )
