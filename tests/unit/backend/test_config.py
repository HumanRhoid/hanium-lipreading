"""백엔드 환경 설정 계약 테스트."""

import pytest
from pydantic import ValidationError

from src.backend.core.config import Settings

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"


def test_settings_accept_async_postgresql_url():
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=TEST_DATABASE_URL,
        inference_backend="fake",
    )

    assert settings.database_url.startswith("postgresql+asyncpg://")

    assert settings.allowed_origins == ["http://localhost:5173"]


def test_redis_url_has_safe_default():
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=TEST_DATABASE_URL,
    )

    assert settings.redis_url == "redis://localhost:6379/0"


def test_redis_url_is_loaded_from_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "REDIS_URL",
        "redis://localhost:6380/2",
    )

    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=TEST_DATABASE_URL,
    )

    assert settings.redis_url == "redis://localhost:6380/2"


def test_settings_accept_rediss_url():
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=TEST_DATABASE_URL,
        redis_url="rediss://redis.example.com:6380/1",
    )

    assert settings.redis_url == "rediss://redis.example.com:6380/1"


@pytest.mark.parametrize(
    "redis_url",
    [
        "",
        "http://localhost:6379/0",
        "redis://",
    ],
)
def test_settings_reject_invalid_redis_url(
    redis_url,
):
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            app_env="test",
            database_url=TEST_DATABASE_URL,
            redis_url=redis_url,
        )


def test_redis_password_is_hidden_from_settings_repr_and_validation_error():
    password = "synthetic-redis-secret"

    redis_url = f"redis://default:{password}@localhost:6379/0"

    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=TEST_DATABASE_URL,
        redis_url=redis_url,
    )

    assert password not in repr(settings)

    assert password not in str(settings)

    invalid_url = f"http://default:{password}@localhost:6379/0"

    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            app_env="test",
            database_url=TEST_DATABASE_URL,
            redis_url=invalid_url,
        )

    assert password not in str(exc_info.value)

    assert password not in repr(exc_info.value)


def test_websocket_timeouts_are_loaded_from_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "SEND_TIMEOUT_SECONDS",
        "0.25",
    )

    monkeypatch.setenv(
        "STREAM_IDLE_TIMEOUT_SECONDS",
        "1.5",
    )

    monkeypatch.setenv(
        "MAX_SESSION_SECONDS",
        "12.0",
    )

    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=TEST_DATABASE_URL,
    )

    assert settings.send_timeout_seconds == 0.25

    assert settings.stream_idle_timeout_seconds == 1.5

    assert settings.max_session_seconds == 12.0


def test_video_upload_limit_has_safe_default():
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=TEST_DATABASE_URL,
    )

    assert settings.max_video_upload_bytes == 64 * 1024 * 1024


def test_video_upload_limit_is_loaded_from_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "MAX_VIDEO_UPLOAD_BYTES",
        "10485760",
    )

    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=TEST_DATABASE_URL,
    )

    assert settings.max_video_upload_bytes == 10 * 1024 * 1024


def test_model_frame_count_is_not_a_deployment_setting():
    assert "frame_window_size" not in Settings.model_fields

    assert "frame_stride" not in Settings.model_fields

    assert "input_frame_count" not in Settings.model_fields


def test_database_password_is_hidden_from_settings_repr_and_validation_error():
    password = "synthetic-super-secret-password"

    database_url = f"postgresql+asyncpg://postgres:{password}@localhost:5432/test_db"

    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=database_url,
    )

    assert password not in repr(settings)

    assert password not in str(settings)

    invalid_url = f"postgresql://postgres:{password}@localhost:5432/test_db"

    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            app_env="test",
            database_url=invalid_url,
        )

    assert password not in str(exc_info.value)

    assert password not in repr(exc_info.value)


def test_settings_reject_non_async_postgresql_url():
    with pytest.raises(
        ValidationError,
        match=r"postgresql\+asyncpg",
    ):
        Settings(
            _env_file=None,
            app_env="test",
            database_url="sqlite+aiosqlite:///test.db",
        )


def test_settings_reject_fake_inference_in_production():
    with pytest.raises(
        ValidationError,
        match="fake",
    ):
        Settings(
            _env_file=None,
            app_env="production",
            database_url=(
                "postgresql+asyncpg://postgres:postgres@localhost:5432/prod_db"
            ),
            inference_backend="fake",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "max_active_sessions",
            0,
        ),
        (
            "max_inference_concurrency",
            0,
        ),
        (
            "start_timeout_seconds",
            0,
        ),
        (
            "readiness_timeout_seconds",
            0,
        ),
        (
            "send_timeout_seconds",
            0,
        ),
        (
            "stream_idle_timeout_seconds",
            0,
        ),
        (
            "max_session_seconds",
            0,
        ),
        (
            "max_video_upload_bytes",
            0,
        ),
    ],
)
def test_settings_reject_non_positive_limits(
    field,
    value,
):
    values = {
        "_env_file": None,
        "app_env": "test",
        "database_url": TEST_DATABASE_URL,
        field: value,
    }

    with pytest.raises(ValidationError):
        Settings(**values)
