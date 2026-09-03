"""환경변수 기반 백엔드 설정."""

from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 실행 설정과 안전한 기본값."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    app_env: Literal[
        "local",
        "test",
        "production",
    ] = "local"

    database_url: str = Field(
        default=(
            "postgresql+asyncpg://postgres:postgres@localhost:5432/hanium_lipreading"
        ),
        repr=False,
    )

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        repr=False,
    )

    inference_job_terminal_ttl_seconds: PositiveInt = 24 * 60 * 60
    inference_job_max_retries: PositiveInt = 3

    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
        ]
    )

    inference_backend: Literal[
        "fake",
        "unavailable",
        "local",
    ] = "unavailable"
    model_dir: str = "checkpoints"
    model_prefix: str = "release192"

    max_active_sessions: PositiveInt = 1
    max_inference_concurrency: PositiveInt = 1

    start_timeout_seconds: PositiveFloat = 5.0
    readiness_timeout_seconds: PositiveFloat = 2.0
    send_timeout_seconds: PositiveFloat = 2.0
    stream_idle_timeout_seconds: PositiveFloat = 10.0
    max_session_seconds: PositiveFloat = 300.0

    database_pool_size: PositiveInt = 5
    database_max_overflow: int = Field(
        default=5,
        ge=0,
    )
    database_pool_timeout_seconds: PositiveFloat = 10.0

    max_video_upload_bytes: PositiveInt = 64 * 1024 * 1024

    object_storage_endpoint_url: str = "http://localhost:9000"

    object_storage_access_key: str = Field(
        default="minioadmin",
        repr=False,
    )

    object_storage_secret_key: str = Field(
        default="minioadmin123",
        repr=False,
    )

    object_storage_bucket: str = "recognition-videos"

    object_storage_region: str = "us-east-1"

    @field_validator("database_url")
    @classmethod
    def validate_database_url(
        cls,
        value: str,
    ) -> str:
        """PostgreSQL asyncpg URL만 허용한다."""

        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("database_url은 postgresql+asyncpg URL이어야 합니다.")

        return value

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(
        cls,
        value: str,
    ) -> str:
        """Redis TCP/TLS URL만 허용한다."""

        value = value.strip()

        if not value:
            raise ValueError("redis_url은 비어 있을 수 없습니다.")

        parsed = urlsplit(value)

        if parsed.scheme not in {
            "redis",
            "rediss",
        }:
            raise ValueError("redis_url은 redis:// 또는 rediss:// URL이어야 합니다.")

        if parsed.hostname is None:
            raise ValueError("redis_url에는 Redis host가 필요합니다.")

        return value

    @field_validator("object_storage_endpoint_url")
    @classmethod
    def validate_object_storage_endpoint_url(
        cls,
        value: str,
    ) -> str:
        """S3 호환 HTTP/HTTPS endpoint만 허용한다."""

        if not value.startswith(
            (
                "http://",
                "https://",
            )
        ):
            raise ValueError(
                "object_storage_endpoint_url은 http:// 또는 https:// URL이어야 합니다."
            )

        return value.rstrip("/")

    @field_validator(
        "object_storage_access_key",
        "object_storage_secret_key",
        "object_storage_bucket",
        "object_storage_region",
    )
    @classmethod
    def validate_non_blank_storage_setting(
        cls,
        value: str,
    ) -> str:
        """Object Storage 필수 설정은 빈 값을 허용하지 않는다."""

        value = value.strip()

        if not value:
            raise ValueError("Object Storage 설정값은 비어 있을 수 없습니다.")

        return value

    @model_validator(mode="after")
    def validate_environment(
        self,
    ) -> Self:
        """운영 환경에서 fake 추론 backend 사용을 방지한다."""

        if self.app_env == "production" and self.inference_backend == "fake":
            raise ValueError("production 환경에서는 fake 추론을 사용할 수 없습니다.")

        return self
