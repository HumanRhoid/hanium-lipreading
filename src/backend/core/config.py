"""환경변수 기반 백엔드 설정."""

from typing import Literal, Self

from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 실행 설정과 안전한 기본값."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    app_env: Literal["local", "test", "production"] = "local"
    database_url: str = Field(
        default=(
            "postgresql+asyncpg://postgres:postgres@localhost:5432/hanium_lipreading"
        ),
        repr=False,
    )
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )
    inference_backend: Literal["fake", "unavailable"] = "unavailable"

    max_active_sessions: PositiveInt = 1
    max_inference_concurrency: PositiveInt = 1
    start_timeout_seconds: PositiveFloat = 5.0
    readiness_timeout_seconds: PositiveFloat = 2.0
    send_timeout_seconds: PositiveFloat = 2.0
    stream_idle_timeout_seconds: PositiveFloat = 10.0
    max_session_seconds: PositiveFloat = 300.0

    database_pool_size: PositiveInt = 5
    database_max_overflow: int = Field(default=5, ge=0)
    database_pool_timeout_seconds: PositiveFloat = 10.0

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        """동기 driver나 SQLite가 운영 경로에 섞이는 것을 방지한다."""

        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("database_url은 postgresql+asyncpg URL이어야 합니다")
        return value

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """운영 환경에서 가짜 모델 결과가 노출되지 않게 한다."""

        if self.app_env == "production" and self.inference_backend == "fake":
            raise ValueError("production 환경에서는 fake 추론을 사용할 수 없습니다")
        return self
