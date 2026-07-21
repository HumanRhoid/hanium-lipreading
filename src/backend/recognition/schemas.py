"""프런트엔드와 공유하는 인식 API 메시지 스키마."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from src.backend.recognition.domain import RecognitionMode


class StrictSchema(BaseModel):
    """명세에 없는 필드를 거부하는 API 스키마."""

    model_config = ConfigDict(extra="forbid")


class StartCommand(StrictSchema):
    type: Literal["start"]
    mode: RecognitionMode = RecognitionMode.CLOSED


class StopCommand(StrictSchema):
    type: Literal["stop"]


ClientCommand = Annotated[StartCommand | StopCommand, Field(discriminator="type")]
_client_command_adapter = TypeAdapter(ClientCommand)


def parse_client_command(payload: str) -> StartCommand | StopCommand:
    """JSON 문자열을 명세에 정의된 클라이언트 명령으로 변환한다."""

    return _client_command_adapter.validate_json(payload)


class ReadyEvent(StrictSchema):
    type: Literal["ready"] = "ready"


class ResultEvent(StrictSchema):
    type: Literal["result"] = "result"
    text: str = Field(min_length=1, max_length=200)
    final: Literal[True] = True
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text는 공백일 수 없습니다")
        return value


ErrorCode = Literal[
    "INVALID_MESSAGE",
    "UNSUPPORTED_MODE",
    "INVALID_FRAME",
    "FRAME_TOO_LARGE",
    "VIDEO_TOO_LONG",
    "VIDEO_TOO_LARGE",
    "INSUFFICIENT_FRAMES",
    "MODEL_NOT_READY",
    "SERVER_BUSY",
    "STREAM_IDLE_TIMEOUT",
    "SESSION_LIMIT_REACHED",
    "INTERNAL_ERROR",
]


class ErrorEvent(StrictSchema):
    type: Literal["error"] = "error"
    code: ErrorCode
    message: str = Field(min_length=1, max_length=200)


class StoppedEvent(StrictSchema):
    type: Literal["stopped"] = "stopped"
