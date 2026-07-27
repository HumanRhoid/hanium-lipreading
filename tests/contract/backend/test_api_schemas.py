"""프런트엔드와 공유하는 WebSocket DTO 계약 테스트."""

import pytest
from pydantic import ValidationError

from src.backend.recognition.domain import RecognitionMode
from src.backend.recognition.schemas import (
    ErrorEvent,
    ResultEvent,
    StartCommand,
    StopCommand,
    parse_client_command,
)


def test_start_command_accepts_only_documented_fields():
    command = parse_client_command('{"type":"start","mode":"CLOSED"}')

    assert command == StartCommand(type="start", mode=RecognitionMode.CLOSED)

    with pytest.raises(ValidationError):
        parse_client_command('{"type":"start","mode":"CLOSED","device_info":"browser"}')


def test_start_command_defaults_to_closed_mode_for_minimal_frontend_dto():
    command = parse_client_command('{"type":"start"}')

    assert command == StartCommand(type="start", mode=RecognitionMode.CLOSED)


def test_stop_command_has_no_additional_payload():
    command = parse_client_command('{"type":"stop"}')

    assert command == StopCommand(type="stop")

    with pytest.raises(ValidationError):
        parse_client_command('{"type":"stop","reason":"user"}')


def test_result_event_omits_confidence_when_model_does_not_provide_it():
    event = ResultEvent(type="result", text="안녕하세요", final=True)

    assert event.model_dump(mode="json", exclude_none=True) == {
        "type": "result",
        "text": "안녕하세요",
        "final": True,
    }


def test_result_event_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        ResultEvent(type="result", text="물 주세요", confidence=1.1)


@pytest.mark.parametrize("text", ["", "   ", "\t\n", "문" * 201])
def test_result_event_rejects_text_outside_public_contract(text):
    with pytest.raises(ValidationError):
        ResultEvent(text=text)


def test_v1_result_event_rejects_partial_results():
    with pytest.raises(ValidationError):
        ResultEvent(text="물 주세요", final=False)


def test_error_event_contains_only_stable_code_and_message():
    event = ErrorEvent(
        type="error",
        code="MODEL_NOT_READY",
        message="인식 모델이 준비되지 않았습니다.",
    )

    assert event.model_dump(mode="json") == {
        "type": "error",
        "code": "MODEL_NOT_READY",
        "message": "인식 모델이 준비되지 않았습니다.",
    }

    with pytest.raises(ValidationError):
        ErrorEvent(code="MODEL_BOOT_FAILED", message="임의 코드")


@pytest.mark.parametrize(
    "code",
    ["STREAM_IDLE_TIMEOUT", "SESSION_LIMIT_REACHED"],
)
def test_error_event_has_distinct_retryable_timeout_codes(code):
    event = ErrorEvent(code=code, message="연결 제한 시간에 도달했습니다.")

    assert event.code == code


@pytest.mark.parametrize("code", ["VIDEO_TOO_LONG", "VIDEO_TOO_LARGE"])
def test_error_event_includes_bounded_video_codes(code):
    event = ErrorEvent(code=code, message="영상 입력 상한을 초과했습니다.")

    assert event.code == code
