"""인식 도메인의 값 객체."""

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real

# ── 뷰 → 서버 전송 규격 ──────────────────────────────────────────
# v1 WebSocket 스트리밍 경로가 받는 프레임의 계약이다. 모델 입력 규격이 아니다.
# 이 둘을 한 이름(INPUT_FRAME_*)으로 섞어 쓰다 Worker가 크롭 결과(60·96·192)를
# 전송 규격(30·360·640)으로 검증하는 사고가 났다. 이름으로 의미를 가른다.
STREAM_FRAME_WIDTH = 640
STREAM_FRAME_HEIGHT = 360
STREAM_FRAME_FPS = 25
MAX_FRAME_BYTES = 512 * 1024

# ── 모델 입력 프레임 수 ──────────────────────────────────────────
# 원천은 src/ml/preprocess/normalize.py의 FIXED_FRAME_COUNT(60)다. ml 모듈은
# import 시 cv2를 끌고 오므로 여기에는 값만 복제한다. 어긋나면
# tests/unit/backend/test_domain_models.py의 동기화 테스트가 잡는다.
MODEL_INPUT_FRAME_COUNT = 60
MIN_VIDEO_FRAMES = MODEL_INPUT_FRAME_COUNT
MAX_VIDEO_FRAMES = 250
MAX_VIDEO_BYTES = 64 * 1024 * 1024


def _validate_text(value: str, *, field_name: str, max_length: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}은 공백이 아닌 문자열이어야 합니다")
    if len(value) > max_length:
        raise ValueError(f"{field_name}은 {max_length}자 이하여야 합니다")


def _validate_confidence(value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("confidence는 실수여야 합니다")
    if not math.isfinite(float(value)) or not 0 <= value <= 1:
        raise ValueError("confidence는 0에서 1 사이의 유한한 실수여야 합니다")


def _validate_phrase_code(value: str | None) -> None:
    if value is not None:
        _validate_text(value, field_name="phrase_code", max_length=64)


class RecognitionMode(str, Enum):
    """립리딩 모델의 인식 방식."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"


class PhraseCategory(str, Enum):
    """폐쇄형 문구의 사용자 목적 분류."""

    PAIN = "PAIN"
    REQUEST = "REQUEST"
    REPLY = "REPLY"
    ETC = "ETC"


def _validate_positive_integer(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name}은 양의 정수여야 합니다")


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """교체 가능한 모델 bundle이 따라야 하는 입력·출력 계약."""

    bundle_version: str
    supported_modes: frozenset[RecognitionMode]
    frame_width: int
    frame_height: int
    fps: int
    input_frame_count: int
    label_map_version: str | None = None
    input_codec: str = "image/jpeg"

    def __post_init__(self) -> None:
        _validate_text(
            self.bundle_version,
            field_name="bundle_version",
            max_length=100,
        )
        modes = frozenset(self.supported_modes)
        if not modes or any(not isinstance(mode, RecognitionMode) for mode in modes):
            raise ValueError(
                "supported_modes는 한 개 이상의 RecognitionMode여야 합니다"
            )
        object.__setattr__(self, "supported_modes", modes)

        _validate_positive_integer(self.frame_width, field_name="frame_width")
        _validate_positive_integer(self.frame_height, field_name="frame_height")
        _validate_positive_integer(self.fps, field_name="fps")
        _validate_positive_integer(
            self.input_frame_count,
            field_name="input_frame_count",
        )

        _validate_text(self.input_codec, field_name="input_codec", max_length=100)
        if self.label_map_version is not None:
            _validate_text(
                self.label_map_version,
                field_name="label_map_version",
                max_length=100,
            )


@dataclass(frozen=True, slots=True)
class Prediction:
    """모델 어댑터가 반환하는 교정 전 인식 결과."""

    text: str
    confidence: float | None = None
    phrase_code: str | None = None

    def __post_init__(self) -> None:
        _validate_text(self.text, field_name="text", max_length=200)
        _validate_confidence(self.confidence)
        _validate_phrase_code(self.phrase_code)


@dataclass(frozen=True, slots=True)
class RecognitionOutput:
    """후처리를 거친 저장 가능한 최종 인식 결과."""

    raw_text: str
    corrected_text: str | None = None
    confidence: float | None = None
    phrase_code: str | None = None

    def __post_init__(self) -> None:
        _validate_text(self.raw_text, field_name="raw_text", max_length=200)
        if self.corrected_text is not None:
            _validate_text(
                self.corrected_text,
                field_name="corrected_text",
                max_length=200,
            )
        _validate_confidence(self.confidence)
        _validate_phrase_code(self.phrase_code)

    @property
    def display_text(self) -> str:
        """외부 API에 노출할 최종 문장을 반환한다."""

        return self.corrected_text or self.raw_text
