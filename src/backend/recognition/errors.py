"""인식 요청에서 클라이언트가 복구할 수 있는 오류."""


class RecognitionError(Exception):
    """립리딩 요청 처리 중 예상 가능한 오류의 기준 클래스."""


class ModelNotReadyError(RecognitionError):
    """완성된 추론 bundle이 준비되지 않은 상태."""


class UnsupportedRecognitionModeError(RecognitionError):
    """현재 모델 bundle이 요청한 인식 방식을 지원하지 않는 상태."""


class InferenceBusyError(RecognitionError):
    """제한된 동기 추론 worker가 이미 사용 중인 상태."""


class SessionBusyError(RecognitionError):
    """프로세스가 허용하는 활성 인식 세션 수를 넘은 상태."""


class FrameValidationBusyError(RecognitionError):
    """제한된 JPEG 검증 worker가 이미 사용 중인 상태."""


class InvalidFrameError(RecognitionError):
    """JPEG 디코딩 또는 해상도 검증에 실패한 상태."""


class InsufficientFramesError(RecognitionError):
    """최종 인식에 필요한 최소 프레임 수를 채우지 못한 상태."""


class VideoTooLongError(RecognitionError):
    """한 영상이 v1 계약의 최대 프레임 수를 넘은 상태."""


class VideoTooLargeError(RecognitionError):
    """한 영상의 누적 JPEG byte가 v1 계약의 최대값을 넘은 상태."""


class SessionClosedError(RecognitionError):
    """이미 종료 절차가 시작된 인식 세션을 다시 사용한 상태."""


class SessionAlreadyEndedError(RecognitionError):
    """이미 종료된 DB 세션에 최종 발화를 추가하려는 상태."""
