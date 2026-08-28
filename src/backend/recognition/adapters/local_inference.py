"""checkpoints의 배포 모델 3개로 실제 추론을 수행하는 SyncPredictor 구현.

Worker의 전처리(MlStoredVideoPreprocessor)가 이미 크롭을 마쳤으므로, 이
파일이 받는 frames는 원본 영상이 아니라 192x96 입술 크롭을 JPEG로 구운
bytes 60개다. 여기서는 디코딩·검증·앙상블만 한다.
"""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch

from src.backend.recognition.domain import (
    MODEL_INPUT_FRAME_COUNT,
    STREAM_FRAME_FPS,
    STREAM_FRAME_HEIGHT,
    STREAM_FRAME_WIDTH,
    ModelManifest,
    Prediction,
    RecognitionMode,
)
from src.ml.models import LipReadingModel
from src.ml.preprocess.normalize import FIXED_FRAME_COUNT, TARGET_HEIGHT, TARGET_WIDTH

# 모델 라벨(학습 파일명 기준) → DB 문구 코드. 표기가 달라 문자열 매칭 대신 표로 박는다.
LABEL_TO_CODE = {
    "가래가있어요": "SYMPTOM_PHLEGM",
    "간호사불러주세요": "REQUEST_NURSE",
    "더워요": "STATE_HOT",
    "도와주세요": "REQUEST_HELP",
    "물주세요": "REQUEST_WATER",
    "배고파요": "STATE_HUNGRY",
    "보호자불러주세요": "REQUEST_GUARDIAN",
    "숨쉬기힘들어요": "SYMPTOM_BREATHING_DIFFICULTY",
    "아파요": "PAIN_GENERAL",
    "어지러워요": "SYMPTOM_DIZZINESS",
    "자세바꿔주세요": "REQUEST_REPOSITION",
    "진통제주세요": "REQUEST_PAINKILLER",
    "추워요": "STATE_COLD",
    "토할거같아요": "SYMPTOM_NAUSEA",
    "화장실가고싶어요": "REQUEST_TOILET",
}


class LocalSyncPredictor:
    """전용 worker 스레드에서만 호출되는 동기 앙상블 추론기."""

    def __init__(self, model_dir: Path, model_prefix: str):
        # __init__은 조립 시점(create_gateway)에 동기로 불린다. 무거운 로드 금지.
        self._model_dir = Path(model_dir)
        self._model_prefix = model_prefix
        self._models: list[LipReadingModel] = []
        self._labels: list[str] = []
        self._device = "cpu"
        self.ready = False
        # service._validate_model_manifest가 WS 스트림 계약(640x360·25fps)과
        # 대조해 다르면 기동을 거부하므로 당분간 그 값으로 선언한다.
        # WS 경로를 검사 채널로 개조할 때 진실(192x96)로 바로잡는다.
        self._manifest = ModelManifest(
            bundle_version=f"{model_prefix}-v1.0.0",
            supported_modes=frozenset({RecognitionMode.CLOSED}),
            frame_width=STREAM_FRAME_WIDTH,
            frame_height=STREAM_FRAME_HEIGHT,
            fps=STREAM_FRAME_FPS,
            input_frame_count=MODEL_INPUT_FRAME_COUNT,
            label_map_version="closed-15-v1",
        )

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    def start(self) -> None:
        if self.ready:
            return
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        paths = sorted(self._model_dir.glob(f"{self._model_prefix}_seed*.pt"))
        if not paths:
            raise FileNotFoundError(
                f"{self._model_dir}에 {self._model_prefix}_seed*.pt가 없습니다"
            )

        labels: list[str] | None = None
        models: list[LipReadingModel] = []
        for path in paths:
            ck = torch.load(path, map_location=self._device)
            if "labels" not in ck:
                raise ValueError(
                    f"{path.name}에 문구 목록이 없습니다. 교차검증 체크포인트로 보입니다"
                )
            if labels is None:
                labels = ck["labels"]
            elif ck["labels"] != labels:
                raise ValueError(f"{path.name}의 문구 목록이 앞 체크포인트와 다릅니다")
            if ck["frames"] != FIXED_FRAME_COUNT:
                raise ValueError(
                    f"{path.name}은 {ck['frames']}프레임 학습본인데 "
                    f"현재 규격은 {FIXED_FRAME_COUNT}프레임입니다"
                )
            model = LipReadingModel(
                num_classes=ck["num_classes"],
                hidden_dim=ck["hidden_dim"],
                num_layer=ck["num_layer"],
                dropout=ck["dropout"],
            ).to(self._device)
            model.load_state_dict(ck["model_state"])
            model.eval()
            models.append(model)

        missing = set(labels) - set(LABEL_TO_CODE)
        if missing:
            raise ValueError(f"문구 코드가 없는 라벨: {sorted(missing)}")

        self._models = models
        self._labels = labels
        self.ready = True

    def predict(
        self,
        frames: Sequence[bytes],
        mode: RecognitionMode,
    ) -> Prediction:
        if not self.ready:
            raise RuntimeError("start()가 호출되지 않았습니다")
        if mode is not RecognitionMode.CLOSED:
            raise ValueError("CLOSED 모드만 지원합니다")
        if len(frames) != FIXED_FRAME_COUNT:
            raise ValueError(
                f"정확히 {FIXED_FRAME_COUNT}프레임이 필요합니다 (받음: {len(frames)})"
            )

        # 역직렬화 + 계약 검증. resize로 덮지 않는다 — 규격 위반은 즉시 드러나야 한다.
        decoded = []
        for index, blob in enumerate(frames):
            frame = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.shape != (TARGET_HEIGHT, TARGET_WIDTH, 3):
                got = None if frame is None else frame.shape
                raise ValueError(f"{index}번째 프레임이 크롭 규격이 아닙니다: {got}")
            decoded.append(frame)

        # 학습(dataset.py)·predict.py와 같은 변환이어야 한다. (B, C, T, H, W)
        clip = np.stack(decoded)                                  # (60, 96, 192, 3)
        x = (
            torch.from_numpy(clip)
            .permute(3, 0, 1, 2)
            .float()
            .div_(255)
            .unsqueeze(0)
            .to(self._device)
        )

        # softmax를 평균한 뒤 argmax — 시드 편차(0.0725)를 앙상블로 줄인다.
        with torch.no_grad():
            probs = sum(
                torch.softmax(model(x).float(), dim=1) for model in self._models
            ) / len(self._models)

        index = int(probs.argmax(dim=1))
        text = self._labels[index]
        return Prediction(
            text=text,
            confidence=float(probs[0, index]),
            phrase_code=LABEL_TO_CODE[text],
        )

    def close(self) -> None:
        self._models = []
        self._labels = []
        self.ready = False