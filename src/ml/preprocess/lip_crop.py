import math
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.ml.preprocess.normalize import TARGET_HEIGHT, TARGET_WIDTH

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker"
    "/face_landmarker/float16/1/face_landmarker.task"
)

LIP_LANDMARKS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
]


def create_landmarker(model_path=DEFAULT_MODEL_PATH):
    """FaceLandmarker를 만든다. 모델 파일이 없으면 내려받는다.

    3.7MB짜리라 저장소에 넣지 않는다. 노트북에서 내려받는 셀을 건너뛰면
    추론이 FileNotFoundError로 죽었다. 체크포인트만 받아 쓰는 사람도
    같은 곳에서 막히므로 여기서 처리한다.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"랜드마커 모델을 내려받습니다: {model_path}")
        urllib.request.urlretrieve(LANDMARKER_URL, str(model_path))

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)


# 입 벌어짐(openness) 계산용 랜드마크 (안쪽 위/아래 입술, 좌/우 입꼬리)
UPPER_INNER_LIP = 13
LOWER_INNER_LIP = 14
LEFT_CORNER = 61
RIGHT_CORNER = 291


def lip_openness(landmarks, w, h):
    """입 세로 벌어짐 ÷ 입 너비 → 얼굴 크기·거리에 무관한 비율 (발화 구간 검출용)."""
    vertical = abs((landmarks[UPPER_INNER_LIP].y - landmarks[LOWER_INNER_LIP].y) * h)
    width = abs((landmarks[LEFT_CORNER].x - landmarks[RIGHT_CORNER].x) * w) + 1e-6
    return vertical / width


# 입 너비가 출력 가로의 이 비율이 되도록 맞춘다. 0.5는 예전 축 정렬 크롭
# (상자 + 좌우 50% 여백 = 상자의 2배)과 같은 배율이라 확대되는 화자가 없다.
MOUTH_WIDTH_RATIO = 0.5


def alignment_matrix(landmarks, w, h, out_w, out_h, mouth_ratio=MOUTH_WIDTH_RATIO):
    """입꼬리 두 점으로 회전·크기·위치를 맞추는 2x3 닮음 변환을 만든다.

    축 정렬 사각형에는 문제가 둘 있었다. 고개가 기울면 입이 대각선으로 들어가고,
    상자 종횡비가 화자마다 달라 가로와 세로가 서로 다른 배율로 눌렸다.
    닮음 변환은 회전을 펴고 한 배율만 쓰므로 둘 다 사라진다.
    """
    left, right = landmarks[LEFT_CORNER], landmarks[RIGHT_CORNER]
    dx = (right.x - left.x) * w
    dy = (right.y - left.y) * h
    mouth_width = math.hypot(dx, dy)
    if mouth_width < 1e-6:
        return None

    # 입꼬리를 잇는 선을 수평으로 돌린다.
    angle = math.degrees(math.atan2(dy, dx))
    scale = (out_w * mouth_ratio) / mouth_width

    # 21점의 무게중심을 중심으로 쓴다. 최소·최대는 극단 한 점에 흔들린다.
    cx = sum(landmarks[i].x for i in LIP_LANDMARKS) / len(LIP_LANDMARKS) * w
    cy = sum(landmarks[i].y for i in LIP_LANDMARKS) / len(LIP_LANDMARKS) * h

    matrix = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    matrix[0, 2] += out_w / 2 - cx
    matrix[1, 2] += out_h / 2 - cy
    return matrix


def crop_lip(frame, landmarker, out_w=TARGET_WIDTH, out_h=TARGET_HEIGHT):
    """정렬된 입 ROI 크롭과 openness를 함께 반환. 얼굴 미검출 시 None."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    landmarks = result.face_landmarks[0]

    matrix = alignment_matrix(landmarks, w, h, out_w, out_h)
    if matrix is None:
        return None
    # 입이 화면 가장자리에 붙어도 검은 띠를 만들지 않는다. CLAHE가 그 경계를
    # 대비로 증폭하므로 없던 특징이 생긴다.
    lip = cv2.warpAffine(
        frame, matrix, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE
    )
    return lip, lip_openness(landmarks, w, h)


def crop_lip_frames(video_path, landmarker, out_w=TARGET_WIDTH, out_h=TARGET_HEIGHT):
    """영상의 각 프레임에서 정렬된 입 ROI + openness를 리스트로 반환.

    반환: (lips, opennesses) — 프레임별 입 크롭 이미지와 벌어짐 비율.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")

    lips, opennesses = [], []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = crop_lip(frame, landmarker, out_w, out_h)
            if result is not None:
                lip, openness = result
                lips.append(lip)
                opennesses.append(openness)
    finally:
        cap.release()

    return lips, opennesses


if __name__ == "__main__":
    landmarker = create_landmarker()
    try:
        sample_video = PROJECT_ROOT / "data" / "raw" / "WIN_20260614_10_27_35_Pro.mp4"
        lips, _ = crop_lip_frames(sample_video, landmarker)
        print("검출된 입 프레임:", len(lips))
    finally:
        landmarker.close()
