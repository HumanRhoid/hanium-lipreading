from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

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


def crop_lip(frame, landmarker, margin=0.5):
    """입 ROI 크롭과 openness를 함께 반환. 얼굴 미검출 시 None."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    landmarks = result.face_landmarks[0]
    xs = [landmarks[i].x * w for i in LIP_LANDMARKS]
    ys = [landmarks[i].y * h for i in LIP_LANDMARKS]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    mw = int((x_max - x_min) * margin)
    mh = int((y_max - y_min) * margin)
    x_min, x_max = max(0, x_min - mw), min(w, x_max + mw)
    y_min, y_max = max(0, y_min - mh), min(h, y_max + mh)
    return frame[y_min:y_max, x_min:x_max], lip_openness(landmarks, w, h)


def crop_lip_frames(video_path, landmarker, margin=0.5):
    """영상의 각 프레임에서 입 ROI 크롭 + openness를 리스트로 반환 (중간 영상 파일 없음).

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
            result = crop_lip(frame, landmarker, margin=margin)
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
