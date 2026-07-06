from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def crop_lip(frame, landmarker, margin=0.5):
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
    return frame[y_min:y_max, x_min:x_max]


def crop_lip_frames(video_path, landmarker, margin=0.5):
    """영상의 각 프레임에서 입 ROI를 크롭해 메모리 상의 배열 리스트로 반환 (중간 영상 파일 없음)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")

    lips = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            lip = crop_lip(frame, landmarker, margin=margin)
            if lip is not None:
                lips.append(lip)
    finally:
        cap.release()

    return lips


if __name__ == "__main__":
    landmarker = create_landmarker()
    try:
        sample_video = PROJECT_ROOT / "data" / "raw" / "WIN_20260614_10_27_35_Pro.mp4"
        lips = crop_lip_frames(sample_video, landmarker)
        print("검출된 입 프레임:", len(lips))
    finally:
        landmarker.close()
