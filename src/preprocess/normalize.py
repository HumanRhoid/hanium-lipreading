import cv2
import numpy as np

# 모델 학습용 고정 해상도 설정
TARGET_WIDTH = 112
TARGET_HEIGHT = 80
FIXED_FRAME_COUNT = 30  # 프레임 수 맞추기 기준값

# CLAHE (조명 불균일 보정) 설정
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)


def to_grayscale_clahe(frame):
    """프레임을 흑백화 + CLAHE 밝기 보정 후, 채널 수는 3으로 유지 (ResNet-18 사전학습 가중치 호환)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = _clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


def normalize_frames(
    frames,
    target_width=TARGET_WIDTH,
    target_height=TARGET_HEIGHT,
    fixed_frame_count=FIXED_FRAME_COUNT,
):
    """크롭된 프레임 리스트를 고정 해상도 · 흑백+밝기 보정 · 고정 프레임 수의 uint8 NumPy 배열로 변환"""
    if not frames:
        return None

    resized = [cv2.resize(frame, (target_width, target_height)) for frame in frames]
    resized = [to_grayscale_clahe(frame) for frame in resized]

    # 프레임 길이 맞추기 (시간축 패딩 또는 절삭)
    if len(resized) < fixed_frame_count:
        resized += [resized[-1]] * (fixed_frame_count - len(resized))
    elif len(resized) > fixed_frame_count:
        resized = resized[:fixed_frame_count]

    return np.array(resized, dtype=np.uint8)
