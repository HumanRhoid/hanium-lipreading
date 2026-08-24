import cv2
import numpy as np

# 모델 학습용 고정 해상도 설정.
# 2026-08-22에 112x80에서 올렸다. 종횡비가 1.4:1이라 2.2:1인 입술 크롭을
# 넣으면 가로가 세로보다 1.6배 더 눌렸다. 2:1로 바꿔 그 왜곡을 없앤다.
# s05의 크롭이 약 268x122px이라 여기까지는 확대되는 화자가 없다.
TARGET_WIDTH = 192
TARGET_HEIGHT = 96
# 2026-08-19b에 60으로 올렸다. 8화자 3시드에서 +0.033(t 1.34)으로
# 판정선 0.042는 못 넘었지만 세 번의 독립 측정이 모두 양수였다.
# 무엇보다 지금 데이터 1,234개가 전부 60프레임이라 30으로 뽑으면 섞을 수 없다.
FIXED_FRAME_COUNT = 60

# CLAHE (조명 불균일 보정) 설정
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)


def to_grayscale_clahe(frame):
    """프레임을 흑백화 + CLAHE 밝기 보정 후, 채널 수는 3으로 유지 (ResNet-18 사전학습 가중치 호환)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = _clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


# 발화 구간 검출 설정
ACTIVE_RATIO = 0.3  # (max-min) 대비 이 비율 이상 벌어지면 '발화 중'으로 간주
TRIM_MARGIN = 2  # 발화 구간 앞뒤로 남길 여유 프레임


def trim_to_speech(frames, opennesses, active_ratio=ACTIVE_RATIO, margin=TRIM_MARGIN):
    """openness가 낮게 유지되는 앞뒤 정적(공백) 구간을 잘라내고 발화 구간만 남긴다.

    fps·영상 길이와 무관하게 '실제 말한 부분'만 추출 → 뒤 공백 문제 방지.
    움직임이 거의 없으면(정지 영상) 원본 그대로 반환.
    """
    if not opennesses:
        return frames
    op = np.asarray(opennesses)
    lo, hi = np.percentile(op, 10), op.max()
    if hi - lo < 1e-6:
        return frames
    threshold = lo + active_ratio * (hi - lo)
    active = np.where(op > threshold)[0]
    if len(active) == 0:
        return frames
    start = max(0, active[0] - margin)
    end = min(len(frames), active[-1] + 1 + margin)
    return frames[start:end]


def resample_frames(frames, count):
    """프레임 리스트를 균등 간격으로 뽑아 정확히 count개로 맞춘다 (앞부분만 자르지 않음).

    길면 균등 축소, 짧으면 균등 복제 → fps·길이 편차를 흡수.
    """
    n = len(frames)
    if n == count:
        return frames
    idx = np.linspace(0, n - 1, count).round().astype(int)
    return [frames[i] for i in idx]


def normalize_frames(
    frames,
    opennesses=None,
    target_width=TARGET_WIDTH,
    target_height=TARGET_HEIGHT,
    fixed_frame_count=FIXED_FRAME_COUNT,
):
    """크롭된 프레임 리스트를 발화 구간 트리밍 · 고정 프레임 수(균등 리샘플링) ·
    고정 해상도 · 흑백+밝기 보정의 uint8 NumPy 배열로 변환.

    opennesses가 주어지면 앞뒤 정적을 잘라낸 뒤 리샘플링한다.
    """
    if not frames:
        return None

    if opennesses is not None:
        frames = trim_to_speech(frames, opennesses)

    frames = resample_frames(frames, fixed_frame_count)

    resized = [cv2.resize(frame, (target_width, target_height)) for frame in frames]
    resized = [to_grayscale_clahe(frame) for frame in resized]

    return np.array(resized, dtype=np.uint8)
