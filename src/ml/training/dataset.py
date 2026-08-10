"""매니페스트 기반 립리딩 학습 데이터셋."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ml.preprocess.augmentation import VideoAugmentation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_ROOT / "manifest.csv"

PIXEL_MAX = 255.0


def load_manifest(manifest_path):
    """매니페스트 CSV를 읽어 클립 정보 목록으로 반환한다."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"매니페스트를 찾을 수 없습니다: {manifest_path}")

    with open(manifest_path, newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))

    if not rows:
        raise ValueError(f"매니페스트가 비어 있습니다: {manifest_path}")

    required = {"clip_path", "label_id", "speaker_id"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"매니페스트에 필요한 열이 없습니다: {sorted(missing)}")

    return rows


class LipReadingDataset(Dataset):
    """전처리된 .npy 클립과 문구 라벨을 모델 입력 형태로 공급한다.

    ``__getitem__``은 ``[C, T, H, W]`` 실수 텐서와 정수 라벨을 반환한다.
    증강은 uint8 단계에서만 적용하고 그 뒤에 실수로 변환한다.

    Args:
        manifest_path: 매니페스트 CSV 경로.
        data_root: ``clip_path``의 기준이 되는 데이터 폴더.
        augmentation: 학습용 증강기. ``None``이면 증강하지 않는다.
    """

    def __init__(
        self,
        manifest_path=DEFAULT_MANIFEST_PATH,
        data_root=DEFAULT_DATA_ROOT,
        augmentation=None,
    ):
        if augmentation is not None and not isinstance(augmentation, VideoAugmentation):
            raise TypeError("augmentation must be a VideoAugmentation or None")

        self.data_root = Path(data_root)
        self.rows = load_manifest(manifest_path)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        clip = np.load(self.data_root / row["clip_path"])

        if self.augmentation is not None:
            clip = self.augmentation(clip)

        # (T, H, W, C) uint8 → [C, T, H, W] float
        frames = torch.from_numpy(np.ascontiguousarray(clip))
        frames = frames.permute(3, 0, 1, 2).float().div_(PIXEL_MAX)

        label = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return frames, label

    @property
    def speaker_ids(self):
        """화자 분리 분할에 사용할 클립별 화자 목록."""
        return [row["speaker_id"] for row in self.rows]
