"""립리딩 모델 학습 스크립트."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.ml.models import LipReadingModel
from src.ml.preprocess.augmentation import VideoAugmentation
from src.ml.training.dataset import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST_PATH,
    LipReadingDataset,
)

DEFAULT_CHECKPOINT_PATH = Path("checkpoints/best.pt")


def resolve_device(prefer_gpu=True):
    """사용 가능한 장치를 고른다. GPU가 있으면 GPU를 우선한다."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_by_speaker(dataset, val_ratio=0.2, seed=42):
    """화자 단위로 학습·검증 인덱스를 나눈다.

    같은 화자의 클립이 양쪽에 섞이면 검증 정확도가 부풀려지므로
    화자를 통째로 한쪽에만 배치한다.
    """
    speakers = sorted(set(dataset.speaker_ids))
    if len(speakers) < 2:
        raise ValueError("화자 분리 분할에는 화자가 둘 이상 필요합니다")

    rng = random.Random(seed)
    shuffled = speakers.copy()
    rng.shuffle(shuffled)

    val_size = max(1, round(len(shuffled) * val_ratio))
    val_speakers = set(shuffled[:val_size])

    train_indices = [
        index
        for index, speaker in enumerate(dataset.speaker_ids)
        if speaker not in val_speakers
    ]
    val_indices = [
        index
        for index, speaker in enumerate(dataset.speaker_ids)
        if speaker in val_speakers
    ]

    if not train_indices:
        raise ValueError("학습 분할이 비었습니다. val_ratio를 낮추세요")
    return train_indices, val_indices


def run_epoch(model, loader, criterion, device, optimizer=None):
    """한 에폭을 실행하고 평균 손실과 정확도를 반환한다."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.set_grad_enabled(is_training):
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def train(
    manifest_path=DEFAULT_MANIFEST_PATH,
    data_root=DEFAULT_DATA_ROOT,
    epochs=30,
    batch_size=4,
    learning_rate=1e-4,
    val_ratio=0.2,
    seed=42,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    num_workers=0,
    prefer_gpu=True,
):
    set_seed(seed)
    device = resolve_device(prefer_gpu)

    augmented = LipReadingDataset(
        manifest_path, data_root, augmentation=VideoAugmentation(seed=seed)
    )
    plain = LipReadingDataset(manifest_path, data_root)
    num_classes = len({row["label_id"] for row in plain.rows})

    train_indices, val_indices = split_by_speaker(plain, val_ratio, seed)
    train_loader = DataLoader(
        Subset(augmented, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(plain, val_indices),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = LipReadingModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"장치: {device} | 클래스: {num_classes}개")
    print(f"학습 {len(train_indices)}개 · 검증 {len(val_indices)}개 클립")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

        print(
            f"[{epoch:3d}/{epochs}] "
            f"train loss {train_loss:.4f} acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.3f}"
        )

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

    print(f"최고 검증 정확도 {best_accuracy:.3f} · 저장 위치 {checkpoint_path}")
    return best_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="립리딩 모델을 학습한다.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cpu", action="store_true", help="GPU가 있어도 CPU로 학습한다"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        manifest_path=args.manifest,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        num_workers=args.num_workers,
        prefer_gpu=not args.cpu,
    )


if __name__ == "__main__":
    main()
