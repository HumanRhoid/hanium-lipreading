"""립리딩 모델 학습 스크립트."""

import argparse
import random
from collections import deque
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
    hidden_dim=256,
    num_layer=2,
    dropout=0.2,
    weight_decay=0.01,
    smoothing=3,
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

    model = LipReadingModel(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # 후반으로 갈수록 보폭을 좁혀 검증 손실이 급등하는 구간을 줄인다.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"장치: {device} | 클래스: {num_classes}개")
    print(f"학습 {len(train_indices)}개 · 검증 {len(val_indices)}개 클립")
    print(
        f"모델 hidden {hidden_dim} · layer {num_layer} · dropout {dropout} "
        f"· wd {weight_decay} | 저장 기준 최근 {smoothing}에폭 평균"
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0
    best_smoothed = 0.0
    recent = deque(maxlen=smoothing)

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # 검증 세트가 작아 단일 에폭 정확도는 크게 진동한다.
        # 최근 몇 에폭의 평균이 최고일 때 저장해 우연한 고점을 걸러낸다.
        recent.append(val_accuracy)
        smoothed = sum(recent) / len(recent)

        print(
            f"[{epoch:3d}/{epochs}] "
            f"train loss {train_loss:.4f} acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f} acc {val_accuracy:.3f} avg {smoothed:.3f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        if smoothed >= best_smoothed:
            best_smoothed = smoothed
            best_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "val_accuracy": val_accuracy,
                    "smoothed_accuracy": smoothed,
                    "hidden_dim": hidden_dim,
                    "num_layer": num_layer,
                    "dropout": dropout,
                },
                checkpoint_path,
            )

    print(
        f"저장 모델 검증 정확도 {best_accuracy:.3f} "
        f"(최근 {smoothing}에폭 평균 {best_smoothed:.3f}) · {checkpoint_path}"
    )
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
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--smoothing", type=int, default=3, help="체크포인트 판정에 쓸 에폭 수"
    )
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
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layer,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        smoothing=args.smoothing,
    )


if __name__ == "__main__":
    main()
