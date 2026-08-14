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


def start_tracking(project, run_name, config):
    """W&B 실험 추적을 시작한다. 미설치이거나 project가 없으면 None을 돌려준다."""
    if project is None:
        return None

    try:
        import wandb
    except ImportError:
        print("[알림] wandb가 없어 실험 추적을 건너뜁니다. pip install wandb")
        return None

    wandb.init(project=project, name=run_name, config=config)
    return wandb


def split_by_speaker(dataset, val_ratio=0.2, seed=42, val_speakers=None):
    """화자 단위로 학습·검증 인덱스를 나눈다.

    같은 화자의 클립이 양쪽에 섞이면 검증 정확도가 부풀려지므로
    화자를 통째로 한쪽에만 배치한다.

    ``val_speakers``를 지정하면 그 화자를 검증으로 쓴다. 화자 구성이 바뀌면
    무작위 선택 결과도 달라져 실험 간 비교가 깨지므로, 교차검증이나
    설정 비교에서는 검증 화자를 고정하는 편이 안전하다.
    """
    speakers = sorted(set(dataset.speaker_ids))
    if len(speakers) < 2:
        raise ValueError("화자 분리 분할에는 화자가 둘 이상 필요합니다")

    if val_speakers is None:
        rng = random.Random(seed)
        shuffled = speakers.copy()
        rng.shuffle(shuffled)
        val_size = max(1, round(len(shuffled) * val_ratio))
        val_speakers = set(shuffled[:val_size])
    else:
        val_speakers = set(val_speakers)
        unknown = val_speakers - set(speakers)
        if unknown:
            raise ValueError(f"매니페스트에 없는 화자입니다: {sorted(unknown)}")

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
    if not val_indices:
        raise ValueError(f"검증 분할이 비었습니다: {sorted(val_speakers)}")
    return train_indices, val_indices, sorted(val_speakers)


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
    val_speakers=None,
    seed=42,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    num_workers=0,
    prefer_gpu=True,
    hidden_dim=256,
    num_layer=2,
    dropout=0.2,
    weight_decay=0.01,
    smoothing=3,
    augment=True,
    augmentation_config=None,
    pretrained=False,
    freeze_backbone=False,
    wandb_project=None,
    run_name=None,
):
    set_seed(seed)
    device = resolve_device(prefer_gpu)

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_ratio": val_ratio,
        "seed": seed,
        "hidden_dim": hidden_dim,
        "num_layer": num_layer,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "smoothing": smoothing,
        "augment": augment,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
    }

    # 증강은 학습 분할에만 적용한다. 검증은 원본이어야 성능을 정직하게 잰다.
    augmentation = (
        VideoAugmentation(config=augmentation_config, seed=seed) if augment else None
    )
    # 사전학습 백본은 ImageNet 분포를 전제하므로 입력 정규화를 함께 맞춘다.
    train_dataset = LipReadingDataset(
        manifest_path,
        data_root,
        augmentation=augmentation,
        imagenet_norm=pretrained,
    )
    plain = LipReadingDataset(manifest_path, data_root, imagenet_norm=pretrained)
    num_classes = len({row["label_id"] for row in plain.rows})

    train_indices, val_indices, held_out = split_by_speaker(
        plain, val_ratio, seed, val_speakers
    )
    # 검증 화자가 정해진 뒤에 기록을 시작해야 실험 간 비교 기준이 남는다.
    config["val_speakers"] = ",".join(held_out)
    tracker = start_tracking(wandb_project, run_name, config)

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
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
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    # 동결된 파라미터는 옵티마이저에 넣지 않는다.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    # 후반으로 갈수록 보폭을 좁혀 검증 손실이 급등하는 구간을 줄인다.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())

    print(f"장치: {device} | 클래스: {num_classes}개")
    print(
        f"학습 {len(train_indices)}개 · 검증 {len(val_indices)}개 클립 "
        f"| 검증 화자 {held_out}"
    )
    print(
        f"모델 hidden {hidden_dim} · layer {num_layer} · dropout {dropout} "
        f"· wd {weight_decay} | 증강 {'켬' if augment else '끔'} "
        f"· 사전학습 {'켬' if pretrained else '끔'} "
        f"· 동결 {'켬' if freeze_backbone else '끔'}"
    )
    print(
        f"학습 파라미터 {trainable_count / 1e6:.2f}M / 전체 {total_count / 1e6:.2f}M "
        f"| 저장 기준 최근 {smoothing}에폭 평균"
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

        if tracker is not None:
            tracker.log(
                {
                    "train/loss": train_loss,
                    "train/acc": train_accuracy,
                    "val/loss": val_loss,
                    "val/acc": val_accuracy,
                    "val/acc_smoothed": smoothed,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
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

    if tracker is not None:
        tracker.summary["best_val_acc"] = best_accuracy
        tracker.summary["best_val_acc_smoothed"] = best_smoothed
        tracker.finish()

    return best_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="립리딩 모델을 학습한다.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--val-speakers",
        nargs="+",
        default=None,
        help="검증에 쓸 화자를 직접 지정한다. 예) --val-speakers s04",
    )
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
        "--no-augment", action="store_true", help="증강 없이 학습해 효과를 비교한다"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="ImageNet 가중치로 백본을 초기화한다",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="ResNet 층을 고정하고 stem·GRU·헤드만 학습한다",
    )
    parser.add_argument(
        "--wandb-project", default=None, help="지정하면 W&B로 실험을 기록한다"
    )
    parser.add_argument("--run-name", default=None, help="W&B 실행 이름")
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
        val_speakers=args.val_speakers,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        num_workers=args.num_workers,
        prefer_gpu=not args.cpu,
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layer,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        smoothing=args.smoothing,
        augment=not args.no_augment,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
