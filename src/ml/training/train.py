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


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN은 합성곱·GRU 역전파에 비결정적 알고리즘을 고른다. 2026-08-25 align
    # 재측정에서 같은 시드·같은 설정이 s01에서 평균 -0.030 어긋났다. 조건을
    # 아무것도 안 바꿔도 판정선(0.042)에 육박하는 잡음이 섞이므로 기본을 켬으로
    # 둔다. benchmark 자동 탐색을 끄는 만큼 느려진다.
    # 주의: cuDNN 밖의 비결정 연산(atomics 등)은 이걸로 안 잡힌다.
    # torch.use_deterministic_algorithms는 대체 구현이 없는 연산에서 예외를
    # 던져 학습이 멈추므로 쓰지 않는다.
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


class WeightAverage:
    """가중치의 지수이동평균을 유지한다.

    후반 에폭에서 가중치가 진동해도 평균값은 완만하게 움직이므로
    검증 성능이 안정된다. 학습에는 원본 가중치를 그대로 쓴다.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if param.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach(), alpha=1 - self.decay
                )

    def apply_to(self, model):
        """평균 가중치를 모델에 넣고, 원래 값을 돌려준다."""
        backup = {name: model.state_dict()[name].clone() for name in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        return backup

    def restore(self, model, backup):
        model.load_state_dict(backup, strict=False)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    amp=False,
    grad_clip=None,
    averager=None,
):
    """한 에폭을 실행하고 평균 손실과 정확도를 반환한다."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.set_grad_enabled(is_training):
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # bfloat16은 fp16과 달리 표현 범위가 넓어 GradScaler 없이 안전하다.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(frames)
                loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # 드물게 튀는 그래디언트가 가중치를 크게 흔드는 것을 막는다.
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if averager is not None:
                    averager.update(model)

            total_loss += loss.float().item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def collect_errors(model, loader, device, amp=False):
    """마지막 에폭 모델의 오답을 (정답, 예측) 쌍으로 센다.

    어느 문구끼리 헷갈리는지가 다음 개입을 정한다. 2026-08-19에 손으로 뽑아
    s07의 물주세요-토할거같아요 쌍을 찾은 작업을 자동화한 것이다.
    보고값과 맞추기 위해 저장 시점이 아니라 마지막 에폭 가중치를 쓴다.
    """
    model.eval()
    counts = {}
    total = 0
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                predicted = model(frames).argmax(dim=1)
            for true_id, pred_id in zip(labels.tolist(), predicted.tolist()):
                total += 1
                if true_id != pred_id:
                    key = (true_id, pred_id)
                    counts[key] = counts.get(key, 0) + 1
    return [[t, p, n] for (t, p), n in sorted(counts.items())], total


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
    amp=True,
    hidden_dim=256,
    num_layer=2,
    dropout=0.2,
    weight_decay=0.01,
    smoothing=3,
    label_smoothing=0.1,
    grad_clip=1.0,
    ema_decay=0.999,
    augment=True,
    augmentation_config=None,
    deterministic=True,
    pretrained=False,
    freeze_backbone=False,
    wandb_project=None,
    run_name=None,
):
    set_seed(seed, deterministic)
    device = resolve_device(prefer_gpu)
    # 혼합정밀도는 CUDA에서만 의미가 있다.
    amp = amp and device.type == "cuda"

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
        "label_smoothing": label_smoothing,
        "grad_clip": grad_clip,
        "ema_decay": ema_decay,
        "augment": augment,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
        "deterministic": deterministic,
        "amp": amp,
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

    # 재현에 필요한데 기록되지 않던 것들. 어떤 데이터로 어떤 증강을 걸고 돌았는지
    # 로그에도 결과 파일에도 안 남아서, 끝난 실험을 나중에 되짚을 수 없었다.
    config["manifest"] = str(manifest_path)
    config["data_root"] = str(data_root)
    config["augmentation_config"] = repr(augmentation.config) if augmentation else None
    # 노트북에서 VideoAugmentation.__call__을 갈아끼워 시간축 증강을 거는 실험이
    # 있다. 패치가 남은 채로 돈 런이 실제로 있었으므로(2026-08-20) 어느 구현으로
    # 돌았는지 남긴다. 원본이면 "VideoAugmentation.__call__"이다.
    config["augmentation_call"] = VideoAugmentation.__call__.__qualname__
    num_classes = len({row["label_id"] for row in plain.rows})

    train_indices, val_indices, held_out = split_by_speaker(
        plain, val_ratio, seed, val_speakers
    )
    # 검증 화자가 정해진 뒤에 기록을 시작해야 실험 간 비교 기준이 남는다.
    config["val_speakers"] = ",".join(held_out)
    tracker = start_tracking(wandb_project, run_name, config)

    # worker를 에폭마다 새로 만들면 생성 비용이 반복된다. pin_memory는 GPU 전송을 앞당긴다.
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = 4

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        Subset(plain, val_indices),
        batch_size=batch_size,
        **loader_options,
    )

    model = LipReadingModel(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        dropout=dropout,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)
    # label smoothing은 정답에 100% 확신하지 못하게 해 과신을 줄인다.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    # 동결된 파라미터는 옵티마이저에 넣지 않는다.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    # 후반으로 갈수록 보폭을 좁혀 검증 손실이 급등하는 구간을 줄인다.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    averager = WeightAverage(model, decay=ema_decay) if ema_decay else None

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
        f"| AMP {'켬' if amp else '끔'} · worker {num_workers}"
    )
    print(
        f"label smoothing {label_smoothing} · grad clip {grad_clip} "
        f"· EMA {ema_decay or '끔'} | 저장 기준 최근 {smoothing}에폭 평균"
    )
    print(f"매니페스트 {manifest_path}")
    print(f"데이터 {data_root} | cuDNN 결정성 {'켬' if deterministic else '끔'}")
    if config["augmentation_call"] != "VideoAugmentation.__call__":
        print(f"[주의] 증강 구현이 원본이 아니다: {config['augmentation_call']}")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0
    best_smoothed = 0.0
    best_epoch = 0
    # 저장값과 별개로, 학습 중 한 번이라도 닿은 최고점을 따로 남긴다.
    # 아무도 고를 수 없는 값이지만 상한이 어디인지 알려준다.
    peak_accuracy = 0.0
    peak_epoch = 0
    # 학습 데이터를 다 맞히는 시점. 이후는 개선 신호 없이 도는 구간이다.
    saturation_epoch = 0
    recent = deque(maxlen=smoothing)

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            amp=amp,
            grad_clip=grad_clip,
            averager=averager,
        )

        # 평균 가중치로 검증한 뒤 학습용 가중치를 되돌린다.
        backup = averager.apply_to(model) if averager is not None else None
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device, amp=amp
        )
        if averager is not None:
            averager.restore(model, backup)

        scheduler.step()

        # 검증 세트가 작아 단일 에폭 정확도는 크게 진동한다.
        # 최근 몇 에폭의 평균이 최고일 때 저장해 우연한 고점을 걸러낸다.
        recent.append(val_accuracy)
        smoothed = sum(recent) / len(recent)

        if val_accuracy > peak_accuracy:
            peak_accuracy = val_accuracy
            peak_epoch = epoch
        if not saturation_epoch and train_accuracy >= 0.999:
            saturation_epoch = epoch

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
            best_epoch = epoch
            # 검증에 쓴 것과 같은 가중치를 저장해야 재현된다.
            saved = averager.apply_to(model) if averager is not None else None
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
            if averager is not None:
                averager.restore(model, saved)

    errors, val_total = collect_errors(model, val_loader, device, amp)

    # 저장값은 검증 화자를 보고 고른 값이라 낙관 편향이 있다.
    # 마지막 에폭 값이 실사용에 가까우므로 둘 다 남긴다.
    print(
        f"저장 모델 검증 정확도 {best_accuracy:.3f} "
        f"(최근 {smoothing}에폭 평균 {best_smoothed:.3f}) · "
        f"{best_epoch}/{epochs}에폭 · {checkpoint_path}"
    )
    print(
        f"최고점 {peak_accuracy:.3f} ({peak_epoch}에폭) · "
        f"마지막 에폭 {val_accuracy:.3f} (보고용) · "
        f"학습 포화 {saturation_epoch or '-'}에폭"
    )

    if tracker is not None:
        tracker.summary["best_val_acc"] = best_accuracy
        tracker.summary["best_val_acc_smoothed"] = best_smoothed
        tracker.summary["best_epoch"] = best_epoch
        tracker.summary["peak_val_acc"] = peak_accuracy
        tracker.summary["peak_epoch"] = peak_epoch
        tracker.summary["saturation_epoch"] = saturation_epoch
        tracker.finish()

    return {
        "best": best_accuracy,
        "best_smoothed": best_smoothed,
        "best_epoch": best_epoch,
        "peak": peak_accuracy,
        "peak_epoch": peak_epoch,
        "saturation_epoch": saturation_epoch,
        "errors": errors,
        "val_size": val_total,
        "last": val_accuracy,
        "trainable_params": trainable_count,
        "config": config,
    }


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
    parser.add_argument(
        "--no-amp", action="store_true", help="혼합정밀도 없이 fp32로 학습한다"
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--smoothing", type=int, default=3, help="체크포인트 판정에 쓸 에폭 수"
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="0이면 클리핑하지 않는다"
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.999, help="0이면 가중치 평균을 쓰지 않는다"
    )
    parser.add_argument(
        "--no-augment", action="store_true", help="증강 없이 학습해 효과를 비교한다"
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cuDNN 결정성. 기본 켬. --no-deterministic으로 끄면 빨라지지만 "
        "같은 시드가 같은 값을 안 낸다",
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
        amp=not args.no_amp,
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layer,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        smoothing=args.smoothing,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        augment=not args.no_augment,
        deterministic=args.deterministic,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
