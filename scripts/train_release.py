"""실사용 모델을 만든다. 화자를 빼지 않고 전원으로 학습한다.

교차검증 체크포인트(cv60_*.pt)는 한 명씩 빼고 학습한 측정용이다. 실사용에
쓰면 이유 없이 그 한 명을 못 본 모델을 쓰게 된다.

검증 화자가 없으니 "최근 3에폭 평균이 최고일 때 저장"을 쓸 수 없다. 대신
에폭을 고정한다. 50은 2026-08-21의 24런을 10단위로 끊어 잰 값이다. 검증을
보지 않고 에폭만 고정했을 때 평균이 가장 높았고(0.599), 40~80이 0.012 안에
들어 있어 그 구간 어디를 잡아도 사실상 같다.

시드마다 파일이 하나씩 나온다. 추론에서 확률을 평균하면 시드 편차 0.0725가
줄어든다. 같은 영상에 답이 갈리는 것을 막는 값싼 방법이다.

사용:
    python scripts/train_release.py --seeds 42 1 7
"""

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 검증 없이 고를 수 있는 최선의 에폭. 위 docstring 참고.
DEFAULT_EPOCHS = 50


def label_names(manifest_path):
    """label_id 순서대로 문구를 늘어놓는다. 이 순서가 모델 출력 순서다."""
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    table = {int(r["label_id"]): r["label_text"] for r in rows}
    ids = sorted(table)
    if ids != list(range(len(ids))):
        raise SystemExit(f"label_id가 0부터 연속이 아니다: {ids}")
    return [table[i] for i in ids]


def train_one(args, seed, labels, checkpoint_path):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    import numpy as np

    from src.ml.models import LipReadingModel
    from src.ml.preprocess.augmentation import VideoAugmentation
    from src.ml.training.dataset import LipReadingDataset
    from src.ml.training.train import (
        WeightAverage,
        resolve_device,
        run_epoch,
        set_seed,
    )

    set_seed(seed)
    device = resolve_device()
    amp = device.type == "cuda"

    augmentation = VideoAugmentation(seed=seed) if args.augment else None
    dataset = LipReadingDataset(args.manifest, args.data_root, augmentation=augmentation)

    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = 4
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, **loader_options
    )

    model = LipReadingModel(
        num_classes=len(labels),
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layer,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    averager = WeightAverage(model, decay=args.ema_decay)

    # 상수가 아니라 실제 데이터에서 읽는다. 30프레임 데이터로 학습해 놓고
    # 체크포인트에 60이라 적히면 추론이 그 불일치를 못 잡는다.
    sample = np.load(Path(args.data_root) / dataset.rows[0]["clip_path"])
    frame_count = int(sample.shape[0])

    speakers = sorted(set(dataset.speaker_ids))
    print(f"장치: {device} | 클래스 {len(labels)}개 | 시드 {seed}")
    print(
        f"학습 {len(dataset)}개 클립 · 화자 {len(speakers)}명 {speakers} "
        f"· {frame_count}프레임 · 검증 없음"
    )
    print(
        f"에폭 {args.epochs} 고정 · batch {args.batch_size} · lr {args.learning_rate} "
        f"| 증강 {'켬' if args.augment else '끔'}"
    )

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        loss, accuracy = run_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer,
            amp=amp,
            grad_clip=args.grad_clip,
            averager=averager,
        )
        scheduler.step()
        print(
            f"[{epoch:3d}/{args.epochs}] train loss {loss:.4f} acc {accuracy:.3f} "
            f"| lr {scheduler.get_last_lr()[0]:.2e}"
        )

    # 검증이 없으므로 되돌릴 이유가 없다. 평균 가중치를 그대로 저장한다.
    averager.apply_to(model)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            # 문구를 함께 넣어야 추론이 매니페스트 없이 돈다. label_id는
            # 문구를 하나만 늘려도 통째로 밀리므로 번호만 믿으면 안 된다.
            "labels": labels,
            "num_classes": len(labels),
            "hidden_dim": args.hidden_dim,
            "num_layer": args.num_layer,
            "dropout": args.dropout,
            "frames": frame_count,
            "epochs": args.epochs,
            "seed": seed,
            "clips": len(dataset),
            "speakers": speakers,
        },
        checkpoint_path,
    )
    print(f"저장 {checkpoint_path} · {round(time.time() - started)}초\n")


def parse_args():
    p = argparse.ArgumentParser(description="실사용 모델을 전 화자로 학습한다.")
    p.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifest.csv")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    p.add_argument("--name", default="release", help="체크포인트 이름 앞부분")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 7])
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    # 아래 기본값은 run_experiment.py와 같아야 한다. 다르면 측정한 성능이 아니다.
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--hidden-dim", type=int, default=300)
    p.add_argument("--num-layer", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.998)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.augment = not args.no_augment
    labels = label_names(args.manifest)

    for seed in args.seeds:
        path = args.checkpoint_dir / f"{args.name}_seed{seed}.pt"
        if path.exists():
            print(f"이미 있음, 건너뜀: {path}")
            continue
        print(f"======== 실사용 학습 · seed {seed} ========")
        train_one(args, seed, labels, path)

    print(f"문구 {len(labels)}개: {', '.join(labels)}")


if __name__ == "__main__":
    main()
