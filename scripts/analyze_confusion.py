"""저장된 교차검증 체크포인트로 혼동 구조만 뽑는다. 재학습하지 않는다.

각 체크포인트는 자기 검증 화자를 안 보고 학습했으므로, 그 화자 클립에
추론하면 화자 독립 조건이 그대로 유지된다.

45프레임 시절(2026-08-19)에는 화자 8명의 1위 혼동 쌍이 전부 달랐고, s06에서
찾은 "뒤 5음절이 같으면 헷갈린다"는 8화자 합계 상위 10에 들지 못해 철회됐다.
60프레임에서도 같은 그림인지 확인하는 것이 목적이다.

사용:
    python scripts/analyze_confusion.py --checkpoint-dir <드라이브>/checkpoints
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def label_names(manifest_path):
    rows = csv.DictReader(open(manifest_path, encoding="utf-8"))
    table = {int(r["label_id"]): r["label_text"] for r in rows}
    return [table[i] for i in sorted(table)]


def parse_args():
    p = argparse.ArgumentParser(description="체크포인트에서 혼동 구조를 뽑는다.")
    p.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifest.csv")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    p.add_argument("--name", default="cv60", help="체크포인트 이름 앞부분")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 7])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader, Subset

    from src.ml.models import LipReadingModel
    from src.ml.training.dataset import LipReadingDataset
    from src.ml.training.train import collect_errors, resolve_device

    names = label_names(args.manifest)
    device = resolve_device()
    amp = device.type == "cuda"

    # 증강 없이, 학습의 검증 경로와 같은 조건으로 읽는다.
    dataset = LipReadingDataset(args.manifest, args.data_root)
    speakers = sorted(set(dataset.speaker_ids))

    pairs = Counter()       # 합계 혼동 쌍
    per_speaker = {}        # 화자별 혼동 쌍
    wrong = Counter()       # 문구별 오답 수
    total = Counter()       # 문구별 추론 수
    runs = 0

    for speaker in speakers:
        indices = [i for i, s in enumerate(dataset.speaker_ids) if s == speaker]
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size)
        local = Counter()
        used = 0

        for seed in args.seeds:
            path = args.checkpoint_dir / f"{args.name}_{speaker}_seed{seed}.pt"
            if not path.exists():
                print(f"없음, 건너뜀: {path.name}")
                continue

            saved = torch.load(path, map_location=device)
            model = LipReadingModel(
                num_classes=saved["num_classes"],
                hidden_dim=saved["hidden_dim"],
                num_layer=saved["num_layer"],
                dropout=saved["dropout"],
            ).to(device)
            model.load_state_dict(saved["model_state"])

            errors, _ = collect_errors(model, loader, device, amp)
            for true_id, pred_id, count in errors:
                # 방향을 합친다. 2026-08-19 표와 같은 기준이어야 비교된다.
                local[tuple(sorted((true_id, pred_id)))] += count
                wrong[true_id] += count
            used += 1
            runs += 1

        counts = Counter(int(dataset.rows[i]["label_id"]) for i in indices)
        for label_id, n in counts.items():
            total[label_id] += n * used
        per_speaker[speaker] = local
        pairs.update(local)
        print(f"{speaker} 완료 · 시드 {used}개 · 클립 {len(indices)}개")

    print(f"\n체크포인트 {runs}개 · 추론 {sum(total.values())}건")

    print(f"\n=== 합계 혼동 상위 {args.top}쌍 ===")
    for (a, b), n in pairs.most_common(args.top):
        print(f"{n:5d}  {names[a]} ↔ {names[b]}")

    print("\n=== 화자별 1위 ===")
    for speaker in sorted(per_speaker):
        local = per_speaker[speaker]
        if not local:
            print(f"{speaker}  오답 없음")
            continue
        (a, b), n = local.most_common(1)[0]
        print(f"{speaker}  {names[a]} ↔ {names[b]} ({n}회)")

    print("\n=== 문구별 오답률 ===")
    rates = sorted(
        ((wrong[i] / total[i], i) for i in range(len(names)) if total[i]),
        reverse=True,
    )
    for rate, i in rates:
        print(f"{rate:6.1%}  {names[i]:<10} ({wrong[i]}/{total[i]})")


if __name__ == "__main__":
    main()
