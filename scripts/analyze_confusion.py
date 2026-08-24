"""저장된 교차검증 체크포인트로 혼동 구조만 뽑는다. 재학습하지 않는다.

각 체크포인트는 자기 검증 화자를 안 보고 학습했으므로, 그 화자 클립에
추론하면 화자 독립 조건이 그대로 유지된다.

45프레임 시절(2026-08-19)에는 화자 8명의 1위 혼동 쌍이 전부 달랐으나
60프레임에서는 넷이 겹쳤고 물주세요가 오답률 72.3%로 단독 최악이었다.

--drop을 주면 그 문구를 정답에서 빼고 예측 후보에서도 막는다. 문구를 줄여
다시 학습했을 때의 결정과 같아지므로, 5시간짜리 재학습 없이 결과를 먼저 본다.
다만 재학습은 학습 클립도 함께 줄어들어 이 값보다 낮을 수 있다.

사용:
    python scripts/analyze_confusion.py --checkpoint-dir <드라이브>/checkpoints
    python scripts/analyze_confusion.py --drop 물주세요 추워요 도와주세요
"""

import argparse
import csv
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def label_names(manifest_path):
    rows = csv.DictReader(open(manifest_path, encoding="utf-8"))
    table = {int(r["label_id"]): r["label_text"] for r in rows}
    return [table[i] for i in sorted(table)]


def resolve_drop(names, dropped):
    """문구 이름을 번호로 바꾼다. 오타는 조용히 무시되면 안 되므로 중단한다."""
    # build_manifest가 NFC로 저장하므로 입력도 맞춘다.
    table = {unicodedata.normalize("NFC", n): i for i, n in enumerate(names)}
    ids = []
    for phrase in dropped:
        key = unicodedata.normalize("NFC", phrase)
        if key not in table:
            raise SystemExit(f"그런 문구가 없다: {phrase}\n있는 문구: {', '.join(names)}")
        ids.append(table[key])
    return ids


def count_errors(model, loader, device, amp, drop_ids):
    """맞은 수와 오답 쌍을 센다. drop_ids는 정답에서 빼고 후보에서도 막는다."""
    import torch

    model.eval()
    dropped = set(drop_ids)
    counts = Counter()
    correct = seen = 0

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(frames)
            logits = logits.float()
            if drop_ids:
                # 뺀 문구는 고를 수 없게 한다. 12문구 모델의 결정과 같아진다.
                logits[:, drop_ids] = float("-inf")
            predicted = logits.argmax(dim=1)

            for true_id, pred_id in zip(labels.tolist(), predicted.tolist()):
                if true_id in dropped:
                    continue
                seen += 1
                if true_id == pred_id:
                    correct += 1
                else:
                    counts[(true_id, pred_id)] += 1

    return counts, correct, seen


def parse_args():
    p = argparse.ArgumentParser(description="체크포인트에서 혼동 구조를 뽑는다.")
    p.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifest.csv")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    p.add_argument("--name", default="cv60", help="체크포인트 이름 앞부분")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 7])
    p.add_argument("--drop", nargs="*", default=[], help="채점에서 뺄 문구")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader, Subset

    from src.ml.models import LipReadingModel
    from src.ml.training.dataset import LipReadingDataset
    from src.ml.training.train import resolve_device

    names = label_names(args.manifest)
    drop_ids = resolve_drop(names, args.drop)
    device = resolve_device()
    amp = device.type == "cuda"

    # 증강 없이, 학습의 검증 경로와 같은 조건으로 읽는다.
    dataset = LipReadingDataset(args.manifest, args.data_root)
    speakers = sorted(set(dataset.speaker_ids))

    pairs = Counter()       # 합계 혼동 쌍
    per_speaker = {}        # 화자별 혼동 쌍
    accuracy = {}           # 화자별 (맞은 수, 전체)
    wrong = Counter()       # 문구별 오답 수
    total = Counter()       # 문구별 추론 수
    runs = 0

    for speaker in speakers:
        indices = [i for i, s in enumerate(dataset.speaker_ids) if s == speaker]
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size)
        local = Counter()
        correct = seen = used = 0

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

            counts, hit, total_seen = count_errors(model, loader, device, amp, drop_ids)
            for (true_id, pred_id), n in counts.items():
                # 방향을 합친다. 2026-08-19 표와 같은 기준이어야 비교된다.
                local[tuple(sorted((true_id, pred_id)))] += n
                wrong[true_id] += n
            correct += hit
            seen += total_seen
            used += 1
            runs += 1

        counts = Counter(int(dataset.rows[i]["label_id"]) for i in indices)
        for label_id, n in counts.items():
            if label_id not in set(drop_ids):
                total[label_id] += n * used
        per_speaker[speaker] = local
        accuracy[speaker] = (correct, seen)
        pairs.update(local)
        print(f"{speaker} 완료 · 시드 {used}개 · 채점 {seen}건")

    kept = len(names) - len(drop_ids)
    print(f"\n체크포인트 {runs}개 · 문구 {kept}개 · 추론 {sum(total.values())}건")
    if drop_ids:
        print(f"제외 {len(drop_ids)}개: {', '.join(args.drop)}")
        print(f"찍기 확률 {1 / len(names):.3f} → {1 / kept:.3f}")

    hits = sum(c for c, _ in accuracy.values())
    seen = sum(s for _, s in accuracy.values())
    print(f"\n=== 전체 정확도 {hits / seen:.3f} ({hits}/{seen}) ===")

    print("\n=== 화자별 정확도 ===")
    for speaker in sorted(accuracy, key=lambda s: -accuracy[s][0] / max(1, accuracy[s][1])):
        correct, seen = accuracy[speaker]
        if seen:
            print(f"{speaker}  {correct / seen:.3f}  ({correct}/{seen})")

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
