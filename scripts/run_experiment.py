"""화자 독립 교차검증을 한 번에 돌리고 결과를 한 파일에 모은다.

노트북에서 셀을 조립하지 않는 이유는 커널에 상태가 남기 때문이다. 실제로
2026-08-20에 모델 패치가 남은 채로 실험 두 개가 오염됐다. 이 스크립트는
설정을 기록하고 매 실행마다 대조하므로 같은 사고가 다시 나지 않는다.

사용:
    python scripts/run_experiment.py --name base60 --seeds 42
    python scripts/run_experiment.py --name no5 --exclude s05 --seeds 42 1 7
    python scripts/run_experiment.py --name no5 --summary --baseline base60
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"

# 조건-화자 한 칸의 3시드 표준편차. 2026-08-20에 5칸으로 측정했다.
# 판정선은 여기서 나오므로 새 측정이 쌓이면 갱신할 것.
CELL_SD = 0.0725

MANIFEST_FIELDS = ["clip_path", "label_id", "label_text", "speaker_id", "take"]


def filter_manifest(manifest_path, excluded, out_path):
    """지정한 화자의 행을 뺀 매니페스트를 새로 쓴다. 문구 수가 줄면 중단한다."""
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    kept = [r for r in rows if r["speaker_id"] not in excluded]
    phrases = {r["label_text"] for r in kept}
    if len(phrases) != len({r["label_text"] for r in rows}):
        raise SystemExit("제외 후 문구가 줄었다. label_id가 어긋나므로 중단한다.")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(kept)

    speakers = sorted({r["speaker_id"] for r in kept})
    print(f"매니페스트 {len(rows)} → {len(kept)}행 · 문구 {len(phrases)}개 · 화자 {speakers}")
    return kept


def speakers_in(manifest_path):
    rows = csv.DictReader(open(manifest_path, encoding="utf-8"))
    return sorted({row["speaker_id"] for row in rows})


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def check_config(records, config):
    """이전 런과 설정이 다르면 중단한다. 오염과 이어받기 사고를 막는다."""
    if not records:
        return
    previous = records[0].get("config")
    if previous is None:
        return
    changed = {
        key: (previous.get(key), value)
        for key, value in config.items()
        if key != "seed" and previous.get(key) != value
    }
    if changed:
        lines = [f"  {k}: {old} → {new}" for k, (old, new) in changed.items()]
        raise SystemExit(
            "이미 기록된 런과 설정이 다르다. 다른 이름으로 돌리거나 결과 파일을 지울 것.\n"
            + "\n".join(lines)
        )


def label_names(manifest_path):
    """label_id를 문구로 바꾸는 표. 매니페스트가 없으면 번호를 그대로 쓴다."""
    if not Path(manifest_path).exists():
        return {}
    rows = csv.DictReader(open(manifest_path, encoding="utf-8"))
    return {int(r["label_id"]): r["label_text"] for r in rows}


def report_confusion(records, names, top=8):
    """오답을 문구 쌍으로 모아 자주 헷갈리는 순으로 보여준다."""
    pairs, per_phrase = {}, {}
    total_errors = total_clips = 0
    for record in records:
        total_clips += record.get("val_size") or 0
        for true_id, pred_id, count in record.get("errors") or []:
            pairs[(true_id, pred_id)] = pairs.get((true_id, pred_id), 0) + count
            per_phrase[true_id] = per_phrase.get(true_id, 0) + count
            total_errors += count
    if not pairs:
        return

    def name(i):
        return names.get(i, str(i))

    print()
    print(f"오답 {total_errors}개 / 클립 {total_clips}개")
    print("자주 헷갈리는 쌍")
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])[:top]
    for (true_id, pred_id), count in ranked:
        share = 100.0 * count / total_errors
        print(f"  {name(true_id)} → {name(pred_id)}   {count}회 ({share:.0f}%)")

    print("틀리는 문구")
    worst = sorted(per_phrase.items(), key=lambda kv: -kv[1])[:5]
    for label_id, count in worst:
        print(f"  {name(label_id)}   {count}회")


def summarize(records, baseline_records=None, names=None):
    by_speaker = {}
    for r in records:
        by_speaker.setdefault(r["speaker"], []).append(r)

    seed_counts = {len(v) for v in by_speaker.values()}
    print()
    print("화자   최고점   저장값   마지막   최고에폭   시드 수")
    best_means, last_means, peak_means = {}, {}, {}
    peaks = []

    def column(values, width=8, digits=3):
        if not values:
            return f"{'-':>{width}s}"
        return f"{statistics.mean(values):{width}.{digits}f}"

    for speaker in sorted(by_speaker):
        runs = by_speaker[speaker]
        best_means[speaker] = statistics.mean(r["best"] for r in runs)
        lasts = [r["last"] for r in runs if r.get("last") is not None]
        peak_values = [r["peak"] for r in runs if r.get("peak") is not None]
        last_means[speaker] = statistics.mean(lasts) if lasts else None
        peak_means[speaker] = statistics.mean(peak_values) if peak_values else None
        epochs = [r["peak_epoch"] for r in runs if r.get("peak_epoch")]
        peaks.extend(
            (r["peak_epoch"], r["config"]["epochs"])
            for r in runs
            if r.get("peak_epoch") and r.get("config")
        )
        print(
            f"{speaker:5s} {column(peak_values)} {best_means[speaker]:8.3f}"
            f" {column(lasts)} {column(epochs, 10, 0)} {len(runs):8d}"
        )
    known_last = [v for v in last_means.values() if v is not None]
    known_peak = [v for v in peak_means.values() if v is not None]
    print(
        f"{'평균':5s} {column(known_peak)} "
        f"{statistics.mean(best_means.values()):8.3f} {column(known_last)}"
    )

    print()
    report_health(records, best_means, last_means, peak_means, peaks)
    if names is not None:
        report_confusion(records, names)

    if baseline_records is None:
        return

    base = {}
    for r in baseline_records:
        base.setdefault(r["speaker"], []).append(r["best"])
    shared = sorted(set(best_means) & set(base))
    if not shared:
        print("\n기준선과 겹치는 화자가 없다.")
        return

    diffs = [best_means[s] - statistics.mean(base[s]) for s in shared]
    mean = statistics.mean(diffs)
    base_worst = min(shared, key=lambda s: statistics.mean(base[s]))
    now_worst = min(shared, key=lambda s: best_means[s])
    print()
    print("화자   기준선   이번    차이")
    for speaker, diff in zip(shared, diffs):
        print(
            f"{speaker:5s} {statistics.mean(base[speaker]):8.3f}"
            f" {best_means[speaker]:8.3f} {diff:+8.3f}"
        )
    print(f"{'평균':5s} {'':8s} {'':8s} {mean:+8.3f}")
    # 평균이 올라도 최저 화자가 그대로면 화자 독립 관점에서는 진전이 아니다.
    print(
        f"{'최저':5s} {statistics.mean(base[base_worst]):8.3f}"
        f" {best_means[now_worst]:8.3f}"
        f" {best_means[now_worst] - statistics.mean(base[base_worst]):+8.3f}"
        f"   ({base_worst} → {now_worst})"
    )

    wins = sum(1 for d in diffs if d > 0)
    line = None
    if len(shared) > 1 and len(seed_counts) == 1:
        seeds = seed_counts.pop()
        line = 2 * CELL_SD / math.sqrt(seeds) * math.sqrt(2) / math.sqrt(len(shared))

    print()
    if len(diffs) > 1:
        se = statistics.stdev(diffs) / math.sqrt(len(diffs))
        t = mean / se if se else float("inf")
        print(f"짝 t검정  평균 {mean:+.3f} · SE {se:.3f} · t {t:.2f} · df {len(diffs) - 1}")
    print(f"부호      {wins}승 {len(diffs) - wins}패")
    if line is not None:
        verdict = "넘었다" if abs(mean) > line else "못 넘었다"
        print(f"판정선    {line:.3f} · {verdict}")
    else:
        print("판정선    화자별 시드 수가 달라 계산하지 않았다.")


def report_health(records, best_means, last_means, peak_means, peaks):
    """평균만 보면 놓치는 것들을 한 자리에 모은다.

    화자 독립 시스템에서 실질 성능은 평균이 아니라 최저 화자다. 지금까지 통한
    개입이 전부 상위 화자만 올리고 격차를 벌렸는데(0.307 → 0.406), 요약에
    최저 화자가 없어 늦게 발견했다.
    """
    worst = min(best_means, key=best_means.get)
    top = max(best_means, key=best_means.get)
    values = list(best_means.values())
    print(f"최저 화자   {worst} {best_means[worst]:.3f}   ← 실질 성능")
    print(
        f"격차        {best_means[top] - best_means[worst]:.3f}"
        f"  ({top} {best_means[top]:.3f} − {worst} {best_means[worst]:.3f})"
    )
    if len(values) > 1:
        print(f"화자간 SD   {statistics.stdev(values):.3f}")

    # 최고점과 마지막의 차이가 곧 검증 집합 선택 편향의 크기다.
    gaps = [
        peak_means[k] - last_means[k]
        for k in peak_means
        if peak_means[k] is not None and last_means[k] is not None
    ]
    if gaps:
        print(f"부풀림      {statistics.mean(gaps):.3f}  (최고점 − 마지막)")

    saturations = [
        (r["saturation_epoch"], r["config"]["epochs"])
        for r in records
        if r.get("saturation_epoch") and r.get("config")
    ]
    if saturations:
        mean_sat = statistics.mean(e for e, _ in saturations)
        total = statistics.mean(t for _, t in saturations)
        print(
            f"학습 포화   {mean_sat:.0f}에폭 / {total:.0f}"
            f"  (이후 {total - mean_sat:.0f}에폭은 개선 신호 없이 돈다)"
        )

    if peaks:
        ratios = [100.0 * e / total for e, total in peaks]
        print(
            f"최고점 위치 전체 에폭의 {min(ratios):.0f}% ~ {max(ratios):.0f}%"
            f" (중앙값 {statistics.median(ratios):.0f}%)"
        )


def parse_args():
    p = argparse.ArgumentParser(description="화자 독립 교차검증 실험 러너")
    p.add_argument("--name", required=True, help="실험 이름. results/<이름>.json에 쌓인다")
    p.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "manifest.csv")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    p.add_argument("--exclude", nargs="*", default=[], help="학습과 평가 양쪽에서 뺄 화자")
    p.add_argument("--speakers", nargs="*", default=None, help="검증할 화자. 생략하면 전체")
    p.add_argument("--seeds", nargs="*", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--hidden-dim", type=int, default=300)
    p.add_argument("--num-layer", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--ema-decay", type=float, default=0.998)
    p.add_argument("--smoothing", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-augment", action="store_true", help="공간 증강을 끈다")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--summary", action="store_true", help="학습 없이 요약만 출력")
    p.add_argument("--baseline", default=None, help="비교할 실험 이름")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    result_path = RESULTS_DIR / f"{args.name}.json"
    records = load(result_path)

    baseline = None
    if args.baseline:
        baseline_path = RESULTS_DIR / f"{args.baseline}.json"
        if not baseline_path.exists():
            raise SystemExit(f"기준선 파일이 없다: {baseline_path}")
        baseline = load(baseline_path)

    names = label_names(args.manifest)
    if args.summary:
        summarize(records, baseline, names)
        return

    manifest = args.manifest
    if args.exclude:
        manifest = RESULTS_DIR / f"manifest_{args.name}.csv"
        filter_manifest(args.manifest, set(args.exclude), manifest)

    from src.ml.training.train import train

    speakers = args.speakers or speakers_in(manifest)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    done = {(r["speaker"], r["seed"]) for r in records}
    started = time.time()

    for speaker in speakers:
        for seed in args.seeds:
            if (speaker, seed) in done:
                continue
            print()
            print(f"======== {args.name} · {speaker} · seed {seed} ========")
            outcome = train(
                manifest_path=manifest,
                data_root=args.data_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=seed,
                val_speakers=[speaker],
                checkpoint_path=args.checkpoint_dir
                / f"{args.name}_{speaker}_seed{seed}.pt",
                num_workers=args.num_workers,
                amp=True,
                ema_decay=args.ema_decay,
                hidden_dim=args.hidden_dim,
                num_layer=args.num_layer,
                dropout=args.dropout,
                smoothing=args.smoothing,
                augment=not args.no_augment,
                wandb_project=args.wandb_project,
                run_name=f"{args.name}_{speaker}_seed{seed}",
            )
            config = dict(outcome["config"], excluded=sorted(args.exclude))
            check_config(records, config)
            records.append(
                {
                    "speaker": speaker,
                    "seed": seed,
                    "best": round(outcome["best"], 4),
                    "best_smoothed": round(outcome["best_smoothed"], 4),
                    "best_epoch": outcome["best_epoch"],
                    "peak": round(outcome["peak"], 4),
                    "peak_epoch": outcome["peak_epoch"],
                    "saturation_epoch": outcome["saturation_epoch"],
                    "errors": outcome["errors"],
                    "val_size": outcome["val_size"],
                    "last": round(outcome["last"], 4),
                    "trainable_params": outcome["trainable_params"],
                    "config": config,
                }
            )
            result_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            print(f"저장 · {len(records)}런 · 경과 {round((time.time() - started) / 60)}분")

    summarize(records, baseline, names)


if __name__ == "__main__":
    main()
