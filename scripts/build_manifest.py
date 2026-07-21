"""전처리된 .npy 파일명을 파싱해 매니페스트 CSV를 자동 생성.

파일명 규칙:  {화자}_{문구}_{번호}.npy   예) s01_물주세요_01.npy
    - 화자/문구/번호는 밑줄(_)로 구분 → 문구 안에는 밑줄을 쓰지 말 것

사용:
    python scripts/build_manifest.py

결과: data/manifest.csv  (clip_path, label_id, label_text, speaker_id, take)
"""
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"


def parse_stem(stem):
    """s01_물주세요_01 → (speaker, phrase, take). 규칙 불일치 시 None."""
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    speaker, phrase, take = parts
    if not (speaker and phrase and take):
        return None
    return speaker, phrase, take


def build(processed_dir=PROCESSED_DIR, manifest_path=MANIFEST_PATH):
    processed_dir = Path(processed_dir)
    npy_files = sorted(processed_dir.glob("*.npy"))
    if not npy_files:
        print(f"전처리된 .npy가 없습니다: {processed_dir}")
        return

    rows, skipped = [], []
    for npy in npy_files:
        parsed = parse_stem(npy.stem)
        if parsed is None:
            skipped.append(npy.name)
            continue
        speaker, phrase, take = parsed
        rows.append(
            {
                "clip_path": f"processed/{npy.name}",
                "label_text": phrase,
                "speaker_id": speaker,
                "take": take,
            }
        )

    if not rows:
        print("규칙에 맞는 파일이 없습니다. 파일명: 화자_문구_번호.npy")
        if skipped:
            print("건너뛴 파일:", ", ".join(skipped))
        return

    # 문구 → label_id (정렬 순서로 안정적 부여)
    labels = sorted({r["label_text"] for r in rows})
    label_to_id = {text: i for i, text in enumerate(labels)}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clip_path", "label_id", "label_text", "speaker_id", "take"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "label_id": label_to_id[r["label_text"]]})

    print(f"매니페스트 생성: {manifest_path}")
    print(f"  클립 {len(rows)}개 · 문구 {len(labels)}개 · 화자 {len({r['speaker_id'] for r in rows})}명")
    print("  라벨 매핑:", ", ".join(f"{i}={t}" for t, i in label_to_id.items()))
    if skipped:
        print(f"  [주의] 규칙 불일치로 건너뜀 {len(skipped)}개:", ", ".join(skipped))


if __name__ == "__main__":
    build()
