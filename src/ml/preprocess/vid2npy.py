import argparse
from pathlib import Path

import numpy as np

from src.ml.preprocess.lip_crop import create_landmarker, crop_lip_frames
from src.ml.preprocess.normalize import FIXED_FRAME_COUNT, normalize_frames

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VIDEO_EXTS = (".mp4", ".avi", ".mov")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def process_video(video_path, landmarker, frames=FIXED_FRAME_COUNT):
    lips, opennesses = crop_lip_frames(video_path, landmarker)
    if not lips:
        return None
    return normalize_frames(lips, opennesses, fixed_frame_count=frames)


def run_batch(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, frames=FIXED_FRAME_COUNT):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not video_paths:
        print(f"처리할 영상이 없습니다: {raw_dir}")
        return

    landmarker = create_landmarker()
    processed, skipped, failed = 0, 0, 0
    try:
        for video_path in video_paths:
            out_path = processed_dir / f"{video_path.stem}.npy"
            if out_path.exists():
                print(f"이미 존재함, 건너뜀: {out_path}")
                skipped += 1
                continue

            print(f"처리 중: {video_path}")
            npy_data = process_video(video_path, landmarker, frames)
            if npy_data is None:
                print(f"  입 검출 실패, 건너뜀: {video_path}")
                failed += 1
                continue

            np.save(out_path, npy_data)
            print(f"  저장 완료: {out_path} (shape={npy_data.shape})")
            processed += 1
    finally:
        landmarker.close()

    print(
        f"\n총 {len(video_paths)}개 중 처리 {processed} / 건너뜀 {skipped} / 실패 {failed}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="영상을 학습용 .npy로 전처리한다.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    # 기존 데이터가 60프레임이므로 섞어 쓰려면 같은 값이어야 한다.
    parser.add_argument("--frames", type=int, default=FIXED_FRAME_COUNT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"프레임 {args.frames}개로 전처리한다")
    run_batch(args.raw_dir, args.processed_dir, args.frames)
