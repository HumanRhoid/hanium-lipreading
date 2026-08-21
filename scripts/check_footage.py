"""촬영본을 전처리 전에 점검한다.

입술 크기로 합격을 매기지는 않는다. 2026-08-17에 규격을 충족한 s07(0.351)이
미달인 s09(0.393)보다 낮아 크기가 성적을 예측하지 못하는 것이 확인됐다.
대신 명백한 결함만 잡는다.

    얼굴 미검출     그 프레임은 통째로 빠진다
    원본 해상도     s05만 720p였고 최하위였다
    세로 정보 손실  크롭 높이가 80 미만이면 늘려 쓴다
    문구 균형       클립 수가 문구마다 다르면 학습이 기운다
    발화 길이       너무 짧으면 60프레임을 복제로 채운다

사용:
    python scripts/check_footage.py --src <촬영본 폴더> --speaker s09
    python scripts/check_footage.py --src <폴더> --preview out.png

미리보기 이미지에는 입술이 그대로 남는다. 저장소에 커밋하지 말 것.
"""

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_EXTS = (".mp4", ".avi", ".mov")

# 전처리와 같은 값. 여기가 바뀌면 판정 기준도 함께 바뀐다.
MARGIN = 0.5
TARGET_W, TARGET_H = 112, 80
FIXED_FRAMES = 60


def find_clips(source):
    """문구별 하위 폴더든 평평한 폴더든 받는다. 윈도우 중복은 걸러낸다."""
    source = Path(source)
    groups = {}
    subdirs = sorted(p for p in source.iterdir() if p.is_dir())
    for directory in subdirs or [source]:
        clips = sorted(
            {p.resolve() for p in directory.iterdir()
             if p.suffix.lower() in VIDEO_EXTS}
        )
        if clips:
            groups[directory.name] = clips
    return groups


def measure(clips, landmarker, frames_per_clip, clips_per_group):
    import cv2
    import mediapipe as mp
    from src.ml.preprocess.lip_crop import LIP_LANDMARKS

    widths, heights, sizes, lengths = [], [], set(), []
    detected = missed = 0
    sample = None

    for path in clips[:clips_per_group]:
        capture = cv2.VideoCapture(str(path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        lengths.append(total)
        step = max(1, total // (frames_per_clip * 2))
        taken = 0
        for index in range(total // 4, 3 * total // 4, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            sizes.add((width, height))
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect(image)
            if not result.face_landmarks:
                missed += 1
                continue
            detected += 1
            marks = result.face_landmarks[0]
            xs = [marks[i].x * width for i in LIP_LANDMARKS]
            ys = [marks[i].y * height for i in LIP_LANDMARKS]
            x0, x1 = int(min(xs)), int(max(xs))
            y0, y1 = int(min(ys)), int(max(ys))
            widths.append(x1 - x0)
            heights.append(y1 - y0)
            if sample is None:
                mx = int((x1 - x0) * MARGIN)
                my = int((y1 - y0) * MARGIN)
                crop = frame[
                    max(0, y0 - my) : min(height, y1 + my),
                    max(0, x0 - mx) : min(width, x1 + mx),
                ]
                if crop.size:
                    sample = cv2.resize(crop, (TARGET_W, TARGET_H))
            taken += 1
            if taken >= frames_per_clip:
                break
        capture.release()

    return {
        "widths": widths,
        "heights": heights,
        "sizes": sizes,
        "lengths": lengths,
        "detected": detected,
        "missed": missed,
        "sample": sample,
    }


def report(groups, results, speaker):
    print("문구                  클립  원본       상자      세로배율  미검출  프레임")
    problems = []
    for name, clips in groups.items():
        found = results[name]
        if not found["heights"]:
            print(f"{name.ljust(20)} {len(clips):5d}   얼굴을 한 번도 못 찾았다")
            problems.append(f"{name}: 얼굴 미검출")
            continue

        box_w = statistics.median(found["widths"])
        box_h = statistics.median(found["heights"])
        # 크롭은 상자의 2배이고 그것을 80으로 줄인다. 1을 넘으면 늘려 쓰는 것이다.
        scale = TARGET_H / (2 * box_h)
        sizes = "·".join(f"{w}x{h}" for w, h in sorted(found["sizes"]))
        miss_rate = found["missed"] / max(1, found["detected"] + found["missed"])
        frames = statistics.median(found["lengths"])

        print(
            f"{name.ljust(20)} {len(clips):5d}  {sizes:10s} "
            f"{int(box_w):3d}x{int(box_h):3d}  {'x' + format(scale, '.2f'):>8s} "
            f"{miss_rate * 100:6.0f}% {int(frames):7d}"
        )
        if scale > 1.0:
            problems.append(f"{name}: 세로를 x{scale:.2f}로 늘려 쓴다")
        if miss_rate > 0.05:
            problems.append(f"{name}: 얼굴 미검출 {miss_rate * 100:.0f}%")
        if frames < FIXED_FRAMES:
            problems.append(f"{name}: {int(frames)}프레임뿐이라 복제로 채운다")

    counts = [len(c) for c in groups.values()]
    all_sizes = set().union(*(r["sizes"] for r in results.values() if r["sizes"]))
    print()
    print(f"화자 {speaker} · 문구 {len(groups)}개 · 클립 {sum(counts)}개")
    if len(set(counts)) > 1:
        print(f"문구별 클립 수가 {min(counts)}~{max(counts)}로 다르다")
    if len(all_sizes) > 1:
        problems.append(f"원본 해상도가 섞여 있다: {sorted(all_sizes)}")

    print()
    if problems:
        print("문제")
        for line in problems:
            print(" ", line)
    else:
        print("명백한 결함은 없다.")
        print("입술 크기는 성적을 예측하지 못하므로 크기만으로 판단하지 말 것.")


def save_preview(results, path, columns=5, zoom=3):
    import cv2
    import numpy as np

    tiles, names = [], []
    for name, found in results.items():
        if found["sample"] is not None:
            tiles.append(
                cv2.resize(
                    found["sample"],
                    (TARGET_W * zoom, TARGET_H * zoom),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            names.append(name)
    if not tiles:
        return
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack(
        [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    )
    cv2.imwrite(str(path), grid)
    print()
    print(f"미리보기 {path} · 입술이 남으므로 커밋하지 말 것")
    print("순서", " · ".join(names))


def parse_args():
    parser = argparse.ArgumentParser(description="촬영본 점검")
    parser.add_argument("--src", type=Path, required=True, help="촬영본 폴더")
    parser.add_argument("--speaker", default="?", help="화자 이름. 출력에만 쓴다")
    parser.add_argument("--frames", type=int, default=6, help="클립당 볼 프레임 수")
    parser.add_argument("--clips", type=int, default=2, help="문구당 볼 클립 수")
    parser.add_argument("--preview", type=Path, default=None, help="크롭 미리보기 PNG")
    return parser.parse_args()


def main():
    args = parse_args()
    groups = find_clips(args.src)
    if not groups:
        raise SystemExit(f"영상을 찾지 못했다: {args.src}")

    from src.ml.preprocess.lip_crop import create_landmarker

    landmarker = create_landmarker()
    try:
        results = {
            name: measure(clips, landmarker, args.frames, args.clips)
            for name, clips in groups.items()
        }
    finally:
        landmarker.close()

    report(groups, results, args.speaker)
    if args.preview:
        save_preview(results, args.preview)


if __name__ == "__main__":
    main()
