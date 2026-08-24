"""영상 하나를 문구 하나로 바꾼다.

전처리는 학습과 같은 함수를 그대로 부른다. 여기서 한 줄이라도 달라지면
성능이 조용히 떨어지고 원인을 찾기 어렵다. 그래서 크롭과 정규화를 다시
구현하지 않고 vid2npy.process_video를 쓴다.

체크포인트를 여러 개 주면 확률을 평균한다. 조건-화자 한 칸의 시드 편차가
0.0725라 한 모델만 쓰면 같은 영상에도 답이 갈린다.

사용:
    python scripts/predict.py 영상.mp4
    python scripts/predict.py 영상1.mp4 영상2.mp4 --verbose
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_models(paths, device):
    import torch

    from src.ml.models import LipReadingModel
    from src.ml.preprocess.normalize import FIXED_FRAME_COUNT

    models, labels = [], None
    for path in paths:
        saved = torch.load(path, map_location=device)
        if "labels" not in saved:
            raise SystemExit(
                f"{path}에 문구 목록이 없다. 교차검증 체크포인트로 보인다.\n"
                "실사용 모델은 scripts/train_release.py로 만들어야 한다."
            )
        if labels is None:
            labels = saved["labels"]
        elif saved["labels"] != labels:
            raise SystemExit(f"{path}의 문구 목록이 앞 체크포인트와 다르다")
        if saved["frames"] != FIXED_FRAME_COUNT:
            raise SystemExit(
                f"{path}는 {saved['frames']}프레임으로 학습했는데 "
                f"지금 전처리는 {FIXED_FRAME_COUNT}프레임이다"
            )

        model = LipReadingModel(
            num_classes=saved["num_classes"],
            hidden_dim=saved["hidden_dim"],
            num_layer=saved["num_layer"],
            dropout=saved["dropout"],
        ).to(device)
        model.load_state_dict(saved["model_state"])
        model.eval()
        models.append(model)

    return models, labels


def predict(models, clip, device):
    """전처리된 클립 하나에 대해 모델들의 확률을 평균한다."""
    import numpy as np
    import torch

    from src.ml.training.dataset import PIXEL_MAX

    # 학습 때 Dataset이 하던 변환과 같아야 한다. 증강과 ImageNet 정규화는
    # 검증 경로에서도 쓰지 않았으므로 여기서도 쓰지 않는다.
    frames = torch.from_numpy(np.ascontiguousarray(clip))
    frames = frames.permute(3, 0, 1, 2).float().div_(PIXEL_MAX)
    frames = frames.unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = [torch.softmax(m(frames), dim=1) for m in models]
    return torch.stack(probabilities).mean(dim=0).squeeze(0).cpu()


def parse_args():
    p = argparse.ArgumentParser(description="영상에서 문구를 예측한다.")
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument(
        "--checkpoint",
        nargs="+",
        type=Path,
        default=None,
        help="생략하면 checkpoints/release_seed*.pt를 모두 쓴다",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--verbose", action="store_true", help="진단용. 문구별 확률을 전부 찍는다"
    )
    return p.parse_args()


def main():
    args = parse_args()

    paths = args.checkpoint
    if paths is None:
        paths = sorted((PROJECT_ROOT / "checkpoints").glob("release_seed*.pt"))
    if not paths:
        raise SystemExit(
            "체크포인트가 없다. 먼저 python scripts/train_release.py 를 돌릴 것"
        )

    from src.ml.preprocess.lip_crop import create_landmarker
    from src.ml.preprocess.vid2npy import process_video
    from src.ml.training.train import resolve_device

    device = resolve_device(prefer_gpu=not args.cpu)
    models, labels = load_models(paths, device)
    print(f"장치 {device} · 모델 {len(models)}개 · 문구 {len(labels)}개")

    landmarker = create_landmarker()
    try:
        for video in args.videos:
            if not video.exists():
                print(f"{video.name}: 파일이 없다")
                continue

            clip = process_video(video, landmarker)
            if clip is None:
                print(f"{video.name}: 입을 한 번도 못 찾았다")
                continue

            probabilities = predict(models, clip, device)
            best = int(probabilities.argmax())
            print(f"{video.name}: {labels[best]} ({probabilities[best]:.2f})")

            if args.verbose:
                order = probabilities.argsort(descending=True)
                for i in order:
                    print(f"    {labels[i]:<12} {probabilities[i]:.3f}")
    finally:
        landmarker.close()


if __name__ == "__main__":
    main()
