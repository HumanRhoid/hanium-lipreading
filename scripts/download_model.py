"""MediaPipe Face Landmarker 모델 다운로드.

git에 바이너리를 넣지 않기 위해, 필요할 때 이 스크립트로 받는다.
    python scripts/download_model.py
"""
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEST = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def main():
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"이미 있음: {DEST}")
        return
    print(f"다운로드 중… {URL}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"완료: {DEST} ({DEST.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
