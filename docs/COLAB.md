# 코랩 실행 순서

셀 세 개로 끝난다. 실험 조합은 저장소의 `scripts/run_experiment.py`가 맡으므로
노트북에서 모델이나 데이터셋을 고치지 않는다.

2026-08-20에 노트북에 남은 패치 때문에 실험 두 개가 오염됐다. 매번 새 프로세스로
돌리고 설정을 파일에 기록하는 것이 그 사고의 대책이다.

## 1. 셋업

런타임을 새로 시작할 때마다 한 번 돌린다. 40초쯤 걸린다.

```python
from google.colab import drive
drive.mount("/content/drive")

import os, sys, shutil, time, subprocess
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/hanium-lipreading")
PROCESSED = DRIVE_ROOT / "processed_f60"
REPO = Path("/content/hanium-lipreading")

n_drive = len(list(PROCESSED.glob("*.npy")))
print("Drive 전처리본", n_drive, "개")
assert n_drive > 0, "전처리본이 없다"

os.chdir("/content")
if (REPO / ".git").exists():
    subprocess.run(["git", "-C", str(REPO), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "develop"], check=True)
    subprocess.run(["git", "-C", str(REPO), "pull"], check=True)
else:
    subprocess.run(["git", "clone", "-b", "develop",
                    "https://github.com/HumanRhoid/hanium-lipreading.git",
                    str(REPO)], check=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
subprocess.run(["pip", "install", "--quiet", "wandb"], check=False)

from scripts.build_manifest import build
MANIFEST = DRIVE_ROOT / "manifest_f60.csv"
build(processed_dir=PROCESSED, manifest_path=MANIFEST)

# Drive에서 직접 읽으면 느려서 로컬 디스크로 옮긴다.
# build_manifest가 clip_path를 "processed/…"로 쓰므로 폴더명을 바꾸면 안 된다.
DATA_ROOT = Path("/content/data_f60")
LOCAL = DATA_ROOT / "processed"
started = time.time()
if len(list(LOCAL.glob("*.npy"))) != n_drive:
    shutil.rmtree(DATA_ROOT, ignore_errors=True)
    shutil.copytree(PROCESSED, LOCAL)

n_local = len(list(LOCAL.glob("*.npy")))
empty = [p.name for p in LOCAL.glob("*.npy") if p.stat().st_size == 0]
print("로컬", n_local, "개 ·", round(time.time() - started), "초 · 빈 파일", len(empty), "개")
assert n_local == n_drive and not empty, "로컬 복사가 불완전하다"
# end
```

## 2. 실험

한 줄이다. 중간에 끊겨도 다시 돌리면 `results/<이름>.json`을 보고 이어간다.

```python
!python scripts/run_experiment.py \
  --name no5 --exclude s05 --seeds 42 1 7 \
  --manifest {MANIFEST} --data-root {DATA_ROOT} \
  --checkpoint-dir /content/drive/MyDrive/hanium-lipreading/checkpoints \
  --wandb-project lipreading
# end
```

자주 쓰는 형태다.

| 목적 | 명령 |
|---|---|
| 기준선 8화자 | `--name base60 --seeds 42 1 7` |
| s05 제외 | `--name no5 --exclude s05 --seeds 42 1 7` |
| 증강 없이 | `--name noaug --seeds 42 1 7 --no-augment` |
| 에폭 120 | `--name e120 --seeds 42 1 7 --epochs 120` |

**결과 파일은 실험마다 따로 둔다.** 같은 이름에 다른 설정을 이어붙이면 스크립트가
설정을 대조해 중단시킨다.

## 3. 요약

학습 없이 표만 본다. 로컬 PC에서도 돈다.

```python
!python scripts/run_experiment.py --name no5 --summary --baseline base60
# end
```

출력에서 볼 것은 셋이다.

```
저장값     검증 화자를 보고 고른 값. 낙관 편향이 있다
마지막     마지막 에폭 값. 발표에 쓸 숫자는 이쪽이다
정점에폭   저장 시점. 한곳에 모이면 에폭을 줄일 여지가 있다
판정선     이 크기를 넘어야 노이즈와 구분된다
```

정점 위치는 요약 끝에 전체 에폭 대비 비율로 나온다. 2026-08-20의 24런에서는
38%에서 100%까지 흩어졌다. 흩어진다는 것은 화자마다 적정 에폭이 다르다는 뜻이고,
그 시점은 검증 화자를 봐야 알 수 있으므로 실사용에서는 못 고른다.

노트북에서 옮긴 기록(`*_notebook.json`)에는 정점 에폭이 없어 `-`로 나온다.

판정선은 조건-화자 한 칸의 시드 표준편차 0.0725에서 계산한다. 화자 수와 시드 수가
늘수록 내려간다.

```
화자 7 · 시드 1    0.077
화자 7 · 시드 3    0.045
화자 8 · 시드 3    0.042
```

## 결과를 저장소에 남기기

`results/*.json`은 커밋한다. 다음 세션에서 기준선으로 바로 쓸 수 있다.

```python
!git -C /content/hanium-lipreading add results
!git -C /content/hanium-lipreading commit -m "Exp: no5 8화자 3시드"
# end
```

## 하지 말 것

**노트북에서 모델·데이터셋·학습 루프를 덮어쓰지 않는다.** 새 방식을 시험하려면
저장소에 옵션으로 넣고 `run_experiment.py`에 인자를 추가한다. 커널에 남은 패치는
다음 셀까지 살아 있어서 원인을 찾기 어렵다.

첫 런 출력의 **학습 파라미터 수**를 확인하는 습관을 들일 것. 기본 설정은 14.31M이다.
