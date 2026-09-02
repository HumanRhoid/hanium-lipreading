# 👄 구순을 읽자 — 한국어 립리딩 비전 AI

> 발성이 어려운 환자가 **입모양만으로 의사를 전달**하도록 돕는 한국어 립리딩 비전 AI
> 한이음 드림업 팀 프로젝트 · 노현수 · 박제형 · 서지민 · 이예빈

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-DL-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Framework-DEEP%20LEARNING-EE4C2C" alt="Deep Learning">
</p>
<p>
  <img src="https://img.shields.io/badge/OpenCV-Vision-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-FaceMesh-0097A7?logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Preprocess-VISION-5C3EE8" alt="Vision">
</p>
<p>
  <img src="https://img.shields.io/badge/Compute-Google%20Colab%20GPU-F9AB00?logo=googlecolab&logoColor=white" alt="Colab">
</p>
<p>
  <img src="https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-Web%20App-61DAFB?logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/GitHub-Collab-181717?logo=github&logoColor=white" alt="GitHub">
  <img src="https://img.shields.io/badge/Serving-INFRASTRUCTURE-009688" alt="Infrastructure">
</p>

---

## Project Overview

**구순을 읽자**는 질병·수술·신경질환 등으로 **목소리를 내기 어려운 환자**가, 카메라 앞에서 **입모양만으로** 필요한 말을 전달할 수 있도록 돕는 한국어 립리딩 비전 AI입니다.

기존 AAC(보완대체의사소통) 도구가 버튼·그림판 터치에 의존하는 것과 달리, 본 프로젝트는 **입술 움직임 영상**을 입력받아 한국어 문구로 인식하고, 그 결과를 화면 텍스트·확신도와 **TTS 음성**으로 출력합니다.

**1단계는 완성되었습니다(v1.0.0)**: 인공기도 환자 요구 조사에서 도출한 **필수 의료 문구 15개**를 폐쇄형으로 인식하며, 브라우저 녹화 → 업로드 → 비동기 추론 → 결과 표시까지 전 구간이 실동작합니다. **2단계(로드맵)** 는 동일 인코더 위에서 사전학습 백본(AV-HuBERT 등) 전이학습으로 **자유 문장(seq2seq)** 확장입니다.

---

## 1. 프로젝트 한눈에

| 항목 | 내용 |
|---|---|
| **무엇을** | 입모양 영상 → 한국어 **필수 의료 문구 15개** 인식 (v1.0.0 완성) |
| **누구를 위해** | 발성 곤란 환자·회진 상황 / 청각장애인↔비수어인 소통 |
| **어떻게** | 3D-Conv+ResNet-18 → BiGRU → 분류, **3시드 앙상블** |
| **성능** | 화자 분리 교차검증(8화자×3시드=24런) 평균 **0.621** (우연 0.067) |
| **다음** | 동일 인코더 위 자유 문장 seq2seq 확장 · 의료진 대시보드 |

> 📐 상세 설계: [docs/한이음_립리딩_설계.md](docs/한이음_립리딩_설계.md)
> ⚠️ 추가 고려사항·리스크: [docs/고려사항.md](docs/고려사항.md)

---

## 2. 동작 흐름

```
[브라우저 녹화 영상]
   → 업로드    POST /recognition/videos (202 + job_id)
   → 저장/큐   MinIO(영상) · Redis Stream(작업)
   → Worker    발화 구간 트리밍 → 닮음 변환 정렬 크롭(96×192)
               → 60프레임 균등 추출·CLAHE → 3D-Conv+ResNet-18 → BiGRU
               → 3시드 앙상블 → 문구·확신도
   → 결과      PostgreSQL 저장 → 프런트 폴링 → 텍스트·확신도·TTS
```
자유 문장 seq2seq(CTC/Attention)는 동일 인코더 위 확장 로드맵입니다.

---

## 3. 기술 스택

| Category | Technology | Details |
|---|---|---|
| **Language** | Python | 3.11 |
| **DL Framework** | PyTorch | 학습 프레임워크 |
| **Vision** | OpenCV · MediaPipe | 프레임 처리 · Face Mesh 입술 랜드마크 검출 |
| **Model** | 3D-Conv + ResNet-18 → BiGRU(2층) | 시각 인코더 + 시간 모델 (1,431만 파라미터) |
| **Head** | ① 15문구 분류 (완성) / ② CTC+Attention (로드맵) | 3시드 앙상블 확률 평균 |
| **Pretrained** | AV-HuBERT / Auto-AVSR | 개방형 백본 전이학습 (자소/음절 토크나이저) |
| **Video I/O** | OpenCV | mp4/webm 디코딩 · 발화 구간 트리밍(입 벌어짐) |
| **Training** | CrossEntropy · AdamW · EMA 0.998 · 시드 3벌 | Colab GPU 학습 |
| **Metric** | 화자 분리 교차검증 정확도 · 재현 잡음 판정선(±0.042) | 개방형은 CER/WER 예정 |
| **Tracking** | TensorBoard | 손실 · 지표 기록 |
| **Serving** | FastAPI + MinIO + Redis Stream + 전용 Worker | 비동기 업로드·큐·폴링 |
| **Web App** | React / TypeScript (Vite) | lipread-connect 저장소 |
| **TTS** | Web Speech API | 브라우저에서 인식 문장 낭독 (AAC 핵심) |
| **Data** | 자체 녹화 8화자 1,238클립 (15문구 균형) | 화자 분리 원칙 관리 |
| **Compute** | Google Colab GPU | 시드 3벌 재현 학습 |
| **Collab** | GitHub + 클라우드 드라이브 | 코드는 git · 데이터는 드라이브 |

---

## 4. 패키지 구조

```
hanium-lipreading
 ├── src
 │    ├── backend         # FastAPI 모듈러 모놀리스 + 추론 Worker (worker_main)
 │    │    └── recognition/adapters   # DB·MinIO·Redis·LocalSyncPredictor
 │    └── ml              # 전처리(normalize·lip_crop) · 모델 · 학습 (학습·추론 공유)
 ├── checkpoints          # 배포 모델 release192_seed{42,1,7}.pt (3시드 앙상블)
 ├── alembic              # PostgreSQL schema migration
 ├── scripts              # dev_up/dev_down(원클릭 기동·종료) · 운영 스크립트
 ├── compose.yaml         # PostgreSQL · MinIO · Redis (Docker Compose)
 ├── configs              # 하이퍼파라미터·경로 설정
 ├── data                 # (git 제외) 녹화·전처리 결과 → data/README.md
 ├── docs                 # 설계서·실험 기록(docs/experiments) · 릴리스 노트
 └── notebooks            # Colab 학습·릴리스 노트북
```

---

## 5. 개발 환경 세팅

이 프로젝트의 기본 Python 개발 환경 도구는 **uv**입니다.
`pyproject.toml`과 `uv.lock`을 기준으로 팀원이 같은 의존성 해석 결과를 재현합니다.

```bash
# 1) uv 설치 확인
uv --version

# 2) Python 3.11 가상환경 생성 + 기본 의존성 설치
uv sync

# 3) 가상환경에서 명령 실행
uv run python --version
uv run python -c "import cv2, mediapipe, numpy; print('deps ok')"

# 4) PyTorch는 GPU 환경별 ROCm/CUDA 빌드를 별도 설치
# ROCm(Windows): https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
# CUDA: https://pytorch.org/get-started/locally/
uv run python -c "import torch; print(torch.cuda.is_available())"   # GPU 사용 가능 여부 확인
```

> 의존성의 단일 기준은 `pyproject.toml` / `uv.lock`입니다.
> Colab처럼 현재 Python 환경에 직접 설치해야 하는 경우에도 `requirements.txt`를 커밋하지 않고,
> 필요한 패키지는 `pyproject.toml`을 기준으로 설치하거나 임시 export해서 사용합니다.

---

## 6. 백엔드 API 개발

백엔드는 FastAPI 단일 프로세스와 PostgreSQL을 사용하는 도메인 중심 모듈러 모놀리스입니다. 현재 계약과 스키마의 기준은 다음 문서입니다.

- [백엔드 Source of Truth](docs/backend-sot.md)
- [HTTP·WebSocket API 명세](docs/api-spec.md)
- [PostgreSQL ERD 명세](docs/erd-spec.md)

### 로컬 실행

Docker와 Docker Compose, Python 3.11, uv가 필요합니다.

**원클릭(Windows)**: 아래 수동 절차를 스크립트 하나로 대신할 수 있습니다.

```powershell
powershell -File scripts\dev_up.ps1              # 인프라→마이그레이션→API·Worker 창
powershell -File scripts\dev_up.ps1 -Front <lipread-connect 경로>   # 프론트까지
powershell -File scripts\dev_down.ps1            # 전체 종료 (데이터 보존)
```

실모델 추론은 `.env`에 `INFERENCE_BACKEND=local`과 `checkpoints/release192_seed*.pt`
3개가 필요합니다. DB 주소는 `localhost` 대신 `127.0.0.1`을 권장합니다(Windows에서
IPv6 폴백으로 첫 연결이 2초 지연되어 health 검사가 실패할 수 있습니다).

```bash
# 1) 로컬 설정을 준비합니다. .env는 Git에 커밋하지 않습니다.
cp .env.example .env

# 2) 인프라(PostgreSQL·MinIO·Redis)를 시작하고 Python 의존성을 설치합니다.
docker compose up -d --wait postgres minio minio-init redis
uv sync --locked --dev

# 3) 스키마와 확정된 폐쇄형 문구 데이터를 동기화합니다.
uv run alembic upgrade head
uv run python scripts/sync_closed_phrases.py

# 4) API를 실행합니다.
uv run uvicorn src.backend.main:app --reload --no-access-log --ws-max-size 524288

# 5) 별도 터미널에서 업로드 영상 추론 Worker를 실행합니다.
uv run python -m src.backend.recognition.worker_main
```

프런트엔드의 비동기 영상 인식은 `POST /api/v1/recognition/videos`가 반환한
`job_id`를 `GET /api/v1/inference-jobs/{job_id}`로 조회합니다. 완료 상태는
`SUCCEEDED`이며 응답의 `result`에 문장, 문구 코드, 신뢰도, 모델 버전과 생성
시각이 포함됩니다. API와 Worker를 모두 실행해야 Job이 `QUEUED`에서 진행됩니다.

로컬 `.env.example`은 프런트엔드 연동을 즉시 검증할 수 있게 `INFERENCE_BACKEND=fake`를 사용합니다. 가중치와 전처리 asset이 준비되지 않은 운영 환경에서는 `INFERENCE_BACKEND=unavailable`로 두어야 하며, `production` 환경에서 `fake`는 설정 검증 단계에서 거부됩니다. `.env.example`의 `postgres/postgres` 계정은 loopback 로컬 데모 전용이므로 운영에서 재사용하면 안 됩니다.

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

WebSocket은 `ws://127.0.0.1:8000/api/v1/recognition/stream`을 사용합니다. v1은 프레임마다 결과를 보내는 실시간 추론이 아닙니다. `ready` 이후 영상 전체를 전송하고 `stop`을 보내면 서버가 정확히 한 번 추론해 `result(final=true)`와 `stopped`를 보냅니다. 영상은 60~250프레임, 누적 64 MiB까지 허용하며 60프레임을 초과하면 전체 구간에서 모델 입력 60장을 균등 추출합니다. 정확한 시작 명령, JPEG 해상도, timeout과 오류·재연결 정책은 [API 명세](docs/api-spec.md)를 따릅니다.

### 프런트엔드 통합 시 주의사항

별도 `lipread-connect` 프런트엔드는 **영상 업로드 + 폴링 방식으로 통합 완료**되었습니다(MediaRecorder 녹화 → `POST /recognition/videos` → `job_id` 폴링 → 결과·TTS). 아래 WebSocket 경로는 v1 유산으로, 실시간 품질 검사 채널로 개조가 예정되어 있습니다. 전송 frame은 정확히 `640x360` JPEG로 축소·인코딩하고, `ready` 이후 raw binary로 약 `25fps` 보내야 합니다. 최초 명령 `{"type":"start"}`에는 서버가 `CLOSED`를 기본 적용합니다. Data URL·base64 text를 보내지 않고, `WebSocket.bufferedAmount`가 상한을 넘으면 새 frame을 로컬 queue에 쌓지 말고 건너뜁니다. 이는 브라우저 transport backpressure이고, 서버는 별도로 연결당 최대 250프레임·64 MiB bounded buffer를 강제해 메모리가 무제한 증가하지 않게 합니다. 사용자 종료 시 `{"type":"stop"}`을 보낸 뒤 최종 결과와 `stopped`를 기다립니다. 자세한 순서는 [프런트엔드 통합 기준](docs/api-spec.md#8-lipread-connect-프런트엔드-통합-기준)을 따릅니다.

### 운영 배포

- FastAPI를 인터넷에 직접 노출하지 않고 TLS reverse proxy 뒤의 사설 upstream으로 둡니다. 브라우저 endpoint는 `wss`만 허용하고 proxy에서 WebSocket Upgrade를 전달합니다.
- 운영 DB에는 기본 계정·기본 비밀번호를 쓰지 않습니다. 최소 권한 전용 계정과 secret 저장소에서 주입한 고유 비밀번호를 사용합니다.
- 운영 log 설정에서 reverse proxy access log와 Uvicorn의 access·WebSocket handshake IP log를 끄고 frame body·WebSocket message·인식 문장은 기록하지 않습니다. `--no-access-log`만으로 일부 `uvicorn.error` handshake log가 남을 수 있습니다. 보안상 필요한 접속 로그만 IP를 마스킹 또는 가명화하여 접근 제한 저장소에 최대 7일 보관한 뒤 자동 삭제합니다.
- 인식 세션과 최종 발화는 현재 자동 만료되지 않습니다. 실제 사용자 데이터를 받기 전에 보존기간과 정기 purge를 정하고, 그 전에는 제한된 데모 데이터만 저장합니다.

### Migration과 관리 명령

```bash
# 현재 migration과 적용 상태 확인
uv run alembic history
uv run alembic current

# 작성된 migration 적용
uv run alembic upgrade head

# ORM 변경 후 migration 초안 생성
uv run alembic revision --autogenerate -m "describe schema change"
```

자동 생성된 migration은 반드시 직접 검토하고, ERD 명세·제약조건 테스트와 같은 PR에서 갱신합니다. 폐쇄형 문구 데이터는 migration에 포함하지 않고 다음 명령으로 따로 관리합니다. 이 명령은 확정 문구를 멱등적으로 upsert하고 목록에 없는 기존 문구를 삭제해 DB를 정확히 동기화합니다. 삭제된 문구와 연결된 과거 발화는 유지되며 `phrase_id`만 `NULL`이 됩니다.

```bash
uv run python scripts/sync_closed_phrases.py
```

인식 세션과 연결된 발화를 삭제하는 명령은 정확히 하나의 조건과 `--confirm`을 모두 요구합니다. `--before`에는 timezone이 포함된 ISO 8601 시각을 사용합니다.

```bash
uv run python scripts/purge_recognition_data.py --help
uv run python scripts/purge_recognition_data.py --session-id 123 --confirm
uv run python scripts/purge_recognition_data.py \
  --before 2026-07-01T00:00:00+09:00 --confirm
```

프로세스 강제 종료 후 최대 세션 시간과 종료 여유를 넘긴 `ended_at IS NULL` 세션이 남았으면, 실행 중인 정상 세션보다 충분히 이전인 기준 시각으로 종료 처리합니다. 이 명령은 세션이나 발화를 삭제하지 않습니다.

```bash
uv run python scripts/reconcile_abandoned_sessions.py \
  --before 2026-07-21T09:00:00+09:00 --confirm
```

### 테스트

일반 개발은 외부 DB가 필요 없는 테스트를 먼저 실행합니다.

```bash
uv run pytest -m "not integration"
uv run ruff check .
uv run ruff format --check .
```

PostgreSQL 통합 테스트는 테이블을 생성·삭제하므로 개발 DB가 아닌 **빈 테스트 전용 DB**를 지정해야 합니다. CI는 일회성 PostgreSQL 16 서비스에서 migration과 통합 테스트를 실행합니다.

```bash
# 개발 DB와 분리된 임시 PostgreSQL 16을 5433 포트에 시작
docker compose --profile test up -d --wait postgres-test

ALLOW_DESTRUCTIVE_DB_TESTS=1 \
  TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/hanium_lipreading_test \
  uv run pytest -m integration

docker compose stop postgres-test
```

---

## 7. 브랜치 전략

| 브랜치 | 용도 | 예시 |
|---|---|---|
| `main` | 배포용 (항상 동작하는 안정 버전) | `main` |
| `develop` | 개발 통합 (PR 도착지) | `develop` |
| `feature/` | 새 기능 개발 | `feature/preprocess-roi`, `feature/train-loop` |
| `fix/` | 버그 수정 (개발 중) | `fix/fps-mismatch`, `fix/label-parse` |

> 흐름: `feature/*` → PR → `develop` → 안정화 후 → `main`
> 커밋 컨벤션: `Feat / Fix / Design / Refactor / Chore / Docs`

---

## 8. 학습과 병행할 작업

> 학습은 Google Colab GPU에서 수행한다. 한 명이 학습을 돌리는 동안,
> 나머지는 **GPU 없이 가능한 작업**을 병행해 일정 정체를 막는다.

| 학습 중 병행 작업 | 담당 워크스트림 | GPU 필요? |
|---|---|---|
| 자체 녹화 데이터 추가 수집·라벨링 | R1 데이터 | ❌ |
| 입 ROI 전처리 개선 · 증강 실험 준비 | R2 전처리 | ❌ (CPU) |
| 웹 앱·대시보드 개발 | R4 시스템 | ❌ |
| 평가 스크립트 · confusion matrix 분석 | R3 모델 | ❌ |
| 중간평가 서류 · 설계서 갱신 | 전원 | ❌ |

**운영 규칙**
- 🗓️ 학습 일정·GPU 점유를 사전에 공유 (시간 겹침 방지)
- 💾 배포 체크포인트 3개는 `checkpoints/`에 커밋, **영상·원본 데이터는 git 밖**(드라이브)
- 🔁 재현성: 시드 고정 · `uv.lock` 고정 · `manifest.csv`로 데이터 버전 관리

---

## 9. 데이터 정책 ⚠️ (중요)

- **얼굴 영상·원본 데이터는 절대 커밋 금지** (개인정보) — `.gitignore`로 제외됨. 서비스 영상 재활용은 사용자 동의 체계로 구분
- 데이터는 팀 공유 드라이브로 관리 → [data/README.md](data/README.md)
