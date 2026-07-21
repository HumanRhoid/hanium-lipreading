# 👄 구순을 읽자 — 한국어 립리딩 비전 AI

> 발성이 어려운 환자가 **입모양만으로 의사를 전달**하도록 돕는 한국어 립리딩 비전 AI
> 한이음 드림업 팀 프로젝트 · 노현수 · 박제형 · 서지민 · 이예빈

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-DL-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Framework-DEEP%20LEARNING-EE4C2C" alt="Deep Learning">
  <img src="https://img.shields.io/badge/Qwen-LLM%20교정-615CED?logo=qwen&logoColor=white" alt="Qwen LLM">
</p>
<p>
  <img src="https://img.shields.io/badge/OpenCV-Vision-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-FaceMesh-0097A7?logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Preprocess-VISION-5C3EE8" alt="Vision">
</p>
<p>
  <img src="https://img.shields.io/badge/Compute-GPU%20(논의중)-181717" alt="Compute TBD">
</p>
<p>
  <img src="https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/GitHub-Collab-181717?logo=github&logoColor=white" alt="GitHub">
  <img src="https://img.shields.io/badge/Serving-INFRASTRUCTURE-009688" alt="Infrastructure">
</p>

---

## Project Overview

**구순을 읽자**는 질병·수술·신경질환 등으로 **목소리를 내기 어려운 환자**가, 카메라 앞에서 **입모양만으로** 필요한 말을 전달할 수 있도록 돕는 한국어 립리딩 비전 AI입니다.

기존 AAC(보완대체의사소통) 도구가 버튼·그림판 터치에 의존하는 것과 달리, 본 프로젝트는 **입술 움직임 영상**을 입력받아 **한국어 문장으로 인식**하고, 그 결과를 화면 텍스트와 **TTS 음성**으로 출력합니다. **수술 직후 회진 상황**이나 **청각장애인과 비수어인의 대화**처럼, 미리 정해두지 않은 **자유로운 대화(개방형)** 를 다루는 것이 목표입니다.

초심자 4인 팀이 4개월 안에 **동작하는 결과물**을 확보하기 위해 **2트랙**으로 접근합니다. **① 1단계(안전판)**: 고정 문구 폐쇄형 분류로 전처리·학습 파이프라인을 먼저 완성하고, **② 목표**: 사전학습 백본(AV-HuBERT / Auto-AVSR)을 **전이학습**해 **자유 문장(개방형) 인식(seq2seq)** 으로 확장합니다. 인식 결과의 오탈자는 **오픈소스 한국어 LLM(Qwen)** 으로 문맥 후처리 교정합니다.

---

## 1. 프로젝트 한눈에

| 항목 | 내용 |
|---|---|
| **무엇을** | 입모양 영상 → 한국어 **문장** 인식 (개방형 목표) |
| **누구를 위해** | 발성 곤란 환자·회진 상황 / 청각장애인↔비수어인 소통 |
| **어떻게** | **① 폐쇄형 분류(1단계 안전판)** → **② 자유 문장 seq2seq(목표)** |
| **언어 교정** | 인식 결과를 **오픈소스 한국어 LLM(Qwen)** 로 문맥 후처리 |
| **왜 2트랙** | 확실한 결과물(폐쇄형) 확보 + 개방형 도전 → 리스크 분산 |

> 📐 상세 설계: [docs/한이음_립리딩_설계.md](docs/한이음_립리딩_설계.md)
> ⚠️ 추가 고려사항·리스크: [docs/고려사항.md](docs/고려사항.md)

---

## 2. 동작 흐름

```
[입력 영상/웹캠]
   → 전처리   프레임화 → 얼굴/입 검출 → 입 ROI 크롭·정렬
   → 모델     시각 프론트엔드 → 시간 모델(Conformer)
              ├ ① 폐쇄형: 분류 헤드(FC)
              └ ② 개방형: CTC/Attention 디코더 → 글자 시퀀스
   → 후처리   한국어 LLM(Qwen) 문맥 교정 + 신뢰도 판단
   → 출력     화면 텍스트 / TTS 음성 / 자막
```

---

## 3. 기술 스택

| Category | Technology | Details |
|---|---|---|
| **Language** | Python | 3.11 |
| **DL Framework** | PyTorch | 학습 프레임워크 |
| **Vision** | OpenCV · MediaPipe | 프레임 처리 · Face Mesh 입술 랜드마크 검출 |
| **Model** | 3D-Conv + ResNet-18 → Conformer | 시각 프론트엔드 + 시간 모델 |
| **Head** | ① FC 분류 / ② CTC + Attention | ① 폐쇄형 / ② 개방형 seq2seq |
| **Pretrained** | AV-HuBERT / Auto-AVSR | 개방형 백본 전이학습 (자소/음절 토크나이저) |
| **LM (후처리)** | **Qwen** (오픈소스 한국어 LLM) | 인식 텍스트 문맥 교정 (로컬 추론, 학습 X) |
| **Video I/O** | ffmpeg | 디코딩 · 25fps 정규화 |
| **Training** | CrossEntropy(폐쇄형) · CTC+Attention(개방형) · Adam | 학습 손실 |
| **Metric** | 폐쇄형: 정확도/Top-3 · 개방형: **CER/WER** | 평가 지표 |
| **Tracking** | TensorBoard | 손실 · 지표 기록 |
| **Serving** | FastAPI / 소켓 스트리밍 | 모델 → 앱 연결 |
| **Demo UI** | Streamlit / Gradio | 빠른 시연 |
| **TTS** | pyttsx3 / 클로바 TTS | 인식 문장 음성 출력 (AAC 핵심) |
| **Data** | AI Hub(문장형) + 자체 녹화 | 한국어 립리딩 (영상↔텍스트 쌍) |
| **Compute** | 학습 GPU 논의 중 / 보조 Colab | 환경 확정 후 반영 |
| **Collab** | GitHub + 클라우드 드라이브 | 코드는 git · 데이터는 드라이브 |

---

## 4. 패키지 구조

```
hanium-lipreading
 ├── src                  # 소스 코드
 │    ├── backend         # FastAPI 도메인 중심 모듈러 모놀리스
 │    ├── preprocess      # 전처리: 영상 → 프레임 → 입 ROI 크롭·정렬
 │    ├── model           # 모델: 백본 + 시간 모델 + 분류 헤드
 │    ├── train           # 학습·평가 루프 (speaker-independent split)
 │    └── infer           # 추론: 오프라인(영상) / 실시간(웹캠)
 ├── alembic              # PostgreSQL schema migration
 ├── scripts              # 데이터 다운로드·라벨 탐색 등 보조 스크립트
 ├── compose.yaml          # 로컬·테스트 PostgreSQL 16
 ├── configs              # 하이퍼파라미터·경로 설정
 ├── data                 # (git 제외) AI Hub·녹화·전처리 결과 → data/README.md
 ├── docs                 # 설계서·문서
 └── notebooks            # 실험 노트북
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

```bash
# 1) 로컬 설정을 준비합니다. .env는 Git에 커밋하지 않습니다.
cp .env.example .env

# 2) PostgreSQL 16을 시작하고 Python 의존성을 설치합니다.
docker compose up -d --wait postgres
uv sync --locked --dev

# 3) 스키마와 확정된 폐쇄형 문구 데이터를 동기화합니다.
uv run alembic upgrade head
uv run python scripts/sync_closed_phrases.py

# 4) API를 실행합니다.
uv run uvicorn src.backend.main:app --reload --no-access-log --ws-max-size 524288
```

로컬 `.env.example`은 프런트엔드 연동을 즉시 검증할 수 있게 `INFERENCE_BACKEND=fake`를 사용합니다. 가중치와 전처리 asset이 준비되지 않은 운영 환경에서는 `INFERENCE_BACKEND=unavailable`로 두어야 하며, `production` 환경에서 `fake`는 설정 검증 단계에서 거부됩니다. `.env.example`의 `postgres/postgres` 계정은 loopback 로컬 데모 전용이므로 운영에서 재사용하면 안 됩니다.

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

WebSocket은 `ws://127.0.0.1:8000/api/v1/recognition/stream`을 사용합니다. v1은 프레임마다 결과를 보내는 실시간 추론이 아닙니다. `ready` 이후 영상 전체를 전송하고 `stop`을 보내면 서버가 정확히 한 번 추론해 `result(final=true)`와 `stopped`를 보냅니다. 영상은 30~250프레임, 누적 64 MiB까지 허용하며 30프레임을 초과하면 전체 구간에서 모델 입력 30장을 균등 추출합니다. 정확한 시작 명령, JPEG 해상도, timeout과 오류·재연결 정책은 [API 명세](docs/api-spec.md)를 따릅니다.

### 프런트엔드 통합 시 주의사항

별도 `lipread-connect` 프런트엔드는 현재 `1280x720` 카메라 미리보기 중심이며 WebSocket 송수신은 아직 연결되지 않았습니다. 전송 frame은 정확히 `640x360` JPEG로 축소·인코딩하고, `ready` 이후 raw binary로 약 `25fps` 보내야 합니다. 최초 명령 `{"type":"start"}`에는 서버가 `CLOSED`를 기본 적용합니다. Data URL·base64 text를 보내지 않고, `WebSocket.bufferedAmount`가 상한을 넘으면 새 frame을 로컬 queue에 쌓지 말고 건너뜁니다. 이는 브라우저 transport backpressure이고, 서버는 별도로 연결당 최대 250프레임·64 MiB bounded buffer를 강제해 메모리가 무제한 증가하지 않게 합니다. 사용자 종료 시 `{"type":"stop"}`을 보낸 뒤 최종 결과와 `stopped`를 기다립니다. 자세한 순서는 [프런트엔드 통합 기준](docs/api-spec.md#8-lipread-connect-프런트엔드-통합-기준)을 따릅니다.

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

> 학습 GPU 환경은 아직 논의 중이다. 한 명이 학습을 돌리는 동안,
> 나머지는 **GPU 없이 가능한 작업**을 병행해 일정 정체를 막는다.

| 학습 중 병행 작업 | 담당 워크스트림 | GPU 필요? |
|---|---|---|
| 자체 녹화 데이터 추가 수집·라벨링 | R1 데이터 | ❌ |
| 입 ROI 전처리 개선 · 증강 실험 준비 | R2 전처리 | ❌ (CPU) |
| 추론·데모 UI(Streamlit/TTS) 개발 | R4 시스템 | ❌ |
| 평가 스크립트 · confusion matrix 분석 | R3 모델 | ❌ |
| 중간평가 서류 · 설계서 갱신 | 전원 | ❌ |

**운영 규칙**
- 🗓️ 학습 일정·GPU 점유를 사전에 공유 (시간 겹침 방지)
- 💾 체크포인트(`.pt`)·영상은 git 밖(드라이브/Releases)으로 — 코드·설정·문서만 커밋
- 🔁 재현성: 시드 고정 · `uv.lock` 고정 · `manifest.csv`로 데이터 버전 관리

---

## 9. 데이터 정책 ⚠️ (중요)

- **AI Hub 데이터·영상·가중치는 절대 커밋 금지** (라이선스 + 얼굴/개인정보) — `.gitignore`로 제외됨
- 데이터는 팀 공유 드라이브로 관리 → [data/README.md](data/README.md)
