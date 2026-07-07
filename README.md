# 👄 구순을 읽자 — 한국어 립리딩 비전 AI

> 발성이 어려운 환자가 **입모양만으로 의사를 전달**하도록 돕는 한국어 립리딩 비전 AI
> 한이음 드림업 팀 프로젝트 · 노현수 · 박제형 · 서지민 · 이예빈

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-ROCm-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch ROCm">
  <img src="https://img.shields.io/badge/Framework-DEEP%20LEARNING-EE4C2C" alt="Deep Learning">
</p>
<p>
  <img src="https://img.shields.io/badge/OpenCV-Vision-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-FaceMesh-0097A7?logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Preprocess-VISION-5C3EE8" alt="Vision">
</p>
<p>
  <img src="https://img.shields.io/badge/AMD-RX%207900%20XTX-ED1C24?logo=amd&logoColor=white" alt="AMD RX 7900 XTX">
  <img src="https://img.shields.io/badge/ROCm-Windows-ED1C24" alt="ROCm Windows">
  <img src="https://img.shields.io/badge/Compute-GPU-181717" alt="GPU">
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

기존 AAC(보완대체의사소통) 도구가 버튼·그림판 터치에 의존하는 것과 달리, 본 프로젝트는 **입술 움직임 영상**을 입력받아 "아파요 / 물 주세요 / 화장실" 같은 **핵심 문구 N개를 분류**하고, 그 결과를 화면 텍스트와 **TTS 음성**으로 출력합니다. 손을 쓰기 어렵거나 글을 입력하기 힘든 환자도 의사를 전할 수 있게 하는 것이 목표입니다.

초심자 4인 팀이 한 학기 안에 **동작하는 결과물**을 만들기 위해, 자유 문장 받아쓰기 대신 **폐쇄형(고정 문구) 분류**로 범위를 좁히고, 외부 데이터 승인을 기다리지 않도록 **자체 녹화 데이터**로 베이스라인을 먼저 확보합니다. 코어(전처리→모델→후처리)는 도메인이 바뀌어도 재사용 가능하게 설계해, 이후 **실시간 인식·자유 문장**으로 확장할 수 있습니다.

---

## 1. 프로젝트 한눈에

| 항목 | 내용 |
|---|---|
| **무엇을** | 입모양 영상 → 한국어 문구 인식 |
| **누구를 위해** | 발성 곤란 환자 (AAC, 보완대체의사소통) |
| **어떻게** | 고정 문구 N개를 분류하는 **폐쇄형 인식** (자유 문장 받아쓰기 ❌) |
| **예시 문구** | "아파요 / 물 주세요 / 화장실 / 추워요" 등 20~30개 |
| **왜 폐쇄형** | 초심자 팀이 학기 내 완성 가능 + 자체 녹화로 데이터 확보 + 데모 가치 명확 |

> 📐 상세 설계: [docs/한이음_립리딩_설계.md](docs/한이음_립리딩_설계.md)
> ⚠️ 추가 고려사항·리스크: [docs/고려사항.md](docs/고려사항.md)

---

## 2. 동작 흐름

```
[입력 영상/웹캠]
   → 전처리   프레임화 → 얼굴/입 검출 → 입 ROI 크롭·정렬
   → 모델     시각 프론트엔드 → 시간 모델 → 분류 헤드
   → 후처리   문구 매핑 + 신뢰도 판단
   → 출력     화면 텍스트 / TTS 음성 / 자막
```

---

## 3. 기술 스택

| Category | Technology | Details |
|---|---|---|
| **Language** | Python | 3.11 |
| **DL Framework** | PyTorch | ROCm 빌드 (RX 7900 XTX, Windows 네이티브) |
| **Vision** | OpenCV · MediaPipe | 프레임 처리 · Face Mesh 입술 랜드마크 검출 |
| **Model** | 3D-Conv + ResNet-18 + BiGRU | 시각 프론트엔드 + 시간 모델 + FC 헤드 |
| **Video I/O** | ffmpeg | 디코딩 · 25fps 정규화 |
| **Training** | CrossEntropy + Adam | 폐쇄형 N-클래스 분류 |
| **Tracking** | TensorBoard | 손실 · 정확도 · confusion matrix |
| **Serving** | FastAPI / 소켓 스트리밍 | 모델 → 앱 연결 |
| **Demo UI** | Streamlit / Gradio | 빠른 시연 |
| **TTS** | pyttsx3 / 클로바 TTS | 인식 문구 음성 출력 (AAC 핵심) |
| **Data** | AI Hub + 자체 녹화 | 한국어 립리딩 |
| **Compute** | AMD RX 7900 XTX (24GB) | 로컬 학습 / 팀원은 Colab |
| **Collab** | GitHub + 클라우드 드라이브 | 코드는 git · 데이터는 드라이브 |

---

## 4. 패키지 구조

```
hanium-lipreading
 ├── src                  # 소스 코드
 │    ├── preprocess      # 전처리: 영상 → 프레임 → 입 ROI 크롭·정렬
 │    ├── model           # 모델: 백본 + 시간 모델 + 분류 헤드
 │    ├── train           # 학습·평가 루프 (speaker-independent split)
 │    └── infer           # 추론: 오프라인(영상) / 실시간(웹캠)
 ├── scripts              # 데이터 다운로드·라벨 탐색 등 보조 스크립트
 ├── configs              # 하이퍼파라미터·경로 설정
 ├── data                 # (git 제외) AI Hub·녹화·전처리 결과 → data/README.md
 ├── docs                 # 설계서·문서
 └── notebooks            # 실험 노트북
```

---

## 5. 개발 환경 세팅

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# PyTorch는 별도 설치 (ROCm / CUDA 빌드) — requirements.txt 주석 참고
python -c "import torch; print(torch.cuda.is_available())"   # True 확인
```

---

## 6. 브랜치 전략

| 브랜치 | 용도 | 예시 |
|---|---|---|
| `main` | 배포용 (항상 동작하는 안정 버전) | `main` |
| `develop` | 개발 통합 (PR 도착지) | `develop` |
| `feature/` | 새 기능 개발 | `feature/preprocess-roi`, `feature/train-loop` |
| `fix/` | 버그 수정 (개발 중) | `fix/fps-mismatch`, `fix/label-parse` |

> 흐름: `feature/*` → PR → `develop` → 안정화 후 → `main`
> 커밋 컨벤션: `Feat / Fix / Design / Refactor / Chore / Docs`

---

## 7. 학습과 병행할 작업 (단일 GPU 운영 전략)

> 학습 노드는 **RX 7900 XTX 1대**뿐. 한 명이 학습을 돌리는 동안 GPU는 점유되므로,
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
- 🔁 재현성: 시드 고정 · `requirements` 핀 · `manifest.csv`로 데이터 버전 관리

---

## 8. 데이터 정책 ⚠️ (중요)

- **AI Hub 데이터·영상·가중치는 절대 커밋 금지** (라이선스 + 얼굴/개인정보) — `.gitignore`로 제외됨
- 데이터는 팀 공유 드라이브로 관리 → [data/README.md](data/README.md)
