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
  <img src="https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?logo=googlecolab&logoColor=white" alt="Google Colab">
  <img src="https://img.shields.io/badge/Compute-CLOUD-181717" alt="Cloud">
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
   → 전처리   프레임화 → 얼굴/입 검출 → 입 ROI 크롭 → 흑백·밝기(CLAHE) 보정
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
| **DL Framework** | PyTorch | Google Colab 기본 제공 (클라우드 GPU) |
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
| **Compute** | Google Colab | 개발·학습 전부 Colab에서 진행 |
| **Collab** | GitHub + 클라우드 드라이브 | 코드는 git · 데이터는 드라이브 |

---

## 4. 패키지 구조

```
hanium-lipreading
 ├── src                  # 소스 코드
 │    ├── preprocess      # 전처리: 영상 → 프레임 → 입 ROI 크롭 → 흑백·밝기 보정
 │    ├── model           # 모델: 백본 + 시간 모델 + 분류 헤드
 │    ├── train           # 학습·평가 루프 (speaker-independent split)
 │    └── infer           # 추론: 오프라인(영상) / 실시간(웹캠)
 ├── scripts              # 데이터 다운로드·라벨 탐색 등 보조 스크립트
 ├── configs              # 하이퍼파라미터·경로 설정
 ├── models               # (git 제외) MediaPipe 등 사전학습 모델 가중치
 ├── data                 # (git 제외) AI Hub·녹화·전처리 결과 → data/README.md
 ├── docs                 # 설계서·문서
 └── notebooks            # 실험 노트북
```

---

## 5. 개발 환경 세팅

개발·학습은 **Google Colab**에서 진행합니다.

```python
!pip install -r requirements.txt
import torch; print(torch.cuda.is_available())   # True 확인 (런타임 유형: GPU)
```

로컬에서는 GPU가 필요 없는 전처리 코드(`src/preprocess`)만 실행합니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 6. 브랜치 전략

| 브랜치 | 용도 | 예시 |
|---|---|---|
| `main` | 배포용 (항상 동작하는 안정 버전) | `main` |
| `develop` | 개발 통합 (PR 도착지) | `develop` |
| `feature/` | 새 기능 개발 | `feature/preprocess-roi`, `feature/train-loop` |
| `fix/` | 버그 수정 (개발 중) | `fix/fps-mismatch`, `fix/label-parse` |
| `docs/` | 문서 작성·정리 | `docs/pr-template-guide` |
| `chore/` | 설정·저장소 관리 | `chore/update-gitignore` |

> 흐름: `feature/*` / `fix/*` / `docs/*` → PR → `develop` → 안정화 후 → `main`
> 커밋 컨벤션: `feat / fix / docs / refactor / design / chore / test`
> PR 작성 기준: [docs/pr-guide.md](docs/pr-guide.md)

---

## 7. 학습과 병행할 작업 (Colab 운영 전략)

> 학습은 **Google Colab**(클라우드 GPU)에서 돌린다. Colab은 세션 시간·GPU 할당 제한이 있어
> 학습이 항상 대기 없이 되진 않으므로, 학습을 돌리는 동안 **GPU 없이 가능한 작업**을 병행해 일정 정체를 막는다.

| 학습 중 병행 작업 | 담당 워크스트림 | GPU 필요? |
|---|---|---|
| 자체 녹화 데이터 추가 수집·라벨링 | R1 데이터 | ❌ |
| 입 ROI 전처리 개선 · 증강 실험 준비 | R2 전처리 | ❌ (CPU) |
| 추론·데모 UI(Streamlit/TTS) 개발 | R4 시스템 | ❌ |
| 평가 스크립트 · confusion matrix 분석 | R3 모델 | ❌ |
| 중간평가 서류 · 설계서 갱신 | 전원 | ❌ |

**운영 규칙**
- ☁️ Colab 세션은 끊길 수 있음 → 체크포인트를 **구글 드라이브에 주기적 저장**(마운트)
- 💾 체크포인트(`.pt`)·영상은 git 밖(드라이브/Releases)으로 — 코드·설정·문서만 커밋
- 🔁 재현성: 시드 고정 · `requirements` 핀 · `manifest.csv`로 데이터 버전 관리

---

## 8. 데이터 정책 ⚠️ (중요)

- **AI Hub 데이터·영상·가중치는 절대 커밋 금지** (라이선스 + 얼굴/개인정보) — `.gitignore`로 제외됨
- 데이터는 팀 공유 드라이브로 관리 → [data/README.md](data/README.md)
