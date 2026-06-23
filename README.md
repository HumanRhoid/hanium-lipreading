# 한이음 립리딩 (구순을 읽자) — 비전 AI

발성이 어려운 환자가 **입모양만으로 의사를 전달**하도록 돕는 한국어 립리딩 비전 AI.
한이음 드림업 팀 프로젝트.

> 상세 설계: [docs/한이음_립리딩_설계.md](docs/한이음_립리딩_설계.md)
> 추가 고려사항·리스크: [docs/고려사항.md](docs/고려사항.md)

## 한눈에
- **도메인**: 환자 전용 AAC, **한국어 전용**, **폐쇄형(고정 문구 N개) 분류**
- **접근**: 사전학습 백본 전이학습(한국어→한국어 우선) + 자체 녹화 데이터
- **컴퓨팅**: 로컬 RX 7900 XTX(Windows ROCm) / Colab / 필요시 클라우드

## 디렉터리 구조
```
src/preprocess/  영상→프레임→얼굴/입 검출→입 ROI 크롭
src/model/       백본 + 시간모델 + 분류 헤드
src/train/       학습·평가 루프
src/infer/       추론(오프라인/실시간)
scripts/         데이터 다운로드·라벨 탐색 등 보조 스크립트
configs/         하이퍼파라미터·경로 설정
data/            (git 제외) 데이터 보관 — data/README.md 참고
docs/            설계서·문서
notebooks/       실험 노트북
```

## 개발 환경 세팅
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
# PyTorch는 별도 설치 (ROCm/CUDA 빌드) — requirements.txt 주석 참고
```

## ⚠️ 데이터 정책 (중요)
- **AI Hub 데이터·영상·가중치는 절대 커밋 금지** (라이선스 + 얼굴/개인정보). `.gitignore`로 제외됨.
- 데이터는 팀 공유 드라이브/클라우드로 관리. `data/README.md` 참고.
- 자체 녹화 영상은 **촬영·활용 동의** 후 사용.

## 커밋 컨벤션
`Feat / Fix / Design / Refactor / Chore / Docs` + 이모지 (팀 합의 형식)

## 팀
노현수 · 박제형 · 서지민 · 이예빈
