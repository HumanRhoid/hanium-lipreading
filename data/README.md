# data/ — 데이터 보관 (git 추적 제외)

이 폴더 내용은 `.gitignore`로 **커밋되지 않습니다** (이 README만 예외).

## 왜 커밋하지 않나
- **AI Hub 데이터**: 사용 협약상 재배포 제한 가능 → 공개·사설 레포 업로드 금지
- **얼굴 영상**: 개인정보/생체정보 → 공유 시 동의 필요
- **용량**: 영상 tar는 개당 수십 GB

## 로컬 폴더 구성(권장)
```
data/
├── aihub/        # AI Hub 다운로드 (TL/TS 등) — 일부만
├── recorded/     # 자체 녹화 원본
├── processed/    # 입 ROI 전처리 결과(작음, 학습 입력)
└── manifest.csv  # clip_path,label_id,label_text,speaker_id,condition,split
```
공유는 팀 클라우드 드라이브 사용.

## AI Hub 다운로드 요령
- 전체(~20TB+) 받지 말 것. 거의 전부 `원천데이터`(영상).
- **라벨(TL) 먼저** 받아 포맷·내용 확인 → 필요한 `원천(TS)` **1~3개만**.
- 상세: [../docs/한이음_립리딩_설계.md](../docs/한이음_립리딩_설계.md) L1 참고.
