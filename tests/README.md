# 테스트 디렉터리 구조

테스트는 가능한 한 `src/`의 책임 경계를 반영해 배치한다. 서로 다른 계층의 테스트를 한 디렉터리에 섞지 않아, 로컬 개발과 CI에서 필요한 범위만 선택해 실행할 수 있게 한다.

```text
tests/
├── unit/
│   ├── model/      # src/model: 백본, 시간축 모델, 분류 헤드 단위 테스트
│   ├── preprocess/ # src/preprocess: 디코딩, ROI, 정규화 단위 테스트
│   └── backend/    # API, WebSocket, 요청/응답 처리 단위 테스트
├── integration/    # 둘 이상의 계층을 연결하는 통합 테스트
├── conftest.py     # 여러 최상위 계층이 공유하는 fixture가 생길 때만 추가
└── README.md
```

아직 테스트가 없는 디렉터리는 미리 만들지 않는다.

## 배치 원칙

- 하나의 모듈만 검증하면 해당 계층 디렉터리에 둔다.
- WebSocket bytes부터 모델 응답까지 여러 계층을 연결하면 `integration/`에 둔다.
- 특정 계층에서만 쓰는 fixture는 그 계층의 `conftest.py`에 둔다.
- 데이터셋, 네트워크, GPU가 필요 없는 테스트를 기본 CI 대상으로 삼는다.
- GPU 테스트는 `gpu` marker를 붙이고 실제 CUDA/ROCm 환경에서 명시적으로 선택 실행한다.
- 파일명은 `test_<대상>.py`, 테스트명은 검증하려는 동작을 드러내게 작성한다.

기본 CI는 PyTorch를 설치하지 않는다. 따라서 모델 테스트는 PyTorch가 없는 환경에서 명시적으로 skip되며, PyTorch가 설치된 로컬 환경에서 전체 계약을 검증한다.

## 실행 예시

```powershell
# 전체 CPU 테스트
pytest -m "not gpu"

# 모델 계층만
pytest tests/unit/model

# 백엔드 계층만
pytest tests/unit/backend

# 통합 테스트만
pytest tests/integration

# GPU 환경에서만
pytest -m gpu
```
