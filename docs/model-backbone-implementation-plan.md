# 립리딩 시각 백본 구현 및 검증 계획

- 작업 브랜치: `feature/model-backbone`
- 기준 브랜치: `origin/develop`
- 작성일: 2026-07-14
- 핵심 구현 상태: 완료

백본의 규범적 계약은 [립리딩 시각 백본 SoT](model-backbone-sot.md)를 따른다. 이 문서는 구현 순서, 검증 방법, 현재 완료 상태와 후속 운영 작업을 관리한다.

## 1. 작업 목표

전처리된 입술 프레임을 받아 시간 길이를 보존하는 512차원 특징 시퀀스를 출력하는 `3D-Conv + ResNet-18` 백본을 구현한다. WebSocket, 전처리, 시간축 모델과 분류 헤드는 이번 구현 범위에서 제외한다.

## 2. 산출물과 상태

| 산출물 | 경로 | 상태 |
|---|---|---|
| 백본 SoT | `docs/model-backbone-sot.md` | 완료 |
| 백본 구현 | `src/model/backbone.py` | 완료 |
| 공개 import | `src/model/__init__.py` | 완료 |
| 백본 계약 테스트 | `tests/unit/model/test_backbone.py` | 완료 |
| 테스트 계층 규칙 | `tests/README.md` | 완료 |
| PyTorch 미설치 CI의 명시적 skip | `tests/unit/model/test_backbone.py` | 완료 |
| 백엔드 경량 CI | `.github/workflows/backend-ci.yml` | develop 반영 |
| ROCm GPU smoke test | GPU 테스트 모듈 | 후속 작업 |

## 3. TDD 구현 순서

1. **계약 정의 — 완료**
   - 입력 `[B,3,T,80,112]`과 출력 `[B,T,512]`를 정의한다.
   - 가변 `T`, 입력 검증과 gradient 전달을 완료 조건으로 정한다.
2. **Red — 완료**
   - 구현 파일 없이 계약 테스트를 먼저 작성한다.
   - `LipReadingBackbone` import 실패를 확인한다.
3. **Green — 완료**
   - 3D stem과 ResNet-18 BasicBlock trunk를 구현한다.
   - 계약 테스트를 통과시키는 최소 공개 API를 제공한다.
4. **Refactor — 완료**
   - 모델과 테스트 계층을 분리한다.
   - 주석과 docstring을 한글로 통일한다.
   - SoT와 작업 계획을 분리한다.

## 4. CPU 검증 범위

기본 검증은 데이터셋, 네트워크, 사전학습 가중치와 GPU 없이 실행한다.

- 3D stem의 kernel, stride, padding
- ResNet-18의 `[2,2,2,2]` block 깊이
- 기준 및 가변 프레임 길이 출력 shape
- 입력과 3D stem·ResNet 마지막 계층까지의 gradient
- 잘못된 rank, 채널, 공간 크기, 빈 축과 정수 dtype 거부
- 전체 테스트 수집과 Python 문법

현재 로컬 검증 명령은 다음과 같다.

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/model -q
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src/model
```

최근 검증 결과는 `13 passed`이며, 학습 가능한 파라미터는 `11,214,144`개다.

## 5. 의존성과 경량 CI 정책

`develop`의 저장소 정책은 개발자별 CUDA·ROCm wheel 차이를 고려해 PyTorch를 `pyproject.toml`과 `uv.lock`에 고정하지 않는다. 백엔드 CI도 PyTorch를 설치하지 않는다.

- 모델 테스트 모듈은 `pytest.importorskip()`으로 PyTorch 설치 여부를 먼저 확인한다.
- PyTorch가 없는 CI에서는 모델 테스트를 실패가 아닌 명시적 skip으로 보고한다.
- PyTorch가 설치된 로컬 환경에서는 동일한 테스트 파일이 전체 forward·backward 계약을 검증한다.
- Ruff와 저장소 정책 검사는 PyTorch 없이 항상 실행한다.
- CI에서는 실제 학습, 데이터셋·사전학습 가중치 다운로드와 GPU 검증을 수행하지 않는다.

이 정책은 CI 자원을 작게 유지하기 위한 현재 기준이다. 향후 모델 CPU 테스트를 필수 CI로 승격할 때만 CPU optional extra와 별도 job 도입을 다시 검토한다.

## 6. GPU 검증 후속 계획

AMD ROCm 호환성은 CPU CI가 보장하지 않는다. 백본 구조나 장치 처리 방식이 변경될 때 실제 대상 GPU에서 다음 smoke test를 수동 실행한다.

- ROCm PyTorch와 대상 GPU 인식 확인
- 작은 배치의 GPU forward와 backward
- 출력과 gradient의 유한값 확인
- GPU 동기화 시 런타임 오류 확인
- 필요할 때만 추론 시간과 메모리 사용량 측정

GPU 테스트는 `gpu` marker로 일반 CPU 테스트와 분리한다. self-hosted GPU runner 도입 전까지 PR 필수 조건으로 사용하지 않는다.

## 7. 현재 작업 완료 조건

- [x] 백본 계약이 SoT에 정의되어 있다.
- [x] 공개 API가 SoT와 일치한다.
- [x] TDD의 Red와 Green 단계를 확인했다.
- [x] CPU 계약 테스트가 통과한다.
- [x] 모델·백엔드 확장을 고려한 테스트 계층이 정의되어 있다.
- [x] 백본 범위를 벗어난 기능이 구현에 섞이지 않았다.
- [x] PyTorch가 없는 경량 CI에서 모델 테스트가 명시적으로 skip된다.
- [ ] 실제 AMD ROCm 환경에서 GPU smoke test를 추가한다.

마지막 GPU 항목은 백본 공개 API 구현의 완료 조건이 아니라 운영 환경 검증을 위한 후속 작업이다.
