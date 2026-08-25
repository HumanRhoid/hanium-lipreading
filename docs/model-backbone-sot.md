# 립리딩 시각 백본 Source of Truth

## 1. 문서의 권위

이 문서는 립리딩 시각 백본의 **책임, 공개 인터페이스, 구조적 불변 조건**을 정의하는 단일 기준이다. 백본 구현과 이를 사용하는 상위 모델은 이 문서의 계약을 따라야 한다.

계약을 변경할 때는 이 문서, 구현, 계약 테스트와 영향을 받는 호출부를 같은 변경 단위에서 함께 수정한다. 작업 순서와 완료 현황은 이 문서의 범위에 포함하지 않는다.

## 2. 역할과 경계

백본은 전처리된 입술 프레임에서 짧은 움직임과 공간 특징을 추출한다.

```text
WebSocket 영상 bytes
  -> 영상 디코딩
  -> 입술 ROI 전처리
  -> [시각 백본]
  -> 시간축 모델
  -> 분류 헤드 또는 문장 디코더
```

백본은 다음 책임을 갖지 않는다.

- WebSocket 메시지 처리와 영상 코덱 디코딩
- 얼굴·입술 검출, ROI 크롭, 리사이즈와 픽셀 정규화
- 배치 패딩과 padding mask 생성
- Conformer, GRU, LSTM 등의 시간축 해석
- 클래스 로짓, 문구 확률과 신뢰도 임계값 계산
- 데이터 로딩, 학습·평가 루프와 체크포인트 저장 정책

## 3. 공개 API

공개 진입점은 `src.ml.models.LipReadingBackbone`이다.

```python
from src.ml.models import LipReadingBackbone

backbone = LipReadingBackbone()
features = backbone(frames)
```

모듈 import는 파일·네트워크 접근을 발생시키지 않는다. 모델 생성도 기본값 `pretrained=False`에서는 마찬가지다.

**예외**: `pretrained=True`로 생성하면 생성자가 `load_imagenet_weights()`를 불러 torchvision의 ImageNet 가중치를 내려받는다. 이 경로는 네트워크에 접근한다. 장치 선택과 `train()`·`eval()` 모드 전환은 호출자의 책임이다.

## 4. 입력 계약

백본은 영상 파일이나 bytes가 아닌 전처리 완료 텐서를 받는다.

| 항목 | 계약 |
|---|---|
| 타입 | `torch.Tensor` |
| shape | `[B, 3, T, 96, 192]` (`NCTHW`) |
| dtype | 부동소수점 dtype |
| `B` | 1 이상의 배치 크기 |
| `T` | 1 이상의 프레임 길이 |
| 채널 | 3채널 |
| 공간 크기 | 높이 96, 너비 192 |

NumPy의 `[T, 96, 192, 3]` `uint8` 배열을 사용할 경우, 호출자는 백본 호출 전에 축 순서 변경, 배치 축 추가, 부동소수점 변환과 픽셀 정규화를 완료해야 한다.

```python
frames = torch.from_numpy(array).permute(3, 0, 1, 2)
frames = frames.unsqueeze(0).float().div(255.0)
```

공간 크기는 전처리 출력과 한 값이다. 바꿀 때는 `src/ml/preprocess/normalize.py`의
`TARGET_HEIGHT`·`TARGET_WIDTH`, 이 문서, 계약 테스트를 **같은 변경 단위에서 함께**
고친다. 2026-08-22에 `80x112`에서 `96x192`로 올렸다.

백본은 입력을 암묵적으로 스케일링하거나 mean·std 표준화를 수행하지 않는다. 학습과 추론은 동일한 상위 전처리 계약을 사용해야 한다.

## 5. 출력 계약

| 항목 | 계약 |
|---|---|
| 타입 | `torch.Tensor` |
| shape | `[B, T, 512]` (`NTD`) |
| 시간 길이 | 입력 `T`와 동일 |
| 특징 차원 | 512 |

출력은 각 시점의 시각 특징이며 클래스 로짓이나 확률이 아니다. 클래스 수와 디코더 종류는 출력 shape에 영향을 주지 않는다.

백본은 입력과 모델 파라미터의 dtype 또는 장치를 내부에서 임의로 변환하지 않는다. 호출자는 입력과 모델이 서로 호환되는 dtype과 동일한 장치에 있도록 보장해야 한다.

## 6. 구조적 불변 조건

백본은 `3D-Conv stem + ResNet-18 trunk` 구조를 사용한다. 3D stem의 모든 시간축 stride는 1이며 입력 시간 길이를 보존한다.

| 단계 | 연산 | 출력 shape |
|---|---|---|
| 입력 | - | `[B, 3, T, 96, 192]` |
| 3D stem | `Conv3d(3, 64, kernel=(5,7,7), stride=(1,2,2), padding=(2,3,3))` + BN + ReLU | `[B, 64, T, 48, 96]` |
| 공간 풀링 | `MaxPool3d(kernel=(1,3,3), stride=(1,2,2), padding=(0,1,1))` | `[B, 64, T, 24, 48]` |
| 프레임 변환 | `[B,C,T,H,W] -> [B*T,C,H,W]` | `[B*T, 64, 24, 48]` |
| ResNet layer1 | BasicBlock 2개, 64채널 | `[B*T, 64, 24, 48]` |
| ResNet layer2 | BasicBlock 2개, 128채널 | `[B*T, 128, 12, 24]` |
| ResNet layer3 | BasicBlock 2개, 256채널 | `[B*T, 256, 6, 12]` |
| ResNet layer4 | BasicBlock 2개, 512채널 | `[B*T, 512, 3, 6]` |
| 공간 집계 | Adaptive average pooling | `[B*T, 512]` |
| 출력 복원 | `[B*T,D] -> [B,T,D]` | `[B, T, 512]` |

일반 ResNet-18의 첫 `7x7 Conv2d` stem은 사용하지 않는다. 시간 정보를 함께 처리하는 3D stem이 이를 대체한다. ResNet trunk는 각 layer에 BasicBlock을 `[2, 2, 2, 2]`개 사용한다.

## 7. 가변 길이와 패딩 계약

- 백본은 모든 `T >= 1` 입력을 받아 동일한 `T` 길이의 특징을 반환한다.
- 백본은 프레임을 고정 개수로 자르거나 복제하지 않는다.
- 배치 내 길이 정렬과 padding mask는 호출자가 관리한다.
- 시간축이 축소되지 않으므로 호출자는 입력 길이 또는 mask를 후속 모델에 그대로 전달할 수 있다.
- 3D 합성곱은 인접 프레임을 참조하므로 패딩 시점의 출력은 후속 시간축 모델에서 mask 처리해야 한다.

## 8. 초기화와 가중치 계약

- Conv2d와 Conv3d 가중치는 ReLU 기준 Kaiming 정규분포로 초기화한다.
- BatchNorm의 scale은 1, bias는 0으로 초기화한다.
- 모델 상태는 표준 PyTorch `state_dict` 형식과 호환되어야 한다.
- 사전학습 가중치를 사용하더라도 생성자와 `forward`의 공개 계약은 변경하지 않는다.
- 백본은 기본값 `pretrained=False`에서 외부 가중치를 검색하거나 다운로드하지 않는다. `pretrained=True`는 예외이며 §3을 따른다.

## 9. 입력 검증과 실패 방식

| 잘못된 입력 | 실패 방식 |
|---|---|
| `torch.Tensor`가 아님 | `TypeError` |
| 5차원 `[B,C,T,H,W]`가 아님 | `ValueError` |
| 채널 수가 3이 아님 | `ValueError` |
| 공간 크기가 `(96, 192)`가 아님 | `ValueError` |
| `B` 또는 `T`가 0 | `ValueError` |
| 정수형 텐서 | `TypeError` |

장치 불일치와 모델 파라미터가 지원하지 않는 부동소수점 dtype 오류는 PyTorch 런타임 오류를 그대로 전달한다.

## 10. 호환성과 변경 관리

다음 변경은 백본 소비자에게 영향을 주는 계약 변경이다.

- 공개 import 경로 또는 생성자 변경
- 입력 축 순서, 채널 수 또는 공간 크기 변경
- 출력 특징 차원 또는 축 순서 변경
- 시간축 stride 도입으로 출력 `T` 변경
- 입력 검증 책임을 다른 계층으로 이동

계약 변경은 문서와 테스트를 먼저 갱신한 뒤 구현에 반영한다. 내부 리팩터링은 공개 API와 이 문서의 불변 조건을 유지하는 한 호환 변경으로 본다.

## 11. 기준 구현과 계약 테스트

- 기준 구현: [`src/ml/models/backbone.py`](../src/ml/models/backbone.py)
- 공개 모듈: [`src/ml/models/__init__.py`](../src/ml/models/__init__.py)
- 계약 테스트: [`tests/unit/models/test_backbone.py`](../tests/unit/models/test_backbone.py)

테스트는 이 문서의 계약을 실행 가능한 형태로 검증하지만, 테스트 코드가 이 문서를 대신하지는 않는다.
