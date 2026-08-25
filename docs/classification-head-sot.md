# 폐쇄형 분류 헤드 Source of Truth

## 1. 역할과 경계

분류 헤드는 시간 모델이 출력한 시점별 특징을 고정 문구 N개의 로짓으로 변환한다.

```text
시각 백본 `[B, T, 512]`
  -> 시간 모델 `[B, T, D]`
  -> [분류 헤드]
  -> 클래스 로짓 `[B, N]`
```

헤드는 softmax, 정답 라벨 매핑, `CrossEntropyLoss`, 신뢰도 임계값 판단을 담당하지 않는다. 학습 루프는 반환된 로짓을 `CrossEntropyLoss`에 직접 전달하고, 추론 계층은 필요한 경우 softmax를 적용한다.

## 2. 공개 API

공개 진입점은 `src.ml.models.ClassificationHead`이다.

```python
from src.ml.models import ClassificationHead

head = ClassificationHead(input_dim=512, num_classes=20, dropout=0.2)
logits = head(features, padding_mask=padding_mask)
```

## 3. 입출력 계약

| 구분 | 계약 |
|---|---|
| 특징 타입 | 부동소수점 `torch.Tensor` |
| 특징 shape | `[B, T, D]` |
| padding mask | 선택 입력, bool tensor `[B, T]` |
| mask 의미 | `True`는 padding, `False`는 유효 시점 |
| 출력 shape | `[B, N]` |
| 출력 의미 | softmax 적용 전 클래스 로짓 |

기본 `input_dim`은 현재 백본 특징 차원과 같은 512이다. 시간 모델이 다른 특징 차원을 반환한다면 생성자의 `input_dim`을 해당 값으로 설정한다.

## 4. 시간축 집계

mask가 없으면 모든 시점의 산술 평균을 사용한다. mask가 있으면 유효한 시점만 평균에 포함한다.

```text
pooled[b] = sum(features[b, t] for valid t) / valid_length[b]
logits = Linear(D, N)(Dropout(pooled))
```

유효 시점이 하나도 없는 샘플은 정상적인 입력이 아니므로 `ValueError`를 발생시킨다. 이 계약은 빈 시퀀스가 학습 중 조용히 0 또는 NaN 특징으로 변환되는 것을 막는다.

### 현재 도달하지 않는 경로

헤드는 `padding_mask`를 완전히 구현하지만 **지금 학습·추론 경로는 그것을 넘기지
않는다.** `LipReadingModel.forward(self, frames)`가 마스크를 받는 인자가 없고
`self.head(features)`로만 부르기 때문이다. 클립을 전부 같은 프레임 수로
리샘플링하므로 지금은 패딩이 생기지 않아 동작에는 문제가 없다.

가변 길이를 살리려면 `LipReadingModel.forward`에 인자를 뚫는 것이 먼저다.
위 §2의 예시는 헤드를 직접 부를 때의 계약이다.

## 5. 학습과 추론

학습에서는 로짓과 정수 클래스 ID를 사용한다.

```python
logits = head(features, padding_mask)
loss = torch.nn.functional.cross_entropy(logits, labels)
```

추론에서 확률이나 Top-k가 필요할 때만 별도로 계산한다.

```python
probabilities = logits.softmax(dim=-1)
confidence, label_ids = probabilities.topk(k=3, dim=-1)
```

## 6. 기준 구현과 테스트

- 기준 구현: [`src/ml/models/classification_head.py`](../src/ml/models/classification_head.py)
- 공개 모듈: [`src/ml/models/__init__.py`](../src/ml/models/__init__.py)
- 계약 테스트: [`tests/unit/models/test_classification_head.py`](../tests/unit/models/test_classification_head.py)
