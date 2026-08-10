# 립리딩 백엔드 Source of Truth

## 1. 문서의 권위

이 문서는 립리딩 데모 백엔드의 범위, 계층 경계, 동시성 정책과 변경 절차를 정의하는 단일 기준이다.

- 외부 HTTP·WebSocket 계약은 [`api-spec.md`](api-spec.md)를 따른다.
- PostgreSQL 구조와 데이터 제약은 [`erd-spec.md`](erd-spec.md)를 따른다.
- 구현을 변경할 때는 영향을 받는 SoT, 명세, 테스트와 코드를 같은 변경 단위에서 갱신한다.
- REST OpenAPI와 Alembic migration은 실행 가능한 산출물이지만, 설계 의도가 충돌하면 먼저 해당 명세를 수정하고 구현을 맞춘다.

## 2. 현재 목표와 범위

현재 1차 데모 목표는 `lipread-connect` 웹 클라이언트가 카메라 프레임을 모두 전송한 뒤 `stop`을 보내면 한국어 립리딩 결과를 한 번 받는 API를 제공하는 것이다. v1은 프레임 수신 중 추론하거나 부분 결과를 보내지 않는다. rolling-window 실시간 추론은 현재 범위가 아니며, 향후 호환되지 않는 `/api/v2/recognition/stream` 계약으로 설계한다.

포함 범위:

- FastAPI 애플리케이션과 PostgreSQL 연결
- 서비스 생존·준비 상태 확인
- JPEG 프레임 기반 WebSocket 인식 스트림
- 인식 세션, 폐쇄형 문구와 최종 발화 저장
- 실제 모델과 분리된 추론 포트 및 로컬 테스트용 fake 구현
- Alembic migration, 테스트와 CI

제외 범위:

- 로그인, 사용자 프로필, Google OAuth
- 발화 이력 조회와 사용자 피드백 API
- TTS 실행 또는 TTS 재생 이력
- 원본 영상·프레임·오디오 저장
- 모델 가중치, MediaPipe asset과 실제 PyTorch 추론 구현
- 다중 GPU·다중 서버 분산 추론

## 3. 도메인 중심 모듈러 모놀리스

백엔드는 하나의 프로세스와 하나의 PostgreSQL을 사용하는 모놀리스로 배포하되, 코드 경계는 비즈니스 도메인별 모듈로 나누는 **도메인 중심 모듈러 모놀리스**를 사용한다. 현재 규모에서 독립 배포와 분산 트랜잭션 비용을 피하면서도, 모델·저장소·전송 계층을 교체할 수 있는 경계를 유지하기 위한 선택이다.

```text
src/backend/
  core/                   설정, SQLAlchemy Base와 engine
  recognition/
    domain.py             framework와 무관한 enum·결과 값·모델 manifest
    errors.py             인식 요청의 예상 가능한 오류
    ports.py              repository·RecognitionGateway·corrector 교체 계약
    schemas.py            엄격한 WebSocket DTO
    service.py            bounded frame buffer와 최종 추론 수명주기 조정
    api.py                FastAPI WebSocket adapter
    adapters/             PostgreSQL, SyncPredictor worker, JPEG 검증
  health.py               공통 health endpoint
  main.py                 composition root
```

- 현재 비즈니스 도메인은 `recognition` 하나다.
- 모듈 경계는 독립 배포 단위가 아니라 import·책임 경계다. 별도 프로세스나 네트워크 호출을 만들지 않는다.
- `recognition` 내부의 도메인 값과 포트는 FastAPI, SQLAlchemy, NumPy와 PyTorch에 의존하지 않는다.
- `main.py`만 구체 DB, repository, gateway와 router를 조립한다.
- WebSocket handler는 SQLAlchemy ORM이나 동기 predictor를 직접 호출하지 않는다.
- `core`는 `recognition`을 import하지 않고, 인식 전용 ORM과 adapter는 도메인 안에 둔다.
- 별도 DI framework, generic repository와 독립 microservice는 도입하지 않는다.
- 향후 Google OAuth가 필요하면 기존 도메인을 섞지 않고 `identity/` 도메인을 추가한다.

## 4. 비동기와 동기 경계

HTTP, WebSocket과 PostgreSQL I/O는 비동기로 처리한다. OpenCV, MediaPipe와 PyTorch 연산은 `async def` 내부에서 직접 실행하지 않고 제한된 전용 worker에서 동기 실행한다.

```text
Async WebSocket / asyncpg PostgreSQL
  -> RecognitionService
    -> RecognitionGateway
      -> BoundedLocalRecognitionGateway
        -> SyncPredictor (bounded worker)
```

`RecognitionGateway`는 서비스가 의존하는 비동기 포트이고, `SyncPredictor`는 로컬 전처리·모델 구현을 감싸는 동기 교체 계약이다. `BoundedLocalRecognitionGateway`가 두 실행 방식 사이의 제한 실행 경계를 담당한다.

v1 외부 입력 계약인 JPEG, 정확히 `640x360`, 목표 `25fps`, message당 최대 `512 KiB`, 영상당 `30..250`프레임과 누적 최대 `64 MiB`는 `Settings`로 바꿀 수 없는 고정값이다. DB URL·pool, 허용 Origin, 동시성, timeout과 adapter 선택은 배포 설정이지만 고정 입력 계약을 변경하지 않는다. 고정 입력 계약을 바꾸려면 새 API 버전과 모델 bundle을 함께 제공한다.

서버는 `ready` 이후 프레임을 순서대로 검증해 연결별 bounded memory buffer에만 보관한다. 30프레임을 초과한 입력은 `stop` 시 전체 `N`프레임에서 `floor(i × (N-1) / 29)` (`i=0..29`) index로 정확히 30프레임을 균등 추출한다. 이 정책은 처음과 마지막 프레임을 포함하며 시간 보간이나 중복 프레임 생성을 하지 않는다.

서버 backpressure는 연결별 최대 250프레임·64 MiB의 bounded buffer로 구현한다. 어느 상한이든 초과하면 더 쌓거나 오래된 프레임을 조용히 버리지 않고 계약된 오류로 연결을 종료한다. 브라우저 transport queue는 서버가 관찰할 수 없으므로 프런트엔드가 `WebSocket.bufferedAmount`를 감시하고 상한을 넘는 동안 새 프레임을 건너뛴다.

초기 데모 기본값:

- 프로세스당 활성 인식 세션: 1개
- 동시에 실행되는 추론: 1개
- 세션별 입력 buffer: 최대 250프레임 및 누적 64 MiB
- 세션별 terminal 추론: `stop` 이후 최대 1개
- GPU를 사용하는 배포의 Uvicorn worker: 1개

이미 시작된 native/GPU 작업은 WebSocket 연결이 끊겨도 즉시 취소할 수 없다고 가정한다. 실행 슬롯은 작업이 실제로 끝난 뒤에만 반환한다.

## 5. 추론 교체 계약

안정적인 포트는 비동기 `RecognitionGateway`다. API와 애플리케이션은 tensor shape, class index, 체크포인트 형식과 프레임 정규화 방법을 알지 않는다.

추론 결과는 다음 의미만 제공한다.

- 인식된 원문 `text`
- 선택적인 `confidence`
- 선택적인 폐쇄형 `phrase_code`

숫자 class index는 모델 어댑터가 bundle의 라벨 맵으로 해석한다. 어댑터 밖의 서비스, 외부 API와 DB는 숫자 라벨을 저장하거나 노출하지 않고 `phrase_code`와 인식 문구만 사용한다.

실제 모델 어댑터는 전처리, FPS, 입력 프레임 수, 입력 크기, 정규화, 체크포인트와 라벨 맵을 하나의 versioned bundle로 묶는다. `SyncPredictor`가 제공하고 `RecognitionGateway`가 노출하는 `ModelManifest`는 `bundle_version`, 선택적인 `label_map_version`, `supported_modes`, `input_codec`, frame 크기·FPS와 `input_frame_count=30`으로 bundle 능력을 서비스에 알리는 계약이다.

- 서비스 조립 시 manifest의 codec, frame 크기와 FPS가 v1 고정 계약인 JPEG, `640x360`, `25fps`와 일치하는지 검사한다.
- 서비스 조립 시 manifest의 `input_frame_count`가 v1 정규화 출력인 30과 일치하는지 검사한다.
- `open_session()`과 `recognize()`에서 요청한 `RecognitionMode`를 manifest가 지원하는지 검사한다.

manifest는 운영 설정을 대신하지 않고, `Settings`가 모델 내부 사양을 덮어쓰지도 않는다. 불일치는 모델 준비 실패로 처리하여 잘못된 shape나 label map으로 추론하지 않는다.

현재 구현은 다음 두 어댑터만 제공한다.

- `fake`: local/test 환경에서 통신과 저장 흐름을 검증한다.
- `unavailable`: 실제 모델이 없음을 표시하며 인식 요청을 거부한다.

운영 환경에서는 `fake` 선택을 허용하지 않는다. 모델이 준비되지 않으면 readiness는 실패하고 WebSocket은 `MODEL_NOT_READY`를 반환한다.

문맥 교정은 별도 `TextCorrector` 포트로 둔다. 초기 구현은 입력을 변경하지 않는 no-op이며, 이후 Qwen 어댑터로 교체할 수 있다.

## 6. 애플리케이션 수명주기

composition root인 `create_app()`은 `Settings`를 애플리케이션당 한 번 생성·검증하고 lifespan closure에 고정한다. 테스트에서는 같은 경계로 설정과 adapter 대역을 주입한다. FastAPI lifespan은 DB engine, repository, frame validator, `RecognitionGateway`와 `RecognitionService`를 프로세스당 한 번 조립하고 gateway의 `start()`를 호출한다.

- liveness는 프로세스가 요청에 응답하는지만 확인한다.
- readiness는 draining 상태가 아니면서 짧은 PostgreSQL ping과 추론 gateway 준비 상태가 모두 정상인지 확인한다.
- WebSocket 전체 수명 동안 DB session이나 transaction을 보유하지 않는다.
- 세션 생성, 최종 결과 저장, 세션 종료마다 짧은 transaction을 연다.
- 종료를 시작하면 먼저 draining 상태로 전환해 readiness를 실패시키고 신규 WebSocket은 close code `1012`로 거부한다.
- 기존 요청이 끝난 뒤 frame validator와 gateway를 먼저 닫는다. 두 adapter는 서로 독립적으로 drain하되, 이미 시작된 검증·동기 추론이 실제로 끝날 때까지 capacity와 worker를 유지한다.
- validator와 gateway 정리가 모두 끝난 뒤 DB engine을 마지막으로 dispose한다. 세션 종료 기록에 DB가 필요하므로 DB와 실행 adapter를 동시에 닫지 않는다.
- `Settings`는 요청마다 다시 읽지 않는다. 환경변수 변경은 프로세스를 재시작해 새 lifespan에 반영한다.

`stop` 전 WebSocket이 예고 없이 끊기면 buffer를 비우고 추론이나 발화 저장 없이 세션만 종료한다. `stop`을 수락한 뒤에는 disconnect나 handler cancellation이 발생해도 terminal task를 끝까지 보호하여 추론, 교정과 최종 발화 저장을 각각 정확히 한 번 수행한다. buffer를 비우고 세션의 `ended_at`을 기록한 뒤에만 session capacity를 반환한다. 현재 프로토콜에는 프레임 재전송이나 세션 재개가 없으므로 재연결은 항상 새 WebSocket과 새 인식 세션으로 시작한다.

프로세스 강제 종료처럼 cleanup 자체가 실행되지 않으면 오래된 `ended_at IS NULL` 세션이 남을 수 있다. 운영자는 최대 세션 시간과 종료 여유보다 이전인 기준 시각을 선택해 `scripts/reconcile_abandoned_sessions.py --before <timezone-aware ISO 8601> --confirm`으로 해당 열린 세션을 종료 처리한다. 실행 중인 정상 세션을 건드리지 않도록 현재 시각을 그대로 기준으로 사용하지 않는다.

## 7. 개인정보와 운영 정책

- JPEG bytes와 디코딩 프레임은 메모리에서만 처리한다.
- 영상, 프레임, 오디오, IP, User-Agent, 로컬 경로를 DB에 저장하지 않는다.
- 프레임 bytes와 인식 문장을 애플리케이션 로그에 기록하지 않는다.
- `.env`, DB 비밀번호와 외부 asset 경로를 Git에 커밋하지 않는다.
- 모델 가중치와 MediaPipe `.task` 파일은 외부 경로에서 주입한다.
- 저장된 세션과 발화는 자동 만료하지 않으며 명시적인 관리 명령으로만 삭제한다.

운영 배포는 다음 조건을 추가로 지킨다.

- Origin 검사는 인증이 아니다. 인증·인가, 요청 제한과 남용 방지가 없는 현재 버전은 공개 인터넷이나 실제 환자 데이터에 사용하지 않고 승인된 사용자와 제한된 데모 데이터가 있는 통제된 환경에서만 운영한다.
- 브라우저와 reverse proxy 사이는 TLS가 적용된 `wss`만 허용한다. FastAPI는 외부에 직접 노출하지 않고, 신뢰하는 reverse proxy의 사설 upstream으로 둔다.
- reverse proxy는 WebSocket Upgrade를 전달하되 frame body나 WebSocket message를 기록하지 않는다. 전달 헤더는 신뢰할 proxy에서 온 값만 사용한다.
- 운영 log 설정에서 reverse proxy access log와 Uvicorn의 access·WebSocket handshake IP log를 끈다. `--no-access-log`만으로 일부 `uvicorn.error` handshake log가 남을 수 있으므로 해당 logger도 함께 제한한다. 보안상 접속 기록이 꼭 필요하면 IP를 마스킹 또는 가명화하고, User-Agent·인식 문장·query/body는 제외하며, 접근 권한을 제한한 저장소에서 **최대 7일** 보관 후 자동 삭제한다.
- `.env.example`의 `postgres/postgres` 값은 loopback 로컬 데모 전용이다. 운영에서는 기본 계정·기본 비밀번호를 금지하고, 별도 최소 권한 계정과 secret 저장소에서 주입한 고유 비밀번호를 사용한다.
- 인식 세션·최종 발화의 데모 기본 정책은 자동 만료 없이 명시적 purge다. 실제 사용자 데이터를 받기 전 목적에 맞는 별도 보존기간과 정기 삭제 작업을 승인·설정해야 하며, 그 전에는 제한된 데모 데이터만 저장한다.
- 프로세스 관리자의 graceful shutdown 제한은 최악의 단일 추론 시간과 세션 종료 DB 기록 시간을 합친 값보다 길게 둔다. 제한을 넘겨 강제 종료됐다면 오래된 열린 세션 reconciliation을 수행한다.

## 8. TDD와 변경 절차

모든 동작 변경은 Red-Green-Refactor 순서로 진행한다.

1. 이 문서 또는 하위 명세에서 기대 동작을 먼저 확정한다.
2. 해당 동작을 검증하는 테스트를 작성하고 의도한 이유로 실패하는지 확인한다.
3. 테스트를 통과시키는 최소 구현을 작성한다.
4. 전체 테스트가 통과하는 상태에서 중복과 경계를 정리한다.

실패하는 중간 상태는 커밋하지 않는다. PR의 각 커밋은 repository policy, lint와 관련 테스트를 통과해야 한다.
