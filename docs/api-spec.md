# 립리딩 백엔드 API 명세

## 1. 범위와 버전

이 문서는 프런트엔드가 사용하는 HTTP와 WebSocket wire contract의 Source of Truth다. 백엔드 범위와 운영 원칙은 [`backend-sot.md`](backend-sot.md), 저장 구조는 [`erd-spec.md`](erd-spec.md)를 따른다. JSON 필드는 `snake_case`, 문자열은 UTF-8을 사용한다.

- HTTP health endpoint는 버전 경로 밖에 둔다.
- 립리딩 WebSocket v1은 `/api/v1` 아래에 두며 전체 영상을 받은 뒤 `stop`에서 한 번 추론한다.
- rolling-window 부분 결과를 제공하는 실시간 인식은 향후 `/api/v2/recognition/stream` 범위다.
- 호환되지 않는 필드나 상태 전이 변경은 새 API 버전으로 제공한다.

## 2. HTTP API

### `GET /health/live`

프로세스가 요청을 처리할 수 있는지 확인한다.

성공 응답 `200`:

```json
{"status":"ok"}
```

### `GET /health/ready`

PostgreSQL과 추론 gateway가 요청을 처리할 준비가 되었는지 확인한다.

성공 응답 `200`:

```json
{"status":"ready","database":"ready","inference":"ready"}
```

실패 응답 `503`:

```json
{"status":"not_ready","database":"ready","inference":"not_ready"}
```

readiness 응답은 내부 예외, DB URL, asset 경로를 노출하지 않는다.

## 3. WebSocket endpoint

### `WS /api/v1/recognition/stream`

하나의 WebSocket 연결은 하나의 인식 세션을 나타낸다.

정상 상태 전이와 server event:

```text
connected -> started -> buffering -> inferencing -> stopped -> closed
             ready          stop       result(final=true) -> stopped -> close 1000
```

연결 후 `START_TIMEOUT_SECONDS` 안에 `start` 명령이 없으면 `INVALID_MESSAGE`로 종료한다. 기본값은 **5초**이며 배포 설정으로 제한 시간을 바꿀 수 있다. `start` 이전의 binary frame, 두 번째 `start`, `stop` 이후의 frame은 허용하지 않는다.

프레임 수신 중에는 JPEG 검증과 bounded memory buffer 적재만 수행한다. 추론·문맥 교정·최종 발화 저장은 `stop`을 받은 뒤 각각 한 번만 수행하며 v1은 부분 결과를 만들지 않는다.

스트림에는 다음 운영 제한을 적용한다.

| 설정 | 기본값 | 기준 | 종료 event |
|---|---:|---|---|
| `STREAM_IDLE_TIMEOUT_SECONDS` | 10초 | `ready` 이후 다음 client message를 기다리는 최대 시간 | `STREAM_IDLE_TIMEOUT`, close `1008` |
| `MAX_SESSION_SECONDS` | 300초 | 인식 세션이 열린 뒤 유지할 수 있는 최대 시간 | `SESSION_LIMIT_REACHED`, close `1008` |

두 값은 배포 설정으로 바꿀 수 있으므로 클라이언트가 숫자를 하드코딩하지 않는다. 서버가 오류 event를 보낼 수 없는 네트워크 단절에서는 클라이언트가 비정상 close만 관찰할 수 있다.

## 4. Client message

### 인식 시작

```json
{"type":"start"}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `type` | `"start"` | 예 | 명령 종류 |
| `mode` | `"CLOSED" \| "OPEN"` | 아니오 | 인식 모드. 생략하면 `CLOSED` |

현재 모드 UI가 없는 프런트엔드는 최소 DTO인 `{"type":"start"}`를 사용한다. 개방형 모델을 명시적으로 선택할 때만 `{"type":"start","mode":"OPEN"}`을 보낸다.

### 영상 프레임

`start` 이후 JPEG 인코딩 결과의 raw bytes를 WebSocket binary message로 전송한다. base64, Data URL, JSON 배열과 text message로 감싸지 않는다.

- 해상도: 정확히 `640x360`
- 목표 전송률: `25fps`
- 단일 message 최대 크기: `512 KiB`
- 영상 최소 길이: `30`프레임
- 영상 최대 길이: `250`프레임
- 영상의 JPEG 누적 최대 크기: `64 MiB` (`67,108,864` bytes)
- 오디오: 전송하지 않음
- 서버 저장: 하지 않음

위 입력값은 v1의 고정 wire contract이며 환경변수로 변경할 수 없다. 서버는 JPEG 크기와 해상도를 검증하지만 프레임 간 시간을 보간하거나 강제로 `25fps`로 재표본화하지 않으므로, 클라이언트가 약 40ms 간격을 맞춘다. 계약 변경은 새 API 버전으로 제공한다.

30프레임을 초과하면 서버는 `stop` 시 전체 `N`프레임에서 다음 index로 정확히 30프레임을 균등 추출해 모델에 전달한다.

```text
floor(i × (N-1) / 29), i = 0..29
```

따라서 처음과 마지막 프레임을 항상 포함한다. 프레임이 30개보다 적으면 추론하지 않고 `INSUFFICIENT_FRAMES`를 보낸다. 251번째 프레임은 `VIDEO_TOO_LONG`, 누적 크기가 64 MiB를 넘게 하는 프레임은 `VIDEO_TOO_LARGE`로 거부한다.

transport에도 message 상한을 적용하므로 너무 큰 frame은 JSON 오류 이벤트 전에 close code `1009`로 거부될 수 있다. 클라이언트는 `FRAME_TOO_LARGE`와 close code `1009`를 같은 입력 오류로 처리한다.

### 인식 종료

```json
{"type":"stop"}
```

## 5. Server event

### 준비 완료

DB session 생성과 추론 접수가 준비된 뒤 전송한다.

```json
{"type":"ready"}
```

### 인식 결과

```json
{"type":"result","text":"물 주세요","final":true,"confidence":0.91}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `type` | `"result"` | 예 | 이벤트 종류 |
| `text` | string | 예 | 공백이 아닌 1~200자. `corrected_text`가 있으면 교정문, 없으면 원문 |
| `final` | `true` | 예 | v1 결과는 항상 최종 발화 |
| `confidence` | number | 아니오 | 모델이 제공하는 경우에만 포함, 범위 `0..1` |

v1은 `final=false` 결과를 보내지 않는다. `stop`을 받으면 전체 입력을 30프레임으로 정규화해 한 번 추론하고 최종 결과를 저장한 뒤 `final=true`로 한 번 전송한다.

외부 DTO는 렌더링과 상태 전이에 필요한 값만 제공한다. DB ID, 내부 `phrase_code`, `raw_text`·`corrected_text` 구분, 사용자·피드백·TTS 필드는 현재 API에 노출하지 않는다.

### 오류

```json
{"type":"error","code":"MODEL_NOT_READY","message":"인식 모델이 준비되지 않았습니다."}
```

클라이언트에는 안정적인 영문 `code`와 사용자에게 표시 가능한 한글 `message`만 제공한다.

| code | 의미 | close code |
|---|---|---|
| `INVALID_MESSAGE` | JSON 형식·순서·명령 오류 | `1008` |
| `UNSUPPORTED_MODE` | 지원하지 않는 인식 모드 | `1008` |
| `INVALID_FRAME` | JPEG 디코딩 실패 또는 해상도 불일치 | `1003` |
| `FRAME_TOO_LARGE` | binary message 크기 초과 | `1009` |
| `VIDEO_TOO_LONG` | 영상이 250프레임을 초과 | `1008` |
| `VIDEO_TOO_LARGE` | JPEG 누적 크기가 64 MiB를 초과 | `1009` |
| `INSUFFICIENT_FRAMES` | `stop` 시 영상이 30프레임 미만 | `stopped` 뒤 `1000` |
| `MODEL_NOT_READY` | 추론 gateway 미준비 | `1013` |
| `SERVER_BUSY` | 활성 세션·추론·프레임 검증 용량 초과 | `1013` |
| `STREAM_IDLE_TIMEOUT` | client message 입력 유휴 제한 초과 | `1008` |
| `SESSION_LIMIT_REACHED` | 인식 세션 최대 이용 시간 초과 | `1008` |
| `INTERNAL_ERROR` | 복구할 수 없는 서버 오류 | `1011` |

내부 예외 내용과 stack trace는 WebSocket으로 보내지 않는다. 로그에는 인식 문장이나 DB parameter가 섞이지 않도록 예외 타입과 안전한 운영 코드만 남긴다.

### 세션 종료

```json
{"type":"stopped"}
```

정상 `stop` 처리 완료 후 전송하고 WebSocket close code `1000`으로 닫는다. 30프레임 미만인 `stop`은 추론·저장 없이 `INSUFFICIENT_FRAMES -> stopped -> close 1000` 순서로 처리한다.

## 6. 동시성과 백프레셔

- 초기 데모는 프로세스당 활성 세션을 1개로 제한한다.
- 추론은 동시에 1개만 실행한다.
- 프레임은 연결별 최대 250개 및 누적 64 MiB의 bounded buffer에만 보관한다. 한 상한이라도 넘으면 해당 연결을 정해진 오류로 종료한다.
- 프레임 수신 중에는 추론 worker를 점유하지 않는다. `stop`을 수락한 뒤 하나의 terminal task만 추론 capacity를 획득한다.
- capacity가 없으면 연결을 무한 대기시키지 않고 `SERVER_BUSY`로 거부한다.
- `stop` 전 연결이 끊기면 buffer를 비우고 추론·발화 저장 없이 세션을 종료한다.
- `stop` 뒤 연결이 끊기거나 handler가 취소돼도 terminal task는 추론·교정·최종 저장을 끝낸다. 실제 작업과 세션 종료가 끝나기 전에는 inference 또는 session capacity를 반환하지 않는다.
- 현재 버전은 세션 재개나 과거 frame 재전송을 지원하지 않는다. 재연결은 새 세션으로 시작한다.

클라이언트는 close code보다 수신한 `error.code`를 우선해 재연결 여부를 판단한다. 재연결 정책:

| 관찰한 종료 | 처리 |
|---|---|
| 정상 `1000` | 자동 재연결하지 않음 |
| 서버 재시작 `1012`, 비정상 종료로 관찰한 `1006` | 지수 backoff와 jitter를 적용해 제한적으로 재연결 |
| `SERVER_BUSY` | 지수 backoff 후 제한적으로 재연결 |
| `MODEL_NOT_READY` | readiness가 회복된 뒤 재연결 |
| `STREAM_IDLE_TIMEOUT` | 카메라와 전송 loop가 다시 활성화된 뒤 새 세션으로 연결 |
| `SESSION_LIMIT_REACHED` | 사용자에게 세션 종료를 알리고 명시적으로 새 세션 시작 |
| `INVALID_MESSAGE`, `UNSUPPORTED_MODE`, `INVALID_FRAME`, `FRAME_TOO_LARGE`, `VIDEO_TOO_LONG`, `VIDEO_TOO_LARGE` | 같은 입력을 자동 재전송하지 않고 원인을 수정한 뒤 연결 |

close code `1006`은 서버가 전송하는 값이 아니라 정상 close frame을 받지 못한 클라이언트가 관찰하는 상태다. 재시도 간격은 예를 들어 `0.5, 1, 2, 4, 8초`로 늘리고 최대 간격과 시도 횟수를 제한한다. 연결이 끝나면 frame timer와 로컬 queue를 정리한다.

## 7. Origin과 배포

- 브라우저 WebSocket의 `Origin`을 설정된 허용 목록과 비교한다.
- local 기본 허용 origin은 `http://localhost:5173`이다.
- 카메라 사용 환경은 localhost 또는 HTTPS이며 배포에서는 TLS reverse proxy를 통한 `wss`만 사용한다.
- reverse proxy는 WebSocket Upgrade를 전달하고 frame body와 message를 access log에 기록하지 않는다.
- 운영 log 설정에서 proxy access log와 Uvicorn의 access·WebSocket handshake IP log를 비활성화한다. `--no-access-log`만으로 일부 handshake log가 남을 수 있다. 불가피한 보안 로그는 IP 마스킹·가명화, 최소 접근 권한, 최대 7일 자동 삭제 정책을 적용한다.
- Origin 검사는 인증을 대체하지 않는다. 인증이 없는 현재 버전은 제한된 데모 환경에서만 운영한다.

## 8. `lipread-connect` 프런트엔드 통합 기준

현재 프런트엔드는 `1280x720` 카메라 미리보기 중심이며 이 WebSocket 송수신이 연결되어 있지 않다. 백엔드 계약에 맞추려면 다음 통합 작업이 별도로 필요하다.

1. 카메라 원본이 `1280x720`이어도 전송 직전에 offscreen canvas 등으로 정확히 `640x360`으로 축소하고 JPEG로 인코딩한다.
2. WebSocket이 열린 직후 기본 `CLOSED` 모드인 `{"type":"start"}`를 text message로 한 번 보낸다. `OPEN` UI가 생긴 뒤에만 mode를 명시한다.
3. `ready`를 받은 뒤 JPEG `ArrayBuffer`를 binary message로 약 `25fps` 전송한다. Data URL 문자열은 보내지 않는다.
4. `WebSocket.bufferedAmount`가 상한을 넘으면 새 frame을 queue에 쌓지 말고 건너뛴 뒤, 연결이 회복되면 최신 frame부터 보낸다.
5. 사용자 종료 시 `{"type":"stop"}`을 보내고, `final=true` 결과와 `stopped`를 처리한 뒤 연결 종료를 반영한다.
6. `error.code`를 기준으로 사용자 메시지와 재시도 가능 여부를 처리한다.
7. `onclose`에서는 frame timer와 전송 queue를 즉시 비우고, 재시도 가능한 종료만 지수 backoff로 새 세션에 연결한다.
8. 탭이 background이면 자동 재연결을 멈추고 `visibilitychange`로 다시 활성화됐을 때 연결한다.
