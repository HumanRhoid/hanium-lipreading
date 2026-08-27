"""Redis Streams 기반 비동기 추론 Job queue."""

from datetime import datetime
from typing import Final, cast
from uuid import UUID

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from src.backend.core.config import Settings
from src.backend.recognition.domain import RecognitionMode
from src.backend.recognition.ports import (
    InferenceJobEnqueueResult,
    InferenceJobRecord,
    InferenceJobStatus,
)

INFERENCE_JOB_STREAM: Final = "stream:inference:jobs"

_JOB_KEY_PREFIX: Final = "inference:job:"
_VIDEO_JOB_KEY_PREFIX: Final = "inference:video:"

_VALID_JOB_STATUSES: Final = frozenset(
    {
        "QUEUED",
        "PROCESSING",
        "SUCCEEDED",
        "FAILED",
    }
)

_ENQUEUE_OR_GET_SCRIPT: Final = """
local existing_job_id = redis.call("GET", KEYS[1])

if existing_job_id then
    return {existing_job_id, "0"}
end

if redis.call("EXISTS", KEYS[2]) == 1 then
    return redis.error_reply("JOB_ID_ALREADY_EXISTS")
end

redis.call(
    "HSET",
    KEYS[2],
    "job_id", ARGV[1],
    "utterance_id", ARGV[2],
    "video_id", ARGV[3],
    "object_key", ARGV[4],
    "mode", ARGV[5],
    "status", ARGV[6],
    "created_at", ARGV[7],
    "updated_at", ARGV[8],
    "error_code", ARGV[9]
)

redis.call(
    "SET",
    KEYS[1],
    ARGV[1]
)

redis.call(
    "XADD",
    KEYS[3],
    "*",
    "job_id", ARGV[1],
    "utterance_id", ARGV[2],
    "video_id", ARGV[3],
    "object_key", ARGV[4],
    "mode", ARGV[5],
    "created_at", ARGV[7]
)

return {ARGV[1], "1"}
"""


class RedisInferenceJobQueue:
    """Redis hash와 Stream을 이용한 추론 Job queue."""

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self._redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    async def ping(
        self,
    ) -> bool:
        """Redis 연결 가능 여부를 확인한다."""

        return bool(await self._redis.ping())

    async def close(
        self,
    ) -> None:
        """Redis 연결 pool을 정리한다."""

        await self._redis.aclose()

    async def enqueue_or_get(
        self,
        *,
        job_id: str,
        utterance_id: int,
        video_id: int,
        object_key: str,
        mode: RecognitionMode,
        created_at: datetime,
    ) -> InferenceJobEnqueueResult:
        """video_id 기준으로 Job을 한 번만 enqueue한다."""

        canonical_job_id = self._canonical_job_id(job_id)

        self._validate_positive_id(
            utterance_id,
            field_name="utterance_id",
        )

        self._validate_positive_id(
            video_id,
            field_name="video_id",
        )

        object_key = object_key.strip()

        if not object_key:
            raise ValueError("object_key는 비어 있을 수 없습니다.")

        created_at_text = self._serialize_datetime(created_at)

        new_job = InferenceJobRecord(
            job_id=canonical_job_id,
            utterance_id=utterance_id,
            video_id=video_id,
            object_key=object_key,
            mode=mode,
            status="QUEUED",
            created_at=created_at,
            updated_at=created_at,
            error_code=None,
        )

        video_job_key = self._video_job_key(video_id)

        job_key = self._job_key(canonical_job_id)

        try:
            raw_result = await self._redis.eval(
                _ENQUEUE_OR_GET_SCRIPT,
                3,
                video_job_key,
                job_key,
                INFERENCE_JOB_STREAM,
                canonical_job_id,
                str(utterance_id),
                str(video_id),
                object_key,
                mode.value,
                "QUEUED",
                created_at_text,
                created_at_text,
                "",
            )
        except ResponseError as exc:
            if "JOB_ID_ALREADY_EXISTS" in str(exc):
                raise RuntimeError(
                    "생성하려는 추론 job_id가 이미 사용 중입니다."
                ) from exc

            raise

        (
            returned_job_id,
            created,
        ) = self._parse_enqueue_result(raw_result)

        if created:
            if returned_job_id != canonical_job_id:
                raise RuntimeError(
                    "Redis가 생성한 추론 Job 식별자가 요청과 일치하지 않습니다."
                )

            return InferenceJobEnqueueResult(
                job=new_job,
                created=True,
            )

        existing_job = await self.get_job(job_id=returned_job_id)

        if existing_job is None:
            raise RuntimeError("영상에 연결된 기존 추론 Job 상태를 찾을 수 없습니다.")

        self._ensure_same_job_request(
            existing_job,
            utterance_id=utterance_id,
            video_id=video_id,
            object_key=object_key,
            mode=mode,
        )

        return InferenceJobEnqueueResult(
            job=existing_job,
            created=False,
        )

    async def get_job(
        self,
        *,
        job_id: str,
    ) -> InferenceJobRecord | None:
        """job_id에 대응하는 현재 추론 상태를 조회한다."""

        canonical_job_id = self._canonical_job_id(job_id)

        raw_job = await self._redis.hgetall(self._job_key(canonical_job_id))

        if not raw_job:
            return None

        return self._parse_job(raw_job)

    @staticmethod
    def _canonical_job_id(
        job_id: str,
    ) -> str:
        """내부 Job ID를 canonical UUID 문자열로 정규화한다."""

        try:
            return str(UUID(job_id.strip()))
        except (
            AttributeError,
            ValueError,
        ) as exc:
            raise ValueError("job_id는 UUID 형식이어야 합니다.") from exc

    @staticmethod
    def _validate_positive_id(
        value: int,
        *,
        field_name: str,
    ) -> None:
        """DB 식별자는 양수만 허용한다."""

        if value <= 0:
            raise ValueError(f"{field_name}는 양수여야 합니다.")

    @staticmethod
    def _serialize_datetime(
        value: datetime,
    ) -> str:
        """timezone-aware datetime만 Redis에 저장한다."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("추론 Job 시간은 timezone-aware datetime이어야 합니다.")

        return value.isoformat()

    @staticmethod
    def _parse_datetime(
        value: str,
    ) -> datetime:
        """Redis datetime 문자열을 검증하며 복원한다."""

        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise RuntimeError(
                "Redis 추론 Job의 시간 형식이 올바르지 않습니다."
            ) from exc

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise RuntimeError(
                "Redis 추론 Job 시간이 timezone 정보를 포함하지 않습니다."
            )

        return parsed

    @staticmethod
    def _parse_enqueue_result(
        raw_result: object,
    ) -> tuple[str, bool]:
        """Lua enqueue 결과를 안전하게 해석한다."""

        if (
            not isinstance(
                raw_result,
                (
                    list,
                    tuple,
                ),
            )
            or len(raw_result) != 2
        ):
            raise RuntimeError("Redis 추론 Job enqueue 결과 형식이 올바르지 않습니다.")

        returned_job_id = str(raw_result[0])

        created_flag = str(raw_result[1])

        if created_flag not in {
            "0",
            "1",
        }:
            raise RuntimeError("Redis 추론 Job 생성 여부 값이 올바르지 않습니다.")

        return (
            returned_job_id,
            created_flag == "1",
        )

    def _parse_job(
        self,
        raw_job: dict[str, str],
    ) -> InferenceJobRecord:
        """Redis hash를 애플리케이션 Job 표현으로 변환한다."""

        required_fields = {
            "job_id",
            "utterance_id",
            "video_id",
            "object_key",
            "mode",
            "status",
            "created_at",
            "updated_at",
            "error_code",
        }

        if not required_fields.issubset(raw_job):
            raise RuntimeError("Redis 추론 Job 데이터가 불완전합니다.")

        status_value = raw_job["status"]

        if status_value not in _VALID_JOB_STATUSES:
            raise RuntimeError("Redis 추론 Job status가 올바르지 않습니다.")

        try:
            utterance_id = int(raw_job["utterance_id"])

            video_id = int(raw_job["video_id"])

            mode = RecognitionMode(raw_job["mode"])
        except ValueError as exc:
            raise RuntimeError(
                "Redis 추론 Job 데이터 형식이 올바르지 않습니다."
            ) from exc

        self._validate_stored_positive_id(
            utterance_id,
            field_name="utterance_id",
        )

        self._validate_stored_positive_id(
            video_id,
            field_name="video_id",
        )

        job_id = self._canonical_stored_job_id(raw_job["job_id"])

        object_key = raw_job["object_key"].strip()

        if not object_key:
            raise RuntimeError("Redis 추론 Job object_key가 비어 있습니다.")

        return InferenceJobRecord(
            job_id=job_id,
            utterance_id=utterance_id,
            video_id=video_id,
            object_key=object_key,
            mode=mode,
            status=cast(
                InferenceJobStatus,
                status_value,
            ),
            created_at=self._parse_datetime(raw_job["created_at"]),
            updated_at=self._parse_datetime(raw_job["updated_at"]),
            error_code=(raw_job["error_code"] or None),
        )

    @staticmethod
    def _canonical_stored_job_id(
        job_id: str,
    ) -> str:
        """Redis에 저장된 Job ID 손상을 감지한다."""

        try:
            return str(UUID(job_id))
        except ValueError as exc:
            raise RuntimeError("Redis 추론 Job ID 형식이 올바르지 않습니다.") from exc

    @staticmethod
    def _validate_stored_positive_id(
        value: int,
        *,
        field_name: str,
    ) -> None:
        """Redis에 저장된 DB 식별자 손상을 감지한다."""

        if value <= 0:
            raise RuntimeError(f"Redis 추론 Job {field_name}가 올바르지 않습니다.")

    @staticmethod
    def _ensure_same_job_request(
        existing_job: InferenceJobRecord,
        *,
        utterance_id: int,
        video_id: int,
        object_key: str,
        mode: RecognitionMode,
    ) -> None:
        """같은 video_id가 다른 추론 요청으로 재사용되는 것을 방지한다."""

        if (
            existing_job.utterance_id != utterance_id
            or existing_job.video_id != video_id
            or existing_job.object_key != object_key
            or existing_job.mode is not mode
        ):
            raise RuntimeError("기존 추론 Job이 영상 메타데이터와 일치하지 않습니다.")

    @staticmethod
    def _job_key(
        job_id: str,
    ) -> str:
        return f"{_JOB_KEY_PREFIX}{job_id}"

    @staticmethod
    def _video_job_key(
        video_id: int,
    ) -> str:
        return f"{_VIDEO_JOB_KEY_PREFIX}{video_id}:job"
