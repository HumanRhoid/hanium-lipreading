"""로그인 사용자의 비동기 추론 Job 상태 조회 서비스."""

from src.backend.recognition.ports import (
    InferenceJobQueue,
    InferenceJobStatusView,
    VideoUploadRepository,
)


class InferenceJobStatusService:
    """Redis Job과 PostgreSQL 영상 소유권을 함께 확인한다."""

    def __init__(
        self,
        *,
        repository: VideoUploadRepository,
        inference_job_queue: InferenceJobQueue,
    ) -> None:
        self._repository = repository
        self._inference_job_queue = inference_job_queue

    async def get_for_user(
        self,
        *,
        user_id: int,
        job_id: str,
    ) -> InferenceJobStatusView | None:
        """현재 사용자가 소유한 추론 Job만 반환한다."""

        if user_id <= 0:
            raise ValueError("user_id는 양의 정수여야 합니다.")

        job = await self._inference_job_queue.get_job(
            job_id=job_id,
        )

        if job is None:
            return None

        asset = await self._repository.find_video_asset_by_id(
            video_id=job.video_id,
        )

        if asset is None:
            return None

        if asset.user_id != user_id:
            return None

        result = None

        if job.status == "SUCCEEDED":
            result = await self._repository.get_inference_result(
                utterance_id=job.utterance_id,
            )

        return InferenceJobStatusView(
            job=job,
            result=result,
        )
