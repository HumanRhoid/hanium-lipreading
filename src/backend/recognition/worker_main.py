"""Redis 비동기 추론 Worker 실행 진입점."""

import asyncio
import logging
import os
import socket

from src.backend.core import Settings, SQLAlchemyDatabase
from src.backend.main import create_gateway
from src.backend.recognition.adapters.object_storage import S3ObjectStorage
from src.backend.recognition.adapters.redis_job_queue import RedisInferenceJobQueue
from src.backend.recognition.adapters.repository import (
    SQLAlchemyRecognitionRepository,
)
from src.backend.recognition.inference_worker import (
    InferenceWorker,
    MlStoredVideoPreprocessor,
)
from src.backend.recognition.training_candidate_publisher import (
    TrainingCandidatePublisher,
    TrainingCandidatePublisherRunner,
)

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    """설정 기반 자원을 조립하고 추론 Job을 계속 처리한다."""

    settings = Settings()
    database = SQLAlchemyDatabase(settings)
    repository = SQLAlchemyRecognitionRepository(database.session_factory)
    queue = RedisInferenceJobQueue(settings)
    object_storage = S3ObjectStorage(settings)

    training_candidate_publisher = (
        TrainingCandidatePublisher(
            settings=settings,
            repository=repository,
        )
    )

    training_candidate_runner = (
        TrainingCandidatePublisherRunner(
            publisher=training_candidate_publisher
        )
    )
    gateway = create_gateway(settings)
    consumer_name = f"{socket.gethostname()}-{os.getpid()}"

    try:
        await database.ping()
        await object_storage.ensure_bucket()
        await gateway.start()
        await training_candidate_runner.start()

        worker = InferenceWorker(
            queue=queue,
            object_storage=object_storage,
            preprocessor=MlStoredVideoPreprocessor(),
            gateway=gateway,
            repository=repository,
            consumer_name=consumer_name,
        )

        logger.info("추론 Worker 시작: consumer=%s", consumer_name)
        await worker.run_forever()
    finally:
        await training_candidate_runner.close()
        await training_candidate_publisher.close()
        await gateway.close()
        await queue.close()
        await database.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
