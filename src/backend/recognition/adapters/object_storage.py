"""MinIO와 AWS S3에서 공통으로 사용하는 Object Storage adapter."""

import asyncio

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from src.backend.core.config import Settings


class S3ObjectStorage:
    """boto3 S3 API를 이용하는 private Object Storage 구현."""

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self._bucket = settings.object_storage_bucket

        self._client = boto3.client(
            "s3",
            endpoint_url=settings.object_storage_endpoint_url,
            aws_access_key_id=settings.object_storage_access_key,
            aws_secret_access_key=settings.object_storage_secret_key,
            region_name=settings.object_storage_region,
            config=Config(
                signature_version="s3v4",
                s3={
                    "addressing_style": "path",
                },
            ),
        )

    @property
    def bucket(self) -> str:
        """현재 adapter가 사용하는 bucket 이름."""

        return self._bucket

    @staticmethod
    def _validate_object_key(
        object_key: str,
    ) -> str:
        """Object Storage 내부 key가 안전한 상대 경로인지 검증한다."""

        object_key = object_key.strip()

        if not object_key:
            raise ValueError("object_key는 비어 있을 수 없습니다.")

        if object_key.startswith("/"):
            raise ValueError("object_key는 /로 시작할 수 없습니다.")

        parts = object_key.split("/")

        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("object_key에 잘못된 경로 요소가 포함되어 있습니다.")

        return object_key

    async def ensure_bucket(self) -> None:
        """설정된 bucket이 존재하고 현재 자격증명으로 접근 가능한지 확인한다."""

        await asyncio.to_thread(
            self._client.head_bucket,
            Bucket=self._bucket,
        )

    async def put(
        self,
        *,
        object_key: str,
        data: bytes,
        content_type: str,
        checksum: str,
    ) -> None:
        """객체와 서버에서 계산한 SHA-256 checksum metadata를 저장한다."""

        key = self._validate_object_key(object_key)

        if not data:
            raise ValueError("빈 객체는 저장할 수 없습니다.")

        content_type = content_type.strip()

        if not content_type:
            raise ValueError("content_type은 비어 있을 수 없습니다.")

        checksum = checksum.strip().lower()

        if len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise ValueError("checksum은 SHA-256 hex 문자열이어야 합니다.")

        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={
                "sha256": checksum,
            },
        )

    async def get(
        self,
        object_key: str,
    ) -> bytes:
        """private 객체를 읽고 body stream을 닫는다."""

        key = self._validate_object_key(object_key)

        response = await asyncio.to_thread(
            self._client.get_object,
            Bucket=self._bucket,
            Key=key,
        )

        body = response["Body"]

        try:
            return await asyncio.to_thread(body.read)
        finally:
            body.close()

    async def exists(
        self,
        object_key: str,
    ) -> bool:
        """객체 존재 여부를 HEAD 요청으로 확인한다."""

        key = self._validate_object_key(object_key)

        try:
            await asyncio.to_thread(
                self._client.head_object,
                Bucket=self._bucket,
                Key=key,
            )
        except ClientError as exc:
            error_code = str(
                exc.response.get(
                    "Error",
                    {},
                ).get(
                    "Code",
                    "",
                )
            )

            if error_code in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                return False

            raise

        return True

    async def delete(
        self,
        object_key: str,
    ) -> None:
        """객체를 삭제한다."""

        key = self._validate_object_key(object_key)

        await asyncio.to_thread(
            self._client.delete_object,
            Bucket=self._bucket,
            Key=key,
        )
