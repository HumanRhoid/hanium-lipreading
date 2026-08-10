"""배포 환경에서 사용하는 공통 health endpoint."""

import asyncio
from typing import Literal

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict


class LivenessResponse(BaseModel):
    """process 생존 여부 응답."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    """DB와 추론 gateway를 각각 표시하는 준비 상태."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ready", "not_ready"]
    database: Literal["ready", "not_ready"]
    inference: Literal["ready", "not_ready"]


router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    return LivenessResponse()


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "PostgreSQL 또는 추론 gateway가 준비되지 않음",
        }
    },
)
async def readiness(request: Request):
    database_state = "ready"
    try:
        async with asyncio.timeout(
            request.app.state.settings.readiness_timeout_seconds
        ):
            await request.app.state.database.ping()
    except Exception:
        database_state = "not_ready"

    inference_state = "ready" if request.app.state.gateway.ready else "not_ready"
    ready = (
        not getattr(request.app.state, "draining", True)
        and database_state == "ready"
        and inference_state == "ready"
    )
    payload = ReadinessResponse(
        status="ready" if ready else "not_ready",
        database=database_state,
        inference=inference_state,
    )
    if ready:
        return payload
    return JSONResponse(
        payload.model_dump(mode="json"),
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )
