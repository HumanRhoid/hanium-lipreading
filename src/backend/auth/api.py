"""회원가입, 로그인 및 로그인 세션 FastAPI 엔드포인트."""

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, status

from src.backend.auth.schemas import (
    CurrentUserResponse,
    LoginRequest,
    LoginResponse,
    LogoutResponse,
    SignupRequest,
    SignupResponse,
)
from src.backend.auth.service import (
    AuthService,
    InvalidCredentialsError,
    InvalidSessionError,
    UsernameAlreadyExistsError,
)

router = APIRouter(
    prefix="/api/v1/auth",
    tags=["auth"],
)


def get_auth_service(request: Request) -> AuthService:
    """FastAPI 애플리케이션에 등록된 AuthService를 반환한다."""

    return request.app.state.auth_service


@router.post(
    "/signup",
    response_model=SignupResponse,
    status_code=status.HTTP_201_CREATED,
)
async def signup(
    payload: SignupRequest,
    request: Request,
) -> SignupResponse:
    """새 의료진 회원을 생성한다."""

    service = get_auth_service(request)

    try:
        user = await service.signup(
            username=payload.username,
            password=payload.password,
            name=payload.name,
            hospital=payload.hospital,
            ward=payload.ward,
        )
    except UsernameAlreadyExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="이미 사용 중인 아이디입니다.",
        ) from exc

    return SignupResponse(
        user_id=user.user_id,
        username=user.username,
        name=user.name,
        hospital=user.hospital,
        ward=user.ward,
    )


@router.post(
    "/login",
    response_model=LoginResponse,
)
async def login(
    payload: LoginRequest,
    request: Request,
) -> LoginResponse:
    """아이디와 비밀번호를 확인하고 로그인 세션을 생성한다."""

    service = get_auth_service(request)

    try:
        login_session = await service.login(
            username=payload.username,
            password=payload.password,
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        ) from exc

    return LoginResponse(
        session_token=login_session.session_token,
        expires_at=login_session.expires_at,
    )


@router.get(
    "/me",
    response_model=CurrentUserResponse,
)
async def get_current_user(
    request: Request,
    session_token: Annotated[
        str,
        Header(alias="X-Session-Token"),
    ],
) -> CurrentUserResponse:
    """현재 로그인한 의료진의 정보를 반환한다."""

    service = get_auth_service(request)

    try:
        user = await service.get_current_user(session_token)
    except InvalidSessionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 만료된 로그인 세션입니다.",
        ) from exc

    return CurrentUserResponse(
        user_id=user.user_id,
        username=user.username,
        name=user.name,
        hospital=user.hospital,
        ward=user.ward,
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
)
async def logout(
    request: Request,
    session_token: Annotated[
        str,
        Header(alias="X-Session-Token"),
    ],
) -> LogoutResponse:
    """현재 로그인 세션을 무효화한다."""

    service = get_auth_service(request)

    try:
        await service.logout(session_token)
    except InvalidSessionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 이미 로그아웃된 세션입니다.",
        ) from exc

    return LogoutResponse()
