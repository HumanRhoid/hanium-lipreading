"""회원가입 및 로그인 API의 요청/응답 데이터 형식."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SignupRequest(BaseModel):
    """회원가입 요청."""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=4, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(min_length=1, max_length=50)
    hospital: str = Field(min_length=1, max_length=100)
    ward: str | None = Field(default=None, max_length=100)


class SignupResponse(BaseModel):
    """회원가입 성공 응답."""

    model_config = ConfigDict(extra="forbid")

    user_id: int
    username: str
    name: str
    hospital: str
    ward: str | None


class LoginRequest(BaseModel):
    """로그인 요청."""

    model_config = ConfigDict(extra="forbid")

    username: str
    password: str


class LoginResponse(BaseModel):
    """로그인 성공 응답."""

    model_config = ConfigDict(extra="forbid")

    session_token: str
    expires_at: datetime


class CurrentUserResponse(BaseModel):
    """현재 로그인 사용자 정보."""

    model_config = ConfigDict(extra="forbid")

    user_id: int
    username: str
    name: str
    hospital: str
    ward: str | None


class LogoutResponse(BaseModel):
    """로그아웃 성공 응답."""

    model_config = ConfigDict(extra="forbid")

    message: str = "로그아웃되었습니다."