"""HTTP request / response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.jobs.models import Job


class YouTubeRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    solo_piano: bool = False
    piano_video: bool = False


class SubmissionResponse(BaseModel):
    job: Job
    cached: bool
    deduplicated: bool


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ClientConfig(BaseModel):
    """Limits the web client mirrors for upfront validation."""

    max_upload_bytes: int
    max_duration_seconds: int
    allowed_extensions: list[str]
    youtube_enabled: bool
    result_retention_seconds: int
    # Desktop app only: show the model settings panel.
    desktop: bool = False
    # False when the score PDF comes from the desktop shell's print engine rather than the server.
    server_pdf: bool = True
