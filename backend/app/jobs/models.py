"""Job domain model shared by the API, the worker and the web client."""

from __future__ import annotations

import time
from enum import StrEnum

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Stage(StrEnum):
    """User-visible pipeline stage."""

    QUEUED = "queued"
    INGESTING = "ingesting"  # downloading / validating / normalising
    SEPARATING = "separating"
    TRANSCRIBING = "transcribing"
    ARRANGING = "arranging"  # generating the piano arrangement of a song
    ENGRAVING = "engraving"  # writing the score and rendering it
    READY = "ready"
    FAILED = "failed"


class SourceKind(StrEnum):
    UPLOAD = "upload"
    YOUTUBE = "youtube"


class JobSource(BaseModel):
    kind: SourceKind
    filename: str | None = None
    extension: str | None = None
    media_kind: str | None = None
    content_sha256: str | None = None
    size_bytes: int | None = None
    video_id: str | None = None


class JobOptions(BaseModel):
    solo_piano: bool = False
    # Model and arrangement settings chosen in the desktop app (see app.desktop.settings).
    # None on the web server, where the server's own configuration applies.
    models: dict[str, str | int | float] | None = Field(default=None, exclude_if=lambda value: value is None)

    def cache_fragment(self) -> str:
        """Stable encoding of every option that changes the output."""
        fragment = f"solo={int(self.solo_piano)}"
        if self.models:
            fragment += ";" + ",".join(f"{k}={self.models[k]}" for k in sorted(self.models))
        return fragment


class JobErrorInfo(BaseModel):
    code: str
    message: str
    retryable: bool = False


class JobResult(BaseModel):
    mode: str  # "arrangement" (a song arranged for piano) or "transcription" (solo piano)
    key: str
    key_changes: int = 0
    time_signature: str
    tempo_bpm: float
    tempo_unit: str = "quarter"
    measures: int
    notes: int
    pages: int
    duration_seconds: float
    # Share (0..1) of the detected melody that the arrangement plays in time; None for solo piano.
    melody_fidelity: float | None = None
    artifacts: dict[str, int] = Field(default_factory=dict)  # name -> size in bytes


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.QUEUED
    stage: Stage = Stage.QUEUED
    progress: float = 0.0  # overall, 0..1
    message: str = "Waiting for a worker"
    title: str = "Untitled"
    source: JobSource
    options: JobOptions = Field(default_factory=JobOptions)
    error: JobErrorInfo | None = None
    result: JobResult | None = None
    attempts: int = 0
    dead_lettered: bool = False
    timings: dict[str, float] = Field(default_factory=dict)  # stage -> seconds
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def terminal(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)
