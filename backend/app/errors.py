"""Typed, user-presentable pipeline errors."""

from __future__ import annotations


class JobError(Exception):
    """A failure with a stable code and an actionable, user-safe message.

    ``retryable`` errors are transient (the job system retries them with
    backoff); non-retryable ones are final and reported to the user as-is.
    """

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class ValidationError(JobError):
    """Input rejected before any processing (maps to HTTP 4xx)."""

    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(code, message, retryable=False)
        self.status = status
