"""YouTube ingestion: strict URL validation and sandboxed audio extraction.

SSRF defence: the user's URL is only ever *parsed*. A video id is extracted
after validating scheme and host against an allow-list, and the downloader is
invoked with a canonical URL rebuilt from that id, so no user-controlled host,
port, path, credential or redirect target ever reaches the network layer.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from app.config import Settings
from app.errors import JobError, ValidationError
from app.ingest.sandbox import ResourceLimits, SandboxResult, run_sandboxed
from app.logging import get_logger

log = get_logger(__name__)

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_WATCH_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
_SHORT_HOSTS = {"youtu.be"}
_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")
_MAX_URL_LENGTH = 2048

# stderr fragments that mean retrying cannot help, with the message shown to the user.
_PERMANENT: list[tuple[str, str, str]] = [
    ("private video", "video_private", "This video is private."),
    ("video unavailable", "video_unavailable", "This video is unavailable."),
    ("has been removed", "video_unavailable", "This video has been removed."),
    ("not available in your country", "video_geo_blocked", "This video is not available in the server's region."),
    ("confirm your age", "video_age_restricted", "This video is age-restricted and cannot be downloaded."),
    ("members-only", "video_members_only", "This video is for channel members only."),
    ("requires payment", "video_paid", "This video requires payment."),
    ("live event will begin", "video_not_started", "This live stream has not started yet."),
    ("copyright", "video_blocked", "This video is blocked on copyright grounds."),
    ("confirm you’re not a bot", "youtube_blocked", "YouTube is refusing downloads from this server right now. Upload the file instead."),
    ("confirm you're not a bot", "youtube_blocked", "YouTube is refusing downloads from this server right now. Upload the file instead."),
    # What YouTube answers an address it has flagged (cloud servers, CI runners), for any video.
    ("content isn’t available", "youtube_blocked", "YouTube is refusing downloads from this server right now. Upload the file instead."),
    ("content isn't available", "youtube_blocked", "YouTube is refusing downloads from this server right now. Upload the file instead."),
]  # fmt: skip


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    title: str
    duration: float


@dataclass(frozen=True)
class DownloadedAudio:
    path: Path
    metadata: VideoMetadata


def parse_video_id(url: str) -> str:
    """Extract the 11-character video id from a YouTube URL, or raise.

    Accepts watch, short, shorts, embed and live links on the allow-listed
    hosts only; anything else (other hosts, IP literals, credentials, custom
    ports, non-HTTP schemes) is rejected.
    """
    url = url.strip()
    invalid = ValidationError(
        "invalid_youtube_url", "Enter a valid YouTube video link, e.g. https://www.youtube.com/watch?v=…"
    )
    if not url or len(url) > _MAX_URL_LENGTH or any(ord(c) < 0x21 or ord(c) == 0x7F for c in url):
        raise invalid
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise invalid from exc
    if parts.scheme not in ("http", "https") or parts.username or parts.password or port not in (None, 80, 443):
        raise invalid
    host = (parts.hostname or "").lower().rstrip(".")
    candidate: str | None = None
    if host in _SHORT_HOSTS:
        candidate = parts.path.lstrip("/").split("/", 1)[0]
    elif host in _WATCH_HOSTS:
        if parts.path in ("/watch", "/watch/"):
            candidate = (parse_qs(parts.query).get("v") or [""])[0]
        else:
            for prefix in _PATH_PREFIXES:
                if parts.path.startswith(prefix):
                    candidate = parts.path[len(prefix) :].split("/", 1)[0]
                    break
    if not candidate or not _VIDEO_ID.match(candidate):
        raise invalid
    return candidate


def canonical_url(video_id: str) -> str:
    if not _VIDEO_ID.match(video_id):
        raise ValueError("invalid video id")
    return f"https://www.youtube.com/watch?v={video_id}"


def placeholder_title(video_id: str) -> str:
    """A YouTube job's title until the video's own title is known."""
    return f"YouTube video {video_id}"


def classify_failure(stderr: str) -> JobError | None:
    """Map downloader output to a permanent, user-facing error (None = transient)."""
    lowered = stderr.lower()
    for needle, code, message in _PERMANENT:
        if needle in lowered:
            return JobError(code, message)
    return None


def backoff_delay(attempt: int, settings: Settings) -> float:
    """Exponential backoff with full jitter for retry ``attempt`` (1-based)."""
    ceiling = min(settings.youtube_backoff_max_seconds, settings.youtube_backoff_base_seconds * 2 ** (attempt - 1))
    return random.uniform(ceiling / 2, ceiling)  # noqa: S311 - jitter, not security


_COOKIES = ".cookies.txt"  # the operator's cookies, copied into the download folder for one run


def _base_args(settings: Settings) -> list[str]:
    return [
        settings.ytdlp_bin, "--ignore-config", "--no-playlist", "--no-warnings", "--no-progress",
        "--no-cache-dir", "--socket-timeout", "20", "--retries", "2", "--extractor-retries", "2",
        "--no-exec", "--restrict-filenames",
        *(["--cookies", _COOKIES] if settings.youtube_cookies_file else []),
    ]  # fmt: skip


def _run(args: list[str], cwd: Path, timeout: float, settings: Settings) -> SandboxResult:
    cookies = cwd / _COOKIES
    if settings.youtube_cookies_file:
        # yt-dlp writes its cookie jar back when it exits: give it a private copy to write.
        cwd.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.open(cookies, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as copy:
            copy.write(settings.youtube_cookies_file.read_bytes())
    try:
        return run_sandboxed(
            args, cwd=cwd, timeout=timeout, limits=ResourceLimits.for_downloader(settings), check=False
        )
    finally:
        cookies.unlink(missing_ok=True)


def _with_retries(label: str, args: list[str], cwd: Path, timeout: float, settings: Settings) -> SandboxResult:
    last: SandboxResult | None = None
    for attempt in range(1, settings.youtube_max_attempts + 1):
        last = _run(args, cwd, timeout, settings)
        if last.returncode == 0 and not last.timed_out:
            return last
        permanent = classify_failure(last.stderr)
        if permanent is not None:
            raise permanent
        log.warning(
            "youtube.attempt_failed", step=label, attempt=attempt, timed_out=last.timed_out,
            returncode=last.returncode, stderr=last.stderr[-400:],
        )  # fmt: skip
        if attempt < settings.youtube_max_attempts:
            time.sleep(backoff_delay(attempt, settings))
    if last is not None and last.timed_out:
        raise JobError("youtube_timeout", "Downloading from YouTube timed out. Try again, or upload the file instead.")
    raise JobError(
        "youtube_failed",
        "The audio could not be fetched from YouTube after several attempts. Try again later, or upload the file instead.",
    )


def fetch_metadata(video_id: str, workdir: Path, settings: Settings) -> VideoMetadata:
    """Fetch title and duration, enforcing the duration cap before any download."""
    result = _with_retries(
        "metadata",
        [*_base_args(settings), "--skip-download", "--dump-json", "--", canonical_url(video_id)],
        workdir,
        settings.youtube_metadata_timeout_seconds,
        settings,
    )
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise JobError("youtube_failed", "YouTube returned unreadable video information.", retryable=True) from exc
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise JobError("video_live", "Live streams cannot be transcribed. Try again once the stream has ended.")
    duration = float(info.get("duration") or 0.0)
    if duration <= 0:
        raise JobError("video_unavailable", "This video has no playable audio.")
    if duration > settings.max_duration_seconds:
        raise JobError(
            "too_long",
            f"This video is {int(duration // 60)}:{int(duration % 60):02d} long; the limit is "
            f"{settings.max_duration_seconds // 60}:{settings.max_duration_seconds % 60:02d}.",
        )
    return VideoMetadata(video_id=video_id, title=str(info.get("title") or video_id)[:200], duration=duration)


def download_video(video_id: str, workdir: Path, settings: Settings) -> Path:
    """Download the picture of ``video_id`` (video only, no sound) into ``workdir``.

    For piano-tutorial videos, whose notes are read from the frames. The sound is fetched
    separately by ``download_audio``; both start at the same instant of the video.
    """
    if not settings.youtube_enabled:
        raise JobError("youtube_disabled", "YouTube links are disabled on this server. Upload a file instead.")
    workdir.mkdir(parents=True, exist_ok=True)
    for stale in workdir.glob("picture.*"):
        stale.unlink()
    height = settings.youtube_video_height
    _with_retries(
        "download-video",
        [
            *_base_args(settings),
            "-f",
            f"bv*[height<={height}][fps<=60]/bv*[height<={height}]/bv*/b",
            "--max-filesize",
            f"{settings.youtube_max_filesize_mb}M",
            "--no-part",
            "--no-mtime",
            "-o",
            "picture.%(ext)s",
            "--",
            canonical_url(video_id),
        ],  # fmt: skip
        workdir,
        settings.youtube_timeout_seconds,
        settings,
    )
    files = [p for p in workdir.glob("picture.*") if p.is_file() and p.stat().st_size > 0]
    if not files:
        raise JobError(
            "youtube_failed", "YouTube did not return the picture of this video (it may exceed the size limit)."
        )
    return files[0]


def download_audio(video_id: str, workdir: Path, settings: Settings) -> DownloadedAudio:
    """Download the best audio-only stream for ``video_id`` into ``workdir``."""
    if not settings.youtube_enabled:
        raise JobError("youtube_disabled", "YouTube links are disabled on this server. Upload a file instead.")
    workdir.mkdir(parents=True, exist_ok=True)
    metadata = fetch_metadata(video_id, workdir, settings)
    for stale in workdir.glob("source.*"):
        stale.unlink()
    _with_retries(
        "download",
        [
            *_base_args(settings),
            "-f",
            "bestaudio/best",
            "--max-filesize",
            f"{settings.youtube_max_filesize_mb}M",
            "--no-part",
            "--no-mtime",
            "-o",
            "source.%(ext)s",
            "--",
            canonical_url(video_id),
        ],  # fmt: skip
        workdir,
        settings.youtube_timeout_seconds,
        settings,
    )
    files = [p for p in workdir.glob("source.*") if p.is_file() and p.stat().st_size > 0]
    if not files:
        raise JobError(
            "youtube_failed", "YouTube did not return any audio for this video (it may exceed the size limit)."
        )
    return DownloadedAudio(path=files[0], metadata=metadata)
