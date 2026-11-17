"""Probe and decode untrusted media with ffprobe / ffmpeg inside the sandbox."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.errors import JobError
from app.ingest.audio import peak_normalize, read_wav, write_wav
from app.ingest.sandbox import ResourceLimits, SandboxError, run_sandboxed
from app.ingest.signatures import PROBE_FORMATS, MediaKind
from app.logging import get_logger

log = get_logger(__name__)

NORMALIZED_SAMPLE_RATE = 44_100
# Only local files and pipes: blocks playlist/concat/HTTP tricks that make
# ffmpeg fetch URLs or read other files named inside a crafted container.
_PROTOCOLS = "file,pipe"


@dataclass(frozen=True)
class MediaInfo:
    format_name: str
    duration: float
    audio_codec: str
    sample_rate: int
    channels: int


def _format_duration(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def probe(path: Path, settings: Settings, *, expected: MediaKind | None = None) -> MediaInfo:
    """Inspect ``path`` and enforce container, stream and duration policy."""
    try:
        result = run_sandboxed(
            [
                settings.ffprobe_bin,
                "-v",
                "error",
                "-protocol_whitelist",
                _PROTOCOLS,
                "-show_entries",
                "format=format_name,duration:stream=codec_type,codec_name,sample_rate,channels,duration",
                "-of",
                "json",
                "--",
                str(path),
            ],  # fmt: skip
            cwd=path.parent,
            timeout=60,
            limits=ResourceLimits.for_media(settings),
        )
        data = json.loads(result.stdout or "{}")
    except (SandboxError, json.JSONDecodeError) as exc:
        raise JobError(
            "unreadable_media", "The file could not be read as audio or video. It may be corrupted."
        ) from exc

    fmt = data.get("format", {})
    format_names = set(str(fmt.get("format_name", "")).split(","))
    if expected is not None and not format_names & PROBE_FORMATS[expected]:
        raise JobError("content_mismatch", "The file's contents do not match its extension.")
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise JobError("no_audio_stream", "This file has no audio track to transcribe.")
    try:
        duration = float(fmt.get("duration") or audio.get("duration") or 0.0)
    except ValueError:
        duration = 0.0
    if duration <= 0.5:
        raise JobError("empty_audio", "The audio is empty or too short to transcribe.")
    if duration > settings.max_duration_seconds:
        raise JobError(
            "too_long",
            f"The audio is {_format_duration(duration)} long; the limit is "
            f"{_format_duration(settings.max_duration_seconds)}. Trim it and try again.",
        )
    return MediaInfo(
        format_name=str(fmt.get("format_name", "")),
        duration=duration,
        audio_codec=str(audio.get("codec_name", "")),
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
    )


def decode_and_normalize(source: Path, target: Path, settings: Settings) -> float:
    """Decode the first audio stream to 44.1 kHz stereo PCM and peak-normalise it.

    Returns the decoded duration in seconds. Decoding (the untrusted step)
    happens in the sandbox; normalisation then operates on plain PCM.
    """
    decoded = target.with_suffix(".decoded.wav")
    try:
        run_sandboxed(
            [
                settings.ffmpeg_bin,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-protocol_whitelist",
                _PROTOCOLS,
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                "-t",
                str(settings.max_duration_seconds),
                "-ac",
                "2",
                "-ar",
                str(NORMALIZED_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                str(decoded),
            ],  # fmt: skip
            cwd=target.parent,
            timeout=settings.ffmpeg_timeout_seconds,
            limits=ResourceLimits.for_media(settings),
        )
    except SandboxError as exc:
        decoded.unlink(missing_ok=True)
        if exc.result.timed_out:
            raise JobError(
                "decode_timeout", "Decoding the media took too long. Try a shorter or simpler file."
            ) from exc
        log.warning("media.decode_failed", stderr=exc.result.stderr[-500:])
        raise JobError("decode_failed", "The audio track could not be decoded. The file may be corrupted.") from exc

    try:
        audio, sample_rate = read_wav(decoded)
    finally:
        decoded.unlink(missing_ok=True)
    if audio.shape[-1] < sample_rate // 2:
        raise JobError("empty_audio", "The audio is empty or too short to transcribe.")
    normalized = peak_normalize(audio)
    if float(abs(normalized).max()) < 1e-4:
        raise JobError("silent_audio", "The audio track is silent.")
    write_wav(target, normalized, sample_rate)
    return float(audio.shape[-1]) / sample_rate
