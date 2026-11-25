"""Decode video frames with ffmpeg inside the sandbox.

Video frames are untrusted media, so they are decoded the same way audio is: by
ffmpeg, as a sandboxed subprocess restricted to local files. This module only ever
sees raw pixels on a pipe.

Two kinds of read are offered. ``sample_frames`` returns a few dozen whole frames
spread over the video, for finding the keyboard. ``read_strips`` returns a few
horizontal strips of every frame, which is all the note reader needs: a strip
of the falling notes just above the keyboard, and strips across the keys.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.errors import JobError
from app.ingest.sandbox import ResourceLimits, SandboxError, run_sandboxed, stream_sandboxed
from app.logging import get_logger

log = get_logger(__name__)

Frames = NDArray[np.uint8]  # [frames, rows, columns, 3], RGB
_PROTOCOLS = "file,pipe"
_CHUNK_BYTES = 48 * 1024 * 1024


@dataclass(frozen=True)
class VideoInfo:
    width: int  # as analysed: the picture is scaled down to at most Settings.visual_max_width
    height: int
    fps: float  # frames per second, as analysed (constant)
    duration: float  # seconds
    source_width: int
    source_height: int

    @property
    def frames(self) -> int:
        return int(round(self.duration * self.fps))


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def probe_video(path: Path, settings: Settings) -> VideoInfo:
    """Size, frame rate and duration of the first video stream of ``path``."""
    try:
        result = run_sandboxed(
            [
                settings.ffprobe_bin,
                "-v",
                "error",
                "-protocol_whitelist",
                _PROTOCOLS,
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,duration,codec_type,disposition:format=duration",
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
        raise JobError("unreadable_media", "The file could not be read as video. It may be corrupted.") from exc
    streams = [s for s in data.get("streams", []) if not (s.get("disposition") or {}).get("attached_pic")]
    if not streams:
        raise JobError(
            "no_video_stream",
            "This file has no video to read notes from. The piano-video option needs a video of falling notes "
            "over a keyboard.",
        )
    stream = streams[0]
    try:
        width, height = int(stream["width"]), int(stream["height"])
        rate = Fraction(stream.get("avg_frame_rate") or "0/1")
        if rate <= 0:
            rate = Fraction(stream.get("r_frame_rate") or "0/1")
        duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise JobError("unreadable_media", "The video stream could not be understood.") from exc
    if width < 320 or height < 180 or rate <= 0 or duration <= 0.5:
        raise JobError("video_too_small", "The video is too small or too short to read notes from.")
    fps = min(float(rate), float(settings.visual_max_fps))
    scale = min(1.0, settings.visual_max_width / width)
    return VideoInfo(
        width=_even(width * scale),
        height=_even(height * scale),
        fps=round(fps, 3),
        duration=duration,
        source_width=width,
        source_height=height,
    )


def _input_args(path: Path, settings: Settings, *, keyframes_only: bool = False) -> list[str]:
    return [
        settings.ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        _PROTOCOLS,
        *(["-skip_frame", "nokey"] if keyframes_only else []),
        "-i",
        str(path),
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-t",
        str(settings.max_duration_seconds),
    ]


def _read_exactly(pipe: object, size: int) -> bytes:
    """Read ``size`` bytes, or fewer at the end of the stream."""
    parts: list[bytes] = []
    remaining = size
    while remaining > 0:
        part = pipe.read(remaining)  # type: ignore[attr-defined]
        if not part:
            break
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def _decode_error(exc: SandboxError) -> JobError:
    if exc.result.timed_out:
        return JobError("decode_timeout", "Reading the video took too long. Try a shorter or smaller file.")
    log.warning("visual.decode_failed", stderr=exc.result.stderr[-500:])
    return JobError("decode_failed", "The video could not be decoded. The file may be corrupted.")


def sample_frames(path: Path, info: VideoInfo, settings: Settings, *, count: int) -> Frames:
    """About ``count`` whole frames spread over the video, no two the same.

    Only key frames are decoded where the video has enough of them, which is many times
    faster than decoding everything; otherwise every frame is decoded and the wanted ones kept.
    """
    step = max(0.25, info.duration / count)
    frame_bytes = info.width * info.height * 3
    # Frames at least ``step`` apart. (An ``fps`` filter would repeat a frame to fill a gap,
    # and repeated frames defeat the medians taken over the sample.)
    spaced = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{step:.4f})'"

    def read(keyframes_only: bool) -> Frames:
        argv = [
            *_input_args(path, settings, keyframes_only=keyframes_only),
            "-vf",
            f"{spaced},scale={info.width}:{info.height}:flags=bilinear",
            "-fps_mode",
            "vfr",
            "-frames:v",
            str(count + 2),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            with stream_sandboxed(
                argv,
                cwd=path.parent,
                timeout=settings.ffmpeg_timeout_seconds,
                limits=ResourceLimits.for_media(settings),
            ) as pipe:
                data = _read_exactly(pipe, frame_bytes * (count + 2))
                pipe.read()  # nothing more is expected; let the tool finish
        except SandboxError as exc:
            raise _decode_error(exc) from exc
        usable = len(data) // frame_bytes
        return np.frombuffer(data[: usable * frame_bytes], dtype=np.uint8).reshape(usable, info.height, info.width, 3)

    frames = read(keyframes_only=True)
    if len(frames) < min(count, 24):
        frames = read(keyframes_only=False)
    return frames


def read_strips(
    path: Path,
    info: VideoInfo,
    settings: Settings,
    strips: list[tuple[int, int]],
    *,
    timeout: float | None = None,
) -> Iterator[Frames]:
    """Yield, for every frame in order, the rows of each strip stacked into one picture.

    ``strips`` are ``(top row, height)`` pairs with even numbers (rows of the analysed size).
    Each yielded array holds a run of consecutive frames: ``[frames, total rows, width, 3]``.
    The frame rate is made constant, so frame ``i`` is at ``i / info.fps`` seconds.
    """
    for top, height in strips:
        if top % 2 or height % 2 or height <= 0 or top < 0 or top + height > info.height:
            raise ValueError(f"strip ({top}, {height}) does not fit a {info.height}-row picture on even rows")
    labels = [f"s{i}" for i in range(len(strips))]
    graph = [
        f"[0:v]fps={info.fps},scale={info.width}:{info.height}:flags=bilinear,"
        f"split={len(strips)}{''.join(f'[i{label}]' for label in labels)}"
    ]
    for label, (top, height) in zip(labels, strips, strict=True):
        graph.append(f"[i{label}]crop={info.width}:{height}:0:{top}:exact=1[{label}]")
    stacked = "".join(f"[{label}]" for label in labels)
    graph.append(f"{stacked}vstack=inputs={len(strips)}[out]" if len(strips) > 1 else f"{stacked}null[out]")
    rows = sum(height for _, height in strips)
    frame_bytes = rows * info.width * 3
    per_chunk = max(1, _CHUNK_BYTES // frame_bytes)
    argv = [
        *_input_args(path, settings),
        "-filter_complex",
        ";".join(graph),
        "-map",
        "[out]",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        with stream_sandboxed(
            argv,
            cwd=path.parent,
            timeout=timeout or settings.visual_timeout_seconds,
            limits=ResourceLimits.for_media(settings),
        ) as pipe:
            while True:
                data = _read_exactly(pipe, frame_bytes * per_chunk)
                usable = len(data) // frame_bytes
                if usable:
                    yield np.frombuffer(data[: usable * frame_bytes], dtype=np.uint8).reshape(
                        usable, rows, info.width, 3
                    )
                if len(data) < frame_bytes * per_chunk:
                    break
    except SandboxError as exc:
        raise _decode_error(exc) from exc
