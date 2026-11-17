"""Upload streaming, sandboxed execution and retention."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.config import Settings
from app.errors import ValidationError
from app.ingest.sandbox import ResourceLimits, SandboxError, run_sandboxed
from app.ingest.upload import receive_upload, sanitize_filename, title_from_filename
from app.jobs.models import Job, JobSource, SourceKind
from app.storage.janitor import purge

WAV_HEAD = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64


async def chunks(data: bytes, size: int = 1000) -> AsyncIterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


async def test_upload_is_streamed_hashed_and_stored_by_content(settings: Settings) -> None:
    data = WAV_HEAD + os.urandom(20_000)
    stored = await receive_upload(chunks(data), filename="My Song.wav", declared_length=len(data), settings=settings)
    assert stored.sha256 == hashlib.sha256(data).hexdigest()
    assert stored.size == len(data)
    assert stored.path == settings.uploads_dir / f"{stored.sha256}.wav"
    assert stored.path.read_bytes() == data
    assert not list(settings.uploads_dir.glob(".incoming-*"))


async def test_oversized_upload_is_rejected_early_and_cleaned_up(settings: Settings) -> None:
    small = settings.model_copy(update={"max_upload_mb": 1})
    with pytest.raises(ValidationError) as declared:
        await receive_upload(chunks(WAV_HEAD), filename="a.wav", declared_length=5 * 1024 * 1024, settings=small)
    assert declared.value.status == 413

    consumed = 0

    async def endless() -> AsyncIterator[bytes]:
        nonlocal consumed
        yield WAV_HEAD
        while True:  # lies about its length; must be cut off at the limit
            consumed += 65536
            yield b"\x00" * 65536

    with pytest.raises(ValidationError) as streamed:
        await receive_upload(endless(), filename="a.wav", declared_length=None, settings=small)
    assert streamed.value.code == "file_too_large"
    assert consumed <= 2 * 1024 * 1024
    assert list(small.uploads_dir.iterdir()) == []


async def test_mislabeled_content_is_rejected_before_the_whole_body_arrives(settings: Settings) -> None:
    sent = 0

    async def executable() -> AsyncIterator[bytes]:
        nonlocal sent
        for _ in range(1000):
            sent += 8192
            yield b"MZ\x90\x00" + b"\x00" * 8188

    with pytest.raises(ValidationError) as excinfo:
        await receive_upload(executable(), filename="song.mp3", declared_length=None, settings=settings)
    assert excinfo.value.code == "unrecognized_content"
    assert sent <= 16384
    assert list(settings.uploads_dir.iterdir()) == []


async def test_bad_extension_and_empty_upload(settings: Settings) -> None:
    with pytest.raises(ValidationError):
        await receive_upload(chunks(WAV_HEAD), filename="a.exe", declared_length=None, settings=settings)
    with pytest.raises(ValidationError) as empty:
        await receive_upload(chunks(b""), filename="a.wav", declared_length=0, settings=settings)
    assert empty.value.code == "empty_file"


def test_filenames_are_never_trusted() -> None:
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\Users\\me\\song.mp3") == "song.mp3"
    assert sanitize_filename("<script>alert(1)</script>.wav") == "script.wav"
    assert sanitize_filename("") == "upload"
    assert title_from_filename("moonlight_sonata-1st.mp3") == "moonlight sonata 1st"


def limits(**overrides: int | None) -> ResourceLimits:
    base = {"memory_bytes": 512 * 1024 * 1024, "cpu_seconds": 5, "file_bytes": 1024 * 1024, "open_files": 64}
    return ResourceLimits(**{**base, **overrides})  # type: ignore[arg-type]


def test_sandbox_runs_and_captures_output(tmp_path: Path) -> None:
    result = run_sandboxed([sys.executable, "-c", "print('hi')"], cwd=tmp_path, timeout=20, limits=limits())
    assert result.returncode == 0 and result.stdout.strip() == "hi" and not result.timed_out
    assert list(tmp_path.iterdir()) == []


def test_sandbox_kills_on_timeout_including_children(tmp_path: Path) -> None:
    script = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"
    started = time.monotonic()
    with pytest.raises(SandboxError) as excinfo:
        run_sandboxed([sys.executable, "-c", script], cwd=tmp_path, timeout=1.0, limits=limits(cpu_seconds=30))
    assert excinfo.value.result.timed_out and time.monotonic() - started < 10


def test_sandbox_enforces_file_size_and_hides_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PS_SECRET_TOKEN", "do-not-leak")
    leak = run_sandboxed(
        [sys.executable, "-c", "import os; print(os.environ.get('PS_SECRET_TOKEN'))"],
        cwd=tmp_path,
        timeout=20,
        limits=limits(),
    )
    assert leak.stdout.strip() == "None"
    big = run_sandboxed(
        [sys.executable, "-c", "open('big.bin','wb').write(b'x' * 5_000_000)"],
        cwd=tmp_path, timeout=20, limits=limits(), check=False,
    )  # fmt: skip
    assert big.returncode != 0
    assert not (tmp_path / "big.bin").exists() or (tmp_path / "big.bin").stat().st_size <= 1024 * 1024


def test_sandbox_enforces_memory_limit(tmp_path: Path) -> None:
    result = run_sandboxed(
        [sys.executable, "-c", "x = bytearray(900 * 1024 * 1024)"],
        cwd=tmp_path,
        timeout=20,
        limits=limits(),
        check=False,
    )
    assert result.returncode != 0


def test_janitor_purges_expired_files_only(settings: Settings, store, artifacts) -> None:
    now = time.time()
    old, fresh = now - settings.raw_retention_seconds - 60, now - 10

    def touch(path: Path, mtime: float) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        os.utime(path, (mtime, mtime))
        return path

    old_upload = touch(settings.uploads_dir / "old.wav", old)
    new_upload = touch(settings.uploads_dir / "new.wav", fresh)
    old_work = touch(settings.work_dir / ("a" * 32) / "normalized.wav", old)
    os.utime(old_work.parent, (old, old))
    busy_work = touch(settings.work_dir / ("b" * 32) / "normalized.wav", old)
    os.utime(busy_work.parent, (old, old))
    lock = store.lock("b" * 32, 30)
    assert lock.acquire()

    expired_id, kept_id = "c" * 32, "d" * 32
    for job_id in (expired_id, kept_id):
        store.create_if_absent(Job(id=job_id, source=JobSource(kind=SourceKind.YOUTUBE, video_id="dQw4w9WgXcQ")))
        artifacts.put_all(job_id, {"score.pdf": b"%PDF", "score.html": b"<html>", "score.musicxml": b"<x/>"})
    long_ago = now - settings.result_retention_seconds - 60
    os.utime(settings.artifacts_dir / expired_id, (long_ago, long_ago))

    try:
        report = purge(settings, store, artifacts, now=now)
    finally:
        lock.release()
    assert (report.raw_files, report.work_dirs, report.results) == (1, 1, 1)
    assert not old_upload.exists() and new_upload.exists()
    assert not old_work.parent.exists() and busy_work.exists()
    assert not artifacts.complete(expired_id) and store.get(expired_id) is None
    assert artifacts.complete(kept_id) and store.get(kept_id) is not None


def test_sandbox_sets_no_new_privs_and_limits(tmp_path: Path) -> None:
    script = (
        "import resource\n"
        "print(open('/proc/self/status').read().split('NoNewPrivs:')[1].split()[0])\n"
        "print(resource.getrlimit(resource.RLIMIT_NOFILE)[0], resource.getrlimit(resource.RLIMIT_CORE)[0])\n"
        "print(resource.getrlimit(resource.RLIMIT_CPU)[0], resource.getrlimit(resource.RLIMIT_AS)[0])\n"
    )
    result = run_sandboxed([sys.executable, "-c", script], cwd=tmp_path, timeout=20, limits=limits())
    assert result.stdout.split() == ["1", "64", "0", "5", str(512 * 1024 * 1024)]
