"""A YouTube link given to the local runner (CLI, CI): the job it builds, and yt-dlp's cookies."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from app.config import Settings
from app.errors import ValidationError
from app.ingest import youtube
from app.ingest.sandbox import SandboxResult
from app.jobs.keys import youtube_job_id
from app.jobs.models import Job, JobOptions, SourceKind
from app.pipeline import local

VIDEO = "dQw4w9WgXcQ"


class Captured(Exception):
    pass


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[Job]:
    """The jobs the local runner hands to the pipeline (which is not run)."""
    jobs: list[Job] = []

    class Pipeline:
        def __init__(self, job: Job, **_: object) -> None:
            jobs.append(job)

        def run(self) -> Job:
            raise Captured

    monkeypatch.setattr(local, "Pipeline", Pipeline)
    return jobs


def test_a_youtube_link_becomes_a_youtube_job(settings: Settings, tmp_path: Path, started: list[Job]) -> None:
    with pytest.raises(Captured):
        local.run_local(f"https://youtu.be/{VIDEO}?si=x", tmp_path / "out", settings, video=True)
    (job,) = started
    assert job.source.kind == SourceKind.YOUTUBE and job.source.video_id == VIDEO
    assert job.id == youtube_job_id(VIDEO, JobOptions(piano_video=True))
    # Same id as the web server gives the same link: the video's own title replaces the placeholder.
    assert job.title == youtube.placeholder_title(VIDEO)


def test_a_title_given_for_a_youtube_link_is_kept(settings: Settings, tmp_path: Path, started: list[Job]) -> None:
    with pytest.raises(Captured):
        local.run_local(f"https://www.youtube.com/watch?v={VIDEO}", tmp_path / "out", settings, title="Etude")
    assert started[0].title == "Etude"


def test_a_link_to_another_site_is_refused(settings: Settings, tmp_path: Path, started: list[Job]) -> None:
    with pytest.raises(ValidationError):
        local.run_local("https://example.com/watch?v=dQw4w9WgXcQ", tmp_path / "out", settings)
    assert started == []


def test_cookies_are_lent_to_one_download_and_taken_back(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "cookies.txt"
    original.write_text("# Netscape HTTP Cookie File\n")
    settings = settings.model_copy(update={"youtube_cookies_file": original})
    seen: list[tuple[list[str], str, int]] = []

    def fake(argv: list[str], *, cwd: Path, **_: object) -> SandboxResult:
        copy = cwd / argv[argv.index("--cookies") + 1]
        seen.append((argv, copy.read_text(), stat.S_IMODE(copy.stat().st_mode)))
        copy.write_text("rewritten by yt-dlp")
        return SandboxResult(returncode=0, stdout="", stderr="", seconds=0.0, timed_out=False)

    monkeypatch.setattr(youtube, "run_sandboxed", fake)
    work = tmp_path / "download"
    youtube._run([*youtube._base_args(settings), "--", youtube.canonical_url(VIDEO)], work, 10, settings)
    ((argv, text, mode),) = seen
    assert text == "# Netscape HTTP Cookie File\n" and mode == 0o600
    assert argv.index("--cookies") < argv.index("--")
    assert original.read_text() == "# Netscape HTTP Cookie File\n"  # never written
    assert list(work.iterdir()) == []  # the copy is gone


def test_no_cookies_unless_configured(settings: Settings) -> None:
    assert "--cookies" not in youtube._base_args(settings)
