"""URL sanitisation: only genuine YouTube video links may reach the downloader."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.errors import ValidationError
from app.ingest.youtube import backoff_delay, canonical_url, classify_failure, parse_video_id

VIDEO = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO}",
        f"http://youtube.com/watch?v={VIDEO}&t=42s",
        f"https://m.youtube.com/watch?feature=share&v={VIDEO}",
        f"https://music.youtube.com/watch?v={VIDEO}&list=RDAMVM",
        f"https://youtu.be/{VIDEO}",
        f"https://youtu.be/{VIDEO}?si=abc",
        f"https://www.youtube.com/shorts/{VIDEO}",
        f"https://www.youtube.com/embed/{VIDEO}?autoplay=1",
        f"https://www.youtube.com/live/{VIDEO}",
        f"  https://WWW.YouTube.com/watch?v={VIDEO}  ",
        f"https://www.youtube.com:443/watch?v={VIDEO}",
    ],
)
def test_accepts_youtube_video_links(url: str) -> None:
    assert parse_video_id(url) == VIDEO


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not a url",
        VIDEO,
        f"www.youtube.com/watch?v={VIDEO}",  # no scheme
        f"ftp://www.youtube.com/watch?v={VIDEO}",
        f"file:///etc/passwd?v={VIDEO}",
        f"javascript:alert(1)//youtube.com/watch?v={VIDEO}",
        # SSRF: other hosts, lookalikes, internal addresses
        "http://169.254.169.254/latest/meta-data/",
        f"http://localhost/watch?v={VIDEO}",
        f"http://127.0.0.1:6379/watch?v={VIDEO}",
        f"https://[::1]/watch?v={VIDEO}",
        f"https://evil.example/watch?v={VIDEO}",
        f"https://youtube.com.evil.example/watch?v={VIDEO}",
        f"https://evilyoutube.com/watch?v={VIDEO}",
        f"https://www.youtube.com@evil.example/watch?v={VIDEO}",
        f"https://user:pass@www.youtube.com/watch?v={VIDEO}",
        f"https://www.youtube.com:8080/watch?v={VIDEO}",
        f"https://www.youtube.com\\@evil.example/watch?v={VIDEO}",
        f"https://www.youtube.com/watch?v={VIDEO}\nHost: evil.example",
        f"https://www.youtube.com/redirect?q=http://evil.example&v={VIDEO}",
        # not a single video
        "https://www.youtube.com/playlist?list=PL1234567890",
        "https://www.youtube.com/@somechannel",
        "https://www.youtube.com/watch?v=tooshort",
        f"https://www.youtube.com/watch?v={VIDEO}extra",
        "https://www.youtube.com/watch?v=../../../etc",
        f"https://youtu.be/{VIDEO[:-1]}$",
        "https://www.youtube.com/watch?v=" + "a" * 3000,
    ],
)
def test_rejects_everything_else(url: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        parse_video_id(url)
    assert excinfo.value.code == "invalid_youtube_url"


def test_downloader_only_ever_sees_a_rebuilt_canonical_url() -> None:
    assert canonical_url(VIDEO) == f"https://www.youtube.com/watch?v={VIDEO}"
    with pytest.raises(ValueError):
        canonical_url("../../etc/passwd")
    with pytest.raises(ValueError):
        canonical_url(f"{VIDEO}&x=1")


def test_permanent_failures_are_not_retried() -> None:
    error = classify_failure("ERROR: [youtube] abc: Private video. Sign in if you've been granted access")
    assert error is not None and error.code == "video_private" and not error.retryable
    assert classify_failure("ERROR: [youtube] abc: Video unavailable") is not None
    assert classify_failure("ERROR: unable to download video data: HTTP Error 503: Service Unavailable") is None
    assert classify_failure("ERROR: Read timed out.") is None


def test_backoff_grows_and_is_capped() -> None:
    settings = Settings(youtube_backoff_base_seconds=2, youtube_backoff_max_seconds=10)
    for attempt, ceiling in [(1, 2), (2, 4), (3, 8), (4, 10), (9, 10)]:
        for _ in range(20):
            assert ceiling / 2 <= backoff_delay(attempt, settings) <= ceiling
