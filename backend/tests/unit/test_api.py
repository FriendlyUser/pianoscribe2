"""HTTP API behaviour with an in-memory Redis and queue."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.api.main import create_app
from app.jobs.models import JobStatus, Stage
from app.storage.artifacts import REQUIRED

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + os.urandom(4096)


@pytest.fixture
def client(settings, store, queue, artifacts) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[deps.settings_dep] = lambda: settings
    app.dependency_overrides[deps.store_dep] = lambda: store
    app.dependency_overrides[deps.queue_dep] = lambda: queue
    app.dependency_overrides[deps.artifacts_dep] = lambda: artifacts
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_config(client: TestClient, settings) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    config = client.get("/api/config").json()
    assert config["max_upload_bytes"] == settings.max_upload_bytes
    assert set(config["allowed_extensions"]) == {"mp3", "wav", "mp4", "mov"}


def test_upload_creates_a_job_and_repeat_is_deduplicated(client: TestClient, queue, settings) -> None:
    first = client.post("/api/jobs/upload", params={"filename": "Für Elise.wav", "solo_piano": "true"}, content=WAV)
    assert first.status_code == 202
    body = first.json()
    assert body["job"]["status"] == "queued" and body["job"]["title"] == "Für Elise"
    assert body["job"]["options"] == {"solo_piano": True} and not body["cached"]
    assert len(list(settings.uploads_dir.iterdir())) == 1

    second = client.post("/api/jobs/upload", params={"filename": "renamed.wav", "solo_piano": "true"}, content=WAV)
    assert second.json()["job"]["id"] == body["job"]["id"] and second.json()["deduplicated"]
    assert len(queue.sent) == 1

    other = client.post("/api/jobs/upload", params={"filename": "renamed.wav", "solo_piano": "false"}, content=WAV)
    assert other.json()["job"]["id"] != body["job"]["id"]


def test_cached_upload_returns_immediately_without_keeping_the_file(
    client: TestClient, store, artifacts, queue, settings
) -> None:
    job_id = client.post("/api/jobs/upload", params={"filename": "a.wav"}, content=WAV).json()["job"]["id"]
    for path in settings.uploads_dir.iterdir():
        path.unlink()
    artifacts.put_all(job_id, {name: b"x" for name in REQUIRED})
    store.update(job_id, status=JobStatus.SUCCEEDED, stage=Stage.READY, progress=1.0)
    again = client.post("/api/jobs/upload", params={"filename": "a.wav"}, content=WAV).json()
    assert again["cached"] and again["job"]["status"] == "succeeded"
    assert len(queue.sent) == 1 and list(settings.uploads_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("filename", "content", "status", "code"),
    [
        ("virus.exe", WAV, 415, "unsupported_format"),
        ("song.mp3", WAV, 415, "content_mismatch"),
        ("song.wav", b"<html>not audio</html>" * 300, 415, "unrecognized_content"),
        ("song.wav", b"", 400, "empty_file"),
    ],
)
def test_invalid_uploads_get_actionable_errors(
    client: TestClient, filename: str, content: bytes, status: int, code: str
) -> None:
    response = client.post("/api/jobs/upload", params={"filename": filename}, content=content)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code and response.json()["error"]["message"]


def test_upload_size_limit(client: TestClient, settings) -> None:
    settings.max_upload_mb = 1
    response = client.post(
        "/api/jobs/upload", params={"filename": "big.wav"}, content=WAV + b"\x00" * (2 * 1024 * 1024)
    )
    assert response.status_code == 413 and response.json()["error"]["code"] == "file_too_large"


def test_youtube_submission_and_dedupe_by_video_id(client: TestClient, queue) -> None:
    a = client.post("/api/jobs/youtube", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
    b = client.post("/api/jobs/youtube", json={"url": "https://youtu.be/dQw4w9WgXcQ?t=30"})
    assert a.status_code == 202 and a.json()["job"]["source"]["video_id"] == "dQw4w9WgXcQ"
    assert b.json()["job"]["id"] == a.json()["job"]["id"] and b.json()["deduplicated"]
    assert len(queue.sent) == 1


@pytest.mark.parametrize(
    "url", ["http://169.254.169.254/", "https://evil.example/watch?v=dQw4w9WgXcQ", "file:///etc/passwd", ""]
)
def test_non_youtube_urls_are_rejected(client: TestClient, queue, url: str) -> None:
    response = client.post("/api/jobs/youtube", json={"url": url})
    assert response.status_code in (400, 422) and queue.sent == []


def test_job_lookup_and_artifacts(client: TestClient, store, artifacts) -> None:
    assert client.get("/api/jobs/" + "0" * 32).status_code == 404
    assert client.get("/api/jobs/not-a-job-id").status_code == 404
    job_id = client.post("/api/jobs/upload", params={"filename": "My Piece.wav"}, content=WAV).json()["job"]["id"]
    assert client.get(f"/api/jobs/{job_id}").json()["stage"] == "queued"
    assert client.get(f"/api/jobs/{job_id}/artifacts/score.pdf").status_code == 404

    artifacts.put_all(job_id, {"score.pdf": b"%PDF-1.7", "score.html": b"<html></html>", "score.musicxml": b"<x/>"})
    pdf = client.get(f"/api/jobs/{job_id}/artifacts/score.pdf", params={"download": "true"})
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert 'filename="my-piece.pdf"' in pdf.headers["content-disposition"]
    html = client.get(f"/api/jobs/{job_id}/artifacts/score.html")
    assert "default-src 'none'" in html.headers["content-security-policy"]
    artifacts.put_all(job_id, {"score.html": b"<html></html>", "score.musicxml": b"<x/>", "score.mp3": b"ID3\x03"})
    played = client.get(f"/api/jobs/{job_id}/artifacts/score.mp3", params={"download": "true"})
    assert played.status_code == 200 and played.headers["content-type"] == "audio/mpeg"
    assert 'filename="my-piece.mp3"' in played.headers["content-disposition"]
    assert client.get(f"/api/jobs/{job_id}/artifacts/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/artifacts/secret.txt").status_code == 404


def test_events_stream_returns_snapshot_for_finished_job(client: TestClient, store) -> None:
    job_id = client.post("/api/jobs/upload", params={"filename": "a.wav"}, content=WAV).json()["job"]["id"]
    store.update(job_id, status=JobStatus.FAILED, stage=Stage.FAILED, message="nope")

    class NoRedis:
        def pubsub(self):  # the snapshot path must not need a live subscription to finish
            class P:
                async def subscribe(self, *_: str) -> None: ...
                async def unsubscribe(self, *_: str) -> None: ...
                async def aclose(self) -> None: ...

            return P()

    client.app.dependency_overrides[deps.async_redis_dep] = lambda: NoRedis()  # type: ignore[attr-defined]
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    assert body.startswith("data: ") and '"status":"failed"' in body
