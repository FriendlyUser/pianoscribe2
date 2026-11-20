"""Desktop mode: the same API in one process, with no Redis server and no queue service."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.desktop import settings as desktop_settings
from app.desktop.server import build_app
from app.jobs.models import Job, JobOptions, JobResult, JobStatus, Stage


@pytest.fixture
def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("app.desktop.server._system_info", lambda _settings: {"cuda": False, "mps": False})
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models", desktop=True, pdf_renderer="none")
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>PianoScribe</title>")
    return TestClient(build_app(settings, web))


def test_the_web_interface_and_the_api_share_one_port(desktop: TestClient) -> None:
    assert "PianoScribe" in desktop.get("/").text
    config = desktop.get("/api/config").json()
    assert config["desktop"] is True and config["server_pdf"] is False
    assert desktop.get("/api/health").status_code == 200


def test_model_choices_are_saved_and_attached_to_new_jobs(desktop: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []
    monkeypatch.setattr("app.desktop.server.run_job", lambda job_id, **_: started.append(job_id))
    state = desktop.get("/api/desktop/state").json()
    assert state["settings"]["solo_transcriber"] == "tsumugi"
    assert {m["id"] for m in state["models"]} >= {"tsumugi", "muscriptor-medium", "sheetsage2-refined"}

    chosen = {**state["settings"], "cover_seed": 7, "cover_density": 1.2}
    assert desktop.put("/api/desktop/settings", json=chosen).status_code == 200
    first = desktop.post("/api/jobs/youtube", json={"url": "https://youtu.be/dQw4w9WgXcQ"}).json()["job"]
    assert first["options"]["models"]["cover_seed"] == 7

    # A different choice is a different score: it must not come back from the cache.
    desktop.put("/api/desktop/settings", json={**chosen, "cover_seed": 8})
    second = desktop.post("/api/jobs/youtube", json={"url": "https://youtu.be/dQw4w9WgXcQ"}).json()["job"]
    assert second["id"] != first["id"]
    for _ in range(50):
        if len(started) == 2:
            break
        time.sleep(0.05)
    assert started == [first["id"], second["id"]]  # the worker thread picked both up


def test_a_model_that_is_not_downloaded_cannot_be_chosen(desktop: TestClient) -> None:
    settings = desktop.get("/api/desktop/state").json()["settings"]
    response = desktop.put("/api/desktop/settings", json={**settings, "solo_transcriber": "muscriptor-large"})
    assert response.status_code == 409


def test_job_choices_change_the_pipeline_settings() -> None:
    from app.jobs.runner import _default_execute

    seen: list[Settings] = []

    class Spy:
        def __init__(self, job: Job, *, settings: Settings, **_: object) -> None:
            seen.append(settings)

        def run(self) -> Job:
            return job

    import app.pipeline.runner as pipeline_runner

    original = pipeline_runner.Pipeline
    pipeline_runner.Pipeline = Spy  # type: ignore[misc, assignment]
    try:
        job = Job(
            id="a" * 32,
            source={"kind": "youtube", "video_id": "dQw4w9WgXcQ"},
            options=JobOptions(models={"cover_seed": 5}),
        )
        _default_execute(None, None, Settings())(job)  # type: ignore[arg-type]
    finally:
        pipeline_runner.Pipeline = original  # type: ignore[misc]
    assert seen[0].cover_seed == 5


def test_the_job_list_survives_a_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fakeredis

    from app.desktop.server import JobArchive

    first = fakeredis.FakeRedis()
    done = Job(id="b" * 32, source={"kind": "youtube", "video_id": "dQw4w9WgXcQ"}, status=JobStatus.SUCCEEDED,
               stage=Stage.READY, result=JobResult(mode="arrangement", key="C major", time_signature="4/4",
               tempo_bpm=120, measures=1, notes=1, pages=1, duration_seconds=1.0))  # fmt: skip
    running = Job(id="c" * 32, source={"kind": "youtube", "video_id": "dQw4w9WgXcQ"}, status=JobStatus.RUNNING)
    first.set(f"ps:job:{done.id}", done.model_dump_json())
    first.set(f"ps:job:{running.id}", running.model_dump_json())
    JobArchive(first, tmp_path / "jobs.json").save()

    second = fakeredis.FakeRedis(server=fakeredis.FakeServer())
    JobArchive(second, tmp_path / "jobs.json").load()
    assert second.get(f"ps:job:{done.id}") is not None
    assert second.get(f"ps:job:{running.id}") is None  # interrupted work is not resurrected


def test_server_jobs_keep_their_cache_keys() -> None:
    assert JobOptions(solo_piano=True).cache_fragment() == "solo=1"  # unchanged: existing cached scores stay valid
    assert desktop_settings.DesktopSettings().job_models()["solo_transcriber"] == "tsumugi"
