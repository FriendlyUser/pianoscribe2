"""The piano-video option through the API, the job model, and the pipeline with stand-in models."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.api.main import create_app
from app.config import Settings
from app.jobs.keys import upload_job_id
from app.jobs.models import JobOptions
from app.visual.check import score_fidelity

MP4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + os.urandom(4096)
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


def test_the_option_is_a_job_of_its_own_and_absent_from_other_jobs() -> None:
    plain, video = JobOptions(), JobOptions(piano_video=True)
    assert "piano_video" not in plain.model_dump() and video.model_dump()["piano_video"] is True
    assert JobOptions.model_validate_json(video.model_dump_json()).piano_video is True
    assert plain.cache_fragment() == "solo=0" and video.cache_fragment() == "solo=0;video=1"
    assert upload_job_id("ab" * 32, plain) != upload_job_id("ab" * 32, video)


def test_a_video_upload_can_ask_for_its_picture_to_be_read(client: TestClient) -> None:
    sent = client.post("/api/jobs/upload", params={"filename": "tutorial.mp4", "piano_video": "true"}, content=MP4)
    assert sent.status_code == 202
    assert sent.json()["job"]["options"] == {"solo_piano": False, "piano_video": True}
    by_ear = client.post("/api/jobs/upload", params={"filename": "tutorial.mp4"}, content=MP4)
    assert by_ear.json()["job"]["id"] != sent.json()["job"]["id"]


def test_a_sound_file_has_no_picture_to_read(client: TestClient, queue) -> None:
    refused = client.post("/api/jobs/upload", params={"filename": "song.wav", "piano_video": "true"}, content=WAV)
    assert refused.status_code == 415 and refused.json()["error"]["code"] == "video_required"
    assert queue.sent == []


def test_a_youtube_link_can_ask_for_its_picture_to_be_read(client: TestClient) -> None:
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    sent = client.post("/api/jobs/youtube", json={"url": url, "piano_video": True})
    assert sent.status_code == 202 and sent.json()["job"]["options"]["piano_video"] is True
    assert client.post("/api/jobs/youtube", json={"url": url}).json()["job"]["id"] != sent.json()["job"]["id"]


SCORE = """<score-partwise><part id="P1"><measure number="1">
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><tie type="start"/></note>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><tie type="stop"/></note>
<note><pitch><step>F</step><alter>1</alter><octave>3</octave></pitch><duration>4</duration></note>
<note><rest/><duration>4</duration></note>
</measure></part></score-partwise>"""


def test_the_score_is_checked_against_the_notes_read() -> None:
    assert score_fidelity(SCORE, [60, 54]) == (1.0, 1.0)  # a tied note is one note
    assert score_fidelity(SCORE, [60, 54, 67]) == (pytest.approx(2 / 3), 1.0)  # one read note is not in the score
    assert score_fidelity(SCORE, [60]) == (1.0, 0.5)  # one written note was never read


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_pipeline_reads_the_video_and_writes_what_it_read(settings: Settings, tmp_path: Path, monkeypatch) -> None:
    """The whole job with the two audio models stood in for: the notes come from the picture."""
    from app.models import song
    from app.pipeline.local import run_local
    from tests import piano_video, synth

    notes = piano_video.from_beats(synth.ode_to_joy(), synth.ODE_BPM)[:16]
    truth = [piano_video.VideoNote(n.pitch, n.start + 3.0, n.end + 3.0, n.hand) for n in notes]
    video = piano_video.render(truth, tmp_path / "ode.mp4", piano_video.Style(title_seconds=0.5))
    seen: dict[str, object] = {}

    def annotate(audio: Path, out_json: Path, _settings: Settings) -> Path:
        out_json.write_text(json.dumps({"beats": [0.5 * i for i in range(40)], "bars": [], "melody": []}))
        return out_json

    def write(notes_json: Path, annotation: Path, out_xml: Path, _settings: Settings, *, title: str, keys=None):
        read = json.loads(notes_json.read_text())["notes"]
        seen["notes"] = read
        body = "".join(
            f"<note><pitch><step>C</step><alter>{n['pitch'] - 60}</alter><octave>4</octave></pitch>"
            "<duration>1</duration></note>"
            for n in read
        )
        out_xml.write_text(
            '<score-partwise version="3.1"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part>'
            '</part-list><part id="P1"><measure number="1"><attributes><divisions>1</divisions><key><fifths>1</fifths>'
            f"</key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>{body}</measure></part>"
            "</score-partwise>"
        )
        return {}

    monkeypatch.setattr(song, "annotate_audio", annotate)
    monkeypatch.setattr(song, "write_exact_score", write)
    monkeypatch.setattr(song, "track_keys", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key tracker here")))
    settings = settings.model_copy(update={"sandbox": "basic", "pdf_renderer": "none"})
    job = run_local(video, tmp_path / "out", settings, video=True, title="Ode to Joy")

    assert job.result is not None and job.result.mode == "video"
    read = job.result.video
    assert read is not None and read.notes == len(truth) == len(seen["notes"])  # type: ignore[arg-type]
    assert read.right_hand + read.left_hand == len(truth) and read.hands_from == "colours"
    assert read.confirmed_by_keys == len(truth) and (read.lowest_key, read.highest_key) == ("A1", "A6")
    assert read.score_kept == 1.0 and read.score_backed == 1.0
    assert (tmp_path / "out" / "performance.mid").stat().st_size > 0
    assert not list((settings.work_dir).glob("*/video.*"))  # the picture is deleted once read
