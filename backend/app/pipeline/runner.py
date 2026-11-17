"""The job pipeline: ingest -> transcribe -> arrange -> notate -> engrave.

Each stage writes its output atomically into the job's working directory and
is skipped when that output already exists, so a retried or redelivered job
resumes after the last completed stage instead of repeating minutes of GPU
work. Stage durations are logged and stored on the job record.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import Settings
from app.errors import JobError
from app.jobs.models import Job, JobResult, JobStatus, SourceKind, Stage
from app.jobs.store import JobStore
from app.logging import get_logger
from app.models.runtime import ModelStepError, as_job_error
from app.storage.artifacts import ArtifactStore

log = get_logger(__name__)

# Share of the overall progress bar given to each user-visible stage.
_SPANS: dict[Stage, tuple[float, float]] = {
    Stage.INGESTING: (0.0, 0.05),
    Stage.SEPARATING: (0.05, 0.25),
    Stage.TRANSCRIBING: (0.25, 0.55),
    Stage.ARRANGING: (0.55, 0.75),
    Stage.ENGRAVING: (0.75, 1.0),
}


class Pipeline:
    def __init__(self, job: Job, *, store: JobStore, artifacts: ArtifactStore, settings: Settings) -> None:
        self.job = job
        self.store = store
        self.artifacts = artifacts
        self.settings = settings
        self.workdir = settings.work_dir / job.id
        self.timings: dict[str, float] = dict(job.timings)
        self._last_publish = 0.0
        self._last_fraction = -1.0

    # -- progress / timing ------------------------------------------------

    def _report(self, stage: Stage, fraction: float, message: str, *, force: bool = False) -> None:
        low, high = _SPANS[stage]
        overall = round(low + (high - low) * min(max(fraction, 0.0), 1.0), 4)
        now = time.monotonic()
        if not force and now - self._last_publish < 0.4 and abs(overall - self._last_fraction) < 0.01:
            return
        self._last_publish, self._last_fraction = now, overall
        self.store.update(self.job.id, status=JobStatus.RUNNING, stage=stage, progress=overall, message=message)

    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            seconds = round(time.perf_counter() - started, 3)
            self.timings[name] = round(self.timings.get(name, 0.0) + seconds, 3)
            log.info("pipeline.stage", stage=name, seconds=seconds)

    # -- stages -----------------------------------------------------------

    def _ingest(self) -> Path:
        from app.ingest import media, youtube
        from app.ingest.signatures import MediaKind

        normalized = self.workdir / "audio.wav"
        if normalized.exists():
            return normalized
        source = self.job.source
        if source.kind == SourceKind.YOUTUBE:
            assert source.video_id is not None
            self._report(Stage.INGESTING, 0.1, "Downloading audio from YouTube", force=True)
            download_dir = self.workdir / "download"
            downloaded = youtube.download_audio(source.video_id, download_dir, self.settings)
            self.job = self.store.update(self.job.id, title=downloaded.metadata.title) or self.job
            raw, expected, cleanup = downloaded.path, None, download_dir
        else:
            raw = self.settings.uploads_dir / f"{source.content_sha256}.{source.extension}"
            if not raw.is_file():
                raise JobError(
                    "source_expired", "The uploaded file is no longer stored on the server. Please upload it again."
                )
            expected, cleanup = MediaKind(source.media_kind) if source.media_kind else None, raw
        self._report(Stage.INGESTING, 0.6, "Validating and normalizing audio", force=True)
        try:
            media.probe(raw, self.settings, expected=expected)
            media.decode_and_normalize(raw, normalized, self.settings)
        finally:
            # Raw media is deleted as soon as it has been decoded (or rejected).
            if cleanup.is_dir():
                shutil.rmtree(cleanup, ignore_errors=True)
            else:
                cleanup.unlink(missing_ok=True)
        return normalized

    def _transcribe_progress(self) -> Callable[[float, str], None]:
        def report(fraction: float, message: str) -> None:
            if fraction < 0.3:
                self._report(Stage.SEPARATING, fraction / 0.3, message)
            else:
                self._report(Stage.TRANSCRIBING, (fraction - 0.3) / 0.7, message)

        return report

    def _arrangement(self, audio: Path) -> tuple[Path, Path, dict[str, object]]:
        """Song -> (score.musicxml, performance.mid, facts). Each model stage is checkpointed."""
        from app.models import fidelity, song

        source = self.workdir / "transcription.mid"
        if not source.exists():
            self._report(Stage.SEPARATING, 0.0, "Separating the instruments", force=True)
            with self._timed("transcribe"):
                song.transcribe_song(audio, source, self.settings, self._transcribe_progress())

        cover_dir = self.workdir / "cover"
        chosen = self.workdir / "performance.mid"
        annotation_path = self.workdir / "annotation.json"
        facts_path = self.workdir / "arrangement.json"
        if not (chosen.exists() and annotation_path.exists() and facts_path.exists()):
            self._report(Stage.ARRANGING, 0.0, "Finding beats and melody, arranging for piano", force=True)
            with self._timed("arrange"):
                result = song.generate_covers(source, audio, cover_dir, self.settings)
                annotation = json.loads(result.annotation.read_text(encoding="utf-8"))
                best, scores = fidelity.choose_cover(result.candidates, annotation)
                facts = {
                    "candidates": len(result.candidates),
                    "fidelity_scores": [round(s, 4) for s in scores],
                    "melody_fidelity": round(fidelity.melody_fidelity(best, annotation), 4),
                }
                shutil.copyfile(result.annotation, annotation_path)
                shutil.copyfile(best, chosen)
                facts_path.write_text(json.dumps(facts))
            shutil.rmtree(cover_dir, ignore_errors=True)
            log.info("pipeline.arranged", **facts)

        score = self.workdir / "score.musicxml"
        if not score.exists():
            self._report(Stage.ENGRAVING, 0.0, "Writing the score", force=True)
            with self._timed("notate"):
                song.write_score(
                    chosen,
                    annotation_path,
                    source,
                    score,
                    self.settings,
                    title=self.job.title,
                    progress=lambda f: self._report(Stage.ENGRAVING, 0.7 * f, "Writing the score"),
                )
        return score, chosen, json.loads(facts_path.read_text())

    def _solo(self, audio: Path) -> tuple[Path, Path, dict[str, object]]:
        """Solo piano -> (score.musicxml, performance.mid, facts): transcribed, not arranged."""
        from app.models import song

        performance = self.workdir / "performance.mid"
        if not performance.exists():
            self._report(Stage.TRANSCRIBING, 0.0, "Transcribing piano notes", force=True)
            with self._timed("transcribe"):
                song.transcribe_piano(
                    audio, performance, self.settings, lambda f, m: self._report(Stage.TRANSCRIBING, f, m, force=True)
                )
        annotation = self.workdir / "annotation.json"
        if not annotation.exists():
            self._report(Stage.ARRANGING, 0.0, "Finding beats and bar lines", force=True)
            with self._timed("annotate"):
                song.annotate_audio(audio, annotation, self.settings)
        score = self.workdir / "score.musicxml"
        if not score.exists():
            self._report(Stage.ENGRAVING, 0.0, "Writing the score", force=True)
            with self._timed("notate"):
                song.write_score(
                    performance,
                    annotation,
                    None,
                    score,
                    self.settings,
                    title=self.job.title,
                    progress=lambda f: self._report(Stage.ENGRAVING, 0.7 * f, "Writing the score"),
                )
        return score, performance, {}

    # -- orchestration ----------------------------------------------------

    def run(self) -> Job:
        from app.engrave import export_musicxml
        from app.engrave.summary import summarize
        from app.ingest.audio import read_wav

        self.workdir.mkdir(parents=True, exist_ok=True)
        with self._timed("ingest"):
            audio = self._ingest()
        try:
            solo = self.job.options.solo_piano
            score_path, performance, facts = self._solo(audio) if solo else self._arrangement(audio)
        except ModelStepError as error:
            raise as_job_error(error) from error

        musicxml = score_path.read_text(encoding="utf-8")
        summary = summarize(musicxml)
        self._report(Stage.ENGRAVING, 0.8, "Engraving sheet music", force=True)
        with self._timed("engrave"):
            exports = export_musicxml(
                musicxml, title=self.job.title, summary=summary.text, page_size=self.settings.page_size
            )
            files = {
                "score.musicxml": musicxml.encode("utf-8"),
                "score.pdf": exports.pdf,
                "score.html": exports.html.encode("utf-8"),
                "score.mid": exports.midi,
                "performance.mid": performance.read_bytes(),
            }
            transcription = self.workdir / "transcription.mid"
            if transcription.exists():
                files["transcription.mid"] = transcription.read_bytes()
            sizes = self.artifacts.put_all(self.job.id, files)

        samples, rate = read_wav(audio)
        fidelity_value = facts.get("melody_fidelity")
        result = JobResult(
            mode="transcription" if self.job.options.solo_piano else "arrangement",
            key=summary.key,
            key_changes=summary.key_changes,
            time_signature=summary.time_signature,
            tempo_bpm=summary.tempo_bpm,
            tempo_unit=summary.tempo_unit,
            measures=summary.measures,
            notes=summary.notes,
            pages=exports.pages,
            duration_seconds=round(samples.shape[-1] / rate, 2),
            melody_fidelity=float(fidelity_value) if isinstance(fidelity_value, (int, float)) else None,
            artifacts=sizes,
        )
        finished = self.store.update(
            self.job.id,
            status=JobStatus.SUCCEEDED,
            stage=Stage.READY,
            progress=1.0,
            message="Sheet music is ready",
            result=result,
            error=None,
            timings=self.timings,
            finished_at=time.time(),
        )
        shutil.rmtree(self.workdir, ignore_errors=True)
        log.info("pipeline.done", timings=self.timings, total=round(sum(self.timings.values()), 2))
        return finished or self.job
