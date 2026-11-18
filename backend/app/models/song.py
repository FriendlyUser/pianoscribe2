"""Recording -> piano score with the tsumugi model family.

A **song** (band, vocals, orchestra, game soundtrack ...) is arranged for piano in
three model stages, each a subprocess in the model's own environment:

1. **Transcribe** (tsumugi): separate the recording into stems and transcribe
   every instrument, with velocities, beats, chords and keys, into one MIDI
   file. This is the exact configuration the cover model was trained on.
2. **Arrange** (SheetSage2-Refined + tsumugi-piano-cover-v2): find beats, bar
   lines and the melody in the audio, then generate a piano performance of
   the song on that beat grid.
3. **Notate** (piano_perf2score): write the performance as a two-staff piano
   score in MusicXML, with bar lines taken from the beat grid of the
   original recording.

A **solo piano** recording is not arranged, only transcribed: tsumugi's notes
(with its velocity model), the beat grid of the recording, then the same
notation stage.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.models.runtime import run_step

Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class CoverResult:
    candidates: list[Path]  # generated piano performances (MIDI), one per sample
    annotation: Path  # SheetSage2-Refined beats / bars / melody of the original (JSON)


def transcribe_song(audio: Path, out_midi: Path, settings: Settings, progress: Progress | None = None) -> Path:
    """Stage 1: multi-instrument transcription of ``audio`` to ``out_midi``."""
    stems_done = 0

    def on_line(line: str) -> None:
        nonlocal stems_done
        if progress is None or not line.startswith("@@"):
            return
        kind, _, payload = line[2:].partition(" ")
        data = json.loads(payload) if payload else {}
        if kind == "progress" and data.get("stage") == "separate" and data.get("total"):
            progress(0.3 * data["done"] / data["total"], "Separating the instruments")
        elif kind == "stem":
            progress(0.3 + 0.6 * min(stems_done, 5) / 6, f"Transcribing the {data.get('stem', 'next')} part")
        elif kind == "stem_done":
            stems_done += 1

    work = out_midi.parent / "_tsumugi_work"
    run_step(
        "transcribe",
        [
            settings.tsumugi_python,
            settings.cover_dir / "cover_studio" / "tsumugi_worker.py",
            "--tsumugi-dir",
            settings.tsumugi_dir,
            "--audio",
            audio,
            "--out",
            out_midi,
            "--work",
            work,
            "--device",
            settings.model_device,
            "--window-batch-size",
            str(settings.tsumugi_window_batch_size),
        ],  # fmt: skip
        cwd=settings.tsumugi_dir,
        timeout=settings.transcribe_timeout_seconds,
        on_line=on_line,
    )
    if not out_midi.is_file() or out_midi.stat().st_size == 0:
        raise RuntimeError("tsumugi produced no MIDI")
    return out_midi


def generate_covers(
    source_midi: Path,
    audio: Path,
    out_dir: Path,
    settings: Settings,
    *,
    seed: int | None = None,
    annotation: Path | None = None,
) -> CoverResult:
    """Stage 2: generate piano covers of the song on the beat grid of the recording.

    Generation is sampled, so several candidates are produced in one run (same
    analysis, same loaded model); the caller keeps the most faithful one. ``seed``
    overrides the configured seed, and ``annotation`` reuses the beat and melody
    analysis of an earlier run instead of repeating it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    reuse: list[str | Path] = ["--sheetsage", annotation] if annotation is not None else []
    for stale in out_dir.glob("*.mid"):
        stale.unlink()
    run_step(
        "arrange",
        [
            settings.cover_python,
            "-m",
            "piano_cover.generate",
            "--checkpoint",
            settings.cover_checkpoint,
            "--source",
            source_midi,
            "--audio",
            audio,
            "--out-dir",
            out_dir,
            "--num-samples",
            str(max(1, settings.cover_candidates)),
            "--onset-bias",
            str(settings.cover_onset_bias),
            "--dynamics",
            str(settings.cover_dynamics),
            "--density",
            str(settings.cover_density),
            "--fill",
            str(settings.cover_fill),
            "--above",
            str(settings.cover_above),
            "--span",
            str(settings.cover_span),
            "--source-cfg",
            str(settings.cover_source_cfg),
            "--temperature",
            str(settings.cover_temperature),
            "--top-p",
            str(settings.cover_top_p),
            "--seed",
            str(settings.cover_seed if seed is None else seed),
            *reuse,
        ],  # fmt: skip
        cwd=settings.cover_dir,
        timeout=settings.arrange_timeout_seconds,
    )
    candidates = sorted(out_dir.glob("*.mid"))
    produced = annotation if annotation is not None else out_dir / f"{audio.stem}.sheetsage.json"
    if not candidates or not produced.is_file():
        raise RuntimeError("the cover model produced no output")
    return CoverResult(candidates=candidates, annotation=produced)


def write_score(
    cover_midi: Path,
    annotation: Path,
    source_midi: Path | None,
    out_xml: Path,
    settings: Settings,
    *,
    title: str,
    keys: Path | None = None,
    progress: Callable[[float], None] | None = None,
) -> Path:
    """Stage 3: write a performance as MusicXML (see ``model_scripts/notate_cover.py``).

    ``source_midi`` supplies chord symbols (the multi-instrument transcription of a song);
    pass None for solo piano.
    """

    def on_line(line: str) -> None:
        if progress is None or not line.startswith("@@progress "):
            return
        data = json.loads(line[len("@@progress ") :])
        if data.get("total"):
            progress(min(1.0, data["done"] / data["total"]))

    partial = out_xml.with_suffix(".partial.musicxml")
    chords: list[str | Path] = ["--chords", source_midi] if source_midi is not None else []
    # A song's key signatures follow the key tracker; a solo recording's are only corrected by it.
    sections: list[str | Path] = ["--section-bars", str(settings.solo_section_bars)] if source_midi is None else []
    tracked: list[str | Path] = (
        ["--keys", keys, "--key-policy", "follow" if source_midi is not None else "correct"] if keys else []
    )
    run_step(
        "notate",
        [
            settings.cover_python,
            settings.scripts_dir / "notate_cover.py",
            "--checkpoint",
            settings.cover_checkpoint,
            "--cover",
            cover_midi,
            "--annotation",
            annotation,
            "--out",
            partial,
            "--title",
            title,
            *chords,
            *tracked,
            *sections,
        ],  # fmt: skip
        cwd=settings.cover_dir,
        timeout=settings.notate_timeout_seconds,
        env={"PYTHONPATH": str(settings.cover_dir)},
        on_line=on_line,
    )
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError("the score model produced no MusicXML")
    partial.replace(out_xml)
    return out_xml


def transcribe_piano(audio: Path, out_midi: Path, settings: Settings, progress: Progress | None = None) -> Path:
    """Solo piano: transcribe ``audio`` to a performance MIDI with note velocities."""
    raw = out_midi.with_suffix(".notes.mid")
    if progress:
        progress(0.0, "Transcribing piano notes")
    run_step(
        "transcribe",
        [
            settings.tsumugi_python,
            "-m",
            "instrument_agnostic_amt.amt.cli.infer",
            "--audio",
            audio,
            "--output-midi",
            raw,
            "--allowed-instruments",
            "piano",
            "--amp",
            "--device",
            settings.model_device,
            "--window-batch-size",
            str(settings.tsumugi_window_batch_size),
        ],  # fmt: skip
        cwd=settings.tsumugi_dir,
        timeout=settings.transcribe_timeout_seconds,
    )
    if progress:
        progress(0.7, "Estimating dynamics")
    # The velocity model reads loudness from the stem a track came from; for a solo
    # recording the piano "stem" is the recording itself.
    stems = out_midi.parent / "_stems"
    stems.mkdir(parents=True, exist_ok=True)
    link = stems / "piano.wav"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(audio.resolve())
    try:
        run_step(
            "velocity",
            [
                settings.tsumugi_python,
                "-m",
                "instrument_agnostic_amt.velocity.cli.infer_velocity",
                "--midi",
                raw,
                "--stems-dir",
                stems,
                "--output-midi",
                out_midi,
            ],  # fmt: skip
            cwd=settings.tsumugi_dir,
            timeout=settings.transcribe_timeout_seconds,
        )
    finally:
        link.unlink(missing_ok=True)
        stems.rmdir()
        raw.unlink(missing_ok=True)
    if not out_midi.is_file() or out_midi.stat().st_size == 0:
        raise RuntimeError("tsumugi produced no MIDI")
    return out_midi


def annotate_audio(audio: Path, out_json: Path, settings: Settings) -> Path:
    """Beats, bar lines, time signatures and melody of ``audio`` (SheetSage2-Refined)."""
    run_step(
        "annotate",
        [
            settings.cover_python,
            settings.scripts_dir / "annotate_audio.py",
            "--audio",
            audio,
            "--out",
            out_json,
            "--device",
            settings.model_device,
        ],  # fmt: skip
        cwd=settings.cover_dir,
        timeout=settings.arrange_timeout_seconds,
        env={"PYTHONPATH": str(settings.cover_dir)},
    )
    if not out_json.is_file():
        raise RuntimeError("the beat tracker produced no annotation")
    return out_json


def track_keys(audio: Path, out_json: Path, settings: Settings) -> Path:
    """Keys and key changes of ``audio`` (SheetSage2-Refined's key tracker), as ``{"keys": [...]}``."""
    scratch = out_json.parent / "_keys"
    scratch.mkdir(parents=True, exist_ok=True)
    device = [] if settings.model_device == "auto" else ["--device", settings.model_device]
    try:
        run_step(
            "keys",
            [
                settings.cover_python,
                "-m",
                "sheetsage2_refined.keys",
                audio,
                "--out-dir",
                scratch,
                "--memory-fraction",
                str(settings.analysis_memory_fraction),
                *device,
            ],  # fmt: skip
            cwd=settings.cover_dir,
            timeout=settings.arrange_timeout_seconds,
        )
        produced = scratch / f"{audio.stem}.keys.json"
        if not produced.is_file():
            raise RuntimeError("the key tracker produced no output")
        produced.replace(out_json)
    finally:
        for leftover in scratch.glob("*"):
            leftover.unlink()
        scratch.rmdir()
    return out_json
