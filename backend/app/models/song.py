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

A **piano-tutorial video** needs no transcriber: its notes are read from the
picture (``app.visual``). The beat grid and the key still come from the audio,
and the notes are then written down by rule (``write_exact_score``).
"""

from __future__ import annotations

import json
import os
import shutil
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
    ffmpeg_libraries: list[str | Path] = (
        ["--ffmpeg-dir", settings.ffmpeg_shared_dir] if settings.ffmpeg_shared_dir else []
    )
    run_step(
        "transcribe",
        [
            settings.tsumugi_python,
            *([settings.scripts_dir / "song_transcribe.py"] if settings.audio_io == "soundfile" else []),
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
            *ffmpeg_libraries,
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


def write_exact_score(
    notes: Path,
    annotation: Path,
    out_xml: Path,
    settings: Settings,
    *,
    title: str,
    keys: Path | None = None,
) -> dict[str, object]:
    """Write notes read from a piano video (``app.visual``: visual.json) as MusicXML, by rule.

    See ``model_scripts/notate_exact.py``: the notes are placed on the beat grid as they
    were read; no network is run. Returns what the writer reports about its work.
    """
    facts: dict[str, object] = {}

    def on_line(line: str) -> None:
        if line.startswith("@@facts "):
            facts.update(json.loads(line[len("@@facts ") :]))

    partial = out_xml.with_suffix(".partial.musicxml")
    tracked: list[str | Path] = ["--keys", keys] if keys else []
    run_step(
        "notate",
        [
            settings.cover_python,
            settings.scripts_dir / "notate_exact.py",
            "--notes",
            notes,
            "--annotation",
            annotation,
            "--out",
            partial,
            "--title",
            title,
            *tracked,
        ],  # fmt: skip
        cwd=settings.cover_dir,
        timeout=settings.notate_timeout_seconds,
        env={"PYTHONPATH": str(settings.cover_dir)},
        on_line=on_line,
    )
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError("the score writer produced no MusicXML")
    partial.replace(out_xml)
    return facts


def link_or_copy(source: Path, target: Path) -> None:
    """Make ``target`` show the contents of ``source`` as cheaply as the platform allows.

    A symbolic link needs administrator rights or Developer Mode on Windows; a hard link
    does not (same volume); a copy always works.
    """
    try:
        target.symlink_to(source)
        return
    except OSError:
        pass
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def transcribe_piano(audio: Path, out_midi: Path, settings: Settings, progress: Progress | None = None) -> Path:
    """Solo piano: transcribe ``audio`` to a performance MIDI with note velocities."""
    raw = out_midi.with_suffix(".notes.mid")
    if progress:
        progress(0.0, "Transcribing piano notes")
    if settings.solo_transcriber.startswith("muscriptor-"):
        _muscriptor_piano(audio, raw, settings)
    else:
        _tsumugi_piano(audio, raw, settings)
    if progress:
        progress(0.7, "Estimating dynamics")
    _add_velocities(audio, raw, out_midi, settings)
    if not out_midi.is_file() or out_midi.stat().st_size == 0:
        raise RuntimeError("the transcription produced no MIDI")
    return out_midi


def _tsumugi_piano(audio: Path, raw: Path, settings: Settings) -> None:
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


def _muscriptor_piano(audio: Path, raw: Path, settings: Settings) -> None:
    """MuScriptor's piano notes, as one track named the way tsumugi's velocity model expects."""
    import mido

    size = settings.solo_transcriber.removeprefix("muscriptor-")
    produced = raw.with_suffix(".muscriptor.mid")
    run_step(
        "transcribe",
        [
            settings.tsumugi_python,
            "-m",
            "muscriptor",
            "transcribe",
            "--model",
            size,
            "--instruments",
            "acoustic_piano",
            "--detect-tempo",
            "false",
            "-o",
            produced,
            audio,
        ],  # fmt: skip
        cwd=raw.parent,
        timeout=settings.transcribe_timeout_seconds,
    )
    source = mido.MidiFile(str(produced))
    piano = mido.MidiFile(ticks_per_beat=source.ticks_per_beat)
    track = mido.MidiTrack([mido.MetaMessage("track_name", name="piano", time=0)])
    now, last = 0, 0
    for message in mido.merge_tracks(source.tracks):
        now += message.time
        if message.is_meta and message.type != "set_tempo":
            continue
        if not message.is_meta and getattr(message, "channel", 0) == 9:
            continue
        track.append(message.copy(time=now - last))
        last = now
    piano.tracks.append(track)
    piano.save(str(raw))
    produced.unlink(missing_ok=True)


def _add_velocities(audio: Path, raw: Path, out_midi: Path, settings: Settings) -> None:
    # The velocity model reads loudness from the stem a track came from; for a solo
    # recording the piano "stem" is the recording itself.
    stems = out_midi.parent / "_stems"
    stems.mkdir(parents=True, exist_ok=True)
    link = stems / "piano.wav"
    if link.is_symlink() or link.exists():
        link.unlink()
    link_or_copy(audio.resolve(), link)
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
