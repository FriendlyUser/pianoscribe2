"""The score, played: an MP3 of ``score.mid`` on a sampled grand piano.

The MIDI that Verovio renders from the score (the notes as written, each bar at its own tempo)
is played by FluidSynth on a General MIDI SoundFont, whose first program is an acoustic grand
piano (in FluidR3_GM, the SoundFont the Debian and Ubuntu package ``fluid-soundfont-gm`` installs,
a sampled Yamaha grand), and the sound is encoded to MP3 by ffmpeg. The input is the app's own,
but both run as the media tools do (``app.ingest.sandbox``). The level comes from the rendering
itself: its loudest moment is put one decibel below full scale.

Where there is no synthesizer or no SoundFont (the desktop builds ship neither) there is no MP3,
and nothing else depends on it: the score's own page still plays it with its built-in voice.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np

from app.config import Settings
from app.ingest.sandbox import ResourceLimits, SandboxError, run_sandboxed
from app.logging import get_logger

log = get_logger(__name__)

# Where the systems that package FluidR3_GM keep it.
SOUNDFONTS = (
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),  # Debian, Ubuntu: fluid-soundfont-gm
    Path("/usr/share/soundfonts/FluidR3_GM.sf2"),  # Fedora, Arch
    Path("/usr/share/sounds/sf2/default-GM.sf2"),
    Path("/usr/share/soundfonts/default.sf2"),
)
RATE = 44_100  # samples a second
GAIN = 0.5  # FluidSynth's master gain: dense chords stay well clear of full scale, the level is set after
PEAK_DB = -1.0  # the loudest sample, after levelling, in decibels below full scale
QUALITY = 2  # LAME's variable bit rate quality, 0 (best) to 9: about 190 kbit/s


def find_soundfont(settings: Settings) -> Path | None:
    """The SoundFont the score is played with: the one set (``PS_SOUNDFONT_FILE``), else FluidR3_GM where kept."""
    if settings.soundfont_file is not None:
        return settings.soundfont_file if settings.soundfont_file.is_file() else None
    return next((path for path in SOUNDFONTS if path.is_file()), None)


def synthesizer(settings: Settings) -> tuple[str, Path] | None:
    """(FluidSynth, SoundFont) to play the score with; None when either is missing or the MP3 is turned off."""
    if not settings.score_audio:
        return None
    binary = shutil.which(settings.fluidsynth_bin)
    font = find_soundfont(settings)
    if binary is None or font is None:
        return None
    return binary, font


def render_mp3(midi: bytes, workdir: Path, settings: Settings, *, title: str) -> bytes | None:
    """``midi`` played on a sampled piano and encoded as MP3; None when that cannot be done here."""
    found = synthesizer(settings)
    if found is None:
        log.info(
            "audio.skipped",
            fluidsynth=shutil.which(settings.fluidsynth_bin) is not None,
            soundfont=str(find_soundfont(settings) or ""),
            enabled=settings.score_audio,
        )
        return None
    binary, font = found
    folder = workdir / "audio"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    limits = ResourceLimits.for_media(settings)
    try:
        (folder / "score.mid").write_bytes(midi)
        play = [
            binary,
            "-n",  # no MIDI input
            "-i",  # no shell
            "-q",
            *("-C", "0"),  # no chorus: a piano has none
            *("-R", "1"),  # the room
            *("-g", str(GAIN)),
            *("-r", str(RATE)),
            *("-o", "synth.lock-memory=0"),  # nothing to pin in memory for one rendering
            *("-o", "synth.dynamic-sample-loading=1"),  # only the samples the score uses: a fifth of the memory
            *("-T", "raw", "-O", "float", "-E", "little"),
            *("-F", "score.raw"),
            str(font),
            "score.mid",
        ]
        run_sandboxed(play, cwd=folder, timeout=settings.ffmpeg_timeout_seconds, limits=limits)
        sound = np.fromfile(folder / "score.raw", dtype="<f4")
        peak = float(np.abs(sound).max()) if sound.size else 0.0
        del sound
        if not peak > 0.0:
            log.warning("audio.silent", bytes=len(midi))
            return None
        level = PEAK_DB - 20.0 * math.log10(peak)
        encode = [
            settings.ffmpeg_bin,
            *("-nostdin", "-hide_banner", "-loglevel", "error"),
            *("-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", "score.raw"),
            *("-af", f"volume={level:.2f}dB"),
            *("-c:a", "libmp3lame", "-q:a", str(QUALITY)),
            *("-id3v2_version", "3", "-metadata", f"title={title}"),
            *("-y", "score.mp3"),
        ]
        run_sandboxed(encode, cwd=folder, timeout=settings.ffmpeg_timeout_seconds, limits=limits)
        data = (folder / "score.mp3").read_bytes()
    except (SandboxError, OSError) as error:
        detail = error.result.stderr[-300:] if isinstance(error, SandboxError) else str(error)
        log.warning("audio.failed", error=str(error), detail=detail)
        return None
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    log.info("audio.done", bytes=len(data), level_db=round(level, 2), soundfont=font.name)
    return data
