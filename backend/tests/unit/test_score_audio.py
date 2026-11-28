"""The score, played (``app.engrave.audio``): what comes out is the notes written, each at its moment.

Needs FluidSynth and a General MIDI SoundFont (``apt-get install fluidsynth fluid-soundfont-gm``)
for the MP3 itself; without them the MP3 is left out, which is tested everywhere.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import mido
import numpy as np
import pytest

from app.config import Settings
from app.engrave.audio import render_mp3, synthesizer
from app.ingest import media
from app.ingest.audio import read_wav
from app.visual import audio as sound

BPM = 100.0
SCALE = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74]  # C3 to D5, a beat each


def scale_midi() -> bytes:
    events = []
    for beat, pitch in enumerate(SCALE):
        events += [(beat * 480, 1, pitch), (round((beat + 0.9) * 480), 0, pitch)]
    track = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0)])
    now = 0
    for tick, on, pitch in sorted(events, key=lambda event: (event[0], event[1])):
        track.append(mido.Message("note_on", note=pitch, velocity=90 if on else 0, time=tick - now))
        now = tick
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(track)
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


@pytest.fixture
def basic(settings: Settings) -> Settings:
    return settings.model_copy(update={"sandbox": "basic"})


def test_the_score_is_played_note_for_note(basic: Settings, tmp_path: Path) -> None:
    if synthesizer(basic) is None or shutil.which(basic.ffmpeg_bin) is None:
        pytest.skip("needs FluidSynth, a SoundFont and ffmpeg")
    data = render_mp3(scale_midi(), tmp_path / "work", basic, title="A scale")
    assert data is not None and data[:3] == b"ID3"
    assert not (tmp_path / "work" / "audio").exists()  # nothing left behind
    (tmp_path / "scale.mp3").write_bytes(data)
    seconds = media.decode_audio(tmp_path / "scale.mp3", tmp_path / "scale.wav", basic)
    audio, rate = read_wav(tmp_path / "scale.wav")
    beat = 60.0 / BPM
    last = (len(SCALE) - 1 + 0.9) * beat
    assert last < seconds < last + 4.0  # the last note, and its ring
    assert 0.8 < float(np.abs(audio).max()) < 0.95  # its loudest moment a decibel below full scale

    # Each note at its moment: the attacks heard are the notes' onsets, to a few milliseconds.
    onsets = [k * beat for k in range(len(SCALE))]
    attacks = sound.attack_times(audio, rate)
    assert len(attacks) == len(SCALE)
    assert max(abs(heard - written) for heard, written in zip(attacks, onsets, strict=True)) < 0.02
    offset = sound.audio_offset(onsets, audio, rate)
    assert offset is not None and offset.reliable and abs(offset.seconds) < 0.02
    # And at its pitch: louder than the semitones and tones around it, and in its own octave.
    notes = [(start, start + 0.9 * beat, pitch) for start, pitch in zip(onsets, SCALE, strict=True)]
    octaves = sound.octave_scores(notes, audio, rate)
    assert max(octaves, key=lambda shift: octaves[shift]) == 0
    bands, per_second = sound.pitch_spectrum(audio, rate)
    for start, _, pitch in notes:
        first = int((start + 0.05) * per_second)
        heard = bands[first : max(first + 1, int((start + 0.45) * per_second))].mean(axis=0)
        own = heard[pitch - sound.LOWEST]
        assert all(heard[pitch + step - sound.LOWEST] < 0.5 * own for step in (-2, -1, 1, 2)), pitch
        assert heard[pitch - 12 - sound.LOWEST] < 0.2 * own, pitch


def test_without_a_synthesizer_or_a_soundfont_there_is_no_mp3(basic: Settings, tmp_path: Path) -> None:
    midi = scale_midi()
    assert synthesizer(basic.model_copy(update={"score_audio": False})) is None
    no_synth = basic.model_copy(update={"fluidsynth_bin": "no-such-synthesizer"})
    assert synthesizer(no_synth) is None and render_mp3(midi, tmp_path, no_synth, title="x") is None
    no_font = basic.model_copy(update={"soundfont_file": tmp_path / "missing.sf2"})
    assert synthesizer(no_font) is None and render_mp3(midi, tmp_path, no_font, title="x") is None


@pytest.mark.skipif(shutil.which("false") is None, reason="needs a program that fails")
def test_a_synthesizer_that_fails_leaves_the_score_without_an_mp3(basic: Settings, tmp_path: Path) -> None:
    font = tmp_path / "font.sf2"
    font.write_bytes(b"not a soundfont")
    failing = basic.model_copy(update={"fluidsynth_bin": "false", "soundfont_file": font})
    assert synthesizer(failing) is not None
    assert render_mp3(scale_midi(), tmp_path / "work", failing, title="x") is None
    assert not (tmp_path / "work" / "audio").exists()
