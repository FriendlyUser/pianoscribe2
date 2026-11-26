"""The piano-video reader against videos whose every note is known (tests/piano_video.py).

No model is involved: the reader measures. A video is rendered from a list of notes, encoded
with H.264 like an upload, read back, and the notes that come out are compared with the
ones that went in.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.errors import JobError
from app.ingest import media
from app.ingest.audio import read_wav
from app.visual.reader import VisualReading, read_video, write_midi
from tests import piano_video, synth
from tests.piano_video import Style, VideoNote

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
SMALL = Style(width=960, height=540, title_seconds=0.5)  # quick to render; a white key is 27 pixels wide


@pytest.fixture
def read(settings: Settings, tmp_path: Path) -> Callable[..., tuple[VisualReading, float]]:
    """read(notes, style) -> (the reading of a video of those notes, seconds before the roll appears)."""
    settings = settings.model_copy(update={"sandbox": "basic"})

    def run(notes: list[VideoNote], style: Style = SMALL, *, audio: bool = True) -> tuple[VisualReading, float]:
        video = piano_video.render(notes, tmp_path / "video.mp4", style, audio=audio)
        sound = None
        if audio:
            media.decode_audio(video, tmp_path / "audio.wav", settings)
            sound = read_wav(tmp_path / "audio.wav")
        return read_video(video, settings, audio=sound), piano_video.lead_in(style)

    return run


def falling(notes: list[VideoNote], after: float = 3.0) -> list[VideoNote]:
    """The notes moved later, so that the first ones fall from the top of the picture."""
    return [VideoNote(n.pitch, n.start + after, n.end + after, n.hand) for n in notes]


def compare(reading: VisualReading, truth: list[VideoNote], lead: float) -> tuple[np.ndarray, np.ndarray]:
    """Pair every true note with the note read for it; (onset errors, end errors) in seconds."""
    assert len(reading.notes) == len(truth), reading.facts["notes"]
    left = sorted(reading.notes, key=lambda n: (n.pitch, n.start))
    right = sorted(truth, key=lambda n: (n.pitch, n.start))
    assert [n.pitch for n in left] == [n.pitch for n in right]
    assert [n.hand for n in left] == [n.hand for n in right]
    return (
        np.array([a.start - (b.start + lead) for a, b in zip(left, right, strict=True)]),
        np.array([a.end - (b.end + lead) for a, b in zip(left, right, strict=True)]),
    )


def test_every_note_of_a_piece_is_read_with_its_hand_and_timing(read, tmp_path: Path) -> None:
    truth = falling(piano_video.from_beats(synth.ode_to_joy(), synth.ODE_BPM))
    reading, lead = read(truth, Style())  # 720p, as most uploads are
    onsets, ends = compare(reading, truth, lead)
    assert np.abs(onsets).max() < 0.025 and np.abs(ends).max() < 0.04
    facts = reading.facts
    # 61 keys are shown, so the picture cannot name the octave: the audio does.
    assert facts["keyboard"]["lowest"] == "A1" and facts["keyboard"]["highest"] == "A6"
    assert facts["octave"]["source"] == "audio" and facts["octave"]["shift"] == 0
    assert facts["notes"]["hands_from"] == "colours"
    assert facts["key_check"]["bars_confirmed"] == len(truth) and facts["key_check"]["lit_without_a_bar"] == 0
    assert facts["roll"]["steady"] is True
    # The notes go on to the score writer as a MIDI file and as JSON.
    midi = write_midi(reading.notes, tmp_path / "performance.mid")
    reading.save(tmp_path / "visual.json")
    assert midi.stat().st_size > 0 and len(VisualReading.load(tmp_path / "visual.json").notes) == len(truth)


def test_a_filmed_keyboard_with_a_film_behind_the_notes(read) -> None:
    # Some tutorials film a real keyboard, whose keys do not light and over which the player's
    # hands move, and play a dimmed film behind the falling notes. However the film changes it is
    # not a note, and what the hands do to the keys says nothing about the notes.
    truth = falling(piano_video.from_beats(synth.ode_to_joy(), synth.ODE_BPM))
    style = Style(width=1920, height=1080, film=True, filmed=True, keys_light=False)
    reading, lead = read(truth, style)
    onsets, ends = compare(reading, truth, lead)
    assert np.abs(onsets).max() < 0.04 and np.abs(ends).max() < 0.05
    facts = reading.facts
    assert facts["roll"]["film_behind"] is True
    assert facts["key_check"]["keys_light"] is False and facts["notes"]["from_lit_keys_only"] == 0
    assert facts["notes"]["hands_from"] == "colours"
    assert facts["keyboard"]["lowest"] == "A1" and facts["keyboard"]["highest"] == "A6"


def test_notes_that_touch_are_told_apart_by_their_corners(read) -> None:
    # Repeated notes with no gap at all between their bars, on a black key, next to one long
    # note that must stay whole. At 1080p, where the rounded corners of a bar are several pixels.
    truth = [VideoNote(49, 0.5 * i, 0.5 * (i + 1), "left") for i in range(6)]
    truth += [VideoNote(64, 0.0, 3.0, "right"), VideoNote(45, 0.0, 3.0, "left")]
    style = Style(width=1920, height=1080, title_seconds=0.5, labels=False)
    reading, lead = read(falling(truth), style)
    onsets, ends = compare(reading, falling(truth), lead)
    assert np.abs(onsets).max() < 0.03 and np.abs(ends).max() < 0.04
    assert reading.facts["notes"]["separated_by_corners"] == 5


def test_every_seam_between_touching_notes_is_found(read) -> None:
    # Six touching bars on a white key and six on a black one. One of the five seams on the white
    # key leaves a notch at one corner only (it falls between two rows of the roll): too faint to
    # cut on outright, deep enough to count (``faint_notches``). It used to go unread.
    truth = [VideoNote(71, 0.5 * i, 0.5 * (i + 1), "right") for i in range(6)]
    truth += [VideoNote(49, 0.5 * i, 0.5 * (i + 1), "left") for i in range(6)]
    truth += [VideoNote(64, 0.0, 3.0, "right"), VideoNote(45, 0.0, 3.0, "left")]
    style = Style(width=1920, height=1080, title_seconds=0.5, labels=False)
    reading, _ = read(falling(truth), style)
    assert len(reading.notes) == len(truth)


@pytest.mark.xfail(strict=True, reason="known limit: see the test")
def test_a_note_name_on_a_narrow_bar_is_not_a_seam(read) -> None:
    # Pins a known limit. At 720p with 61 keys shown, the bar of a black key is about 20 pixels
    # wide and a note name printed on it covers its whole width; its strokes can then read as
    # the dark line between two touching bars, and the note is written as two. At 1080p the
    # same notes are read whole (the test after this one).
    truth = falling([VideoNote(p, 0.2 * i, 2.5 + 0.2 * i, "left") for i, p in enumerate((49, 54, 58, 61, 66, 70, 75))])
    reading, _ = read(truth, Style(title_seconds=0.5), audio=False)
    assert len(reading.notes) == len(truth)


def test_long_notes_with_their_names_printed_on_them_stay_whole(read) -> None:
    pitches = (45, 49, 54, 58, 61, 64, 66, 70, 71, 75)
    truth = falling(
        [VideoNote(p, 0.2 * i, 2.5 + 0.2 * i, "right" if p >= 60 else "left") for i, p in enumerate(pitches)]
    )
    reading, _ = read(truth, Style(width=1920, height=1080, title_seconds=0.5), audio=False)
    assert sorted(note.pitch for note in reading.notes) == sorted(pitches)
    assert reading.facts["notes"]["separated_by_corners"] == 0
    assert reading.facts["roll"]["film_behind"] is False  # a still background is not taken for a film


def test_a_black_key_between_two_sounding_white_keys(read) -> None:
    # C and D together, with and without the C sharp between them, whose bar lies over both.
    truth = [
        VideoNote(60, 0.0, 0.8, "right"), VideoNote(62, 0.0, 0.8, "right"),
        VideoNote(60, 1.2, 2.0, "right"), VideoNote(61, 1.2, 2.0, "right"), VideoNote(62, 1.2, 2.0, "right"),
        VideoNote(61, 2.4, 3.2, "right"), VideoNote(62, 2.4, 3.2, "right"),
        VideoNote(60, 3.6, 4.4, "right"), VideoNote(61, 3.9, 4.4, "right"),
    ]  # fmt: skip
    reading, lead = read(falling(truth), Style(labels=False, title_seconds=0.5))
    onsets, _ = compare(reading, falling(truth), lead)
    assert np.abs(onsets).max() < 0.05


def test_one_colour_for_both_hands_is_split_at_middle_c(read) -> None:
    truth = falling(piano_video.from_beats(synth.ode_to_joy(), synth.ODE_BPM))[:24]
    pink = Style(width=960, height=540, title_seconds=0.5, left=(253, 186, 254), left_black=(166, 111, 173))
    reading, _ = read(truth, pink)
    assert reading.facts["notes"]["hands_from"] == "pitch"
    assert all((note.hand == "right") == (note.pitch >= 60) for note in reading.notes)


def test_a_full_keyboard_names_the_octave_without_the_audio(read) -> None:
    truth = falling(
        [VideoNote(21, 0.0, 0.5, "left"), VideoNote(60, 0.5, 1.0, "right"), VideoNote(108, 1.0, 1.5, "right")]
    )
    truth += falling([VideoNote(60 + i, 2.0 + 0.3 * i, 2.25 + 0.3 * i, "right") for i in range(8)])
    reading, _ = read(
        truth, Style(width=1920, height=1080, lowest=21, highest=108, title_seconds=0.5, labels=False), audio=False
    )
    assert reading.facts["octave"]["source"] == "full keyboard"
    assert sorted(note.pitch for note in reading.notes) == sorted(note.pitch for note in truth)


def test_the_octave_of_a_cropped_keyboard_comes_from_the_audio(read) -> None:
    # Keys C3 to C6 only, and music high on them: the middle of the keyboard is the wrong guess.
    notes = [VideoNote(72 + [0, 4, 7, 12, 7, 4][i % 6], 0.4 * i, 0.4 * i + 0.36, "right") for i in range(24)]
    notes += [VideoNote(48 + [0, 7][i % 2], 0.8 * i, 0.8 * i + 0.7, "left") for i in range(12)]
    truth = falling(notes)
    reading, _ = read(truth, Style(width=960, height=540, lowest=48, highest=84, title_seconds=0.5))
    assert reading.facts["octave"]["source"] == "audio"
    assert sorted(note.pitch for note in reading.notes) == sorted(note.pitch for note in truth)


def test_a_video_without_a_keyboard_is_refused(settings: Settings, tmp_path: Path) -> None:
    import subprocess

    video = tmp_path / "noise.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=4",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    with pytest.raises(JobError) as refused:
        read_video(video, settings.model_copy(update={"sandbox": "basic"}))
    assert refused.value.code == "keyboard_not_found"
