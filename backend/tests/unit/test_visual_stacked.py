"""The reader of small pictures and filmed keyboards (app.visual.stacked) against videos whose notes are known.

A video of a real piano filmed from above, 480 lines tall: all 88 keys, 16 pixels to a white
key, dim and tinted by a strip of lights, wider at the ends than in the middle as a camera
sees them; the player's hands over them; every note a white block in a glow of one colour,
with sparks where it lands. ``tests/piano_video.py`` renders one from a list of notes, and
the notes read back are compared with the ones that went in.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.ingest import media
from app.ingest.audio import read_wav
from app.visual import stacked
from app.visual.hands import LEFT, RIGHT, divide_hands, hands_by_divide
from app.visual.keyboard import filmed_rows, find_keyboard, measure_filmed_keyboard
from app.visual.reader import VisualReading, read_video
from tests import piano_video, synth
from tests.piano_video import Style, VideoNote

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
FILMED = Style(
    width=854,
    height=480,
    lowest=21,
    highest=108,
    filmed=True,
    keys_light=False,
    glow=True,
    sparks=True,
    strip_light=True,
    stretch=0.04,
    key_brightness=0.6,
    labels=False,
    right=(150, 60, 255),
    title_seconds=0.5,
)


@pytest.fixture
def read(settings: Settings, tmp_path: Path) -> Callable[..., tuple[VisualReading, float]]:
    settings = settings.model_copy(update={"sandbox": "basic"})

    def run(notes: list[VideoNote], style: Style = FILMED, *, audio: bool = True) -> tuple[VisualReading, float]:
        video = piano_video.render(notes, tmp_path / "video.mp4", style, audio=audio)
        sound = None
        if audio:
            media.decode_audio(video, tmp_path / "audio.wav", settings)
            sound = read_wav(tmp_path / "audio.wav")
        return read_video(video, settings, audio=sound), piano_video.lead_in(style)

    return run


def falling(notes: list[VideoNote], after: float = 3.0) -> list[VideoNote]:
    return [VideoNote(n.pitch, n.start + after, n.end + after, n.hand) for n in notes]


def pair(reading: VisualReading, truth: list[VideoNote]) -> list[tuple[object, VideoNote]]:
    assert len(reading.notes) == len(truth), reading.facts["notes"]
    read_notes = sorted(reading.notes, key=lambda n: (n.pitch, n.start))
    true_notes = sorted(truth, key=lambda n: (n.pitch, n.start))
    assert [n.pitch for n in read_notes] == [n.pitch for n in true_notes]
    return list(zip(read_notes, true_notes, strict=True))


def test_a_small_filmed_keyboard_is_read_from_one_picture_of_its_roll(read) -> None:
    truth = falling(piano_video.from_beats(synth.ode_to_joy(), synth.ODE_BPM))
    reading, lead = read(truth)
    pairs = pair(reading, truth)
    onsets = np.array([a.start - (b.start + lead) for a, b in pairs])
    ends = np.array([a.end - (b.end + lead) for a, b in pairs])
    assert np.abs(onsets).max() < 0.02 and np.abs(ends).max() < 0.03
    # Both hands' notes are drawn alike: the hands are told apart by where the player's hands are.
    assert [a.hand for a, _ in pairs] == [b.hand for _, b in pairs]
    facts = reading.facts
    assert facts["notes"]["hands_from"] == "hands"
    assert facts["keyboard"]["filmed"] is True
    assert facts["keyboard"]["lowest"] == "A0" and facts["keyboard"]["highest"] == "C8"
    assert facts["octave"]["source"] == "full keyboard"
    # The lanes are laid evenly; the camera does not film the keys evenly, and they part ways at the ends.
    assert abs(facts["lanes"]["white_step"] - 854 / 52) < 0.05
    assert facts["lanes"]["furthest_from_filmed_key"] > 3.0
    # Everything lit on the picture is a block of a key that is on, but for a fringe at blocks' ends.
    check = facts["picture_check"]
    assert check["most_rows_in_a_row_unexplained"] <= 4
    assert check["lit_without_a_key"] < 0.02 * check["lit_columns"]
    assert facts["audio_check"]["attacks_without_a_note"] == 0


SCALE = [VideoNote(pitch, 0.25 * i, 0.25 * i + 0.2, "right") for i, pitch in enumerate(range(55, 73))]


def test_blocks_side_by_side_are_told_apart_by_where_they_begin_and_end(read) -> None:
    # A black key struck with the white key beside it, on either side, covers part of that
    # key's block and is covered by it in part; three neighbouring white keys make one wide
    # block; and a black key alone must not be taken for the white key under it. (After a
    # scale: where each key's blocks fall is measured on blocks that fall alone.)
    chords = [(63, 64), (61, 60), (60, 62, 64), (66,), (65, 67), (70, 71), (58, 59, 61), (56,), (55, 57)]
    truth = falling(
        SCALE
        + [VideoNote(pitch, 5.0 + 0.8 * i, 5.6 + 0.8 * i, "right") for i, chord in enumerate(chords) for pitch in chord]
    )
    reading, lead = read(truth)
    pairs = pair(reading, truth)
    assert max(abs(a.start - (b.start + lead)) for a, b in pairs) < 0.03
    # Four times two white keys a tone apart sounded together, for 0.6 seconds each: the black key
    # between them (C sharp, D sharp, F sharp, G sharp) could not have been seen then.
    assert 2.2 < reading.facts["notes"]["seconds_a_black_key_was_out_of_sight"] < 2.6


def test_blocks_that_meet_end_to_end_are_separate_notes(read) -> None:
    # A note struck again as its block ends: the two blocks touch, and only their rounded corners
    # show where one stops and the next begins. Next to them, a long note that must stay whole.
    touch = 0.01  # seconds: a block ends on the row the next begins on
    again = [VideoNote(64, 5.0 + 0.4 * i, 5.4 + 0.4 * i + touch, "right") for i in range(5)]
    black = [VideoNote(70, 5.0 + 0.3 * i, 5.3 + 0.3 * i + touch, "right") for i in range(4)]
    truth = falling([*SCALE, *again, *black, VideoNote(48, 5.0, 7.2, "left")])
    reading, lead = read(truth)
    pairs = pair(reading, truth)
    assert max(abs(a.start - (b.start + lead)) for a, b in pairs) < 0.03
    assert reading.facts["notes"]["separated_by_corners"] == 7


def test_a_filmed_keyboard_in_uneven_light_is_found_and_named() -> None:
    # A dim keyboard under a strip of coloured light, its keys wider at the ends: fixed grey levels
    # find no keyboard in it. Each stretch's own two tones do, and the black keys name every key.
    style = FILMED
    layout = piano_video._layout(style)
    keys = piano_video._filmed_keyboard(style, layout)
    frame = np.zeros((style.height, style.width, 3), dtype=np.uint8)
    frame[style.keyboard_top :] = keys
    grey = frame.astype(np.float32).mean(axis=-1)
    rows = filmed_rows(grey)
    assert rows is not None and abs(rows[0] - style.keyboard_top) <= 6
    keyboard = measure_filmed_keyboard(grey, rows)
    assert keyboard is not None and keyboard.octave_certain
    assert keyboard.lowest == 21 and keyboard.highest == 108 and len(keyboard.keys) == 88
    # Each key is where the camera shows it: up to five columns from where an even keyboard has it.
    seen = {key.pitch: key.center for key in keyboard.keys}
    for pitch in (24, 60, 61, 84, 106):
        even = sum(piano_video._span(pitch, layout)) / 2
        assert abs(seen[pitch] - piano_video._seen_at(even, style.width, style.stretch)) < 1.5
    view = find_keyboard(np.repeat(frame[None], 4, axis=0))
    assert view.filmed and view.keyboard.lowest == 21


def test_the_divide_between_two_hands_follows_them() -> None:
    # Two hands, eighty columns wide, the left one leaping back and forth; for a while they touch.
    frames, width = 300, 400
    skin = np.zeros((frames, width), dtype=np.uint8)
    left_at = np.where((np.arange(frames) // 40) % 2 == 0, 60, 150)
    right_at = np.full(frames, 250)
    right_at[200:240] = 232  # the right hand comes up against the left
    for frame in range(frames):
        skin[frame, left_at[frame] : left_at[frame] + 80] = 30
        skin[frame, right_at[frame] : right_at[frame] + 80] = 30
        if 200 <= frame < 240 and left_at[frame] == 150:
            skin[frame, 228:234] = 8  # two fingers side by side: little skin where the hands meet
    divide = divide_hands(skin, 30)
    for frame in range(frames):
        assert left_at[frame] + 78 <= divide[frame] <= right_at[frame] + 4, frame
    hands = hands_by_divide([100.0, 200.0, 240.0, 300.0], [10.0, 50.0, 60.0, 100.0], divide)
    assert hands == [LEFT, LEFT, RIGHT, RIGHT]


def test_the_block_light_is_read_where_every_block_is_bright() -> None:
    # White blocks in a violet glow are read in the channel the glow lacks, so that the glow is
    # no part of a block; blocks that go from one colour to another, in the channel both have.
    keyboard = find_keyboard(_frames_with_keys())
    rows, columns = keyboard.keyboard.top, 854
    still = np.zeros((6, 480, columns, 3), dtype=np.uint8)
    still[:] = _frames_with_keys()[:1]
    white = still.copy()
    for frame in range(6):
        top = 40 + 30 * frame
        white[frame, top : top + 60, 300:340] = (120, 50, 200)  # the glow
        white[frame, top + 10 : top + 50, 310:330] = (252, 244, 255)  # the block
    light = stacked.block_light(white, keyboard)
    assert light.channels == (0, 1, 2) and light.level > 230
    read = light.read(white[:, :rows])
    assert read[0, 60, 320] > 230 and read[0, 45, 305] < 100  # block; glow
    turning = still.copy()
    for frame in range(6):
        top = 40 + 30 * frame
        colour = (40, 230, 250) if frame < 3 else (250, 90, 250)
        turning[frame, top : top + 50, 310:330] = colour
    assert stacked.block_light(turning, keyboard).channels == (2,)


def _frames_with_keys() -> np.ndarray:
    layout = piano_video._layout(FILMED)
    frame = np.zeros((FILMED.height, FILMED.width, 3), dtype=np.uint8)
    frame[FILMED.keyboard_top :] = piano_video._filmed_keyboard(FILMED, layout)
    return np.repeat(frame[None], 6, axis=0)


def test_notes_that_ring_on_after_their_keys_are_up_are_heard_to() -> None:
    # Thirty keys touched for a tenth of a second each. Damped as the keys come up, nothing is
    # left of a note a third of a second later; with the pedal down, most of it still is. The
    # score writer is told which (a note the pedal holds is written out to the next).
    from app.visual.audio import RINGS, rings_on

    rate = synth.SAMPLE_RATE
    scale = (60, 62, 64, 65, 67, 69, 71)
    notes = [(0.5 + 0.25 * k, 0.6 + 0.25 * k, scale[k % 7] + 12 * (k % 2)) for k in range(30)]

    def sound(seconds: float) -> np.ndarray:
        audio = np.zeros(int(9.5 * rate))
        for start, _, pitch in notes:
            wave = synth._note_wave(pitch, seconds, 80, rate)
            first = int(start * rate)
            audio[first : first + len(wave)] += wave[: len(audio) - first]
        return (audio / np.abs(audio).max() * 0.8).astype(np.float32)

    damped = rings_on(notes, sound(0.1), rate)
    held = rings_on(notes, sound(1.5), rate)
    assert damped is not None and held is not None
    assert damped < 0.5 * RINGS < RINGS < held
    assert rings_on(notes[:5], sound(1.5), rate) is None  # too few notes to say
