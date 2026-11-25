"""Render a piano-tutorial video from notes that are known, to test the reader against.

The picture imitates the tutorial videos the reader is meant for: a keyboard along the
bottom, bars with rounded corners and dark outlines falling onto it, a note name printed
at the foot of each bar, a darker shade for the bars of black keys, a white flare where
a note lands, keys that light while they sound, octave lines behind the notes, a title
card before the roll and a fade into it. It is then encoded with H.264 like any upload.

    from tests import piano_video
    piano_video.render(notes, path, piano_video.Style())

Nothing here is used by the application; like ``tests.synth`` it only builds fixtures.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tests import synth

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
BLACK = (1, 3, 6, 8, 10)
# How far a black key sits off the boundary between its white neighbours, in white-key widths.
BLACK_OFFSET = {1: -0.08, 3: 0.09, 6: -0.14, 8: 0.0, 10: 0.16}


@dataclass(frozen=True)
class VideoNote:
    pitch: int
    start: float  # seconds from the first frame of the roll
    end: float
    hand: str = "right"  # "right" or "left"


@dataclass(frozen=True)
class Style:
    width: int = 1280
    height: int = 720
    fps: int = 30
    lowest: int = 33  # A1: the lowest and highest keys shown (white keys)
    highest: int = 93  # A6
    seconds_on_screen: float = 2.9  # how long a note takes to fall
    right: tuple[int, int, int] = (253, 186, 254)
    right_black: tuple[int, int, int] = (166, 111, 173)
    left: tuple[int, int, int] = (223, 245, 76)
    left_black: tuple[int, int, int] = (67, 214, 10)
    background: tuple[int, int, int] = (20, 20, 20)
    labels: bool = True
    flare: bool = True
    keys_light: bool = True
    title_seconds: float = 1.0  # a title card before the roll appears
    fade_frames: int = 6
    crf: int = 26  # H.264 quality; streaming sites are about this lossy

    @property
    def keyboard_top(self) -> int:
        return int(self.height * 0.69)


def _is_black(pitch: int) -> bool:
    return pitch % 12 in BLACK


@dataclass(frozen=True)
class _Layout:
    white: dict[int, tuple[float, float]]
    black: dict[int, tuple[float, float]]
    key_width: float


def _layout(style: Style) -> _Layout:
    whites = [p for p in range(style.lowest, style.highest + 1) if not _is_black(p)]
    width = style.width / len(whites)
    white = {p: (i * width, (i + 1) * width) for i, p in enumerate(whites)}
    black = {}
    for p in range(style.lowest, style.highest + 1):
        if _is_black(p) and p - 1 in white and p + 1 in white:
            centre = white[p - 1][1] + BLACK_OFFSET[p % 12] * width
            black[p] = (centre - 0.31 * width, centre + 0.31 * width)
    return _Layout(white, black, width)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _colour(note: VideoNote, style: Style) -> tuple[int, int, int]:
    if note.hand == "left":
        return style.left_black if _is_black(note.pitch) else style.left
    return style.right_black if _is_black(note.pitch) else style.right


def _span(pitch: int, layout: _Layout) -> tuple[float, float]:
    if pitch in layout.black:
        x0, x1 = layout.black[pitch]
        return x0 + 0.03 * layout.key_width, x1 - 0.03 * layout.key_width
    return layout.white[pitch]


def _roll(notes: list[VideoNote], style: Style, layout: _Layout, speed: float, seconds: float) -> np.ndarray:
    """The whole piano roll as one tall picture; row 0 is the latest moment."""
    top = style.keyboard_top
    height = int(np.ceil(seconds * speed)) + top + 4
    image = Image.new("RGB", (style.width, height), style.background)
    draw = ImageDraw.Draw(image)
    for pitch, (x0, _) in layout.white.items():
        if pitch % 12 == 0:
            draw.line([(round(x0), 0), (round(x0), height)], fill=(54, 54, 54), width=1)
        elif pitch % 12 == 5:
            draw.line([(round(x0), 0), (round(x0), height)], fill=(26, 26, 26), width=1)
    radius = max(2, round(0.15 * layout.key_width))
    font = _font(max(8, round(0.62 * layout.key_width)))
    small = _font(max(7, round(0.42 * layout.key_width)))
    # White keys first, so that the bar of a black key lies over its neighbours as in the videos.
    for note in sorted(notes, key=lambda n: (_is_black(n.pitch), n.start)):
        x0, x1 = _span(note.pitch, layout)
        low = height - 1 - note.start * speed  # the bar's lower edge: where the note is struck
        high = height - 1 - note.end * speed
        fill = _colour(note, style)
        outline = tuple(int(0.22 * c) for c in fill)
        box = [round(x0), round(high) + 1, round(x1) - 1, round(low)]  # up to, not onto, the next bar
        if box[3] - box[1] < 2:
            continue
        r = min(radius, (box[3] - box[1]) // 2, (box[2] - box[0]) // 2)
        draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=max(1, round(0.03 * layout.key_width)))
        if style.labels:
            text = NAMES[note.pitch % 12]
            use = small if _is_black(note.pitch) else font
            box_text = draw.textbbox((0, 0), text, font=use)
            tall = box_text[3] - box_text[1]
            if box[3] - box[1] > tall + 6:
                x = (box[0] + box[2]) / 2 - (box_text[2] - box_text[0]) / 2 - box_text[0]
                y = box[3] - tall - 5 - box_text[1]
                draw.text((x, y), text, font=use, fill=(253, 253, 253), stroke_width=2, stroke_fill=outline)
    return np.asarray(image)


def _keyboard(style: Style, layout: _Layout, lit: dict[int, tuple[int, int, int]]) -> np.ndarray:
    top = style.keyboard_top
    depth = style.height - top
    image = Image.new("RGB", (style.width, depth), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    for pitch, (x0, x1) in layout.white.items():
        if pitch in lit:
            draw.rectangle([round(x0) + 1, 4, round(x1) - 1, depth], fill=lit[pitch])
        draw.line([(round(x1), 0), (round(x1), depth)], fill=(90, 90, 90), width=max(1, round(0.03 * layout.key_width)))
    for pitch, (x0, x1) in layout.black.items():
        colour = lit.get(pitch, (12, 12, 12))
        draw.rectangle([round(x0), 4, round(x1) - 1, round(0.65 * depth)], fill=colour)
    draw.rectangle([0, 0, style.width, 3], fill=(221, 13, 19))
    return np.asarray(image)


def render(
    notes: list[VideoNote],
    path: Path,
    style: Style = Style(),
    *,
    audio: bool = True,
    tail: float = 1.0,  # noqa: B008
) -> Path:
    """Write ``notes`` as a tutorial video (with their sound, unless ``audio`` is False). Returns ``path``.

    A note's ``start`` is the moment its bar touches the keyboard, counted from when the roll
    appears; add ``lead_in(style)`` for its time in the finished video.
    """
    layout = _layout(style)
    top = style.keyboard_top
    speed = top / style.seconds_on_screen
    seconds = max(note.end for note in notes) + tail
    roll = _roll(notes, style, layout, speed, seconds)
    title_frames = round(style.title_seconds * style.fps)
    frames = round(seconds * style.fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{style.width}x{style.height}", "-r", str(style.fps), "-i", "pipe:0",
    ]  # fmt: skip
    wav = path.with_suffix(".sound.wav")
    if audio:
        _sound(notes, wav, style.title_seconds, seconds)
        command += ["-i", str(wav), "-c:a", "aac", "-b:a", "128k"]
    command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(style.crf), "-pix_fmt", "yuv420p", str(path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    card = Image.new("RGB", (style.width, style.height), (8, 8, 8))
    ImageDraw.Draw(card).rectangle([0, 0, style.width, style.height // 4], fill=(240, 220, 180))
    ImageDraw.Draw(card).text(
        (style.width // 3, style.height // 2), "SPECIAL THANKS", font=_font(40), fill=(110, 220, 240)
    )
    for _ in range(title_frames):
        encoder.stdin.write(card.tobytes())
    for index in range(frames):
        now = index / style.fps
        offset = roll.shape[0] - top - round(now * speed)
        frame = np.empty((style.height, style.width, 3), dtype=np.uint8)
        frame[:top] = roll[offset : offset + top]
        sounding = [note for note in notes if note.start <= now < note.end]
        lit = {note.pitch: _colour(note, style) for note in sounding} if style.keys_light else {}
        frame[top:] = _keyboard(style, layout, lit)
        if style.flare and sounding:
            glow = Image.new("L", (style.width, style.height), 0)
            draw = ImageDraw.Draw(glow)
            for note in sounding:
                x0, x1 = _span(note.pitch, layout)
                centre, reach = (x0 + x1) / 2, 0.7 * layout.key_width
                draw.ellipse([centre - reach, top - 0.8 * reach, centre + reach, top + 0.2 * reach], fill=255)
            glow = glow.filter(ImageFilter.GaussianBlur(0.12 * layout.key_width))
            mask = np.asarray(glow, dtype=np.float32)[..., None] / 255.0
            frame = (frame * (1 - mask) + 255.0 * mask).astype(np.uint8)
        if index < style.fade_frames:
            frame = (frame * ((index + 1) / (style.fade_frames + 1))).astype(np.uint8)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg could not encode the test video")
    wav.unlink(missing_ok=True)
    return path


def lead_in(style: Style = Style()) -> float:  # noqa: B008
    """Seconds from the start of the video to the moment the roll appears."""
    return round(style.title_seconds * style.fps) / style.fps


def _sound(notes: list[VideoNote], path: Path, before: float, seconds: float) -> None:
    rate = synth.SAMPLE_RATE
    audio = np.zeros(int((before + seconds + 0.5) * rate))
    for note in notes:
        wave = synth._note_wave(note.pitch, max(0.05, note.end - note.start), 92 if note.hand == "right" else 66, rate)
        start = int((before + note.start) * rate)
        audio[start : start + len(wave)] += wave[: len(audio) - start]
    peak = float(np.max(np.abs(audio))) or 1.0
    synth.write_wav(path, (audio / peak * 0.8).astype(np.float32), rate)


def from_beats(
    notes: list[synth.SynthNote], bpm: float, *, right_velocity: int = 92, gate: float = 0.9
) -> list[VideoNote]:
    """``tests.synth`` notes (in beats) as video notes; the melody's velocity marks the right hand."""
    beat = 60.0 / bpm
    return [
        VideoNote(
            note.pitch,
            note.start_beat * beat,
            (note.start_beat + note.duration_beats * gate) * beat,
            "right" if note.velocity == right_velocity else "left",
        )
        for note in notes
    ]
