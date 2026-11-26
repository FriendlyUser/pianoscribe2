"""Render a piano-tutorial video from notes that are known, to test the reader against.

The picture imitates the tutorial videos the reader is meant for: a keyboard along the
bottom, bars with rounded corners and dark outlines falling onto it, a note name printed
at the foot of each bar, a darker shade for the bars of black keys, a white flare where
a note lands, keys that light while they sound, octave lines behind the notes, a title
card before the roll and a fade into it. It is then encoded with H.264 like any upload.

Some tutorials are made differently, and ``Style`` can imitate them too: a dimmed film
playing behind the notes (``film``), and a filmed keyboard, shaded by the light of the room,
whose keys do not light and over which the player's hands move (``filmed``). Videos of a
real piano filmed from above often draw both hands' notes alike, as white blocks in a glow
of colour (``glow``), with sparks rising where they land (``sparks``) and a strip of
coloured light along the top of the keys (``strip_light``); and the camera does not film
the keys as evenly as the lanes of the notes are laid (``stretch``).

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
    film: bool = False  # a dimmed film plays behind the notes instead of the still background
    afterglow: bool = False  # a screen recording's smear: a faint band a few rows past the end of every bar
    filmed: bool = False  # a filmed keyboard: shaded keys that do not light, and the player's hands over them
    glow: bool = False  # both hands' notes alike: white blocks with round ends, in a glow of ``right``
    sparks: bool = False  # sparks rise from where the notes land
    strip_light: bool = False  # a strip of light in ``right`` along the top of a filmed keyboard, tinting the keys
    stretch: float = 0.0  # how much wider a filmed keyboard's keys are at its ends than in its middle (0.04: 4%)
    key_brightness: float = 1.0  # a filmed keyboard in a dark room: 0.6
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
    if style.glow:
        return _glowing_roll(notes, style, layout, speed, height)
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
        if style.afterglow:
            # Not background, and nowhere near the colour of a bar: a third of the way to the darkest outline.
            faint = tuple(b + 34 for b in style.background)
            draw.rectangle([box[0] + 1, box[1] - 11, box[2] - 1, box[1] - 7], fill=faint)
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


def _block_span(pitch: int, layout: _Layout) -> tuple[float, float]:
    """Where a glowing block falls: a white key's as wide as the key, a black key's half as wide."""
    if pitch in layout.black:
        x0, x1 = layout.black[pitch]
        centre = (x0 + x1) / 2
        return centre - 0.25 * layout.key_width, centre + 0.25 * layout.key_width
    x0, x1 = layout.white[pitch]
    return x0 + 0.5, x1 - 0.5


def _glowing_roll(notes: list[VideoNote], style: Style, layout: _Layout, speed: float, height: int) -> np.ndarray:
    """The roll of a video that draws every note as a white block with round ends in a glow of colour.

    Drawn four times the size and scaled down, so that the blocks' edges fall between pixels as
    they do in a video scaled to a small picture.
    """
    scale = 4
    size = (style.width * scale, height * scale)
    cores = Image.new("L", size, 0)
    draw = ImageDraw.Draw(cores)
    for note in notes:
        x0, x1 = _block_span(note.pitch, layout)
        low = height - 1 - note.start * speed
        high = height - 1 - note.end * speed
        box = [x0 * scale, (high + 1) * scale, x1 * scale, low * scale]
        if box[3] - box[1] < 2 * scale:
            continue
        draw.rounded_rectangle(
            box, radius=min((box[2] - box[0]) / 2, (box[3] - box[1]) / 2, 0.3 * layout.key_width * scale), fill=255
        )
    cores = cores.resize((style.width, height), Image.Resampling.BOX)
    core = np.asarray(cores, dtype=np.float32)[..., None] / 255.0
    halo = (
        np.asarray(cores.filter(ImageFilter.GaussianBlur(0.35 * layout.key_width)), dtype=np.float32)[..., None] / 255.0
    )
    colour = np.array(style.right, dtype=np.float32)
    picture = np.array(style.background, dtype=np.float32) + 1.6 * halo * colour
    picture = picture * (1.0 - core) + np.array([252.0, 244.0, 255.0]) * core
    return np.clip(picture, 0, 255).astype(np.uint8)


def _sparks(frame: np.ndarray, sounding: list[VideoNote], style: Style, layout: _Layout, index: int) -> None:
    """Sparks above the notes that sound: bright specks that drift up on their own, not with the roll."""
    top = style.keyboard_top
    for note in sounding:
        x0, x1 = _block_span(note.pitch, layout)
        rng = np.random.default_rng(note.pitch * 7919 + int(note.start * 1000))
        for _ in range(14):
            born, rise = rng.integers(0, 40), rng.uniform(1.5, 5.0)
            drift, side = rng.uniform(-0.6, 0.6), rng.uniform(x0 - layout.key_width, x1 + layout.key_width)
            age = (index - born) % 40
            x, y = int(side + drift * age), int(top - 4 - rise * age)
            if 0 <= x < style.width - 1 and 0 <= y < top - 1:
                frame[y : y + 2, x : x + 2] = (255, 235, 255)


def _stretched(keyboard: np.ndarray, stretch: float) -> np.ndarray:
    """``keyboard`` as a camera with a wide lens sees it: its keys wider at the ends than in the middle."""
    width = keyboard.shape[1]
    along = (np.arange(width) + 0.5) / width
    # The column of the even keyboard seen at each column of the picture: squeezed at the ends.
    source = (along - stretch * np.sin(2 * np.pi * along) / (2 * np.pi)) * width - 0.5
    low = np.clip(np.floor(source).astype(int), 0, width - 1)
    high = np.clip(low + 1, 0, width - 1)
    share = np.clip(source - low, 0.0, 1.0)[None, :, None]
    return (keyboard[:, low] * (1 - share) + keyboard[:, high] * share).astype(np.uint8)


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


def _filmed_keyboard(style: Style, layout: _Layout) -> np.ndarray:
    """A keyboard as a camera sees it: white keys shaded towards the sides of the picture, black
    keys whose sloping sides catch the light, grey gaps between the keys, a darker front edge."""
    top = style.keyboard_top
    depth = style.height - top
    rows = np.arange(depth, dtype=np.float32)[:, None]
    columns = np.arange(style.width, dtype=np.float32)[None, :]
    # Measured on a filmed tutorial: about a fifth of each row of the keyboard is neither clearly
    # bright nor clearly dark (shading towards the sides, reflections, the gaps between keys).
    level = 215.0 + 15.0 * np.minimum(1.0, rows / (0.7 * depth))
    level = np.where(rows > 0.94 * depth, 190.0, level)
    level = level * (1.0 - 0.35 * ((columns - style.width / 2) / (style.width / 2)) ** 2)
    image = Image.fromarray(np.repeat(level[..., None], 3, axis=-1).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for _, x1 in layout.white.values():
        draw.line([(round(x1), 0), (round(x1), depth)], fill=(95, 95, 95), width=max(1, round(0.08 * layout.key_width)))
    for x0, x1 in layout.black.values():
        draw.rectangle([round(x0), 4, round(x1) - 1, round(0.65 * depth)], fill=(25, 25, 28))
        bevel = 0.1 * (x1 - x0)  # the sloping sides of a black key catch the light
        draw.rectangle([round(x0), 4, round(x0 + bevel), round(0.65 * depth)], fill=(105,) * 3)
        draw.rectangle([round(x1 - bevel), 4, round(x1) - 1, round(0.65 * depth)], fill=(105,) * 3)
    draw.rectangle([0, 0, style.width, 3], fill=(140, 20, 24))
    keys = np.asarray(image).astype(np.float32) * style.key_brightness
    if style.strip_light:
        # A strip of lights along the top of the keys: a bright line, and its colour over the keys below it.
        colour = np.array(style.right, dtype=np.float32)
        reach = np.exp(-rows / (0.12 * depth))[..., None]
        keys = keys * (1.0 - 0.5 * reach) + 0.75 * reach * colour * (keys / 255.0) ** 0.5
        keys[:3] = colour
    keys = np.clip(keys, 0, 255).astype(np.uint8)
    return _stretched(keys, style.stretch) if style.stretch else keys


def _hands(
    keyboard: np.ndarray, notes: list[VideoNote], now: float, layout: _Layout, stretch: float = 0.0
) -> np.ndarray:
    """The player's hands over a filmed keyboard, each above the notes it plays (or played last)."""
    image = Image.fromarray(keyboard)
    draw = ImageDraw.Draw(image)
    depth = keyboard.shape[0]
    for hand in ("left", "right"):
        own = [note for note in notes if note.hand == hand]
        if not own:
            continue
        sounding = [note for note in own if note.start <= now < note.end]
        played = [note for note in own if note.start <= now]
        chosen = sounding or played[-1:] or own[:1]
        centre = float(np.mean([sum(_span(note.pitch, layout)) / 2 for note in chosen]))
        if stretch:  # the hand is over the key where the camera shows it
            width = keyboard.shape[1]
            centre = _seen_at(centre, width, stretch)
        reach = 1.6 * layout.key_width
        draw.ellipse([centre - reach, 0.3 * depth, centre + reach, 1.25 * depth], fill=(226, 184, 154))
    return np.asarray(image)


def _seen_at(column: float, width: int, stretch: float) -> float:
    """Where a camera with a wide lens shows column ``column`` of an even keyboard (see ``_stretched``)."""
    along = (np.arange(width) + 0.5) / width
    source = (along - stretch * np.sin(2 * np.pi * along) / (2 * np.pi)) * width - 0.5
    return float(np.interp(column, source, np.arange(width)))


def _film(style: Style, index: int) -> np.ndarray:
    """The film behind the notes at frame ``index``: dimmed shots of soft colour and sharp-edged
    shapes, a new shot every second and a half, each panning slowly sideways."""
    shot_frames = round(1.5 * style.fps)
    shot = _shot(style.width, style.keyboard_top, index // shot_frames)
    pan = 2 * (index % shot_frames)
    return shot[:, pan : pan + style.width]


_SHOTS: dict[tuple[int, int, int], np.ndarray] = {}


def _shot(width: int, height: int, number: int) -> np.ndarray:
    key = (width, height, number)
    if key not in _SHOTS:
        rng = np.random.default_rng(number)
        field = Image.fromarray(rng.uniform(5, 95, size=(4, 7, 3)).astype(np.uint8)).resize(
            (width + 200, height), Image.Resampling.BICUBIC
        )
        draw = ImageDraw.Draw(field)
        for _ in range(4):
            x, y, r = rng.uniform(0, width + 200), rng.uniform(0, height), rng.uniform(0.04, 0.15) * width
            draw.ellipse([x - r, y - 1.5 * r, x + r, y + 1.5 * r], fill=tuple(int(c) for c in rng.uniform(15, 105, 3)))
        _SHOTS[key] = np.asarray(field)
    return _SHOTS[key]


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
    # With a film behind the notes, whatever the roll drew over its background (bars, their names
    # and outlines, the octave lines) lies over the film.
    drawn = (roll != np.array(style.background, dtype=np.uint8)).any(axis=-1, keepdims=True) if style.film else None
    filmed_keys = _filmed_keyboard(style, layout) if style.filmed else None
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
        if drawn is not None:
            frame[:top] = np.where(drawn[offset : offset + top], frame[:top], _film(style, index))
        sounding = [note for note in notes if note.start <= now < note.end]
        lit = {note.pitch: _colour(note, style) for note in sounding} if style.keys_light else {}
        if filmed_keys is not None:
            frame[top:] = _hands(filmed_keys, notes, now, layout, style.stretch)
        else:
            frame[top:] = _keyboard(style, layout, lit)
        if style.sparks:
            _sparks(frame, sounding, style, layout, index)
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
