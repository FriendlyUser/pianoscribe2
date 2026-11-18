"""Melody fidelity of a cover, and the facts read from a finished score."""

from __future__ import annotations

from pathlib import Path

import mido

from app.engrave.summary import key_name, summarize
from app.models.fidelity import choose_cover, melody_fidelity, read_note_onsets


def write_midi(path: Path, notes: list[tuple[float, int]], *, bpm: float = 120.0) -> Path:
    """notes: (onset seconds, pitch); each lasts 0.2 s."""
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    events = []
    for onset, pitch in notes:
        events.append((onset, mido.Message("note_on", note=pitch, velocity=80)))
        events.append((onset + 0.2, mido.Message("note_off", note=pitch, velocity=0)))
    events.sort(key=lambda e: e[0])
    last = 0
    for seconds, message in events:
        tick = round(mido.second2tick(seconds, 480, mido.bpm2tempo(bpm)))
        track.append(message.copy(time=tick - last))
        last = tick
    midi.save(str(path))
    return path


MELODY = [
    {"start": 1.0 + i * 0.5, "end": 1.4 + i * 0.5, "pitch": p} for i, p in enumerate([69, 74, 76, 77, 79, 76, 81, 84])
]
ANNOTATION = {"melody": MELODY}


def test_onsets_are_read_in_seconds(tmp_path: Path) -> None:
    onsets, pitches = read_note_onsets(write_midi(tmp_path / "a.mid", [(0.5, 60), (1.25, 64)], bpm=90))
    assert [round(float(t), 2) for t in onsets] == [0.5, 1.25] and pitches.tolist() == [60, 64]


def test_a_cover_that_plays_the_melody_in_time_scores_one(tmp_path: Path) -> None:
    faithful = [(n["start"] + 0.03, n["pitch"]) for n in MELODY] + [(n["start"], 48) for n in MELODY]
    assert melody_fidelity(write_midi(tmp_path / "f.mid", faithful), ANNOTATION) == 1.0
    octave_up = [(n["start"], n["pitch"] + 12) for n in MELODY]
    assert melody_fidelity(write_midi(tmp_path / "o.mid", octave_up), ANNOTATION) == 1.0


def test_right_notes_at_the_wrong_time_and_wrong_notes_both_lose(tmp_path: Path) -> None:
    rushed = [(1.0 + i * 0.25, n["pitch"]) for i, n in enumerate(MELODY)]  # twice as fast
    wrong = [(n["start"], n["pitch"] + 1) for n in MELODY]
    assert melody_fidelity(write_midi(tmp_path / "r.mid", rushed), ANNOTATION) < 0.5
    assert melody_fidelity(write_midi(tmp_path / "w.mid", wrong), ANNOTATION) == 0.0
    assert melody_fidelity(write_midi(tmp_path / "e.mid", [(0.0, 60)]), {"melody": []}) == 0.0


def test_the_most_faithful_candidate_is_chosen(tmp_path: Path) -> None:
    good = write_midi(tmp_path / "good.mid", [(n["start"], n["pitch"]) for n in MELODY])
    partial = write_midi(tmp_path / "partial.mid", [(n["start"], n["pitch"]) for n in MELODY[:4]])
    bad = write_midi(tmp_path / "bad.mid", [(n["start"], n["pitch"] + 1) for n in MELODY])
    best, scores = choose_cover([bad, partial, good], ANNOTATION)
    assert best == good and scores[0] < scores[1] < scores[2]


SCORE = """<?xml version="1.0"?><score-partwise version="3.1"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="1"><attributes><divisions>2</divisions><key><fifths>-1</fifths></key><time><beats>6</beats><beat-type>8</beat-type></time><staves>2</staves></attributes>
<direction><direction-type><metronome><beat-unit>quarter</beat-unit><beat-unit-dot/><per-minute>55</per-minute></metronome></direction-type></direction>
<note><pitch><step>A</step><octave>4</octave></pitch><duration>3</duration><tie type="start"/><staff>1</staff></note>
<note><pitch><step>D</step><octave>5</octave></pitch><duration>3</duration><staff>1</staff></note></measure>
<measure number="2"><note><pitch><step>A</step><octave>4</octave></pitch><duration>3</duration><tie type="stop"/><staff>1</staff></note>
<note><rest/><duration>3</duration><staff>1</staff></note></measure>
<measure number="3"><attributes><key><fifths>2</fifths></key></attributes>
<note><grace/><pitch><step>E</step><octave>5</octave></pitch><staff>1</staff></note>
<note><pitch><step>F</step><alter>1</alter><octave>5</octave></pitch><duration>6</duration><staff>1</staff></note></measure>
</part></score-partwise>"""


def test_summary_reads_key_meter_tempo_and_counts() -> None:
    summary = summarize(SCORE)
    assert summary.key == "F major" and summary.key_changes == 1
    assert summary.time_signature == "6/8" and summary.measures == 3
    assert summary.tempo_bpm == 55 and summary.tempo_unit == "dotted quarter"
    assert summary.notes == 3  # the tied continuation and the grace note are not counted
    assert summary.text == "F major · 6/8 · dotted quarter = 55 · 3 bars"


def test_key_names() -> None:
    assert key_name(0, None) == "C major" and key_name(2, "major") == "D major" and key_name(-3, "minor") == "C minor"
    assert key_name(0, "minor") == "A minor" and key_name(99, None) == "C# major"


def test_opening_fidelity_separates_a_rushed_opening_from_a_sound_one(tmp_path: Path) -> None:
    import mido

    from app.models.fidelity import opening_fidelity

    def midi(name: str, notes: list[tuple[float, int]]) -> Path:
        track, now = mido.MidiTrack(), 0.0
        track.append(mido.MetaMessage("set_tempo", tempo=500_000))
        events = sorted([(t, "note_on", p) for t, p in notes] + [(t + 0.1, "note_off", p) for t, p in notes])
        for when, kind, pitch in events:
            ticks = round((when - now) * 960)
            track.append(mido.Message(kind, note=pitch, velocity=80, time=ticks))
            now += ticks / 960
        file = mido.MidiFile(ticks_per_beat=480)
        file.tracks.append(track)
        path = tmp_path / name
        file.save(path)
        return path

    melody = [(0.53, 69), (0.90, 74), (1.26, 76), (1.62, 77), (1.98, 79), (2.70, 76), (3.07, 81), (3.43, 84)]
    annotation = {
        "downbeats": [0.54, 2.73, 4.91, 7.09],
        "melody": [{"start": t, "end": t + 0.3, "pitch": p} for t, p in melody],
    }
    sound = midi(
        "sound.mid",
        [(0.51, 46), (0.57, 69), (0.91, 74), (1.27, 76), (1.63, 77), (2.0, 79), (2.73, 76), (3.09, 81), (3.45, 84)],
    )
    rushed = midi(
        "rushed.mid", [(0.91, 46), (1.09, 74), (1.27, 76), (1.45, 77), (1.63, 77), (2.73, 76), (3.09, 81), (3.45, 84)]
    )
    assert opening_fidelity(sound, annotation) == 1.0
    assert opening_fidelity(rushed, annotation) < 0.8
    assert opening_fidelity(sound, {"downbeats": [0.54], "melody": annotation["melody"]}) is None  # too short to judge


def test_cjk_text_gets_a_font_that_has_the_glyphs() -> None:
    from app.engrave.pdf import with_cjk_font

    svg = (
        '<text><tspan class="text"><tspan font-size="607px">光るなら／Goose house</tspan></tspan></text>'
        '<text><tspan font-size="405px" font-family="Times">Piano</tspan></text>'
        '<text><tspan font-family="Times" font-size="405px">【Piano】夏影</tspan></text>'
    )
    fixed = with_cjk_font(svg, "Noto Serif CJK JP")
    assert '<tspan font-size="607px" font-family="Noto Serif CJK JP">光るなら／Goose house</tspan>' in fixed
    assert '<tspan font-size="405px" font-family="Times">Piano</tspan>' in fixed  # Latin text is left alone
    assert '<tspan font-size="405px" font-family="Noto Serif CJK JP">【Piano】夏影</tspan>' in fixed
    assert with_cjk_font(svg, None) == svg  # no such font installed: nothing to do
    assert (
        with_cjk_font("<text><tspan>Ode to Joy</tspan></text>", "Noto Serif CJK JP")
        == "<text><tspan>Ode to Joy</tspan></text>"
    )
