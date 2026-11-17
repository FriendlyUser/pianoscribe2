"""The scoring harness itself must be trustworthy: perfect input scores 1, damaged input scores less."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.arrangement import Reference, evaluate
from eval.scoreio import read_musicxml

STEPS = {0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("E", -1), 4: ("E", 0), 5: ("F", 0), 6: ("F", 1), 7: ("G", 0), 8: ("A", -1), 9: ("A", 0), 10: ("B", -1), 11: ("B", 0)}  # fmt: skip
DIV = 4  # divisions per quarter


def note_xml(pitch: int, ticks: int, staff: int, chord: bool = False) -> str:
    step, alter = STEPS[pitch % 12]
    alter_xml = f"<alter>{alter}</alter>" if alter else ""
    return (
        f"<note>{'<chord/>' if chord else ''}<pitch><step>{step}</step>{alter_xml}<octave>{pitch // 12 - 1}</octave></pitch>"
        f"<duration>{ticks}</duration><voice>{staff}</voice><staff>{staff}</staff></note>"
    )


def rest_xml(ticks: int, staff: int) -> str:
    return f"<note><rest/><duration>{ticks}</duration><voice>{staff}</voice><staff>{staff}</staff></note>"


def write_score(
    path: Path, bars: list[tuple[list[tuple[float, float, int]], int | None]], meter=(6, 8), fifths=-1
) -> Path:
    """bars: [(melody [(start_eighths, length_eighths, pitch)], bass pitch or None)]"""
    bar_ticks = int(DIV * 4 * meter[0] / meter[1])
    parts = []
    for i, (melody, bass) in enumerate(bars):
        xml = ""
        if i == 0:
            xml += (
                f"<attributes><divisions>{DIV}</divisions><key><fifths>{fifths}</fifths></key>"
                f"<time><beats>{meter[0]}</beats><beat-type>{meter[1]}</beat-type></time><staves>2</staves></attributes>"
            )
        cursor = 0
        for start, length, pitch in melody:
            s, d = int(start * DIV / 2), int(length * DIV / 2)
            if s > cursor:
                xml += rest_xml(s - cursor, 1)
            xml += note_xml(pitch, d, 1)
            cursor = s + d
        if cursor < bar_ticks:
            xml += rest_xml(bar_ticks - cursor, 1)
        xml += f"<backup><duration>{bar_ticks}</duration></backup>"
        xml += note_xml(bass, bar_ticks, 2) if bass is not None else rest_xml(bar_ticks, 2)
        parts.append(f'<measure number="{i + 1}">{xml}</measure>')
    path.write_text(
        '<?xml version="1.0"?><score-partwise version="4.0"><part-list><score-part id="P1"><part-name>P</part-name>'
        f'</score-part></part-list><part id="P1">{"".join(parts)}</part></score-partwise>'
    )
    return path


TUNE = [
    ([(0, 1, 69), (1, 1, 74), (2, 1, 76), (3, 1, 77), (4, 2, 79)], 46),
    ([(1, 1, 81), (2, 1, 84), (3, 3, 76)], 45),
    ([(0, 2, 74), (2, 1, 77), (3, 1, 72), (4, 1, 74), (5, 1, 81)], 46),
    ([(0, 2, 81), (2, 1, 82), (3, 2, 81), (5, 1, 79)], 41),
] * 6


@pytest.fixture
def reference(tmp_path: Path) -> Reference:
    data = {
        "title": "test", "meter": [6, 8], "bars": len(TUNE),
        "key_regions": [{"bar": 1, "fifths": -1}],
        "melody": {str(i + 1): [[s, d, p] for s, d, p in melody] for i, (melody, _) in enumerate(TUNE)},
        "bass": {str(i + 1): [[0, bass % 12]] for i, (_, bass) in enumerate(TUNE)},
    }  # fmt: skip
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(data))
    return Reference.load(path)


def test_reader_recovers_notes_positions_and_ties(tmp_path: Path) -> None:
    score = read_musicxml(write_score(tmp_path / "s.xml", TUNE))
    assert len(score.bars) == len(TUNE) and score.bars[0].meter == (6, 8) and score.bars[0].fifths == -1
    assert float(score.bars[1].start) == 3.0 and float(score.length) == 3.0 * len(TUNE)
    first = [n for n in score.notes if n.bar == 0 and n.staff == 1]
    assert [(float(n.pos), n.pitch) for n in first] == [(0.0, 69), (0.5, 74), (1.0, 76), (1.5, 77), (2.0, 79)]
    assert [n.pitch for n in score.notes if n.bar == 0 and n.staff == 2] == [46]


def test_identical_score_is_perfect(tmp_path: Path, reference: Reference) -> None:
    result = evaluate(reference, read_musicxml(write_score(tmp_path / "s.xml", TUNE)))
    assert result.melody_recall == 1 and result.melody_precision == 1 and result.melody_octave_exact == 1
    assert result.rhythm_agreement == 1 and result.bar_placement == 1 and result.bass_accuracy == 1
    assert result.meter_correct and abs(result.bar_count_ratio - 1) < 1e-9 and result.key_accuracy == 1
    assert result.overall == pytest.approx(1.0)


def test_octave_doubling_and_extra_inner_notes_are_tolerated(tmp_path: Path, reference: Reference) -> None:
    up = [([(s, d, p + 12) for s, d, p in melody], bass - 12) for melody, bass in TUNE]
    result = evaluate(reference, read_musicxml(write_score(tmp_path / "s.xml", up)))
    assert result.melody_recall == 1 and result.melody_octave_exact == 0 and result.bass_accuracy == 1


def test_wrong_melody_scores_low(tmp_path: Path, reference: Reference) -> None:
    wrong = [
        ([(s, d, 60 + (i * 5 + k * 3) % 11) for k, (s, d, _) in enumerate(melody)], bass + 6)
        for i, (melody, bass) in enumerate(TUNE)
    ]
    result = evaluate(reference, read_musicxml(write_score(tmp_path / "s.xml", wrong)))
    assert result.melody_f1 < 0.45 and result.bass_accuracy == 0 and result.overall < 0.4


def test_shifted_bar_lines_keep_the_melody_but_lose_bar_placement(tmp_path: Path, reference: Reference) -> None:
    # Same notes, but every bar line is one eighth late (a pickup the reference does not have).
    flat = [(i * 6 + s, d, p) for i, (melody, _) in enumerate(TUNE) for s, d, p in melody]
    shifted: list[tuple[list[tuple[float, float, int]], int | None]] = [([], None) for _ in range(len(TUNE) + 1)]
    for start, length, pitch in flat:
        t = start + 1
        length = min(length, 6 - t % 6)
        shifted[int(t // 6)][0].append((t % 6, length, pitch))
    result = evaluate(reference, read_musicxml(write_score(tmp_path / "s.xml", shifted)))
    assert result.melody_recall > 0.95 and result.bar_placement < 0.1
    assert result.overall < 0.75


def test_wrong_meter_and_key_are_reported(tmp_path: Path, reference: Reference) -> None:
    result = evaluate(reference, read_musicxml(write_score(tmp_path / "s.xml", TUNE, fifths=2)))
    assert result.key_accuracy == 0 and result.meter_correct
    three_eight = []
    for melody, bass in TUNE:
        three_eight.append(([(s, min(d, 3 - s), p) for s, d, p in melody if s < 3], bass))
        three_eight.append(([(s - 3, d, p) for s, d, p in melody if s >= 3], None))
    halved = evaluate(reference, read_musicxml(write_score(tmp_path / "t.xml", three_eight, meter=(3, 8))))
    assert not halved.meter_correct and halved.melody_recall > 0.95


def test_note_values_twice_as_long_are_not_an_error(tmp_path: Path, reference: Reference) -> None:
    doubled = [([(s * 2, d * 2, p) for s, d, p in melody], bass) for melody, bass in TUNE]
    result = evaluate(reference, read_musicxml(write_score(tmp_path / "s.xml", doubled, meter=(6, 4))))
    assert result.melody_recall == 1 and result.rhythm_agreement == 1 and result.bar_placement == 1
    assert not result.meter_correct


def test_excerpt_reference_is_compared_with_the_opening_only(tmp_path: Path) -> None:
    opening = TUNE[:4]
    data = {
        "title": "opening", "meter": [6, 8], "bars": 4, "excerpt_only": True,
        "key_regions": [{"bar": 1, "fifths": -1}],
        "melody": {str(i + 1): [[s, d, p] for s, d, p in melody] for i, (melody, _) in enumerate(opening)},
        "bass": {str(i + 1): [[0, bass % 12]] for i, (_, bass) in enumerate(opening)},
    }  # fmt: skip
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(data))
    reference = Reference.load(path)
    # The candidate is the whole piece; everything after the opening is different music.
    rest = [([(0, 3, 60 + (i * 7) % 12), (3, 3, 62 + (i * 5) % 12)], 40) for i in range(20)]
    full = evaluate(reference, read_musicxml(write_score(tmp_path / "s.xml", opening + rest)))
    assert (
        full.melody_recall == 1 and full.rhythm_agreement == 1 and full.bar_placement == 1 and full.bass_accuracy == 1
    )
    wrong = evaluate(reference, read_musicxml(write_score(tmp_path / "w.xml", rest + opening)))
    assert wrong.melody_recall < 0.5


def test_a_pickup_bar_is_read_against_the_bar_line_that_follows(tmp_path: Path) -> None:
    from eval.arrangement import top_line

    head = f"<attributes><divisions>{DIV}</divisions><time><beats>3</beats><beat-type>4</beat-type></time></attributes>"
    pickup = f'<measure number="1" implicit="yes">{head}{note_xml(76, DIV, 1)}</measure>'
    full = f'<measure number="2">{note_xml(69, DIV * 3, 1)}</measure>'
    path = tmp_path / "pickup.xml"
    path.write_text(
        '<?xml version="1.0"?><score-partwise version="4.0"><part-list><score-part id="P1"><part-name>P</part-name>'
        f'</score-part></part-list><part id="P1">{pickup}{full}</part></score-partwise>'
    )
    line = top_line(read_musicxml(path))
    assert [(pitch, round(share, 3)) for _, pitch, share, _ in line] == [(76, 0.667), (69, 0.0)]
