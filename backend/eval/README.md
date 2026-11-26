# Grounded evaluation

Accuracy here is measured against ground truth, never judged by ear or against the system's own
output. There are three kinds of ground truth, from strictest to most musical:

| Ground truth | What it checks | Where |
|---|---|---|
| **Synthesized piano** with every note known | notes, key, meter, tempo, bar count, hands | `tests/synth.py`, always available |
| **Rendered piano video** of the same notes | the picture reader: key, onset, length and hand of every note | `tests/piano_video.py`, always available |
| **Disklavier MIDI** recorded with the audio (MAESTRO v3) | note onsets and lengths of the piano transcriber | `eval/data/maestro/` (downloaded) |
| **A printed score** of the piece | melody, rhythm, bar lines, bass, key, meter of the written score | `eval/references/*.json`, `eval/private/*.json` |

## Running

```bash
# inside the tests image (see the repository README)
pytest -m grounded                       # every grounded test whose material is present
python -m eval.fetch_maestro             # fetch the six MAESTRO test pieces (~125 MB, CC BY-NC-SA 4.0)
python -m eval.maestro_report            # per-piece note scores of the shipped piano transcriber

# score files directly
python -m eval.run arrangement eval/references/fur_elise_opening.json mine=score.musicxml other=other.musicxml
python -m eval.run transcription eval/data/maestro OUTPUT_DIR tsumugi=tsumugi
```

Tests skip, rather than fail, when their recording or reference is absent.

## Checking a result without a reference

Most recordings have no reference score. Two checks still apply to any finished job:

```bash
python -m eval.field_check RECORDING.wav ARTIFACT_DIR    # ARTIFACT_DIR: the job's downloaded files
```

- **Note fidelity** (`eval/fidelity.py`): the written score against the performance MIDI it was
  written from, as multisets of pitches. The score model may respell and regroup, but it should
  neither invent notes nor lose them. Asserted in the grounded tests (97% both ways).
- **Harmony agreement** (`eval/field_check.py`): the pitch-class content of the audio, from a
  plain spectrum, against the notes of the MIDI in half-second windows. No model is involved on
  the audio side, and the same notes shifted in time or transposed give the baseline.

Neither can say whether each melody note is right; they confirm that the result follows the
recording and that the notation is faithful to what was transcribed.

### Field test, 2026-11-17

Four recordings picked from YouTube searches, run through the full stack from their links.

| Recording | Path | Published key, meter, tempo | Harmony agreement (transposed baseline) | Notes written vs performed |
|---|---|---|---|---|
| "Hikaru Nara" (Your Lie in April), 4:19 | song | A major, 4/4, 160: all matched | 0.76 (0.35) | +1% |
| "Tori no Uta" (Air), 6:08 | song | B minor, 4/4, 122: matched (120); the change to G♯ minor followed | 0.70 (0.36) | +0% |
| "Natsukage" (Air), piano cover, 3:44 | solo | E♭ major, 4/4: matched | 0.86 (0.34) | -2% |
| "Merry-Go-Round" (Howl's Moving Castle), piano solo, 4:28 | solo | G minor, 3/4: matched | 0.89 (0.34) | +1% (was **+39%**) |

The first run of this test found two defects, both since fixed. The 203-bar solo piece was written
with 39% more notes than were played, because the score model degrades on long solo input; long
solo pieces are now written in sections. And Japanese titles were blank in the PDF for want of a
font. For scale on the harmony column: the benchmark arrangement that scores 0.87 against its
human reference gives 0.72 here, and the Für Elise clip 0.84.

## Piano tutorial videos

The reader (`app/visual`) is tested against videos rendered from known notes
(`tests/piano_video.py`): a keyboard, falling bars with rounded corners and dark outlines, a note
name printed on each bar, a flare where notes land, lit keys, a title card and a fade, encoded
with H.264. `tests/unit/test_visual_reader.py` asserts the notes that come out against the notes
that went in, and `tests/grounded/test_piano_video.py` holds the finished score to the piece.

| | Result |
|---|---|
| Reader alone, "Ode to Joy" at 720p (62 notes) | 62 read, every key and hand right; onset error -1.9 ms on average, 6.8 ms at most; end error 10.6 ms at most |
| Whole pipeline on that video, score against the piece | overall 1.00 (melody, rhythm, bar placement, bass, key, meter); each hand on its own staff; 8 bars; quarter = 112 |

Three real videos were used in development. None has a reference score, so these are
cross-checks, not accuracy figures. Their outputs are not in the repository: all are other
people's arrangements of copyrighted songs.

| | Rendered tutorial, 1080p60, 4:21 | Synthesia screen recording, 720p30, 4:16 | Filmed keyboard, film behind the notes, 1080p60 (AV1), 4:16 |
|---|---|---|---|
| Keyboard read | A1 to A6 (61 keys); the "C4" printed on the keyboard is where middle C was put | A1 to B5 (51 keys), note names printed on every key | F1 to A6 (65 keys), filmed, the player's hands over it |
| Notes read | 929 (right hand 622, left 307) | 1415 (right hand 549, left 866) | 2251 (hands split at middle C: right 1332, left 919) |
| Separated from a touching note | 154 | 58 (12 of them at faint seams) | 671 |
| Bars whose key was seen lit | 929 of 929 | 1412 of 1415 | the keys do not light: not used |
| Keys lit with no bar | 0 | 0 | not used |
| Octave, by agreement with the audio | as placed: 0.70, against 0.44 an octave down and 0.38 up | as placed: 0.56, against 0.44 and 0.35 | as placed: 0.66, against 0.53 and 0.42 |
| Attacks in the audio (jumps in its rise) with no note onset within 60 ms | 0 of 689 | 6 of 363, none beyond 150 ms | 21 of 656 (counted the earlier way, as below) |
| Frame-to-frame motion | even | uneven (0 to 10 rows a frame): followed frame by frame | even |
| Score | E major, 4/4 with two bars of 2/4, quarter = 106, then 118, 120, 124 and 65; every note read is in it and nothing else; every note on its hand's staff | F major, 4/4, quarter = 100, the tempo the recording itself displays; every note read is in it and nothing else; 41 notes of chords too wide for one hand on the other staff | E major, 4/4, quarter = 105, 113 bars; every note read is in it, and 99.6% of its notes are notes read |
| The same notes through the score model instead | 936 notes written for 929, three on the wrong staff, the opening triplets written as sixteenths | not run | not run |

What this does and does not establish. For the first video every check agrees, and one more that
only it allows: each bar carries exactly one printed note name. The first video read again at
720p gives 929 notes of which 5 are split and 5 merged against the 1080p reading (see the limits
pinned in `tests/unit/test_visual_reader.py`).

The second video's "strong attacks with no note" in the table above were mostly no attacks. They
were counted as peaks of the audio's rise above its mean plus two standard deviations, and a
ringing chord's rise ripples at about that level for some tenths of a second. Counted as jumps in
the rise instead (`app.visual.audio.attack_times`), the second video has 363 attacks while notes
are played, 357 with a note onset within 60 ms and every one within 150 ms; the first has 689,
every one with an onset within 60 ms.

What was wrong in the second reading was found in the frames instead. Four notes were read as two
each, one piece a sliver of 20 to 50 ms: joined back. And in runs of repeated notes whose bars
touch, some seams left a notch at one corner only, or one blurred to half the strip, and two notes
were read as one: twelve such notes are now separated (`faint_notches`), each a second block with
its own printed name in the frames. The second video now reads 1415 notes (right hand 549, left
866), 1412 with their key seen lit; the first is unchanged at 929. Not every such seam need be
found: one that leaves no notch at all at 720p would still be read as one long note, and nothing
measured here would show it.

How the scores are written was measured on the same two videos. Each note's onset was compared
with the grid it is written on:

| | Rendered tutorial, 1080p60 | Synthesia screen recording, 720p30 |
|---|---|---|
| Steady stretches found | 5: 106, 118, 120, 124 and 65 to the minute (the tracker's bars gave 105 and one change, to 120, at the wrong bar) | 8, all within one beat a minute of 100; one tempo mark |
| Onsets from the nearest twelfth of the fitted beat | median 1.6 ms; 2 of 850 over 10 ms | median 5 ms; 95% within 19 ms |
| Beats taken from the notes / left to the tracker | 449 / 35 (before the first note, after the last) | 371 / 34 |
| Bars | 114: 4/4 with two bars of 2/4; the slowing bar before the last section keeps its four beats on the notes and is marked *rit.* | 97, all 4/4 |
| Tempo marks | 106 at bar 1, 118 at 34, 120 at 54, 124 at 70, 65 at 111 | 100 |
| Triplets written | in the introduction only, where the notes sit on thirds of the beat to 2 ms | none (the earlier score had them wherever a note was read a frame late) |
| Notes moved to the other staff | 0 | 41: the upper part of chords two octaves wide drawn in the left-hand colour |
| Every note read is in the score, and nothing else | yes | yes |

These are measurements of consistency, not of correctness against a reference: neither video has
a published score here to compare with. What they do show is that the beat the notes are written
on is the one the notes keep, and that the same figure is written the same way each time.

Reading the third video at first changed how every video is read, and the first two then read
worse: the strip higher above the keyboard and a looser test for the keyboard's rows cost the
first video three notes (repeated B4s whose seams went unseen) and gave the second its title
banner as 30 notes at once and its notes 87 ms early (the audio's timing no longer sharp enough
to apply). Both changes now apply only where they are needed (a film behind the notes; no drawn
keyboard found), banners are dropped wherever they are read, and the first two videos read as
before: every note with the same key and hand, starts and ends within 4 ms (first) and 16 ms
(second) of the earlier reading.

## Score references

A reference is a JSON file transcribed by hand from a printed score:

```json
{
  "title": "...", "meter": [6, 8], "bars": 119,
  "key_regions": [{"bar": 1, "fifths": -1}, {"bar": 23, "fifths": 2}],
  "melody": {"1": [[0, 2, 69], [2, 1, 74]]},
  "bass": {"1": [[0, 2]]},
  "unsure_bars": [57],
  "excerpt_only": false
}
```

`melody` maps a bar number to `[start, duration, MIDI pitch]` triples of the top line, in eighth
notes from the bar line. `bass` maps a bar to `[position, pitch class]` of the lowest note at the
main beats. Bars listed in `unsure_bars` count half. `excerpt_only` marks a reference that covers
just the opening of the piece.

`eval/arrangement.py` compares a MusicXML score with a reference:

| Metric | Meaning |
|---|---|
| `melody_recall` | reference melody notes found in the candidate's top line, in order (any octave), by banded sequence alignment |
| `melody_precision` | candidate top-line notes, over the stretch the reference covers, that belong to the melody |
| `melody_octave_exact` | of the found notes, the share in the reference's octave |
| `rhythm_agreement` | consecutive melody notes whose spacing matches the reference |
| `bar_placement` | found notes that sit at the reference's position within the bar |
| `bass_accuracy` | reference bass points where the candidate's lowest left-hand note has the same pitch class |
| `meter_correct`, `bar_count_ratio`, `key_accuracy` | time signature, number of bars, key signature per region |
| `overall` | 0.35 melody F1 + 0.20 rhythm + 0.20 bar placement + 0.15 bass + 0.05 meter + 0.05 key |

A candidate written in note values twice as long or short as the reference is not penalised on
melody or rhythm (that is caught by `meter_correct`). The harness has its own tests
(`tests/unit/test_eval_harness.py`): a perfect score must get 1.0, and each kind of damage must
lower exactly the metric that describes it.

## Material that is not in the repository

- `eval/data/` (git-ignored): recordings. `eval/data/maestro/<name>.wav|.midi`, for the
  arrangement benchmark `eval/data/scene_shifts_there.wav`, and for the long-solo test any solo
  piano recording of more than about three minutes as `eval/data/long_solo.wav`.
- `eval/private/` (git-ignored): references read from scores that are under copyright. A reference
  encodes the melody of the work, so it is kept out of the repository; the test that uses it
  (`tests/grounded/test_scene_shifts.py`) skips when it is missing.

To add a piece: write its reference JSON (public-domain scores go in `eval/references/`), put the
recording in `eval/data/` or `samples/`, and add a test next to the existing ones with thresholds a
little below what the current system scores, so a regression fails the build.
