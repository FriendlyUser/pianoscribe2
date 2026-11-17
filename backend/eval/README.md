# Grounded evaluation

Accuracy here is measured against ground truth, never judged by ear or against the system's own
output. There are three kinds of ground truth, from strictest to most musical:

| Ground truth | What it checks | Where |
|---|---|---|
| **Synthesized piano** with every note known | notes, key, meter, tempo, bar count, hands | `tests/synth.py`, always available |
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

- `eval/data/` (git-ignored): recordings. `eval/data/maestro/<name>.wav|.midi` and, for the
  arrangement benchmark, `eval/data/scene_shifts_there.wav`.
- `eval/private/` (git-ignored): references read from scores that are under copyright. A reference
  encodes the melody of the work, so it is kept out of the repository; the test that uses it
  (`tests/grounded/test_scene_shifts.py`) skips when it is missing.

To add a piece: write its reference JSON (public-domain scores go in `eval/references/`), put the
recording in `eval/data/` or `samples/`, and add a test next to the existing ones with thresholds a
little below what the current system scores, so a regression fails the build.
