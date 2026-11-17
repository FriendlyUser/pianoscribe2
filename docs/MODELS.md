# Models: survey, benchmarks, licences

Surveyed in November 2026 for one question: which openly downloadable models give the most
accurate piano score from a recording? Candidates were compared on the grounded benchmarks in
[backend/eval](../backend/eval/README.md), on an RTX 4060 (8 GB).

## What is used

| Stage | Model | Source | Released |
|---|---|---|---|
| Stem separation and transcription of every instrument; piano transcription with dynamics | **tsumugi** | [code](https://github.com/anime-song/tsumugi), weights fetched by the code | 2026 |
| Beats, downbeats, per-bar time signatures, key, melody, chords | **SheetSage2-Refined** (built on SheetSage2 and MERT) | [anime-song/SheetSage2-Refined](https://huggingface.co/anime-song/SheetSage2-Refined) | 2026 |
| Song → piano arrangement | **tsumugi-piano-cover-v2** | [anime-song/tsumugi-piano-cover-v2](https://huggingface.co/anime-song/tsumugi-piano-cover-v2), [code](https://github.com/anime-song/tsumugi-piano-cover) | Oct-Nov 2026 |
| Performance → notation (MusicXML) | **piano_perf2score** | same repository and checkpoint as the cover model | Oct-Nov 2026 |

The three come from one author and are designed to chain: the cover model was fine-tuned on piano
covers paired with tsumugi transcriptions of the original audio, it generates on
SheetSage2-Refined's beat grid, and the score model reads its output on that same grid. Using them together avoids the mismatch between stages that limited
version 1, where a rule-based quantizer had to guess the beat grid of a transcription made by an
unrelated model.

Pinned versions: tsumugi `eba1c7d`, tsumugi-piano-cover `36f50c1`, SheetSage2-Refined `6b966e2`
(`backend/Dockerfile`, `backend/requirements/cover-env.txt`).

## Benchmarks

### Arranging a song: against a human piano arrangement

"Scene Shifts There" (band recording, 4 min 23 s). Reference: the first 83 of 119 bars of a
published piano arrangement, read by hand: 263 melody notes, 73 bass points, 6/8, key regions
F, D, C, F. Metrics are defined in [backend/eval/README.md](../backend/eval/README.md).

| System | Overall | Melody recall | Melody F1 | Rhythm | Bar placement | Bass | Meter | Key |
|---|---|---|---|---|---|---|---|---|
| v1 first release: Demucs + Kong piano model + rule-based notation | 0.404 | | | | | | | |
| v1 second pass: + Basic Pitch hybrid and arrangement rules | 0.733 | | | | | | | |
| tsumugi + cover v2 + perf2score, as shipped by their command line | 0.776 | | | | | | 3/4 ✗ | |
| + missing bar meters completed | 0.829 | | | | | | 6/8 ✓ | |
| + anticipated first chord settled, warm-up bars | 0.856 | 0.932 | 0.843 | 0.893 | 0.885 | 0.726 | 6/8 ✓ | 0.933 |
| + bar regularization, pickup notation (same arrangement as the row above) | 0.863 | 0.932 | 0.845 | 0.893 | 0.914 | 0.726 | 6/8 ✓ | 0.933 |
| **Full stack, from the YouTube link** (`docker compose up`, default settings) | **0.832** | 0.914 | 0.844 | 0.881 | 0.895 | 0.740 | 6/8 ✓ | 0.420 |
| Test image, from the local recording (`pytest -m grounded`) | 0.864 | 0.908 | 0.870 | 0.894 | 0.903 | 0.767 | 6/8 ✓ | 0.697 |

The rows above the last two were measured on one arrangement (one sample of the cover model) while
the fixes were developed; the last two are what the shipped pipeline produced, each from its own
sample. The cover is sampled, and samples differ: fourteen arrangements of this song (different
seeds, guidance strengths and batch positions) scored between 0.82 and 0.86 overall. Differences
of a few hundredths between two runs are therefore noise. The steps in the upper part of the table
are larger than that, were measured on the same arrangement, or were confirmed on a second piece.

The widest spread is in the key signatures (0.42 to 0.93 of bars): the song changes key three
times (F, D, C, F) and the score model does not always follow. The 8-bar D major passage is
written with one sharp in every arrangement that marks it at all, and the return to F for the
last 61 bars is missed in several, including the run from the YouTube link. Melody,
rhythm and bar placement vary by two or three points.

### Transcribing piano: against Disklavier MIDI

Six pieces from the MAESTRO v3 test split (Scarlatti, Schubert, Chopin, Debussy, Scriabin,
Rachmaninoff; 11.9 minutes). Note-level F1 from `mir_eval`: onset within 50 ms; the stricter score
also requires the note length within 20%, with reference lengths extended through the sustain
pedal (the MAESTRO convention).

| Model | Onset F1 | Onset + offset F1 |
|---|---|---|
| Kong et al. 2021, high-resolution piano transcription (version 1's model) | 0.981 | 0.864 |
| **tsumugi** (as shipped: piano label only, plus velocity model) | 0.978 | 0.874 |

On clean solo piano the two are equivalent. tsumugi is used because it is the same model the song
path needs, it comes with a velocity model, and it does not depend on a separate source-separation
step when other instruments are present.

### Solo piano, recording to written score

| Piece | Result |
|---|---|
| Ode to Joy, synthesized, every note known | G major, 4/4, 8 bars, 112 bpm, all 62 notes: exact |
| Für Elise, bars 1-9, 32-second clip against the printed score | overall 0.89: every note present, rhythm 1.00, bar placement 1.00, bass 1.00. Two errors of notation: 3/4 with doubled note values instead of 3/8, and a key signature of one sharp instead of none. Two middle Cs are on the lower staff where the edition has them in the right hand. |
| Für Elise, same bars from the full 75-second recording | overall 0.90: melody recall 1.00, rhythm 1.00, bass 1.00, key correct; some left-hand notes written on the upper staff (melody precision 0.78) |
| Mozart K. 545 (i), opening | C major, 4/4, opening bars match the printed score (checked by eye) |
| Satie, Gymnopédie No. 1 | D major, 3/4 (checked by eye) |

The score writer has a second mode that needs no beat grid. On the full Für Elise recording it
scored 0.42 where the beat-grid mode, before this repository's fixes, scored 0.76, so the beat
grid is always used.

## Candidates not used

| Model | What it does | Why not |
|---|---|---|
| **MuScriptor** (small / medium / large), [MuScriptor/muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium) | Multi-instrument audio-to-MIDI transcription | **Not benchmarked: the weights are gated.** Downloading needs a Hugging Face account that has accepted the model's terms, and a token. It is the most interesting untested alternative for the transcription stage; see below. Licence CC BY-NC 4.0. |
| Kong et al. high-resolution piano transcription | Piano transcription | Benchmarked above; equal on solo piano, no use on band recordings. |
| Transkun | Piano transcription | Not downloaded (not approved for this project), so not benchmarked. |
| Pop2Piano, PiCoGen2, AMT-APC, Etude | Song to piano cover | Earlier systems for the task tsumugi-piano-cover-v2 addresses. Not benchmarked here: none of them comes with a matching score-writing model, which is the stage that decided readability in version 1. |
| MIDI2ScoreTransformer | Performance MIDI to score | An alternative to perf2score. Not benchmarked; perf2score was chosen because it reads the same beat grid the cover model writes on. |
| Beat This | Beat and downbeat tracking | Tracks beats only. SheetSage2-Refined also gives per-bar time signatures, key, melody and chords, which the later stages need. |

### Where MuScriptor would fit

MuScriptor would replace tsumugi in the first stage. On the evidence so far that stage is not the
weak point: note accuracy on piano is 0.98, and the errors that remain in arrangements (bass
choices, the occasional extra top note) come from the arranging stage. A better transcription
could still help on dense band recordings, where there is no note-level ground truth in this
repository. To test it once access is granted:

1. accept the terms on the model page and create a read token;
2. transcribe the MAESTRO pieces and compare with `python -m eval.run transcription`;
3. feed its MIDI of "Scene Shifts There" to the arranging and notation stages in place of
   `transcription.mid` and compare with `python -m eval.run arrangement`.

Adopt it if step 3 improves the overall score by more than the 0.04 spread between samples,
measured over several samples each.

## Licences

| Component | Licence |
|---|---|
| tsumugi, tsumugi-piano-cover, SheetSage2-Refined **code** | MIT |
| tsumugi **weights** | no separate licence stated; the repository is MIT |
| tsumugi-piano-cover-v2 (cover and score models) and SheetSage2-Refined **weights** | **CC BY-NC 4.0** (attribution, non-commercial) |
| MAESTRO v3 (evaluation data, not redistributed here) | CC BY-NC-SA 4.0 |
| `samples/fur_elise_opening.mp3` | CC BY-SA 4.0 (see [samples/README.md](../samples/README.md)) |

Every score, song or solo piano, passes through SheetSage2-Refined and the score model, so any
deployment of this system is non-commercial. Scores produced for
personal use, study or research are within the licence; selling the service or its output is not.
Check each model card for the current terms before relying on this summary.
