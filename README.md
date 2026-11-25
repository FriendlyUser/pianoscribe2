# PianoScribe 2

Turn a recording into piano sheet music, with accuracy measured against ground truth.

Upload an audio/video file (MP3, WAV, MP4, MOV) or paste a YouTube link:

- a **song** (band, vocals, orchestra, game soundtrack) is *arranged* for piano: its melody, bass
  and harmony rewritten as a part two hands can play;
- a **solo piano** recording is *transcribed*: the notes that were played, written down;
- a **piano tutorial video** (notes falling onto a keyboard, "Synthesia style") is *read*: every
  note, its length and the hand that plays it are measured in the picture and written down as shown.

Each way you get an engraved two-staff score to play back in the browser, print, or download as
PDF, standalone HTML, MusicXML and MIDI.

This is a ground-up rebuild of [PianoScribe](https://github.com/FriendlyUser/pianoscribe). The
service around the music (API, job queue, sandboxed ingestion, engraving, web UI) is the same; every
musical stage is new, built on current models from Hugging Face, and held to tests against
reference scores and reference recordings.

## How accurate is it?

Measured, not estimated. The full method is in [backend/eval/README.md](backend/eval/README.md) and
the model comparison in [docs/MODELS.md](docs/MODELS.md).

**Arranging a song**, against a published human piano arrangement of the same piece
("Scene Shifts There", 119 bars, 6/8, three key changes):

| | v1 first release | v1 second pass | **PianoScribe 2** |
|---|---|---|---|
| Overall agreement | 0.40 | 0.73 | **0.85** from the YouTube link (0.84 to 0.88 over 10 arrangements) |

The arrangement is sampled, so the same song can come out a little differently; ten arrangements
of the benchmark song landed between 0.84 and 0.88. The run through the full stack from the
YouTube link scored 0.85:

| Melody notes present | Rhythm | Place in the bar | Bass | Time signature | Bars under the right key signature | First bar |
|---|---|---|---|---|---|---|
| 92% | 83% | 87% | 77% | 6/8, correct | 93%: both lasting key changes followed; an 8-bar passage in D major is written with accidentals | all five notes of the opening phrase, on the downbeat |

**Transcribing solo piano:**

| Ground truth | Result |
|---|---|
| MAESTRO v3 test recordings (Disklavier MIDI), six pieces | note onset F1 0.978 (onset and length: 0.874) |
| Synthesized piece with every note known | all notes, key, meter, tempo and bar count exact |
| Für Elise, 32-second recording against the printed score (bars 1-9) | overall 0.95: the melody, rhythm, bar lines, bass and key signature correct; written in 3/4 with doubled note values instead of 3/8 |

**Reading a piano tutorial video.** The notes are measured in the frames, not heard, and written
down by rule, so the score is held to a stricter standard: the notes shown, all of them, no others.

| Ground truth | Result |
|---|---|
| Rendered video of a piece with every note known (720p, H.264), reader alone | 62 of 62 notes, each with the right key and hand; onsets within 7 ms, ends within 11 ms |
| The same video through the whole pipeline, against the piece | overall 1.00: every note in its bar, at its beat, with its length, on the staff of its hand; key, meter and tempo exact |

Real videos have no reference score, so each reading is cross-checked three ways instead (the
keys lighting up, the attacks in the audio, the pitch of the audio); the two videos the reader was
developed on, and what those checks can and cannot show, are in
[backend/eval/README.md](backend/eval/README.md#piano-tutorial-videos). In short: a 1080p video
read to 929 notes with every check passing; lower resolutions and screen recordings read with a
few notes in a hundred in doubt. See also [Known limitations](ARCHITECTURE.md#known-limitations).

A field test on four more recordings from anime and visual novels (two songs, two piano solos)
had key, meter and tempo right on all four; the method and numbers are in
[backend/eval/README.md](backend/eval/README.md#field-test-2026-11-17).

One arrangement benchmark and a handful of piano pieces are a small evaluation. They are enough to
catch regressions and to compare approaches; they are not a claim about every genre. See
[Known limitations](ARCHITECTURE.md#known-limitations).

## Licence of the models: non-commercial

Two of the three models this project downloads (SheetSage2-Refined and tsumugi-piano-cover-v2,
which includes the score writer) publish their weights under **CC BY-NC 4.0**, and every score
goes through both. You may therefore use this system for personal, research and other
non-commercial purposes only. The code in this repository does not change that.
Details and attributions: [docs/MODELS.md](docs/MODELS.md#licences).

## Quick start

Prerequisites: Docker 24+ with Compose v2, an NVIDIA GPU with 8 GB or more and the NVIDIA Container
Toolkit, about 25 GB of free disk (the worker image is 11 GB, and 4.5 GB of weights download on the
first job).

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Then open <http://localhost:8080>. Try `samples/fur_elise_opening.mp3` with "This is already solo
piano" ticked.

Without a GPU, `docker compose up --build` runs the same code on the CPU. It works, but a song then
takes a large multiple of the GPU time; treat it as a way to check the installation.

Measured on an RTX 4060 (8 GB) with the weights already downloaded: a 30-second piano clip in
about 1.5 minutes, a 4.4-minute song from a YouTube link in 6 to 9 minutes. Most of that is the model
stages loading and running one after another.

Check the installation from inside the worker:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm worker python -m app.cli doctor
```

### What runs

| Service   | Purpose                                                               |
|-----------|-----------------------------------------------------------------------|
| `web`     | nginx: the React UI, and a reverse proxy for `/api` (port 8080)        |
| `api`     | FastAPI: validation, streamed uploads, job submission, progress (SSE)  |
| `worker`  | Dramatiq worker: ingestion sandbox, model stages, engraving            |
| `janitor` | Purges raw media, stale work files and expired results                 |
| `redis`   | Job queue, job state, progress pub/sub (append-only persistence)       |

Storage is two named volumes: `app-data` (uploads, work files, finished scores) and `model-cache`
(model weights).

## Desktop app

The same app runs on Windows (x64, NVIDIA GPU recommended) and on Macs with Apple Silicon as a
desktop app with the models included, so nothing leaves your computer. It adds a **Models** panel:
which device to run on, the solo-piano transcriber (tsumugi, or MuScriptor as an optional download),
and how closely and densely songs are arranged. Builds come from the GitHub Action "Desktop app"
as a draft release; see [desktop/README.md](desktop/README.md).

## Tests

```bash
# Unit tests (a few seconds; no models needed)
docker compose --profile test run --rm --build tests

# Grounded accuracy tests: the real models against reference scores and recordings (about 10 minutes)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile test run --rm tests pytest -m grounded
```

To check any finished job against its recording, without a reference score:

```bash
docker compose --profile test run --rm -v "$PWD/my-job:/job:ro" tests python -m eval.field_check /job/recording.wav /job
```

The unit tests include the piano-video reader against rendered videos whose notes are known
(they need ffmpeg, which the tests image has). The grounded suite always runs the
synthesized-piano, piano-video and Für Elise tests. Two more need material
that is not in the repository and are skipped without it:

```bash
# MAESTRO note accuracy: fetch six test recordings (~125 MB, CC BY-NC-SA 4.0) into backend/eval/data
docker compose --profile test run --rm --user "$(id -u):$(id -g)" -v "$PWD/backend/eval/data:/out" tests python -m eval.fetch_maestro /out/maestro
```

The song-arrangement benchmark needs the recording at `backend/eval/data/scene_shifts_there.wav`
and its reference at `backend/eval/private/scene_shifts_there.json`. The reference is read from a
published arrangement of a copyrighted work, so it is not committed.

## Using it without the web UI

```bash
# A local file or a YouTube link straight to ./out (no queue, no Redis). Add --solo for solo
# piano, or --video for a piano tutorial video (an MP4 or MOV, or a YouTube link, whose picture
# is to be read).
mkdir -p out
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm -T \
  -v "$PWD/samples:/in:ro" worker sh -c \
  'python -m app.cli transcribe /in/fur_elise_opening.mp3 /tmp/out --solo >&2 && tar -C /tmp/out -cf - .' \
  | tar -xf - -C out

# HTTP API
curl -X POST --data-binary @samples/fur_elise_opening.mp3 \
  "http://localhost:8080/api/jobs/upload?filename=fur_elise_opening.mp3&solo_piano=true"
curl -X POST -H 'Content-Type: application/json' \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID", "solo_piano": false}' \
  http://localhost:8080/api/jobs/youtube
curl -X POST --data-binary @tutorial.mp4 \
  "http://localhost:8080/api/jobs/upload?filename=tutorial.mp4&piano_video=true"
curl -N http://localhost:8080/api/jobs/JOB_ID/events          # live progress (Server-Sent Events)
curl -O "http://localhost:8080/api/jobs/JOB_ID/artifacts/score.pdf?download=true"
```

**On GitHub Actions.** The workflow "Piano tutorial video"
([.github/workflows/piano-video.yml](.github/workflows/piano-video.yml)) takes a YouTube link to a
piano tutorial video and runs the same CLI on a GitHub runner, on its processor: Actions tab >
Piano tutorial video > Run workflow. The score comes back as the run's artifact `score` (kept for
3 days; in a public repository, any signed-in GitHub user can download it while it is kept), with
the reading's checks in the run's summary. If YouTube asks the runner to sign in, put a
`cookies.txt` for youtube.com into the repository secret `YOUTUBE_COOKIES`.

Artifacts per job: `score.pdf`, `score.html`, `score.musicxml`, `score.mid` (the score as MIDI),
`performance.mid` (the piano part before notation, with original timing and dynamics) and, for
songs, `transcription.mid` (every instrument the transcriber heard).

## Configuration

Environment variables with the `PS_` prefix; copy `.env.example` to `.env`. The most useful:

| Variable | Default | Meaning |
|---|---|---|
| `PS_WEB_PORT` / `PS_WEB_BIND` | `8080` / `127.0.0.1` | Host port and interface of the UI (localhost only by default; there is no login) |
| `PS_MAX_UPLOAD_MB` | `200` | Upload size limit (enforced while streaming) |
| `PS_MAX_DURATION_SECONDS` | `600` | Longest audio accepted |
| `PS_YOUTUBE_ENABLED` | `true` | Allow YouTube links |
| `PS_WORKER_MEMORY` | `12g` | Memory ceiling of the worker container |
| `PS_MODEL_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `PS_TSUMUGI_WINDOW_BATCH_SIZE` | `2` | Transcription batch size; 2 fits an 8 GB card, raise it on larger ones |
| `PS_COVER_SOURCE_CFG` | `2.0` | How closely an arrangement follows the original (higher: closer) |
| `PS_COVER_DENSITY` | `1.0` | Note density of arrangements |
| `PS_COVER_SEED` | `0` | Arrangements are sampled; a fixed seed makes them repeatable |
| `PS_COVER_OPENING_RETRIES` | `2` | Times an arrangement is generated again when its opening does not play the detected melody |
| `PS_VIDEO_SCORE_WRITER` | `exact` | Piano videos: `exact` writes the notes read down by rule; `model` hands them to the score model |
| `PS_YOUTUBE_VIDEO_HEIGHT` | `1080` | Largest picture fetched for a YouTube piano video (touching notes need the detail) |
| `PS_PAGE_SIZE` | `A4` | `A4` or `Letter` |

The full list is in [backend/app/config.py](backend/app/config.py).

## Operations

```bash
docker compose logs -f worker                                   # structured JSON logs, one job_id per job
docker compose exec worker python -m app.cli dead-letters        # jobs that exhausted their retries
docker compose exec worker python -m app.cli requeue JOB_ID      # retry a failed job
docker compose exec janitor python -m app.storage.janitor --once # run a retention pass now
```

## Repository layout

```
backend/
  app/
    api/        HTTP routes, schemas, SSE
    ingest/     magic-byte checks, streamed uploads, sandboxed ffmpeg and yt-dlp
    jobs/       job model, dedupe keys, Redis store, Dramatiq queue, idempotent runner
    models/     model stages as subprocesses: transcription, arrangement, notation
    visual/     piano tutorial videos: keyboard, falling notes, hands, checks (no model)
    engrave/    Verovio rendering, PDF, standalone HTML, score summary
    pipeline/   stage orchestration with checkpoints and timings
    storage/    artifact store, retention janitor
  model_scripts/  drivers that run inside the model environments, the score fixes, and the
                  rule-based writer for notes read from a video
  eval/         grounded evaluation: metrics, references, data fetchers
  tests/        unit/ and grounded/
  requirements/ *.in sources, hash-pinned *.txt locks, the cover model's environment
docs/MODELS.md  model survey, benchmarks, licences
frontend/       React + TypeScript UI, nginx config
samples/        test recordings
```

## Responsible use

Only transcribe recordings you have the right to use. YouTube downloads are subject to YouTube's
terms; the feature can be switched off with `PS_YOUTUBE_ENABLED=false`. An arrangement of a
copyrighted song is a derivative of that song.
