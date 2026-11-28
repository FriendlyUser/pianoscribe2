# syntax=docker/dockerfile:1.7
# Targets:
#   api     - HTTP API and janitor (small, no models)
#   worker  - ingestion, model environments, engraving (GPU strongly recommended)
#   dev     - worker + test tooling (used by the `tests` compose service)
#   video   - piano-tutorial videos only, on the processor: the CLI from a YouTube link or a file
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --home-dir /home/app app \
 && mkdir -p /data /models \
 && chown app:app /data /models
WORKDIR /app

# ---------------------------------------------------------------- api ----
FROM base AS api
COPY requirements/api.txt /tmp/requirements.txt
RUN uv pip install --require-hashes -r /tmp/requirements.txt
COPY app ./app
USER app
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-server-header"]

# ------------------------------------------------------------- models ----
# The model programs at pinned commits, each with the environment its authors lock.
FROM base AS models
ARG TSUMUGI_REF=eba1c7d05e4d6a53cc4e8916d742d9623a3fdc38
ARG COVER_REF=36f50c1d76ac36b11c7c9de701d231649ba17680
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# Use the image's own interpreter, so the environments still work when copied into the worker stage.
ENV UV_NO_CACHE=0 UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy UV_SYSTEM_PYTHON=0 UV_PYTHON_DOWNLOADS=never
WORKDIR /opt/models
RUN git init -q tsumugi && cd tsumugi \
 && git fetch -q --depth 1 https://github.com/anime-song/tsumugi.git ${TSUMUGI_REF} && git checkout -q FETCH_HEAD \
 && rm -rf .git \
 && uv sync --locked --no-dev --extra stem --extra sheetsage \
 && rm -rf /tmp/uv-cache
COPY requirements/cover-env.txt /tmp/cover-env.txt
RUN git init -q tsumugi-piano-cover && cd tsumugi-piano-cover \
 && git fetch -q --depth 1 https://github.com/anime-song/tsumugi-piano-cover.git ${COVER_REF} && git checkout -q FETCH_HEAD \
 && rm -rf .git web docs notebooks \
 && uv venv --python 3.12 .venv \
 && uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu130 \
      --index-strategy unsafe-best-match -r /tmp/cover-env.txt \
 && rm -rf /tmp/uv-cache
# Model weights are downloaded on first use into the /models volume: tsumugi keeps
# its checkpoints next to its code, so those two folders are links into the volume.
RUN ln -s /models/tsumugi/checkpoints tsumugi/checkpoints \
 && ln -s /models/tsumugi/beat_chord_checkpoints tsumugi/beat_chord_checkpoints

# ------------------------------------------------------------- worker ----
FROM base AS worker
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libcairo2 fontconfig fonts-liberation2 util-linux \
      gcc g++ libc6-dev fonts-noto-cjk fluidsynth fluid-soundfont-gm \
 && rm -rf /var/lib/apt/lists/*
# (gcc/g++: the transcriber compiles its GPU kernels on first use, via torch.compile.
#  fluidsynth, fluid-soundfont-gm: the score played on a sampled piano, score.mp3.)
# JavaScript runtime required by yt-dlp for YouTube extraction.
COPY --from=denoland/deno:bin-2.9.8 /deno /usr/local/bin/deno
COPY requirements/worker.txt /tmp/requirements.txt
RUN uv pip install --require-hashes -r /tmp/requirements.txt
COPY --from=models /opt/models /opt/models
COPY app ./app
COPY model_scripts ./model_scripts
ENV HOME=/models/home \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    NUMBA_CACHE_DIR=/models/cache/numba \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HUB_DISABLE_TELEMETRY=1
# Register Verovio's music font system-wide at build time (HOME is a volume at run time).
RUN HOME=/tmp/fonthome python -c "from app.engrave.pdf import install_music_font; install_music_font()" \
 && mkdir -p /usr/local/share/fonts && cp /tmp/fonthome/.fonts/* /usr/local/share/fonts/ && fc-cache -f \
 && fc-list | grep -qi leipzig && rm -rf /tmp/fonthome
USER app
CMD ["sh", "-c", "mkdir -p /models/home /models/tsumugi/checkpoints /models/tsumugi/beat_chord_checkpoints && exec dramatiq app.jobs.queue --processes ${PS_WORKER_PROCESSES:-1} --threads 1"]

# ---------------------------------------------------------------- dev ----
FROM worker AS dev
USER root
COPY requirements/dev.txt /tmp/requirements-dev.txt
RUN uv pip install --require-hashes -r /tmp/requirements-dev.txt
COPY pyproject.toml ./
COPY tests ./tests
COPY eval ./eval
USER app
CMD ["pytest"]

# ------------------------------------------------------- video-models ----
# Piano-tutorial videos only, on the processor. The notes are read from the picture, so the one
# model environment needed is tsumugi-piano-cover's (beats, bar lines and key, and the score
# writer), here without CUDA, with SheetSage2-Refined's weights at the revision the rest of the
# app pins. The same setup as .github/workflows/piano-video.yml.
FROM base AS video-models
ARG COVER_REF=36f50c1d76ac36b11c7c9de701d231649ba17680
ARG SHEETSAGE_REVISION=5e9c494941b8c0bb322d9f17f365a6f886d5e562
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ENV UV_NO_CACHE=0 UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy UV_SYSTEM_PYTHON=0 UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/hf HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /opt/models
COPY requirements/cover-env-cpu.txt /tmp/cover-env-cpu.txt
RUN git init -q tsumugi-piano-cover && cd tsumugi-piano-cover \
 && git fetch -q --depth 1 https://github.com/anime-song/tsumugi-piano-cover.git ${COVER_REF} && git checkout -q FETCH_HEAD \
 && rm -rf .git web docs notebooks \
 && uv venv --python 3.12 .venv \
 && uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match -r /tmp/cover-env-cpu.txt \
 && rm -rf /tmp/uv-cache
# Offline lookups of "main" resolve to the pinned revision.
RUN tsumugi-piano-cover/.venv/bin/python -c "from pathlib import Path; \
from sheetsage2_refined.pipeline import model_folder; \
print(model_folder('anime-song/SheetSage2-Refined', '${SHEETSAGE_REVISION}')); \
refs = Path('/opt/hf/hub/models--anime-song--SheetSage2-Refined/refs'); refs.mkdir(parents=True, exist_ok=True); \
(refs / 'main').write_text('${SHEETSAGE_REVISION}')" \
 && rm -rf /opt/hf/hub/.locks /opt/hf/xet

# -------------------------------------------------------------- video ----
# The CLI for piano-tutorial videos, from a YouTube link or a file; no queue, no GPU:
#   docker build --target video -t pianoscribe-video backend
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/out:/out" pianoscribe-video \
#     https://www.youtube.com/watch?v=VIDEO_ID /out
# Where YouTube asks to sign in, add  -v "$PWD/cookies.txt:/cookies.txt:ro" -e PS_YOUTUBE_COOKIES_FILE=/cookies.txt
# For a file, add  -v "$PWD/tutorial.mp4:/in/tutorial.mp4:ro"  and give /in/tutorial.mp4 instead of the link.
FROM base AS video
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libcairo2 fontconfig fonts-liberation2 util-linux fonts-noto-cjk \
      fluidsynth fluid-soundfont-gm \
 && rm -rf /var/lib/apt/lists/*
COPY --from=denoland/deno:bin-2.9.8 /deno /usr/local/bin/deno
COPY requirements/worker.txt /tmp/requirements.txt
RUN uv pip install --require-hashes -r /tmp/requirements.txt
COPY --from=video-models /opt/models /opt/models
COPY --from=video-models /opt/hf /opt/hf
COPY app ./app
COPY model_scripts ./model_scripts
# Any user may run it (--user), so everything written goes to /tmp.
ENV HOME=/tmp \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib \
    PS_DATA_DIR=/tmp/data \
    PS_MODEL_DEVICE=cpu
RUN HOME=/tmp/fonthome python -c "from app.engrave.pdf import install_music_font; install_music_font()" \
 && mkdir -p /usr/local/share/fonts && cp /tmp/fonthome/.fonts/* /usr/local/share/fonts/ && fc-cache -f \
 && fc-list | grep -qi leipzig && rm -rf /tmp/fonthome
USER app
ENTRYPOINT ["python", "-m", "app.cli", "transcribe", "--video"]
