"""Environment-driven configuration. Every operational limit lives here."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PS_", env_file=".env", extra="ignore")

    # --- infrastructure -------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    data_dir: Path = Path("/data")
    models_dir: Path = Path("/models")
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    cors_origins: list[str] = Field(default_factory=list)

    # --- ingestion limits -----------------------------------------------
    max_upload_mb: int = 200
    max_duration_seconds: int = 600
    upload_chunk_bytes: int = 1 << 20

    # --- YouTube --------------------------------------------------------
    youtube_enabled: bool = True
    youtube_timeout_seconds: int = 300
    youtube_metadata_timeout_seconds: int = 60
    youtube_max_attempts: int = 3
    youtube_backoff_base_seconds: float = 2.0
    youtube_backoff_max_seconds: float = 30.0
    youtube_max_filesize_mb: int = 400

    # --- sandbox for untrusted media tooling ----------------------------
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ytdlp_bin: str = "yt-dlp"
    ffmpeg_timeout_seconds: int = 300
    sandbox_memory_mb: int = 2048
    sandbox_cpu_seconds: int = 900
    sandbox_max_file_mb: int = 2048
    sandbox_max_open_files: int = 256
    # strict: refuse to run a media tool without its resource limits (servers).
    # basic: time limits only, where the limit tools do not exist (desktop builds on Windows/macOS).
    sandbox: Literal["strict", "basic"] = "strict"

    # --- job orchestration ----------------------------------------------
    queue_name: str = "transcribe"
    job_max_retries: int = 2
    job_retry_min_backoff_ms: int = 5_000
    job_retry_max_backoff_ms: int = 60_000
    job_time_limit_seconds: int = 3600
    job_lock_ttl_seconds: int = 60
    # A worker silent for this long is presumed dead and its in-flight jobs are re-queued.
    queue_heartbeat_timeout_seconds: int = 45
    # Share (per million broker commands) that also run dead-worker recovery.
    queue_maintenance_chance: int = 50_000

    # --- retention ------------------------------------------------------
    raw_retention_seconds: int = 3600
    result_retention_seconds: int = 7 * 24 * 3600
    janitor_interval_seconds: int = 300

    # --- models ---------------------------------------------------------
    # Where the model programs and their Python environments live (see the worker image).
    tsumugi_dir: Path = Path("/opt/models/tsumugi")
    cover_dir: Path = Path("/opt/models/tsumugi-piano-cover")
    scripts_dir: Path = Path("/app/model_scripts")
    # One interpreter for every model program (desktop builds ship a single environment);
    # unset, each program's own .venv is used.
    model_python: Path | None = None
    # Windows: FFmpeg's shared libraries, which the stem separator's audio loader needs.
    ffmpeg_shared_dir: Path | None = None
    # torchcodec: the stem separator reads audio through FFmpeg's shared libraries (servers).
    # soundfile: through soundfile instead (desktop builds, which ship no FFmpeg libraries).
    audio_io: Literal["torchcodec", "soundfile"] = "torchcodec"
    model_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    cover_checkpoint: str = "anime-song/tsumugi-piano-cover-v2"
    # Windows per batch in transcription; 2 fits an 8 GB card.
    tsumugi_window_batch_size: int = 2
    transcribe_timeout_seconds: int = 2400
    arrange_timeout_seconds: int = 1800
    notate_timeout_seconds: int = 1800

    # Cover generation. The defaults are the model authors' recommended values.
    cover_onset_bias: float = 4.0  # pull note onsets toward onsets in the original
    cover_dynamics: float = 1.0
    cover_density: float = 1.0
    cover_fill: float = 0.0
    cover_above: float = 0.0
    cover_span: float = 0.0
    cover_source_cfg: float = 2.0  # higher follows the original more closely
    cover_temperature: float = 1.0
    cover_top_p: float = 0.9
    # Generation is sampled; a fixed seed makes a job reproducible and cacheable.
    cover_seed: int = 0
    # Candidates generated per song; the one most faithful to the detected melody is kept.
    cover_candidates: int = 1
    # An arrangement whose first two bars play less than this share of the detected melody is
    # generated again with the next seed, up to this many times; the best opening is kept.
    cover_opening_min_fidelity: float = 0.8
    cover_opening_retries: int = 2
    # Share of GPU memory the beat / key models may take (their own default, 0.6, is too
    # little on an 8 GB card).
    analysis_memory_fraction: float = 0.95

    # Solo piano: a recording longer than this many bars is written in sections of at most this
    # size. On long solo pieces the score model starts to write notes twice (0 disables).
    solo_section_bars: int = 64

    # --- piano tutorial videos (app.visual) ------------------------------
    # The falling notes are read from the picture, at most this wide and at most this many
    # frames a second. Narrower than about 1280 the bars of black keys get hard to tell apart.
    visual_max_width: int = 1920
    visual_max_fps: int = 60
    # Whole frames sampled across the video to find the keyboard.
    visual_sample_frames: int = 48
    # Wall-clock limit for decoding every frame of the video.
    visual_timeout_seconds: int = 1200
    # exact: the notes read from a video are written down by rule, as read (model_scripts/exact_score.py).
    # model: they are given to the score model, as a transcription by ear is.
    video_score_writer: Literal["exact", "model"] = "exact"
    # Largest picture fetched for a YouTube piano video.
    youtube_video_height: int = 1080
    # A cookies.txt (Netscape format) handed to yt-dlp, for hosts YouTube asks to sign in, such
    # as CI runners. Copied into each download's own folder, so the original is never written.
    youtube_cookies_file: Path | None = None

    page_size: Literal["A4", "Letter"] = "A4"
    # score.mp3: the score played on a sampled piano by FluidSynth (app.engrave.audio). Left out
    # where the synthesizer or a SoundFont is missing, or when turned off.
    score_audio: bool = True
    fluidsynth_bin: str = "fluidsynth"
    # A General MIDI SoundFont whose first program is a piano. None: FluidR3_GM where the system keeps it.
    soundfont_file: Path | None = None
    # cairo: the worker writes score.pdf. none: no server-side PDF; a desktop shell prints
    # the score page with its own browser engine instead.
    pdf_renderer: Literal["cairo", "none"] = "cairo"
    # Running inside the desktop app (app.desktop): enables its settings panel and API.
    desktop: bool = False
    # Solo piano transcription: tsumugi, or MuScriptor (gated download, see docs/MODELS.md).
    solo_transcriber: Literal["tsumugi", "muscriptor-small", "muscriptor-medium", "muscriptor-large"] = "tsumugi"

    # --- derived --------------------------------------------------------
    @property
    def tsumugi_python(self) -> Path:
        return self.model_python or _venv_python(self.tsumugi_dir / ".venv")

    @property
    def cover_python(self) -> Path:
        return self.model_python or _venv_python(self.cover_dir / ".venv")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    def ensure_dirs(self) -> None:
        for path in (self.uploads_dir, self.work_dir, self.artifacts_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


def _venv_python(venv: Path) -> Path:
    return venv / "Scripts" / "python.exe" if sys.platform == "win32" else venv / "bin" / "python"


@lru_cache
def get_settings() -> Settings:
    return Settings()
