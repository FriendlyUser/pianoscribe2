"""Environment-driven configuration. Every operational limit lives here."""

from __future__ import annotations

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
    model_device: Literal["auto", "cuda", "cpu"] = "auto"
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

    page_size: Literal["A4", "Letter"] = "A4"

    # --- derived --------------------------------------------------------
    @property
    def tsumugi_python(self) -> Path:
        return self.tsumugi_dir / ".venv" / "bin" / "python"

    @property
    def cover_python(self) -> Path:
        return self.cover_dir / ".venv" / "bin" / "python"

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
