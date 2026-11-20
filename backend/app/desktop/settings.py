"""The user's model choices in the desktop app, and the catalogue they choose from."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Device = Literal["auto", "cuda", "mps", "cpu"]
SoloTranscriber = Literal["tsumugi", "muscriptor-small", "muscriptor-medium", "muscriptor-large"]


class DesktopSettings(BaseModel):
    """Choices that change the score, applied per job (and part of each job's cache key)."""

    model_device: Device = "auto"
    solo_transcriber: SoloTranscriber = "tsumugi"
    # Arrangement: how closely the piano part follows the recording, how many notes it plays,
    # and which of the model's possible arrangements is written (a different seed, a different take).
    cover_source_cfg: float = Field(2.0, ge=1.0, le=4.0)
    cover_density: float = Field(1.0, ge=0.5, le=1.5)
    cover_seed: int = Field(0, ge=0, le=999_999)
    # Solo piano longer than this many bars is written in sections (0: one pass).
    solo_section_bars: int = Field(64, ge=0, le=256)

    def job_models(self) -> dict[str, str | int | float]:
        return self.model_dump()


def load(path: Path) -> DesktopSettings:
    try:
        return DesktopSettings.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DesktopSettings()


def save(path: Path, settings: DesktopSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    partial.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    partial.replace(path)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    stage: str  # "solo" (piano transcription), "song" (everything a song needs), "analysis", "notation"
    name: str
    repo: str  # Hugging Face repository the weights come from
    size_gb: float
    licence: str
    bundled: bool  # shipped with the app; otherwise downloaded on request
    gated: bool  # the download needs the user's Hugging Face token and accepted terms
    measured: str  # what the grounded benchmarks found (docs/MODELS.md)
    setting: str | None = None  # the DesktopSettings value that selects it, if any


CATALOGUE: tuple[ModelInfo, ...] = (
    ModelInfo(
        "tsumugi",
        "solo",
        "tsumugi",
        "anime-song/instrument_agnostic_amt",
        0.52,
        "MIT",
        True,
        False,
        "Piano note accuracy 0.978 on MAESTRO (onset F1); also separates and transcribes songs.",
        "tsumugi",
    ),
    ModelInfo(
        "muscriptor-small",
        "solo",
        "MuScriptor small",
        "MuScriptor/muscriptor-small",
        0.41,
        "CC BY-NC 4.0 + terms",
        False,
        True,
        "Not measured here; the medium model scored 0.807 on MAESTRO.",
        "muscriptor-small",
    ),
    ModelInfo(
        "muscriptor-medium",
        "solo",
        "MuScriptor medium",
        "MuScriptor/muscriptor-medium",
        1.23,
        "CC BY-NC 4.0 + terms",
        False,
        True,
        "Piano note accuracy 0.807 on MAESTRO (onset F1), against tsumugi's 0.978.",
        "muscriptor-medium",
    ),
    ModelInfo(
        "muscriptor-large",
        "solo",
        "MuScriptor large",
        "MuScriptor/muscriptor-large",
        5.47,
        "CC BY-NC 4.0 + terms",
        False,
        True,
        "Not measured here; its authors report about one point above medium.",
        "muscriptor-large",
    ),
    ModelInfo(
        "stem-splitter",
        "song",
        "Stem splitter",
        "anime-song/stem-splitter",
        0.35,
        "MIT",
        True,
        False,
        "Separates a song into instruments before tsumugi transcribes each.",
    ),
    ModelInfo(
        "tsumugi-piano-cover-v2",
        "song",
        "tsumugi piano cover v2",
        "anime-song/tsumugi-piano-cover-v2",
        1.23,
        "CC BY-NC 4.0",
        True,
        False,
        "Arrangements score 0.84 to 0.88 against a human piano arrangement.",
    ),
    ModelInfo(
        "sheetsage2-refined",
        "analysis",
        "SheetSage2-Refined",
        "anime-song/SheetSage2-Refined",
        3.01,
        "CC BY-NC 4.0",
        True,
        False,
        "Beats, bar lines, meter, melody and key changes (key signatures right for 93% of bars).",
    ),
)


def hub_dir() -> Path:
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def installed(model: ModelInfo) -> bool:
    """Whether the weights are on this computer (bundled models are checked like any other)."""
    if model.id == "tsumugi":
        tsumugi = os.environ.get("PS_TSUMUGI_DIR")
        return tsumugi is not None and (Path(tsumugi) / "checkpoints" / "best_model.pth").is_file()
    if model.id == "stem-splitter":
        cache = os.environ.get("STEM_SPLITTER_CACHE")
        return cache is not None and any(Path(cache).rglob("*.pt"))
    snapshots = hub_dir() / f"models--{model.repo.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(any(p.iterdir()) for p in snapshots.iterdir() if p.is_dir())


def catalogue_state() -> list[dict[str, object]]:
    return [{**asdict(model), "installed": installed(model)} for model in CATALOGUE]


def by_id(model_id: str) -> ModelInfo | None:
    return next((m for m in CATALOGUE if m.id == model_id), None)


def dumps_state(settings: DesktopSettings) -> str:
    return json.dumps({"settings": settings.model_dump(), "models": catalogue_state()})
