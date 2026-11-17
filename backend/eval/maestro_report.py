"""Transcribe the MAESTRO pieces with the shipped solo-piano stage and print note scores.

python -m eval.maestro_report [DATA_DIR]      default: $PS_GROUNDED_DATA/maestro or eval/data/maestro
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from app.models import song
from eval.transcription import note_scores


def main() -> int:
    default = Path(os.environ.get("PS_GROUNDED_DATA", Path(__file__).resolve().parent / "data")) / "maestro"
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    pieces = sorted(p for p in data.glob("*.midi") if p.with_suffix(".wav").is_file())
    if not pieces:
        print(f"no recordings in {data} (python -m eval.fetch_maestro)", file=sys.stderr)
        return 1
    settings = Settings()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for reference in pieces:
            midi = song.transcribe_piano(reference.with_suffix(".wav"), Path(tmp) / f"{reference.stem}.mid", settings)
            scores = note_scores(reference, midi)
            rows.append(scores)
            print(f"{reference.stem:24s} onset F1 {scores['onset_f1']:.3f}   onset+offset F1 {scores['offset_f1']:.3f}")
    mean = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
    print(f"{'mean':24s} onset F1 {mean('onset_f1'):.3f}   onset+offset F1 {mean('offset_f1'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
