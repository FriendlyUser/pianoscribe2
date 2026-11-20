"""Beats, bar lines, time signatures and melody of a recording. Runs in the tsumugi-piano-cover environment.

    python annotate_audio.py --audio song.wav --out annotation.json

A thin wrapper around ``piano_ar.sheetsage.SheetSageAnnotator`` (SheetSage2-Refined),
producing the same JSON the cover model uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def best_device() -> str:
    """NVIDIA GPU, else Apple GPU, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from piano_ar.sheetsage import SheetSageAnnotator

    device = best_device() if args.device == "auto" else args.device
    annotation = SheetSageAnnotator(None, device=device).annotate(Path(args.audio))
    out = Path(args.out)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(annotation, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    print(
        f"{len(annotation['beats'])} beats, {len(annotation['bars'])} bars, {len(annotation['melody'])} melody notes -> {out}"
    )


if __name__ == "__main__":
    main()
