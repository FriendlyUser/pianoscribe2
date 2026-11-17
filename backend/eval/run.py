"""Command line.

    python -m eval.run arrangement REFERENCE.json label=score.musicxml [label=score.musicxml ...]
    python -m eval.run transcription REFERENCE_DIR ESTIMATE_DIR label=suffix [label=suffix ...]

``transcription`` compares ``ESTIMATE_DIR/<name>.<suffix>.mid`` with ``REFERENCE_DIR/<name>.midi``
for every reference file and prints per-piece and mean note scores.
"""

from __future__ import annotations

import json
import sys

from eval.arrangement import Reference, evaluate
from eval.scoreio import read_musicxml


def _table(rows: dict[str, dict[str, object]]) -> str:
    keys = list(next(iter(rows.values())))
    width = max(len(k) for k in keys)
    lines = [" " * (width + 2) + "  ".join(f"{label:>12s}" for label in rows)]
    for key in keys:
        cells = []
        for values in rows.values():
            v = values[key]
            cells.append(f"{v:>12}" if isinstance(v, bool) else f"{v:>12.3f}")
        lines.append(f"{key:<{width}}  " + "  ".join(cells))
    return "\n".join(lines)


def _transcription(argv: list[str]) -> int:
    from pathlib import Path

    from eval.transcription import note_scores

    reference_dir, estimate_dir = Path(argv[0]), Path(argv[1])
    systems = dict(item.split("=", 1) for item in argv[2:])
    totals: dict[str, dict[str, list[float]]] = {label: {} for label in systems}
    print(f"{'piece':24s} " + "  ".join(f"{label + ' on/off F1':>22s}" for label in systems))
    for reference in sorted(reference_dir.glob("*.midi")):
        cells = []
        for label, suffix in systems.items():
            estimate = estimate_dir / f"{reference.stem}.{suffix}.mid"
            if not estimate.exists():
                cells.append(f"{'missing':>22s}")
                continue
            scores = note_scores(reference, estimate)
            for key, value in scores.items():
                totals[label].setdefault(key, []).append(value)
            cells.append(f"{scores['onset_f1']:>10.3f} / {scores['offset_f1']:<9.3f}")
        print(f"{reference.stem:24s} " + "  ".join(cells))
    print("-" * 60)
    for label, values in totals.items():
        if values:
            mean = {k: sum(v) / len(v) for k, v in values.items()}
            print(
                f"{label:12s} onset F1 {mean['onset_f1']:.3f} (P {mean['onset_precision']:.3f} R {mean['onset_recall']:.3f})"
                f"  onset+offset F1 {mean['offset_f1']:.3f}  over {len(values['onset_f1'])} pieces"
            )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[0] == "transcription":
        return _transcription(argv[1:])
    if len(argv) < 3 or argv[0] != "arrangement":
        print(__doc__)
        return 2
    reference = Reference.load(argv[1])
    rows = {}
    for item in argv[2:]:
        label, path = item.split("=", 1)
        rows[label] = evaluate(reference, read_musicxml(path)).as_dict()
    print(f"{reference.title}: {len(reference.melody)} reference melody notes, {len(reference.bass)} bass points")
    print(_table(rows))
    if "--json" in argv:
        print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
