"""Download the MAESTRO v3 test recordings used by the grounded tests (about 125 MB).

    python -m eval.fetch_maestro [DEST]        default: eval/data/maestro

MAESTRO (Hawthorne et al., 2019) is published by Google Magenta under CC BY-NC-SA 4.0.
The files are not part of this repository; this script fetches six test-split pieces
(Scarlatti to Rachmaninoff, about two minutes each) from a Hugging Face mirror.
"""

from __future__ import annotations

import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

MIRROR = "https://huggingface.co/datasets/ddPn08/maestro-v3.0.0/resolve/main/"
PIECES = {
    "scarlatti_k9": "2008/MIDI-Unprocessed_10_R2_2008_01-05_ORIG_MID--AUDIO_10_R2_2008_wav--1",
    "schubert_op90_4": "2015/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_03_R2_2015_wav--1",
    "chopin_op10_9": "2011/MIDI-Unprocessed_24_R1_2011_MID--AUDIO_R1-D9_11_Track11_wav",
    "debussy_etude_7": "2004/MIDI-Unprocessed_SMF_17_R1_2004_03-06_ORIG_MID--AUDIO_20_R2_2004_12_Track12_wav--1",
    "scriabin_op42_4_5": "2009/MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_04_WAV",
    "rachmaninoff_op32_8": "2015/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_11_R1_2015_wav--5",
}


def main() -> int:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data" / "maestro"
    dest.mkdir(parents=True, exist_ok=True)
    for name, stem in PIECES.items():
        for remote, local in ((".midi", ".midi"), (".wav", ".wav")):
            target = dest / f"{name}{local}"
            if target.exists():
                continue
            url = MIRROR + urllib.parse.quote(stem + remote)
            print(f"{name}{local}  <-  {url}")
            partial = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as fh:  # noqa: S310
                shutil.copyfileobj(response, fh)
            partial.replace(target)
    print(f"{len(list(dest.glob('*.midi')))} pieces in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
