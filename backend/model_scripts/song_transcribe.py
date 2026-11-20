"""Run tsumugi's song transcription with audio read and written by soundfile.

    python song_transcribe.py WORKER.py [worker arguments]

``WORKER.py`` is ``cover_studio/tsumugi_worker.py`` from the tsumugi-piano-cover repository.
The stem separator reads and writes audio through torchaudio's TorchCodec backend, which
needs FFmpeg's shared libraries; desktop builds do not ship those. The pipeline only ever
hands the models a decoded WAV file, which soundfile reads exactly, so the two calls are
replaced and the worker runs unchanged in this process.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def load(path: str | Path, *_: object, **__: object) -> tuple[torch.Tensor, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), int(rate)


def save_wav(out_path: str | Path, samples: torch.Tensor, sample_rate: int) -> None:
    sf.write(str(out_path), samples.detach().cpu().to(torch.float32).numpy().T, int(sample_rate), subtype="FLOAT")


def main() -> None:
    import stem_splitter.inference as separator

    torchaudio.load = load  # type: ignore[assignment]
    separator._save_wav_no_warning = save_wav
    worker = Path(sys.argv[1])
    sys.argv = [str(worker), *sys.argv[2:]]
    sys.path.insert(0, str(worker.parent))
    runpy.run_path(str(worker), run_name="__main__")


if __name__ == "__main__":
    main()
