"""Filesystem artifact store for finished scores.

The interface is deliberately small (put a set of files atomically, open one,
check completeness, delete) so it can be backed by object storage in a
multi-node deployment without touching callers.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

# name -> (media type, download extension)
ARTIFACTS: dict[str, tuple[str, str]] = {
    "score.pdf": ("application/pdf", "pdf"),
    "score.html": ("text/html; charset=utf-8", "html"),
    "score.musicxml": ("application/vnd.recordare.musicxml+xml", "musicxml"),
    "score.mid": ("audio/midi", "mid"),
    "performance.mid": ("audio/midi", "performance.mid"),
    "transcription.mid": ("audio/midi", "transcription.mid"),
}
# score.pdf is not required: a desktop build leaves PDF output to its shell (Settings.pdf_renderer).
REQUIRED = ("score.html", "score.musicxml")


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _dir(self, job_id: str) -> Path:
        if not job_id.isalnum():
            raise ValueError("invalid job id")
        return self.root / job_id

    def put_all(self, job_id: str, files: dict[str, bytes]) -> dict[str, int]:
        """Write all files for a job atomically (directory rename)."""
        final = self._dir(job_id)
        staging = self.root / f".staging-{job_id}-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        try:
            for name, data in files.items():
                with (staging / name).open("wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
            if final.exists():
                shutil.rmtree(final)
            staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return {name: len(data) for name, data in files.items()}

    def path(self, job_id: str, name: str) -> Path | None:
        if name not in ARTIFACTS:
            return None
        candidate = self._dir(job_id) / name
        return candidate if candidate.is_file() else None

    def complete(self, job_id: str) -> bool:
        return all((self._dir(job_id) / name).is_file() for name in REQUIRED)

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self._dir(job_id), ignore_errors=True)
