"""Run a model program in its own Python environment and working directory.

The models (tsumugi, SheetSage2-Refined, tsumugi-piano-cover) are research
code with their own pinned dependencies, so they are not imported into the
worker. Each step is a subprocess: it gets a clean GPU context, cannot leak
memory into the worker between jobs, can be killed on a time limit, and its
console output is captured for the job log and parsed for progress.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

from app.errors import JobError
from app.logging import get_logger
from app.proc import group_kwargs, kill_tree

log = get_logger(__name__)
_TAIL_LINES = 40


class ModelStepError(RuntimeError):
    """A model program exited with an error or ran out of time."""

    def __init__(self, step: str, message: str, tail: list[str], *, timed_out: bool = False) -> None:
        super().__init__(f"{step}: {message}")
        self.step = step
        self.tail = tail
        self.timed_out = timed_out

    @property
    def out_of_memory(self) -> bool:
        text = "\n".join(self.tail).lower()
        return "out of memory" in text or "cuda oom" in text


def _offline() -> dict[str, str]:
    """Desktop builds pin their bundled weights: model programs must not look for newer ones online."""
    return {"HF_HUB_OFFLINE": "1"} if os.environ.get("PS_MODELS_OFFLINE") == "1" else {}


def run_step(
    step: str,
    argv: Sequence[str | Path],
    *,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> list[str]:
    """Run ``argv`` to completion; return the last lines of its output.

    ``on_line`` receives every output line as it is printed (for progress).
    The whole process group is killed if the time limit passes.
    """
    started = time.perf_counter()
    tail: deque[str] = deque(maxlen=_TAIL_LINES)
    process = subprocess.Popen(  # noqa: S603 - argv is assembled from configuration, never a shell string
        [str(a) for a in argv],
        cwd=str(cwd),
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **_offline(), **(env or {})},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **group_kwargs(),
    )
    timed_out = threading.Event()
    finished = threading.Event()

    def watchdog() -> None:
        if finished.wait(timeout):
            return
        timed_out.set()
        kill_tree(process)

    guard = threading.Thread(target=watchdog, name=f"{step}-watchdog", daemon=True)
    guard.start()
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            if on_line is not None:
                try:
                    on_line(line)
                except Exception:  # noqa: BLE001 - progress parsing must never fail the step
                    log.debug("model.progress_parse_failed", step=step)
            if not line.startswith("@@"):
                tail.append(line[-500:])
        returncode = process.wait()
    finally:
        finished.set()
        kill_tree(process)  # reap any helper left behind
    seconds = round(time.perf_counter() - started, 2)
    if timed_out.is_set():
        log.warning("model.step", step=step, outcome="timeout", seconds=seconds)
        raise ModelStepError(step, f"exceeded its {int(timeout)} s time limit", list(tail), timed_out=True)
    if returncode != 0:
        log.warning(
            "model.step", step=step, outcome="error", returncode=returncode, seconds=seconds, tail=list(tail)[-8:]
        )
        raise ModelStepError(step, f"exited with status {returncode}", list(tail))
    log.info("model.step", step=step, outcome="ok", seconds=seconds)
    return list(tail)


def as_job_error(error: ModelStepError) -> JobError:
    """Translate a model failure into what the user should be told."""
    if error.out_of_memory:
        return JobError(
            "gpu_out_of_memory",
            "The graphics card ran out of memory on this track. Try a shorter recording.",
            retryable=False,
        )
    if error.timed_out:
        return JobError("model_timeout", "Processing this track took too long. Try a shorter recording.")
    return JobError("model_failed", "A model step failed on this recording. Please try again.", retryable=True)
