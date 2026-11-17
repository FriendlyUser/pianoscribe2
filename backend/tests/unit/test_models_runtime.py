"""Running model programs as subprocesses: output capture, progress lines, failures, time limits."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from app.models.runtime import ModelStepError, as_job_error, run_step


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_returns_output_tail_and_streams_lines(tmp_path: Path) -> None:
    seen: list[str] = []
    tail = run_step(
        "demo",
        py("print('@@progress {\"done\": 1}'); print('working'); print('done')"),
        cwd=tmp_path,
        timeout=30,
        on_line=seen.append,
    )
    assert seen == ['@@progress {"done": 1}', "working", "done"]
    assert tail == ["working", "done"]  # protocol lines are not part of the log tail


def test_environment_is_extended_not_replaced(tmp_path: Path) -> None:
    tail = run_step(
        "env", py("import os; print(os.environ['PS_TEST_MARK'], bool(os.environ.get('PATH')))"),
        cwd=tmp_path, timeout=30, env={"PS_TEST_MARK": "yes"},
    )  # fmt: skip
    assert tail == ["yes True"]


def test_runs_in_the_given_directory(tmp_path: Path) -> None:
    assert run_step("cwd", py("import os; print(os.getcwd())"), cwd=tmp_path, timeout=30) == [str(tmp_path)]


def test_nonzero_exit_raises_with_the_log_tail(tmp_path: Path) -> None:
    with pytest.raises(ModelStepError) as excinfo:
        run_step("boom", py("import sys; print('about to fail'); sys.exit(3)"), cwd=tmp_path, timeout=30)
    error = excinfo.value
    assert error.step == "boom" and not error.timed_out and error.tail == ["about to fail"]
    assert as_job_error(error).retryable  # an unexplained model failure is worth one more try


def test_out_of_memory_is_reported_as_such(tmp_path: Path) -> None:
    code = "import sys; print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB'); sys.exit(1)"
    with pytest.raises(ModelStepError) as excinfo:
        run_step("oom", py(code), cwd=tmp_path, timeout=30)
    assert excinfo.value.out_of_memory
    job_error = as_job_error(excinfo.value)
    assert job_error.code == "gpu_out_of_memory" and not job_error.retryable


def test_time_limit_kills_the_whole_process_group(tmp_path: Path) -> None:
    code = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    started = time.monotonic()
    with pytest.raises(ModelStepError) as excinfo:
        run_step("slow", py(code), cwd=tmp_path, timeout=1.0)
    assert excinfo.value.timed_out and time.monotonic() - started < 10
    assert as_job_error(excinfo.value).code == "model_timeout"


def test_a_failing_progress_callback_does_not_fail_the_step(tmp_path: Path) -> None:
    def explode(_: str) -> None:
        raise ValueError("bad progress line")

    assert run_step("robust", py("print('fine')"), cwd=tmp_path, timeout=30, on_line=explode) == ["fine"]
