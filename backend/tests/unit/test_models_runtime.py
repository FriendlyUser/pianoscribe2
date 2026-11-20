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


def test_a_timed_out_step_takes_its_helpers_down_with_it(tmp_path) -> None:  # noqa: ANN001
    import sys
    import time

    import pytest

    from app.models.runtime import ModelStepError, run_step

    marker = tmp_path / "helper-still-running"
    helper = tmp_path / "helper.py"
    helper.write_text(f"import pathlib, time\ntime.sleep(3)\npathlib.Path({str(marker)!r}).touch()\n")
    # The step starts a helper that would write a file after 3 s, then hangs itself.
    step = tmp_path / "step.py"
    step.write_text(
        f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, {str(helper)!r}])\ntime.sleep(60)\n"
    )
    script = str(step)
    started = time.monotonic()
    with pytest.raises(ModelStepError) as caught:
        run_step("hang", [sys.executable, script], cwd=tmp_path, timeout=1.0)
    assert caught.value.timed_out and time.monotonic() - started < 10
    time.sleep(3.5)
    assert not marker.exists()  # the helper was killed with the step


def test_desktop_sandbox_runs_without_the_linux_limit_tools(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import shutil
    import sys

    import pytest

    from app.ingest import sandbox

    monkeypatch.setattr(shutil, "which", lambda name: None)  # as on Windows and macOS
    strict = sandbox.ResourceLimits(memory_bytes=None, cpu_seconds=10, file_bytes=1 << 20, open_files=64)
    with pytest.raises(RuntimeError):
        sandbox.run_sandboxed([sys.executable, "-c", "print(1)"], cwd=tmp_path, timeout=10, limits=strict)
    basic = sandbox.ResourceLimits(memory_bytes=None, cpu_seconds=10, file_bytes=1 << 20, open_files=64, required=False)
    result = sandbox.run_sandboxed([sys.executable, "-c", "print('ok')"], cwd=tmp_path, timeout=10, limits=basic)
    assert result.returncode == 0 and result.stdout.strip() == "ok"


def test_one_interpreter_can_serve_every_model_program(tmp_path) -> None:  # noqa: ANN001
    from app.config import Settings

    shared = tmp_path / "python.exe"
    settings = Settings(model_python=shared)
    assert settings.tsumugi_python == shared and settings.cover_python == shared
    default = Settings(model_python=None)
    assert default.tsumugi_python.parent.parent.name == ".venv"


def test_link_or_copy_falls_back_when_links_are_not_allowed(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    from pathlib import Path

    from app.models.song import link_or_copy

    source = tmp_path / "audio.wav"
    source.write_bytes(b"RIFF....WAVE")

    def refuse(*_: object, **__: object) -> None:
        raise OSError(1314, "A required privilege is not held by the client")  # what Windows says

    monkeypatch.setattr(Path, "symlink_to", refuse)
    hard = tmp_path / "hard.wav"
    link_or_copy(source, hard)
    assert hard.read_bytes() == source.read_bytes()
    monkeypatch.setattr("os.link", refuse)
    copied = tmp_path / "copied.wav"
    link_or_copy(source, copied)
    assert copied.read_bytes() == source.read_bytes()
