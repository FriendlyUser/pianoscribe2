"""Constrained subprocess execution for tools that parse untrusted media.

ffprobe, ffmpeg and yt-dlp all consume attacker-controlled bytes. Each
invocation runs:

* without a shell, with a minimal environment and a private working directory;
* in its own session, so the whole process group can be killed on timeout;
* under hard resource limits (address space, CPU seconds, file size, open
  files, no core dumps) applied by ``prlimit``, with ``no_new_privs`` set by
  ``setpriv`` (both from util-linux), so no code runs between fork and exec
  in the multi-threaded worker;
* with a wall-clock timeout.

This is one layer of defence; the worker container adds the outer ones
(non-root user, read-only root filesystem, dropped capabilities, cgroup
memory/CPU/pid limits).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.logging import get_logger
from app.proc import WINDOWS, group_kwargs, kill_tree

log = get_logger(__name__)
_OUTPUT_LIMIT = 64 * 1024 * 1024


@dataclass(frozen=True)
class ResourceLimits:
    memory_bytes: int | None  # RLIMIT_AS; None for runtimes that reserve huge virtual ranges
    cpu_seconds: int
    file_bytes: int
    open_files: int
    # False (desktop builds): run the tool with its time limit only when the limit tools are
    # missing. The server keeps the default and refuses to run a media tool without them.
    required: bool = True

    @classmethod
    def for_media(cls, settings: Settings) -> ResourceLimits:
        return cls(
            memory_bytes=settings.sandbox_memory_mb * 1024 * 1024,
            cpu_seconds=settings.sandbox_cpu_seconds,
            file_bytes=settings.sandbox_max_file_mb * 1024 * 1024,
            open_files=settings.sandbox_max_open_files,
            required=settings.sandbox == "strict",
        )

    @classmethod
    def for_downloader(cls, settings: Settings) -> ResourceLimits:
        # yt-dlp embeds a JavaScript runtime whose allocator reserves far more
        # address space than it uses, so RLIMIT_AS cannot be applied; the
        # container memory cgroup bounds it instead.
        return cls(
            memory_bytes=None,
            cpu_seconds=settings.sandbox_cpu_seconds,
            file_bytes=settings.youtube_max_filesize_mb * 1024 * 1024,
            open_files=max(settings.sandbox_max_open_files, 1024),
            required=settings.sandbox == "strict",
        )


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool


class SandboxError(RuntimeError):
    def __init__(self, message: str, result: SandboxResult) -> None:
        super().__init__(message)
        self.result = result


def _launcher(limits: ResourceLimits) -> list[str]:
    """Command prefix that applies the limits and then execs the real tool."""
    setpriv, prlimit = shutil.which("setpriv"), shutil.which("prlimit")
    if setpriv is None or prlimit is None:
        if not limits.required:
            return []  # desktop: the user's own machine and files; the time limit still applies
        # Fail closed: never run a media tool without its limits.
        raise RuntimeError("sandbox requires `setpriv` and `prlimit` (util-linux) on PATH")
    args = [
        setpriv,
        "--no-new-privs",
        prlimit,
        "--core=0",
        f"--cpu={limits.cpu_seconds}:{limits.cpu_seconds + 5}",
        f"--fsize={limits.file_bytes}",
        f"--nofile={limits.open_files}",
    ]
    if limits.memory_bytes is not None:
        args.append(f"--as={limits.memory_bytes}")
    return [*args, "--"]


def _minimal_env(extra: dict[str, str] | None, home: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "TMPDIR": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
    }
    if WINDOWS:
        # Programs on Windows need these to start at all (sockets, temp files, the user profile).
        for name in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT"):
            if name in os.environ:
                env[name] = os.environ[name]
        env.update({"TEMP": str(home), "TMP": str(home), "USERPROFILE": str(home), "APPDATA": str(home)})
    for passthrough in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "SSL_CERT_FILE"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    env.update(extra or {})
    return env


def run_sandboxed(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    limits: ResourceLimits,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> SandboxResult:
    """Run ``argv`` under the sandbox policy and return its captured output."""
    cwd.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stdout_path = cwd / f".sandbox-{os.getpid()}-{time.monotonic_ns()}.out"
    stderr_path = stdout_path.with_suffix(".err")
    timed_out = False
    process_options: dict[str, Any] = {**group_kwargs(), **({} if WINDOWS else {"umask": 0o077})}
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        process = subprocess.Popen(  # argv is built by the caller, never a shell string
            [*_launcher(limits), *argv],
            cwd=str(cwd),
            env=_minimal_env(env, cwd),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            close_fds=True,
            **process_options,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Kill the whole group: also reaps helpers left behind on normal exit.
            kill_tree(process)
            process.wait()
    try:
        stdout = stdout_path.read_bytes()[:_OUTPUT_LIMIT].decode("utf-8", "replace")
        stderr = stderr_path.read_bytes()[:_OUTPUT_LIMIT].decode("utf-8", "replace")
    finally:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)

    result = SandboxResult(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        seconds=round(time.perf_counter() - started, 3),
        timed_out=timed_out,
    )
    log.debug(
        "sandbox.exit",
        tool=Path(argv[0]).name,
        returncode=result.returncode,
        seconds=result.seconds,
        timed_out=timed_out,
    )
    if check and (timed_out or result.returncode != 0):
        reason = "timed out" if timed_out else f"exited with status {result.returncode}"
        raise SandboxError(f"{Path(argv[0]).name} {reason}", result)
    return result
