"""Process groups on every platform: start a child in its own group, kill the whole tree.

Model programs and media tools start helpers of their own (data loaders, ffmpeg,
a JavaScript runtime). A time limit or a finished job must take all of them down.
On Linux and macOS the child leads a new session and the group gets SIGKILL; on
Windows it gets a new process group and ``taskkill /T /F`` ends the tree.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from typing import Any

WINDOWS = sys.platform == "win32"


def group_kwargs() -> dict[str, Any]:
    """Keyword arguments for ``subprocess.Popen`` that put the child in a group of its own."""
    if WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
    return {"start_new_session": True}


def kill_tree(process: subprocess.Popen[Any]) -> None:
    """Kill ``process`` and everything it started. Safe to call on a process that has exited."""
    if WINDOWS:
        # Sweep the tree even when the parent has exited: its helpers may still be running.
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(  # noqa: S603 - fixed argv, pid is an int
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],  # noqa: S607
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
            )
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
