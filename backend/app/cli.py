"""Operator CLI.

python -m app.cli transcribe INPUT OUTPUT_DIR [--solo] [--work DIR]   run the pipeline locally, no queue
python -m app.cli doctor                                              check the model environments and GPU
python -m app.cli dead-letters                                        list dead-lettered jobs
python -m app.cli requeue JOB_ID                                      retry a failed / dead-lettered job
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import Settings, get_settings
from app.logging import configure_logging


def _transcribe(args: argparse.Namespace) -> int:
    from app.errors import JobError
    from app.pipeline.local import print_progress, run_local

    source = Path(args.input).resolve()
    if not source.is_file():
        print(f"no such file: {source}", file=sys.stderr)
        return 2
    out = Path(args.output)
    scratch = Path(args.work).resolve() if args.work else Path(tempfile.mkdtemp(prefix="pianoscribe-"))
    try:
        settings: Settings = get_settings().model_copy(update={"data_dir": scratch})
        try:
            finished = run_local(source, out, settings, solo=args.solo, title=args.title, on_update=print_progress())
        except JobError as error:
            print(f"failed ({error.code}): {error.message}", file=sys.stderr)
            return 1
        report = {
            **(finished.result.model_dump(exclude={"artifacts"}) if finished.result else {}),
            "timings": finished.timings,
            "output": str(out),
        }
        print(json.dumps(report, indent=2))
        return 0
    finally:
        if not args.work:
            shutil.rmtree(scratch, ignore_errors=True)


def _doctor(_: argparse.Namespace) -> int:
    settings = get_settings()
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"{'ok  ' if passed else 'FAIL'}  {label}{': ' + detail if detail else ''}")

    for binary in (settings.ffmpeg_bin, settings.ffprobe_bin):
        check(binary, shutil.which(binary) is not None)
    probe = "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
    for label, python in (("tsumugi", settings.tsumugi_python), ("tsumugi-piano-cover", settings.cover_python)):
        if not python.is_file():
            check(f"{label} environment", False, f"{python} is missing")
            continue
        done = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, timeout=120, check=False)
        detail = done.stdout.strip() or done.stderr.strip().splitlines()[-1:]
        check(f"{label} environment", done.returncode == 0, str(detail))
        if done.returncode == 0 and " True " not in f" {done.stdout} ":
            print("      no GPU visible: the models will run on the CPU, many times slower")
    check("score writer", (settings.scripts_dir / "notate_cover.py").is_file(), str(settings.scripts_dir))
    return 0 if ok else 1


def _dead_letters(_: argparse.Namespace) -> int:
    from app.jobs.queue import dead_letters

    print(json.dumps(dead_letters(), indent=2))
    return 0


def _requeue(args: argparse.Namespace) -> int:
    from app.jobs.queue import DramatiqQueue, get_artifacts, get_store
    from app.jobs.service import submit_job

    store = get_store()
    job = store.get(args.job_id)
    if job is None:
        print("no such job", file=sys.stderr)
        return 1
    submission = submit_job(
        job_id=job.id, source=job.source, options=job.options, title=job.title,
        store=store, queue=DramatiqQueue(), artifacts=get_artifacts(), settings=get_settings(),
    )  # fmt: skip
    print(submission.job.status.value)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="pianoscribe")
    sub = parser.add_subparsers(dest="command", required=True)
    transcribe = sub.add_parser("transcribe", help="turn a local recording into a score, without the job queue")
    transcribe.add_argument("input")
    transcribe.add_argument("output")
    transcribe.add_argument(
        "--solo", action="store_true", help="the recording is solo piano: transcribe, do not arrange"
    )
    transcribe.add_argument("--title")
    transcribe.add_argument("--work", help="keep work files here; after a failure, a rerun resumes from them")
    transcribe.set_defaults(func=_transcribe)
    sub.add_parser("doctor", help="check the model environments and the GPU").set_defaults(func=_doctor)
    sub.add_parser("dead-letters", help="list dead-lettered jobs").set_defaults(func=_dead_letters)
    requeue = sub.add_parser("requeue", help="retry a failed job")
    requeue.add_argument("job_id")
    requeue.set_defaults(func=_requeue)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
