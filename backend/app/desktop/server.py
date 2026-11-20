"""``python -m app.desktop``: the API, a worker thread and the web interface in one process."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import fakeredis
import fakeredis.aioredis
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api import deps
from app.config import Settings, get_settings
from app.desktop import settings as desktop_settings
from app.desktop.settings import DesktopSettings
from app.jobs.runner import dead_letter, run_job
from app.jobs.store import JobStore
from app.logging import get_logger
from app.storage.artifacts import ArtifactStore

log = get_logger(__name__)
_SNAPSHOT_SECONDS = 5.0


class LocalQueue:
    """:class:`~app.jobs.service.JobQueue` served by one worker thread (one job at a time, like one GPU)."""

    def __init__(self, store: JobStore, artifacts: ArtifactStore, settings: Settings) -> None:
        self._items: queue.PriorityQueue[tuple[float, int, str]] = queue.PriorityQueue()
        self._counter = 0
        self._failures: dict[str, int] = {}
        self._store, self._artifacts, self._settings = store, artifacts, settings
        self._thread = threading.Thread(target=self._work, name="desktop-worker", daemon=True)
        self._thread.start()

    def enqueue(self, job_id: str, *, delay_ms: int = 0) -> None:
        self._counter += 1
        self._items.put((time.monotonic() + delay_ms / 1000, self._counter, job_id))

    def _work(self) -> None:
        while True:
            due, _, job_id = self._items.get()
            wait = due - time.monotonic()
            if wait > 0:
                time.sleep(min(wait, 1.0))
                self._items.put((due, self._counter, job_id))
                continue
            try:
                run_job(job_id, store=self._store, queue=self, artifacts=self._artifacts, settings=self._settings)
                self._failures.pop(job_id, None)
            except Exception:  # noqa: BLE001 - the queue retries, like the server's broker
                failures = self._failures.get(job_id, 0) + 1
                self._failures[job_id] = failures
                if failures > self._settings.job_max_retries:
                    dead_letter(job_id, store=self._store, settings=self._settings)
                    self._failures.pop(job_id, None)
                else:
                    self.enqueue(job_id, delay_ms=self._settings.job_retry_min_backoff_ms * failures)


class JobArchive:
    """Keeps the job list across restarts: the in-process store is written to a file now and then."""

    def __init__(self, client: fakeredis.FakeRedis, path: Path) -> None:
        self._client, self._path = client, path

    def load(self) -> int:
        try:
            records: dict[str, str] = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        for key, value in records.items():
            if '"status":"running"' in value or '"status":"queued"' in value:
                continue  # interrupted by the app closing; the user can submit again
            self._client.set(key, value)
        return len(records)

    def save(self) -> None:
        records: dict[str, str] = {}
        for key in self._client.scan_iter(match="ps:job:*"):
            value = self._client.get(key)
            if isinstance(value, bytes) and isinstance(key, bytes):
                records[key.decode()] = value.decode()
        partial = self._path.with_suffix(".tmp")
        partial.write_text(json.dumps(records), encoding="utf-8")
        partial.replace(self._path)

    def keep_saving(self) -> None:
        def loop() -> None:
            while True:
                time.sleep(_SNAPSHOT_SECONDS)
                try:
                    self.save()
                except OSError as error:
                    log.warning("desktop.archive_failed", error=str(error))

        threading.Thread(target=loop, name="desktop-archive", daemon=True).start()


class Download(BaseModel):
    token: str | None = None


def _system_info(settings: Settings) -> dict[str, Any]:
    """Which accelerator the models will use (asked of the model environment, once)."""
    probe = (
        "import json, torch; m = getattr(torch.backends, 'mps', None); "
        "print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), "
        "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'mps': bool(m and m.is_available())}))"
    )
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv
            [str(settings.tsumugi_python), "-c", probe], capture_output=True, text=True, timeout=180, check=False
        )
        return (
            json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else {"error": done.stderr[-300:]}
        )
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"error": str(error)}


def build_app(settings: Settings, web_dir: Path | None) -> FastAPI:
    from app.api.main import create_app

    settings.ensure_dirs()
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server)
    store = JobStore(client, record_ttl_seconds=10 * 365 * 24 * 3600)
    artifacts = ArtifactStore(settings.artifacts_dir)
    archive = JobArchive(client, settings.data_dir / "jobs.json")
    log.info("desktop.jobs_restored", count=archive.load())
    archive.keep_saving()
    local_queue = LocalQueue(store, artifacts, settings)
    async_client = fakeredis.aioredis.FakeRedis(server=server)
    settings_path = settings.data_dir / "desktop-settings.json"
    system: dict[str, Any] = {"checking": True}

    def probe() -> None:
        found = _system_info(settings)
        system.update(found)
        system.pop("checking", None)

    threading.Thread(target=probe, name="desktop-probe", daemon=True).start()

    app = create_app()
    app.dependency_overrides[deps.settings_dep] = lambda: settings
    app.dependency_overrides[deps.store_dep] = lambda: store
    app.dependency_overrides[deps.queue_dep] = lambda: local_queue
    app.dependency_overrides[deps.artifacts_dep] = lambda: artifacts
    app.dependency_overrides[deps.async_redis_dep] = lambda: async_client
    app.dependency_overrides[deps.job_models_dep] = lambda: desktop_settings.load(settings_path).job_models()

    router = APIRouter(prefix="/api/desktop")

    @router.get("/state")
    def state() -> dict[str, Any]:
        return {
            "settings": desktop_settings.load(settings_path).model_dump(),
            "models": desktop_settings.catalogue_state(),
            "system": dict(system),
        }

    @router.put("/settings")
    def update(choice: DesktopSettings) -> dict[str, Any]:
        model = desktop_settings.by_id(choice.solo_transcriber)
        if model is not None and not model.bundled and not desktop_settings.installed(model):
            raise HTTPException(409, f"{model.name} is not downloaded yet.")
        desktop_settings.save(settings_path, choice)
        return state()

    @router.post("/models/{model_id}/download")
    def download(model_id: str, body: Download) -> dict[str, Any]:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

        model = desktop_settings.by_id(model_id)
        if model is None or model.bundled:
            raise HTTPException(404, "No such downloadable model.")
        try:
            # The token is used for this request only and is never stored.
            snapshot_download(model.repo, token=body.token or None)
        except GatedRepoError as error:
            raise HTTPException(
                403, "Hugging Face refused the download. Accept the model's terms on its page, then use a read token."
            ) from error
        except (HfHubHTTPError, OSError) as error:
            raise HTTPException(502, f"The download failed: {str(error)[:200]}") from error
        return state()

    app.include_router(router)
    if web_dir is not None and web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="python -m app.desktop", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--web", type=Path, default=None, help="the built web interface (frontend/dist)")
    args = parser.parse_args()
    settings = get_settings().model_copy(update={"desktop": True})
    app = build_app(settings, args.web)
    print(f"@@ready http://{args.host}:{args.port}", flush=True)
    # A long keep-alive: the shell polls the API every couple of seconds over the same connection.
    uvicorn.run(app, host=args.host, port=args.port, log_config=None, access_log=False, timeout_keep_alive=75)
