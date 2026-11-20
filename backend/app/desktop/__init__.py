"""Desktop mode: the same API and web interface in one local process.

The server deployment runs the API, a Dramatiq worker, Redis and nginx as separate
services. The desktop app runs ``python -m app.desktop``: the same FastAPI routes,
an in-process Redis stand-in (fakeredis) for job state and live progress, a worker
thread in place of the queue service, and the built web interface served from the
same port. The Electron shell in ``desktop/`` starts it and opens a window on it.
"""
