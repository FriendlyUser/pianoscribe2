"""Deterministic job identifiers: the basis of deduplication and caching.

A job id is a hash of *what* is being transcribed (file content hash or
YouTube video id), *how* (options that affect output) and the pipeline
version. Identical submissions therefore map to the same job record, which
makes enqueueing idempotent and lets repeat submissions reuse finished work.
"""

from __future__ import annotations

import hashlib

from app import PIPELINE_VERSION
from app.jobs.models import JobOptions


def _job_id(kind: str, identity: str, options: JobOptions, pipeline_version: str) -> str:
    material = f"pianoscribe|v{pipeline_version}|{kind}|{identity}|{options.cache_fragment()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def upload_job_id(content_sha256: str, options: JobOptions, pipeline_version: str = PIPELINE_VERSION) -> str:
    return _job_id("upload", content_sha256.lower(), options, pipeline_version)


def youtube_job_id(video_id: str, options: JobOptions, pipeline_version: str = PIPELINE_VERSION) -> str:
    # Video ids are case-sensitive; they are used verbatim.
    return _job_id("youtube", video_id, options, pipeline_version)
