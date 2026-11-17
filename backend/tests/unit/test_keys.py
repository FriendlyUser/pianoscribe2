"""Deduplication keys: identical work maps to one job id, different work never does."""

from __future__ import annotations

from app.jobs.keys import upload_job_id, youtube_job_id
from app.jobs.models import JobOptions

SHA = "ee948fc37cef8e39d88df47327711d03eaeacc4e8cba91f974c30c558c7544c0"


def test_same_content_and_options_share_an_id() -> None:
    assert upload_job_id(SHA, JobOptions()) == upload_job_id(SHA, JobOptions())
    assert upload_job_id(SHA.upper(), JobOptions()) == upload_job_id(SHA, JobOptions())
    assert youtube_job_id("dQw4w9WgXcQ", JobOptions()) == youtube_job_id("dQw4w9WgXcQ", JobOptions())


def test_ids_are_32_hex_characters() -> None:
    job_id = upload_job_id(SHA, JobOptions())
    assert len(job_id) == 32 and int(job_id, 16) >= 0


def test_options_that_change_output_change_the_id() -> None:
    assert upload_job_id(SHA, JobOptions(solo_piano=True)) != upload_job_id(SHA, JobOptions(solo_piano=False))
    assert youtube_job_id("dQw4w9WgXcQ", JobOptions(solo_piano=True)) != youtube_job_id("dQw4w9WgXcQ", JobOptions())


def test_different_content_or_source_kind_never_collide() -> None:
    other = SHA.replace("e", "f", 1)
    assert upload_job_id(SHA, JobOptions()) != upload_job_id(other, JobOptions())
    # A video id and an upload hash with the same text must not alias.
    assert upload_job_id("dQw4w9WgXcQ", JobOptions()) != youtube_job_id("dQw4w9WgXcQ", JobOptions())


def test_video_ids_are_case_sensitive() -> None:
    assert youtube_job_id("dQw4w9WgXcQ", JobOptions()) != youtube_job_id("DQW4W9WGXCQ", JobOptions())


def test_pipeline_version_invalidates_cached_results() -> None:
    assert upload_job_id(SHA, JobOptions(), "1") != upload_job_id(SHA, JobOptions(), "2")
