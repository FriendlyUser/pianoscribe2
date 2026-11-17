"""Streaming upload receiver: bounded size, signature check, content hashing."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.errors import ValidationError
from app.ingest.signatures import ALLOWED_EXTENSIONS, SNIFF_BYTES, MediaKind, extension_of, verify_signature

_SAFE_NAME = re.compile(r"[^\w\s.\-()]+", re.UNICODE)


@dataclass(frozen=True)
class StoredUpload:
    sha256: str
    size: int
    path: Path
    kind: MediaKind
    display_name: str


def sanitize_filename(filename: str) -> str:
    """Reduce a client filename to a safe display label (never used as a path)."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = _SAFE_NAME.sub("", name).strip(" .")
    return name[:120] or "upload"


def title_from_filename(filename: str) -> str:
    stem = sanitize_filename(filename).rsplit(".", 1)[0]
    return re.sub(r"[_\-]+", " ", stem).strip() or "Untitled"


async def receive_upload(
    chunks: AsyncIterator[bytes],
    *,
    filename: str,
    declared_length: int | None,
    settings: Settings,
) -> StoredUpload:
    """Stream an upload to disk, enforcing limits as bytes arrive.

    The body is never buffered in memory. The size cap is checked against the
    declared Content-Length up front and again on the running byte count; the
    binary signature is verified as soon as enough bytes have arrived, so a
    mislabeled file is rejected before it is fully received.
    """
    limit = settings.max_upload_bytes
    too_large = ValidationError(
        "file_too_large", f"The file exceeds the {settings.max_upload_mb} MB upload limit.", status=413
    )
    if declared_length is not None and declared_length > limit:
        raise too_large
    extension = extension_of(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            "unsupported_format", "Unsupported file type. Upload an MP3, WAV, MP4 or MOV file.", status=415
        )

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    temp = settings.uploads_dir / f".incoming-{uuid.uuid4().hex}"
    digest = hashlib.sha256()
    head = bytearray()
    kind: MediaKind | None = None
    size = 0
    try:
        with temp.open("wb") as fh:
            async for chunk in chunks:
                if not chunk:
                    continue
                size += len(chunk)
                if size > limit:
                    raise too_large
                if kind is None:
                    head.extend(chunk[: SNIFF_BYTES - len(head)])
                    if len(head) >= SNIFF_BYTES:
                        kind = verify_signature(filename, bytes(head))
                digest.update(chunk)
                fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
        if size == 0:
            raise ValidationError("empty_file", "The uploaded file is empty.")
        if kind is None:
            kind = verify_signature(filename, bytes(head))
        sha256 = digest.hexdigest()
        final = settings.uploads_dir / f"{sha256}.{extension}"
        temp.replace(final)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return StoredUpload(sha256=sha256, size=size, path=final, kind=kind, display_name=sanitize_filename(filename))
