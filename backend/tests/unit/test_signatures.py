from __future__ import annotations

import pytest

from app.errors import ValidationError
from app.ingest.signatures import MediaKind, sniff, verify_signature

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
MP4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00" * 16
MOV = b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00qt  " + b"\x00" * 16
MP3_FRAME = b"\xff\xfb\x90\x64" + b"\x00" * 32
MP3_ID3 = b"ID3\x04\x00\x00\x00\x00\x00\x0a" + b"\x00" * 10 + MP3_FRAME


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (WAV, MediaKind.WAV),
        (MP4, MediaKind.ISO_BMFF),
        (MOV, MediaKind.ISO_BMFF),
        (MP3_FRAME, MediaKind.MP3),
        (MP3_ID3, MediaKind.MP3),
    ],
)
def test_sniff_recognises_supported_containers(data: bytes, kind: MediaKind) -> None:
    assert sniff(data) == kind


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"hello world, definitely not media",
        b"%PDF-1.7\n",
        b"MZ\x90\x00" + b"\x00" * 60,  # Windows executable
        b"\x7fELF" + b"\x00" * 60,
        b"<html><script>alert(1)</script>",
        b"RIFF\x24\x00\x00\x00AVI LIST",  # RIFF but not WAVE
        b"\x00\x00\x00\x20ftypXXXX" + b"\x00" * 16,  # unknown brand
        b"ID3\x04\x00\x00\x00\x00\x00\x0a" + b"\x00" * 10 + b"not an mpeg frame",
        b"\xff\xff\xff\xff",  # sync bits but reserved fields
    ],
)
def test_sniff_rejects_other_content(data: bytes) -> None:
    assert sniff(data) is None


def test_extension_and_content_must_agree() -> None:
    assert verify_signature("song.WAV", WAV) == MediaKind.WAV
    assert verify_signature("clip.mov", MOV) == MediaKind.ISO_BMFF
    assert verify_signature("clip.mp4", MP4) == MediaKind.ISO_BMFF
    with pytest.raises(ValidationError) as mismatch:
        verify_signature("song.mp3", WAV)
    assert mismatch.value.code == "content_mismatch"
    with pytest.raises(ValidationError) as unknown:
        verify_signature("song.mp3", b"#!/bin/sh\nrm -rf /\n")
    assert unknown.value.code == "unrecognized_content"
    with pytest.raises(ValidationError) as unsupported:
        verify_signature("song.flac", WAV)
    assert unsupported.value.code == "unsupported_format"
    with pytest.raises(ValidationError):
        verify_signature("no_extension", WAV)
