"""Binary-signature (magic number) detection for accepted media containers.

Uploads are identified by their leading bytes, never by the client-supplied
filename or Content-Type alone. The extension must additionally agree with
the detected container.
"""

from __future__ import annotations

from enum import StrEnum

from app.errors import ValidationError

SNIFF_BYTES = 4096


class MediaKind(StrEnum):
    MP3 = "mp3"
    WAV = "wav"
    ISO_BMFF = "isobmff"  # MP4 / MOV / M4A family


ALLOWED_EXTENSIONS: dict[str, MediaKind] = {
    "mp3": MediaKind.MP3,
    "wav": MediaKind.WAV,
    "mp4": MediaKind.ISO_BMFF,
    "mov": MediaKind.ISO_BMFF,
}
# ffprobe ``format_name`` values acceptable for each detected kind.
PROBE_FORMATS: dict[MediaKind, frozenset[str]] = {
    MediaKind.MP3: frozenset({"mp3"}),
    MediaKind.WAV: frozenset({"wav"}),
    MediaKind.ISO_BMFF: frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}),
}
_ISO_BRANDS = {
    b"isom", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1", b"hvc1", b"hev1",
    b"dash", b"M4V ", b"M4A ", b"M4B ", b"f4v ", b"mmp4", b"MSNV", b"3gp4", b"3gp5", b"3gp6", b"3g2a", b"qt  ",
}  # fmt: skip
_QUICKTIME_ATOMS = {b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}


def _skip_id3(data: bytes) -> bytes | None:
    """Return the bytes following an ID3v2 tag, or None if the tag exceeds ``data``."""
    if len(data) < 10 or data[:3] != b"ID3":
        return data
    size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
    end = 10 + size
    return data[end:] if end < len(data) else None


def _is_mpeg_audio_frame(header: bytes) -> bool:
    if len(header) < 4 or header[0] != 0xFF or (header[1] & 0xE0) != 0xE0:
        return False
    version = (header[1] >> 3) & 0x03
    layer = (header[1] >> 1) & 0x03
    bitrate = (header[2] >> 4) & 0x0F
    sample_rate = (header[2] >> 2) & 0x03
    return version != 0x01 and layer != 0x00 and bitrate not in (0x00, 0x0F) and sample_rate != 0x03


def sniff(data: bytes) -> MediaKind | None:
    """Identify the container from the first bytes of a file."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return MediaKind.WAV
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return MediaKind.ISO_BMFF if data[8:12] in _ISO_BRANDS else None
    if len(data) >= 8 and data[4:8] in _QUICKTIME_ATOMS:
        return MediaKind.ISO_BMFF
    if data[:3] == b"ID3":
        rest = _skip_id3(data)
        # A large tag (e.g. cover art) can exceed the sniffed prefix; the ID3
        # magic is then the strongest available evidence. ffprobe confirms later.
        if rest is None or _is_mpeg_audio_frame(rest.lstrip(b"\x00")[:4]):
            return MediaKind.MP3
        return None
    if _is_mpeg_audio_frame(data[:4]):
        return MediaKind.MP3
    return None


def extension_of(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def verify_signature(filename: str, head: bytes) -> MediaKind:
    """Validate extension and magic bytes together; raise a user-facing error otherwise."""
    extension = extension_of(filename)
    expected = ALLOWED_EXTENSIONS.get(extension)
    if expected is None:
        raise ValidationError(
            "unsupported_format",
            "Unsupported file type. Upload an MP3, WAV, MP4 or MOV file.",
            status=415,
        )
    detected = sniff(head)
    if detected is None:
        raise ValidationError(
            "unrecognized_content",
            "This file does not look like valid audio or video. It may be corrupted or mislabeled.",
            status=415,
        )
    if detected != expected:
        raise ValidationError(
            "content_mismatch",
            f"The file's contents do not match its .{extension} extension.",
            status=415,
        )
    return detected
