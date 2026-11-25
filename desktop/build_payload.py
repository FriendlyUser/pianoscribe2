"""Assemble the desktop app's payload: Python, packages, model programs, weights and tools.

    python desktop/build_payload.py --out desktop/payload --web frontend/dist [--weights-from DIR]

Runs on the platform it builds for (Windows x64 or macOS on Apple Silicon) and needs
uv on PATH and ``huggingface_hub`` in the Python running it. Everything downloaded is
pinned below and verified: packages by uv's lock (desktop/requirements/*.txt), model
programs by commit, weights by Hugging Face revision, tools by published checksums.

``--weights-from DIR`` copies weights from an earlier payload's ``models`` folder
instead of downloading them (moving them with ``--move-weights``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON_VERSION = "3.12.16"
TSUMUGI = ("anime-song/tsumugi", "eba1c7d05e4d6a53cc4e8916d742d9623a3fdc38")
COVER = ("anime-song/tsumugi-piano-cover", "36f50c1d76ac36b11c7c9de701d231649ba17680")
# Hugging Face weights at the revisions the benchmarks in docs/MODELS.md were run with.
TSUMUGI_WEIGHTS = ("anime-song/instrument_agnostic_amt", "c6ee8a22626cd7185031ee71ce7cea27ec862781")
TSUMUGI_CHECKPOINTS = (
    "best_model.pth", "best_model_bass_v2.pth", "best_model_drums_v1_5.pth", "best_model_guitar_v1_5.pth",
    "best_model_other_v1_5.pth", "best_model_vocal_harmony_v1_5.pth", "best_instrument_refinement.pth",
    "best_velocity_model.pth",
)  # fmt: skip
TSUMUGI_BEAT_CHECKPOINT = "best_beat_chord_key.pth"
STEM_SPLITTER = ("anime-song/stem-splitter", "8cf8a201016f76a23db793f4781e436d2bd450b6", "stem_splitter.pt")
HUB_REPOS = {
    "anime-song/SheetSage2-Refined": (
        "5e9c494941b8c0bb322d9f17f365a6f886d5e562",
        ["config.json", "sheetsage2/model.safetensors", "heads/beat-v7u.safetensors",
         "heads/chord-detail-c3.safetensors", "heads/key-k10.safetensors", "heads/melody-m4.safetensors"],
    ),
    "anime-song/tsumugi-piano-cover-v2": ("bd9ad18e8ce6fa0c4f2f7c29715a039359e19db0", ["*"]),
}  # fmt: skip
YTDLP = "2026.11.23"
DENO = "v2.9.8"
FFMPEG_MAC = "https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2"
FFMPEG_WIN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
FFMPEG_WIN_ASSET_PREFIX = "ffmpeg-n8.1"
FFMPEG_WIN_ASSET_SUFFIX = "-win64-lgpl-8.1.zip"
KEEP_OUT = {".git", ".github", "docs", "notebooks", "web", "tests", "evaluation", "recipes", "scripts"}


def log(message: str) -> None:
    print(f"[payload] {message}", flush=True)


def fetch(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(url)
    request = urllib.request.Request(url, headers={"User-Agent": "pianoscribe-build"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310 - fixed https URLs
        return response.read()


def digest_in(text: str) -> str:
    """The SHA-256 digest in a checksum file, whatever its layout ("<hash>  <file>" or PowerShell's)."""
    found = re.search(r"\b[0-9a-fA-F]{64}\b", text)
    if found is None:
        raise SystemExit("no SHA-256 digest in the checksum file")
    return found.group(0)


def verified(data: bytes, expected: str, what: str) -> bytes:
    actual = hashlib.sha256(data).hexdigest()
    if actual.lower() != expected.lower().strip():
        raise SystemExit(f"checksum mismatch for {what}: {actual} != {expected}")
    return data


def platform_name(value: str | None) -> str:
    if value:
        return value
    return {"win32": "windows", "darwin": "macos"}.get(sys.platform) or sys.exit("build on Windows or macOS")


def python_runtime(out: Path, work: Path) -> Path:
    """A relocatable CPython (python-build-standalone, as installed by uv)."""
    target = out / "python"
    if target.exists():
        shutil.rmtree(target)
    install = work / "python"
    subprocess.run(["uv", "python", "install", PYTHON_VERSION, "--install-dir", str(install)], check=True)  # noqa: S607
    found = [p for p in install.iterdir() if p.is_dir() and p.name.startswith(f"cpython-{PYTHON_VERSION}")]
    if not found:
        raise SystemExit("uv did not install the expected Python")
    shutil.copytree(found[0], target, symlinks=True)
    exe = target / "python.exe" if os.name == "nt" else target / "bin" / "python3"
    for marker in target.rglob("EXTERNALLY-MANAGED"):
        marker.unlink()  # this copy belongs to the app: packages go straight into it
    log(f"python {PYTHON_VERSION} -> {target}")
    return exe


def packages(python: Path, platform: str) -> None:
    lock = ROOT / "desktop" / "requirements" / f"{platform}.txt"
    command = ["uv", "pip", "install", "--python", str(python), "--system", "--link-mode", "copy", "-r", str(lock)]
    if platform == "windows":
        command += [
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu130",
            "--index-strategy",
            "unsafe-best-match",
        ]
    subprocess.run(command, check=True)  # noqa: S603
    log(f"packages from {lock.name} installed")


def github_source(repo: str, commit: str, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    data = fetch(f"https://codeload.github.com/{repo}/tar.gz/{commit}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts or parts[0] in KEEP_OUT or member.name.endswith(".ipynb"):
                continue
            member.name = str(Path(*parts))
            archive.extract(member, target, filter="data")
    log(f"{repo}@{commit[:7]} -> {target}")


def materialize(hub: Path) -> None:
    """Hugging Face caches link snapshots to blobs; copy the files in so the payload has no links.

    Blobs live in each repository's ``blobs/`` folder and, for large files fetched through Xet
    (huggingface_hub 1.x and later), in one shared ``hub/blobs/`` store. Both go once the
    snapshots hold real files, or every large weight file would ship twice.
    """
    for repo in hub.glob("models--*"):
        for link in [p for p in (repo / "snapshots").rglob("*") if p.is_symlink()]:
            source = link.resolve()
            link.unlink()
            shutil.copyfile(source, link)
        shutil.rmtree(repo / "blobs", ignore_errors=True)
    for store in (hub / "blobs", hub / ".locks", hub.parent / "xet"):
        shutil.rmtree(store, ignore_errors=True)
    leftover = [p for p in hub.rglob("*") if p.is_symlink()]
    if leftover:
        raise SystemExit(f"links left in the model cache: {leftover[:3]}")


def weights(models: Path, source: Path | None, move: bool) -> None:
    tsumugi = models / "tsumugi"
    places = {
        "hf": models / "hf",
        "stem_splitter": models / "stem_splitter",
        "tsumugi/checkpoints": tsumugi / "checkpoints",
        "tsumugi/beat_chord_checkpoints": tsumugi / "beat_chord_checkpoints",
    }
    if source is not None:
        for relative, target in places.items():
            origin = source / relative
            if not origin.exists():
                raise SystemExit(f"{origin} is missing")
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(origin), str(target))
            else:
                shutil.copytree(origin, target)
        log(f"weights {'moved' if move else 'copied'} from {source}")
        return
    from huggingface_hub import hf_hub_download, snapshot_download

    repo, revision = TSUMUGI_WEIGHTS
    for name in TSUMUGI_CHECKPOINTS:
        hf_hub_download(repo, name, revision=revision, local_dir=places["tsumugi/checkpoints"])
    hf_hub_download(
        repo, TSUMUGI_BEAT_CHECKPOINT, revision=revision, local_dir=places["tsumugi/beat_chord_checkpoints"]
    )
    repo, revision, name = STEM_SPLITTER
    hf_hub_download(repo, name, revision=revision, local_dir=places["stem_splitter"] / "weights")
    hub = places["hf"] / "hub"
    for repo, (revision, patterns) in HUB_REPOS.items():
        snapshot_download(repo, revision=revision, cache_dir=hub, allow_patterns=patterns)
        refs = hub / f"models--{repo.replace('/', '--')}" / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision)  # offline lookups of "main" resolve to the pinned revision
    materialize(hub)
    for leftover in models.rglob(".cache"):
        if leftover.is_dir():
            shutil.rmtree(leftover)
    log("weights downloaded")


def tools(out: Path, platform: str) -> None:
    ffmpeg, deno, bin_dir = out / "ffmpeg", out / "deno", out / "bin"
    for folder in (ffmpeg, deno, bin_dir):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    if platform == "windows":
        sums = fetch(f"{FFMPEG_WIN}/checksums.sha256").decode()
        name, digest = next(
            (line.split()[1], line.split()[0])
            for line in sums.splitlines()
            if line.split()[1].startswith(FFMPEG_WIN_ASSET_PREFIX) and line.split()[1].endswith(FFMPEG_WIN_ASSET_SUFFIX)
        )
        archive = zipfile.ZipFile(io.BytesIO(verified(fetch(f"{FFMPEG_WIN}/{name}"), digest, name)))
        for member in archive.namelist():
            if member.endswith(("/bin/ffmpeg.exe", "/bin/ffprobe.exe")):
                (ffmpeg / Path(member).name).write_bytes(archive.read(member))
            if member.endswith("/LICENSE.txt"):
                (ffmpeg / "LICENSE.txt").write_bytes(archive.read(member))
        log(f"ffmpeg {name}")
        deno_asset, ytdlp_asset, exe = "deno-x86_64-pc-windows-msvc.zip", "yt-dlp.exe", ".exe"
    else:
        for program in ("ffmpeg", "ffprobe"):
            digest = digest_in(fetch(f"{FFMPEG_MAC}/{program}.zip.sha256").decode())
            data = verified(fetch(f"{FFMPEG_MAC}/{program}.zip"), digest, program)
            (ffmpeg / program).write_bytes(zipfile.ZipFile(io.BytesIO(data)).read(program))
            (ffmpeg / program).chmod(0o755)
        log("ffmpeg 9.0.2 (macOS arm64)")
        deno_asset, ytdlp_asset, exe = "deno-aarch64-apple-darwin.zip", "yt-dlp_macos", ""
    deno_url = f"https://github.com/denoland/deno/releases/download/{DENO}/{deno_asset}"
    digest = digest_in(fetch(f"{deno_url}.sha256sum").decode())
    archive = zipfile.ZipFile(io.BytesIO(verified(fetch(deno_url), digest, deno_asset)))
    (deno / f"deno{exe}").write_bytes(archive.read(f"deno{exe}"))
    (deno / f"deno{exe}").chmod(0o755)
    sums = fetch(f"https://github.com/yt-dlp/yt-dlp/releases/download/{YTDLP}/SHA2-256SUMS").decode()
    digest = next(line.split()[0] for line in sums.splitlines() if line.split()[1] == ytdlp_asset)
    data = verified(
        fetch(f"https://github.com/yt-dlp/yt-dlp/releases/download/{YTDLP}/{ytdlp_asset}"), digest, ytdlp_asset
    )
    (bin_dir / f"yt-dlp{exe}").write_bytes(data)
    (bin_dir / f"yt-dlp{exe}").chmod(0o755)
    log(f"deno {DENO}, yt-dlp {YTDLP}")


def app_files(out: Path, web: Path) -> None:
    backend = out / "backend"
    shutil.rmtree(backend, ignore_errors=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "backend" / "app", backend / "app", ignore=ignore)
    shutil.copytree(ROOT / "backend" / "model_scripts", backend / "model_scripts", ignore=ignore)
    shutil.rmtree(out / "web", ignore_errors=True)
    if not (web / "index.html").is_file():
        raise SystemExit(f"{web} is not a built web interface (run npm run build in frontend/)")
    shutil.copytree(web, out / "web")
    shutil.copyfile(ROOT / "desktop" / "LICENCES.txt", out / "LICENCES.txt")
    log("backend, web interface and licences copied")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--platform", choices=("windows", "macos"))
    parser.add_argument("--weights-from", type=Path)
    parser.add_argument("--move-weights", action="store_true")
    parser.add_argument("--work", type=Path, help="scratch folder (default: <out>.work)")
    parser.add_argument(
        "--steps",
        default="python,sources,weights,tools,app",
        help="comma-separated subset to run, e.g. tools,app to finish an interrupted build",
    )
    args = parser.parse_args()
    platform = platform_name(args.platform)
    out = args.out.resolve()
    work = (args.work or out.with_name(out.name + ".work")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    steps = set(args.steps.split(","))
    if "python" in steps:
        packages(python_runtime(out, work), platform)
    if "sources" in steps:
        github_source(*TSUMUGI, out / "models" / "tsumugi")
        github_source(*COVER, out / "models" / "tsumugi-piano-cover")
    if "weights" in steps:
        weights(out / "models", args.weights_from, args.move_weights)
    if "tools" in steps:
        tools(out, platform)
    if "app" in steps:
        app_files(out, args.web.resolve())
    shutil.rmtree(work, ignore_errors=True)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink())
    log(f"done: {size / 1e9:.2f} GB in {out}")


if __name__ == "__main__":
    main()
