# PianoScribe desktop app

The same app as the web version, running entirely on your own computer: the backend, the
models and the web interface in one window. Windows x64 (NVIDIA GPU recommended) and macOS 14+
on Apple Silicon. Intel Macs cannot run the PyTorch version the models need.

## How it fits together

```
PianoScribe.exe / PianoScribe.app        Electron shell (main.cjs): window, PDF printing, process control
  resources/payload/
    python/            relocatable CPython 3.12 with every package (requirements/<platform>.txt)
    backend/           backend/app and backend/model_scripts
    models/            tsumugi + tsumugi-piano-cover code and checkpoints, hf/ (Hugging Face cache),
                       stem_splitter/weights/
    ffmpeg/ deno/ bin/ ffmpeg + ffprobe, Deno, yt-dlp
    web/               the built web interface
```

The shell starts `python -m app.desktop` on a free local port and opens a window on it. That is
the server's own API with an in-process job store and worker thread in place of Redis and Dramatiq
(see `backend/app/desktop/`). The web interface shows a **Models** panel and prints PDFs through the
shell only when the API reports a desktop build, so the hosted web version is unchanged.

User data (scores, settings, downloaded models, `backend.log`) lives in `%APPDATA%\PianoScribe` or
`~/Library/Application Support/PianoScribe`; the app folder is never written to.

## Building

The GitHub Action `.github/workflows/desktop.yml` builds both apps, runs a headless smoke test of
each (it transcribes `samples/fur_elise_opening.mp3` through the real app and prints the PDF) and
attaches them to a draft release in parts under 2 GiB. Run it from the Actions tab, or push a tag
named `desktop-v*`.
A run takes about 40 minutes (most of it compressing the Windows app). On the first full run
the Windows smoke test took 4.5 minutes on the runner's processor; the macOS one ran on the
runner's Apple GPU in 1.5 minutes, which also exercises the Apple GPU code path.

To build by hand on the target platform (needs uv, Node 24, Git and Python with
`huggingface_hub`):

```bash
cd frontend && npm ci && npm run build && cd ..
python desktop/build_payload.py --out desktop/payload --web frontend/dist
cd desktop && npm ci && npx electron-builder --dir
# then move desktop/payload into dist/<platform>-unpacked/resources/payload
# (on macOS: PianoScribe.app/Contents/Resources/payload, then codesign --force --deep --sign -)
```

`build_payload.py --steps tools,app` reruns only some steps, and `--weights-from DIR` takes the
weights from an earlier payload's `models` folder instead of downloading them.

Smoke test any build: `PianoScribe --smoke-test=recording.mp3 --smoke-out=DIR [--smoke-solo]`
writes the score, the PDF and `smoke.json` (timings, accelerator found, interface check) and
exits 0 on success.

To change the Python packages, edit `requirements.in` and run `requirements/compile.sh`.
