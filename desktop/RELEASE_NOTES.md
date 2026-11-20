PianoScribe desktop app: recordings to piano sheet music, entirely on your computer.
The models are included; nothing is uploaded anywhere.

**Windows 10/11, x64.** An NVIDIA GPU is strongly recommended (a four-minute song takes a few
minutes; on the processor alone, close to an hour).
1. Download every `PianoScribe-windows-x64.7z.0NN` part into one folder.
2. Open the `.001` file with [7-Zip](https://www.7-zip.org) and extract it; 7-Zip joins the parts.
3. Move the `PianoScribe` folder somewhere permanent (for example `%LOCALAPPDATA%\Programs`) and
   run `PianoScribe.exe`. The app is not code-signed: if Windows shows "Windows protected your PC",
   choose **More info > Run anyway**.

**macOS 14 or later, Apple Silicon (M1 or newer).** Intel Macs are not supported.
1. Download every `PianoScribe-macos-arm64.tar.gz.part-*` file into one folder.
2. In Terminal, in that folder: `cat PianoScribe-macos-arm64.tar.gz.part-* | tar -xz`
3. Move `PianoScribe.app` to Applications. The app is not notarized: the first time, right-click it
   and choose **Open**, or run `xattr -dr com.apple.quarantine /Applications/PianoScribe.app`.

`SHA256SUMS-*.txt` lists checksums for every part.

**Licence.** The beat and arrangement models are CC BY-NC 4.0: personal and other non-commercial
use only. See `LICENCES.txt` inside the app for every bundled component.
