// PianoScribe desktop shell: starts the bundled backend and shows the web interface.
//
// Layout of the bundled payload (resources/payload, or $PIANOSCRIBE_PAYLOAD when developing):
//   python/        a relocatable CPython with every package installed (desktop/requirements/*.txt)
//   backend/       app/ and model_scripts/ from backend/
//   models/        tsumugi/ and tsumugi-piano-cover/ (code + checkpoints), hf/ (Hugging Face cache),
//                  stem_splitter/weights/
//   ffmpeg/        ffmpeg and ffprobe          deno/   the JavaScript runtime yt-dlp needs
//   bin/           yt-dlp                       web/    the built web interface (frontend/dist)
const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");

const WINDOWS = process.platform === "win32";
const EXE = WINDOWS ? ".exe" : "";
const payload = process.env.PIANOSCRIBE_PAYLOAD || path.join(process.resourcesPath, "payload");
let backend = null;
let base = null;
let quitting = false;

function arg(name) {
  const prefix = `--${name}=`;
  const found = process.argv.find((a) => a.startsWith(prefix));
  return found ? found.slice(prefix.length) : null;
}

function pythonPath() {
  return WINDOWS ? path.join(payload, "python", "python.exe") : path.join(payload, "python", "bin", "python3");
}

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

// Downloaded models (MuScriptor) go to the user's own folder; the bundled ones are linked in,
// so one Hugging Face cache sees both and the app folder is never written to.
function prepareModelCache(hfHome) {
  const bundled = path.join(payload, "models", "hf", "hub");
  const hub = path.join(hfHome, "hub");
  fs.mkdirSync(hub, { recursive: true });
  if (!fs.existsSync(bundled)) return;
  for (const name of fs.readdirSync(bundled)) {
    const target = path.join(hub, name);
    try {
      const stat = fs.lstatSync(target);
      if (stat.isSymbolicLink()) {
        if (fs.readlinkSync(target) === path.join(bundled, name)) continue;
        fs.unlinkSync(target);
      } else {
        continue; // a real folder: something the user downloaded; leave it
      }
    } catch {
      // missing: create below
    }
    fs.symlinkSync(path.join(bundled, name), target, WINDOWS ? "junction" : "dir");
  }
}

function backendEnv(dataRoot) {
  const models = path.join(payload, "models");
  const hfHome = path.join(dataRoot, "hf");
  prepareModelCache(hfHome);
  const cache = path.join(dataRoot, "cache");
  const extraPath = [path.join(payload, "deno"), path.join(payload, "ffmpeg"), path.join(payload, "bin")];
  return {
    ...process.env,
    PATH: [...extraPath, process.env.PATH || ""].join(path.delimiter),
    PYTHONUTF8: "1",
    PYTHONIOENCODING: "utf-8",
    PYTHONDONTWRITEBYTECODE: "1",
    PYTHONNOUSERSITE: "1",
    HF_HOME: hfHome,
    HF_HUB_DISABLE_TELEMETRY: "1",
    STEM_SPLITTER_CACHE: path.join(models, "stem_splitter", "weights"),
    TORCHINDUCTOR_CACHE_DIR: path.join(cache, "torchinductor"),
    TRITON_CACHE_DIR: path.join(cache, "triton"),
    PYTORCH_ENABLE_MPS_FALLBACK: "1",
    PS_DATA_DIR: path.join(dataRoot, "data"),
    PS_MODELS_DIR: path.join(dataRoot, "models"),
    PS_TSUMUGI_DIR: path.join(models, "tsumugi"),
    PS_COVER_DIR: path.join(models, "tsumugi-piano-cover"),
    PS_SCRIPTS_DIR: path.join(payload, "backend", "model_scripts"),
    PS_MODEL_PYTHON: pythonPath(),
    PS_FFMPEG_BIN: path.join(payload, "ffmpeg", `ffmpeg${EXE}`),
    PS_FFPROBE_BIN: path.join(payload, "ffmpeg", `ffprobe${EXE}`),
    PS_YTDLP_BIN: path.join(payload, "bin", `yt-dlp${EXE}`),
    PS_SANDBOX: "basic",
    PS_PDF_RENDERER: "none",
    PS_AUDIO_IO: "soundfile",
    PS_MODELS_OFFLINE: "1",
    PS_LOG_FORMAT: "console",
    PS_MAX_DURATION_SECONDS: "1200",
  };
}

async function startBackend() {
  const port = await freePort();
  const dataRoot = app.getPath("userData");
  const logPath = path.join(dataRoot, "backend.log");
  fs.mkdirSync(dataRoot, { recursive: true });
  const log = fs.openSync(logPath, "a");
  backend = spawn(
    pythonPath(),
    ["-m", "app.desktop", "--host", "127.0.0.1", "--port", String(port), "--web", path.join(payload, "web")],
    {
      cwd: path.join(payload, "backend"),
      env: backendEnv(dataRoot),
      stdio: ["ignore", log, log],
      detached: !WINDOWS, // its own process group on macOS, so quitting takes down the model programs too
      windowsHide: true,
    },
  );
  backend.on("exit", (code) => {
    if (!quitting) {
      dialog.showErrorBox("PianoScribe stopped", `The PianoScribe service exited (code ${code}). Details: ${logPath}`);
      app.quit();
    }
  });
  base = `http://127.0.0.1:${port}`;
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${base}/api/health`);
      if (response.ok) return base;
    } catch {
      // not listening yet
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`The PianoScribe service did not start. Details: ${logPath}`);
}

function stopBackend() {
  quitting = true;
  if (!backend || backend.exitCode !== null) return;
  if (WINDOWS) {
    spawnSync("taskkill", ["/PID", String(backend.pid), "/T", "/F"], { windowsHide: true });
  } else {
    try {
      process.kill(-backend.pid, "SIGKILL");
    } catch {
      // already gone
    }
  }
}

// The score page's print layout, printed by Chromium: the same rendering as the web viewer.
async function renderPdf(jobId) {
  const win = new BrowserWindow({ show: false, webPreferences: { sandbox: true } });
  try {
    await win.loadURL(`${base}/api/jobs/${jobId}/artifacts/score.html`);
    await win.webContents.executeJavaScript("document.fonts.ready.then(() => true)");
    return await win.webContents.printToPDF({
      pageSize: "A4",
      printBackground: true,
      margins: { marginType: "none" },
      preferCSSPageSize: true,
    });
  } finally {
    win.destroy();
  }
}

ipcMain.handle("save-pdf", async (event, jobId, title) => {
  if (!/^[0-9a-f]{32}$/.test(jobId)) return { saved: false, error: "invalid job" };
  try {
    const pdf = await renderPdf(jobId);
    const safe = title.replace(/[\\/:*?"<>|]+/g, " ").trim() || "score";
    const choice = await dialog.showSaveDialog(BrowserWindow.fromWebContents(event.sender), {
      defaultPath: path.join(app.getPath("documents"), `${safe}.pdf`),
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });
    if (choice.canceled || !choice.filePath) return { saved: false };
    fs.writeFileSync(choice.filePath, pdf);
    return { saved: true, path: choice.filePath };
  } catch (error) {
    return { saved: false, error: String(error && error.message ? error.message : error) };
  }
});

ipcMain.on("open-external", (_event, url) => {
  if (/^https:\/\//.test(url)) shell.openExternal(url);
});

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 860,
    minWidth: 420,
    minHeight: 520,
    title: "PianoScribe",
    autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, "preload.cjs"), sandbox: true, contextIsolation: true },
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https:\/\//.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(base)) {
      event.preventDefault();
      if (/^https:\/\//.test(url)) shell.openExternal(url);
    }
  });
  return win;
}

// Node's fetch can pick an idle keep-alive connection just as the server closes it; retry such requests.
async function call(url, init) {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await fetch(url, init);
    } catch (error) {
      if (attempt >= 4) throw error;
      await new Promise((r) => setTimeout(r, 500 * attempt));
    }
  }
}

// Headless check used by the build: transcribe a recording through the real app and print its PDF.
async function smokeTest(audio, outDir, solo) {
  const started = Date.now();
  const bytes = fs.readFileSync(audio);
  const query = new URLSearchParams({ filename: path.basename(audio), solo_piano: solo ? "true" : "false" });
  const submitted = await (await call(`${base}/api/jobs/upload?${query}`, { method: "POST", body: bytes })).json();
  const jobId = submitted.job.id;
  let job = submitted.job;
  while (!["succeeded", "failed"].includes(job.status)) {
    if (Date.now() - started > 3 * 3600_000) throw new Error("timed out");
    await new Promise((r) => setTimeout(r, 2000));
    job = await (await call(`${base}/api/jobs/${jobId}`)).json();
  }
  fs.mkdirSync(outDir, { recursive: true });
  if (job.status !== "succeeded") throw new Error(`job failed: ${JSON.stringify(job.error)}`);
  for (const name of ["score.musicxml", "performance.mid"]) {
    const data = Buffer.from(await (await call(`${base}/api/jobs/${jobId}/artifacts/${name}`)).arrayBuffer());
    fs.writeFileSync(path.join(outDir, name), data);
  }
  const pdf = await renderPdf(jobId);
  fs.writeFileSync(path.join(outDir, "score.pdf"), pdf);
  // The interface itself: it loads in the shell and shows the desktop-only Models link.
  const ui = new BrowserWindow({ show: false, webPreferences: { preload: path.join(__dirname, "preload.cjs"), sandbox: true } });
  await ui.loadURL(base);
  const uiCheck = await ui.webContents.executeJavaScript(`new Promise((resolve) => {
    const deadline = Date.now() + 20000;
    const look = () => {
      const models = [...document.querySelectorAll("a")].some((a) => a.textContent.trim() === "Models");
      if (models || Date.now() > deadline) resolve({ models, bridge: typeof window.pianoscribe === "object", title: document.title });
      else setTimeout(look, 200);
    };
    look();
  })`);
  ui.destroy();
  if (!uiCheck.models || !uiCheck.bridge) throw new Error(`interface check failed: ${JSON.stringify(uiCheck)}`);
  let state = await (await call(`${base}/api/desktop/state`)).json();
  for (let i = 0; i < 90 && state.system.checking; i += 1) {
    await new Promise((r) => setTimeout(r, 2000));
    state = await (await call(`${base}/api/desktop/state`)).json();
  }
  const report = {
    seconds: Math.round((Date.now() - started) / 1000),
    user_data: app.getPath("userData"),
    result: job.result,
    timings: job.timings,
    system: state.system,
    interface: uiCheck,
    pdf_bytes: pdf.length,
  };
  fs.writeFileSync(path.join(outDir, "smoke.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    const [win] = BrowserWindow.getAllWindows();
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });
  app.on("before-quit", stopBackend);
  // In smoke-test mode the only windows are the hidden ones that print PDFs; closing them must not quit.
  app.on("window-all-closed", () => {
    if (!arg("smoke-test")) app.quit();
  });
  app.whenReady().then(async () => {
    const smokeAudio = arg("smoke-test");
    let splash = null;
    if (!smokeAudio) {
      splash = new BrowserWindow({ width: 360, height: 260, frame: false, resizable: false, show: true });
      splash.loadFile(path.join(__dirname, "splash.html"));
    }
    try {
      await startBackend();
    } catch (error) {
      if (splash) splash.destroy();
      if (smokeAudio) {
        console.error(String(error));
        app.exit(2);
        return;
      }
      dialog.showErrorBox("PianoScribe could not start", String(error && error.message ? error.message : error));
      app.quit();
      return;
    }
    if (smokeAudio) {
      try {
        await smokeTest(smokeAudio, arg("smoke-out") || path.join(app.getPath("temp"), "pianoscribe-smoke"), process.argv.includes("--smoke-solo"));
        stopBackend();
        app.exit(0);
      } catch (error) {
        console.error(String(error && error.stack ? error.stack : error));
        const logPath = path.join(app.getPath("userData"), "backend.log");
        try {
          console.error(`--- ${logPath}\n` + fs.readFileSync(logPath, "utf8").split("\n").slice(-40).join("\n"));
        } catch {
          console.error(`no log at ${logPath}`);
        }
        stopBackend();
        app.exit(1);
      }
      return;
    }
    const win = createWindow();
    await win.loadURL(base);
    if (splash) splash.destroy();
  });
}
