import { useCallback, useId, useRef, useState, type DragEvent, type FormEvent } from "react";
import { ApiError, submitUpload, submitYouTube } from "../api";
import type { ClientConfig, Submission } from "../types";
import { formatBytes, parseYouTubeUrl, validateFile } from "../validation";

type Mode = "upload" | "youtube";

interface Props {
  config: ClientConfig;
  onSubmitted: (submission: Submission) => void;
}

export function SubmitForm({ config, onSubmitted }: Props) {
  const [mode, setMode] = useState<Mode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [solo, setSolo] = useState(false);
  const [video, setVideo] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const errorId = useId();
  const formats = config.allowed_extensions.map((e) => e.toUpperCase()).join(", ");
  const minutes = Math.floor(config.max_duration_seconds / 60);

  const chooseFile = useCallback(
    async (candidate: File | undefined) => {
      if (!candidate) return;
      const check = await validateFile(candidate, config);
      if (check.ok) {
        setFile(candidate);
        setError(null);
      } else {
        setFile(null);
        setError(check.message);
      }
    },
    [config],
  );

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files.length > 1) {
      setError("Drop one file at a time.");
      return;
    }
    void chooseFile(event.dataTransfer.files[0]);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (busy) return;
    setError(null);
    try {
      if (mode === "upload") {
        if (!file) {
          setError("Choose a file to transcribe.");
          return;
        }
        if (video && !/\.(mp4|mov)$/i.test(file.name)) {
          setError("The piano-video option reads the picture: choose an MP4 or MOV video.");
          return;
        }
        setBusy(true);
        setUploading(0);
        onSubmitted(await submitUpload(file, { soloPiano: solo && !video, pianoVideo: video }, setUploading));
      } else {
        const parsed = parseYouTubeUrl(url);
        if (!parsed.ok) {
          setError(parsed.message);
          return;
        }
        setBusy(true);
        onSubmitted(await submitYouTube(url.trim(), { soloPiano: solo && !video, pianoVideo: video }));
      }
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
      setUploading(null);
    }
  };

  return (
    <form className="card submit" onSubmit={submit} noValidate>
      <div className="tabs" role="tablist" aria-label="Source">
        <button type="button" role="tab" aria-selected={mode === "upload"} onClick={() => { setMode("upload"); setError(null); }}>
          Upload a file
        </button>
        {config.youtube_enabled && (
          <button type="button" role="tab" aria-selected={mode === "youtube"} onClick={() => { setMode("youtube"); setError(null); }}>
            YouTube link
          </button>
        )}
      </div>

      {mode === "upload" ? (
        <div
          className={`dropzone${dragging ? " dragging" : ""}${file ? " has-file" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => input.current?.click()}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.current?.click(); } }}
          role="button"
          tabIndex={0}
          aria-describedby={error ? errorId : undefined}
        >
          <input
            ref={input}
            type="file"
            hidden
            accept={config.allowed_extensions.map((e) => `.${e}`).join(",")}
            onChange={(e) => { void chooseFile(e.target.files?.[0]); e.target.value = ""; }}
          />
          {file ? (
            <>
              <strong className="file-name">{file.name}</strong>
              <span className="hint">{formatBytes(file.size)} · click or drop to replace</span>
            </>
          ) : (
            <>
              <strong>Drop an audio or video file here</strong>
              <span className="hint">
                or click to browse · {formats} · up to {formatBytes(config.max_upload_bytes)} and {minutes} minutes
              </span>
            </>
          )}
        </div>
      ) : (
        <label className="field">
          <span>YouTube video link</span>
          <input
            type="url"
            inputMode="url"
            placeholder="https://www.youtube.com/watch?v=…"
            value={url}
            onChange={(e) => { setUrl(e.target.value); setError(null); }}
            aria-invalid={error !== null}
            aria-describedby={error ? errorId : undefined}
            autoComplete="off"
            spellCheck={false}
          />
          <span className="hint">Videos up to {minutes} minutes. Playlists and live streams are not supported.</span>
        </label>
      )}

      <label className="toggle">
        <input type="checkbox" checked={video} onChange={(e) => setVideo(e.target.checked)} />
        <span>
          <strong>This is a piano tutorial video</strong>
          <span className="hint">
            Notes fall onto a keyboard along the bottom of the picture. They are read from the picture itself: every
            note, its length and the hand that plays it are written down as shown.
          </span>
        </span>
      </label>

      <label className="toggle">
        <input type="checkbox" checked={solo && !video} disabled={video} onChange={(e) => setSolo(e.target.checked)} />
        <span>
          <strong>This is already solo piano</strong>
          <span className="hint">
            The notes are written down as played. Leave it off for songs with a band, strings or vocals: those are
            arranged into a part one pianist can play.
          </span>
        </span>
      </label>

      {error && (
        <p className="form-error" id={errorId} role="alert">
          {error}
        </p>
      )}

      {uploading !== null && (
        <div className="upload-progress" aria-live="polite">
          <progress max={1} value={uploading} aria-label="Upload progress" />
          <span>Uploading {Math.round(uploading * 100)}%</span>
        </div>
      )}

      <button type="submit" className="button primary wide" disabled={busy}>
        {busy ? "Submitting…" : "Transcribe"}
      </button>
    </form>
  );
}
