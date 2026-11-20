import { useEffect, useState } from "react";
import {
  downloadModel,
  getDesktopState,
  saveDesktopSettings,
  shell,
  type DesktopSettings,
  type DesktopState,
  type Device,
  type ModelInfo,
  type SoloTranscriber,
} from "../desktop";

const DEVICES: { value: Device; label: string }[] = [
  { value: "auto", label: "Automatic" },
  { value: "cuda", label: "NVIDIA GPU" },
  { value: "mps", label: "Apple GPU" },
  { value: "cpu", label: "Processor only" },
];

function deviceNote(state: DesktopState): string {
  const s = state.system;
  if (s.checking) return "Checking which accelerator is available…";
  if (s.error) return "Could not check the accelerator; the models will choose for themselves.";
  if (s.cuda) return `Found ${s.gpu ?? "an NVIDIA GPU"}. A four-minute song takes a few minutes.`;
  if (s.mps) return "Found Apple's GPU.";
  return "No GPU found: everything runs on the processor. A four-minute song can take close to an hour.";
}

interface Props {
  onClose: () => void;
}

export function ModelSettings({ onClose }: Props) {
  const [state, setState] = useState<DesktopState | null>(null);
  const [draft, setDraft] = useState<DesktopSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [token, setToken] = useState("");

  useEffect(() => {
    getDesktopState()
      .then((next) => {
        setState(next);
        setDraft(next.settings);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!state?.system.checking) return;
    const timer = setTimeout(() => getDesktopState().then(setState).catch(() => undefined), 3000);
    return () => clearTimeout(timer);
  }, [state]);

  if (!state || !draft) {
    return <section className="card loading">{error ?? "Loading model settings…"}</section>;
  }

  const update = (patch: Partial<DesktopSettings>) => {
    setDraft({ ...draft, ...patch });
    setSaved(false);
  };
  const save = async () => {
    setError(null);
    try {
      const next = await saveDesktopSettings(draft);
      setState(next);
      setDraft(next.settings);
      setSaved(true);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const download = async (model: ModelInfo) => {
    setError(null);
    setBusy(model.id);
    try {
      setState(await downloadModel(model.id, token));
      setToken("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };
  const transcribers = state.models.filter((m) => m.stage === "solo");
  const fixed = state.models.filter((m) => m.stage !== "solo");
  const needsToken = transcribers.some((m) => m.gated && !m.installed);

  return (
    <section className="card models" aria-labelledby="models-heading">
      <div className="models-head">
        <h2 id="models-heading">Models</h2>
        <button type="button" className="button ghost" onClick={onClose}>
          Done
        </button>
      </div>
      <p className="hint">Changes apply to the next recording you submit. Scores already made are kept.</p>

      <fieldset className="models-group">
        <legend>Run the models on</legend>
        <div className="segmented" role="radiogroup">
          {DEVICES.map((d) => (
            <label key={d.value} className={draft.model_device === d.value ? "on" : ""}>
              <input
                type="radio"
                name="device"
                value={d.value}
                checked={draft.model_device === d.value}
                onChange={() => update({ model_device: d.value })}
              />
              {d.label}
            </label>
          ))}
        </div>
        <p className="hint">{deviceNote(state)}</p>
      </fieldset>

      <fieldset className="models-group">
        <legend>Solo piano transcription</legend>
        <ul className="model-list">
          {transcribers.map((m) => (
            <li key={m.id} className={draft.solo_transcriber === m.setting ? "chosen" : ""}>
              <label className="model-choice">
                <input
                  type="radio"
                  name="transcriber"
                  disabled={!m.installed}
                  checked={draft.solo_transcriber === m.setting}
                  onChange={() => update({ solo_transcriber: m.setting as SoloTranscriber })}
                />
                <span>
                  <strong>{m.name}</strong>
                  {m.id === "tsumugi" && <span className="badge">Recommended</span>}
                  <span className="hint">{m.measured}</span>
                  <span className="hint">
                    {m.size_gb.toFixed(1)} GB · {m.licence}
                    {m.bundled ? " · included" : m.installed ? " · downloaded" : ""}
                  </span>
                </span>
              </label>
              {!m.installed && (
                <button type="button" className="button" disabled={busy !== null} onClick={() => download(m)}>
                  {busy === m.id ? "Downloading…" : "Download"}
                </button>
              )}
            </li>
          ))}
        </ul>
        {needsToken && (
          <label className="field">
            Hugging Face token for gated models
            <input
              type="password"
              autoComplete="off"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="hf_…"
            />
            <span className="hint">
              MuScriptor's authors require you to accept their terms on the{" "}
              <a
                href="https://huggingface.co/MuScriptor/muscriptor-medium"
                onClick={(e) => {
                  const bridge = shell();
                  if (bridge) {
                    e.preventDefault();
                    bridge.openExternal("https://huggingface.co/MuScriptor/muscriptor-medium");
                  }
                }}
              >
                model page
              </a>{" "}
              first. The token is used for the download only and is not stored.
            </span>
          </label>
        )}
      </fieldset>

      <fieldset className="models-group">
        <legend>Arranging songs for piano</legend>
        <label className="slider">
          <span>
            Follow the recording <output>{draft.cover_source_cfg.toFixed(1)}</output>
          </span>
          <input
            type="range"
            min={1}
            max={4}
            step={0.5}
            value={draft.cover_source_cfg}
            onChange={(e) => update({ cover_source_cfg: Number(e.target.value) })}
          />
          <span className="hint">Higher keeps closer to the original; lower gives the arranger more freedom.</span>
        </label>
        <label className="slider">
          <span>
            Notes <output>{draft.cover_density.toFixed(2)}×</output>
          </span>
          <input
            type="range"
            min={0.5}
            max={1.5}
            step={0.05}
            value={draft.cover_density}
            onChange={(e) => update({ cover_density: Number(e.target.value) })}
          />
          <span className="hint">Fewer notes is easier to play.</span>
        </label>
        <div className="seed">
          <span>
            Arrangement number <strong>{draft.cover_seed + 1}</strong>
          </span>
          <button type="button" className="button" onClick={() => update({ cover_seed: draft.cover_seed + 1 })}>
            Try another arrangement
          </button>
        </div>
        <p className="hint">
          The arranger is <strong>tsumugi piano cover v2</strong>, with SheetSage2-Refined for beats and keys.
        </p>
      </fieldset>

      <details className="models-group">
        <summary>Included models</summary>
        <ul className="model-list compact">
          {fixed.map((m) => (
            <li key={m.id}>
              <span>
                <strong>{m.name}</strong>
                <span className="hint">{m.measured}</span>
                <span className="hint">
                  {m.size_gb.toFixed(1)} GB · {m.licence}
                </span>
              </span>
            </li>
          ))}
        </ul>
      </details>

      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
      <div className="models-actions">
        {saved && <span className="saved">Saved</span>}
        <button type="button" className="button primary" onClick={save}>
          Save
        </button>
      </div>
    </section>
  );
}
