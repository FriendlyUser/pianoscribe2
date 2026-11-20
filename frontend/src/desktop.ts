// Desktop app only: model settings API and the bridge to the Electron shell (desktop/preload.cjs).

export type Device = "auto" | "cuda" | "mps" | "cpu";
export type SoloTranscriber = "tsumugi" | "muscriptor-small" | "muscriptor-medium" | "muscriptor-large";

export interface DesktopSettings {
  model_device: Device;
  solo_transcriber: SoloTranscriber;
  cover_source_cfg: number;
  cover_density: number;
  cover_seed: number;
  solo_section_bars: number;
}

export interface ModelInfo {
  id: string;
  stage: "solo" | "song" | "analysis" | "notation";
  name: string;
  repo: string;
  size_gb: number;
  licence: string;
  bundled: boolean;
  gated: boolean;
  measured: string;
  setting: string | null;
  installed: boolean;
}

export interface SystemInfo {
  checking?: boolean;
  torch?: string;
  cuda?: boolean;
  gpu?: string | null;
  mps?: boolean;
  error?: string;
}

export interface DesktopState {
  settings: DesktopSettings;
  models: ModelInfo[];
  system: SystemInfo;
}

interface ShellBridge {
  savePdf(jobId: string, title: string): Promise<{ saved: boolean; path?: string; error?: string }>;
  openExternal(url: string): void;
}

export const shell = (): ShellBridge | undefined => (window as unknown as { pianoscribe?: ShellBridge }).pianoscribe;

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    throw new Error(typeof detail === "string" ? detail : "The desktop service did not answer.");
  }
  return body as T;
}

export const getDesktopState = () => call<DesktopState>("/api/desktop/state");

export const saveDesktopSettings = (settings: DesktopSettings) =>
  call<DesktopState>("/api/desktop/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });

export const downloadModel = (id: string, token: string) =>
  call<DesktopState>(`/api/desktop/models/${id}/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: token || null }),
  });
