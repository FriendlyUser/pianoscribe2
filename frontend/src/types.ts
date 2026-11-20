// Mirrors the backend job model (backend/app/jobs/models.py).

export type JobStatus = "queued" | "running" | "succeeded" | "failed";
export type Stage =
  | "queued"
  | "ingesting"
  | "separating"
  | "transcribing"
  | "arranging"
  | "engraving"
  | "ready"
  | "failed";

export interface JobSource {
  kind: "upload" | "youtube";
  filename: string | null;
  video_id: string | null;
  size_bytes: number | null;
}

export interface JobError {
  code: string;
  message: string;
  retryable: boolean;
}

export interface JobResult {
  /** "arrangement": a song arranged for piano. "transcription": a solo piano recording written down. */
  mode: "arrangement" | "transcription";
  key: string;
  key_changes: number;
  time_signature: string;
  tempo_bpm: number;
  /** Note value the tempo counts: "quarter", "dotted quarter", "eighth", ... */
  tempo_unit: string;
  measures: number;
  notes: number;
  pages: number;
  duration_seconds: number;
  /** Share (0..1) of the detected melody that the arrangement plays in time; null for solo piano. */
  melody_fidelity: number | null;
  artifacts: Record<string, number>;
}

export interface Job {
  id: string;
  status: JobStatus;
  stage: Stage;
  progress: number;
  message: string;
  title: string;
  source: JobSource;
  options: { solo_piano: boolean };
  error: JobError | null;
  result: JobResult | null;
  attempts: number;
  dead_lettered: boolean;
  timings: Record<string, number>;
  created_at: number;
  updated_at: number;
  finished_at: number | null;
}

export interface Submission {
  job: Job;
  cached: boolean;
  deduplicated: boolean;
}

export interface ClientConfig {
  max_upload_bytes: number;
  max_duration_seconds: number;
  allowed_extensions: string[];
  youtube_enabled: boolean;
  result_retention_seconds: number;
  /** Desktop app only: shows the model settings. Absent from the web server's answer. */
  desktop?: boolean;
  /** False when PDFs come from the desktop shell's print engine instead of the server. */
  server_pdf?: boolean;
}

export const DEFAULT_CONFIG: ClientConfig = {
  max_upload_bytes: 200 * 1024 * 1024,
  max_duration_seconds: 600,
  allowed_extensions: ["mp3", "wav", "mp4", "mov"],
  youtube_enabled: true,
  result_retention_seconds: 7 * 24 * 3600,
};
