import type { ClientConfig, Job, Submission } from "./types";

export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly status: number,
  ) {
    super(message);
  }
}

function toApiError(status: number, body: unknown): ApiError {
  const error = (body as { error?: { code?: string; message?: string } } | null)?.error;
  if (error?.message) return new ApiError(error.code ?? "error", error.message, status);
  if (status === 413) return new ApiError("file_too_large", "This file is larger than the server accepts.", status);
  if (status === 429) return new ApiError("rate_limited", "Too many requests. Wait a moment and try again.", status);
  return new ApiError("server_error", "The server could not process the request. Please try again.", status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError("network", "Cannot reach the server. Check your connection and try again.", 0);
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) throw toApiError(response.status, body);
  return body as T;
}

export const getConfig = () => request<ClientConfig>("/api/config");
export const getJob = (id: string) => request<Job>(`/api/jobs/${id}`);

/** How a recording is to be read: by ear (a song, or solo piano), or from the picture of a piano video. */
export interface SubmitOptions {
  soloPiano: boolean;
  pianoVideo: boolean;
}

export const submitYouTube = (url: string, options: SubmitOptions) =>
  request<Submission>("/api/jobs/youtube", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, solo_piano: options.soloPiano, piano_video: options.pianoVideo }),
  });

/** Upload a file as the raw request body, reporting upload progress (0..1). */
export function submitUpload(
  file: File,
  options: SubmitOptions,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<Submission> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const query = new URLSearchParams({
      filename: file.name,
      solo_piano: String(options.soloPiano),
      piano_video: String(options.pianoVideo),
    });
    xhr.open("POST", `/api/jobs/upload?${query.toString()}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300 && xhr.response) resolve(xhr.response as Submission);
      else reject(toApiError(xhr.status, xhr.response));
    };
    xhr.onerror = () => reject(new ApiError("network", "The upload was interrupted. Check your connection and try again.", 0));
    xhr.onabort = () => reject(new ApiError("aborted", "Upload cancelled.", 0));
    signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(file);
  });
}

export const artifactUrl = (jobId: string, name: string, download = false) =>
  `/api/jobs/${jobId}/artifacts/${name}${download ? "?download=true" : ""}`;

/**
 * Follow a job until it reaches a terminal state. Uses Server-Sent Events and
 * falls back to polling if the stream cannot be established.
 */
export function watchJob(jobId: string, onUpdate: (job: Job) => void, onError: (error: ApiError) => void): () => void {
  let closed = false;
  let source: EventSource | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let failures = 0;

  const finish = () => {
    closed = true;
    source?.close();
    if (timer) clearTimeout(timer);
  };
  const deliver = (job: Job) => {
    if (closed) return;
    onUpdate(job);
    if (job.status === "succeeded" || job.status === "failed") finish();
  };
  const poll = async () => {
    if (closed) return;
    try {
      deliver(await getJob(jobId));
      failures = 0;
    } catch (error) {
      failures += 1;
      if (error instanceof ApiError && (error.status === 404 || failures >= 5)) {
        onError(error);
        finish();
        return;
      }
    }
    if (!closed) timer = setTimeout(poll, 2000);
  };

  if (typeof EventSource === "undefined") {
    void poll();
  } else {
    source = new EventSource(`/api/jobs/${jobId}/events`);
    source.onmessage = (event) => deliver(JSON.parse(event.data as string) as Job);
    source.onerror = () => {
      // The stream also ends normally when the job finishes; only fall back if still open.
      if (closed) return;
      source?.close();
      source = null;
      void poll();
    };
  }
  return finish;
}
