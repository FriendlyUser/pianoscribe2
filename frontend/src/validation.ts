// Upfront client-side validation. These checks mirror the server's rules so
// users get immediate feedback; the server remains the source of truth.

import type { ClientConfig } from "./types";

export type Validation<T> = { ok: true; value: T } | { ok: false; message: string };

const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
const WATCH_HOSTS = new Set(["youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"]);
const PATH_PREFIXES = ["/shorts/", "/embed/", "/live/", "/v/"];
const INVALID_URL = "Enter a valid YouTube video link, e.g. https://www.youtube.com/watch?v=…";

/** Extract the video id from a YouTube URL, accepting only real YouTube hosts. */
export function parseYouTubeUrl(input: string): Validation<string> {
  const text = input.trim();
  if (!text) return { ok: false, message: "Paste a YouTube link first." };
  if (text.length > 2048 || /\s/.test(text)) return { ok: false, message: INVALID_URL };
  let url: URL;
  try {
    url = new URL(text);
  } catch {
    return { ok: false, message: INVALID_URL };
  }
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.port) {
    return { ok: false, message: INVALID_URL };
  }
  const host = url.hostname.toLowerCase().replace(/\.$/, "");
  let candidate: string | undefined;
  if (host === "youtu.be") {
    candidate = url.pathname.replace(/^\/+/, "").split("/")[0];
  } else if (WATCH_HOSTS.has(host)) {
    if (url.pathname === "/watch" || url.pathname === "/watch/") {
      candidate = url.searchParams.get("v") ?? undefined;
    } else {
      const prefix = PATH_PREFIXES.find((p) => url.pathname.startsWith(p));
      if (prefix) candidate = url.pathname.slice(prefix.length).split("/")[0];
    }
  } else {
    return { ok: false, message: "Only YouTube links are supported (youtube.com or youtu.be)." };
  }
  if (!candidate || !VIDEO_ID.test(candidate)) return { ok: false, message: INVALID_URL };
  return { ok: true, value: candidate };
}

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot + 1).toLowerCase();
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export type MediaKind = "mp3" | "wav" | "isobmff";

const EXTENSION_KIND: Record<string, MediaKind> = { mp3: "mp3", wav: "wav", mp4: "isobmff", mov: "isobmff" };
const ascii = (bytes: Uint8Array, start: number, end: number) => String.fromCharCode(...bytes.subarray(start, end));

/** Identify a container from its first bytes (same signatures the server checks). */
export function sniffKind(bytes: Uint8Array): MediaKind | null {
  if (bytes.length >= 12 && ascii(bytes, 0, 4) === "RIFF" && ascii(bytes, 8, 12) === "WAVE") return "wav";
  if (bytes.length >= 8) {
    const atom = ascii(bytes, 4, 8);
    if (["ftyp", "moov", "mdat", "wide", "free", "skip", "pnot"].includes(atom)) return "isobmff";
  }
  if (bytes.length >= 3 && ascii(bytes, 0, 3) === "ID3") return "mp3";
  if (bytes.length >= 2 && bytes[0] === 0xff && ((bytes[1] ?? 0) & 0xe0) === 0xe0) return "mp3";
  return null;
}

interface FileLike {
  name: string;
  size: number;
}

/** Synchronous checks: extension allow-list and size limits. */
export function validateFileMeta(file: FileLike, config: ClientConfig): Validation<string> {
  const extension = extensionOf(file.name);
  if (!config.allowed_extensions.includes(extension)) {
    const list = config.allowed_extensions.map((e) => e.toUpperCase()).join(", ");
    return { ok: false, message: `Unsupported file type. Choose one of: ${list}.` };
  }
  if (file.size === 0) return { ok: false, message: "This file is empty." };
  if (file.size > config.max_upload_bytes) {
    return {
      ok: false,
      message: `This file is ${formatBytes(file.size)}; the limit is ${formatBytes(config.max_upload_bytes)}.`,
    };
  }
  return { ok: true, value: extension };
}

/** Full validation including a look at the file's leading bytes. */
export async function validateFile(file: File, config: ClientConfig): Promise<Validation<string>> {
  const meta = validateFileMeta(file, config);
  if (!meta.ok) return meta;
  const head = new Uint8Array(await file.slice(0, 16).arrayBuffer());
  const kind = sniffKind(head);
  if (kind === null) {
    return { ok: false, message: "This file does not look like audio or video. It may be corrupted or mislabeled." };
  }
  if (kind !== EXTENSION_KIND[meta.value]) {
    return { ok: false, message: `The file's contents do not match its .${meta.value} extension.` };
  }
  return meta;
}
