import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "./types";
import { parseYouTubeUrl, sniffKind, validateFileMeta } from "./validation";

const bytes = (text: string) => Uint8Array.from(text, (c) => c.charCodeAt(0));

describe("parseYouTubeUrl", () => {
  it.each([
    ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"],
    ["https://youtu.be/dQw4w9WgXcQ?t=10", "dQw4w9WgXcQ"],
    ["https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=x", "dQw4w9WgXcQ"],
    ["https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"],
    ["  http://m.youtube.com/embed/dQw4w9WgXcQ  ", "dQw4w9WgXcQ"],
  ])("accepts %s", (url, id) => {
    expect(parseYouTubeUrl(url)).toEqual({ ok: true, value: id });
  });

  it.each([
    "",
    "not a url",
    "ftp://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://evil.example/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com.evil.example/watch?v=dQw4w9WgXcQ",
    "https://user:pw@www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com:8443/watch?v=dQw4w9WgXcQ",
    "http://169.254.169.254/latest/meta-data",
    "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/playlist?list=PL123",
  ])("rejects %s", (url) => {
    expect(parseYouTubeUrl(url).ok).toBe(false);
  });
});

describe("validateFileMeta", () => {
  it("accepts supported files within the size limit", () => {
    expect(validateFileMeta({ name: "Take 1.MP3", size: 1024 }, DEFAULT_CONFIG).ok).toBe(true);
  });
  it("rejects unsupported extensions, empty and oversized files", () => {
    expect(validateFileMeta({ name: "notes.pdf", size: 10 }, DEFAULT_CONFIG).ok).toBe(false);
    expect(validateFileMeta({ name: "a.wav", size: 0 }, DEFAULT_CONFIG).ok).toBe(false);
    expect(validateFileMeta({ name: "a.wav", size: DEFAULT_CONFIG.max_upload_bytes + 1 }, DEFAULT_CONFIG).ok).toBe(false);
  });
});

describe("sniffKind", () => {
  it("recognises container signatures", () => {
    expect(sniffKind(bytes("RIFF\x00\x00\x00\x00WAVEfmt "))).toBe("wav");
    expect(sniffKind(bytes("\x00\x00\x00\x18ftypmp42"))).toBe("isobmff");
    expect(sniffKind(bytes("ID3\x04\x00\x00\x00\x00\x00\x00"))).toBe("mp3");
    expect(sniffKind(Uint8Array.from([0xff, 0xfb, 0x90, 0x64]))).toBe("mp3");
  });
  it("rejects unknown content", () => {
    expect(sniffKind(bytes("%PDF-1.7 not audio"))).toBeNull();
    expect(sniffKind(new Uint8Array())).toBeNull();
  });
});
