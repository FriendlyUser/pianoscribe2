// Recently viewed jobs, kept in this browser only.

export interface RecentJob {
  id: string;
  title: string;
  at: number;
}

const KEY = "pianoscribe.recent";
const LIMIT = 6;

export function loadRecent(): RecentJob[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(parsed) ? (parsed as RecentJob[]).slice(0, LIMIT) : [];
  } catch {
    return [];
  }
}

export function rememberJob(id: string, title: string): RecentJob[] {
  const next = [{ id, title, at: Date.now() }, ...loadRecent().filter((j) => j.id !== id)].slice(0, LIMIT);
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* storage unavailable: recents are a convenience only */
  }
  return next;
}

export function forgetJob(id: string): RecentJob[] {
  const next = loadRecent().filter((j) => j.id !== id);
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* ignore */
  }
  return next;
}
