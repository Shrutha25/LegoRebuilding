import type { MatchPart, Resolved, Scan, SearchResult, SetMatch } from "./types";

/**
 * Backend base URL. NEXT_PUBLIC_API_URL wins; otherwise assume the backend runs on port 8000
 * of whatever host served this page, so a phone opening http://<laptop-ip>:3000 talks to
 * http://<laptop-ip>:8000 without any configuration.
 */
export function apiBase(): string {
  const fromEnv = process.env.NEXT_PUBLIC_API_URL;
  if (fromEnv) return fromEnv.replace(/\/$/, "");
  if (typeof window !== "undefined") return `${window.location.protocol}//${window.location.hostname}:8000`;
  return "http://localhost:8000";
}

export const absUrl = (path: string | null | undefined) =>
  !path ? "" : path.startsWith("http") ? path : `${apiBase()}${path}`;

/** Brickognize thumbnails are blocked by browsers (wrong content type), so route them via our proxy. */
export const imgSrc = (url: string | null | undefined) =>
  !url ? "" : url.startsWith("https://storage.googleapis.com/brickognize-static/")
    ? `${apiBase()}/api/img?url=${encodeURIComponent(url)}`
    : url;

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${apiBase()}${path}`, init);
  } catch {
    throw new ApiError(0, "Can't reach the server. Check your connection and that the backend is running.");
  }
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).detail;
    } catch {}
    throw new ApiError(res.status, typeof detail === "string" && detail ? detail : `Request failed (${res.status})`);
  }
  return res.json();
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export async function startScan(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  return (await call<{ job_id: string }>("/api/scan", { method: "POST", body: form })).job_id;
}

export const getScan = (id: string) => call<Scan>(`/api/scan/${id}`);

export const resolveParts = async (parts: string[]) =>
  (await call<{ results: Record<string, Resolved> }>("/api/parts/resolve", json({ parts }))).results;

export const searchParts = async (q: string) =>
  (await call<{ results: SearchResult[] }>(`/api/parts/search?q=${encodeURIComponent(q)}`)).results;

export const matchSets = (parts: MatchPart[], matchColor: boolean) =>
  call<{ sets: SetMatch[]; unmapped: string[] }>(
    "/api/match",
    json({ parts, match_color: matchColor, max_parts: 150 }),
  );
