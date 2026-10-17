import type { InstructionInfo } from "./instructions";
import type { BuildMeta, BuildPartIn, NearbySets, RecipeBuilds, SetBuild } from "./build";
import type { Idea, IdeaPart, MatchPart, Resolved, Scan, SearchResult, SetMatch } from "./types";

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

/** Upload a photo of the empty background; the scan that passes the returned id is compared with it. */
export async function uploadReference(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  return (await call<{ reference_id: string }>("/api/reference", { method: "POST", body: form })).reference_id;
}

export async function startScan(file: File, referenceId?: string | null): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  if (referenceId) form.append("reference_id", referenceId);
  return (await call<{ job_id: string }>("/api/scan", { method: "POST", body: form })).job_id;
}

export const getScan = (id: string) => call<Scan>(`/api/scan/${id}`);

/** Merge two detected pieces (their local piece numbers in this photo) into one; the merged crop is recognised again. */
export const mergePieces = (jobId: string, indices: [number, number]) =>
  call<Scan>(`/api/scan/${jobId}/merge`, json({ indices }));

/** Cut one detected piece along a line (x0, y0, x1, y1 in the photo's own pixels); each part is recognised. */
export const splitPiece = (jobId: string, index: number, line: [number, number, number, number]) =>
  call<Scan>(`/api/scan/${jobId}/split`, json({ index, line }));

export const resolveParts = async (parts: string[], colors: Record<string, string | null> = {}) =>
  (await call<{ results: Record<string, Resolved> }>("/api/parts/resolve", json({ parts, colors }))).results;

export const searchParts = async (q: string) =>
  (await call<{ results: SearchResult[] }>(`/api/parts/search?q=${encodeURIComponent(q)}`)).results;

export const matchSets = (parts: MatchPart[], matchColor: boolean) =>
  call<{ sets: SetMatch[]; unmapped: string[] }>(
    "/api/match",
    json({ parts, match_color: matchColor, max_parts: 150 }),
  );

export const getIdeas = (parts: IdeaPart[], matchColor = true) => call<{ ideas: Idea[] }>("/api/ideas", json({ parts, match_color: matchColor }));

/** A 3D build with step-by-step placement for the user's pieces (AI), optionally for a chosen idea. Takes 10-40 s. */
export const createAiBuild = (parts: BuildPartIn[], idea?: { name: string; description: string }, matchColor = true) =>
  call<BuildMeta>("/api/builds/ai", json({ parts, idea: idea ?? null, match_color: matchColor }));

/** Builds the code designs from the user's pieces (recipes), named by the AI when it is available. Same pieces, same builds. */
export const getRecipeBuilds = (parts: BuildPartIn[], useAi = true, matchColor = true) =>
  call<RecipeBuilds>("/api/builds/recipes", json({ parts, use_ai: useAi, match_color: matchColor }));

/** Official sets the pieces can build now, and sets that are almost there (with the missing parts). */
export const getNearbySets = (parts: BuildPartIn[], matchColor?: boolean) => call<NearbySets>("/api/sets/nearby", json({ parts, match_color: matchColor ?? null }));

/** An official set: its LDraw model (when one exists), the parts list with have/missing, and LEGO's instructions link. */
export const getSetBuild = (setNum: string, parts: BuildPartIn[], setName?: string) =>
  call<SetBuild>(`/api/sets/${encodeURIComponent(setNum)}/build`, json({ parts, set_name: setName ?? null }));

/** Instruction state per set (3D model / official LEGO link / none). Answers from caches; `pending` entries are filled in by a background lookup. */
export const getSetInstructions = async (setNums: string[]) =>
  (await call<{ sets: Record<string, InstructionInfo> }>("/api/sets/instructions", json({ set_nums: setNums }))).sets;
