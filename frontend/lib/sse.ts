import type { BuildMeta } from "./build";

/** What the server sends while it designs: see backend/designs/pipeline.py. */
export type DesignEvent =
  | { event: "start"; data: { pieces: number; kinds: number; cells: number; t_ms: number } }
  | { event: "design"; data: { build: BuildMeta & { design?: { concept: string; coverage: number; use_share: number; low_use: boolean } }; t_ms: number } }
  | { event: "verdict"; data: { id: string; title: string; ok: boolean; choice: string | null; t_ms: number } }
  | { event: "note"; data: { kind: string; message: string; reasons?: { concept: string; why: string }[] } }
  | { event: "done"; data: { designs: number; calls: number; t_ms?: number } };

/** Parse one server-sent-events frame ("event: x\ndata: {...}") */
export function parseFrame(frame: string): DesignEvent | null {
  let event = "";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!event || !data) return null;
  try {
    return { event, data: JSON.parse(data) } as DesignEvent;
  } catch {
    return null;
  }
}

