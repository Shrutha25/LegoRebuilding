import { ApiError, apiBase } from "./api";
import type { BuildPartIn } from "./build";
import { parseFrame, type DesignEvent } from "./sse";

export type { DesignEvent } from "./sse";

/**
 * POST the pile and call `onEvent` for every event as it arrives. Resolves when the stream ends. A plain fetch with a stream reader (EventSource
 * cannot POST). Aborting `signal` stops the request, and the server stops designing.
 */
export async function streamDesigns(parts: BuildPartIn[], matchColor: boolean, onEvent: (e: DesignEvent) => void, signal?: AbortSignal): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${apiBase()}/api/designs/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parts, match_color: matchColor }),
      signal,
    });
  } catch (e) {
    if (signal?.aborted) return;
    throw new ApiError(0, "Can't reach the server. Check your connection and that the backend is running.");
  }
  if (!res.ok || !res.body) throw new ApiError(res.status, `Request failed (${res.status})`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut: number;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const ev = parseFrame(buffer.slice(0, cut));
        buffer = buffer.slice(cut + 2);
        if (ev) onEvent(ev);
      }
    }
  } catch (e) {
    if (!signal?.aborted) throw e;
  }
}
