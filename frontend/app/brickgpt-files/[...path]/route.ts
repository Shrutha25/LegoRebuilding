import { readFile } from "node:fs/promises";
import path from "node:path";

/**
 * Dev-only file server for the BrickGPT experiment: serves experiments/brickgpt/gallery/ (gitignored, nothing in public/).
 * Returns 404 outside `next dev`, and only plain file names are accepted.
 */
const DIR = path.resolve(process.cwd(), "..", "experiments", "brickgpt", "gallery");

export async function GET(_req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  if (process.env.NODE_ENV !== "development") return new Response("Not found", { status: 404 });
  const { path: parts } = await ctx.params;
  if (parts.length !== 1 || !/^[\w.-]+$/.test(parts[0]) || parts[0].includes("..")) return new Response("Not found", { status: 404 });
  try {
    const body = await readFile(path.join(DIR, parts[0]));
    const type = parts[0].endsWith(".json") ? "application/json" : parts[0].endsWith(".png") ? "image/png" : "text/plain; charset=utf-8";
    return new Response(body, { headers: { "content-type": type, "cache-control": "no-store" } });
  } catch {
    return new Response("Not found", { status: 404 });
  }
}
