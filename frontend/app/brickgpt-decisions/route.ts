import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

/**
 * Dev-only: the curation decisions of the BrickGPT library, kept in experiments/brickgpt/library-decisions.json.
 * GET -> {designId: "approved" | "rejected"};  POST {id, decision: "approved" | "rejected" | null} sets or clears one.
 * 404 outside `next dev`. Only designs approved here are meant to be used.
 */
const FILE = path.resolve(process.cwd(), "..", "experiments", "brickgpt", "library-decisions.json");

async function load(): Promise<Record<string, string>> {
  try {
    return JSON.parse(await readFile(FILE, "utf-8"));
  } catch {
    return {};
  }
}

export async function GET() {
  if (process.env.NODE_ENV !== "development") return new Response("Not found", { status: 404 });
  return Response.json(await load(), { headers: { "cache-control": "no-store" } });
}

export async function POST(req: Request) {
  if (process.env.NODE_ENV !== "development") return new Response("Not found", { status: 404 });
  if (!req.headers.get("content-type")?.includes("application/json")) return new Response("Expected JSON", { status: 415 });
  const body = (await req.json().catch(() => null)) as { id?: unknown; decision?: unknown } | null;
  const id = body?.id;
  const decision = body?.decision;
  if (typeof id !== "string" || !/^[\w-]{1,80}$/.test(id) || !(decision === null || decision === "approved" || decision === "rejected")) {
    return new Response("Bad request", { status: 400 });
  }
  const all = await load();
  if (decision === null) delete all[id];
  else all[id] = decision;
  await writeFile(FILE, JSON.stringify(all, null, 1) + "\n", "utf-8");
  return Response.json(all, { headers: { "cache-control": "no-store" } });
}
