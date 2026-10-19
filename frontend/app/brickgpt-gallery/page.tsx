"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import BuildViewer from "@/components/BuildViewerLazy";
import type { BuildMeta } from "@/lib/build";

/**
 * Dev page for the BrickGPT experiment (nothing here ships: the files route returns 404 outside `next dev`).
 * Every generated build with its run numbers; "3D" opens it in the real step viewer.  ?open=<id>&done=1 opens one directly.
 * Stability is BrickGPT's connectivity check (no Gurobi): "connected", not physically verified.
 */
type Run = {
  id: string;
  prompt: string;
  mode: string;
  inventory: string | null;
  inventory_bricks: number | null;
  bricks: number;
  steps: number;
  end_reason: string;
  connected: boolean;
  seconds: number;
  rollbacks: number;
};

const END: Record<string, string> = { finished: "finished", ran_out: "ran out of bricks", no_valid_brick: "no valid brick left" };

function Gallery() {
  const q = useSearchParams();
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [viewing, setViewing] = useState<BuildMeta | null>(null);
  const [error, setError] = useState<string | null>(null);

  const open = async (id: string) => {
    const meta: BuildMeta = await (await fetch(`/brickgpt-files/${id}.meta.json`)).json();
    // the viewer wants absolute model URLs
    setViewing({ ...meta, model_url: `${location.origin}${meta.model_url}`, lite_model_url: `${location.origin}${meta.lite_model_url}` });
  };

  useEffect(() => {
    fetch("/brickgpt-files/index.json")
      .then(async (r) => {
        if (!r.ok) throw new Error("No gallery yet: run experiments/brickgpt/pack_gallery.py (and use `next dev`, the route is dev-only).");
        setRuns(await r.json());
        const id = q.get("open");
        if (id) await open(id);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load the gallery."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="mx-auto w-full max-w-[90rem] px-4 py-6">
      <h1 className="title-display text-3xl">
        BrickGPT <span>gallery</span>
      </h1>
      <p className="mt-1 text-sm text-muted">Experiment, seed 42. “Connected” is a connectivity check, not a physical stability check.</p>
      {error && (
        <p role="alert" className="mt-4 rounded-2xl bg-red-50 p-4 text-sm text-red-800">
          {error}
        </p>
      )}
      {!runs && !error && <p className="mt-6 text-sm">Loading…</p>}
      {runs && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="eyebrow">
                <th className="py-2 pr-3">Build</th>
                <th className="pr-3">Mode</th>
                <th className="pr-3">Inventory</th>
                <th className="pr-3">Bricks used</th>
                <th className="pr-3">Ended</th>
                <th className="pr-3">Connected</th>
                <th className="pr-3">Time</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} className="border-t border-black/10">
                  <td className="py-2 pr-3 font-bold">
                    {r.prompt}
                    <span className="block text-[11px] font-normal text-zinc-500">{r.id}</span>
                  </td>
                  <td className="pr-3">{r.mode}</td>
                  <td className="pr-3">{r.inventory ? `${r.inventory} (${r.inventory_bricks})` : "none"}</td>
                  <td className="pr-3">
                    {r.bricks}
                    {r.inventory_bricks ? ` / ${r.inventory_bricks}` : ""}
                  </td>
                  <td className="pr-3">{END[r.end_reason] ?? r.end_reason}</td>
                  <td className="pr-3">{r.connected ? "yes" : "no"}</td>
                  <td className="pr-3">{r.seconds}s</td>
                  <td>
                    <button onClick={() => open(r.id)} className="h-8 rounded-full btn-glass px-4 text-xs font-bold">
                      3D
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {viewing && (
        <BuildViewer
          key={viewing.id}
          build={viewing}
          ldconfigUrl="/demo/LDConfig.ldr"
          startFinished={q.get("done") === "1"}
          exposeScene={(s) => ((window as unknown as { __scene?: unknown }).__scene = s)}
          onClose={() => setViewing(null)}
        />
      )}
    </main>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<p className="p-6">Loading…</p>}>
      <Gallery />
    </Suspense>
  );
}
