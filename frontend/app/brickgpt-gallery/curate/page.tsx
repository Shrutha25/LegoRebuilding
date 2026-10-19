"use client";

import { useEffect, useMemo, useState } from "react";
import BuildViewer from "@/components/BuildViewerLazy";
import type { BuildMeta } from "@/lib/build";

/**
 * Dev page: curate the offline BrickGPT library. Every design with approve / reject; the decision is saved at once to
 * experiments/brickgpt/library-decisions.json (dev-only route). Only approved designs are meant to be used.
 * "Not one piece" marks designs whose bricks are not all linked through stud overlaps (BrickGPT's own "connected" only
 * checks that every brick reaches the ground). Nothing here ships: the routes return 404 outside `next dev`.
 */
type Design = {
  id: string; gid: string; thumb: string; tag: string; prompt: string; category: string; seed: number; bricks: number; seconds: number;
  rollbacks: number; connected: boolean; single_piece: boolean; linked_pieces: number; linked_sizes: number[];
};
type Decision = "approved" | "rejected";
type Filter = "all" | "undecided" | "approved" | "rejected" | "standard" | "small" | "not-one-piece";

const FILTERS: [Filter, string][] = [["all", "All"], ["undecided", "Undecided"], ["approved", "Approved"], ["rejected", "Rejected"], ["standard", "Standard"], ["small", "Small"], ["not-one-piece", "Not one piece"]];

export default function Curate() {
  const [designs, setDesigns] = useState<Design[] | null>(null);
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [filter, setFilter] = useState<Filter>("all");
  const [error, setError] = useState<string | null>(null);
  const [viewing, setViewing] = useState<BuildMeta | null>(null);

  useEffect(() => {
    Promise.all([fetch("/brickgpt-files/library-index.json"), fetch("/brickgpt-decisions")])
      .then(async ([a, b]) => {
        if (!a.ok) throw new Error("No library yet: run experiments/brickgpt/pack_library.py (and use `next dev`, the routes are dev-only).");
        setDesigns(await a.json());
        if (b.ok) setDecisions(await b.json());
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load the library."));
  }, []);

  const decide = async (id: string, next: Decision | null) => {
    const before = decisions;
    const after = { ...decisions };
    if (next) after[id] = next;
    else delete after[id];
    setDecisions(after); // optimistic, undone if the save fails
    try {
      const r = await fetch("/brickgpt-decisions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ id, decision: next }) });
      if (!r.ok) throw new Error(`Save failed (${r.status})`);
      setError(null);
    } catch (e) {
      setDecisions(before);
      setError(e instanceof Error ? e.message : "Save failed.");
    }
  };

  const open = async (gid: string) => {
    const meta: BuildMeta = await (await fetch(`/brickgpt-files/${gid}.meta.json`)).json();
    setViewing({ ...meta, model_url: `${location.origin}${meta.model_url}`, lite_model_url: `${location.origin}${meta.lite_model_url}` });
  };

  const counts = useMemo(() => {
    const all = designs ?? [];
    return {
      all: all.length,
      undecided: all.filter((d) => !decisions[d.id]).length,
      approved: all.filter((d) => decisions[d.id] === "approved").length,
      rejected: all.filter((d) => decisions[d.id] === "rejected").length,
      standard: all.filter((d) => d.tag === "standard").length,
      small: all.filter((d) => d.tag === "small").length,
      "not-one-piece": all.filter((d) => !d.single_piece).length,
    } satisfies Record<Filter, number>;
  }, [designs, decisions]);

  const shown = useMemo(
    () =>
      (designs ?? []).filter((d) =>
        filter === "all" ? true
        : filter === "undecided" ? !decisions[d.id]
        : filter === "not-one-piece" ? !d.single_piece
        : filter === "standard" || filter === "small" ? d.tag === filter
        : decisions[d.id] === filter,
      ),
    [designs, decisions, filter],
  );

  return (
    <main className="mx-auto w-full max-w-[100rem] px-4 py-6">
      <h1 className="title-display text-3xl">
        BrickGPT <span>library</span>
      </h1>
      <p className="mt-1 text-sm text-muted">
        Approve or reject each design; it is saved straight away. Only approved designs are used. Colours are not tuned yet (all lime).
      </p>
      <div className="mt-4 flex flex-wrap gap-2" role="tablist" aria-label="Filter designs">
        {FILTERS.map(([key, label]) => (
          <button
            key={key}
            role="tab"
            aria-selected={filter === key}
            onClick={() => setFilter(key)}
            className={`h-9 rounded-full px-4 text-xs font-bold ${filter === key ? "bg-black text-white" : "btn-glass"}`}
          >
            {label} ({counts[key]})
          </button>
        ))}
      </div>
      {error && (
        <p role="alert" className="mt-4 rounded-2xl bg-red-50 p-4 text-sm text-red-800">
          {error}
        </p>
      )}
      {!designs && !error && <p className="mt-6 text-sm">Loading…</p>}
      <ul className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        {shown.map((d) => {
          const dec = decisions[d.id];
          return (
            <li key={d.id} className={`glass-sm flex flex-col p-2 ${dec === "approved" ? "ring-2 ring-green-500" : dec === "rejected" ? "opacity-60 ring-2 ring-red-400" : ""}`}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={d.thumb} alt={`${d.category}, seed ${d.seed}`} className="aspect-[13/11] w-full rounded-xl bg-white object-contain" />
              <p className="mt-2 text-sm font-bold">
                {d.category} <span className="font-normal text-zinc-500">· seed {d.seed}</span>
                {d.tag === "small" && <span className="ml-2 rounded-full bg-black/10 px-2 py-0.5 text-[10px] font-bold uppercase">small</span>}
              </p>
              <p className="line-clamp-2 text-[11px] text-zinc-500">{d.prompt}</p>
              <p className="mt-1 text-xs text-muted">
                {d.bricks} bricks · {d.seconds}s{d.rollbacks ? ` · ${d.rollbacks} rollbacks` : ""}
              </p>
              <p className="mt-1 text-xs font-bold">
                {d.single_piece ? (
                  <span className="text-green-700">One piece</span>
                ) : (
                  <span className="text-red-700">
                    Not one piece: {d.linked_pieces} parts ({d.linked_sizes.join(" + ")} bricks)
                  </span>
                )}
                {!d.connected && <span className="ml-2 text-red-700">· not grounded</span>}
              </p>
              <div className="mt-2 flex gap-2">
                <button
                  onClick={() => decide(d.id, dec === "approved" ? null : "approved")}
                  aria-pressed={dec === "approved"}
                  className={`h-8 flex-1 rounded-full text-xs font-bold ${dec === "approved" ? "bg-green-600 text-white" : "btn-glass"}`}
                >
                  Approve
                </button>
                <button
                  onClick={() => decide(d.id, dec === "rejected" ? null : "rejected")}
                  aria-pressed={dec === "rejected"}
                  className={`h-8 flex-1 rounded-full text-xs font-bold ${dec === "rejected" ? "bg-red-600 text-white" : "btn-glass"}`}
                >
                  Reject
                </button>
                <button onClick={() => open(d.gid)} className="h-8 rounded-full btn-glass px-3 text-xs font-bold">
                  3D
                </button>
              </div>
            </li>
          );
        })}
      </ul>
      {designs && shown.length === 0 && <p className="mt-6 text-sm text-muted">Nothing in this filter.</p>}
      {viewing && <BuildViewer key={viewing.id} build={viewing} ldconfigUrl="/demo/LDConfig.ldr" onClose={() => setViewing(null)} />}
    </main>
  );
}
