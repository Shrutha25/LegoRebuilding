"use client";

import { useEffect, useRef, useState } from "react";
import { absUrl } from "@/lib/api";
import type { BuildMeta, BuildPartIn } from "@/lib/build";
import { streamDesigns, type DesignEvent } from "@/lib/designs";
import type { IdeaPart } from "@/lib/types";
import BuildViewer from "./BuildViewerLazy";

type Design = BuildMeta & { design?: { concept: string; coverage: number; use_share: number; low_use: boolean } };
type State = { designs: Design[]; status: "idle" | "designing" | "done" | "error"; note: string | null; message: string | null; start: { pieces: number } | null };
const SKELETONS = 3;

/**
 * "Designed for your pile": the AI imagines shapes for the pieces, the server fits the exact bricks, and every design that passes the checks appears here
 * the moment it is ready (a stream), with placeholders while the rest are made. A design the final look check does not recognise disappears again.
 * Each opens in the 3D step viewer. It starts when the tab opens and starts again when the pieces or "Consider color" change.
 */
export default function DesignedBuilds({ parts, active, matchColour }: { parts: IdeaPart[]; active: boolean; matchColour: boolean }) {
  const [state, setState] = useState<State>({ designs: [], status: "idle", note: null, message: null, start: null });
  const [viewing, setViewing] = useState<BuildMeta | null>(null);
  const key = JSON.stringify(parts.map((p) => [p.part_num, p.color, p.qty]));

  useEffect(() => {
    if (!active) return;
    const ctl = new AbortController();
    const body: BuildPartIn[] = parts.map((p) => ({ part_num: p.part_num, name: p.name, color: p.color, qty: p.qty }));
    setState({ designs: [], status: "designing", note: null, message: null, start: null });
    const on = (e: DesignEvent) => {
      setState((s) => {
        if (e.event === "start") return { ...s, start: { pieces: e.data.pieces } };
        if (e.event === "design") return { ...s, designs: [...s.designs, e.data.build] };
        if (e.event === "verdict") return e.data.ok ? s : { ...s, designs: s.designs.filter((d) => d.id !== e.data.id) };
        if (e.event === "note") return { ...s, note: e.data.message };
        if (e.event === "done") return { ...s, status: "done" };
        return s;
      });
    };
    streamDesigns(body, matchColour, on, ctl.signal)
      .then(() => setState((s) => (s.status === "designing" ? { ...s, status: "done" } : s)))
      .catch((e) => setState((s) => ({ ...s, status: "error", message: e instanceof Error ? e.message : "Something went wrong." })));
    return () => ctl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, key, matchColour]);

  if (!active) return null;
  const { designs, status } = state;
  return (
    <section aria-label="Designed for your pile" className="mb-8">
      <h2 className="title-display text-2xl">
        Designed for <span>your pile</span>
      </h2>
      <p className="mt-1 text-sm text-muted">Small models imagined for your pieces and built from exactly what you have, with steps you can follow in 3D.</p>

      {status === "error" && (
        <p role="alert" className="mt-3 rounded-2xl bg-red-50 p-4 text-sm text-red-800">
          {state.message}
        </p>
      )}
      {state.note && designs.length === 0 && status === "done" && <p className="glass-sm mt-3 p-4 text-sm">{state.note}</p>}

      {(designs.length > 0 || status === "designing") && (
        <ul className="mt-4 grid grid-cols-1 items-start gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {designs.map((d) => (
            <DesignCard key={d.id} build={d} onOpen={setViewing} />
          ))}
          {status === "designing" &&
            Array.from({ length: Math.max(0, SKELETONS - designs.length) }).map((_, i) => (
              <li key={`sk${i}`} aria-hidden="true" className="glass-sm animate-pulse p-3">
                <div className="aspect-[21/17] w-full rounded-2xl bg-zinc-200/70" />
                <div className="mt-3 h-5 w-2/3 rounded bg-zinc-200/70" />
                <div className="mt-2 h-4 w-full rounded bg-zinc-200/60" />
                <div className="mt-4 h-11 w-full rounded-2xl bg-zinc-200/60" />
              </li>
            ))}
        </ul>
      )}
      {status === "designing" && (
        <p role="status" className="mt-3 text-xs text-muted">
          Designing{state.start ? ` for your ${state.start.pieces} pieces` : ""}…
        </p>
      )}
      {viewing && <BuildViewer build={viewing} onClose={() => setViewing(null)} />}
    </section>
  );
}

function DesignCard({ build, onOpen }: { build: Design; onOpen: (b: BuildMeta) => void }) {
  const ref = useRef<HTMLLIElement>(null);
  return (
    <li ref={ref} className="glass-sm overflow-hidden p-3">
      {build.thumb_url && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={absUrl(build.thumb_url)} alt={`${build.title}, drawn from the front corner`} className="aspect-[21/17] w-full rounded-2xl bg-white object-contain" loading="lazy" />
      )}
      <h3 className="mt-3 text-lg font-bold leading-snug">
        {build.title}
        {build.mixed_colours && <span className="ml-2 rounded-full bg-zinc-100 px-2 py-0.5 align-middle text-[11px] font-semibold uppercase tracking-wide text-zinc-600">Mixed colors</span>}
      </h3>
      <p className="mt-1 text-sm text-zinc-700">{build.description}</p>
      <p className="mt-2 text-xs text-muted">
        {build.part_count} parts · {build.step_count} steps
        {build.design ? ` · uses ${Math.round(build.design.use_share * 100)}% of your pile` : ""}
      </p>
      <button onClick={() => onOpen(build)} className="mt-3 flex h-11 w-full items-center justify-center rounded-2xl btn-primary text-sm font-semibold text-white active:opacity-80">
        See it built in 3D
      </button>
    </li>
  );
}
