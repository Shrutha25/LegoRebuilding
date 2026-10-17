"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, matchSets } from "@/lib/api";
import { toIdeaParts, toMatchParts } from "@/lib/review";
import type { Item, Resolved, SetMatch } from "@/lib/types";
import IdeasTab from "./IdeasTab";
import NearbySets from "./NearbySets";
import SubBuilds from "./SubBuilds";
import SetCard from "./SetCard";
import { useConsiderColour } from "@/lib/settings";
import { useRankedSets } from "@/lib/useInstructions";

type Outcome =
  | { kind: "catalogue" } // backend still loading the LEGO catalogue
  | { kind: "error"; message: string }
  | { kind: "done"; sets: SetMatch[] };
type State = Outcome | { kind: "loading" };

const RETRY_MS = 2000;
const NO_SETS = "No official set can be built 100% with these pieces – here's what you can build instead";

const NO_SET_MATCHES: SetMatch[] = [];

export default function ResultsScreen({
  items,
  resolved,
  onBack,
}: {
  items: Item[];
  resolved: Record<string, Resolved>;
  onBack: () => void;
}) {
  const [tab, setTab] = useState<"sets" | "ideas">("sets");
  const [matchColor, setMatchColor] = useConsiderColour(); // one setting for official sets, designed builds and ideas
  const [result, setResult] = useState<{ key: string } & Outcome>({ key: "", kind: "catalogue" });
  const [attempt, setAttempt] = useState(0);
  const autoSwitched = useRef(false);

  const parts = useMemo(() => toMatchParts(items, resolved), [items, resolved]);
  const userPieces = parts.reduce((n, p) => n + p.qty, 0);
  const ideaParts = useMemo(() => toIdeaParts(items), [items]);
  const partsKey = JSON.stringify(parts);
  const requestKey = `${partsKey}|${matchColor}|${attempt}`;
  // Anything not answering the current request (parts/colour toggled, retry) shows as loading.
  const state: State = result.key === requestKey ? result : { kind: "loading" };
  const { ranked, infoOf } = useRankedSets(state.kind === "done" ? state.sets : NO_SET_MATCHES);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const run = async () => {
      try {
        const r = await matchSets(JSON.parse(partsKey), matchColor);
        if (cancelled) return;
        setResult({ key: requestKey, kind: "done", sets: r.sets });
        // Nothing can be built completely: take the user straight to the AI ideas (once).
        if (r.sets.length === 0 && !autoSwitched.current) {
          autoSwitched.current = true;
          setTab("ideas");
        }
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 503) {
          setResult({ key: requestKey, kind: "catalogue" });
          timer = setTimeout(run, RETRY_MS);
        } else {
          setResult({ key: requestKey, kind: "error", message: e instanceof Error ? e.message : "Something went wrong." });
        }
      }
    };
    run();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [partsKey, matchColor, requestKey]);

  const noSets = state.kind === "done" && state.sets.length === 0;

  const tabClass = (active: boolean) =>
    `h-11 flex-1 rounded-full text-[11px] font-bold uppercase tracking-wide transition ${active ? "bg-ink text-white shadow-[0_8px_18px_-8px_rgba(0,0,0,0.6)]" : "text-zinc-500"}`;

  return (
    <div>
      <button onClick={onBack} className="mb-3 h-10 text-xs font-bold uppercase tracking-widest text-zinc-500">
        ← Back to pieces
      </button>
      <h1 className="title-display text-3xl">
        What you can <span>build</span>
      </h1>

      <div role="tablist" className="mt-4 flex gap-1 rounded-full border border-white bg-zinc-200/60 p-1 shadow-[inset_0_1px_3px_rgba(0,0,0,0.08)]">
        <button role="tab" aria-selected={tab === "sets"} onClick={() => setTab("sets")} className={tabClass(tab === "sets")}>
          Official sets
        </button>
        <button role="tab" aria-selected={tab === "ideas"} onClick={() => setTab("ideas")} className={tabClass(tab === "ideas")}>
          Build it now (AI ideas)
        </button>
      </div>

      <label className="mt-4 flex min-h-11 items-center gap-3 text-sm">
        <input type="checkbox" checked={matchColor} onChange={(e) => setMatchColor(e.target.checked)} className="h-5 w-5 accent-black" />
        <span>
          Consider color
          <span className="block text-xs text-muted">
            {matchColor ? "Pieces are matched by shape and colour." : "Pieces are matched by shape only; builds get colours from what you have."}
          </span>
        </span>
      </label>

      <div className="mt-4">
        {/* always mounted so switching tabs keeps the ideas; it starts only when first opened */}
        {tab === "ideas" && noSets && (
          <p className="mb-4 rounded-2xl bg-amber-50 p-3 text-sm text-amber-900">{NO_SETS}</p>
        )}
        <IdeasTab parts={ideaParts} active={tab === "ideas"} matchColour={matchColor} />

        {tab === "sets" && (
          <>
            {(state.kind === "loading" || state.kind === "catalogue") && (
              <div role="status" className="flex flex-col items-center gap-3 py-16 text-center">
                <span className="h-8 w-8 animate-spin rounded-full border-4 border-zinc-300 border-t-ink" />
                <p className="font-medium">{state.kind === "catalogue" ? "Loading LEGO catalogue…" : "Finding sets…"}</p>
                {state.kind === "catalogue" && (
                  <p className="max-w-xs text-sm text-zinc-500">The first start takes a little while. We&apos;ll keep trying.</p>
                )}
              </div>
            )}

            {state.kind === "error" && (
              <div role="alert" className="rounded-2xl bg-red-50 p-4 text-sm text-red-800">
                <p>{state.message}</p>
                <button onClick={() => setAttempt((a) => a + 1)} className="mt-3 h-10 rounded-full bg-red-800 px-4 font-medium text-white">
                  Try again
                </button>
              </div>
            )}

            {noSets && (
              <div className="glass-sm p-5">
                <p className="font-semibold">{NO_SETS}</p>
                <button onClick={() => setTab("ideas")} className="mt-4 h-11 rounded-full btn-primary px-5 text-sm font-semibold text-white active:opacity-80">
                  See AI build ideas →
                </button>
              </div>
            )}

            {state.kind === "done" && state.sets.length > 0 && (
              <ul className="grid grid-cols-1 items-start gap-3 lg:grid-cols-2 lg:gap-5">
                {ranked.map((s) => (
                  <SetCard key={s.set_num} set={s} userPieces={userPieces} parts={parts} info={infoOf(s)} />
                ))}
              </ul>
            )}

            {state.kind === "done" && parts.length > 0 && <SubBuilds parts={ideaParts.map((p) => ({ part_num: p.part_num, name: p.name, color: p.color, qty: p.qty }))} matchColour={matchColor} />}
            {state.kind === "done" && parts.length > 0 && <NearbySets parts={parts} matchColour={matchColor} />}
          </>
        )}
      </div>
    </div>
  );
}
