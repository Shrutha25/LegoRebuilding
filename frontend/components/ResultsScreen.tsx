"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, matchSets } from "@/lib/api";
import { toIdeaParts, toMatchParts } from "@/lib/review";
import type { Item, Resolved, SetMatch } from "@/lib/types";
import IdeasTab from "./IdeasTab";
import SetCard from "./SetCard";

type Outcome =
  | { kind: "catalogue" } // backend still loading the LEGO catalogue
  | { kind: "error"; message: string }
  | { kind: "done"; sets: SetMatch[] };
type State = Outcome | { kind: "loading" };

const RETRY_MS = 2000;
const NO_SETS = "No official set can be built 100% with these pieces – here's what you can build instead";

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
  const [matchColor, setMatchColor] = useState(false);
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
    `h-11 flex-1 rounded-xl text-sm font-semibold ${active ? "bg-white shadow-sm" : "text-stone-600"}`;

  return (
    <div>
      <button onClick={onBack} className="mb-3 h-10 text-sm font-medium text-stone-600">
        ← Back to pieces
      </button>
      <h1 className="text-xl font-bold">What you can build</h1>

      <div role="tablist" className="mt-3 flex gap-1 rounded-2xl bg-stone-200 p-1">
        <button role="tab" aria-selected={tab === "sets"} onClick={() => setTab("sets")} className={tabClass(tab === "sets")}>
          Official sets
        </button>
        <button role="tab" aria-selected={tab === "ideas"} onClick={() => setTab("ideas")} className={tabClass(tab === "ideas")}>
          Build it now (AI ideas)
        </button>
      </div>

      <div className="mt-4">
        {/* always mounted so switching tabs keeps the ideas; it starts only when first opened */}
        {tab === "ideas" && noSets && (
          <p className="mb-4 rounded-2xl bg-amber-50 p-3 text-sm text-amber-900">{NO_SETS}</p>
        )}
        <IdeasTab parts={ideaParts} active={tab === "ideas"} />

        {tab === "sets" && (
          <>
            <label className="mb-4 flex min-h-11 items-center gap-3 text-sm">
              <input
                type="checkbox"
                checked={matchColor}
                onChange={(e) => setMatchColor(e.target.checked)}
                className="h-5 w-5 accent-[#d01012]"
              />
              Match colours exactly
            </label>

            {(state.kind === "loading" || state.kind === "catalogue") && (
              <div role="status" className="flex flex-col items-center gap-3 py-16 text-center">
                <span className="h-8 w-8 animate-spin rounded-full border-4 border-stone-300 border-t-brick" />
                <p className="font-medium">{state.kind === "catalogue" ? "Loading LEGO catalogue…" : "Finding sets…"}</p>
                {state.kind === "catalogue" && (
                  <p className="max-w-xs text-sm text-stone-500">The first start takes a little while. We&apos;ll keep trying.</p>
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
              <div className="rounded-2xl border border-stone-200 bg-white p-5">
                <p className="font-semibold">{NO_SETS}</p>
                <button onClick={() => setTab("ideas")} className="mt-4 h-11 rounded-full bg-brick px-5 text-sm font-semibold text-white active:opacity-80">
                  See AI build ideas →
                </button>
              </div>
            )}

            {state.kind === "done" && state.sets.length > 0 && (
              <ul className="flex flex-col gap-3">
                {state.sets.map((s) => (
                  <SetCard key={s.set_num} set={s} userPieces={userPieces} />
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </div>
  );
}
