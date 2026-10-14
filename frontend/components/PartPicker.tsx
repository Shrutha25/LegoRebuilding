"use client";

import { useEffect, useState } from "react";
import { ApiError, searchParts } from "@/lib/api";
import type { PartChoice } from "@/lib/review";
import type { SearchResult } from "@/lib/types";

/** Search the Rebrickable parts catalogue by number or name. */
export default function PartPicker({
  onPick,
  onClose,
}: {
  onPick: (choice: PartChoice) => void;
  onClose: () => void;
}) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const term = q.trim();
  const active = term.length >= 2;

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const t = setTimeout(async () => {
      setBusy(true);
      try {
        const r = await searchParts(term);
        if (!cancelled) {
          setResults(r);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof ApiError && e.status === 503 ? "Catalogue is still loading, try again in a moment." : "Search failed. Try again.");
      } finally {
        if (!cancelled) setBusy(false);
      }
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [term, active]);

  return (
    <div className="rounded-xl border border-stone-300 bg-white p-3 shadow-sm">
      <div className="flex gap-2">
        <input
          autoFocus
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Part number or name, e.g. 3001 or slope"
          inputMode="search"
          className="h-11 min-w-0 flex-1 rounded-lg border border-stone-300 px-3 text-base outline-none focus:border-stone-500"
        />
        <button onClick={onClose} className="h-11 rounded-lg px-3 text-sm font-medium text-stone-600 active:bg-stone-100">
          Cancel
        </button>
      </div>
      {active && error && <p className="mt-2 text-sm text-red-700">{error}</p>}
      {active && busy && <p className="mt-2 text-sm text-stone-500">Searching…</p>}
      {active && !busy && !error && results.length === 0 && (
        <p className="mt-2 text-sm text-stone-500">No parts found.</p>
      )}
      <ul className="mt-2 flex max-h-72 flex-col overflow-y-auto">
        {(active ? results : []).map((r) => (
          <li key={r.part_num}>
            <button
              onClick={() => onPick({ part_id: r.part_num, name: r.name, image: r.image })}
              className="flex w-full items-center gap-3 rounded-lg p-2 text-left active:bg-stone-100"
            >
              <span className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-md bg-stone-100">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {r.image && <img src={r.image} alt="" className="h-full w-full object-contain" loading="lazy" />}
              </span>
              <span className="min-w-0">
                <span className="line-clamp-2 text-sm font-medium">{r.name}</span>
                <span className="text-xs text-stone-500">{r.part_num}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
