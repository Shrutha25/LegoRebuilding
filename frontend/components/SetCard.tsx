"use client";

import { useState } from "react";
import type { SetMatch } from "@/lib/types";

export default function SetCard({ set }: { set: SetMatch }) {
  const [open, setOpen] = useState(false);
  const pct = Math.round(set.coverage * 100);

  return (
    <li className="overflow-hidden rounded-2xl border border-stone-200 bg-white">
      <div className="flex gap-3 p-3">
        <div className="flex h-24 w-24 shrink-0 items-center justify-center overflow-hidden rounded-xl bg-stone-100">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          {set.image && <img src={set.image} alt="" className="h-full w-full object-contain" loading="lazy" />}
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="line-clamp-2 font-semibold leading-snug">{set.name}</h3>
          <p className="text-xs text-stone-500">
            {set.set_num} · {set.year}
          </p>
          <div
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`${pct}% of the set`}
            className="mt-2 h-3 overflow-hidden rounded-full bg-stone-200"
          >
            <div className={`h-full rounded-full ${pct >= 70 ? "bg-green-600" : pct >= 40 ? "bg-amber-500" : "bg-brick"}`} style={{ width: `${pct}%` }} />
          </div>
          <p className="mt-1 text-sm">
            You have <strong>{set.parts_you_have}</strong> / {set.total_parts} pieces <span className="text-stone-500">({pct}%)</span>
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-stone-100 px-3 py-2">
        {set.missing.length > 0 ? (
          <button
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="h-10 rounded-full border border-stone-300 px-4 text-sm font-medium active:bg-stone-100"
          >
            {open ? "Hide" : "Show"} missing parts ({set.missing_count})
          </button>
        ) : (
          <span className="text-sm font-medium text-green-700">You have every piece!</span>
        )}
        <div className="ml-auto flex gap-1">
          <a href={set.instructions_url} target="_blank" rel="noreferrer" className="h-10 content-center px-2 text-sm font-medium text-brick underline-offset-2 hover:underline">
            Instructions
          </a>
          <a href={set.rebrickable_url} target="_blank" rel="noreferrer" className="h-10 content-center px-2 text-sm font-medium text-brick underline-offset-2 hover:underline">
            Rebrickable
          </a>
        </div>
      </div>

      {open && (
        <ul className="grid grid-cols-2 gap-2 border-t border-stone-100 bg-stone-50 p-3 sm:grid-cols-3">
          {set.missing.map((m) => (
            <li key={`${m.part}-${m.color}`} className="flex items-center gap-2 rounded-xl bg-white p-2">
              <span className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-stone-100">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {m.image && <img src={m.image} alt="" className="h-full w-full object-contain" loading="lazy" />}
              </span>
              <span className="min-w-0 text-xs">
                <span className="block font-semibold">
                  {m.qty} × {m.part}
                </span>
                {m.color && <span className="block truncate capitalize text-stone-500">{m.color}</span>}
              </span>
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}
