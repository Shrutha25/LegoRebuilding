"use client";

import { useEffect, useState } from "react";
import { getNearbySets } from "@/lib/api";
import type { NearbySet } from "@/lib/build";
import type { MatchPart } from "@/lib/types";
import { useRankedSets } from "@/lib/useInstructions";
import InstructionsAction from "./InstructionsAction";
import Thumb from "./Thumb";

/** Sets that are almost buildable: at most a few parts missing, or 90% of the parts there. The missing parts are listed. */
export default function NearbySets({ parts, matchColour }: { parts: MatchPart[]; matchColour: boolean }) {
  const [sets, setSets] = useState<NearbySet[] | null>(null);
  const key = JSON.stringify(parts);

  useEffect(() => {
    let cancelled = false;
    getNearbySets(JSON.parse(key), matchColour)
      .then((r) => !cancelled && setSets(r.almost))
      .catch(() => !cancelled && setSets([])); // a hint only: never an error screen
    return () => {
      cancelled = true;
    };
  }, [key, matchColour]);

  const { ranked, infoOf } = useRankedSets(sets ?? []);
  if (!sets || sets.length === 0) return null;
  return (
    <section aria-label="Almost there" className="mt-8">
      <h2 className="title-display text-2xl">
        Almost <span>there</span>
      </h2>
      <p className="mt-1 text-sm text-muted">Sets you could build with a few more pieces.</p>
      <ul className="mt-4 grid grid-cols-1 items-start gap-3 lg:grid-cols-2 lg:gap-5">
        {ranked.map((s) => (
          <li key={s.set_num} className="glass-sm p-3">
            <div className="flex gap-3">
              <Thumb srcs={[s.image]} label={s.name} className="h-20 w-20" />
              <div className="min-w-0 flex-1">
                <h3 className="line-clamp-2 font-semibold leading-snug">{s.name}</h3>
                <p className="text-xs text-zinc-500">
                  {s.set_num} · {s.year}
                </p>
                <p className="mt-1 text-sm">
                  Missing <strong>{s.missing_count}</strong> of {s.total_parts} parts
                </p>
              </div>
            </div>
            <ul className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {s.missing.slice(0, 6).map((m) => (
                <li key={`${m.part}-${m.colour}`} className="flex items-center gap-2 rounded-xl bg-zinc-50 p-2">
                  <Thumb srcs={[m.image]} label={m.name} className="h-10 w-10" />
                  <span className="min-w-0 text-xs">
                    <span className="line-clamp-1 block font-semibold">
                      {m.qty} × {m.name}
                    </span>
                    <span className="block text-zinc-500">
                      {m.part}
                      {m.colour ? ` · ${m.colour}` : ""}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
            <InstructionsAction setNum={s.set_num} name={s.name} rebrickableUrl={s.rebrickable_url} info={infoOf(s)} parts={parts} />
          </li>
        ))}
      </ul>
    </section>
  );
}
