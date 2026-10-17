"use client";

import { useEffect, useMemo, useState } from "react";
import { getSetInstructions } from "./api";
import { infoFor, sortByInstructions, type InstructionInfo } from "./instructions";

const POLL_MS = 2500;
const GIVE_UP_MS = 120_000;

type Item = { set_num: string; total_parts: number; missing_count: number; instructions?: InstructionInfo };

/** Sets sorted by instructions state. Sets whose state is still unknown are polled until the server has looked them up. */
export function useRankedSets<T extends Item>(sets: T[]) {
  const [live, setLive] = useState<Record<string, InstructionInfo>>({});
  const pending = sets.filter((s) => infoFor(s.instructions, live, s.set_num)?.pending !== false).map((s) => s.set_num);
  const pendingKey = pending.join(",");

  useEffect(() => {
    if (!pendingKey) return;
    let cancelled = false;
    const started = Date.now();
    const tick = async () => {
      try {
        const r = await getSetInstructions(pendingKey.split(","));
        if (!cancelled) setLive((old) => ({ ...old, ...r }));
      } catch {
        /* a hint only: the cards simply stay without a button */
      }
      if (!cancelled && Date.now() - started < GIVE_UP_MS) timer = setTimeout(tick, POLL_MS);
    };
    let timer = setTimeout(tick, 800);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [pendingKey]);

  const infoOf = (s: T) => infoFor(s.instructions, live, s.set_num);
  const ranked = useMemo(() => sortByInstructions(sets, (s) => infoFor(s.instructions, live, s.set_num)), [sets, live]);
  return { ranked, infoOf };
}
