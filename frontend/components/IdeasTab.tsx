"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getIdeas } from "@/lib/api";
import type { Idea, IdeaPart } from "@/lib/types";
import Thumb from "./Thumb";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "done"; ideas: Idea[] };

/** Three original builds the AI suggests using only the user's pieces (validated server-side). */
export default function IdeasTab({ parts, active }: { parts: IdeaPart[]; active: boolean }) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [slow, setSlow] = useState(false);
  const started = useRef(false);

  const generate = useCallback(async () => {
    setState({ kind: "loading" });
    setSlow(false);
    const timer = setTimeout(() => setSlow(true), 12000);
    try {
      const r = await getIdeas(parts);
      setState({ kind: "done", ideas: r.ideas });
    } catch (e) {
      setState({ kind: "error", message: e instanceof Error ? e.message : "Something went wrong." });
    } finally {
      clearTimeout(timer);
    }
  }, [parts]);

  // Start once, the first time the tab is opened (each call costs money, so not on page load).
  useEffect(() => {
    if (active && !started.current) {
      started.current = true;
      generate();
    }
  }, [active, generate]);

  if (!active) return null;

  return (
    <div>
      {state.kind === "loading" && (
        <div role="status" className="flex flex-col items-center gap-3 py-16 text-center">
          <span className="h-8 w-8 animate-spin rounded-full border-4 border-stone-300 border-t-brick" />
          <p className="font-medium">Thinking up builds for your pieces…</p>
          <p className="max-w-xs text-sm text-stone-500">
            {slow ? "Still working. This can take up to a minute." : "This usually takes 10 to 30 seconds."}
          </p>
        </div>
      )}

      {state.kind === "error" && (
        <div role="alert" className="rounded-2xl bg-red-50 p-4 text-sm text-red-800">
          <p className="font-semibold">Couldn&apos;t get build ideas</p>
          <p className="mt-1">{state.message}</p>
          <button onClick={generate} className="mt-3 h-10 rounded-full bg-red-800 px-4 font-medium text-white active:opacity-80">
            Try again
          </button>
        </div>
      )}

      {state.kind === "done" && (
        <>
          <p className="mb-3 text-xs text-stone-500">
            Ideas are AI-generated and not test-built. Every piece listed is checked against your list.
          </p>
          <ul className="flex flex-col gap-4">
            {state.ideas.map((idea) => (
              <IdeaCard key={idea.name} idea={idea} />
            ))}
          </ul>
          <button onClick={generate} className="mt-4 h-11 w-full rounded-2xl border border-stone-300 bg-white text-sm font-medium active:bg-stone-100">
            Get new ideas
          </button>
        </>
      )}
    </div>
  );
}

function IdeaCard({ idea }: { idea: Idea }) {
  const total = idea.pieces_used.reduce((n, p) => n + p.qty, 0);
  return (
    <li className="rounded-2xl border border-stone-200 bg-white p-4">
      <h3 className="text-lg font-bold leading-snug">{idea.name}</h3>
      <p className="mt-1 text-sm text-stone-700">{idea.description}</p>

      <h4 className="mt-4 text-xs font-semibold uppercase tracking-wide text-stone-500">
        Pieces used ({total})
      </h4>
      <ul className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {idea.pieces_used.map((p) => (
          <li key={p.part_num} className="flex items-center gap-2 rounded-xl bg-stone-50 p-2">
            <Thumb srcs={[p.image]} label={p.name} className="h-10 w-10" />
            <span className="min-w-0 text-xs">
              <span className="block font-semibold">
                {p.qty} × {p.name}
              </span>
              <span className="block text-stone-500">{p.part_num}</span>
            </span>
          </li>
        ))}
      </ul>

      <h4 className="mt-4 text-xs font-semibold uppercase tracking-wide text-stone-500">Steps</h4>
      <ol className="mt-2 list-decimal space-y-2 pl-5 text-sm">
        {idea.steps.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ol>
    </li>
  );
}
