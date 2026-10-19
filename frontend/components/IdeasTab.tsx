"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createAiBuild, getIdeas } from "@/lib/api";
import type { BuildMeta } from "@/lib/build";
import type { Idea, IdeaPart } from "@/lib/types";
import BuildViewer from "./BuildViewerLazy";
import DesignedBuilds from "./DesignedBuilds";
import Thumb from "./Thumb";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "done"; ideas: Idea[] };

/** Three original builds the AI suggests using only the user's pieces (validated server-side). */
export default function IdeasTab({ parts, active, matchColour }: { parts: IdeaPart[]; active: boolean; matchColour: boolean }) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [slow, setSlow] = useState(false);
  const [viewing, setViewing] = useState<BuildMeta | null>(null); // the 3D build that is open
  const considerColour = matchColour;
  const started = useRef(false);
  const shown = useRef<string[]>([]); // every idea name already shown, so "Get new ideas" never repeats one

  const generate = useCallback(async () => {
    setState({ kind: "loading" });
    setSlow(false);
    const timer = setTimeout(() => setSlow(true), 12000);
    try {
      const r = await getIdeas(parts, considerColour, shown.current);
      shown.current = [...shown.current, ...r.ideas.map((i) => i.name)];
      setState({ kind: "done", ideas: r.ideas });
    } catch (e) {
      setState({ kind: "error", message: e instanceof Error ? e.message : "Something went wrong." });
    } finally {
      clearTimeout(timer);
    }
  }, [parts, considerColour]);

  // Start once, the first time the tab is opened (each call costs money, so not on page load).
  useEffect(() => {
    if (active && !started.current) {
      started.current = true;
      generate();
    }
  }, [active, generate]);

  // "Consider color" changed after the first run: the ideas are checked by a different rule, so ask again
  const lastMode = useRef(considerColour);
  useEffect(() => {
    if (lastMode.current !== considerColour) {
      lastMode.current = considerColour;
      if (started.current) generate();
    }
  }, [considerColour, generate]);

  if (!active) return null;

  return (
    <div>
      <DesignedBuilds parts={parts} active={active} matchColour={considerColour} />
      {state.kind !== "idle" && (
        <h2 className="title-display mb-1 text-2xl">
          More <span>ideas</span>
        </h2>
      )}
      {state.kind === "loading" && (
        <div role="status" className="flex flex-col items-center gap-3 py-16 text-center">
          <span className="h-8 w-8 animate-spin rounded-full border-4 border-zinc-300 border-t-ink" />
          <p className="font-medium">Thinking up builds for your pieces…</p>
          <p className="max-w-xs text-sm text-zinc-500">
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
          <p className="mb-3 text-xs text-zinc-500">
            Ideas are AI-generated and not test-built. Every piece listed is checked against your list.
          </p>
          <ul className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2 lg:gap-5">
            {state.ideas.map((idea) => (
              <IdeaCard key={idea.name} idea={idea} parts={parts} onOpen={setViewing} matchColour={considerColour} />
            ))}
          </ul>
          <button onClick={generate} className="mt-4 h-11 w-full rounded-2xl btn-glass text-sm font-medium active:bg-zinc-100">
            Get new ideas
          </button>
        </>
      )}
      {viewing && <BuildViewer build={viewing} onClose={() => setViewing(null)} />}
    </div>
  );
}

function IdeaCard({ idea, parts, onOpen, matchColour }: { idea: Idea; parts: IdeaPart[]; onOpen: (b: BuildMeta) => void; matchColour: boolean }) {
  const total = idea.pieces_used.reduce((n, p) => n + p.qty, 0);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  /** The AI places this idea's pieces on a stud grid; the server checks and repairs it, then we show it step by step. */
  const open3d = async () => {
    setBusy(true);
    setProblem(null);
    try {
      onOpen(await createAiBuild(parts.map((p) => ({ part_num: p.part_num, name: p.name, color: p.color, qty: p.qty })), idea, matchColour));
    } catch (e) {
      setProblem(e instanceof Error ? e.message : "Could not make the 3D build.");
    } finally {
      setBusy(false);
    }
  };
  return (
    <li className="glass-sm p-4">
      <h3 className="text-lg font-bold leading-snug">
        {idea.name}
        {idea.mixed_colours && <span className="ml-2 rounded-full bg-zinc-100 px-2 py-0.5 align-middle text-[11px] font-semibold uppercase tracking-wide text-zinc-600">Mixed colors</span>}
      </h3>
      <p className="mt-1 text-sm text-zinc-700">{idea.description}</p>

      <h4 className="mt-4 text-xs font-semibold uppercase tracking-wide text-zinc-500">
        Pieces used ({total})
      </h4>
      <ul className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {idea.pieces_used.map((p) => (
          <li key={`${p.part_num}|${p.colour ?? ""}`} className="flex items-center gap-2 rounded-xl bg-zinc-50 p-2">
            <Thumb srcs={[p.image]} label={p.name} className="h-10 w-10" />
            <span className="min-w-0 text-xs">
              <span className="block font-semibold">
                {p.qty} × {p.name}
                {p.colour ? ` (${p.colour})` : ""}
              </span>
              <span className="block text-zinc-500">{p.part_num}</span>
            </span>
          </li>
        ))}
      </ul>

      <button
        onClick={open3d}
        disabled={busy}
        className="mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-2xl btn-primary text-base font-semibold text-white disabled:opacity-60 active:opacity-80"
      >
        {busy ? (
          <>
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" /> Designing it in 3D… (up to 40 s)
          </>
        ) : (
          "See it built in 3D, step by step"
        )}
      </button>
      {problem && (
        <p role="alert" className="mt-2 rounded-xl bg-red-50 p-3 text-sm text-red-800">
          {problem}
        </p>
      )}

      <h4 className="mt-4 text-xs font-semibold uppercase tracking-wide text-zinc-500">Steps</h4>
      <ol className="mt-2 list-decimal space-y-2 pl-5 text-sm">
        {idea.steps.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ol>
    </li>
  );
}
