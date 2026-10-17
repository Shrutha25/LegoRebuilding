"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { absUrl, getRecipeBuilds } from "@/lib/api";
import type { BuildMeta, RecipeBuilds as Result, UnlockHint } from "@/lib/build";
import type { IdeaPart } from "@/lib/types";
import BuildViewer from "./BuildViewerLazy";

type State = { kind: "idle" } | { kind: "loading" } | { kind: "error"; message: string } | { kind: "done"; result: Result };

/**
 * Builds the code designs from the user's pieces (recipes: car, house, tower, bridge, rocket ...). The shape always matches the
 * name, and every build opens in the 3D step-by-step viewer. Gemini only picks, names and colours them; without it the builds
 * still appear with standard names. Pieces that would unlock more recipes are shown as a gentle hint.
 */
export default function RecipeBuilds({ parts, active, matchColour = true }: { parts: IdeaPart[]; active: boolean; matchColour?: boolean }) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [viewing, setViewing] = useState<BuildMeta | null>(null);
  const started = useRef(false);

  const load = useCallback(async () => {
    setState({ kind: "loading" });
    try {
      setState({ kind: "done", result: await getRecipeBuilds(parts.map((p) => ({ part_num: p.part_num, name: p.name, color: p.color, qty: p.qty })), true, matchColour) });
    } catch (e) {
      setState({ kind: "error", message: e instanceof Error ? e.message : "Something went wrong." });
    }
  }, [parts, matchColour]);

  // First time the tab opens, and again whenever "Consider color" is switched.
  useEffect(() => {
    if (active) {
      started.current = true;
      load();
    }
  }, [active, load]);

  if (!active) return null;
  return (
    <section aria-label="Builds made from your pieces" className="mb-8">
      <h2 className="title-display text-2xl">
        Made from <span>your pieces</span>
      </h2>
      <p className="mt-1 text-sm text-muted">Designed to fit your bricks exactly, with steps you can follow in 3D.</p>

      {state.kind === "loading" && (
        <div role="status" className="flex items-center gap-3 py-8">
          <span className="h-6 w-6 animate-spin rounded-full border-4 border-zinc-300 border-t-ink" />
          <p className="text-sm font-medium">Designing builds for your pieces…</p>
        </div>
      )}

      {state.kind === "error" && (
        <div role="alert" className="mt-3 rounded-2xl bg-red-50 p-4 text-sm text-red-800">
          <p>{state.message}</p>
          <button onClick={load} className="mt-3 h-10 rounded-full bg-red-800 px-4 font-medium text-white active:opacity-80">
            Try again
          </button>
        </div>
      )}

      {state.kind === "done" && (
        <>
          {state.result.builds.length === 0 ? (
            <div className="glass-sm mt-3 p-4 text-sm">
              <p className="font-semibold">Not enough pieces for a designed build yet.</p>
              <p className="mt-1 text-zinc-600">Scan more bricks and plates (a few long ones help), then try again.</p>
            </div>
          ) : (
            <ul className="mt-4 grid grid-cols-1 items-start gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {state.result.builds.map((b) => (
                <BuildCard key={b.id} build={b} onOpen={setViewing} />
              ))}
            </ul>
          )}
          <Unlock hints={state.result.unlock} />
        </>
      )}
      {viewing && <BuildViewer build={viewing} onClose={() => setViewing(null)} />}
    </section>
  );
}

function BuildCard({ build, onOpen }: { build: BuildMeta; onOpen: (b: BuildMeta) => void }) {
  const colours = Array.from(new Set(Object.values(build.recipe?.colours ?? {})));
  return (
    <li className="glass-sm overflow-hidden p-3">
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
        {build.part_count} parts · {build.step_count} steps{colours.length > 0 && <> · {colours.join(", ")}</>}
      </p>
      <button onClick={() => onOpen(build)} className="mt-3 flex h-11 w-full items-center justify-center rounded-2xl btn-primary text-sm font-semibold text-white active:opacity-80">
        See it built in 3D
      </button>
    </li>
  );
}

function Unlock({ hints }: { hints: UnlockHint[] }) {
  if (!hints.length) return null;
  const text = (h: UnlockHint) => h.options.map((o) => o.map((p) => `${p.qty} more ${p.part}`).join(" + ")).join(", or ");
  return (
    <div className="mt-5 rounded-2xl border border-white bg-white/60 p-4 text-sm">
      <p className="eyebrow">Scan these to unlock</p>
      <p className="mt-1 font-semibold">{hints.map((h) => h.name).join(", ")}</p>
      <ul className="mt-2 space-y-1 text-zinc-600">
        {hints.map((h) => (
          <li key={h.recipe}>
            <span className="font-medium text-zinc-900">{h.name}:</span> {text(h)}
          </li>
        ))}
      </ul>
    </div>
  );
}
