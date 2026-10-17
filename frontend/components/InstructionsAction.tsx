"use client";

import { useState } from "react";
import { getSetBuild } from "@/lib/api";
import type { BuildMeta, SetBuild } from "@/lib/build";
import type { InstructionInfo } from "@/lib/instructions";
import type { MatchPart } from "@/lib/types";
import BuildViewer from "./BuildViewerLazy";
import Thumb from "./Thumb";

const linkBtn = "mt-3 flex h-11 w-full items-center justify-center rounded-xl btn-glass text-sm font-semibold active:bg-zinc-100";

/**
 * The instructions part of a set card, by state:
 *   1  an open 3D model exists          -> "View steps" opens our viewer
 *   2  LEGO has instructions for it     -> "Open LEGO instructions" (a short list when there are several booklets)
 *   3  neither                          -> no button: a note and a Rebrickable link
 *   unknown (lookup running)            -> empty, but the space is reserved so the card does not jump
 */
export default function InstructionsAction({
  setNum, name, rebrickableUrl, info, parts = [],
}: { setNum: string; name: string; rebrickableUrl: string; info?: InstructionInfo; parts?: MatchPart[] }) {
  const [state, setState] = useState<{ kind: "idle" } | { kind: "loading" } | { kind: "error"; message: string } | { kind: "done"; result: SetBuild }>({ kind: "idle" });
  const [viewing, setViewing] = useState<BuildMeta | null>(null);
  const result = state.kind === "done" ? state.result : null;

  const openSteps = async () => {
    setState({ kind: "loading" });
    try {
      const r = await getSetBuild(setNum, parts.map((p) => ({ part_num: p.part_num, color: p.color, qty: p.qty })), name);
      setState({ kind: "done", result: r });
      if (r.available && r.build) setViewing(r.build);
    } catch (e) {
      setState({ kind: "error", message: e instanceof Error ? e.message : "Something went wrong." });
    }
  };

  const slot = "mt-3 min-h-11";
  if (!info || info.state === null) return <div className={slot} aria-hidden="true" />;

  if (info.state === 3) {
    return (
      <div className={`${slot} flex flex-wrap items-center justify-between gap-2 text-sm text-zinc-600`}>
        <span>Instructions aren&apos;t available online for this set</span>
        <a href={rebrickableUrl} target="_blank" rel="noreferrer noopener" className="h-8 content-center text-[11px] font-bold uppercase tracking-wide text-zinc-900 underline-offset-4 hover:underline">
          View on Rebrickable
        </a>
      </div>
    );
  }

  if (info.state === 2) {
    const links = info.links;
    if (links.length === 1) {
      return (
        <a href={links[0].url} target="_blank" rel="noreferrer noopener" className={linkBtn}>
          Open LEGO instructions
        </a>
      );
    }
    return (
      <div className={slot}>
        <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Open LEGO instructions</p>
        <ul className="mt-1 space-y-1">
          {links.map((l, i) => (
            <li key={l.url}>
              <a href={l.url} target="_blank" rel="noreferrer noopener" className="block rounded-lg bg-zinc-50 px-3 py-2 text-sm font-medium underline-offset-4 hover:underline">
                {l.description || `Booklet ${i + 1}`}
              </a>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  return (
    <>
      <button
        onClick={result?.available && result.build ? () => setViewing(result.build) : openSteps}
        disabled={state.kind === "loading"}
        className="mt-3 flex h-11 w-full items-center justify-center gap-2 rounded-xl btn-glass text-sm font-semibold disabled:opacity-60 active:bg-zinc-100"
      >
        {state.kind === "loading" ? (
          <>
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-300 border-t-zinc-900" /> Loading steps…
          </>
        ) : (
          "View steps"
        )}
      </button>

      {state.kind === "error" && (
        <p role="alert" className="mt-2 rounded-xl bg-red-50 p-3 text-sm text-red-800">
          {state.message}
        </p>
      )}

      {result && !result.available && (
        <div className="mt-3 rounded-xl bg-zinc-50 p-3 text-sm">
          <p className="font-semibold">No 3D steps for this set</p>
          <p className="mt-1 text-zinc-600">
            {result.reason === "licence" ? "A model exists but its licence does not allow showing it here." : "The 3D model could not be loaded."}{" "}
            Here are its parts ({result.have_total} of {result.need_total} are in your pieces):
          </p>
          <ul className="mt-3 max-h-64 divide-y divide-zinc-200 overflow-y-auto rounded-lg bg-white">
            {result.inventory.map((row) => (
              <li key={`${row.part}-${row.colour}`} className="flex items-center gap-2 p-2">
                <Thumb srcs={[row.image]} label={row.name} className="h-10 w-10" />
                <span className="min-w-0 flex-1 text-xs">
                  <span className="line-clamp-1 block font-medium">{row.name}</span>
                  <span className="block text-zinc-500">
                    {row.part} · {row.colour}
                  </span>
                </span>
                <span className={`shrink-0 text-xs font-semibold ${row.missing ? "text-amber-700" : "text-green-700"}`}>
                  {row.have}/{row.need}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {viewing && <BuildViewer build={viewing} onClose={() => setViewing(null)} />}
    </>
  );
}
