"use client";

import { useEffect, useState } from "react";
import { getSubBuildModel, getSubBuilds } from "@/lib/api";
import type { BuildMeta, SubBuild, SubBuilds as SubBuildsResult } from "@/lib/build";
import type { BuildPartIn } from "@/lib/build";
import BuildViewer from "./BuildViewerLazy";
import Thumb from "./Thumb";

/**
 * Real LEGO builds from the LDraw Official Model Repository: whole small models and the sub-models of bigger ones (a vehicle, a tree, a gate)
 * that the user's pieces can build exactly, then, in a separate section, builds that are only 1 or 2 parts short (the missing parts are listed).
 * Each card names the author and licence of the model and opens it in the 3D step viewer.
 */
export default function SubBuilds({ parts, matchColour }: { parts: BuildPartIn[]; matchColour: boolean }) {
  const [data, setData] = useState<SubBuildsResult | null>(null);
  const [viewing, setViewing] = useState<BuildMeta | null>(null);
  const key = JSON.stringify(parts);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let tries = 0;
    const load = () =>
      getSubBuilds(JSON.parse(key), matchColour)
        .then((r) => {
          if (cancelled) return;
          if (!r.ready && tries++ < 40) timer = setTimeout(load, 1500); // the index is still loading at server start-up
          else setData(r);
        })
        .catch(() => !cancelled && setData(null)); // a hint only: never an error screen
    load();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [key, matchColour]);

  if (!data || !data.available || (data.buildable.length === 0 && data.almost.length === 0)) return null;
  return (
    <>
      {data.buildable.length > 0 && (
        <Section title="Real LEGO builds you can make" intro="Parts of official LEGO sets that your pieces build exactly." builds={data.buildable} count={data.counts.buildable} onOpen={setViewing} />
      )}
      {data.almost.length > 0 && (
        <Section title="Almost buildable" intro="Real LEGO builds that are only a piece or two short." builds={data.almost} count={data.counts.almost} onOpen={setViewing} />
      )}
      {viewing && <BuildViewer build={viewing} onClose={() => setViewing(null)} />}
    </>
  );
}

function Section({ title, intro, builds, count, onOpen }: { title: string; intro: string; builds: SubBuild[]; count: number; onOpen: (b: BuildMeta) => void }) {
  return (
    <section aria-label={title} className="mt-8">
      <h2 className="title-display text-2xl">{title}</h2>
      <p className="mt-1 text-sm text-muted">
        {intro} {count > builds.length ? `Showing ${builds.length} of ${count}.` : ""}
      </p>
      <ul className="mt-4 grid grid-cols-1 items-start gap-3 lg:grid-cols-2 lg:gap-5">
        {builds.map((b) => (
          <SubBuildCard key={b.id} build={b} onOpen={onOpen} />
        ))}
      </ul>
    </section>
  );
}

function SubBuildCard({ build, onOpen }: { build: SubBuild; onOpen: (b: BuildMeta) => void }) {
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const open = async () => {
    setBusy(true);
    setProblem(null);
    try {
      onOpen(await getSubBuildModel(build.id, build.main ? build.title : `${build.title} (${build.set_name})`));
    } catch (e) {
      setProblem(e instanceof Error ? e.message : "Could not open this build.");
    } finally {
      setBusy(false);
    }
  };
  return (
    <li className="glass-sm p-3">
      <div className="flex gap-3">
        <Thumb srcs={[build.image]} label={build.set_name} className="h-20 w-20" />
        <div className="min-w-0 flex-1">
          <h3 className="line-clamp-2 font-semibold leading-snug">{build.title}</h3>
          <p className="text-xs text-zinc-500">
            {build.main ? "Whole model" : "Part of"} {build.set_name} · {build.set_num} · {build.parts} parts
          </p>
          {build.missing && build.missing.length > 0 && (
            <p className="mt-1 text-sm">
              Missing <strong>{build.missing_count}</strong>:{" "}
              {build.missing.map((m) => `${m.qty} × ${m.name}${m.colour ? ` (${m.colour})` : ""}`).join(", ")}
            </p>
          )}
        </div>
      </div>
      <button onClick={open} disabled={busy} className="mt-3 flex h-11 w-full items-center justify-center rounded-xl btn-glass text-sm font-semibold disabled:opacity-60 active:bg-zinc-100">
        {busy ? "Loading steps…" : "View steps"}
      </button>
      {problem && (
        <p role="alert" className="mt-2 rounded-xl bg-red-50 p-3 text-sm text-red-800">
          {problem}
        </p>
      )}
      <p className="mt-2 text-[11px] text-zinc-500">
        Model by {build.authors.length ? build.authors.join(", ") : "an LDraw contributor"} ·{" "}
        <a href={build.source_url} target="_blank" rel="noreferrer noopener" className="underline underline-offset-2">
          LDraw Official Model Repository
        </a>{" "}
        · {build.licence[0] ? build.licence[0].split(":")[0] : "open licence"}
      </p>
    </li>
  );
}
