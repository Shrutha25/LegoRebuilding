"use client";

import { useEffect, useState } from "react";
import BuildViewer from "@/components/BuildViewerLazy";
import { absUrl, apiBase } from "@/lib/api";
import type { BuildMeta } from "@/lib/build";

/**
 * Dev page for reviewing recipes: every recipe on several random but realistic inventories (the backend needs RECIPE_GALLERY=1).
 * Each card is the build's isometric thumbnail; "3D" opens it in the real step-by-step viewer.  ?seed=2 gives other inventories.
 */
type Row = { seed: number; build: BuildMeta | null; inventory: string; pieces: number };
type Gallery = { recipes: { recipe: string; name: string; rows: Row[] }[] };

export default function RecipeGallery() {
  const [data, setData] = useState<Gallery | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewing, setViewing] = useState<BuildMeta | null>(null);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const seed = Number(q.get("seed") ?? 1) || 1;
    fetch(`${apiBase()}/api/dev/recipe-gallery?seed=${seed}&per=5`)
      .then(async (r) => {
        if (!r.ok) throw new Error(r.status === 404 ? "Start the backend with RECIPE_GALLERY=1 to enable the gallery." : `Request failed (${r.status})`);
        setData(await r.json());
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load the gallery."));
  }, []);

  return (
    <main className="mx-auto w-full max-w-[110rem] px-4 py-6">
      <h1 className="title-display text-3xl">
        Recipe <span>gallery</span>
      </h1>
      <p className="mt-1 text-sm text-muted">Each recipe on 5 random inventories. Open a card in 3D to check the steps.</p>
      {error && (
        <p role="alert" className="mt-4 rounded-2xl bg-red-50 p-4 text-sm text-red-800">
          {error}
        </p>
      )}
      {!data && !error && <p className="mt-6 text-sm">Building the gallery…</p>}
      {data?.recipes.map((r) => (
        <section key={r.recipe} className="mt-8" aria-label={r.name}>
          <h2 className="eyebrow mb-3">{r.name}</h2>
          <ul className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
            {r.rows.map((row) => (
              <li key={row.seed} className="glass-sm flex flex-col p-2">
                {row.build ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={absUrl(row.build.thumb_url)} alt={row.build.title} className="aspect-[21/17] w-full rounded-xl bg-white object-contain" />
                    <p className="mt-2 text-sm font-bold">{row.build.title}</p>
                    <p className="text-xs text-muted">
                      {row.build.part_count} parts · {row.build.step_count} steps · {row.pieces} pieces owned
                    </p>
                    <p className="mt-1 line-clamp-2 text-[11px] text-zinc-500">{row.inventory}</p>
                    <button onClick={() => setViewing(row.build)} className="mt-2 h-9 rounded-full btn-glass text-xs font-bold">
                      3D
                    </button>
                  </>
                ) : (
                  <p className="p-4 text-xs text-zinc-500">No {r.name.toLowerCase()} from this inventory ({row.pieces} pieces).</p>
                )}
              </li>
            ))}
          </ul>
        </section>
      ))}
      {viewing && <BuildViewer build={viewing} onClose={() => setViewing(null)} />}
    </main>
  );
}
