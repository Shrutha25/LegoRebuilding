"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { absUrl, apiBase } from "@/lib/api";
import { DROP_ANIMATION, isTouchDevice, type BuildMeta, type BuildPartUse } from "@/lib/build";
import type { BuildScene } from "@/lib/buildScene";

/**
 * Step-by-step 3D building instructions for ONE build (AI or official set). three.js is imported when the viewer opens, so none
 * of it is in the main bundle. Step 1 .. N, then a finished-model screen.
 */
export default function BuildViewer({
  build,
  onClose,
  ldconfigUrl,
  initialStep = 0,
  startFinished = false,
  exposeScene,
}: {
  build: BuildMeta;
  onClose: () => void; // "Back to suggestions"
  ldconfigUrl?: string;
  initialStep?: number;
  startFinished?: boolean;
  exposeScene?: (scene: BuildScene) => void; // tests and screenshots only
}) {
  const box = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<BuildScene | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");
  const [pos, setPos] = useState(initialStep); // index into `nav`
  const [done, setDone] = useState(startFinished);

  /** Steps that add at least one part (a step number the model skips is not shown). */
  const nav = useMemo(() => {
    const list = build.steps.filter((s) => s.parts.length > 0).map((s) => s.index);
    return list.length ? list : [0];
  }, [build]);
  const last = nav.length - 1;
  const step = build.steps.find((s) => s.index === nav[Math.min(pos, last)]);

  // ---- load three.js, the packed model and the colour table
  useEffect(() => {
    let cancelled = false;
    let scene: BuildScene | null = null;
    (async () => {
      try {
        const lite = isTouchDevice();
        const [{ BuildScene }, modelText, ldconfig] = await Promise.all([
          import("@/lib/buildScene"),
          fetchText(absUrl(lite ? build.lite_model_url : build.model_url)),
          fetchText(ldconfigUrl ?? `${apiBase()}/api/ldraw/LDConfig.ldr`),
        ]);
        if (cancelled || !box.current) return;
        scene = new BuildScene(box.current, { lite, dropIn: DROP_ANIMATION });
        await scene.load(modelText, ldconfig);
        if (cancelled) {
          scene.dispose();
          return;
        }
        sceneRef.current = scene;
        exposeScene?.(scene);
        if (startFinished) scene.finish();
        else scene.setStep(nav[Math.min(initialStep, last)] ?? 0, false);
        setStatus("ready");
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Could not show this build.");
          setStatus("error");
        }
      }
    })();
    return () => {
      cancelled = true;
      sceneRef.current = null;
      scene?.dispose();
    };
    // the scene is built once per build; steps are driven by the effects below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [build.id]);

  // ---- drive the scene from the step / finished state
  useEffect(() => {
    const scene = sceneRef.current;
    if (status !== "ready" || !scene) return;
    if (done) scene.finish();
    else scene.setStep(nav[Math.min(pos, last)] ?? 0, true);
  }, [pos, done, status, nav, last]);

  const go = useCallback(
    (to: number) => {
      if (done) setDone(false);
      setPos(Math.max(0, Math.min(last, to)));
    },
    [done, last],
  );
  const next = useCallback(() => {
    if (pos >= last) setDone(true);
    else go(pos + 1);
  }, [pos, last, go]);
  const back = useCallback(() => {
    if (done) setDone(false);
    else go(pos - 1);
  }, [done, pos, go]);

  // keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") next();
      else if (e.key === "ArrowLeft") back();
      else if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, back, onClose]);

  // swipe on the instruction panel (the canvas keeps drag = rotate)
  const swipe = useRef<{ x: number; y: number } | null>(null);
  const panelHandlers = {
    onPointerDown: (e: React.PointerEvent) => {
      swipe.current = { x: e.clientX, y: e.clientY };
    },
    onPointerUp: (e: React.PointerEvent) => {
      const s = swipe.current;
      swipe.current = null;
      if (!s) return;
      const dx = e.clientX - s.x;
      const dy = e.clientY - s.y;
      if (Math.abs(dx) > 50 && Math.abs(dx) > 2 * Math.abs(dy)) {
        if (dx < 0) next();
        else back();
      }
    },
  };

  const attribution = build.attribution;
  const adjusted = build.repair_summary ? build.repair_summary.dropped + build.repair_summary.swapped + build.repair_summary.nudged : 0;

  return createPortal(
    <div role="dialog" aria-modal="true" aria-label={`3D instructions: ${build.title}`} className="fixed inset-0 z-50 flex flex-col bg-[#f4f4f6]">
      <header className="flex h-14 shrink-0 items-center gap-2 border-b border-white bg-white/80 px-3 shadow-[0_8px_30px_-18px_rgba(18,18,28,0.3)] backdrop-blur-xl">
        <button onClick={onClose} aria-label="Close the 3D view" className="h-10 rounded-full px-3 text-xs font-bold uppercase tracking-wider text-zinc-700 active:bg-zinc-100">
          ✕ Close
        </button>
        <h2 className="min-w-0 flex-1 truncate text-center text-xs font-extrabold uppercase tracking-[0.18em]">{build.title}</h2>
        <span className="w-[4.5rem]" aria-hidden />
      </header>

      <div className="relative min-h-0 flex-1" ref={box}>
        {status === "ready" && (
          <button
            onClick={() => sceneRef.current?.resetView()}
            aria-label="Reset view"
            className="btn-glass absolute right-3 top-3 z-10 h-10 rounded-full px-4 text-[11px] font-bold backdrop-blur-xl"
          >
            ⟲ Reset view
          </button>
        )}
        {status === "loading" && (
          <div role="status" className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-zinc-100 text-center">
            <span className="h-9 w-9 animate-spin rounded-full border-4 border-zinc-300 border-t-ink" />
            <p className="font-medium">Loading the bricks…</p>
            <p className="max-w-xs text-sm text-zinc-500">{build.part_count} parts, {build.step_count} steps</p>
          </div>
        )}
        {status === "error" && (
          <div role="alert" className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-zinc-100 p-6 text-center">
            <p className="font-semibold">Can&apos;t show this build in 3D</p>
            <p className="max-w-sm text-sm text-zinc-600">{error}</p>
            <button onClick={onClose} className="h-11 rounded-full btn-primary px-5 font-semibold text-white">
              Back
            </button>
          </div>
        )}
      </div>

      <section {...panelHandlers} className="pb-safe shrink-0 touch-pan-y rounded-t-[1.75rem] border-t border-white bg-white/85 px-4 pt-4 shadow-[0_-18px_50px_-20px_rgba(18,18,28,0.35)] backdrop-blur-2xl" aria-label="Instructions">
        {done ? (
          <div className="flex flex-col gap-3 pb-3 text-center">
            <p className="title-display text-2xl">Model <span>complete</span></p>
            <p className="text-sm text-zinc-600">{build.title}: {build.part_count} parts in {nav.length} steps.</p>
            <div className="flex gap-3">
              <button onClick={() => { setDone(false); setPos(0); }} className="h-12 flex-1 rounded-full btn-glass font-semibold active:bg-zinc-100">
                Build again
              </button>
              <button onClick={onClose} className="h-12 flex-1 rounded-full btn-primary font-semibold text-white active:opacity-80">
                Back to suggestions
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="flex items-baseline justify-between">
              <p className="text-base font-extrabold uppercase tracking-wide">
                Step {Math.min(pos, last) + 1} of {nav.length}
              </p>
              <p className="text-xs text-zinc-500">{Math.round(((Math.min(pos, last) + 1) / nav.length) * 100)}%</p>
            </div>
            <ProgressBar count={nav.length} value={Math.min(pos, last)} onChange={go} />

            <ul className="mt-3 flex max-h-40 flex-wrap gap-2 overflow-y-auto pb-1" aria-label="Parts for this step">
              {(step?.parts ?? []).map((p) => (
                <PartCard key={`${p.ld_part}:${p.colour_code}`} part={p} sceneRef={sceneRef} ready={status === "ready"} />
              ))}
            </ul>

            <div className="mt-2 flex gap-3 pb-2">
              <button onClick={back} disabled={pos === 0} className="h-12 flex-1 rounded-full btn-glass font-semibold disabled:opacity-30 active:bg-zinc-100">
                ← Previous
              </button>
              <button onClick={next} disabled={status !== "ready"} className="h-12 flex-[2] rounded-full btn-primary text-base font-semibold text-white disabled:opacity-40 active:opacity-80">
                {pos >= last ? "Finish" : "Next step →"}
              </button>
            </div>
          </>
        )}
        <p className="pb-2 text-[11px] leading-snug text-zinc-500">
          {attribution && (
            <>
              {attribution.text} Licence: {attribution.licence.join("; ")}.{" "}
              <a href={attribution.source_url} target="_blank" rel="noreferrer noopener" className="underline">
                LDraw OMR
              </a>
              {build.steps_generated && " This model had no steps; the step order was generated automatically."}{" "}
            </>
          )}
          {!attribution && adjusted > 0 && <>AI design, checked so it holds together: {adjusted} part{adjusted === 1 ? "" : "s"} adjusted automatically. </>}
          3D parts: LDraw.org Parts Library (
          <a href="https://www.ldraw.org" target="_blank" rel="noreferrer noopener" className="underline">
            CC BY 4.0
          </a>
          ).
        </p>
      </section>
    </div>,
    document.body,
  );
}

async function fetchText(url: string): Promise<string> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.status === 503 ? "The 3D parts library is not installed on the server." : `Could not load ${url.split("/").pop()} (${r.status}).`);
  return r.text();
}

/** The progress bar doubles as a scrubber: tap or drag to jump to a step. */
function ProgressBar({ count, value, onChange }: { count: number; value: number; onChange: (i: number) => void }) {
  const el = useRef<HTMLDivElement>(null);
  const at = (clientX: number) => {
    const r = el.current!.getBoundingClientRect();
    onChange(Math.round(Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * (count - 1)));
  };
  const drag = useRef(false);
  return (
    <div
      ref={el}
      role="slider"
      tabIndex={0}
      aria-label="Step"
      aria-valuemin={1}
      aria-valuemax={count}
      aria-valuenow={value + 1}
      className="relative mt-1 h-8 cursor-pointer touch-none select-none"
      onPointerDown={(e) => {
        e.stopPropagation();
        drag.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        at(e.clientX);
      }}
      onPointerMove={(e) => drag.current && at(e.clientX)}
      onPointerUp={(e) => {
        e.stopPropagation();
        drag.current = false;
      }}
      onKeyDown={(e) => {
        if (e.key === "ArrowRight") onChange(value + 1);
        if (e.key === "ArrowLeft") onChange(value - 1);
      }}
    >
      <div className="absolute inset-x-0 top-1/2 h-2 -translate-y-1/2 rounded-full bg-zinc-200" />
      <div className="absolute left-0 top-1/2 h-2 -translate-y-1/2 rounded-full bg-ink" style={{ width: `${count > 1 ? (value / (count - 1)) * 100 : 100}%` }} />
      <div className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-ink shadow" style={{ left: `${count > 1 ? (value / (count - 1)) * 100 : 100}%` }} />
    </div>
  );
}

/** One part the step adds: rendered thumbnail, count, name and colour. */
function PartCard({ part, sceneRef, ready }: { part: BuildPartUse; sceneRef: React.RefObject<BuildScene | null>; ready: boolean }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    const scene = sceneRef.current;
    if (ready && scene) scene.thumbnail(part.ld_part, part.colour_code).then((u) => !cancelled && setUrl(u)).catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [part.ld_part, part.colour_code, ready, sceneRef]);
  return (
    <li className="glass-sm flex w-[calc(50%-0.25rem)] min-w-[10.5rem] items-center gap-2 p-2">
      <span className="flex h-14 w-14 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-white" style={{ boxShadow: `inset 0 -3px 0 ${part.hex}` }}>
        {url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url} alt="" className="h-full w-full object-contain" />
        ) : (
          <span className="h-6 w-8 rounded-sm" style={{ background: part.hex }} aria-hidden />
        )}
      </span>
      <span className="min-w-0 text-xs leading-tight">
        <span className="block text-sm font-bold">{part.qty}×</span>
        <span className="line-clamp-2 block font-medium">{part.name}</span>
        <span className="mt-0.5 flex items-center gap-1 text-zinc-600">
          <span className="inline-block h-2.5 w-2.5 rounded-full border border-zinc-300" style={{ background: part.hex }} aria-hidden />
          {part.colour_name}
        </span>
      </span>
    </li>
  );
}
