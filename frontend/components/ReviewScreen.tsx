"use client";

import { useMemo, useRef, useState } from "react";
import { ApiError, mergePieces, splitPiece } from "@/lib/api";
import { addManual, changePart, confirmItem, isUnmapped, localIndex, pieceOwners, pieceUnderLine, removeItem, scanOfPiece, setQty, toMatchParts, type PartChoice } from "@/lib/review";
import type { Item, Resolved, Scan } from "@/lib/types";
import PartPicker from "./PartPicker";
import PartRow from "./PartRow";
import PhotoBoxes from "./PhotoBoxes";

export default function ReviewScreen({
  scans,
  items,
  setItems,
  resolved,
  onStartOver,
  onAddPhoto,
  onFind,
  onPhotoEdited,
  onDeletePhoto,
}: {
  scans: Scan[];
  items: Item[];
  setItems: (fn: (prev: Item[]) => Item[]) => void;
  resolved: Record<string, Resolved>;
  onStartOver: () => void;
  onAddPhoto: () => void;
  onFind: () => void;
  /** A merge or split changed this photo's pieces: the page swaps in the new scan and rebuilds the list. */
  onPhotoEdited: (oldScan: Scan, newScan: Scan) => void;
  /** Remove a whole photo and the pieces it contributed to the list. */
  onDeletePhoto: (scan: Scan) => void;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [photo, setPhoto] = useState(scans.length - 1); // the photo shown; a newly added photo opens first
  const shown = scans[Math.min(photo, scans.length - 1)];
  const [adding, setAdding] = useState(false);
  const [mode, setMode] = useState<"merge" | "split" | null>(null);
  const [picked, setPicked] = useState<number[]>([]);
  const [menu, setMenu] = useState<{ x: number; y: number; index: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);
  const rowEls = useRef(new Map<string, HTMLLIElement>());

  const owners = useMemo(() => pieceOwners(items), [items]);
  const selected = items.find((i) => i.id === selectedId) ?? null;
  const selectedPieces = useMemo(() => new Set(selected?.pieceIdx ?? []), [selected]);

  const reviewCount = items.filter((i) => i.needsReview).length;
  const usable = toMatchParts(items, resolved);
  const usableQty = usable.reduce((n, p) => n + p.qty, 0);
  const totalQty = items.reduce((n, i) => n + i.qty, 0);
  const unusedQty = items.filter((i) => isUnmapped(i, resolved)).reduce((n, i) => n + i.qty, 0);

  const select = (id: string) => {
    setSelectedId((cur) => (cur === id ? null : id));
    const it = items.find((i) => i.id === id);
    if (it && it.pieceIdx.length > 0) setPhoto(scanOfPiece(scans, it.pieceIdx[0])); // show the photo that holds it
  };

  const edit = async (run: () => Promise<Scan>) => {
    setBusy(true);
    setEditError(null);
    try {
      onPhotoEdited(shown, await run());
      setPicked([]);
      setMode(null);
    } catch (e) {
      setEditError(e instanceof ApiError ? e.message : "That did not work. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  const doMerge = () => {
    if (picked.length !== 2) return;
    edit(() => mergePieces(shown.job_id, [localIndex(shown, picked[0]), localIndex(shown, picked[1])]));
  };

  const onDrawLine = (line: [number, number, number, number]) => {
    const target = pieceUnderLine(shown, line);
    if (!target) {
      setEditError("Draw the line across one of the boxes.");
      return;
    }
    edit(() => splitPiece(shown.job_id, localIndex(shown, target.index), line));
  };

  const onSelectPiece = (pieceIndex: number) => {
    if (mode === "merge") {
      setPicked((cur) => (cur.includes(pieceIndex) ? cur.filter((i) => i !== pieceIndex) : [...cur, pieceIndex].slice(-2)));
      return;
    }
    const owner = owners.get(pieceIndex);
    if (!owner) return;
    setSelectedId(owner.id);
    setPhoto(scanOfPiece(scans, pieceIndex));
    rowEls.current.get(owner.id)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  const deletePhoto = (index: number) => {
    setMenu(null);
    setMode(null);
    setPicked([]);
    setSelectedId(null);
    setPhoto(Math.max(0, Math.min(photo > index ? photo - 1 : photo, scans.length - 2)));
    onDeletePhoto(scans[index]);
  };

  const add = (choice: PartChoice) => {
    setItems((prev) => addManual(prev, choice));
    setAdding(false);
  };

  return (
    <div className="pb-28 md:flex md:h-[calc(100dvh-7.5rem)] md:flex-col md:pb-0">
      {shown.detector_note?.includes("WARNING") && (
        <p role="alert" className="mb-4 rounded-xl bg-amber-50 p-3 text-sm text-amber-900">
          {shown.detector_note.replace(/^.*?WARNING:\s*/, "")}
        </p>
      )}
      <div className="mb-4 md:shrink-0">
        <h1 className="title-display text-3xl">
          Check your <span>pieces</span>
        </h1>
        <p className="mt-1 text-sm text-muted">
          {totalQty} piece{totalQty === 1 ? "" : "s"}
          {reviewCount > 0 ? ` · ${reviewCount} to check` : " · all look good"}
        </p>
      </div>

      <div className="md:grid md:min-h-0 md:flex-1 md:grid-cols-2 md:items-start md:gap-6 lg:gap-10">
        <div className="sticky top-0 z-10 -mx-4 bg-white/85 px-4 pb-3 pt-2 backdrop-blur-xl md:top-3 md:mx-0 md:bg-transparent md:p-0 md:backdrop-blur-none">
          {scans.length > 1 && (
            <div className="mb-2 flex gap-2 overflow-x-auto" role="tablist" aria-label="Your photos">
              {scans.map((s, i) => (
                <button
                  key={s.job_id}
                  role="tab"
                  aria-selected={s === shown}
                  onClick={() => setPhoto(i)}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    setMenu({ x: e.clientX, y: e.clientY, index: i });
                  }}
                  className={`shrink-0 rounded-full border px-3 py-1 text-xs font-medium ${
                    s === shown ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-300 bg-white"
                  }`}
                >
                  Photo {i + 1} · {s.pieces.length}
                </button>
              ))}
            </div>
          )}
          {menu && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setMenu(null)} onContextMenu={(e) => { e.preventDefault(); setMenu(null); }} />
              <div
                role="menu"
                className="fixed z-50 min-w-36 rounded-xl border border-zinc-200 bg-white p-1 text-sm shadow-lg"
                style={{ left: menu.x, top: menu.y }}
              >
                <button role="menuitem" onClick={() => deletePhoto(menu.index)} className="w-full rounded-lg px-3 py-2 text-left font-medium text-red-700 hover:bg-red-50">
                  Delete photo {menu.index + 1}
                </button>
              </div>
            </>
          )}
          <PhotoBoxes
            scan={shown}
            owners={owners}
            selectedPieces={selectedPieces}
            onSelectPiece={onSelectPiece}
            mode={mode}
            picked={picked}
            onDrawLine={onDrawLine}
          />
          <div className="mt-2 flex flex-wrap items-center justify-center gap-2 text-xs">
            <button
              onClick={() => { setMode(mode === "merge" ? null : "merge"); setPicked([]); setEditError(null); }}
              className={`rounded-full border px-3 py-1 font-medium ${mode === "merge" ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-300 bg-white"}`}
            >
              One piece in two boxes? Merge
            </button>
            <button
              onClick={() => { setMode(mode === "split" ? null : "split"); setPicked([]); setEditError(null); }}
              className={`rounded-full border px-3 py-1 font-medium ${mode === "split" ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-300 bg-white"}`}
            >
              Two pieces in one box? Split
            </button>
            {mode === "merge" && (
              <button onClick={doMerge} disabled={picked.length !== 2 || busy} className="rounded-full btn-primary px-3 py-1 font-semibold text-white disabled:opacity-40">
                {busy ? "Merging…" : picked.length === 2 ? "Merge these two" : `Tap ${2 - picked.length} box${picked.length === 1 ? "" : "es"}`}
              </button>
            )}
            {mode === "split" && <span className="text-zinc-600">{busy ? "Splitting…" : "Drag a line across the box where the pieces meet"}</span>}
          </div>
          {editError && <p role="alert" className="mt-1 text-center text-xs text-red-700">{editError}</p>}
          <p className="mt-2 text-center text-xs text-zinc-500">
            <span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-green-600 align-middle" />confident
            <span className="mx-1 ml-3 inline-block h-2.5 w-2.5 rounded-sm bg-amber-500 align-middle" />check this · tap a box to jump to its row
          </p>
        </div>

        <div className="md:h-full md:overflow-y-auto md:pb-2 md:pr-2">
          {items.length === 0 && (
            <div className="glass-sm p-4 text-sm text-zinc-700">
              <p className="font-medium">No pieces found.</p>
              <p className="mt-1 text-zinc-600">
                Try again on a plain, light surface with the pieces spread apart, or add parts by hand below.
              </p>
            </div>
          )}

          <ul className="flex flex-col gap-3">
            {items.map((it) => (
              <PartRow
                key={it.id}
                item={it}
                resolved={it.part_id ? resolved[it.part_id] : undefined}
                resolvedAlts={resolved}
                selected={it.id === selectedId}
                rowRef={(el) => {
                  if (el) rowEls.current.set(it.id, el);
                  else rowEls.current.delete(it.id);
                }}
                onSelect={() => select(it.id)}
                onChangePart={(c) => setItems((prev) => changePart(prev, it.id, c))}
                onConfirm={() => setItems((prev) => confirmItem(prev, it.id))}
                onDelete={() => {
                  setItems((prev) => removeItem(prev, it.id));
                  if (selectedId === it.id) setSelectedId(null);
                }}
                onQty={(q) => setItems((prev) => setQty(prev, it.id, q))}
              />
            ))}
          </ul>

          <div className="mt-4">
            {adding ? (
              <PartPicker onPick={add} onClose={() => setAdding(false)} />
            ) : (
              <button onClick={() => setAdding(true)} className="h-12 w-full rounded-2xl border border-dashed border-zinc-400 text-sm font-medium active:bg-zinc-100">
                + Add a part the scan missed
              </button>
            )}
          </div>

          <p className="mt-3 text-xs text-zinc-500">
            A piece missing, or the same colour as the surface? Tap <span className="font-semibold">+ Add photo</span> and photograph
            it on a surface of a different colour. Its pieces are added to this list.
          </p>

          {unusedQty > 0 && (
            <p className="mt-4 text-xs text-zinc-500">
              {unusedQty} piece{unusedQty === 1 ? "" : "s"} (e.g. minifigure assemblies) can&apos;t be matched against sets and are left out.
            </p>
          )}
        </div>
      </div>

      <div className="pb-safe fixed inset-x-0 bottom-0 z-20 rounded-t-[1.75rem] border-t border-white bg-white/80 px-4 pt-3 shadow-[0_-18px_50px_-20px_rgba(18,18,28,0.35)] backdrop-blur-2xl">
        <div className="mx-auto flex max-w-3xl gap-3 lg:max-w-6xl 2xl:max-w-[88rem] lg:px-8">
          <button
            onClick={() => {
              if (items.length === 0 || window.confirm("Start over? Your list and photos will be cleared.")) onStartOver();
            }}
            className="h-12 whitespace-nowrap rounded-full btn-glass px-4 text-sm font-medium active:bg-zinc-100"
          >
            Start over
          </button>
          <button onClick={onAddPhoto} className="h-12 whitespace-nowrap rounded-full btn-glass px-4 text-sm font-semibold active:bg-zinc-100">
            + Add photo
          </button>
          <button
            onClick={onFind}
            disabled={usable.length === 0}
            className="h-12 flex-1 whitespace-nowrap rounded-full btn-primary text-sm font-semibold text-white disabled:opacity-40 active:opacity-80"
          >
            Find builds ({usableQty})
          </button>
        </div>
      </div>
    </div>
  );
}
