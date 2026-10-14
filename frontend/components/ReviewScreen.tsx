"use client";

import { useMemo, useRef, useState } from "react";
import { addManual, changePart, confirmItem, isUnmapped, pieceOwners, removeItem, setQty, toMatchParts, type PartChoice } from "@/lib/review";
import type { Item, Resolved, Scan } from "@/lib/types";
import PartPicker from "./PartPicker";
import PartRow from "./PartRow";
import PhotoBoxes from "./PhotoBoxes";

export default function ReviewScreen({
  scan,
  items,
  setItems,
  resolved,
  onRetake,
  onFind,
}: {
  scan: Scan;
  items: Item[];
  setItems: (fn: (prev: Item[]) => Item[]) => void;
  resolved: Record<string, Resolved>;
  onRetake: () => void;
  onFind: () => void;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const rowEls = useRef(new Map<string, HTMLLIElement>());

  const owners = useMemo(() => pieceOwners(items), [items]);
  const selected = items.find((i) => i.id === selectedId) ?? null;
  const selectedPieces = useMemo(() => new Set(selected?.pieceIdx ?? []), [selected]);

  const reviewCount = items.filter((i) => i.needsReview).length;
  const usable = toMatchParts(items, resolved);
  const usableQty = usable.reduce((n, p) => n + p.qty, 0);
  const totalQty = items.reduce((n, i) => n + i.qty, 0);
  const unusedQty = items.filter((i) => isUnmapped(i, resolved)).reduce((n, i) => n + i.qty, 0);

  const select = (id: string) => setSelectedId((cur) => (cur === id ? null : id));

  const onSelectPiece = (pieceIndex: number) => {
    const owner = owners.get(pieceIndex);
    if (!owner) return;
    setSelectedId(owner.id);
    rowEls.current.get(owner.id)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  const add = (choice: PartChoice) => {
    setItems((prev) => addManual(prev, choice));
    setAdding(false);
  };

  return (
    <div className="pb-28">
      <div className="mb-4">
        <h1 className="text-xl font-bold">Check your pieces</h1>
        <p className="text-sm text-stone-600">
          {totalQty} piece{totalQty === 1 ? "" : "s"}
          {reviewCount > 0 ? ` · ${reviewCount} to check` : " · all look good"}
        </p>
      </div>

      <div className="md:grid md:grid-cols-2 md:items-start md:gap-6">
        <div className="sticky top-0 z-10 -mx-4 bg-background/95 px-4 pb-3 pt-2 backdrop-blur md:static md:mx-0 md:p-0">
          <PhotoBoxes scan={scan} owners={owners} selectedPieces={selectedPieces} onSelectPiece={onSelectPiece} />
          <p className="mt-2 text-center text-xs text-stone-500">
            <span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-green-600 align-middle" />confident
            <span className="mx-1 ml-3 inline-block h-2.5 w-2.5 rounded-sm bg-amber-500 align-middle" />check this · tap a box to jump to its row
          </p>
        </div>

        <div>
          {items.length === 0 && (
            <div className="rounded-2xl border border-stone-200 bg-white p-4 text-sm text-stone-700">
              <p className="font-medium">No pieces found.</p>
              <p className="mt-1 text-stone-600">
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
              <button onClick={() => setAdding(true)} className="h-12 w-full rounded-2xl border border-dashed border-stone-400 text-sm font-medium active:bg-stone-100">
                + Add a part the scan missed
              </button>
            )}
          </div>

          {unusedQty > 0 && (
            <p className="mt-4 text-xs text-stone-500">
              {unusedQty} piece{unusedQty === 1 ? "" : "s"} (e.g. minifigure assemblies) can&apos;t be matched against sets and are left out.
            </p>
          )}
        </div>
      </div>

      <div className="pb-safe fixed inset-x-0 bottom-0 z-20 border-t border-stone-200 bg-white/95 px-4 pt-3 backdrop-blur">
        <div className="mx-auto flex max-w-3xl gap-3">
          <button onClick={onRetake} className="h-12 rounded-2xl border border-stone-300 px-4 text-sm font-medium active:bg-stone-100">
            Retake
          </button>
          <button
            onClick={onFind}
            disabled={usable.length === 0}
            className="h-12 flex-1 rounded-2xl bg-brick text-base font-semibold text-white disabled:opacity-40 active:opacity-80"
          >
            Find builds ({usableQty} piece{usableQty === 1 ? "" : "s"})
          </button>
        </div>
      </div>
    </div>
  );
}
