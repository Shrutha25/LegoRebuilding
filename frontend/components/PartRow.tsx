"use client";

import { useState } from "react";
import { absUrl, imgSrc } from "@/lib/api";
import type { PartChoice } from "@/lib/review";
import type { Item, Resolved } from "@/lib/types";
import PartPicker from "./PartPicker";
import Thumb from "./Thumb";

export default function PartRow({
  item,
  resolved,
  resolvedAlts,
  selected,
  onSelect,
  onChangePart,
  onConfirm,
  onDelete,
  onQty,
  rowRef,
}: {
  item: Item;
  resolved: Resolved | undefined;
  resolvedAlts: Record<string, Resolved>;
  selected: boolean;
  onSelect: () => void;
  onChangePart: (choice: PartChoice) => void;
  onConfirm: () => void;
  onDelete: () => void;
  onQty: (qty: number) => void;
  rowRef: (el: HTMLLIElement | null) => void;
}) {
  const [changing, setChanging] = useState(false);
  const [searching, setSearching] = useState(false);
  const unmapped = resolved?.unmapped === true;
  const open = item.needsReview || changing;

  const tone = item.needsReview
    ? "border-amber-300 bg-amber-50"
    : unmapped
      ? "border-zinc-200 bg-zinc-100"
      : "border-zinc-200 bg-white";

  const pick = (c: PartChoice) => {
    onChangePart(c);
    setChanging(false);
    setSearching(false);
  };

  return (
    <li ref={rowRef} className={`scroll-mt-[44vh] rounded-2xl border p-3 md:scroll-mt-4 ${tone} ${selected ? "ring-2 ring-zinc-900" : ""}`}>
      <div className="flex items-start gap-3">
        <button onClick={onSelect} className="flex min-w-0 flex-1 items-start gap-3 text-left" aria-label={`Show ${item.name} in the photo`}>
          <Thumb srcs={[item.cropUrl ? absUrl(item.cropUrl) : null]} label="Your piece" />
          <Thumb srcs={[imgSrc(item.image), resolved?.image]} label="Matched part" />
          <span className="min-w-0 flex-1">
            <span className="line-clamp-2 text-sm font-semibold">{item.name}</span>
            <span className="block text-xs text-zinc-600">
              {item.part_id ?? "no match"}
              {item.color ? ` · ${item.color}` : ""}
            </span>
            <span className="mt-1 flex flex-wrap gap-1">
              {item.needsReview && <span className="rounded-full bg-amber-200 px-2 py-0.5 text-xs font-medium text-amber-900">Check this</span>}
              {unmapped && <span className="rounded-full bg-zinc-300 px-2 py-0.5 text-xs font-medium text-zinc-700">Not used for matching</span>}
              {item.pieceIdx.length > 0 && (
                <span className="text-xs text-zinc-500">
                  {item.pieceIdx.length === 1 ? "Piece" : "Pieces"} {item.pieceIdx.slice(0, 6).map((i) => i + 1).join(", ")}
                  {item.pieceIdx.length > 6 ? "…" : ""}
                </span>
              )}
            </span>
          </span>
        </button>

        <div className="flex shrink-0 items-center gap-1" aria-label="Quantity">
          <button onClick={() => onQty(item.qty - 1)} disabled={item.qty <= 1} aria-label="Decrease quantity" className="h-9 w-9 rounded-full btn-glass text-lg disabled:opacity-30 active:bg-zinc-100">
            −
          </button>
          <span className="w-7 text-center text-base font-semibold tabular-nums">{item.qty}</span>
          <button onClick={() => onQty(item.qty + 1)} aria-label="Increase quantity" className="h-9 w-9 rounded-full btn-glass text-lg active:bg-zinc-100">
            +
          </button>
        </div>
      </div>

      {open && !searching && item.alternatives.length > 0 && (
        <div className="mt-3">
          <p className="mb-1 text-xs font-medium text-zinc-600">Or is it one of these?</p>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {item.alternatives.map((a) => (
              <button
                key={a.part_id}
                onClick={() => pick({ part_id: a.part_id, name: a.name, image: a.image })}
                className="flex w-32 shrink-0 flex-col items-center gap-1 rounded-xl btn-glass p-2 text-center active:bg-zinc-100"
              >
                <Thumb srcs={[imgSrc(a.image), resolvedAlts[a.part_id]?.image]} label={a.name} />
                <span className="line-clamp-2 text-xs leading-tight">{a.name}</span>
                <span className="text-[11px] text-zinc-500">{a.part_id}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {searching && (
        <div className="mt-3">
          <PartPicker onPick={pick} onClose={() => setSearching(false)} />
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {item.needsReview && item.part_id && (
          <button onClick={onConfirm} className="h-10 rounded-full bg-green-600 px-4 text-sm font-semibold text-white active:opacity-80">
            ✓ Looks right
          </button>
        )}
        {!item.needsReview && !changing && (
          <button
            onClick={() => {
              setChanging(true);
              if (item.alternatives.length === 0) setSearching(true);
            }}
            className="h-10 rounded-full btn-glass px-4 text-sm font-medium active:bg-zinc-100"
          >
            Change part
          </button>
        )}
        {!item.needsReview && changing && (
          <button
            onClick={() => {
              setChanging(false);
              setSearching(false);
            }}
            className="h-10 rounded-full btn-glass px-4 text-sm font-medium active:bg-zinc-100"
          >
            Done
          </button>
        )}
        {open && !searching && (
          <button onClick={() => setSearching(true)} className="h-10 rounded-full btn-glass px-4 text-sm font-medium active:bg-zinc-100">
            Search parts…
          </button>
        )}
        <button onClick={onDelete} className="ml-auto h-10 rounded-full px-3 text-sm font-medium text-red-700 active:bg-red-50">
          Remove
        </button>
      </div>
    </li>
  );
}
