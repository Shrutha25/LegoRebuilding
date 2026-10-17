import type { Alt, IdeaPart, Item, MatchPart, Piece, Resolved, Scan } from "./types";

const MAX_QTY = 999;

/** Group detected pieces into rows: identical (part, colour) become one row with a qty. */
export function groupPieces(pieces: Piece[]): Item[] {
  const items: Item[] = [];
  const byKey = new Map<string, Item>();
  for (const p of pieces) {
    if (!p.part_id) {
      items.push({
        id: `unknown-${p.index}`,
        part_id: null,
        name: "Unrecognized piece",
        color: null,
        qty: 1,
        pieceIdx: [p.index],
        needsReview: true,
        alternatives: [],
        cropUrl: p.crop_url,
        image: null,
      });
      continue;
    }
    const key = `${p.part_id}|${p.color ?? ""}`;
    const existing = byKey.get(key);
    if (existing) {
      existing.qty += 1;
      existing.pieceIdx.push(p.index);
      existing.needsReview ||= p.needs_review;
      continue;
    }
    const item: Item = {
      id: `part-${key}`,
      part_id: p.part_id,
      name: p.name ?? p.part_id,
      color: p.color,
      qty: 1,
      pieceIdx: [p.index],
      needsReview: p.needs_review,
      alternatives: p.alternatives,
      cropUrl: p.crop_url,
      image: p.image ?? null,
    };
    byKey.set(key, item);
    items.push(item);
  }
  return sortItems(items);
}

/**
 * Pieces are numbered across ALL photos of a session (photo 2 carries on where photo 1 stopped), so a row can own pieces from
 * several photos and a box number is unique. A scan's piece `index` is shifted by `offset`; crop URLs are unaffected.
 */
export function withOffset(scan: Scan, offset: number): Scan {
  return { ...scan, pieces: scan.pieces.map((p) => ({ ...p, index: p.index + offset })) };
}

/** First piece number for the next photo: one past the highest number used so far. */
export const nextOffset = (scans: Scan[]): number =>
  scans.reduce((n, s) => Math.max(n, ...s.pieces.map((p) => p.index + 1)), 0);

/** Add the pieces of a new photo to the list: identical parts across photos become one row with a qty. */
export function addPieces(items: Item[], pieces: Piece[]): Item[] {
  return sortItems(mergeDuplicates([...items, ...groupPieces(pieces)]));
}

/**
 * A photo's pieces changed (two merged, or one split): take the old pieces off the list and add the new ones. Rows keep what the
 * user did to them (a corrected part, a confirmed row); a row whose pieces are all gone disappears.
 */
export function replacePieces(items: Item[], oldPieces: Piece[], newPieces: Piece[]): Item[] {
  const gone = new Set(oldPieces.map((p) => p.index));
  const kept: Item[] = [];
  for (const it of items) {
    const left = it.pieceIdx.filter((i) => !gone.has(i));
    if (left.length === it.pieceIdx.length) kept.push(it);
    else if (left.length > 0) kept.push({ ...it, pieceIdx: left, qty: Math.max(1, it.qty - (it.pieceIdx.length - left.length)) });
  }
  return addPieces(kept, newPieces);
}

/** A photo's piece number as the server knows it (each photo numbers its pieces from 0). */
export const localIndex = (scan: Scan, index: number): number => index - Math.min(...scan.pieces.map((p) => p.index));

/** The piece whose box holds the middle of a line drawn on the photo, else the one the line crosses most. */
export function pieceUnderLine(scan: Scan, line: [number, number, number, number]): Piece | null {
  const [x0, y0, x1, y1] = line;
  let best: Piece | null = null;
  let bestHits = 0;
  for (const p of scan.pieces) {
    const [x, y, w, h] = p.box;
    let hits = 0;
    for (let i = 0; i <= 20; i++) {
      const px = x0 + ((x1 - x0) * i) / 20;
      const py = y0 + ((y1 - y0) * i) / 20;
      if (px >= x && px <= x + w && py >= y && py <= y + h) hits++;
    }
    if (hits > bestHits) {
      best = p;
      bestHits = hits;
    }
  }
  return best;
}

/** The photo that holds a given (global) piece number. */
export const scanOfPiece = (scans: Scan[], index: number): number =>
  Math.max(0, scans.findIndex((s) => s.pieces.some((p) => p.index === index)));

/** Rows that need review first; otherwise keep detection order. */
export function sortItems(items: Item[]): Item[] {
  return items
    .map((it, i) => ({ it, i }))
    .sort((a, b) => Number(b.it.needsReview) - Number(a.it.needsReview) || a.i - b.i)
    .map((x) => x.it);
}

/** Merge rows that ended up with the same part and colour (e.g. after a correction). */
export function mergeDuplicates(items: Item[]): Item[] {
  const out: Item[] = [];
  const seen = new Map<string, Item>();
  for (const it of items) {
    if (!it.part_id) {
      out.push(it);
      continue;
    }
    const key = `${it.part_id}|${it.color ?? ""}`;
    const prev = seen.get(key);
    if (prev) {
      prev.qty = Math.min(MAX_QTY, prev.qty + it.qty);
      prev.pieceIdx = [...prev.pieceIdx, ...it.pieceIdx];
      prev.needsReview ||= it.needsReview;
      prev.cropUrl ??= it.cropUrl;
    } else {
      const copy = { ...it, pieceIdx: [...it.pieceIdx] };
      seen.set(key, copy);
      out.push(copy);
    }
  }
  return out;
}

export type PartChoice = { part_id: string; name: string; image?: string | null };

/** Replace a row's part with the user's pick; the old guess stays available as an alternative. */
export function changePart(items: Item[], id: string, choice: PartChoice): Item[] {
  const next = items.map((it) => {
    if (it.id !== id) return it;
    const old: Alt[] = it.part_id ? [{ part_id: it.part_id, name: it.name, score: 0, image: it.image }] : [];
    const alternatives = [...old, ...it.alternatives].filter((a) => a.part_id !== choice.part_id);
    return {
      ...it,
      part_id: choice.part_id,
      name: choice.name,
      image: choice.image ?? null,
      alternatives,
      needsReview: false,
    };
  });
  return mergeDuplicates(next);
}

export const confirmItem = (items: Item[], id: string): Item[] =>
  items.map((it) => (it.id === id ? { ...it, needsReview: false } : it));

export const removeItem = (items: Item[], id: string): Item[] => items.filter((it) => it.id !== id);

export const setQty = (items: Item[], id: string, qty: number): Item[] =>
  items.map((it) =>
    it.id === id ? { ...it, qty: Math.max(1, Math.min(MAX_QTY, Math.round(qty) || 1)) } : it,
  );

/** A part the user added by hand because the scan missed it. */
export function addManual(items: Item[], choice: PartChoice): Item[] {
  const dup = items.find((it) => it.part_id === choice.part_id && it.color === null);
  if (dup) return setQty(items, dup.id, dup.qty + 1);
  return [
    ...items,
    {
      id: `manual-${choice.part_id}-${items.length}`,
      part_id: choice.part_id,
      name: choice.name,
      color: null,
      qty: 1,
      pieceIdx: [],
      needsReview: false,
      alternatives: [],
      cropUrl: null,
      image: choice.image ?? null,
    },
  ];
}

export const isUnmapped = (it: Item, resolved: Record<string, Resolved>) =>
  !!it.part_id && resolved[it.part_id]?.unmapped === true;

/** Rows that will be sent to /api/match (identified and mappable). */
export function toMatchParts(items: Item[], resolved: Record<string, Resolved>): MatchPart[] {
  return items
    .filter((it) => it.part_id && !isUnmapped(it, resolved))
    .map((it) => ({ part_num: it.part_id as string, color: it.color, qty: it.qty }));
}

/** Map each detected piece index to the row that currently owns it. */
export function pieceOwners(items: Item[]): Map<number, Item> {
  const m = new Map<number, Item>();
  for (const it of items) for (const i of it.pieceIdx) m.set(i, it);
  return m;
}

/** Every identified row, including parts that can't be matched to sets (they are still real pieces). */
export function toIdeaParts(items: Item[]): IdeaPart[] {
  return items
    .filter((it) => it.part_id)
    .map((it) => ({ part_num: it.part_id as string, name: it.name, color: it.color, qty: it.qty }));
}
