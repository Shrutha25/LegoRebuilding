import assert from "node:assert/strict";
import { test } from "node:test";
import {
  addManual, changePart, confirmItem, groupPieces, isUnmapped, pieceOwners, removeItem, setQty,
  toIdeaParts, toMatchParts,
} from "../lib/review.ts";
import type { Piece } from "../lib/types.ts";

const piece = (index: number, part_id: string | null, color: string | null, needs = false, alt: string[] = []): Piece => ({
  index, box: [0, 0, 10, 10], crop_url: `/crop/${index}`, part_id, name: part_id ? `Part ${part_id}` : null,
  color, score: needs ? 0.4 : 0.9, alternatives: alt.map((a) => ({ part_id: a, name: `Part ${a}`, score: 0.3 })),
  needs_review: needs, error: part_id ? null : "timeout",
});

test("identical parts are grouped into one row with qty", () => {
  const items = groupPieces([piece(0, "3001", "Red"), piece(1, "3002", "Red"), piece(2, "3001", "Red")]);
  assert.equal(items.length, 2);
  assert.deepEqual(items.find((i) => i.part_id === "3001")?.pieceIdx, [0, 2]);
  assert.equal(items.find((i) => i.part_id === "3001")?.qty, 2);
});

test("same part in a different colour is a separate row", () => {
  assert.equal(groupPieces([piece(0, "3001", "Red"), piece(1, "3001", "Blue")]).length, 2);
});

test("rows needing review come first; unrecognized pieces are never grouped", () => {
  const items = groupPieces([piece(0, "3001", "Red"), piece(1, null, null), piece(2, "3002", "Red", true), piece(3, null, null)]);
  assert.deepEqual(items.map((i) => i.needsReview), [true, true, true, false]);
  assert.equal(items.filter((i) => i.part_id === null).length, 2);
});

test("a group needs review if any of its pieces does", () => {
  const [it] = groupPieces([piece(0, "3001", "Red"), piece(1, "3001", "Red", true)]);
  assert.equal(it.needsReview, true);
});

test("picking an alternative clears review and keeps the old guess as an option", () => {
  const items = groupPieces([piece(0, "3001", "Red", true, ["3002"])]);
  const out = changePart(items, items[0].id, { part_id: "3002", name: "Part 3002" });
  assert.equal(out[0].part_id, "3002");
  assert.equal(out[0].needsReview, false);
  assert.deepEqual(out[0].alternatives.map((a) => a.part_id), ["3001"]);
});

test("changing a row to a part that already has a row merges them", () => {
  const items = groupPieces([piece(0, "3001", "Red"), piece(1, "3002", "Red", true), piece(2, "3002", "Red", true)]);
  const target = items.find((i) => i.part_id === "3002")!;
  const out = changePart(items, target.id, { part_id: "3001", name: "Part 3001" });
  assert.equal(out.length, 1);
  assert.equal(out[0].qty, 3);
  assert.deepEqual(out[0].pieceIdx.sort(), [0, 1, 2]);
});

test("qty is clamped and rounded; delete removes the row and frees its boxes", () => {
  const items = groupPieces([piece(0, "3001", "Red")]);
  assert.equal(setQty(items, items[0].id, 0)[0].qty, 1);
  assert.equal(setQty(items, items[0].id, 5000)[0].qty, 999);
  assert.equal(setQty(items, items[0].id, NaN)[0].qty, 1);
  assert.equal(pieceOwners(removeItem(items, items[0].id)).size, 0);
});

test("manually added parts have no boxes; adding twice bumps qty", () => {
  let items = addManual([], { part_id: "3001", name: "Brick" });
  items = addManual(items, { part_id: "3001", name: "Brick" });
  assert.equal(items.length, 1);
  assert.equal(items[0].qty, 2);
  assert.deepEqual(items[0].pieceIdx, []);
});

test("unmapped and unidentified rows are left out of the match request", () => {
  const items = groupPieces([piece(0, "3001", "Red"), piece(1, "970c00", "Black"), piece(2, null, null)]);
  const resolved = {
    "3001": { part_num: "3001", unmapped: false, name: null, image: null },
    "970c00": { part_num: "970c00", unmapped: true, name: null, image: null },
  };
  assert.deepEqual(toMatchParts(items, resolved), [{ part_num: "3001", color: "Red", qty: 1 }]);
  assert.equal(isUnmapped(items.find((i) => i.part_id === "970c00")!, resolved), true);
});

test("confirming a row clears its review flag", () => {
  const items = groupPieces([piece(0, "3001", "Red", true)]);
  assert.equal(confirmItem(items, items[0].id)[0].needsReview, false);
});

test("idea parts include unmapped parts but not unidentified pieces", () => {
  const items = groupPieces([piece(0, "3001", "Red"), piece(1, "970c00", "Black"), piece(2, null, null), piece(3, "3001", "Red")]);
  const parts = toIdeaParts(items);
  assert.equal(parts.length, 2);
  assert.deepEqual(parts.find((p) => p.part_num === "3001"), { part_num: "3001", name: "Part 3001", color: "Red", qty: 2 });
  assert.ok(parts.some((p) => p.part_num === "970c00"));
});

// ---------------------------------------------------------------- several photos
import { addPieces, nextOffset, scanOfPiece, withOffset } from "../lib/review.ts";
import type { Scan } from "../lib/types.ts";

const scanOf = (id: string, pieces: Piece[]): Scan => ({
  job_id: id, status: "done", total: pieces.length, image_url: `/img/${id}`, width: 100, height: 100, pieces, error: null,
});

test("piece numbers continue across photos and crop URLs are untouched", () => {
  const first = scanOf("a", [piece(0, "3001", "Red"), piece(1, "3002", "Red")]);
  assert.equal(nextOffset([]), 0);
  assert.equal(nextOffset([first]), 2);
  const second = withOffset(scanOf("b", [piece(0, "3001", "Red")]), nextOffset([first]));
  assert.deepEqual(second.pieces.map((p) => p.index), [2]);
  assert.equal(second.pieces[0].crop_url, "/crop/0");
});

test("a second photo's pieces join the list: same part and colour becomes one row, new parts add rows", () => {
  const first = withOffset(scanOf("a", [piece(0, "3001", "Red"), piece(1, "3002", "Blue")]), 0);
  let items = addPieces([], first.pieces);
  const second = withOffset(scanOf("b", [piece(0, "3001", "Red"), piece(1, "3003", "Black")]), nextOffset([first]));
  items = addPieces(items, second.pieces);
  assert.equal(items.length, 3);
  const red = items.find((i) => i.part_id === "3001")!;
  assert.equal(red.qty, 2);
  assert.deepEqual(red.pieceIdx.sort(), [0, 2]);
  assert.equal(pieceOwners(items).size, 4);
  assert.equal(toMatchParts(items, {}).reduce((n, p) => n + p.qty, 0), 4);
});

test("corrections and manual rows survive adding a photo; unidentified pieces are never merged", () => {
  let items = addPieces([], withOffset(scanOf("a", [piece(0, "3001", "Red", true, ["3002"]), piece(1, null, null)]), 0).pieces);
  items = changePart(items, items.find((i) => i.part_id === "3001")!.id, { part_id: "3002", name: "Part 3002" });
  items = addManual(items, { part_id: "9999", name: "Hand added" });
  items = addPieces(items, withOffset(scanOf("b", [piece(0, null, null), piece(1, "3002", "Red")]), 2).pieces);
  assert.equal(items.filter((i) => i.part_id === null).length, 2);
  assert.equal(items.find((i) => i.part_id === "3002")!.qty, 2);
  assert.ok(items.some((i) => i.part_id === "9999"));
});

test("a piece number maps back to the photo that holds it", () => {
  const a = withOffset(scanOf("a", [piece(0, "1", "R"), piece(1, "2", "R")]), 0);
  const b = withOffset(scanOf("b", [piece(0, "3", "R")]), nextOffset([a]));
  assert.equal(scanOfPiece([a, b], 1), 0);
  assert.equal(scanOfPiece([a, b], 2), 1);
  assert.equal(scanOfPiece([a, b], 99), 0);
});

// ---------------------------------------------------------------- merge / split corrections
import { localIndex, pieceUnderLine, replacePieces } from "../lib/review.ts";

test("a merge replaces two pieces by one and keeps what the user did to the other rows", () => {
  const before = [piece(0, "3001", "Red"), piece(1, "3001", "Red"), piece(2, "3002", "Blue")];
  let items = groupPieces(before);
  items = changePart(items, items.find((i) => i.part_id === "3002")!.id, { part_id: "9999", name: "Corrected" });
  const merged = withOffset(scanOf("m", [piece(0, "3003", "Red")]), 3).pieces;
  const next = replacePieces(items, before.slice(0, 2), merged);
  assert.equal(next.find((i) => i.part_id === "3001"), undefined);          // both red 3001 pieces are gone
  assert.deepEqual(next.find((i) => i.part_id === "3003")?.pieceIdx, [3]);   // the merged piece is a row
  assert.equal(next.find((i) => i.part_id === "9999")?.name, "Corrected");   // the corrected row is untouched
});

test("removing one of several pieces of a row lowers its qty", () => {
  const before = [piece(0, "3001", "Red"), piece(1, "3001", "Red"), piece(2, "3001", "Red")];
  const next = replacePieces(groupPieces(before), [before[1]], []);
  assert.equal(next[0].qty, 2);
  assert.deepEqual(next[0].pieceIdx, [0, 2]);
});

test("server piece numbers are local to a photo; the page's are global", () => {
  const second = withOffset(scanOf("s", [piece(0, "a", null), piece(1, "b", null)]), 7);
  assert.equal(localIndex(second, 8), 1);
});

test("a line drawn across a box picks that box", () => {
  const a = { ...piece(0, "a", null), box: [0, 0, 100, 100] as [number, number, number, number] };
  const b = { ...piece(1, "b", null), box: [200, 0, 100, 100] as [number, number, number, number] };
  const scan = scanOf("l", [a, b]);
  assert.equal(pieceUnderLine(scan, [210, 10, 290, 90])?.index, 1);
  assert.equal(pieceUnderLine(scan, [150, 300, 160, 310]), null);
});
