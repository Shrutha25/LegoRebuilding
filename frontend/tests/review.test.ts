import assert from "node:assert/strict";
import { test } from "node:test";
import {
  addManual, changePart, confirmItem, groupPieces, isUnmapped, pieceOwners, removeItem, setQty,
  toMatchParts,
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
