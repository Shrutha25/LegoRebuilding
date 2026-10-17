import assert from "node:assert/strict";
import { test } from "node:test";
import { infoFor, sortByInstructions, type InstructionInfo } from "../lib/instructions.ts";

const info = (state: 1 | 2 | 3 | null): InstructionInfo => ({ state, links: [], pending: state === null });
const set = (set_num: string, total_parts: number, missing_count = 0) => ({ set_num, total_parts, missing_count });

test("sets are ordered by instructions state, then completeness, then part count", () => {
  const states: Record<string, InstructionInfo> = {
    a: info(1), b1: info(2), b2: info(2), b3: info(2), u: info(null), c: info(3),
  };
  const sets = [set("c", 500), set("b2", 100), set("a", 100, 5), set("b1", 300), set("u", 900), set("b3", 100, 10)];
  const order = sortByInstructions(sets, (s) => states[s.set_num]).map((s) => s.set_num);
  assert.deepEqual(order, ["a", "b1", "b2", "b3", "u", "c"]); // unknown is listed with state 3
});

test("a finished lookup replaces the inline info; a pending one does not", () => {
  const inline = info(null);
  assert.equal(infoFor(inline, { x: info(2) }, "x")?.state, 2);
  assert.equal(infoFor(info(3), { x: info(null) }, "x")?.state, 3);
  assert.equal(infoFor(undefined, {}, "x"), undefined);
});
