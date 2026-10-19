import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFrame } from "../lib/sse.ts";

test("a server-sent-events frame becomes an event", () => {
  const e = parseFrame('event: design\ndata: {"build": {"id": "ai-1", "title": "Pond Duck"}, "t_ms": 5200}');
  assert.equal(e?.event, "design");
  assert.equal((e as { data: { t_ms: number } }).data.t_ms, 5200);
  assert.equal(parseFrame("event: done\ndata: {broken"), null);
  assert.equal(parseFrame(": comment only"), null);
  assert.equal(parseFrame('event: note\ndata: {"kind": "x", "message": "m"}')?.event, "note");
});
