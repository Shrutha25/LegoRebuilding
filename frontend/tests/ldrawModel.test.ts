import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { collectParts, parseModel, stepCounts } from "../lib/ldrawModel.ts";
import type { BuildMeta } from "../lib/build.ts";

// three's FileLoader reports progress with a browser ProgressEvent, which Node lacks
(globalThis as unknown as { ProgressEvent?: unknown }).ProgressEvent ??= class ProgressEvent extends Event {
  constructor(type: string, init: object = {}) {
    super(type);
    Object.assign(this, init);
  }
};

const demo = (name: string) => readFileSync(new URL(`../public/demo/${name}`, import.meta.url), "utf8");
const ldconfig = demo("LDConfig.ldr");

/** {step: parts} the backend says each step adds (the callout list), from the build metadata. */
const expected = (meta: BuildMeta) => Object.fromEntries(meta.steps.filter((s) => s.parts.length).map((s) => [s.index, s.parts.reduce((n, p) => n + p.qty, 0)]));

for (const [label, base] of [["AI build", "ai-build"], ["official set 30051-1", "official-30051"]] as const) {
  test(`${label}: the loader's building steps and parts match the backend's metadata`, async () => {
    const meta: BuildMeta = JSON.parse(demo(`${base}.json`));
    const model = await parseModel(demo(`${base}.mpd`), { ldconfig });
    assert.equal(model.numSteps, meta.step_count);
    assert.deepEqual(stepCounts(model.parts), expected(meta));
    assert.equal(model.parts.length, meta.part_count);
    // every part group sits under the root and is hidden-able on its own
    assert.ok(model.parts.every((p) => p.group.parent !== null));
  });
}

test("the lite model has no conditional lines and the same steps", async () => {
  const full = await parseModel(demo("official-30051.mpd"), { ldconfig });
  const lite = await parseModel(demo("official-30051.lite.mpd"), { ldconfig, lite: true });
  let conditional = 0;
  lite.root.traverse((o) => {
    if ((o as unknown as { isConditionalLine?: boolean }).isConditionalLine) conditional++;
  });
  assert.equal(conditional, 0);
  assert.deepEqual(stepCounts(lite.parts), stepCounts(full.parts));
});

test("colours come from LDConfig: a red part is red, and its colour code is kept", async () => {
  const model = await parseModel(demo("ai-build.mpd"), { ldconfig });
  const red = model.parts.find((p) => p.colour === "4")!;
  let found = false;
  red.group.traverse((o) => {
    const m = (o as unknown as { material?: { color?: { getHexString: () => string }; userData?: { code?: string } } }).material;
    if (m?.color && !Array.isArray(m) && o.type === "Mesh" && m.userData?.code === "4") {
      assert.equal(m.color.getHexString().toLowerCase(), "b40000");
      found = true;
    }
  });
  assert.ok(found);
  assert.equal(collectParts(model.root).length, 12);
});
