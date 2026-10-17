/**
 * Parsing a packed LDraw model with three.js' LDrawLoader and finding what each building step adds. No renderer in here, so it
 * runs in Node as well (see tests/ldrawModel.test.ts, which checks the step numbers against the backend's).
 */
import type { Group } from "three";
import { LDrawLoader } from "three/addons/loaders/LDrawLoader.js";
import { LDrawConditionalLineMaterial } from "three/addons/materials/LDrawConditionalLineMaterial.js";

export type PartRef = {
  group: Group;
  step: number; // 0-based: LDrawLoader's userData.buildingStep
  file: string; // e.g. "3001.dat"
  colour: string; // the colour code written in the model
};

export type ParsedModel = { loader: LDrawLoader; root: Group; numSteps: number; parts: PartRef[] };

/** Part-like LDraw file types (a printed torso is a "Shortcut"); sub-parts and primitives are not parts a builder adds. */
const PART_TYPE = /^(unofficial_)?(part|shortcut)$/i;

export function isPartGroup(g: Group): boolean {
  return PART_TYPE.test(String(g.userData.type ?? ""));
}

/**
 * Parts a builder adds, in order of appearance: groups of a part-like type with no part-like ancestor (a part that is built from
 * other parts, like a moved-part stub, counts once).
 */
export function collectParts(root: Group): PartRef[] {
  const out: PartRef[] = [];
  const visit = (g: Group) => {
    for (const child of g.children) {
      if (!(child as Group).isGroup) continue;
      const cg = child as Group;
      if (isPartGroup(cg)) {
        out.push({ group: cg, step: cg.userData.buildingStep ?? 0, file: String(cg.userData.fileName ?? cg.name), colour: String(cg.userData.colorCode ?? "16") });
      } else {
        visit(cg); // a submodel: look inside
      }
    }
  };
  visit(root);
  return out;
}

/** {step: number of parts} of a model, for comparing with the backend's own numbering. */
export function stepCounts(parts: PartRef[]): Record<number, number> {
  const counts: Record<number, number> = {};
  for (const p of parts) counts[p.step] = (counts[p.step] ?? 0) + 1;
  return counts;
}

export type ParseOptions = {
  /** LDConfig.ldr text: the colour table (code -> material). Without it every colour is the default. */
  ldconfig?: string;
  /** Drop the conditional (soft-edge) lines: faster on phones. */
  lite?: boolean;
  smoothNormals?: boolean;
};

export async function parseModel(text: string, opts: ParseOptions = {}): Promise<ParsedModel> {
  const loader = new LDrawLoader();
  loader.setConditionalLineMaterial(LDrawConditionalLineMaterial);
  loader.smoothNormals = opts.smoothNormals ?? !opts.lite;
  if (opts.ldconfig) {
    // the loader reads its colour table from a URL; a data: URI hands it the text we already have
    await loader.preloadMaterials(`data:text/plain;charset=utf-8,${encodeURIComponent(opts.ldconfig)}`);
  }
  return new Promise((resolve, reject) => {
    loader.parse(
      text,
      (root) => {
        if (opts.lite) {
          const doomed: { parent: { remove: (x: unknown) => void } }[] = [];
          root.traverse((o) => {
            if ((o as unknown as { isConditionalLine?: boolean }).isConditionalLine) doomed.push(o as never);
          });
          for (const o of doomed) o.parent.remove(o);
        }
        resolve({ loader, root, numSteps: root.userData.numBuildingSteps ?? 1, parts: collectParts(root) });
      },
      reject,
    );
  });
}
