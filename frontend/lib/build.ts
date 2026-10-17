import type { InstructionInfo } from "./instructions";
/** Types for the 3D building-instructions viewer (what /api/builds/* and /api/sets/{set}/build return). */

export type BuildPartUse = {
  ld_part: string; // LDraw part number, e.g. "3001"
  name: string;
  colour_code: string; // LDraw colour code as text ("4") or a direct colour ("0x2FF0000")
  colour_name: string;
  hex: string; // "#B40000"
  qty: number;
};

export type BuildStep = { index: number; parts: BuildPartUse[] }; // index is 0-based, like LDrawLoader's buildingStep

export type Attribution = { authors: string[]; licence: string[]; source_url: string; text: string };

export type BuildMeta = {
  id: string;
  kind: "ai" | "set";
  title: string;
  description: string;
  steps: BuildStep[];
  step_count: number;
  part_count: number;
  model_url: string;
  lite_model_url: string;
  attribution: Attribution | null;
  repairs: string[];
  warnings: string[];
  repair_summary?: { dropped: number; swapped: number; nudged: number; steps_regenerated: boolean; retried: boolean; asked_for: number };
  skipped_parts?: { part: string; reason: string }[];
  steps_generated?: boolean;
  set_num?: string;
  /** Present on builds made from a recipe (the code designed the shape; the name fits it). */
  recipe?: RecipeInfo;
  thumb_url?: string;
  /** True when the build was made with "Consider color" off: pieces were counted by part only and coloured afterwards. */
  mixed_colours?: boolean;
  look_check?: "yes" | "partly" | "no";
};

export type RecipeInfo = {
  id: string;
  category: string;
  params: Record<string, number>;
  colours: Record<string, string>; // colour role -> colour name
  score: number;
  named_by_ai: boolean;
  /** The concept the build was made for: its silhouette, bounding box (studs) and the features it asked for. */
  concept?: { silhouette: string; bbox: [number, number, number] | null; features: string[]; source: string } | null;
};

export type UnlockHint = {
  recipe: string;
  name: string;
  options: { part: string; qty: number }[][]; // alternatives: "1 more Brick 2 x 4", or "2 more Brick 2 x 2"
  cost: number;
};

export type RecipeBuilds = {
  builds: BuildMeta[];
  unlock: UnlockHint[];
  ai_used: boolean;
  ai_note: string;
  candidate_count: number;
  freeform: boolean;
  timings_ms: Record<string, number>;
};

export type NearbySet = {
  set_num: string;
  name: string;
  year: string | null;
  image: string | null;
  total_parts: number;
  missing_count: number;
  colour_match: "exact" | "any";
  any_colour: { have: number; need: number };
  exact_colour: { have: number; need: number };
  missing: { part: string; name: string; colour: string | null; qty: number; image: string | null }[];
  rebrickable_url: string;
  instructions_url: string;
  instructions?: InstructionInfo;
};

export type NearbySets = { can_build: NearbySet[]; almost: NearbySet[]; timings_ms: Record<string, number> };

export type SetInventoryRow = {
  part: string;
  name: string;
  colour: string;
  need: number;
  have: number;
  missing: number;
  image: string | null;
};

export type SetBuild = {
  set_num: string;
  name: string | null;
  available: boolean;
  reason: "none" | "licence" | "error" | null;
  build: BuildMeta | null;
  instructions_url: string;
  instructions?: InstructionInfo;
  inventory: SetInventoryRow[];
  need_total: number;
  have_total: number;
};

export type BuildPartIn = { part_num: string; color: string | null; qty: number; name?: string | null };

/** Phones and tablets get the lighter model (no conditional edge lines) and fewer effects. */
export function isTouchDevice(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia?.("(pointer: coarse)").matches || navigator.maxTouchPoints > 0 || window.innerWidth < 768;
}

export const DROP_ANIMATION = process.env.NEXT_PUBLIC_BUILD_ANIMATION === "1";
