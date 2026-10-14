export type Alt = {
  part_id: string;
  name: string;
  score: number;
  image?: string | null;
};

export type Piece = {
  index: number;
  box: [number, number, number, number]; // x, y, w, h in image pixels
  crop_url: string;
  part_id: string | null;
  name: string | null;
  color: string | null;
  score: number | null;
  image?: string | null;
  alternatives: Alt[];
  needs_review: boolean;
  error: string | null;
};

export type Scan = {
  job_id: string;
  status: "queued" | "detecting" | "recognizing" | "done" | "error";
  total: number;
  image_url: string;
  width: number;
  height: number;
  pieces: Piece[];
  error: string | null;
};

/** One row on the review screen: identical parts grouped together. */
export type Item = {
  id: string;
  part_id: string | null; // null = piece Brickognize could not identify
  name: string;
  color: string | null;
  qty: number;
  pieceIdx: number[]; // detected pieces (boxes) belonging to this row
  needsReview: boolean;
  alternatives: Alt[];
  cropUrl: string | null;
  image: string | null;
};

export type Resolved = {
  part_num: string;
  unmapped: boolean;
  name: string | null;
  image: string | null;
};

export type SearchResult = { part_num: string; name: string; image: string | null };

export type MissingPart = { part: string; color: string | null; qty: number; image: string | null };

export type SetMatch = {
  set_num: string;
  name: string;
  year: string;
  image: string;
  total_parts: number;
  parts_you_have: number;
  coverage: number;
  missing_count: number;
  missing: MissingPart[];
  rebrickable_url: string;
  instructions_url: string;
};

export type MatchPart = { part_num: string; color: string | null; qty: number };
