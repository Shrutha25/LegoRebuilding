"use client";

import { absUrl } from "@/lib/api";
import type { Item, Scan } from "@/lib/types";

const GREEN = "#16a34a";
const AMBER = "#f59e0b";

/** The photo with numbered boxes: green = confident, amber = needs review. Tap a box to select its row. */
export default function PhotoBoxes({
  scan,
  owners,
  selectedPieces,
  onSelectPiece,
}: {
  scan: Scan;
  owners: Map<number, Item>;
  selectedPieces: Set<number>;
  onSelectPiece: (pieceIndex: number) => void;
}) {
  const { width, height } = scan;
  const label = Math.max(14, Math.round(Math.max(width, height) * 0.028));
  // Draw selected boxes last so they sit on top of neighbours.
  const visible = scan.pieces
    .filter((p) => owners.has(p.index))
    .sort((a, b) => Number(selectedPieces.has(a.index)) - Number(selectedPieces.has(b.index)));

  return (
    <div
      className="relative mx-auto max-w-full overflow-hidden rounded-2xl border border-stone-200 bg-white [--ph:38vh] md:[--ph:80vh]"
      style={{ width: `min(100%, calc(var(--ph) * ${width / height}))`, aspectRatio: `${width} / ${height}` }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={absUrl(scan.image_url)} alt="Your photo with detected pieces outlined" className="absolute inset-0 h-full w-full" />
      <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 h-full w-full" role="group" aria-label="Detected pieces">
        {visible.map((p) => {
          const [x, y, w, h] = p.box;
          const review = owners.get(p.index)!.needsReview;
          const selected = selectedPieces.has(p.index);
          const color = review ? AMBER : GREEN;
          const tag = String(p.index + 1);
          const tw = label * (0.7 * tag.length + 0.8);
          return (
            <g key={p.index} onClick={() => onSelectPiece(p.index)} className="cursor-pointer">
              <rect
                x={x} y={y} width={w} height={h}
                fill={selected ? `${color}55` : "transparent"}
                stroke={color}
                strokeWidth={selected ? 6 : 3}
                vectorEffect="non-scaling-stroke"
              />
              <rect x={x} y={y} width={tw} height={label * 1.3} fill={color} />
              <text x={x + tw / 2} y={y + label * 1.02} fontSize={label} fontWeight={700} textAnchor="middle" fill="#fff">
                {tag}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
