"use client";

import { useRef, useState } from "react";
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
  mode = null,
  picked = [],
  onDrawLine,
}: {
  scan: Scan;
  owners: Map<number, Item>;
  selectedPieces: Set<number>;
  onSelectPiece: (pieceIndex: number) => void;
  /** "merge": tapping boxes picks them for a merge. "split": drag a line across a box to cut it there. */
  mode?: "merge" | "split" | null;
  picked?: number[];
  onDrawLine?: (line: [number, number, number, number]) => void;
}) {
  const { width, height } = scan;
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<[number, number, number, number] | null>(null);
  const label = Math.max(14, Math.round(Math.max(width, height) * 0.028));
  // Draw selected boxes last so they sit on top of neighbours.
  const visible = scan.pieces
    .filter((p) => owners.has(p.index))
    .sort((a, b) => Number(selectedPieces.has(a.index)) - Number(selectedPieces.has(b.index)));

  return (
    <div
      className="relative mx-auto max-w-full overflow-hidden glass-sm [--ph:38vh] md:[--ph:80vh]"
      style={{ width: `min(100%, calc(var(--ph) * ${width / height}))`, aspectRatio: `${width} / ${height}` }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={absUrl(scan.image_url)} alt="Your photo with detected pieces outlined" className="absolute inset-0 h-full w-full" />
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className={`absolute inset-0 h-full w-full ${mode === "split" ? "cursor-crosshair touch-none" : ""}`}
        role="group"
        aria-label="Detected pieces"
        onPointerDown={(e) => {
          if (mode !== "split") return;
          const pt = toImage(e);
          if (pt) setDrag([pt[0], pt[1], pt[0], pt[1]]);
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const pt = toImage(e);
          if (drag && pt) setDrag([drag[0], drag[1], pt[0], pt[1]]);
        }}
        onPointerUp={() => {
          if (drag && Math.hypot(drag[2] - drag[0], drag[3] - drag[1]) > label) onDrawLine?.(drag);
          setDrag(null);
        }}
      >
        {visible.map((p) => {
          const [x, y, w, h] = p.box;
          const review = owners.get(p.index)!.needsReview;
          const selected = selectedPieces.has(p.index) || picked.includes(p.index);
          const color = review ? AMBER : GREEN;
          const tag = String(p.index + 1);
          const tw = label * (0.7 * tag.length + 0.8);
          return (
            <g key={p.index} onClick={() => mode !== "split" && onSelectPiece(p.index)} className="cursor-pointer">
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
        {drag && <line x1={drag[0]} y1={drag[1]} x2={drag[2]} y2={drag[3]} stroke="#dc2626" strokeWidth={5} strokeLinecap="round" vectorEffect="non-scaling-stroke" />}
      </svg>
    </div>
  );

  /** Pointer position in the photo's own pixels. */
  function toImage(e: React.PointerEvent<SVGSVGElement>): [number, number] | null {
    const r = svgRef.current?.getBoundingClientRect();
    if (!r || r.width === 0) return null;
    return [((e.clientX - r.left) / r.width) * width, ((e.clientY - r.top) / r.height) * height];
  }
}
