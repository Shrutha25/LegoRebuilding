"use client";

import type { Scan } from "@/lib/types";

export default function ScanningScreen({ scan }: { scan: Scan | null }) {
  let label = "Uploading photo…";
  if (scan?.status === "detecting" || scan?.status === "queued") label = "Finding your pieces…";
  if (scan?.status === "recognizing") label = `Identifying ${scan.total} piece${scan.total === 1 ? "" : "s"}…`;

  return (
    <div role="status" aria-live="polite" className="flex flex-1 flex-col items-center justify-center gap-6 py-24 text-center">
      <div className="grid grid-cols-3 gap-2" aria-hidden>
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <span
            key={i}
            className="h-6 w-9 animate-pulse rounded-md bg-ink shadow-[0_8px_18px_-8px_rgba(0,0,0,0.6)]"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
      <p className="title-display text-2xl">{label}</p>
      <p className="max-w-xs text-sm text-zinc-500">This usually takes a few seconds. Each piece is identified separately.</p>
    </div>
  );
}
