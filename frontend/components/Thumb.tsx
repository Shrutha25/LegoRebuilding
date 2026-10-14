"use client";

import { useState } from "react";

/**
 * A small image that tries each source in order (some thumbnails 403/404 upstream) and
 * falls back to a neutral brick placeholder when none loads.
 */
export default function Thumb({
  srcs,
  label,
  className = "h-14 w-14",
}: {
  srcs: (string | null | undefined)[];
  label: string;
  className?: string;
}) {
  const [bad, setBad] = useState<string[]>([]);
  const src = srcs.find((s): s is string => !!s && !bad.includes(s));

  return (
    <span className={`flex shrink-0 items-center justify-center overflow-hidden rounded-lg bg-stone-100 ${className}`} title={label}>
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt="" className="h-full w-full object-contain" loading="lazy" onError={() => setBad((b) => [...b, src])} />
      ) : (
        <svg viewBox="0 0 24 24" className="h-1/2 w-1/2 text-stone-300" fill="currentColor" role="img" aria-label="No image available">
          <rect x="3" y="10" width="18" height="9" rx="1.5" />
          <circle cx="8" cy="7.5" r="2" />
          <circle cx="16" cy="7.5" r="2" />
        </svg>
      )}
    </span>
  );
}
