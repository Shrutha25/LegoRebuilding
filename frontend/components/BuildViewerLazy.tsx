"use client";

import dynamic from "next/dynamic";

/** The viewer (and three.js with it) is a separate chunk, downloaded only when the user opens a build. */
const BuildViewer = dynamic(() => import("./BuildViewer"), {
  ssr: false,
  loading: () => (
    <div role="status" className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-3 bg-zinc-100">
      <span className="h-9 w-9 animate-spin rounded-full border-4 border-zinc-300 border-t-ink" />
      <p className="font-medium">Opening the 3D viewer…</p>
    </div>
  ),
});

export default BuildViewer;
