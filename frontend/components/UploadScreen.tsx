"use client";

import { useRef } from "react";

const RULES = [
  "Spread pieces apart on a plain white or light surface",
  "Good, even light (no harsh shadows)",
  "No pieces touching or overlapping",
];

export default function UploadScreen({
  onFile,
  error,
}: {
  onFile: (file: File) => void;
  error: string | null;
}) {
  const camera = useRef<HTMLInputElement>(null);
  const library = useRef<HTMLInputElement>(null);

  const handle = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow picking the same file again
    if (file) onFile(file);
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">What can I build?</h1>
        <p className="mt-1 text-stone-600">
          Photograph your LEGO pieces and we&apos;ll find sets and ideas you can build.
        </p>
      </div>

      <ul className="flex flex-col gap-2 rounded-2xl border border-stone-200 bg-white p-4">
        {RULES.map((r) => (
          <li key={r} className="flex gap-3 text-sm">
            <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-stud text-xs font-bold">
              ✓
            </span>
            {r}
          </li>
        ))}
      </ul>

      <figure className="overflow-hidden rounded-2xl border border-stone-200 bg-white">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/example.jpg" alt="Example: LEGO pieces laid out apart on a white background" className="w-full" />
        <figcaption className="px-4 py-2 text-xs text-stone-500">Example of a good photo</figcaption>
      </figure>

      {error && (
        <p role="alert" className="rounded-xl bg-red-50 p-3 text-sm text-red-800">
          {error}
        </p>
      )}

      <div className="flex flex-col gap-3">
        {/* capture="environment" opens the rear camera on phones */}
        <input ref={camera} type="file" accept="image/*" capture="environment" className="hidden" onChange={handle} />
        <input ref={library} type="file" accept="image/*" className="hidden" onChange={handle} />
        <button
          onClick={() => camera.current?.click()}
          className="h-14 rounded-2xl bg-brick text-lg font-semibold text-white active:opacity-80"
        >
          Take photo
        </button>
        <button
          onClick={() => library.current?.click()}
          className="h-12 rounded-2xl border border-stone-300 bg-white font-medium active:bg-stone-100"
        >
          Upload from gallery
        </button>
      </div>
    </div>
  );
}
