"use client";

import { useRef } from "react";

const RULES = [
  { title: "Plain surface", text: "Use a surface with no pattern, in a colour different from your pieces (white paper on a table works well)." },
  { title: "Spread them apart", text: "Leave a gap of a few centimetres between pieces. Nothing touching or overlapping." },
  { title: "Even light", text: "Daylight or a lamp from above. Avoid the flash, harsh shadows and glare." },
  { title: "Phone at an angle", text: "Hold the phone at about 45°, 30 to 40 cm away, so every piece is fully in the frame and in focus." },
  { title: "Same colour as the surface?", text: "A black piece on a black surface (or white on white) can be missed. After this photo, tap + Add photo and photograph those pieces again on a different coloured background." },
];

export default function UploadScreen({
  onFile,
  error,
  referenceId,
  onReference,
  uploadingReference = false,
  adding = false,
  onCancel,
}: {
  onFile: (file: File) => void;
  error: string | null;
  /** Set once the empty-background photo (step 1) has been taken. */
  referenceId: string | null;
  onReference: (file: File | null) => void;
  uploadingReference?: boolean;
  /** True when this is an extra photo for a list that already has pieces. */
  adding?: boolean;
  onCancel?: () => void;
}) {
  const emptyCamera = useRef<HTMLInputElement>(null);
  const emptyLibrary = useRef<HTMLInputElement>(null);
  const camera = useRef<HTMLInputElement>(null);
  const library = useRef<HTMLInputElement>(null);

  const handle = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow picking the same file again
    if (file) onFile(file);
  };

  const handleEmpty = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) onReference(file);
  };

  const ready = !!referenceId;

  return (
    <div className="grid flex-1 gap-6 lg:grid-cols-2 lg:grid-rows-[auto_minmax(0,1fr)_auto] lg:items-stretch lg:gap-x-8 lg:gap-y-3">
      <div className="px-2 pb-2 pt-4 text-center lg:col-span-2 lg:pb-0 lg:pt-0">
        <h1 className="title-display text-4xl sm:text-5xl lg:text-4xl">
          {adding ? (
            <>
              Add <span>photo</span>
            </>
          ) : (
            <>
              Brick <span>Finder</span>
            </>
          )}
        </h1>
        <p className="mt-3 text-xl font-semibold tracking-tight text-zinc-900 lg:mt-0.5 lg:text-base">{adding ? "Add more pieces to your list" : "Scan Your Pieces"}</p>
        <p className="mx-auto mt-2 max-w-sm text-sm text-muted lg:hidden">
          {adding
            ? "The pieces in this photo are added to your list."
            : "Photograph your LEGO pieces and we'll find the sets and ideas you can build."}
        </p>
      </div>

      {adding && (
        <div className="glass-sm px-5 py-4 text-sm lg:col-span-2">
          <p className="eyebrow">When to add a photo</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-zinc-700">
            <li>A piece was missed, or it is the same colour as the surface (black on black, white on white).</li>
            <li>Place only those pieces on a surface of a different colour, then photograph them.</li>
            <li>Do not include pieces already on your list: they would be counted twice.</li>
          </ul>
        </div>
      )}

      {/* placeholder picture */}
      <figure className="glass overflow-hidden p-0 lg:flex lg:flex-col lg:self-stretch">
        <div className="lg:relative lg:min-h-0 lg:flex-1">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/landing%20page%20lego.png"
            alt="Colourful LEGO bricks"
            className="aspect-square w-full object-cover object-left-top lg:absolute lg:inset-0 lg:aspect-auto lg:h-full"
          />
        </div>
      </figure>

      <div className="flex flex-col gap-6 lg:gap-4">
        {error && (
          <p role="alert" className="rounded-2xl border border-red-200 bg-red-50/90 p-3 text-sm text-red-800">
            {error}
          </p>
        )}

        <section className="glass flex flex-col gap-3 p-5 lg:flex-1 lg:justify-center lg:gap-2.5 lg:p-6" aria-label="Step 1: the empty background">
          <p className="eyebrow">Step 1</p>
          <p className="text-lg font-bold tracking-tight lg:text-base">Photo of the empty background</p>
          <p className="text-sm text-zinc-600 lg:text-[13px] lg:leading-snug">
            Photograph the surface with <span className="font-bold text-zinc-900">no pieces on it</span>. Then leave the phone where it is,
            or hold it as still as you can: the second photo must be taken from the same spot, at the same angle.
          </p>
          <input ref={emptyCamera} type="file" accept="image/*" capture="environment" className="hidden" onChange={handleEmpty} />
          <input ref={emptyLibrary} type="file" accept="image/*" className="hidden" onChange={handleEmpty} />
          {ready ? (
            <div className="flex items-center justify-between gap-3 rounded-2xl bg-ink px-4 py-3 text-sm font-semibold text-white shadow-[0_10px_24px_-12px_rgba(0,0,0,0.6)]">
              <span>✓ Empty background saved</span>
              <button onClick={() => emptyCamera.current?.click()} className="shrink-0 text-xs uppercase tracking-wider text-white/70 underline underline-offset-4">
                Retake
              </button>
            </div>
          ) : (
            <div className="flex gap-3">
              <button
                onClick={() => emptyCamera.current?.click()}
                disabled={uploadingReference}
                className="h-12 lg:h-11 flex-1 rounded-full btn-primary text-sm font-semibold disabled:opacity-50"
              >
                {uploadingReference ? "Uploading…" : "Take photo"}
              </button>
              <button
                onClick={() => emptyLibrary.current?.click()}
                disabled={uploadingReference}
                className="h-12 lg:h-11 flex-1 rounded-full btn-glass font-semibold disabled:opacity-50"
              >
                Upload
              </button>
            </div>
          )}
        </section>

        <section className={`glass flex flex-col gap-3 p-5 transition-opacity lg:flex-1 lg:justify-center lg:gap-2.5 lg:p-6 ${ready ? "" : "opacity-55"}`} aria-label="Step 2: the pieces">
          <p className="eyebrow">Step 2</p>
          <p className="text-lg font-bold tracking-tight lg:text-base">Photo with the pieces</p>
          <p className="text-sm text-zinc-600">
            {ready ? "Now place the pieces on the same background and take the photo without moving the phone." : "Take the empty-background photo first."}
          </p>
          {/* capture="environment" opens the rear camera on phones */}
          <input ref={camera} type="file" accept="image/*" capture="environment" className="hidden" onChange={handle} />
          <input ref={library} type="file" accept="image/*" className="hidden" onChange={handle} />
          <div className="flex gap-3">
            <button
              onClick={() => camera.current?.click()}
              disabled={!ready}
              className="h-12 lg:h-11 flex-1 rounded-full btn-primary text-sm font-semibold disabled:opacity-40"
            >
              Take photo
            </button>
            <button
              onClick={() => library.current?.click()}
              disabled={!ready}
              className="h-12 lg:h-11 flex-1 rounded-full btn-glass font-semibold disabled:opacity-40"
            >
              Upload
            </button>
          </div>
        </section>
      </div>

      {/* instructions sit below the picture and the two steps */}
      <section className="glass p-5 lg:col-span-2 lg:px-6 lg:py-2" aria-label="How to take the photos">
        <h2 className="eyebrow mb-4 text-center lg:mb-2">How to take the photos</h2>
        <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-[repeat(auto-fit,minmax(11rem,1fr))] lg:gap-5">
          {(adding ? RULES.slice(0, 4) : RULES).map((r, i) => (
            <li key={r.title} className="flex gap-3 text-sm lg:flex-col lg:gap-1.5 lg:text-[12.5px] lg:leading-snug">
              <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-ink text-xs font-bold text-white shadow-[0_4px_10px_-3px_rgba(0,0,0,0.5)]">
                {i + 1}
              </span>
              <span>
                <span className="font-bold text-zinc-900">{r.title}{/[?!.]$/.test(r.title) ? "" : "."}</span> <span className="text-zinc-600">{r.text}</span>
              </span>
            </li>
          ))}
        </ol>
      </section>

      {adding && onCancel && (
        <button onClick={onCancel} className="h-12 rounded-full lg:col-span-2 text-xs font-bold uppercase tracking-widest text-zinc-500 active:bg-zinc-100">
          Back to my list
        </button>
      )}
    </div>
  );
}
