"use client";

import { useCallback, useEffect, useState } from "react";
import ResultsScreen from "@/components/ResultsScreen";
import ReviewScreen from "@/components/ReviewScreen";
import ScanningScreen from "@/components/ScanningScreen";
import UploadScreen from "@/components/UploadScreen";
import { ApiError, getScan, resolveParts, startScan, uploadReference } from "@/lib/api";
import { addPieces, nextOffset, replacePieces, withOffset } from "@/lib/review";
import type { Item, Resolved, Scan } from "@/lib/types";

type Screen = "upload" | "scanning" | "review" | "results";

const POLL_MS = 1000;
const MAX_POLL_FAILURES = 5;

export default function Home() {
  const [screen, setScreen] = useState<Screen>("upload");
  const [scan, setScan] = useState<Scan | null>(null); // the photo being scanned right now
  const [scans, setScans] = useState<Scan[]>([]); // every photo of this session that has been scanned (piece numbers continue across photos)
  const [items, setItems] = useState<Item[]>([]);
  const [resolved, setResolved] = useState<Record<string, Resolved>>({});
  const [error, setError] = useState<string | null>(null);
  const [referenceId, setReferenceId] = useState<string | null>(null); // the empty-background photo for the next photo of pieces
  const [uploadingReference, setUploadingReference] = useState(false);

  const onReference = useCallback(async (file: File | null) => {
    if (!file) {
      setReferenceId(null);
      return;
    }
    setUploadingReference(true);
    try {
      setReferenceId(await uploadReference(file));
      setError(null);
    } catch (e) {
      setReferenceId(null);
      setError(e instanceof Error ? e.message : "Could not upload the empty-background photo.");
    } finally {
      setUploadingReference(false);
    }
  }, []);

  /** Throw everything away and start again. */
  const reset = useCallback((message: string | null = null) => {
    setScreen("upload");
    setScan(null);
    setScans([]);
    setItems([]);
    setResolved({});
    setReferenceId(null);
    setError(message);
  }, []);

  const onFile = useCallback(
    async (file: File) => {
      const keep = scans.length > 0; // an extra photo: keep what is on the list
      const fail = (message: string, expired = false) => {
        if (expired) setReferenceId(null); // the empty-background photo is gone from the server: it has to be taken again
        if (keep) {
          setScreen("upload");
          setScan(null);
          setError(message);
        } else {
          reset(message);
        }
      };
      setError(null);
      setScan(null);
      setScreen("scanning");
      let jobId: string;
      try {
        jobId = await startScan(file, referenceId);
      } catch (e) {
        fail(e instanceof Error ? e.message : "Upload failed.", e instanceof ApiError && e.status === 404);
        return;
      }

      let failures = 0;
      const poll = async () => {
        try {
          const s = await getScan(jobId);
          failures = 0;
          setScan(s);
          if (s.status === "done") {
            const shifted = withOffset(s, nextOffset(scans));
            setReferenceId(null); // each photo of pieces gets its own empty-background photo
            setScans((prev) => [...prev, shifted]);
            setItems((prev) => addPieces(prev, shifted.pieces));
            setScreen("review");
            return;
          }
          if (s.status === "error") {
            fail(s.error ?? "Scanning failed. Please try another photo.");
            return;
          }
        } catch (e) {
          if (e instanceof ApiError && e.status === 404) {
            fail("That scan expired. Please upload the photo again.");
            return;
          }
          if (++failures >= MAX_POLL_FAILURES) {
            fail("Lost connection to the server. Please try again.");
            return;
          }
        }
        setTimeout(poll, POLL_MS);
      };
      poll();
    },
    [reset, scans, referenceId],
  );

  // Look up the Rebrickable mapping for every part on the list (flags parts that can't be matched)
  // and for every alternative (gives them a Rebrickable image when Brickognize's is missing).
  const wanted = new Map<string, string | null>();
  for (const it of items) {
    if (it.part_id && !wanted.has(it.part_id)) wanted.set(it.part_id, it.color);
    for (const a of it.alternatives) if (!wanted.has(a.part_id)) wanted.set(a.part_id, it.color);
  }
  const missingKey = [...wanted]
    .filter(([id]) => !(id in resolved))
    .map(([id, color]) => `${id}${color ?? ""}`)
    .join("");
  useEffect(() => {
    if (!missingKey) return;
    const pairs = missingKey.split("").map((p) => p.split(""));
    const ids = pairs.map(([id]) => id);
    const colors = Object.fromEntries(pairs.map(([id, color]) => [id, color || null]));
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = async () => {
      try {
        const r = await resolveParts(ids, colors);
        if (!cancelled) setResolved((prev) => ({ ...prev, ...r }));
      } catch (e) {
        // Catalogue still loading: try again shortly. Other errors: carry on without it.
        if (!cancelled && e instanceof ApiError && e.status === 503) timer = setTimeout(run, 2000);
      }
    };
    run();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [missingKey]);

  return (
    <>
      <header className="px-4 pb-1 pt-5 text-center lg:pt-2 lg:pb-0">
        <span className="eyebrow tracking-[0.3em] lg:text-[10px]">Brick Finder</span>
      </header>
      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-3 pb-10 pt-3 sm:px-6 lg:max-w-none lg:px-[5px] lg:pb-[5px] lg:pt-1">
        <div className="flex flex-1 flex-col p-1 sm:p-2 lg:p-0">
        {screen === "upload" && (
          <UploadScreen
            onFile={onFile}
            error={error}
            adding={scans.length > 0}
            referenceId={referenceId}
            onReference={onReference}
            uploadingReference={uploadingReference}
            onCancel={() => {
              setError(null);
              setScreen("review");
            }}
          />
        )}
        {screen === "scanning" && <ScanningScreen scan={scan} />}
        {screen === "review" && scans.length > 0 && (
          <ReviewScreen
            scans={scans}
            items={items}
            setItems={(fn) => setItems(fn)}
            resolved={resolved}
            onStartOver={() => reset()}
            onAddPhoto={() => {
              setError(null);
              setScreen("upload");
            }}
            onFind={() => setScreen("results")}
            onDeletePhoto={(gone) => {
              setScans((prev) => prev.filter((s) => s.job_id !== gone.job_id));
              setItems((prev) => replacePieces(prev, gone.pieces, []));
            }}
            onPhotoEdited={(oldScan, newScan) => {
              const fresh = withOffset(newScan, nextOffset(scans)); // new pieces get new numbers: no clash with the other photos
              setScans((prev) => prev.map((s) => (s.job_id === oldScan.job_id ? fresh : s)));
              setItems((prev) => replacePieces(prev, oldScan.pieces, fresh.pieces));
            }}
          />
        )}
        {screen === "results" && <ResultsScreen items={items} resolved={resolved} onBack={() => setScreen("review")} />}
        </div>
      </main>
    </>
  );
}
