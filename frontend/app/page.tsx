"use client";

import { useCallback, useEffect, useState } from "react";
import ResultsScreen from "@/components/ResultsScreen";
import ReviewScreen from "@/components/ReviewScreen";
import ScanningScreen from "@/components/ScanningScreen";
import UploadScreen from "@/components/UploadScreen";
import { ApiError, getScan, resolveParts, startScan } from "@/lib/api";
import { groupPieces } from "@/lib/review";
import type { Item, Resolved, Scan } from "@/lib/types";

type Screen = "upload" | "scanning" | "review" | "results";

const POLL_MS = 1000;
const MAX_POLL_FAILURES = 5;

export default function Home() {
  const [screen, setScreen] = useState<Screen>("upload");
  const [scan, setScan] = useState<Scan | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [resolved, setResolved] = useState<Record<string, Resolved>>({});
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback((message: string | null = null) => {
    setScreen("upload");
    setScan(null);
    setItems([]);
    setResolved({});
    setError(message);
  }, []);

  const onFile = useCallback(
    async (file: File) => {
      reset();
      setScreen("scanning");
      let jobId: string;
      try {
        jobId = await startScan(file);
      } catch (e) {
        reset(e instanceof Error ? e.message : "Upload failed.");
        return;
      }

      let failures = 0;
      const poll = async () => {
        try {
          const s = await getScan(jobId);
          failures = 0;
          setScan(s);
          if (s.status === "done") {
            setItems(groupPieces(s.pieces));
            setScreen("review");
            return;
          }
          if (s.status === "error") {
            reset(s.error ?? "Scanning failed. Please try another photo.");
            return;
          }
        } catch (e) {
          if (e instanceof ApiError && e.status === 404) {
            reset("That scan expired. Please upload the photo again.");
            return;
          }
          if (++failures >= MAX_POLL_FAILURES) {
            reset("Lost connection to the server. Please try again.");
            return;
          }
        }
        setTimeout(poll, POLL_MS);
      };
      poll();
    },
    [reset],
  );

  // Look up the Rebrickable mapping for every part on the list (flags parts that can't be matched).
  const missingKey = [...new Set(items.flatMap((i) => (i.part_id ? [i.part_id] : [])))]
    .filter((id) => !(id in resolved))
    .join("|");
  useEffect(() => {
    if (!missingKey) return;
    const ids = missingKey.split("|");
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = async () => {
      try {
        const r = await resolveParts(ids);
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
      <header className="bg-stud">
        <div className="mx-auto flex h-12 max-w-3xl items-center px-4 font-bold tracking-tight">🧱 Brick Finder</div>
      </header>
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-5">
        {screen === "upload" && <UploadScreen onFile={onFile} error={error} />}
        {screen === "scanning" && <ScanningScreen scan={scan} />}
        {screen === "review" && scan && (
          <ReviewScreen
            scan={scan}
            items={items}
            setItems={(fn) => setItems(fn)}
            resolved={resolved}
            onRetake={() => reset()}
            onFind={() => setScreen("results")}
          />
        )}
        {screen === "results" && <ResultsScreen items={items} resolved={resolved} onBack={() => setScreen("review")} />}
      </main>
    </>
  );
}
