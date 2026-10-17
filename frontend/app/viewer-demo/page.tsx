"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import BuildViewer from "@/components/BuildViewerLazy";
import type { BuildMeta } from "@/lib/build";

/**
 * /viewer-demo?build=ai|set&step=3  (or &done=1): the viewer on the bundled demo builds in public/demo, with no backend.
 * ai  = a Gemini-style build of the fixture inventory (validated, converted, packed by the backend)
 * set = the official LDraw model of set 30051-1 (X-wing Fighter, Mini) by Jude Parrill, CCAL 2.0
 */
function Demo() {
  const q = useSearchParams();
  const param = q.get("build") ?? "ai";
  const which = param === "set" ? "official-30051" : param === "ai" ? "ai-build" : param.replace(/[^a-z0-9-]/gi, ""); // any file in public/demo
  const [build, setBuild] = useState<BuildMeta | null>(null);
  useEffect(() => {
    // model URLs in the demo files are same-origin paths: make them absolute so the viewer does not point them at the API port
    fetch(`/demo/${which}.json`)
      .then((r) => r.json())
      .then((b: BuildMeta) =>
        setBuild({ ...b, model_url: `${location.origin}${b.model_url}`, lite_model_url: `${location.origin}${b.lite_model_url}` }),
      );
  }, [which]);
  if (!build) return <p className="p-6">Loading demo…</p>;
  return (
    <BuildViewer
      key={build.id}
      build={build}
      ldconfigUrl="/demo/LDConfig.ldr"
      initialStep={Number(q.get("step") ?? 0)}
      startFinished={q.get("done") === "1"}
      exposeScene={(s) => ((window as unknown as { __scene?: unknown }).__scene = s)}
      onClose={() => window.history.back()}
    />
  );
}

export default function Page() {
  return (
    <Suspense fallback={<p className="p-6">Loading demo…</p>}>
      <Demo />
    </Suspense>
  );
}
