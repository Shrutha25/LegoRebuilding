"use client";

import { useCallback, useEffect, useState } from "react";

const KEY = "bf-consider-colour-v1";

/**
 * "Consider color" (default on): on = pieces are counted by part AND colour, off = by part only and the build is coloured afterwards.
 * Kept in this browser's localStorage; when storage is blocked the choice just lasts until the page is closed.
 */
export function useConsiderColour(): [boolean, (v: boolean) => void] {
  const [value, setValue] = useState(true);
  useEffect(() => {
    try {
      if (window.localStorage.getItem(KEY) === "0") setValue(false);
    } catch {
      /* blocked: keep the default */
    }
  }, []);
  const set = useCallback((v: boolean) => {
    setValue(v);
    try {
      window.localStorage.setItem(KEY, v ? "1" : "0");
    } catch {
      /* blocked: still applies for this visit */
    }
  }, []);
  return [value, set];
}
