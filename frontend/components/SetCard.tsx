"use client";

import type { SetMatch } from "@/lib/types";
import Thumb from "./Thumb";

/**
 * An official set the user can build. Sets are listed only when they are 100% covered
 * (server setting MIN_COVERAGE), so the card says how many of the user's pieces it uses.
 */
export default function SetCard({ set, userPieces }: { set: SetMatch; userPieces: number }) {
  return (
    <li className="flex gap-3 rounded-2xl border border-stone-200 bg-white p-3">
      <Thumb srcs={[set.image]} label={set.name} className="h-24 w-24" />
      <div className="min-w-0 flex-1">
        <h3 className="line-clamp-2 font-semibold leading-snug">{set.name}</h3>
        <p className="text-xs text-stone-500">
          {set.set_num} · {set.year}
        </p>
        <p className="mt-2 text-sm">
          {set.complete ? (
            <>
              Uses <strong>{set.total_parts}</strong> of your {userPieces} pieces
            </>
          ) : (
            // Only reachable if the server's MIN_COVERAGE is lowered below 1.0
            <>
              You have <strong>{set.parts_you_have}</strong> of the set&apos;s {set.total_parts} pieces
            </>
          )}
        </p>
        <div className="-ml-2 mt-1 flex flex-wrap">
          <a href={set.instructions_url} target="_blank" rel="noreferrer" className="h-10 content-center px-2 text-sm font-medium text-brick underline-offset-2 hover:underline">
            Instructions
          </a>
          <a href={set.rebrickable_url} target="_blank" rel="noreferrer" className="h-10 content-center px-2 text-sm font-medium text-brick underline-offset-2 hover:underline">
            Rebrickable
          </a>
        </div>
      </div>
    </li>
  );
}
