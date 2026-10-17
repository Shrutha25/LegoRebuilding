"use client";

import type { InstructionInfo } from "@/lib/instructions";
import type { MatchPart, SetMatch } from "@/lib/types";
import InstructionsAction from "./InstructionsAction";
import Thumb from "./Thumb";

/**
 * An official set the user can build. Sets are listed only when they are 100% covered
 * (server setting MIN_COVERAGE), so the card says how many of the user's pieces it uses.
 * The instructions part depends on what exists for the set (see InstructionsAction).
 */
export default function SetCard({ set, userPieces, parts = [], info }: { set: SetMatch; userPieces: number; parts?: MatchPart[]; info?: InstructionInfo }) {
  return (
    <li className="glass-sm p-3">
      <div className="flex gap-3">
        <Thumb srcs={[set.image]} label={set.name} className="h-24 w-24" />
        <div className="min-w-0 flex-1">
          <h3 className="line-clamp-2 font-semibold leading-snug">{set.name}</h3>
          <p className="text-xs text-zinc-500">
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
            <a href={set.rebrickable_url} target="_blank" rel="noreferrer" className="h-8 content-center px-2 text-[11px] font-bold uppercase tracking-wide text-zinc-500 underline-offset-4 hover:underline">
              Rebrickable
            </a>
          </div>
        </div>
      </div>

      <InstructionsAction setNum={set.set_num} name={set.name} rebrickableUrl={set.rebrickable_url} info={info} parts={parts} />
    </li>
  );
}
