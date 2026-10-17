/** What a set card can offer: 1 = our 3D steps, 2 = official LEGO instructions, 3 = neither. null while the lookup is still running. */
export type InstructionLink = { url: string; description: string };
export type InstructionInfo = { state: 1 | 2 | 3 | null; links: InstructionLink[]; pending: boolean };

/** The info the server sent with the match, replaced by a later lookup when one has arrived. */
export const infoFor = (
  inline: InstructionInfo | undefined,
  live: Record<string, InstructionInfo>,
  setNum: string,
): InstructionInfo | undefined => (live[setNum] && !live[setNum].pending ? live[setNum] : inline ?? live[setNum]);

type Rankable = { set_num: string; total_parts: number; missing_count: number };

/** Same order as the server: state 1, then 2, then 3 (unknown counts as 3); then completeness; then part count. */
export const rankKey = (s: Rankable, info?: InstructionInfo): [number, number, number, string] => {
  const total = Math.max(1, s.total_parts || 1);
  return [info?.state ?? 3, -(1 - (s.missing_count || 0) / total), -total, s.set_num];
};

export const sortByInstructions = <T extends Rankable>(
  sets: T[],
  infoOf: (s: T) => InstructionInfo | undefined,
): T[] =>
  [...sets].sort((a, b) => {
    const ka = rankKey(a, infoOf(a));
    const kb = rankKey(b, infoOf(b));
    for (let i = 0; i < 3; i++) if (ka[i] !== kb[i]) return (ka[i] as number) - (kb[i] as number);
    return ka[3] < kb[3] ? -1 : ka[3] > kb[3] ? 1 : 0;
  });
