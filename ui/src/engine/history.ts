import { useEffect, useRef, useState } from "react";
import type { EngineState } from "./types";

/** One completed ply, kept for the graph and for seeking back to it. */
export interface PlyRecord {
  ply: number;
  /** Evaluation in centipawns, always from White's point of view. */
  evalWhite: number;
  move: string;
  fen: string;
  depth: number;
  seldepth: number;
  nodes: number;
  nps: number;
}

/** One search iteration, for the search-tree view. */
export interface IterRecord {
  depth: number;
  seldepth: number;
  nodes: number;
  ply: number;
}

/**
 * Accumulates the game as it happens.
 *
 * The engine streams state but keeps no history, so anything that wants to look
 * backwards -- the evaluation graph, seeking to an earlier position, the shape
 * of the search over time -- has to remember it here. Cleared whenever a new
 * game starts, and capped so a long session cannot grow without bound.
 */
export function useGameHistory(state: EngineState | null) {
  const [plies, setPlies] = useState<PlyRecord[]>([]);
  const [iters, setIters] = useState<IterRecord[]>([]);
  const lastPly = useRef(-1);
  const lastGame = useRef(-1);
  const lastIter = useRef("");

  useEffect(() => {
    if (!state) return;
    const g = state.game;

    // A new game (or a jump to a different position) restarts the record.
    if (g.gameIndex !== lastGame.current || g.ply < lastPly.current) {
      lastGame.current = g.gameIndex;
      lastPly.current = g.ply;
      setPlies([]);
      setIters([]);
      return;
    }

    if (g.ply !== lastPly.current) {
      lastPly.current = g.ply;
      // The frame is captured at the PV leaf, so convert with ITS side to move.
      const evalWhite = state.frame
        ? state.frame.eval * (state.frame.sideToMove === 0 ? 1 : -1)
        : state.search.scoreCp * (g.fen.split(" ")[1] === "b" ? -1 : 1);
      setPlies((p) =>
        [
          ...p,
          {
            ply: g.ply,
            evalWhite,
            move: g.lastMove,
            fen: g.fen,
            depth: state.search.depth,
            seldepth: state.search.seldepth,
            nodes: state.search.nodes,
            nps: state.search.nps,
          },
        ].slice(-400),
      );
    }

    // Search iterations: one per depth completed, keyed so repeats are skipped.
    const key = `${g.ply}:${state.search.depth}:${state.search.nodes}`;
    if (state.search.depth > 0 && key !== lastIter.current) {
      lastIter.current = key;
      setIters((it) =>
        [
          ...it,
          {
            depth: state.search.depth,
            seldepth: state.search.seldepth,
            nodes: state.search.nodes,
            ply: g.ply,
          },
        ].slice(-160),
      );
    }
  }, [state]);

  return { plies, iters };
}
