import { useEffect, useRef } from "react";
import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { Key } from "chessground/types";
import type { EngineState } from "../engine/types";

interface Props {
  state: EngineState | null;
  onMove?: (uci: string) => void;
}

/**
 * Lichess's chessground, driven straight from the engine's FEN and its own
 * legal-move list -- the UI never computes legality itself, so what you can
 * play is exactly what the engine accepts.
 */
export function Board({ state, onMove }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const api = useRef<Api | null>(null);
  const moveCb = useRef(onMove);
  moveCb.current = onMove;

  useEffect(() => {
    if (!ref.current) return;
    api.current = Chessground(ref.current, {
      coordinates: false,
      animation: { enabled: true, duration: 180 },
      movable: { free: false, color: undefined, dests: new Map() },
      drawable: { enabled: false },
      highlight: { lastMove: true, check: true },
    });
    return () => {
      api.current?.destroy();
      api.current = null;
    };
  }, []);

  useEffect(() => {
    const cg = api.current;
    if (!cg || !state) return;

    // Group the engine's legal moves into chessground's origin -> targets map.
    const dests = new Map<Key, Key[]>();
    for (const uci of state.legalMoves) {
      const from = uci.slice(0, 2) as Key;
      const to = uci.slice(2, 4) as Key;
      const list = dests.get(from);
      if (list) list.push(to);
      else dests.set(from, [to]);
    }

    const humanTurn =
      state.mode === "human" &&
      !state.game.over &&
      state.frame !== null &&
      state.engineColor !== sideToMoveIndex(state.game.fen);

    const last = state.game.lastMove;
    cg.set({
      fen: state.game.fen.split(" ")[0],
      turnColor: sideToMoveIndex(state.game.fen) === 0 ? "white" : "black",
      lastMove:
        last && last.length >= 4
          ? [last.slice(0, 2) as Key, last.slice(2, 4) as Key]
          : undefined,
      movable: {
        free: false,
        color: humanTurn
          ? sideToMoveIndex(state.game.fen) === 0
            ? "white"
            : "black"
          : undefined,
        dests: humanTurn ? dests : new Map(),
        events: {
          after: (from, to) => {
            // Promotions always send a queen; the engine rejects anything it
            // does not consider legal, so this stays honest.
            const uci = `${from}${to}`;
            const promo = state.legalMoves.find(
              (m) => m.startsWith(uci) && m.length === 5,
            );
            moveCb.current?.(promo ?? uci);
          },
        },
      },
    });
  }, [state]);

  return <div className="board-wrap" ref={ref} />;
}

function sideToMoveIndex(fen: string): number {
  return fen.split(" ")[1] === "b" ? 1 : 0;
}
