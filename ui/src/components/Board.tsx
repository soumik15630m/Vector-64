import { useEffect, useRef } from "react";
import { Chessground } from "chessground";
import type { Api } from "chessground/api";
import type { DrawBrushes } from "chessground/draw";
import type { Key } from "chessground/types";
import type { EngineState } from "../engine/types";

interface Props {
  state: EngineState | null;
  /** A candidate move to preview, drawn instead of the engine's intent. */
  highlight?: string | null;
  onMove?: (uci: string) => void;
}

/**
 * Lichess's chessground, driven straight from the engine's FEN and its own
 * legal-move list -- the UI never computes legality itself, so what you can
 * play is exactly what the engine accepts.
 */
/**
 * The engine's intended move. lineWidth is chessground's own unit (a square is
 * 10), so 5 is half a square -- a pointer, not a bar across the board, and the
 * arrowhead scales with it so a thinner line also gives a sharper head.
 *
 * Amber rather than the interface accent, and that is the whole point: the
 * board is blue, the last move is cyan and so is the eval bar, so a cyan arrow
 * would be one more blue thing on a blue board. --pos is the theme's other
 * principal colour -- the neuron field already reads it as "positive" -- so
 * the board ends up saying cyan for what has happened and amber for what the
 * engine intends next.
 *
 * Near-opaque on purpose: chessground's stock arrows are translucent navy,
 * which over a light square leaves neither the arrow nor the square legible.
 * No mid-tone hue clears 3:1 against both square colours, so what actually
 * delimits the arrow is the ink halo theme.css puts under the shape layer.
 */
const INTENT_BRUSHES: DrawBrushes = {
  intent: { key: "vi", color: "#ffb545", opacity: 0.92, lineWidth: 5 },
  // chessground requires its four standard brushes. Nothing draws with them
  // here, but leaving the stock colours in would let a default arrow appear in
  // a palette this UI never uses.
  green: { key: "vg", color: "#4ade80", opacity: 0.9, lineWidth: 5 },
  red: { key: "vr", color: "#f87171", opacity: 0.9, lineWidth: 5 },
  blue: { key: "vb", color: "#4a90ff", opacity: 0.9, lineWidth: 5 },
  yellow: { key: "vy", color: "#ffb545", opacity: 0.9, lineWidth: 5 },
};

export function Board({ state, highlight, onMove }: Props) {
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
      drawable: {
        enabled: false,
        autoShapes: [],
        // Chessground's stock arrows are thick and navy -- they read as a
        // separate application drawn over the board. This one is the interface
        // accent, thin enough to point without hiding the squares it crosses.
        brushes: INTENT_BRUSHES,
      },
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
    // Show what the engine intends to play next as an arrow, so the board and
    // the network view agree on the decision being made.
    const intend = highlight ?? state.search.pv[0];
    const shapes =
      intend && intend.length >= 4 && !state.game.over
        ? [
            {
              orig: intend.slice(0, 2) as Key,
              dest: intend.slice(2, 4) as Key,
              brush: "intent",
            },
          ]
        : [];
    cg.set({
      fen: state.game.fen.split(" ")[0],
      turnColor: sideToMoveIndex(state.game.fen) === 0 ? "white" : "black",
      lastMove:
        last && last.length >= 4
          ? [last.slice(0, 2) as Key, last.slice(2, 4) as Key]
          : undefined,
      drawable: { enabled: false, autoShapes: shapes, brushes: INTENT_BRUSHES },
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
  }, [state, highlight]);

  return <div className="board-wrap" ref={ref} />;
}

function sideToMoveIndex(fen: string): number {
  return fen.split(" ")[1] === "b" ? 1 : 0;
}
