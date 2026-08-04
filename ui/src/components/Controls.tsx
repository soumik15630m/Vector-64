import { useState } from "react";
import { Chess } from "chess.js";
import type { ControlCommand, EngineState, Mode } from "../engine/types";

/**
 * Turn pasted PGN or a FEN into something the engine accepts.
 *
 * chess.js only parses the notation into a move list; every move is still
 * validated by the engine when the position is applied, so the UI cannot talk
 * the engine into an illegal position.
 */
function parsePosition(
  text: string,
): { fen: string; moves: string[] } | { error: string } {
  const t = text.trim();
  if (!t) return { error: "empty" };

  // A bare FEN: 6 space-separated fields with a side-to-move letter.
  if (/^[1-8pnbrqkPNBRQK/]+\s+[wb]\s/.test(t)) {
    const chess = new Chess();
    try {
      chess.load(t);
    } catch {
      return { error: "invalid FEN" };
    }
    return { fen: t, moves: [] };
  }

  const chess = new Chess();
  try {
    chess.loadPgn(t);
  } catch {
    return { error: "could not parse PGN" };
  }
  const history = chess.history({ verbose: true });
  if (history.length === 0) return { error: "no moves in that PGN" };
  const start = history[0].before;
  return {
    fen: start,
    moves: history.map((m) => `${m.from}${m.to}${m.promotion ?? ""}`),
  };
}

interface Props {
  s: EngineState;
  send: (c: ControlCommand) => void;
}

const MODES: { id: Mode; label: string }[] = [
  { id: "selfplay", label: "self-play" },
  { id: "analysis", label: "analysis" },
  { id: "human", label: "play" },
];

export function Controls({ s, send }: Props) {
  const [fen, setFen] = useState("");
  const [posError, setPosError] = useState<string | null>(null);
  const [delay, setDelay] = useState(300);
  const [nodes, setNodes] = useState(20000);
  // Never offer more threads than the machine has: past the core count lazy
  // SMP gets slower, not faster. The engine clamps too, so a stale UI value
  // cannot push it out of range.
  const maxThreads = Math.max(1, s.hardwareThreads || 1);
  const [threads, setThreads] = useState(Math.min(s.threads, maxThreads));

  return (
    <div className="panel">
      <h3>Control</h3>

      <div className="seg" style={{ marginBottom: 9 }}>
        {MODES.map((m) => (
          <button
            key={m.id}
            className={s.mode === m.id ? "on" : ""}
            onClick={() => send({ cmd: "mode", value: m.id })}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="controls">
        <button
          className={`btn${s.paused ? " on" : ""}`}
          onClick={() => send({ cmd: "pause", value: !s.paused })}
        >
          {s.paused ? "resume" : "pause"}
        </button>
        <button
          className="btn"
          disabled={!s.paused}
          onClick={() => send({ cmd: "step" })}
        >
          step
        </button>
        <button className="btn" onClick={() => send({ cmd: "newgame" })}>
          new game
        </button>
        {s.mode === "human" && (
          <button
            className="btn"
            onClick={() =>
              send({ cmd: "enginecolor", value: s.engineColor ? 0 : 1 })
            }
          >
            engine {s.engineColor ? "black" : "white"}
          </button>
        )}
      </div>

      <div className="field-row">
        <div>
          <label className="lbl">
            delay<b>{delay}ms</b>
          </label>
          <input
            type="range"
            min={0}
            max={2000}
            step={50}
            value={delay}
            onChange={(e) => {
              const v = Number(e.currentTarget.value);
              setDelay(v);
              send({ cmd: "delay", value: v });
            }}
          />
        </div>
        <div>
          <label className="lbl">
            nodes<b>{nodes >= 1000 ? `${Math.round(nodes / 1000)}k` : nodes}</b>
          </label>
          <input
            type="range"
            min={1000}
            max={400000}
            step={1000}
            value={nodes}
            onChange={(e) => {
              const v = Number(e.currentTarget.value);
              setNodes(v);
              send({ cmd: "nodes", value: v });
            }}
          />
        </div>
      </div>

      <div style={{ marginTop: 8 }}>
        <label className="lbl">
          threads
          <b>
            {Math.min(threads, maxThreads)} / {maxThreads}
          </b>
        </label>
        <input
          type="range"
          min={1}
          max={maxThreads}
          step={1}
          disabled={maxThreads === 1}
          value={Math.min(threads, maxThreads)}
          onChange={(e) => {
            const v = Math.min(
              Math.max(1, Number(e.currentTarget.value) || 1),
              maxThreads,
            );
            setThreads(v);
            send({ cmd: "threads", value: v });
          }}
        />
      </div>

      {s.mode === "analysis" && (
        <div style={{ marginTop: 9 }}>
          <label className="lbl">
            position
            {posError && <b style={{ color: "var(--bad)" }}>{posError}</b>}
          </label>
          <textarea
            className="pos-input"
            placeholder="paste a FEN or a PGN, then press Enter"
            value={fen}
            onChange={(e) => {
              setFen(e.currentTarget.value);
              setPosError(null);
            }}
            onKeyDown={(e) => {
              if (e.key !== "Enter" || e.shiftKey) return;
              e.preventDefault();
              const parsed = parsePosition(fen);
              if ("error" in parsed) {
                setPosError(parsed.error);
                return;
              }
              setPosError(null);
              send({ cmd: "position", fen: parsed.fen, moves: parsed.moves });
            }}
          />
        </div>
      )}
    </div>
  );
}
