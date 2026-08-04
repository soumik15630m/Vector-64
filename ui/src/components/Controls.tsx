import { useState } from "react";
import type { ControlCommand, EngineState, Mode } from "../engine/types";

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

  return (
    <div className="panel">
      <h3>Control</h3>

      <div className="seg" style={{ marginBottom: 8 }}>
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

      <div className="controls" style={{ marginBottom: 8 }}>
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
            engine: {s.engineColor ? "black" : "white"}
          </button>
        )}
      </div>

      <label className="k" style={{ fontSize: 11 }}>
        move delay
      </label>
      <input
        type="range"
        min={0}
        max={2000}
        step={50}
        defaultValue={300}
        onChange={(e) =>
          send({ cmd: "delay", value: Number(e.currentTarget.value) })
        }
      />

      <label className="k" style={{ fontSize: 11 }}>
        nodes per move
      </label>
      <input
        type="range"
        min={1000}
        max={400000}
        step={1000}
        defaultValue={20000}
        onChange={(e) =>
          send({ cmd: "nodes", value: Number(e.currentTarget.value) })
        }
      />

      {s.mode === "analysis" && (
        <div style={{ marginTop: 8 }}>
          <label className="k" style={{ fontSize: 11 }}>
            position (FEN)
          </label>
          <input
            type="text"
            placeholder="paste a FEN and press Enter"
            value={fen}
            onChange={(e) => setFen(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && fen.trim()) {
                send({ cmd: "position", fen: fen.trim(), moves: [] });
              }
            }}
          />
        </div>
      )}
    </div>
  );
}
