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
  const [delay, setDelay] = useState(300);
  const [nodes, setNodes] = useState(20000);

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

      {s.mode === "analysis" && (
        <div style={{ marginTop: 9 }}>
          <label className="lbl">position</label>
          <input
            type="text"
            placeholder="paste a FEN, press Enter"
            value={fen}
            onChange={(e) => setFen(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && fen.trim())
                send({ cmd: "position", fen: fen.trim(), moves: [] });
            }}
          />
        </div>
      )}
    </div>
  );
}
