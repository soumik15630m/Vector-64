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

/**
 * Node budgets span five orders of magnitude, so the slider is logarithmic and
 * its top position means "no cap" -- the engine takes uint64 nodes, so there is
 * no ceiling to impose. Research runs need the extremes; watching needs the
 * low end.
 */
const NODE_STEPS = [
  1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7,
  1e8, 2e8, 5e8, 0,
];
const nodeLabel = (n: number) =>
  n === 0
    ? "unlimited"
    : n >= 1e6
      ? `${n / 1e6}M`
      : n >= 1e3
        ? `${n / 1e3}k`
        : `${n}`;

const MODES: { id: Mode; label: string }[] = [
  { id: "selfplay", label: "self-play" },
  { id: "analysis", label: "analysis" },
  { id: "human", label: "play" },
  { id: "datagen", label: "datagen" },
];

export function Controls({ s, send }: Props) {
  const [fen, setFen] = useState("");
  const [posError, setPosError] = useState<string | null>(null);
  const [delay, setDelay] = useState(300);
  const [nodeIdx, setNodeIdx] = useState(
    Math.max(0, NODE_STEPS.indexOf(20000)),
  );
  const [depth, setDepth] = useState(0); // 0 = no cap
  const [nodeVal, setNodeVal] = useState(20000);
  const [nodeText, setNodeText] = useState("20000");
  const [depthText, setDepthText] = useState("0");
  // Seeded from the engine so the control opens showing what is actually in
  // effect rather than a guess.
  const [variety, setVariety] = useState(s.varietyCp);

  // Typed entry: anything the engine accepts is allowed (nodes are uint64 on
  // the engine side, so there is no ceiling to enforce here); depth is clamped
  // to the engine's real maximum. Garbage input reverts rather than being sent.
  const commitNodes = () => {
    const v = Math.floor(Number(nodeText.replace(/[_,\s]/g, "")));
    if (!Number.isFinite(v) || v < 0) {
      setNodeText(String(nodeVal));
      return;
    }
    setNodeVal(v);
    setNodeText(String(v));
    const i = NODE_STEPS.indexOf(v);
    if (i >= 0) setNodeIdx(i);
    send({ cmd: "nodes", value: v });
  };
  const commitDepth = () => {
    const raw = Math.floor(Number(depthText.replace(/[_,\s]/g, "")));
    if (!Number.isFinite(raw) || raw < 0) {
      setDepthText(String(depth));
      return;
    }
    const v = Math.min(raw, s.maxDepth);
    setDepth(v);
    setDepthText(String(v));
    send({ cmd: "depth", value: v });
  };
  // Never offer more threads than the machine has: past the core count lazy
  // SMP gets slower, not faster. The engine clamps too, so a stale UI value
  // cannot push it out of range.
  const maxThreads = Math.max(1, s.hardwareThreads || 1);
  const [threads, setThreads] = useState(Math.min(s.threads, maxThreads));
  // The engine starts with this on (game 1 from the real start position,
  // later games opened randomly), so the toggle must open on too.
  const [randomOpening, setRandomOpening] = useState(true);

  // A dataset built from shifting settings is not one dataset, so everything
  // that would change the data locks while generating. Pause, resume and stop
  // stay live -- they are on the datagen panel.
  const locked = s.datagen.running;

  return (
    <div className="panel">
      <h3>Control{locked && <i>locked while generating</i>}</h3>

      <div className="seg" style={{ marginBottom: 9 }}>
        {MODES.map((m) => (
          <button
            key={m.id}
            className={s.mode === m.id ? "on" : ""}
            disabled={locked}
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
          disabled={locked || !s.paused}
          onClick={() => send({ cmd: "step" })}
        >
          step
        </button>
        <button
          className="btn"
          disabled={locked}
          title={locked ? "would discard the game being recorded" : undefined}
          onClick={() => send({ cmd: "newgame" })}
        >
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
        {s.mode === "selfplay" && (
          <button
            className={`btn${randomOpening ? " on" : ""}`}
            title="start each game from a random balanced opening instead of the initial position"
            onClick={() => {
              setRandomOpening((v) => {
                send({ cmd: "randomopening", value: !v });
                return !v;
              });
            }}
          >
            random opening
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
            disabled={locked}
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
            nodes<b>{nodeLabel(nodeVal)}</b>
          </label>
          <input
            type="range"
            min={0}
            max={NODE_STEPS.length - 1}
            step={1}
            disabled={locked}
            value={nodeIdx}
            onChange={(e) => {
              const i = Number(e.currentTarget.value);
              setNodeIdx(i);
              setNodeVal(NODE_STEPS[i]);
              setNodeText(String(NODE_STEPS[i]));
              send({ cmd: "nodes", value: NODE_STEPS[i] });
            }}
          />
          <input
            className="numbox"
            type="text"
            inputMode="numeric"
            value={nodeText}
            disabled={locked}
            title="exact node budget; 0 = unlimited"
            onChange={(e) => setNodeText(e.currentTarget.value)}
            onBlur={() => commitNodes()}
            onKeyDown={(e) => e.key === "Enter" && commitNodes()}
          />
        </div>
      </div>

      <div style={{ marginTop: 8 }}>
        <label className="lbl">
          depth<b>{depth === 0 ? `auto · max ${s.maxDepth}` : depth}</b>
        </label>
        <div className="with-box">
          <input
            type="range"
            min={0}
            max={s.maxDepth}
            step={1}
            disabled={locked}
            value={Math.min(depth, s.maxDepth)}
            onChange={(e) => {
              const v = Math.min(
                Math.max(0, Number(e.currentTarget.value) || 0),
                s.maxDepth,
              );
              setDepth(v);
              setDepthText(String(v));
              send({ cmd: "depth", value: v });
            }}
          />
          <input
            className="numbox"
            type="text"
            inputMode="numeric"
            value={depthText}
            disabled={locked}
            title={`exact depth; 0 = no cap, engine maximum ${s.maxDepth}`}
            onChange={(e) => setDepthText(e.currentTarget.value)}
            onBlur={() => commitDepth()}
            onKeyDown={(e) => e.key === "Enter" && commitDepth()}
          />
        </div>
      </div>

      <div style={{ marginTop: 8 }}>
        <label className="lbl">
          variety
          <b>{variety === 0 ? "off · always best" : `±${variety} cp`}</b>
        </label>
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          disabled={locked}
          value={variety}
          title={
            "Opening variety. The engine picks at random among the root moves " +
            "it scored within this many centipawns of the best, for the first " +
            "few plies only. A search is deterministic, so at 0 every game " +
            "from the same position is the same game."
          }
          onChange={(e) => {
            const v = Math.min(Math.max(0, Number(e.currentTarget.value) || 0), 100);
            setVariety(v);
            send({ cmd: "variety", value: v });
          }}
        />
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
          disabled={locked || maxThreads === 1}
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
            disabled={locked}
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
