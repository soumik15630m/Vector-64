import { useMemo, useState } from "react";
import { Chess } from "chess.js";
import type { IterRecord, PlyRecord } from "../engine/history";
import type { ControlCommand, EngineState } from "../engine/types";

const FILES = "abcdefgh";

/**
 * The evaluation across the game so far, and a way back into it.
 *
 * Watching the number drift forward tells you little; seeing the whole curve
 * shows you the move where it turned. Clicking a point seeks the engine to that
 * position, so "where did this go wrong" becomes one click rather than a replay.
 */
export function EvalGraph({
  plies,
  onSeek,
}: {
  plies: PlyRecord[];
  onSeek: (p: PlyRecord) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 268;
  const H = 74;

  const { path, zero, pts, lo, hi } = useMemo(() => {
    if (plies.length === 0)
      return { path: "", zero: H / 2, pts: [] as { x: number; y: number }[], lo: 0, hi: 0 };
    const vals = plies.map((p) => p.evalWhite);
    // Symmetric range around zero so "level" always sits on the mid-line.
    const span = Math.max(120, ...vals.map((v) => Math.abs(v)));
    const y = (v: number) => H / 2 - (v / span) * (H / 2 - 4);
    const x = (i: number) =>
      plies.length === 1 ? W / 2 : (i / (plies.length - 1)) * W;
    const pts = plies.map((p, i) => ({ x: x(i), y: y(p.evalWhite) }));
    return {
      path: pts.map((q, i) => `${i ? "L" : "M"}${q.x.toFixed(1)},${q.y.toFixed(1)}`).join(" "),
      zero: H / 2,
      pts,
      lo: -span,
      hi: span,
    };
  }, [plies]);

  if (plies.length === 0)
    return (
      <div className="panel">
        <h3>Evaluation graph</h3>
        <div className="axis">waiting for the first move…</div>
      </div>
    );

  const h = hover != null ? plies[hover] : null;

  return (
    <div className="panel">
      <h3>
        Evaluation graph <i>{h ? `ply ${h.ply}` : `${plies.length} plies`}</i>
      </h3>
      <svg
        className="evalgraph"
        viewBox={`0 0 ${W} ${H}`}
        onMouseLeave={() => setHover(null)}
      >
        <line x1="0" y1={zero} x2={W} y2={zero} className="zero" />
        <path d={`${path} L${W},${zero} L0,${zero} Z`} className="fill" />
        <path d={path} className="line" />
        {h && hover != null && (
          <>
            <line x1={pts[hover].x} y1="0" x2={pts[hover].x} y2={H} className="cursor" />
            <circle cx={pts[hover].x} cy={pts[hover].y} r="3" className="dot" />
          </>
        )}
        {/* One wide hit-target per ply: the line itself is too thin to click. */}
        {pts.map((q, i) => (
          <rect
            key={i}
            x={q.x - W / plies.length / 2}
            y="0"
            width={W / plies.length}
            height={H}
            fill="transparent"
            style={{ cursor: "pointer" }}
            onMouseEnter={() => setHover(i)}
            onClick={() => onSeek(plies[i])}
          />
        ))}
      </svg>
      <div className="axis">
        <span className="num">{(lo / 100).toFixed(1)}</span>
        <span>
          {h
            ? `${h.move} · ${h.evalWhite >= 0 ? "+" : "−"}${Math.abs(h.evalWhite / 100).toFixed(2)} · click to seek`
            : "click a point to seek there"}
        </span>
        <span className="num">+{(hi / 100).toFixed(1)}</span>
      </div>
    </div>
  );
}

/**
 * How the search grew: nominal depth against selective depth, per iteration.
 *
 * The gap between the two is the point -- extensions and quiescence push lines
 * far past the nominal depth, which is exactly why the ply ceiling has to bound
 * the selective figure rather than the one you asked for.
 */
export function SearchTree({ iters }: { iters: IterRecord[] }) {
  if (iters.length === 0)
    return (
      <div className="panel">
        <h3>Search shape</h3>
        <div className="axis">waiting for a search…</div>
      </div>
    );
  const maxSel = Math.max(...iters.map((i) => i.seldepth), 1);
  const last = iters[iters.length - 1];
  return (
    <div className="panel">
      <h3>
        Search shape <i>depth vs seldepth</i>
      </h3>
      <div className="tree">
        {iters.slice(-56).map((it, i) => (
          <div key={i} className="col" title={`depth ${it.depth} · seldepth ${it.seldepth}`}>
            <i className="sel" style={{ height: `${(it.seldepth / maxSel) * 100}%` }} />
            <i className="nom" style={{ height: `${(it.depth / maxSel) * 100}%` }} />
          </div>
        ))}
      </div>
      <div className="axis">
        <span>
          <b className="sw nom" /> nominal <b className="sw sel" /> selective
        </span>
        <span className="num">
          {last.depth} / {last.seldepth}
        </span>
      </div>
    </div>
  );
}

/**
 * Remove a piece and see what the network says.
 *
 * The most direct question you can ask a chess evaluator -- "how much is this
 * piece worth, here?" -- and the answer comes from the real net on the real
 * position, not from a piece-value table.
 */
export function Ablation({
  s,
  send,
}: {
  s: EngineState;
  send: (c: ControlCommand) => void;
}) {
  const [baseline, setBaseline] = useState<{ fen: string; evalWhite: number } | null>(
    null,
  );
  const [removed, setRemoved] = useState<string | null>(null);

  const evalWhite = s.frame
    ? s.frame.eval * (s.frame.sideToMove === 0 ? 1 : -1)
    : s.search.scoreCp * (s.game.fen.split(" ")[1] === "b" ? -1 : 1);

  // Pieces that can actually be removed: kings must stay for a legal position.
  const pieces = useMemo(() => {
    const c = new Chess();
    try {
      c.load(s.game.fen);
    } catch {
      return [];
    }
    const out: { square: string; type: string; color: string }[] = [];
    for (let r = 0; r < 8; r++)
      for (let f = 0; f < 8; f++) {
        const sq = `${FILES[f]}${8 - r}`;
        const p = c.get(sq as never);
        if (p && p.type !== "k") out.push({ square: sq, ...p });
      }
    return out;
  }, [s.game.fen]);

  // Ablation moves the board into analysis mode, which would abandon the game
  // being recorded. The engine refuses it while a run is going; not offering it
  // is the honest version of the same rule.
  const locked = s.datagen.running;

  const ablate = (square: string) => {
    if (locked) return;
    const c = new Chess();
    try {
      c.load(s.game.fen);
    } catch {
      return;
    }
    if (!baseline) setBaseline({ fen: s.game.fen, evalWhite });
    c.remove(square as never);
    setRemoved(square);
    send({ cmd: "mode", value: "analysis" });
    send({ cmd: "position", fen: c.fen(), moves: [] });
  };

  const restore = () => {
    if (locked || !baseline) return;
    send({ cmd: "position", fen: baseline.fen, moves: [] });
    setRemoved(null);
    setBaseline(null);
  };

  const delta = baseline && removed ? evalWhite - baseline.evalWhite : null;

  return (
    <div className="panel">
      <h3>
        Ablation <i>{removed ? `without ${removed}` : "remove a piece"}</i>
      </h3>
      {removed && delta !== null ? (
        <>
          <div className="row">
            <span className="k">with the piece</span>
            <span className="v num">
              {baseline!.evalWhite >= 0 ? "+" : "−"}
              {Math.abs(baseline!.evalWhite / 100).toFixed(2)}
            </span>
          </div>
          <div className="row">
            <span className="k">without it</span>
            <span className="v num">
              {evalWhite >= 0 ? "+" : "−"}
              {Math.abs(evalWhite / 100).toFixed(2)}
            </span>
          </div>
          <div className="row">
            <span className="k">the net values it at</span>
            <span className={`v num ${delta < 0 ? "pos" : "neg"}`}>
              {(Math.abs(delta) / 100).toFixed(2)}
            </span>
          </div>
          <button className="btn" style={{ marginTop: 8 }} onClick={restore}>
            put it back
          </button>
        </>
      ) : (
        <>
          <div className="ablate-grid">
            {pieces.map((p) => (
              <button
                key={p.square}
                className={`ab ${p.color === "w" ? "wp" : "bp"}`}
                title={`remove the ${p.color === "w" ? "white" : "black"} ${p.type.toUpperCase()} on ${p.square}`}
                onClick={() => ablate(p.square)}
              >
                {p.color === "w" ? p.type.toUpperCase() : p.type}
                <em>{p.square}</em>
              </button>
            ))}
          </div>
          <div className="axis">
            <span>switches to analysis on that position</span>
          </div>
        </>
      )}
    </div>
  );
}
