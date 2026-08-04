import { useEffect, useState } from "react";
import type { ControlCommand, EngineState } from "../engine/types";

/**
 * Data generation, driven by the engine's OWN native generator -- the same
 * self-play loop, opening distribution and row format as `ChessEngine datagen`,
 * so a dataset built here is interchangeable with one built from the CLI.
 *
 * While a run is going the settings lock. A dataset assembled from shifting
 * node counts or labelling rules is not one dataset, so the only controls left
 * live are pause, resume and stop.
 */
export function DatagenPanel({
  s,
  send,
  probe,
}: {
  s: EngineState;
  send: (c: ControlCommand) => void;
  probe: (out: string) => Promise<{ resumable: boolean; positions: number; games: number }>;
}) {
  const d = s.datagen;
  const [out, setOut] = useState("data/selfplay.txt");
  const [target, setTarget] = useState("1000000");
  const [nodes, setNodes] = useState("6000");
  const [emit, setEmit] = useState<"raw" | "blend">("raw");
  const [lam, setLam] = useState("0.5");
  const [skipPlies, setSkipPlies] = useState("12");
  const [maxPlies, setMaxPlies] = useState("200");
  const [openingPlies, setOpeningPlies] = useState("8");
  const [balance, setBalance] = useState("150");
  const [seed, setSeed] = useState("12345");
  const [more, setMore] = useState(false);
  const [found, setFound] = useState<{ positions: number; games: number } | null>(
    null,
  );

  // Look for a recoverable run whenever the path changes, so a crashed session
  // is offered back instead of being silently overwritten.
  useEffect(() => {
    let alive = true;
    if (d.running || !out.trim()) {
      setFound(null);
      return;
    }
    const t = setTimeout(() => {
      probe(out.trim())
        .then((r) => alive && setFound(r.resumable ? r : null))
        .catch(() => alive && setFound(null));
    }, 400);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [out, d.running, probe]);

  const num = (v: string, def: number, min = 0) => {
    const x = Number(v.replace(/[_,\s]/g, ""));
    return Number.isFinite(x) && x >= min ? x : def;
  };

  const start = (resume: boolean) =>
    send({
      cmd: "datagen",
      action: "start",
      out: out.trim(),
      targetPositions: Math.max(1, num(target, 1000000, 1)),
      nodes: Math.max(1, num(nodes, 6000, 1)),
      emit,
      lam: Math.min(1, Math.max(0, num(lam, 0.5))),
      skipPlies: num(skipPlies, 12),
      maxPlies: Math.max(1, num(maxPlies, 200, 1)),
      openingPlies: num(openingPlies, 8),
      balance: num(balance, 150),
      seed: num(seed, 12345),
      resume,
    });

  const pct = d.target > 0 ? Math.min(100, (d.positions / d.target) * 100) : 0;
  const n = (x: number) => x.toLocaleString("en-US");

  if (d.running || d.positions > 0) {
    const total = Math.max(1, d.wins + d.draws + d.losses);
    return (
      <div className="panel">
        <h3>
          Datagen <i>{d.running ? (s.paused ? "paused" : "running") : "stopped"}</i>
        </h3>
        <div className="evalbar" style={{ margin: "2px 0 8px" }}>
          <div className="fill" style={{ left: 0, width: `${pct}%` }} />
        </div>
        <div className="row">
          <span className="k">positions</span>
          <span className="v num">
            {n(d.positions)} / {n(d.target)}
          </span>
        </div>
        <div className="row">
          <span className="k">games</span>
          <span className="v num">{n(d.games)}</span>
        </div>
        <div className="row">
          <span className="k">rate</span>
          <span className="v num">{Math.round(d.positionsPerSec)} pos/s</span>
        </div>
        <div className="row">
          <span className="k">eta</span>
          <span className="v num">
            {d.running && d.etaMinutes > 0 ? `${d.etaMinutes.toFixed(1)} min` : "—"}
          </span>
        </div>
        <div className="row">
          <span className="k">w/d/l</span>
          <span className="v num">
            {d.wins}/{d.draws}/{d.losses} ·{" "}
            {(((d.wins + d.draws / 2) / total) * 100).toFixed(0)}%
          </span>
        </div>
        <div className="row">
          <span className="k">file</span>
          <span className="v num" style={{ fontSize: 10 }}>
            {d.out.split(/[/\\]/).pop()}
          </span>
        </div>
        <div className="controls" style={{ marginTop: 8 }}>
          <button
            className={`btn${s.paused ? " on" : ""}`}
            onClick={() => send({ cmd: "pause", value: !s.paused })}
          >
            {s.paused ? "resume" : "pause"}
          </button>
          <button
            className="btn"
            onClick={() => send({ cmd: "datagen", action: "stop" })}
          >
            stop
          </button>
        </div>
        <div className="axis" style={{ marginTop: 6 }}>
          <span>settings locked while generating</span>
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <h3>
        Datagen <i>native generator</i>
      </h3>
      <label className="lbl">output file</label>
      <input
        type="text"
        value={out}
        onChange={(e) => setOut(e.currentTarget.value)}
      />
      <div className="field-row">
        <div>
          <label className="lbl">target positions</label>
          <input
            className="numbox"
            type="text"
            value={target}
            onChange={(e) => setTarget(e.currentTarget.value)}
          />
        </div>
        <div>
          <label className="lbl">nodes / move</label>
          <input
            className="numbox"
            type="text"
            value={nodes}
            onChange={(e) => setNodes(e.currentTarget.value)}
          />
        </div>
      </div>
      <div style={{ marginTop: 8 }}>
        <label className="lbl">label format</label>
        <div className="seg">
          {(["raw", "blend"] as const).map((x) => (
            <button
              key={x}
              className={emit === x ? "on" : ""}
              onClick={() => setEmit(x)}
              title={
                x === "raw"
                  ? "fen | eval | wdl  (bullet-native)"
                  : "fen | blended cp"
              }
            >
              {x}
            </button>
          ))}
        </div>
      </div>

      <button
        className="btn"
        style={{ marginTop: 9, width: "100%" }}
        onClick={() => setMore((v) => !v)}
      >
        {more ? "hide" : "show"} labelling options
      </button>
      {more && (
        <div className="field-row">
          <div>
            <label className="lbl">skip plies</label>
            <input
              className="numbox"
              type="text"
              value={skipPlies}
              title="opening plies left unlabelled"
              onChange={(e) => setSkipPlies(e.currentTarget.value)}
            />
          </div>
          <div>
            <label className="lbl">max plies</label>
            <input
              className="numbox"
              type="text"
              value={maxPlies}
              title="a game longer than this is scored a draw"
              onChange={(e) => setMaxPlies(e.currentTarget.value)}
            />
          </div>
          <div>
            <label className="lbl">opening plies</label>
            <input
              className="numbox"
              type="text"
              value={openingPlies}
              title="random balanced opening length"
              onChange={(e) => setOpeningPlies(e.currentTarget.value)}
            />
          </div>
          <div>
            <label className="lbl">balance cp</label>
            <input
              className="numbox"
              type="text"
              value={balance}
              title="reject an opening more lopsided than this"
              onChange={(e) => setBalance(e.currentTarget.value)}
            />
          </div>
          <div>
            <label className="lbl">seed</label>
            <input
              className="numbox"
              type="text"
              value={seed}
              title="same seed reproduces the same games"
              onChange={(e) => setSeed(e.currentTarget.value)}
            />
          </div>
          {emit === "blend" && (
            <div>
              <label className="lbl">lambda</label>
              <input
                className="numbox"
                type="text"
                value={lam}
                title="WDL weight in the blend, 0..1"
                onChange={(e) => setLam(e.currentTarget.value)}
              />
            </div>
          )}
        </div>
      )}

      {found && (
        <div className="resume-note">
          A previous run left <b>{n(found.positions)}</b> positions across{" "}
          <b>{n(found.games)}</b> games in this file. Resuming appends to it;
          starting fresh overwrites it.
        </div>
      )}

      <div className="controls" style={{ marginTop: 9 }}>
        <button className="btn on" onClick={() => start(false)}>
          {found ? "start fresh" : "start"}
        </button>
        {found && (
          <button className="btn" onClick={() => start(true)}>
            resume
          </button>
        )}
      </div>
    </div>
  );
}
