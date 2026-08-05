import { useEffect, useRef, useState } from "react";
import { connect } from "./engine/connect";
import type { BootProgress } from "./engine/connect";
import type { ConnectionState, EngineSource } from "./engine/source";
import type { ControlCommand, EngineState } from "./engine/types";
import { Board } from "./components/Board";
import { Controls } from "./components/Controls";
import { NeuronField } from "./components/NeuronField";
import { NetInspector } from "./components/NetInspector";
import { DatagenPanel } from "./components/Datagen";
import { Ablation, EvalGraph, SearchTree } from "./components/Analysis";
import { useGameHistory } from "./engine/history";
import {
  CandidatesPanel,
  ClockPanel,
  EvalPanel,
  GamePanel,
  NetworkPanel,
  SearchPanel,
} from "./components/Panels";

export default function App() {
  // One transport-agnostic source: a local ChessEngine-viz if one is running,
  // otherwise the WebAssembly build. Nothing below this line knows which.
  const [source, setSource] = useState<EngineSource | null>(null);
  const [boot, setBoot] = useState<BootProgress | null>(null);
  const [state, setState] = useState<EngineState | null>(null);
  const [conn, setConn] = useState<ConnectionState>("connecting");
  // Hovering a candidate previews it on the board.
  const [hoverMove, setHoverMove] = useState<string | null>(null);
  // Click the board to enlarge it; click the speed chip for the full counters.
  const [boardBig, setBoardBig] = useState(false);
  const [showPerf, setShowPerf] = useState(false);
  // Right column: live telemetry, the net itself, or the analysis tools.
  const [rightTab, setRightTab] = useState<"live" | "net" | "tools">("live");
  // Column widths are user-adjustable; the centre takes whatever is left.
  const [cols, setCols] = useState({ left: 332, right: 300 });
  const drag = useRef<{ side: "left" | "right"; x0: number; w0: number } | null>(
    null,
  );

  useEffect(() => {
    const move = (e: PointerEvent) => {
      const d = drag.current;
      if (!d) return;
      const delta = e.clientX - d.x0;
      const raw = d.side === "left" ? d.w0 + delta : d.w0 - delta;
      // Keep both side columns usable and always leave room for the field.
      const max = Math.max(240, window.innerWidth * 0.34);
      const w = Math.min(max, Math.max(240, raw));
      setCols((c) => ({ ...c, [d.side]: w }));
    };
    const up = () => {
      drag.current = null;
      document.body.style.cursor = "";
      document.querySelectorAll(".gutter.drag").forEach((g) =>
        g.classList.remove("drag"),
      );
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, []);

  const startDrag = (side: "left" | "right") => (e: React.PointerEvent) => {
    drag.current = { side, x0: e.clientX, w0: cols[side] };
    document.body.style.cursor = "col-resize";
    (e.currentTarget as HTMLElement).classList.add("drag");
  };

  useEffect(() => {
    let alive = true;
    connect((p) => alive && setBoot(p))
      .then((s) => {
        if (!alive) return;
        setSource(s);
        setBoot(null);
      })
      .catch(() => alive && setConn("error"));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!source) return;
    const stop = source.subscribe((s) => {
      setState(s);
      setConn("live");
    });
    const t = setTimeout(
      () => setConn((c) => (c === "connecting" ? "error" : c)),
      12000,
    );
    return () => {
      stop();
      clearTimeout(t);
    };
  }, [source]);

  const { plies, iters } = useGameHistory(state);

  const send = (c: ControlCommand) => {
    if (source) void source.send(c);
  };

  // Seek: replay the game up to that ply in analysis mode, so the board AND the
  // network show the position as it was, not just the board.
  const seekTo = (ply: number) => {
    // Seeking switches to analysis mode, which would abandon the game being
    // recorded, so it is off while a datagen run is going.
    if (!state || state.datagen.running) return;
    send({ cmd: "mode", value: "analysis" });
    send({
      cmd: "position",
      fen: state.game.startFen,
      moves: state.game.moves.slice(0, ply),
    });
  };

  // Ask the engine whether an output file has a recoverable run behind it, so
  // a crashed session is offered back instead of silently overwritten.
  const probeDatagen = async (out: string) => {
    const res = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cmd: "datagen", action: "probe", out }),
    });
    return (await res.json()) as {
      resumable: boolean;
      positions: number;
      games: number;
    };
  };

  if (boot) return <BootScreen p={boot} />;

  return (
    <div
      className="app"
      style={{
        gridTemplateColumns: `${cols.left}px minmax(0, 1fr) ${cols.right}px`,
      }}
    >
      <div className="topbar">
        <div className="brand">
          STK<span>·</span>Vector-64 <span>/</span> Vector Scope
        </div>
        <div style={{ flex: 1 }} />
        {state && (
          <>
            <span className={`chip${state.thinking ? " on" : ""}`}>
              {state.thinking ? "thinking" : state.paused ? "paused" : "idle"}
            </span>
            <span className="chip">
              {state.nnueActive ? "NNUE" : "classical"}
            </span>
            <span className="chip">
              <b>{state.threads}</b> thread{state.threads === 1 ? "" : "s"}
            </span>
            <button
              className={`chip act${showPerf ? " on" : ""}`}
              onClick={() => setShowPerf((v) => !v)}
              title="speed and search counters"
            >
              <b>
                {state.search.nps
                  ? (state.search.nps / 1e6).toFixed(2)
                  : "0.00"}
              </b>{" "}
              Mnps
            </button>
          </>
        )}
        <button
          className={`btn${rightTab === "net" ? " on" : ""}`}
          onClick={() =>
            setRightTab((t) => (t === "net" ? "live" : "net"))
          }
        >
          net inspector
        </button>
      </div>

      <div className="col col-left">
        <div
          className="board-slot"
          onClick={() => setBoardBig(true)}
          title="click to enlarge"
        >
          <Board
            state={state}
            highlight={hoverMove}
            onMove={(uci) => send({ cmd: "move", value: uci })}
          />
        </div>
        <div className="col-scroll">
          {state && state.mode === "datagen" && (
            <DatagenPanel s={state} send={send} probe={probeDatagen} />
          )}
          {state && <ClockPanel s={state} />}
          {state && <EvalPanel s={state} />}
          {state && <CandidatesPanel s={state} onHover={setHoverMove} />}
          {state && <Controls s={state} send={send} />}
          {state && <GamePanel s={state} />}
        </div>
      </div>

      <div
        className="gutter"
        style={{ left: cols.left + 12 }}
        onPointerDown={startDrag("left")}
        title="drag to resize"
      />
      <div
        className="gutter"
        style={{ right: cols.right + 12, left: "auto" }}
        onPointerDown={startDrag("right")}
        title="drag to resize"
      />

      <div className="center">
        <NeuronField
          frame={state?.frame ?? null}
          arch={state?.arch ?? null}
          nnueActive={state?.nnueActive ?? false}
          pv={state?.search.pv ?? []}
          depth={state?.search.depth ?? 0}
          thinking={state?.thinking ?? false}
        />
      </div>

      <div className="col col-right">
        <div className="seg" style={{ marginBottom: 0 }}>
          {(["live", "net", "tools"] as const).map((t) => (
            <button
              key={t}
              className={rightTab === t ? "on" : ""}
              onClick={() => setRightTab(t)}
            >
              {t}
            </button>
          ))}
        </div>
        <div className="col-scroll">
          {rightTab === "net" && source && (
            <NetInspector source={source} state={state} />
          )}
          {rightTab === "live" && state && (
            <>
              <SearchPanel s={state} />
              <NetworkPanel s={state} />
            </>
          )}
          {rightTab === "tools" && state && (
            <>
              <EvalGraph plies={plies} onSeek={(p) => seekTo(p.ply)} />
              <SearchTree iters={iters} />
              <Ablation s={state} send={send} />
            </>
          )}
        </div>
      </div>

      {showPerf && state && (
        <PerfOverlay s={state} onClose={() => setShowPerf(false)} />
      )}

      {boardBig && state && (
        <BigBoard
          state={state}
          highlight={hoverMove}
          onMove={(uci) => send({ cmd: "move", value: uci })}
          onClose={() => setBoardBig(false)}
        />
      )}

      <div className="statusbar">
        <span>
          <span
            className={`dot ${conn === "live" ? "live" : conn === "error" ? "err" : ""}`}
          />
          {conn === "live"
            ? `live · ${source?.label ?? ""}`
            : conn === "connecting"
              ? "connecting…"
              : "no engine — is ChessEngine-viz running?"}
        </span>
        {state && (
          <>
            <span className="num">seq {state.seq.toLocaleString("en-US")}</span>
            <span className="num fen">{state.game.fen}</span>
          </>
        )}
        <div style={{ flex: 1 }} />
        <span>every value shown is the engine's own output</span>
      </div>
    </div>
  );
}

/** Full search counters, opened from the speed chip. */
function PerfOverlay({
  s,
  onClose,
}: {
  s: EngineState;
  onClose: () => void;
}) {
  const q = s.search;
  const rows: [string, string][] = [
    ["speed", `${(q.nps / 1e6).toFixed(3)} Mnps`],
    ["nodes this search", q.nodes.toLocaleString("en-US")],
    ["time", `${q.elapsedMs} ms`],
    ["depth / seldepth", `${q.depth} / ${q.seldepth}`],
    ["threads", `${s.threads} of ${s.hardwareThreads}`],
    ["nps per thread", `${(q.nps / Math.max(1, s.threads) / 1e6).toFixed(3)} Mnps`],
    ["tt hit · main", `${q.negamaxTtHitRate.toFixed(1)}%`],
    ["tt hit · qsearch", `${q.qsearchTtHitRate.toFixed(1)}%`],
    ["tb hits", q.tbHits ? q.tbHits.toLocaleString("en-US") : "—"],
    ["candidates searched", `${q.candidates.length}`],
    ["evaluation", s.nnueActive ? "NNUE (H=1024)" : "classical"],
  ];
  return (
    <div className="overlay" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <h3>
          Performance <i>click anywhere to close</i>
        </h3>
        {rows.map(([k, v]) => (
          <div className="row" key={k}>
            <span className="k">{k}</span>
            <span className="v num">{v}</span>
          </div>
        ))}
        <div className="pv" style={{ marginTop: 8 }}>
          {q.pv.join(" ") || "—"}
        </div>
      </div>
    </div>
  );
}

/** The board at full size, with the move list and candidates beside it. */
function BigBoard({
  state,
  highlight,
  onMove,
  onClose,
}: {
  state: EngineState;
  highlight: string | null;
  onMove: (uci: string) => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);

  const g = state.game;
  const stm = g.fen.split(" ")[1] === "b" ? -1 : 1;
  // Frame eval converts with the frame's own side to move (it is the PV leaf).
  const evalWhite = state.frame
    ? state.frame.eval * (state.frame.sideToMove === 0 ? 1 : -1)
    : state.search.scoreCp * stm;
  return (
    <div className="overlay" onClick={onClose}>
      <div className="board-sheet" onClick={(e) => e.stopPropagation()}>
        <div className="big-board">
          <Board state={state} highlight={highlight} onMove={onMove} />
        </div>
        <div className="board-side">
          <div className="panel">
            <h3>
              Position <i>game #{g.gameIndex}</i>
            </h3>
            <div className="row">
              <span className="k">evaluation</span>
              <span className="v num">
                {evalWhite >= 0 ? "+" : "−"}
                {Math.abs(evalWhite / 100).toFixed(2)}
              </span>
            </div>
            <div className="row">
              <span className="k">to move</span>
              <span className="v">{stm === 1 ? "white" : "black"}</span>
            </div>
            <div className="row">
              <span className="k">ply</span>
              <span className="v num">{g.ply}</span>
            </div>
            <div className="row">
              <span className="k">result</span>
              <span className="v">
                {g.over ? `${g.result} · ${g.reason}` : "in progress"}
              </span>
            </div>
            <div className="row">
              <span className="k">fen</span>
              <span className="v" />
            </div>
            <div className="pv" style={{ height: 34 }}>
              {g.fen}
            </div>
          </div>
          {state.search.candidates.length > 0 && (
            <CandidatesPanel s={state} />
          )}
          <div className="panel" style={{ minHeight: 0 }}>
            <h3>Moves</h3>
            <div className="moves" style={{ height: 150 }}>
              {g.moves.length === 0
                ? "—"
                : g.moves.map((m, i) => (
                    <span
                      key={`${i}-${m}`}
                      className={`mv${i === g.moves.length - 1 ? " last" : ""}`}
                    >
                      {i % 2 === 0 ? `${Math.floor(i / 2) + 1}.` : ""}
                      {m}{" "}
                    </span>
                  ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Shown while the WebAssembly engine and the real net are downloading. */
function BootScreen({ p }: { p: BootProgress }) {
  const pct = p.total > 0 ? Math.round((p.loaded / p.total) * 100) : 0;
  return (
    <div
      style={{
        height: "100%",
        display: "grid",
        placeItems: "center",
        gap: 14,
      }}
    >
      <div style={{ width: 420, maxWidth: "80vw", textAlign: "center" }}>
        <div className="brand" style={{ fontSize: 13, marginBottom: 18 }}>
          STK<span>�</span>Vector-64 <span>/</span> Vector Scope
        </div>
        <div className="evalbar" style={{ height: 6 }}>
          <div
            className="fill"
            style={{ left: 0, width: `${pct}%`, transition: "width .2s linear" }}
          />
        </div>
        <div
          className="num"
          style={{ marginTop: 10, fontSize: 11, color: "var(--fg-dim)" }}
        >
          {p.phase === "net" ? "network" : "engine"} � {pct}%
        </div>
        {p.note && (
          <div style={{ marginTop: 6, fontSize: 11, color: "var(--fg-faint)" }}>
            {p.note}
          </div>
        )}
      </div>
    </div>
  );
}
