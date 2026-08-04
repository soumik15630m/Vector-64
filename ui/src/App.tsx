import { useEffect, useState } from "react";
import { connect } from "./engine/connect";
import type { BootProgress } from "./engine/connect";
import type { ConnectionState, EngineSource } from "./engine/source";
import type { ControlCommand, EngineState } from "./engine/types";
import { Board } from "./components/Board";
import { Controls } from "./components/Controls";
import { NeuronField } from "./components/NeuronField";
import { NetInspector } from "./components/NetInspector";
import {
  CandidatesPanel,
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
  const [showNet, setShowNet] = useState(false);
  // Hovering a candidate previews it on the board.
  const [hoverMove, setHoverMove] = useState<string | null>(null);

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

  const send = (c: ControlCommand) => {
    if (source) void source.send(c);
  };

  if (boot) return <BootScreen p={boot} />;

  return (
    <div className="app">
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
            <span className="chip">
              <b>{state.search.nps ? Math.round(state.search.nps / 1000) : 0}</b>k
              nps
            </span>
          </>
        )}
        <button
          className={`btn${showNet ? " on" : ""}`}
          onClick={() => setShowNet((v) => !v)}
        >
          net inspector
        </button>
      </div>

      <div className="col col-left">
        <Board
          state={state}
          highlight={hoverMove}
          onMove={(uci) => send({ cmd: "move", value: uci })}
        />
        <div className="col-scroll">
          {state && <EvalPanel s={state} />}
          {state && <CandidatesPanel s={state} onHover={setHoverMove} />}
          {state && <Controls s={state} send={send} />}
          {state && <GamePanel s={state} />}
        </div>
      </div>

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
        <div className="col-scroll">
          {showNet && source ? (
            <NetInspector source={source} state={state} />
          ) : (
            state && (
              <>
                <SearchPanel s={state} />
                <NetworkPanel s={state} />
              </>
            )
          )}
        </div>
      </div>

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
