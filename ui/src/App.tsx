import { useEffect, useMemo, useState } from "react";
import { HttpEngineSource } from "./engine/httpSource";
import type { ConnectionState } from "./engine/source";
import type { ControlCommand, EngineState } from "./engine/types";
import { Board } from "./components/Board";
import { Controls } from "./components/Controls";
import { NeuronField } from "./components/NeuronField";
import { NetInspector } from "./components/NetInspector";
import {
  EvalPanel,
  GamePanel,
  NetworkPanel,
  SearchPanel,
} from "./components/Panels";

export default function App() {
  // One transport-agnostic source. A WASM build swaps this line and nothing
  // below it changes.
  const source = useMemo(() => new HttpEngineSource(""), []);
  const [state, setState] = useState<EngineState | null>(null);
  const [conn, setConn] = useState<ConnectionState>("connecting");
  const [showNet, setShowNet] = useState(false);

  useEffect(() => {
    const stop = source.subscribe((s) => {
      setState(s);
      setConn("live");
    });
    const t = setTimeout(
      () => setConn((c) => (c === "connecting" ? "error" : c)),
      8000,
    );
    return () => {
      stop();
      clearTimeout(t);
    };
  }, [source]);

  const send = (c: ControlCommand) => void source.send(c);

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          STK<span>·</span>Vector-64 <span>/</span> Vector Scope
        </div>
        <div style={{ flex: 1 }} />
        {state && (
          <div
            className="num"
            style={{ fontSize: 11, color: "var(--fg-dim)" }}
          >
            {state.thinking ? "thinking" : state.paused ? "paused" : "idle"}
            {" · "}
            {state.nnueActive ? "NNUE" : "classical"}
            {" · "}
            {state.threads} thread{state.threads === 1 ? "" : "s"}
          </div>
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
          onMove={(uci) => send({ cmd: "move", value: uci })}
        />
        {state && <EvalPanel s={state} />}
        {state && <Controls s={state} send={send} />}
        {state && <GamePanel s={state} />}
      </div>

      <div className="center">
        <NeuronField
          frame={state?.frame ?? null}
          arch={state?.arch ?? null}
          nnueActive={state?.nnueActive ?? false}
        />
      </div>

      <div className="col col-right">
        {showNet ? (
          <NetInspector source={source} />
        ) : (
          state && (
            <>
              <SearchPanel s={state} />
              <NetworkPanel s={state} />
            </>
          )
        )}
      </div>

      <div className="statusbar">
        <span>
          <span
            className={`dot ${conn === "live" ? "live" : conn === "error" ? "err" : ""}`}
          />
          {conn === "live"
            ? `live · ${source.label}`
            : conn === "connecting"
              ? "connecting…"
              : "no engine — is ChessEngine-viz running?"}
        </span>
        {state && (
          <>
            <span className="num">seq {state.seq.toLocaleString("en-US")}</span>
            <span className="num">{state.game.fen}</span>
          </>
        )}
        <div style={{ flex: 1 }} />
        <span>every value shown is the engine's own output</span>
      </div>
    </div>
  );
}
