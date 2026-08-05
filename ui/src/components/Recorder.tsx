import { useState } from "react";
import type { ControlCommand, EngineState } from "../engine/types";

/**
 * Frame recording: append every published frame to a JSONL log.
 *
 * One line per frame -- position, search telemetry, the candidate moves and,
 * when a net is loaded, the eval broken into its PSQT and positional parts.
 * That is the file you plot from or diff two nets over, and it is the same
 * writer `--record` uses, so a session captured here is interchangeable with
 * one captured from the command line.
 *
 * The button follows the ENGINE's state, not a local flag: a run may already
 * have been started with --record, and a path that cannot be opened has to
 * report as not recording rather than silently look armed.
 */
export function Recorder({
  s,
  send,
}: {
  s: EngineState;
  send: (c: ControlCommand) => Promise<boolean>;
}) {
  const r = s.record;
  const [path, setPath] = useState("frames.jsonl");
  // The engine answers a rejected control with 400; surface that instead of
  // leaving the button looking like it did something.
  const [error, setError] = useState<string | null>(null);

  // Recording is one of the settings datagen locks: the run owns the session,
  // and a log started midway through would describe part of a dataset.
  const locked = s.datagen.running;

  const toggle = async () => {
    setError(null);
    if (r.recording) {
      await send({ cmd: "record", value: "" });
      return;
    }
    const p = path.trim();
    if (!p) {
      setError("needs a file name");
      return;
    }
    // The engine opens the file itself, so it is the only thing that knows
    // whether the path is writable.
    if (!(await send({ cmd: "record", value: p })))
      setError("could not open that file");
  };

  // The engine reports the path it actually opened, which may differ from what
  // is typed here once a recording is running.
  const shown = r.recording ? r.path : path;

  return (
    <div className="panel">
      <h3>
        Recorder
        <i>
          {r.recording
            ? `${r.frames.toLocaleString("en-US")} frames`
            : r.frames > 0
              ? `wrote ${r.frames.toLocaleString("en-US")} frames`
              : "jsonl · one line per frame"}
        </i>
      </h3>

      <label className="lbl">
        file
        {error && <b style={{ color: "var(--bad)" }}>{error}</b>}
      </label>
      <input
        type="text"
        value={shown}
        disabled={locked || r.recording}
        title={
          r.recording
            ? "stop recording to change the file"
            : "written relative to where the visualizer was started"
        }
        onChange={(e) => {
          setPath(e.currentTarget.value);
          setError(null);
        }}
      />

      <div className="controls" style={{ marginTop: 8 }}>
        <button
          className={`btn${r.recording ? " on" : ""}`}
          disabled={locked}
          title={
            locked ? "locked while generating data" : "append every frame"
          }
          onClick={() => void toggle()}
        >
          {r.recording ? "stop" : "record"}
        </button>
      </div>

      {r.recording && (
        <div className="row" style={{ marginTop: 6 }}>
          <span className="k">writing</span>
          <span className="v num" style={{ fontSize: 10 }}>
            {r.path.split(/[/\\]/).pop()}
          </span>
        </div>
      )}
    </div>
  );
}
