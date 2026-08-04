import type { ControlCommand, EngineState } from "./types";

/**
 * The one interface every component talks to.
 *
 * Native builds implement it over HTTP long-poll (httpSource). A WASM build
 * implements it by calling the engine directly, with no networking at all.
 * Because nothing above this layer knows which is in use, adding the WASM
 * backend touches no rendering code.
 */
export interface EngineSource {
  /** Begin streaming. Returns an unsubscribe function. */
  subscribe(onState: (s: EngineState) => void): () => void;
  /** Send a control command. Resolves false if the engine rejected it. */
  send(cmd: ControlCommand): Promise<boolean>;
  /** Net inspector data (static description of the loaded net). */
  netInfo(): Promise<unknown>;
  /** Human-readable description of the connection, for the status bar. */
  readonly label: string;
}

export type ConnectionState = "connecting" | "live" | "error";
