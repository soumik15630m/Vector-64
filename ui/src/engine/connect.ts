import { HttpEngineSource } from "./httpSource";
import type { EngineSource } from "./source";
import { WasmEngineSource } from "./wasmSource";

export interface BootProgress {
  phase: string;
  loaded: number;
  total: number;
  note?: string;
}

/**
 * Pick a transport at runtime, so one bundle serves both deployments.
 *
 * If a local ChessEngine-viz is answering, talk to it (full native speed). If
 * not, fall back to the WebAssembly build shipped alongside the page. Nothing
 * above EngineSource can tell the difference.
 */
export async function connect(
  onProgress: (p: BootProgress) => void,
): Promise<EngineSource> {
  // A native server responds to this immediately; a static host 404s.
  try {
    const res = await fetch("/api/health", {
      cache: "no-store",
      signal: AbortSignal.timeout(2500),
    });
    if (res.ok) return new HttpEngineSource("");
  } catch {
    // fall through to WASM
  }

  onProgress({ phase: "engine", loaded: 0, total: 1 });
  const wasm = new WasmEngineSource();
  await wasm.boot({
    moduleUrl: new URL("./stk-engine.js", location.href).href,
    netUrl: new URL("./stk-vector-64.nnue", location.href).href,
    onProgress: (phase, loaded, total) =>
      onProgress({
        phase,
        loaded,
        total,
        note:
          phase === "net"
            ? "downloading the real 46 MB network (cached after the first visit)"
            : undefined,
      }),
  });
  return wasm;
}

/** True when the browser gave us SharedArrayBuffer (so pthreads can be used). */
export function crossOriginIsolated(): boolean {
  return typeof SharedArrayBuffer !== "undefined" && self.crossOriginIsolated;
}
