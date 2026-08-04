import { decodeState } from "./decode";
import type { EngineSource } from "./source";
import type { ControlCommand, EngineState } from "./types";

/**
 * Native transport: long-polls /api/state.
 *
 * The server holds each request until it has something newer than `since`, so
 * there is no polling interval to tune and no busy loop. Because the next
 * request is only issued after the previous frame has been handed to the
 * renderer, the client applies natural backpressure -- it can never accumulate
 * a backlog of stale frames the way a push stream would.
 */
/**
 * Wait for the next paint before requesting another frame, so we never render
 * faster than the display.
 *
 * requestAnimationFrame stops firing entirely when the page is not compositing
 * (hidden tab, background window, headless capture). Racing it against a timer
 * means the stream keeps flowing instead of stalling forever in those states.
 */
function yieldToRenderer(): Promise<void> {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(finish, 250);
    requestAnimationFrame(finish);
  });
}

export class HttpEngineSource implements EngineSource {
  readonly label: string;
  private base: string;
  private stopped = false;

  constructor(base = "") {
    this.base = base;
    this.label = base || location.host;
  }

  subscribe(onState: (s: EngineState) => void): () => void {
    this.stopped = false;
    let since = 0;
    let backoff = 250;

    const loop = async () => {
      while (!this.stopped) {
        try {
          const res = await fetch(
            `${this.base}/api/state?since=${since}`,
            { cache: "no-store" },
          );
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const state = decodeState(await res.arrayBuffer());
          backoff = 250;
          if (state.seq !== since) {
            since = state.seq;
            onState(state);
            await yieldToRenderer();
          }
        } catch {
          if (this.stopped) return;
          await new Promise((r) => setTimeout(r, backoff));
          backoff = Math.min(backoff * 2, 4000);
        }
      }
    };
    void loop();
    return () => {
      this.stopped = true;
    };
  }

  async send(cmd: ControlCommand): Promise<boolean> {
    try {
      const res = await fetch(`${this.base}/api/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cmd),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  async netInfo(): Promise<unknown> {
    const res = await fetch(`${this.base}/api/net`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }
}
