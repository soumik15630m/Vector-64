import { useEffect, useState } from "react";
import type { EngineSource } from "../engine/source";
import type { EngineState } from "../engine/types";

interface Stats {
  min: number;
  max: number;
  mean: number;
  stddev: number;
  count: number;
}
interface NetInfo {
  loaded: boolean;
  arch: Record<string, number>;
  kingBucketMap: { square: number; bucket: number; mirror: boolean }[];
  buckets?: { bucket: number; l1: Stats; l2: Stats; out: Stats; outBias: number }[];
  ftWeights?: { min: number; max: number; count: number; bins: number[] };
}

/**
 * The net itself.
 *
 * The weight data is static — it is the loaded network — so on its own this
 * panel would never change. What makes it an *inspector* is overlaying the live
 * position on top of it: which PSQT bucket is in use right now, which king
 * buckets the two kings currently select, and how much of the network is
 * actually firing. Static structure, live read-out.
 */
export function NetInspector({
  source,
  state,
}: {
  source: EngineSource;
  state: EngineState | null;
}) {
  const [info, setInfo] = useState<NetInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    source
      .netInfo()
      .then((d) => alive && setInfo(d as NetInfo))
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [source]);

  if (err)
    return (
      <div className="panel">
        <h3>Net inspector</h3>
        <div style={{ color: "var(--bad)", fontSize: 11 }}>{err}</div>
      </div>
    );
  if (!info)
    return (
      <div className="panel">
        <h3>Net inspector</h3>
        <div style={{ color: "var(--fg-faint)", fontSize: 11 }}>loading…</div>
      </div>
    );

  const f = state?.frame ?? null;
  const ft = info.ftWeights;
  const peak = ft ? Math.max(...ft.bins) : 1;
  const nBuckets = info.arch.psqtBuckets ?? 8;

  // Live: how much of each layer is actually firing right now.
  const active = (a: Uint8Array | undefined) =>
    a && a.length ? a.reduce((c, v) => c + (v > 0 ? 1 : 0), 0) : 0;

  return (
    <>
      <div className="panel">
        <h3>
          Active bucket <i>live</i>
        </h3>
        <div className="bucket-bar">
          {Array.from({ length: nBuckets }, (_, i) => (
            <i key={i} className={f && f.bucket === i ? "on" : ""}>
              {i}
            </i>
          ))}
        </div>
        <div style={{ height: 8 }} />
        <div className="row">
          <span className="k">pieces on board</span>
          <span className="v num">
            {f ? f.white.featureCount + 1 : "—"}
          </span>
        </div>
        <div className="row">
          <span className="k">L1 firing</span>
          <span className="v num">
            {f ? `${active(f.l1out)} / ${info.arch.l1}` : "—"}
          </span>
        </div>
        <div className="row">
          <span className="k">L2 firing</span>
          <span className="v num">
            {f ? `${active(f.l2out)} / ${info.arch.l2}` : "—"}
          </span>
        </div>
        <div className="row">
          <span className="k">pairwise firing</span>
          <span className="v num">
            {f ? `${active(f.l1in)} / ${info.arch.hidden}` : "—"}
          </span>
        </div>
      </div>

      <div className="panel">
        <h3>
          King buckets <i>{info.arch.kingBuckets}</i>
        </h3>
        <div className="kb-grid">
          {Array.from({ length: 64 }, (_, i) => {
            const sq = (7 - Math.floor(i / 8)) * 8 + (i % 8); // rank 8 on top
            const e = info.kingBucketMap[sq];
            const t = e ? e.bucket / (info.arch.kingBuckets - 1 || 1) : 0;
            const isWhiteKing = f?.white.kingSquare === sq;
            const isBlackKing = f?.black.kingSquare === sq;
            return (
              <div
                key={i}
                className={
                  isWhiteKing ? "king-w" : isBlackKing ? "king-b" : undefined
                }
                title={`square ${sq} → bucket ${e?.bucket ?? "?"}${
                  e?.mirror ? " (mirrored)" : ""
                }`}
                style={{
                  color: t > 0.55 ? "#05070b" : "var(--fg-faint)",
                  background: `color-mix(in srgb, var(--accent) ${t * 88}%, var(--panel-2))`,
                }}
              >
                {e?.bucket ?? ""}
              </div>
            );
          })}
        </div>
        <div className="axis">
          <span>outlined = current kings</span>
          <span>
            {f ? `w ${f.white.kingBucket} · b ${f.black.kingBucket}` : "—"}
          </span>
        </div>
      </div>

      <div className="panel">
        <h3>
          Feature transformer <i>static</i>
        </h3>
        {ft ? (
          <>
            <div className="hist" title="distribution of every FT weight">
              {ft.bins.map((b, i) => (
                <i key={i} style={{ height: `${Math.max(1, (b / peak) * 100)}%` }} />
              ))}
            </div>
            <div className="axis">
              <span className="num">{ft.min}</span>
              <span className="num">
                {ft.count.toLocaleString("en-US")} weights
              </span>
              <span className="num">{ft.max}</span>
            </div>
          </>
        ) : (
          <div style={{ color: "var(--fg-faint)", fontSize: 11 }}>
            no net loaded
          </div>
        )}
      </div>

      {info.buckets && (
        <div className="panel">
          <h3>
            Dense weights <i>σ per bucket</i>
          </h3>
          {info.buckets.map((b) => (
            <div
              className="row"
              key={b.bucket}
              style={
                f && f.bucket === b.bucket
                  ? { color: "var(--accent)" }
                  : undefined
              }
            >
              <span className="k" style={f && f.bucket === b.bucket ? { color: "var(--accent)" } : undefined}>
                bucket {b.bucket}
                {f && f.bucket === b.bucket ? " ·" : ""}
              </span>
              <span className="v num">
                {b.l1.stddev.toFixed(1)} / {b.l2.stddev.toFixed(1)} /{" "}
                {b.out.stddev.toFixed(1)}
              </span>
            </div>
          ))}
          <div className="axis">
            <span>L1 / L2 / output</span>
          </div>
        </div>
      )}
    </>
  );
}
