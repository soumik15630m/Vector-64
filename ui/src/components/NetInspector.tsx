import { useEffect, useState } from "react";
import type { EngineSource } from "../engine/source";

interface NetInfo {
  loaded: boolean;
  arch: Record<string, number>;
  kingBucketMap: { square: number; bucket: number; mirror: boolean }[];
  buckets?: {
    bucket: number;
    l1: Stats;
    l2: Stats;
    out: Stats;
    outBias: number;
  }[];
  ftWeights?: { min: number; max: number; count: number; bins: number[] };
}
interface Stats {
  min: number;
  max: number;
  mean: number;
  stddev: number;
  count: number;
}

/**
 * Static view of the loaded net itself: the shape of its weights and the
 * king-bucket map. Independent of any game -- this is the net, not a position.
 */
export function NetInspector({ source }: { source: EngineSource }) {
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

  if (err) return <div className="panel">net inspector: {err}</div>;
  if (!info) return <div className="panel">loading net…</div>;

  const ft = info.ftWeights;
  const peak = ft ? Math.max(...ft.bins) : 1;

  return (
    <>
      <div className="panel">
        <h3>Feature transformer</h3>
        {ft ? (
          <>
            <div className="hist" title="distribution of all FT weights">
              {ft.bins.map((b, i) => (
                <i
                  key={i}
                  style={{ height: `${Math.max(1, (b / peak) * 100)}%` }}
                />
              ))}
            </div>
            <div
              className="num"
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 10,
                color: "var(--fg-faint)",
                marginTop: 4,
              }}
            >
              <span>{ft.min}</span>
              <span>{ft.count.toLocaleString("en-US")} weights</span>
              <span>{ft.max}</span>
            </div>
          </>
        ) : (
          <div style={{ color: "var(--fg-faint)" }}>no net loaded</div>
        )}
      </div>

      <div className="panel">
        <h3>King buckets</h3>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(8, 1fr)",
            gap: 2,
          }}
        >
          {Array.from({ length: 64 }, (_, i) => {
            // Render rank 8 at the top.
            const sq = (7 - Math.floor(i / 8)) * 8 + (i % 8);
            const e = info.kingBucketMap[sq];
            const t = e ? e.bucket / (info.arch.kingBuckets - 1 || 1) : 0;
            return (
              <div
                key={i}
                title={`square ${sq} → bucket ${e?.bucket ?? "?"}${
                  e?.mirror ? " (mirrored)" : ""
                }`}
                className="num"
                style={{
                  aspectRatio: "1",
                  display: "grid",
                  placeItems: "center",
                  fontSize: 8,
                  borderRadius: 2,
                  color: t > 0.55 ? "#06080c" : "var(--fg-dim)",
                  background: `color-mix(in srgb, var(--accent) ${
                    t * 85
                  }%, var(--panel-2))`,
                }}
              >
                {e?.bucket ?? ""}
              </div>
            );
          })}
        </div>
      </div>

      {info.buckets && (
        <div className="panel">
          <h3>Dense weights per bucket</h3>
          {info.buckets.map((b) => (
            <div className="row" key={b.bucket}>
              <span className="k">bucket {b.bucket}</span>
              <span className="v num">
                σ {b.l1.stddev.toFixed(1)} / {b.l2.stddev.toFixed(1)} /{" "}
                {b.out.stddev.toFixed(1)}
              </span>
            </div>
          ))}
          <div
            style={{
              fontSize: 10,
              color: "var(--fg-faint)",
              marginTop: 4,
            }}
          >
            standard deviation of L1 / L2 / output weights
          </div>
        </div>
      )}
    </>
  );
}
