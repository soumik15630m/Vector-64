import { useEffect, useRef } from "react";
import type { EngineState } from "../engine/types";

function Row({
  k,
  v,
  tone,
}: {
  k: string;
  v: string;
  tone?: "pos" | "neg";
}) {
  return (
    <div className="row">
      <span className="k">{k}</span>
      <span className={`v num${tone ? " " + tone : ""}`}>{v}</span>
    </div>
  );
}

const n = (x: number) => x.toLocaleString("en-US");
const signed = (x: number) => `${x > 0 ? "+" : ""}${x}`;

export function EvalPanel({ s }: { s: EngineState }) {
  const f = s.frame;
  // Engine scores are side-to-move relative; show White's perspective so the
  // bar does not flip meaning every ply.
  const stm = s.game.fen.split(" ")[1] === "b" ? -1 : 1;
  const cp = f ? f.eval : s.search.scoreCp;
  const white = cp * stm;
  const pct = 50 + 50 * Math.tanh(white / 400);
  const tone = white > 15 ? "pos" : white < -15 ? "neg" : undefined;

  return (
    <div className="panel">
      <h3>
        Evaluation <i>white</i>
      </h3>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <span className={`eval-big${tone ? " " + tone : ""}`}>
          {white >= 0 ? "+" : "−"}
          {Math.abs(white / 100).toFixed(2)}
        </span>
        <span className="eval-side">
          {white > 40 ? "white better" : white < -40 ? "black better" : "level"}
        </span>
      </div>
      <div className="evalbar">
        <div
          className="fill"
          style={
            white >= 0
              ? { left: "50%", width: `${pct - 50}%` }
              : { left: `${pct}%`, width: `${50 - pct}%` }
          }
        />
        <div className="mid" />
      </div>
      {/* Rows are always present so the panel height never changes. */}
      <Row k="psqt" v={f ? signed(f.psqt) : "—"} />
      <Row k="positional" v={f ? signed(f.positional) : "—"} />
      <Row
        k="bucket"
        v={f ? `${f.bucket} / ${s.arch.psqtBuckets}` : "—"}
      />
    </div>
  );
}

export function SearchPanel({ s }: { s: EngineState }) {
  const q = s.search;
  return (
    <div className="panel">
      <h3>
        Search <i>{s.thinking ? "thinking" : "idle"}</i>
      </h3>
      <Row k="depth / seldepth" v={`${q.depth} / ${q.seldepth}`} />
      <Row k="nodes" v={n(q.nodes)} />
      <Row k="nps" v={q.nps ? n(q.nps) : "—"} />
      <Row k="time" v={`${q.elapsedMs} ms`} />
      <Row k="threads" v={`${s.threads}`} />
      <Row k="tt hit · main" v={`${q.negamaxTtHitRate.toFixed(1)}%`} />
      <Row k="tt hit · qsearch" v={`${q.qsearchTtHitRate.toFixed(1)}%`} />
      <Row k="tb hits" v={q.tbHits > 0 ? n(q.tbHits) : "—"} />
      <div className="pv" style={{ marginTop: 6 }}>
        {q.pv.join(" ") || "—"}
      </div>
    </div>
  );
}

export function NetworkPanel({ s }: { s: EngineState }) {
  const f = s.frame;
  const a = s.arch;
  return (
    <div className="panel">
      <h3>Network</h3>
      <Row k="features" v={`${n(a.features)} × 2`} />
      <Row k="accumulator" v={`${a.hidden} int16 × 2`} />
      <Row k="dense" v={`${a.pair * 2} → ${a.l1} → ${a.l2} → 1`} />
      <Row k="king buckets" v={`${a.kingBuckets}`} />
      <Row
        k="active features w/b"
        v={f ? `${f.white.featureCount} / ${f.black.featureCount}` : "—"}
      />
      <Row
        k="king bucket w/b"
        v={f ? `${f.white.kingBucket} / ${f.black.kingBucket}` : "—"}
      />
      <Row
        k="mirrored w/b"
        v={
          f
            ? `${f.white.mirrored ? "yes" : "no"} / ${f.black.mirrored ? "yes" : "no"}`
            : "—"
        }
      />
    </div>
  );
}

export function GamePanel({ s }: { s: EngineState }) {
  const g = s.game;
  const total = Math.max(1, g.wins + g.draws + g.losses);
  const listRef = useRef<HTMLDivElement>(null);

  // Keep the latest move visible without changing the panel's height.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [g.ply, g.gameIndex]);

  return (
    <div className="panel" style={{ minHeight: 0 }}>
      <h3>
        Game <i>#{g.gameIndex}</i>
      </h3>
      <Row k="ply" v={`${g.ply}`} />
      <Row
        k="result"
        v={g.over ? `${g.result} · ${g.reason}` : "in progress"}
      />
      <Row k="session w/d/l" v={`${g.wins}/${g.draws}/${g.losses}`} />
      <Row
        k="white score"
        v={`${(((g.wins + g.draws / 2) / total) * 100).toFixed(1)}%`}
      />
      <div className="moves" ref={listRef} style={{ marginTop: 6 }}>
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
  );
}
