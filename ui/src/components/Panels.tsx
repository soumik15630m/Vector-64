import type { EngineState } from "../engine/types";

function Row({ k, v, mono = true }: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="row">
      <span className="k">{k}</span>
      <span className={mono ? "v num" : "v"}>{v}</span>
    </div>
  );
}

const n = (x: number) => x.toLocaleString("en-US");

export function EvalPanel({ s }: { s: EngineState }) {
  const f = s.frame;
  // Engine scores are side-to-move relative; show white's perspective too so
  // the bar does not flip meaning every ply.
  const stm = s.game.fen.split(" ")[1] === "b" ? -1 : 1;
  const cp = f ? f.eval : s.search.scoreCp;
  const white = cp * stm;
  const pct = 50 + 50 * Math.tanh(white / 400);

  return (
    <div className="panel">
      <h3>Evaluation</h3>
      <div className="evalbar" title="white perspective">
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
      <div style={{ height: 8 }} />
      <Row
        k="eval (white)"
        v={`${white >= 0 ? "+" : ""}${(white / 100).toFixed(2)}`}
      />
      {f && (
        <>
          <Row k="psqt" v={`${f.psqt}`} />
          <Row k="positional" v={`${f.positional}`} />
          <Row k="bucket" v={`${f.bucket} / ${s.arch.psqtBuckets}`} />
        </>
      )}
    </div>
  );
}

export function SearchPanel({ s }: { s: EngineState }) {
  const q = s.search;
  return (
    <div className="panel">
      <h3>Search</h3>
      <Row k="depth" v={`${q.depth} / ${q.seldepth}`} />
      <Row k="nodes" v={n(q.nodes)} />
      <Row k="nps" v={q.nps ? n(q.nps) : "—"} />
      <Row k="time" v={`${q.elapsedMs} ms`} />
      <Row k="threads" v={`${s.threads}`} />
      <Row
        k="tt hit (main)"
        v={`${q.negamaxTtHitRate.toFixed(1)}%`}
      />
      <Row k="tt hit (qs)" v={`${q.qsearchTtHitRate.toFixed(1)}%`} />
      {q.tbHits > 0 && <Row k="tb hits" v={n(q.tbHits)} />}
      <div style={{ marginTop: 6 }}>
        <span className="k" style={{ fontSize: 11, color: "var(--fg-faint)" }}>
          pv
        </span>
        <div className="moves" style={{ maxHeight: 52 }}>
          {q.pv.join(" ") || "—"}
        </div>
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
      <Row k="architecture" v={`${n(a.features)}→${a.hidden}x2`} />
      <Row k="dense" v={`${a.pair * 2}→${a.l1}→${a.l2}→1`} />
      <Row k="king buckets" v={`${a.kingBuckets}`} />
      {f && (
        <>
          <Row
            k="active features"
            v={`${f.white.featureCount} / ${f.black.featureCount}`}
          />
          <Row
            k="king bucket w/b"
            v={`${f.white.kingBucket} / ${f.black.kingBucket}`}
          />
          <Row
            k="mirrored w/b"
            v={`${f.white.mirrored ? "yes" : "no"} / ${
              f.black.mirrored ? "yes" : "no"
            }`}
          />
        </>
      )}
    </div>
  );
}

export function GamePanel({ s }: { s: EngineState }) {
  const g = s.game;
  const total = Math.max(1, g.wins + g.draws + g.losses);
  return (
    <div className="panel">
      <h3>Game</h3>
      <Row k="game" v={`#${g.gameIndex}`} />
      <Row k="ply" v={`${g.ply}`} />
      {g.over && <Row k="result" v={`${g.result} (${g.reason})`} />}
      <Row
        k="session w/d/l"
        v={`${g.wins}/${g.draws}/${g.losses}`}
      />
      <Row
        k="score"
        v={`${(((g.wins + g.draws / 2) / total) * 100).toFixed(1)}%`}
      />
      <div style={{ marginTop: 6 }}>
        <div className="moves">
          {g.moves.map((m, i) => (
            <span
              key={`${i}-${m}`}
              className={`mv${i === g.moves.length - 1 ? " last" : ""}`}
            >
              {i % 2 === 0 ? `${Math.floor(i / 2) + 1}.` : ""}
              {m}{" "}
            </span>
          ))}
          {g.moves.length === 0 && "—"}
        </div>
      </div>
    </div>
  );
}
