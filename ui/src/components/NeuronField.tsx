import { useCallback, useEffect, useRef, useState } from "react";
import { FieldRenderer } from "../field/renderer";
import type { HoverTarget } from "../field/renderer";
import type { Arch, Frame } from "../engine/types";

interface Props {
  frame: Frame | null;
  arch: Arch | null;
  nnueActive: boolean;
  /** The engine's current principal variation; pv[0] is the move it will play. */
  pv: string[];
  depth: number;
  thinking: boolean;
}

// x is the centre of each column, matching FieldRenderer.layout().
const LAYERS = [
  { label: "input", hint: "active features", x: 7 },
  { label: "accumulator", hint: "2 × 1024 int16", x: 27 },
  { label: "pairwise", hint: "clipped ReLU", x: 46 },
  { label: "L1", hint: "16", x: 64 },
  { label: "L2", hint: "32", x: 79 },
  { label: "eval", hint: "centipawns", x: 92 },
];

const FILES = "abcdefgh";
const PIECE = ["", "pawn", "knight", "bishop", "rook", "queen", "king"];

const signed = (x: number) => `${x > 0 ? "+" : ""}${x}`;

/**
 * The big nodes (L1, L2, output) get a card rather than a cursor tooltip: they
 * are the part of the network a person can actually reason about, and each one
 * is explained in terms of the decision it is feeding.
 *
 * Only exact quantities are reported. An L1 neuron's effect on the final
 * evaluation is NOT linear -- the L2 clip and the shifts sit in between -- so
 * its drive into L2 is reported as what it is rather than converted into a
 * centipawn figure that would look precise and not be.
 */
function describeCard(
  h: HoverTarget,
  f: Frame,
  arch: Arch,
  pv: string[],
  depth: number,
  thinking: boolean,
): {
  head: string;
  big: string;
  sub: string;
  rows: [string, string][];
  line?: string;
  tone?: "pos" | "neg";
} | null {
  switch (h.layer) {
    case "out": {
      const tone = f.eval > 0 ? "pos" : f.eval < 0 ? "neg" : undefined;
      return {
        head: thinking ? "currently choosing" : "engine plays",
        big: pv[0] ?? "—",
        sub: `${signed(f.eval)} cp · depth ${depth}`,
        rows: [
          ["psqt", signed(f.psqt)],
          ["positional", signed(f.positional)],
          ["bucket", `${f.bucket} / ${arch.psqtBuckets}`],
        ],
        line: pv.length > 1 ? pv.join(" ") : undefined,
        tone,
      };
    }
    case "l1": {
      const act = f.l1out[h.index] ?? 0;
      let drive = 0;
      let bestO = -1;
      let bestV = 0;
      for (let o = 0; o < arch.l2; o++) {
        const v = f.l2Contrib[o * arch.l1 + h.index];
        drive += v;
        if (Math.abs(v) > Math.abs(bestV)) {
          bestV = v;
          bestO = o;
        }
      }
      const k = f.l1TopK;
      let topIn = 0;
      for (let j = 0; j < k; j++) {
        const v = f.l1Top[(h.index * k + j) * 2 + 1];
        if (Math.abs(v) > Math.abs(topIn)) topIn = v;
      }
      return {
        head: `L1 neuron ${h.index}`,
        big: `${act}`,
        sub: `activation · max ${arch.actMax}`,
        rows: [
          ["strongest input", signed(topIn)],
          ["drive into L2", signed(drive)],
          [
            "feeds most",
            bestO >= 0 ? `L2 ${bestO} (${signed(bestV)})` : "—",
          ],
        ],
        // Naming the move makes the chain concrete: this neuron is part of what
        // produced that choice, not an isolated number.
        line: pv[0] ? `feeds the choice of ${pv[0]}` : undefined,
        tone: drive > 0 ? "pos" : drive < 0 ? "neg" : undefined,
      };
    }
    case "l2": {
      const act = f.l2out[h.index] ?? 0;
      const c = f.outContrib[h.index] ?? 0;
      let total = 0;
      for (let i = 0; i < f.outContrib.length; i++)
        total += Math.abs(f.outContrib[i]);
      return {
        head: `L2 neuron ${h.index}`,
        big: signed(c),
        sub: "contribution to the evaluation",
        rows: [
          ["activation", `${act} / ${arch.actMax}`],
          ["share of output", `${((Math.abs(c) / Math.max(1, total)) * 100).toFixed(1)}%`],
          ["effect", c > 0 ? "raises eval" : c < 0 ? "lowers eval" : "neutral"],
        ],
        line: pv[0] ? `feeds the choice of ${pv[0]}` : undefined,
        tone: c > 0 ? "pos" : c < 0 ? "neg" : undefined,
      };
    }
    default:
      return null;
  }
}

/** Small cells keep a cursor tooltip: there are thousands and they are tiny. */
function describe(
  h: HoverTarget,
  f: Frame,
  arch: Arch,
): { title: string; rows: [string, string][] } | null {
  const whiteIsUs = f.sideToMove === 0;
  const rows: [string, string][] = [];

  switch (h.layer) {
    case "accW":
    case "accB": {
      const white = h.layer === "accW";
      const arr = white === whiteIsUs ? f.accUs : f.accThem;
      rows.push(["value", `${arr[h.index] ?? 0}`]);
      rows.push(["pairs with", `#${(h.index + arch.pair) % arch.hidden}`]);
      return {
        title: `${white ? "White" : "Black"} accumulator #${h.index}`,
        rows,
      };
    }
    case "pairW":
    case "pairB": {
      const white = h.layer === "pairW";
      const off = white === whiteIsUs ? 0 : arch.pair;
      const v = f.l1in[off + h.index] ?? 0;
      rows.push(["activation", `${v} / ${arch.actMax}`]);
      rows.push(["feeds", `all ${arch.l1} L1 neurons`]);
      return {
        title: `${white ? "White" : "Black"} pairwise #${h.index}`,
        rows,
      };
    }
    case "square": {
      const sq = h.index;
      const name = `${FILES[sq % 8]}${Math.floor(sq / 8) + 1}`;
      const feat = f.whiteFeatures.find((x) => x.square === sq);
      if (!feat) return { title: name, rows: [["", "empty"]] };
      rows.push([
        "piece",
        `${feat.pieceColor === 0 ? "white" : "black"} ${PIECE[feat.pieceType] ?? "?"}`,
      ]);
      rows.push(["feature index", `${feat.featureIndex}`]);
      rows.push(["oriented square", `${feat.orientedSquare}`]);
      return { title: name, rows };
    }
    default:
      return null; // L1 / L2 / output are shown as a card instead
  }
}

export function NeuronField({
  frame,
  arch,
  nnueActive,
  pv,
  depth,
  thinking,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<FieldRenderer | null>(null);
  const [hover, setHover] = useState<HoverTarget | null>(null);

  const onHover = useCallback((h: HoverTarget | null) => setHover(h), []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    if (!canvas || !host) return;
    let disposed = false;
    const r = new FieldRenderer();
    void r.init(canvas, host, onHover).then(() => {
      if (disposed) {
        r.destroy();
        return;
      }
      rendererRef.current = r;
    });
    return () => {
      disposed = true;
      rendererRef.current = null;
      r.destroy();
    };
  }, [onHover]);

  // Esc releases a pinned node.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") rendererRef.current?.clearSelection();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const ro = new ResizeObserver(() => {
      if (rendererRef.current && arch) rendererRef.current.layout(arch);
    });
    ro.observe(host);
    return () => ro.disconnect();
  }, [arch]);

  useEffect(() => {
    const r = rendererRef.current;
    if (!r || !arch) return;
    r.layout(arch);
    if (frame) r.update(frame);
  }, [frame, arch]);

  // Big nodes get the card; small cells get the cursor tooltip. Never both.
  const card =
    hover && frame && arch
      ? describeCard(hover, frame, arch, pv, depth, thinking)
      : null;
  const info = hover && frame && arch && !card ? describe(hover, frame, arch) : null;

  return (
    <div ref={hostRef} className="field-host">
      <canvas ref={canvasRef} className="field-canvas" />

      {!nnueActive && (
        <div className="field-empty">
          No NNUE net loaded — the engine is using the classical evaluation, so
          there is no network to show.
        </div>
      )}

      <div className="field-labels">
        {LAYERS.map((l) => (
          <div key={l.label} style={{ left: `${l.x}%` }}>
            {l.label}
            <em>{l.hint}</em>
          </div>
        ))}
      </div>

      {info && hover && (
        <div
          className="field-tip"
          style={{
            left: Math.min(hover.x + 18, 9999),
            top: hover.y,
            transform:
              hover.x > (hostRef.current?.clientWidth ?? 0) * 0.72
                ? "translate(calc(-100% - 36px), -50%)"
                : "translateY(-50%)",
          }}
        >
          <h4>{info.title}</h4>
          {info.rows.map(([k, v]) => (
            <div key={k}>
              <span>{k}</span>
              <b className="num">{v}</b>
            </div>
          ))}
        </div>
      )}

      {/* Shown only while a big node is hovered, so the field stays clean. */}
      {card && (
        <div className="decision">
          <h5>{card.head}</h5>
          <div className={`mv${card.tone ? " " + card.tone : ""}`}>
            {card.big}
          </div>
          <div className="sub">{card.sub}</div>
          <div className="crows">
            {card.rows.map(([k, v]) => (
              <div key={k}>
                <span>{k}</span>
                <b className="num">{v}</b>
              </div>
            ))}
          </div>
          {card.line && <div className="line">{card.line}</div>}
        </div>
      )}

      {/* The two-colour scheme is only readable if it is named. */}
      <div className="field-legend">
        <span className="key">
          <span className="sw neg" /> negative
        </span>
        <span className="key">
          <span className="sw pos" /> positive
        </span>
        <span className="sep" />
        <span className="muted">cells = value · edges = effect on eval</span>
        <span className="sep" />
        <span className="muted">width = |weight × activation|</span>
        <span className="sep" />
        <span className="muted">click a neuron to pin it · esc to release</span>
      </div>
    </div>
  );
}
