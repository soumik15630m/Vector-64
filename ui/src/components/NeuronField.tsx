import { useCallback, useEffect, useRef, useState } from "react";
import { FieldRenderer } from "../field/renderer";
import type { HoverTarget } from "../field/renderer";
import type { Arch, Frame } from "../engine/types";

interface Props {
  frame: Frame | null;
  arch: Arch | null;
  nnueActive: boolean;
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

/** What the hovered node is and what it contributed. All values are exact. */
function describe(
  h: HoverTarget,
  f: Frame,
  arch: Arch,
): { title: string; rows: [string, string][] } {
  const whiteIsUs = f.sideToMove === 0;
  const rows: [string, string][] = [];

  switch (h.layer) {
    case "l1": {
      const v = f.l1out[h.index] ?? 0;
      rows.push(["activation", `${v} / ${arch.actMax}`]);
      // Its own strongest inputs, and what it pushed into L2.
      const k = f.l1TopK;
      if (k > 0) {
        let top = 0;
        for (let j = 0; j < k; j++) {
          const val = f.l1Top[(h.index * k + j) * 2 + 1];
          if (Math.abs(val) > Math.abs(top)) top = val;
        }
        rows.push(["strongest input", `${top > 0 ? "+" : ""}${top}`]);
      }
      let sum = 0;
      for (let o = 0; o < arch.l2; o++) sum += f.l2Contrib[o * arch.l1 + h.index];
      rows.push(["total drive into L2", `${sum > 0 ? "+" : ""}${sum}`]);
      return { title: `L1 neuron ${h.index}`, rows };
    }
    case "l2": {
      const v = f.l2out[h.index] ?? 0;
      const c = f.outContrib[h.index] ?? 0;
      rows.push(["activation", `${v} / ${arch.actMax}`]);
      rows.push(["contribution to eval", `${c > 0 ? "+" : ""}${c}`]);
      rows.push([
        "share of output",
        `${((Math.abs(c) / Math.max(1, f.outContrib.reduce((a, b) => a + Math.abs(b), 0))) * 100).toFixed(1)}%`,
      ]);
      return { title: `L2 neuron ${h.index}`, rows };
    }
    case "out":
      rows.push(["eval (side to move)", `${f.eval > 0 ? "+" : ""}${f.eval} cp`]);
      rows.push(["psqt", `${f.psqt}`]);
      rows.push(["positional", `${f.positional}`]);
      rows.push(["bucket", `${f.bucket} / ${arch.psqtBuckets}`]);
      return { title: "output", rows };
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
  }
}

export function NeuronField({ frame, arch, nnueActive }: Props) {
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

  const info = hover && frame && arch ? describe(hover, frame, arch) : null;

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

      <div className="field-legend">
        <span>−</span>
        <span className="ramp" />
        <span>+</span>
        <span className="sep" />
        <span>edge width = |weight × activation|</span>
        <span className="sep" />
        <span>hover a neuron to trace its path</span>
      </div>
    </div>
  );
}
