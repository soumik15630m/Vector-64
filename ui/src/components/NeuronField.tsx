import { useEffect, useRef } from "react";
import { FieldRenderer } from "../field/renderer";
import type { Arch, Frame } from "../engine/types";

interface Props {
  frame: Frame | null;
  arch: Arch | null;
  nnueActive: boolean;
}

// x is the centre of each column, matching FieldRenderer.layout().
const LAYERS = [
  { label: "input", hint: "active features", x: 7 },
  { label: "accumulator", hint: "2 × 1024 int16", x: 26 },
  { label: "pairwise", hint: "clipped ReLU", x: 45 },
  { label: "L1", hint: "16", x: 63 },
  { label: "L2", hint: "32", x: 78 },
  { label: "eval", hint: "cp", x: 92 },
];

export function NeuronField({ frame, arch, nnueActive }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<FieldRenderer | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    if (!canvas || !host) return;
    let disposed = false;
    const r = new FieldRenderer();
    void r.init(canvas, host).then(() => {
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
  }, []);

  // Re-layout on container resize; the renderer no-ops when nothing changed.
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

  return (
    <div ref={hostRef} style={{ position: "absolute", inset: 0 }}>
      <canvas ref={canvasRef} className="field-canvas" />
      {!nnueActive && (
        <div className="field-empty">
          No NNUE net loaded — the engine is using the classical evaluation, so
          there is no network to show.
        </div>
      )}
      <div
        style={{
          position: "absolute",
          top: 10,
          left: 0,
          right: 0,
          display: "flex",
          pointerEvents: "none",
        }}
      >
        {LAYERS.map((l) => (
          <div
            key={l.label}
            style={{
              position: "absolute",
              left: `${l.x}%`,
              transform: "translateX(-50%)",
              textAlign: "center",
              fontFamily: "var(--mono)",
              fontSize: 9.5,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "var(--fg-faint)",
              whiteSpace: "nowrap",
              lineHeight: 1.45,
            }}
          >
            {l.label}
            <div style={{ opacity: 0.5, fontSize: 8.5, letterSpacing: "0.06em" }}>
              {l.hint}
            </div>
          </div>
        ))}
      </div>
      <div className="field-legend">
        <span>−</span>
        <span className="ramp" />
        <span>+</span>
        <span style={{ marginLeft: 8 }}>
          edge width = |weight × activation|
        </span>
      </div>
    </div>
  );
}
