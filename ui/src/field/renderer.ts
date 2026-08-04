import {
  Application,
  Container,
  Graphics,
  Sprite,
  Text,
  TextStyle,
  Texture,
} from "pixi.js";
import type { Arch, Frame } from "../engine/types";

/**
 * The neuron field.
 *
 * Renders the actual forward pass, left to right:
 *
 *   active features -> accumulator (White / Black) -> pairwise clipped ReLU
 *   -> L1 (16) -> L2 (32) -> output
 *
 * Nothing here is decorative. Every cell's colour is a real activation, and
 * every edge is drawn from the engine's own weight x activation attribution,
 * so the lines you see are the ones that actually moved the evaluation.
 *
 * Perspectives are pinned to White and Black rather than the engine's
 * side-to-move-relative "us/them". The raw arrays swap meaning every ply, which
 * made the two blocks trade places on every move during self-play; pinning them
 * keeps each side's accumulator in one place so changes read as changes.
 */

// Mirrors the CSS custom properties in theme.css so canvas and DOM chrome stay
// one visual system.
const COL = {
  neg: 0x4a90ff,
  pos: 0xffb545,
  zero: 0x151d2b,
  accent: 0x4dd8ff,
  white: 0xe8eef7,
  black: 0x7b8ba3,
  label: 0x5c6b81,
};

export type HoverLayer =
  | "accW"
  | "accB"
  | "pairW"
  | "pairB"
  | "l1"
  | "l2"
  | "out"
  | "square";

export interface HoverTarget {
  layer: HoverLayer;
  index: number;
  /** Screen position of the hovered node, for tooltip placement. */
  x: number;
  y: number;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function mixHex(a: number, b: number, t: number): number {
  const ar = (a >> 16) & 255,
    ag = (a >> 8) & 255,
    ab = a & 255;
  const br = (b >> 16) & 255,
    bg = (b >> 8) & 255,
    bb = b & 255;
  return (
    (Math.round(lerp(ar, br, t)) << 16) |
    (Math.round(lerp(ag, bg, t)) << 8) |
    Math.round(lerp(ab, bb, t))
  );
}

/** Signed value in [-1,1] -> cool / neutral / warm. */
function diverging(t: number): number {
  const c = Math.max(-1, Math.min(1, t));
  return c < 0 ? mixHex(COL.zero, COL.neg, -c) : mixHex(COL.zero, COL.pos, c);
}

/** Soft round sprite texture, reused by every cell. */
function cellTexture(size: number): Texture {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  const grd = g.createRadialGradient(
    size / 2,
    size / 2,
    0,
    size / 2,
    size / 2,
    size / 2,
  );
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.5, "rgba(255,255,255,0.9)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, size, size);
  return Texture.from(c);
}

interface Block {
  sprites: Sprite[];
  count: number;
  cols: number;
  rows: number;
  x: number;
  y: number;
  cell: number;
}

const LABEL_STYLE = new TextStyle({
  fontFamily: "ui-monospace, Menlo, Consolas, monospace",
  fontSize: 9,
  letterSpacing: 1.6,
  fill: COL.label,
});

export class FieldRenderer {
  private app = new Application();
  private ready = false;
  private tex!: Texture;

  private backdrop = new Graphics();
  private edges = new Graphics();
  private highlight = new Graphics();
  private nodes = new Container();
  private labels = new Container();

  private accW?: Block;
  private accB?: Block;
  private pairW?: Block;
  private pairB?: Block;
  private l1: Sprite[] = [];
  private l2: Sprite[] = [];
  private out?: Sprite;
  private squares: Sprite[] = [];

  private arch?: Arch;
  private frame?: Frame;
  private hover: HoverTarget | null = null;
  private onHover?: (h: HoverTarget | null) => void;
  private w = 0;
  private h = 0;

  async init(
    canvas: HTMLCanvasElement,
    parent: HTMLElement,
    onHover: (h: HoverTarget | null) => void,
  ): Promise<void> {
    await this.app.init({
      canvas,
      backgroundAlpha: 0,
      antialias: true,
      resolution: Math.min(window.devicePixelRatio || 1, 2),
      autoDensity: true,
      resizeTo: parent,
    });
    this.onHover = onHover;
    this.tex = cellTexture(48);
    this.app.stage.addChild(this.backdrop);
    this.app.stage.addChild(this.edges);
    this.app.stage.addChild(this.highlight);
    this.app.stage.addChild(this.nodes);
    this.app.stage.addChild(this.labels);

    canvas.addEventListener("pointermove", this.pointerMove);
    canvas.addEventListener("pointerleave", this.pointerLeave);
    this.ready = true;
  }

  destroy(): void {
    if (!this.ready) return;
    const c = this.app.canvas as HTMLCanvasElement;
    c.removeEventListener("pointermove", this.pointerMove);
    c.removeEventListener("pointerleave", this.pointerLeave);
    this.app.destroy(true, { children: true });
    this.ready = false;
  }

  private pointerMove = (ev: PointerEvent) => {
    const rect = (this.app.canvas as HTMLCanvasElement).getBoundingClientRect();
    const hit = this.hitTest(ev.clientX - rect.left, ev.clientY - rect.top);
    const same =
      hit && this.hover
        ? hit.layer === this.hover.layer && hit.index === this.hover.index
        : hit === this.hover;
    if (same) return;
    this.hover = hit;
    this.onHover?.(hit);
    if (this.frame) this.drawEdges(this.frame);
  };

  private pointerLeave = () => {
    if (!this.hover) return;
    this.hover = null;
    this.onHover?.(null);
    if (this.frame) this.drawEdges(this.frame);
  };

  /** Grids are regular, so hit testing is arithmetic rather than a scan. */
  private hitTest(x: number, y: number): HoverTarget | null {
    const inBlock = (b: Block | undefined, layer: HoverLayer) => {
      if (!b) return null;
      const cx = Math.floor((x - b.x) / b.cell);
      const cy = Math.floor((y - b.y) / b.cell);
      if (cx < 0 || cy < 0 || cx >= b.cols || cy >= b.rows) return null;
      const i = cy * b.cols + cx;
      if (i >= b.count) return null;
      return {
        layer,
        index: i,
        x: b.x + cx * b.cell + b.cell / 2,
        y: b.y + cy * b.cell + b.cell / 2,
      };
    };
    const near = (arr: Sprite[], layer: HoverLayer, r: number) => {
      for (let i = 0; i < arr.length; i++) {
        const s = arr[i];
        const dx = s.x - x;
        const dy = s.y - y;
        if (dx * dx + dy * dy <= r * r)
          return { layer, index: i, x: s.x, y: s.y };
      }
      return null;
    };

    return (
      near(this.l1, "l1", 14) ??
      near(this.l2, "l2", 11) ??
      (this.out && Math.hypot(this.out.x - x, this.out.y - y) < 22
        ? { layer: "out" as HoverLayer, index: 0, x: this.out.x, y: this.out.y }
        : null) ??
      near(this.squares, "square", 9) ??
      inBlock(this.accW, "accW") ??
      inBlock(this.accB, "accB") ??
      inBlock(this.pairW, "pairW") ??
      inBlock(this.pairB, "pairB")
    );
  }

  private makeBlock(
    count: number,
    cols: number,
    x: number,
    y: number,
    cell: number,
  ): Block {
    const rows = Math.ceil(count / cols);
    const sprites: Sprite[] = [];
    for (let i = 0; i < count; i++) {
      const s = new Sprite(this.tex);
      s.anchor.set(0.5);
      s.width = s.height = cell * 1.55;
      s.x = x + (i % cols) * cell + cell / 2;
      s.y = y + Math.floor(i / cols) * cell + cell / 2;
      s.tint = COL.zero;
      this.nodes.addChild(s);
      sprites.push(s);
    }
    return { sprites, count, cols, rows, x, y, cell };
  }

  private label(text: string, x: number, y: number, tint = COL.label): void {
    const t = new Text({ text, style: LABEL_STYLE });
    t.tint = tint;
    t.anchor.set(0, 0.5);
    t.x = x;
    t.y = y;
    this.labels.addChild(t);
  }

  layout(arch: Arch): void {
    if (!this.ready) return;
    const w = this.app.renderer.width / this.app.renderer.resolution;
    const h = this.app.renderer.height / this.app.renderer.resolution;
    if (w === this.w && h === this.h && this.arch === arch) return;
    this.w = w;
    this.h = h;
    this.arch = arch;

    this.nodes.removeChildren();
    this.labels.removeChildren();
    this.l1 = [];
    this.l2 = [];
    this.squares = [];

    const padY = 46;
    const usable = h - padY - 38;

    // --- input: a small board, so the position stays recognisable ----------
    const boardSize = Math.min(usable * 0.34, w * 0.105);
    const sq = boardSize / 8;
    const bx = w * 0.07 - boardSize / 2;
    const by = padY + (usable - boardSize) / 2;
    for (let i = 0; i < 64; i++) {
      const s = new Sprite(this.tex);
      s.anchor.set(0.5);
      s.width = s.height = sq * 0.92;
      s.x = bx + (i % 8) * sq + sq / 2;
      s.y = by + (7 - Math.floor(i / 8)) * sq + sq / 2;
      s.tint = COL.zero;
      this.nodes.addChild(s);
      this.squares.push(s);
    }

    // Two stacked blocks per stage, sized to fill and centred as a pair.
    const stack = (count: number, cols: number, cx: number, maxW: number) => {
      const rows = Math.ceil(count / cols);
      const gap = 24;
      const cell = Math.min(maxW / cols, (usable - gap) / 2 / rows);
      const bw = cell * cols;
      const bh = cell * rows;
      const top = padY + (usable - (bh * 2 + gap)) / 2;
      const x = cx - bw / 2;
      const a = this.makeBlock(count, cols, x, top, cell);
      const b = this.makeBlock(count, cols, x, top + bh + gap, cell);
      this.label("WHITE", x, top - 11, COL.white);
      this.label("BLACK", x, top + bh + gap - 11, COL.black);
      return { a, b, x, bw, bh, top, gap };
    };

    const acc = stack(arch.hidden, 32, w * 0.28, w * 0.2);
    this.accW = acc.a;
    this.accB = acc.b;

    const pair = stack(arch.pair, 16, w * 0.47, w * 0.1);
    this.pairW = pair.a;
    this.pairB = pair.b;

    const mkColumn = (count: number, cx: number, r: number, out: Sprite[]) => {
      const spread = Math.min(usable * 0.92, count * r * 3.4);
      const step = count > 1 ? spread / (count - 1) : 0;
      const top = padY + (usable - spread) / 2;
      for (let i = 0; i < count; i++) {
        const s = new Sprite(this.tex);
        s.anchor.set(0.5);
        s.width = s.height = r * 2.4;
        s.x = cx;
        s.y = count > 1 ? top + i * step : padY + usable / 2;
        s.tint = COL.zero;
        this.nodes.addChild(s);
        out.push(s);
      }
    };
    mkColumn(arch.l1, w * 0.64, Math.max(7, usable / 78), this.l1);
    mkColumn(arch.l2, w * 0.79, Math.max(5, usable / 118), this.l2);

    this.out = new Sprite(this.tex);
    this.out.anchor.set(0.5);
    this.out.width = this.out.height = 44;
    this.out.x = w * 0.92;
    this.out.y = padY + usable / 2;
    this.out.tint = COL.accent;
    this.nodes.addChild(this.out);

    this.drawBackdrop(acc, pair, bx, by, boardSize);
    if (this.frame) this.update(this.frame);
  }

  /** Faint panels behind each stage: structure without competing for attention. */
  private drawBackdrop(
    acc: { x: number; bw: number; bh: number; top: number; gap: number },
    pair: { x: number; bw: number; bh: number; top: number; gap: number },
    bx: number,
    by: number,
    boardSize: number,
  ): void {
    const g = this.backdrop;
    g.clear();
    const pad = 9;
    const panel = (x: number, y: number, w: number, h: number) => {
      g.roundRect(x - pad, y - pad, w + pad * 2, h + pad * 2, 8);
      g.fill({ color: 0x0d131d, alpha: 0.55 });
      g.roundRect(x - pad, y - pad, w + pad * 2, h + pad * 2, 8);
      g.stroke({ width: 1, color: 0x1b2432, alpha: 0.8 });
    };
    panel(bx, by, boardSize, boardSize);
    for (const s of [acc, pair]) {
      panel(s.x, s.top, s.bw, s.bh);
      panel(s.x, s.top + s.bh + s.gap, s.bw, s.bh);
    }
  }

  update(f: Frame): void {
    if (!this.ready || !this.arch || !this.accW || !this.accB) return;
    this.frame = f;
    const arch = this.arch;
    const act = arch.actMax || 127;

    // The engine's arrays are side-to-move relative; pin them to White/Black so
    // the two blocks never trade places mid-game.
    const whiteIsUs = f.sideToMove === 0;
    const accWhite = whiteIsUs ? f.accUs : f.accThem;
    const accBlack = whiteIsUs ? f.accThem : f.accUs;
    const pairOffWhite = whiteIsUs ? 0 : arch.pair;
    const pairOffBlack = whiteIsUs ? arch.pair : 0;

    // Normalising by the maximum lets one outlier wash the block out, so scale
    // by a few times the mean magnitude. The mapping stays monotonic: only the
    // contrast is chosen, never the meaning.
    const spread = (a: Int16Array): number => {
      let sum = 0;
      for (let i = 0; i < a.length; i++) sum += a[i] < 0 ? -a[i] : a[i];
      return Math.max(1, (sum / Math.max(1, a.length)) * 2.5);
    };
    const paintAcc = (b: Block, data: Int16Array) => {
      const s = spread(data);
      for (let i = 0; i < b.sprites.length && i < data.length; i++) {
        const t = data[i] / s;
        const sp = b.sprites[i];
        sp.tint = diverging(t);
        sp.alpha = 0.3 + Math.min(1, Math.abs(t)) * 0.7;
      }
    };
    paintAcc(this.accW, accWhite);
    paintAcc(this.accB, accBlack);

    const paintU8 = (b: Block, data: Uint8Array, off: number) => {
      for (let i = 0; i < b.sprites.length; i++) {
        const t = (data[off + i] ?? 0) / act;
        const sp = b.sprites[i];
        sp.tint = t <= 0 ? COL.zero : mixHex(COL.zero, COL.accent, t);
        sp.alpha = 0.25 + t * 0.75;
      }
    };
    if (this.pairW) paintU8(this.pairW, f.l1in, pairOffWhite);
    if (this.pairB) paintU8(this.pairB, f.l1in, pairOffBlack);

    for (let i = 0; i < this.l1.length; i++) {
      const t = (f.l1out[i] ?? 0) / act;
      this.l1[i].tint = t <= 0 ? COL.zero : mixHex(COL.zero, COL.accent, t);
      this.l1[i].alpha = 0.35 + t * 0.65;
    }
    for (let i = 0; i < this.l2.length; i++) {
      const t = (f.l2out[i] ?? 0) / act;
      this.l2[i].tint = t <= 0 ? COL.zero : mixHex(COL.zero, COL.accent, t);
      this.l2[i].alpha = 0.35 + t * 0.65;
    }
    if (this.out) {
      const e = Math.max(-1, Math.min(1, f.eval / 600));
      this.out.tint = diverging(e);
      this.out.alpha = 0.8 + Math.abs(e) * 0.2;
    }

    for (const s of this.squares) {
      s.tint = COL.zero;
      s.alpha = 0.2;
    }
    // Light every occupied square; white and black pieces read differently.
    for (const feat of f.whiteFeatures) {
      const d = this.squares[feat.square];
      if (!d) continue;
      d.tint = feat.pieceColor === 0 ? COL.white : COL.black;
      d.alpha = 0.95;
    }

    this.drawEdges(f);
  }

  /**
   * Edges come from attribution, not topology: for each L1 neuron only its
   * strongest inputs are drawn, and every downstream edge scales with
   * |weight x activation|. When something is hovered, its own path is drawn
   * bright and everything else recedes.
   */
  private drawEdges(f: Frame): void {
    const g = this.edges;
    const hi = this.highlight;
    g.clear();
    hi.clear();
    if (!this.arch || !this.pairW || !this.pairB) return;
    const arch = this.arch;
    const hov = this.hover;
    const dim = hov ? 0.18 : 1;

    const whiteIsUs = f.sideToMove === 0;
    // Map a raw l1in index to its pinned White/Black block position.
    const pairPos = (i: number) => {
      const isUsHalf = i < arch.pair;
      const local = isUsHalf ? i : i - arch.pair;
      const white = isUsHalf === whiteIsUs;
      const b = white ? this.pairW! : this.pairB!;
      return {
        x: b.x + (local % b.cols) * b.cell + b.cell / 2,
        y: b.y + Math.floor(local / b.cols) * b.cell + b.cell / 2,
      };
    };

    const k = f.l1TopK;
    const hasTop = k > 0 && f.l1Top.length >= arch.l1 * k * 2;
    let maxTop = 1;
    if (hasTop)
      for (let i = 1; i < f.l1Top.length; i += 2)
        maxTop = Math.max(maxTop, Math.abs(f.l1Top[i]));

    // l1in -> L1
    if (hasTop) {
      for (let o = 0; o < arch.l1; o++) {
        const node = this.l1[o];
        if (!node) continue;
        const focus = hov?.layer === "l1" && hov.index === o;
        const target = focus ? hi : g;
        for (let j = 0; j < k; j++) {
          const base = (o * k + j) * 2;
          const src = f.l1Top[base];
          const val = f.l1Top[base + 1];
          if (!val) continue;
          const t = Math.abs(val) / maxTop;
          if (t < 0.06 && !focus) continue;
          const p = pairPos(src);
          target.moveTo(p.x, p.y);
          target.lineTo(node.x, node.y);
          target.stroke({
            width: focus ? 0.8 + t * 2.2 : 0.4 + t * 1.4,
            color: val >= 0 ? COL.pos : COL.neg,
            alpha: (focus ? 0.35 + t * 0.55 : 0.06 + t * 0.45) * (focus ? 1 : dim),
          });
        }
      }
    }

    // L1 -> L2
    if (f.l2Contrib.length >= arch.l2 * arch.l1) {
      let maxC = 1;
      for (let i = 0; i < f.l2Contrib.length; i++)
        maxC = Math.max(maxC, Math.abs(f.l2Contrib[i]));
      for (let o = 0; o < arch.l2; o++) {
        const dst = this.l2[o];
        if (!dst) continue;
        for (let j = 0; j < arch.l1; j++) {
          const val = f.l2Contrib[o * arch.l1 + j];
          const t = Math.abs(val) / maxC;
          const focus =
            (hov?.layer === "l2" && hov.index === o) ||
            (hov?.layer === "l1" && hov.index === j);
          if (t < 0.12 && !focus) continue;
          const src = this.l1[j];
          if (!src) continue;
          const target = focus ? hi : g;
          target.moveTo(src.x, src.y);
          target.lineTo(dst.x, dst.y);
          target.stroke({
            width: focus ? 0.8 + t * 1.8 : 0.3 + t * 1,
            color: val >= 0 ? COL.pos : COL.neg,
            alpha: (focus ? 0.3 + t * 0.5 : 0.05 + t * 0.32) * (focus ? 1 : dim),
          });
        }
      }
    }

    // L2 -> output
    if (this.out && f.outContrib.length >= arch.l2) {
      let maxC = 1;
      for (let i = 0; i < f.outContrib.length; i++)
        maxC = Math.max(maxC, Math.abs(f.outContrib[i]));
      for (let j = 0; j < arch.l2; j++) {
        const val = f.outContrib[j];
        const t = Math.abs(val) / maxC;
        const focus =
          hov?.layer === "out" || (hov?.layer === "l2" && hov.index === j);
        if (t < 0.05 && !focus) continue;
        const src = this.l2[j];
        if (!src) continue;
        const target = focus ? hi : g;
        target.moveTo(src.x, src.y);
        target.lineTo(this.out.x, this.out.y);
        target.stroke({
          width: focus ? 1 + t * 2.6 : 0.5 + t * 2,
          color: val >= 0 ? COL.pos : COL.neg,
          alpha: (focus ? 0.4 + t * 0.5 : 0.1 + t * 0.55) * (focus ? 1 : dim),
        });
      }
    }

    // Feature rays: a piece feeds the whole accumulator, so the ray goes to the
    // block edge rather than pretending it targets one cell.
    if (this.accW && this.accB) {
      for (const feat of f.whiteFeatures) {
        const d = this.squares[feat.square];
        if (!d) continue;
        const focus = hov?.layer === "square" && hov.index === feat.square;
        const target = focus ? hi : g;
        for (const b of [this.accW, this.accB]) {
          target.moveTo(d.x, d.y);
          target.lineTo(b.x - 8, b.y + (b.rows * b.cell) / 2);
          target.stroke({
            width: focus ? 1.2 : 0.5,
            color: COL.accent,
            alpha: (focus ? 0.5 : 0.1) * (focus ? 1 : dim),
          });
        }
      }
    }

    // Ring the hovered node so the cursor target is unambiguous.
    if (hov) {
      hi.circle(hov.x, hov.y, 13);
      hi.stroke({ width: 1.2, color: COL.accent, alpha: 0.9 });
    }
  }
}
