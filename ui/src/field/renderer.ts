import { Application, Container, Graphics, Sprite, Texture } from "pixi.js";
import type { Arch, Frame } from "../engine/types";

/**
 * The neuron field.
 *
 * Renders the actual forward pass, left to right:
 *
 *   active features -> accumulator (2 x 1024) -> pairwise (1024) -> L1 (16)
 *   -> L2 (32) -> output
 *
 * Nothing here is decorative. Every cell's colour is a real activation, and
 * every edge is drawn from the engine's own weight x activation attribution,
 * so the lines you see are the ones that actually moved the evaluation.
 *
 * Cells are persistent sprites whose tint/alpha are updated per frame (cheap
 * for ~3k nodes at 60fps); edges are one Graphics redrawn per frame.
 */

// Mirrors the CSS custom properties in theme.css so the canvas and the DOM
// chrome stay one visual system.
const COL = {
  neg: 0x3f8cff,
  pos: 0xffb545,
  zero: 0x1a2434,
  accent: 0x4dd8ff,
};

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

/** Signed value in [-1,1] -> cool/neutral/warm. */
function diverging(t: number): number {
  const c = Math.max(-1, Math.min(1, t));
  return c < 0
    ? mixHex(COL.zero, COL.neg, -c)
    : mixHex(COL.zero, COL.pos, c);
}

/** Build a soft round-square sprite texture once, reused by every cell. */
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
  grd.addColorStop(0.55, "rgba(255,255,255,0.85)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, size, size);
  return Texture.from(c);
}

interface Block {
  sprites: Sprite[];
  cols: number;
  rows: number;
  x: number;
  y: number;
  cw: number;
  ch: number;
}

export class FieldRenderer {
  private app = new Application();
  private ready = false;
  private tex!: Texture;

  private edges = new Graphics();
  private glow = new Graphics();
  private nodes = new Container();

  private accUs?: Block;
  private accThem?: Block;
  private pairUs?: Block;
  private pairThem?: Block;
  private l1: Sprite[] = [];
  private l2: Sprite[] = [];
  private out?: Sprite;
  private featureDots: Sprite[] = [];

  private arch?: Arch;
  private lastFrame?: Frame;
  private w = 0;
  private h = 0;

  async init(canvas: HTMLCanvasElement, parent: HTMLElement): Promise<void> {
    await this.app.init({
      canvas,
      background: 0x0b0f16,
      antialias: true,
      resolution: Math.min(window.devicePixelRatio || 1, 2),
      autoDensity: true,
      resizeTo: parent,
    });
    this.tex = cellTexture(32);
    this.app.stage.addChild(this.glow);
    this.app.stage.addChild(this.edges);
    this.app.stage.addChild(this.nodes);
    this.ready = true;
  }

  destroy(): void {
    if (this.ready) this.app.destroy(true, { children: true });
    this.ready = false;
  }

  private makeBlock(
    count: number,
    cols: number,
    x: number,
    y: number,
    cw: number,
    ch: number,
  ): Block {
    const rows = Math.ceil(count / cols);
    const sprites: Sprite[] = [];
    for (let i = 0; i < count; i++) {
      const s = new Sprite(this.tex);
      s.anchor.set(0.5);
      s.width = cw * 1.6;
      s.height = ch * 1.6;
      s.x = x + (i % cols) * cw + cw / 2;
      s.y = y + Math.floor(i / cols) * ch + ch / 2;
      s.tint = COL.zero;
      s.alpha = 0.5;
      this.nodes.addChild(s);
      sprites.push(s);
    }
    return { sprites, cols, rows, x, y, cw, ch };
  }

  private blockPos(b: Block, i: number): { x: number; y: number } {
    return {
      x: b.x + (i % b.cols) * b.cw + b.cw / 2,
      y: b.y + Math.floor(i / b.cols) * b.ch + b.ch / 2,
    };
  }

  /** (Re)build geometry for the current canvas size and architecture. */
  layout(arch: Arch): void {
    if (!this.ready) return;
    const w = this.app.renderer.width / this.app.renderer.resolution;
    const h = this.app.renderer.height / this.app.renderer.resolution;
    if (w === this.w && h === this.h && this.arch === arch) return;
    this.w = w;
    this.h = h;
    this.arch = arch;

    this.nodes.removeChildren();
    this.l1 = [];
    this.l2 = [];
    this.featureDots = [];

    const padY = 46;
    const usable = h - padY * 2;

    // --- input features: a small board so the position is recognisable -----
    const bx = w * 0.035;
    const boardSize = Math.min(usable * 0.34, w * 0.1);
    const sq = boardSize / 8;
    for (let i = 0; i < 64; i++) {
      const s = new Sprite(this.tex);
      s.anchor.set(0.5);
      s.width = s.height = sq * 0.82;
      s.x = bx + (i % 8) * sq + sq / 2;
      s.y = padY + (7 - Math.floor(i / 8)) * sq + sq / 2;
      s.tint = COL.zero;
      s.alpha = 0.25;
      this.nodes.addChild(s);
      this.featureDots.push(s);
    }

    // Two stacked blocks (us above, them below), sized to fill the column and
    // centred as a pair so the field stays balanced at any window size.
    const stack = (
      count: number,
      cols: number,
      cx: number,
      maxW: number,
      gap: number,
    ) => {
      const rows = Math.ceil(count / cols);
      const cell = Math.min(maxW / cols, (usable - gap) / 2 / rows);
      const bw = cell * cols;
      const bh = cell * rows;
      const top = padY + (usable - (bh * 2 + gap)) / 2;
      const x = cx - bw / 2;
      return {
        a: this.makeBlock(count, cols, x, top, cell, cell),
        b: this.makeBlock(count, cols, x, top + bh + gap, cell, cell),
      };
    };

    const acc = stack(arch.hidden, 32, w * 0.26, w * 0.16, 26);
    this.accUs = acc.a;
    this.accThem = acc.b;

    const pair = stack(arch.pair, 16, w * 0.45, w * 0.085, 26);
    this.pairUs = pair.a;
    this.pairThem = pair.b;

    // --- L1 / L2 / output: few enough to draw as real neurons --------------
    const mkColumn = (
      count: number,
      cx: number,
      radius: number,
      target: Sprite[],
    ) => {
      const spread = Math.min(usable, count * radius * 3.2);
      const step = count > 1 ? spread / (count - 1) : 0;
      const top = padY + (usable - spread) / 2;
      for (let i = 0; i < count; i++) {
        const s = new Sprite(this.tex);
        s.anchor.set(0.5);
        s.width = s.height = radius * 2;
        s.x = cx;
        s.y = count > 1 ? top + i * step : padY + usable / 2;
        s.tint = COL.zero;
        s.alpha = 0.6;
        this.nodes.addChild(s);
        target.push(s);
      }
    };
    mkColumn(arch.l1, w * 0.63, Math.max(7, usable / 90), this.l1);
    mkColumn(arch.l2, w * 0.78, Math.max(5, usable / 130), this.l2);

    this.out = new Sprite(this.tex);
    this.out.anchor.set(0.5);
    this.out.width = this.out.height = 34;
    this.out.x = w * 0.92;
    this.out.y = padY + usable / 2;
    this.out.tint = COL.accent;
    this.nodes.addChild(this.out);

    if (this.lastFrame) this.update(this.lastFrame);
  }

  /** Push one real frame into the field. */
  update(f: Frame): void {
    if (!this.ready || !this.arch || !this.accUs || !this.accThem) return;
    this.lastFrame = f;
    const arch = this.arch;
    const act = arch.actMax || 127;

    // Accumulators are int16 and unbounded in principle. Normalising by the
    // maximum lets a single outlier wash the whole block out, so scale by a few
    // times the mean magnitude instead: typical values then use the ramp, and
    // genuine outliers simply saturate (the mapping stays monotonic, so nothing
    // is misrepresented -- only the contrast is chosen).
    const spread = (a: Int16Array): number => {
      let sum = 0;
      for (let i = 0; i < a.length; i++) sum += a[i] < 0 ? -a[i] : a[i];
      return Math.max(1, (sum / Math.max(1, a.length)) * 2.5);
    };
    const paint = (b: Block, data: Int16Array, scale: number) => {
      for (let i = 0; i < b.sprites.length && i < data.length; i++) {
        const t = data[i] / scale;
        const s = b.sprites[i];
        s.tint = diverging(t);
        s.alpha = 0.28 + Math.min(1, Math.abs(t)) * 0.72;
      }
    };
    const sUs = spread(f.accUs);
    const sThem = spread(f.accThem);
    paint(this.accUs, f.accUs, sUs);
    paint(this.accThem, f.accThem, sThem);

    // Pairwise activations are already clipped to [0, ACT_MAX].
    const paintU8 = (b: Block, data: Uint8Array, off: number) => {
      for (let i = 0; i < b.sprites.length; i++) {
        const v = data[off + i] ?? 0;
        const t = v / act;
        const s = b.sprites[i];
        s.tint = t <= 0 ? COL.zero : mixHex(COL.zero, COL.accent, t);
        s.alpha = 0.22 + t * 0.78;
      }
    };
    if (this.pairUs) paintU8(this.pairUs, f.l1in, 0);
    if (this.pairThem) paintU8(this.pairThem, f.l1in, arch.pair);

    for (let i = 0; i < this.l1.length; i++) {
      const t = (f.l1out[i] ?? 0) / act;
      this.l1[i].tint = t <= 0 ? COL.zero : mixHex(COL.zero, COL.accent, t);
      this.l1[i].alpha = 0.3 + t * 0.7;
    }
    for (let i = 0; i < this.l2.length; i++) {
      const t = (f.l2out[i] ?? 0) / act;
      this.l2[i].tint = t <= 0 ? COL.zero : mixHex(COL.zero, COL.accent, t);
      this.l2[i].alpha = 0.3 + t * 0.7;
    }
    if (this.out) {
      const e = Math.max(-1, Math.min(1, f.eval / 600));
      this.out.tint = diverging(e);
      this.out.alpha = 0.75 + Math.abs(e) * 0.25;
    }

    // Board squares: light the ones that produced a feature this frame.
    for (const s of this.featureDots) {
      s.tint = COL.zero;
      s.alpha = 0.18;
    }
    for (const feat of f.whiteFeatures) {
      const d = this.featureDots[feat.square];
      if (d) {
        d.tint = COL.accent;
        d.alpha = 0.9;
      }
    }

    this.drawEdges(f);
  }

  /**
   * Edges come from attribution, not from the topology: for each L1 neuron we
   * draw only its strongest inputs, and downstream layers are drawn with width
   * and alpha proportional to |weight x activation|. So a thick bright line
   * means "this actually drove the eval".
   */
  private drawEdges(f: Frame): void {
    const g = this.edges;
    g.clear();
    if (!this.arch || !this.pairUs || !this.pairThem) return;
    const arch = this.arch;

    const pairPos = (i: number) =>
      i < arch.pair
        ? this.blockPos(this.pairUs!, i)
        : this.blockPos(this.pairThem!, i - arch.pair);

    // l1in -> L1, from the top-K table.
    const k = f.l1TopK;
    if (k > 0 && f.l1Top.length >= arch.l1 * k * 2) {
      let maxC = 1;
      for (let i = 1; i < f.l1Top.length; i += 2) {
        const v = Math.abs(f.l1Top[i]);
        if (v > maxC) maxC = v;
      }
      for (let o = 0; o < arch.l1; o++) {
        const node = this.l1[o];
        if (!node) continue;
        for (let j = 0; j < k; j++) {
          const base = (o * k + j) * 2;
          const src = f.l1Top[base];
          const val = f.l1Top[base + 1];
          if (!val) continue;
          const t = Math.abs(val) / maxC;
          if (t < 0.06) continue;
          const p = pairPos(src);
          g.moveTo(p.x, p.y);
          g.lineTo(node.x, node.y);
          g.stroke({
            width: 0.4 + t * 1.5,
            color: val >= 0 ? COL.pos : COL.neg,
            alpha: 0.06 + t * 0.5,
          });
        }
      }
    }

    // L1 -> L2 from l2Contrib.
    if (f.l2Contrib.length >= arch.l2 * arch.l1) {
      let maxC = 1;
      for (let i = 0; i < f.l2Contrib.length; i++) {
        const v = Math.abs(f.l2Contrib[i]);
        if (v > maxC) maxC = v;
      }
      for (let o = 0; o < arch.l2; o++) {
        const dst = this.l2[o];
        if (!dst) continue;
        for (let j = 0; j < arch.l1; j++) {
          const val = f.l2Contrib[o * arch.l1 + j];
          const t = Math.abs(val) / maxC;
          if (t < 0.12) continue;
          const src = this.l1[j];
          if (!src) continue;
          g.moveTo(src.x, src.y);
          g.lineTo(dst.x, dst.y);
          g.stroke({
            width: 0.3 + t * 1.1,
            color: val >= 0 ? COL.pos : COL.neg,
            alpha: 0.05 + t * 0.35,
          });
        }
      }
    }

    // L2 -> output from outContrib.
    if (this.out && f.outContrib.length >= arch.l2) {
      let maxC = 1;
      for (let i = 0; i < f.outContrib.length; i++) {
        const v = Math.abs(f.outContrib[i]);
        if (v > maxC) maxC = v;
      }
      for (let j = 0; j < arch.l2; j++) {
        const val = f.outContrib[j];
        const t = Math.abs(val) / maxC;
        if (t < 0.05) continue;
        const src = this.l2[j];
        if (!src) continue;
        g.moveTo(src.x, src.y);
        g.lineTo(this.out.x, this.out.y);
        g.stroke({
          width: 0.5 + t * 2.2,
          color: val >= 0 ? COL.pos : COL.neg,
          alpha: 0.1 + t * 0.6,
        });
      }
    }

    // Feature rays: each active piece feeds the whole accumulator, so the ray
    // goes to the block edge rather than pretending it targets one cell.
    if (this.accUs) {
      const tx = this.accUs.x - 6;
      const ty = this.accUs.y + (this.accUs.rows * this.accUs.ch) / 2;
      for (const feat of f.whiteFeatures) {
        const d = this.featureDots[feat.square];
        if (!d) continue;
        g.moveTo(d.x, d.y);
        g.lineTo(tx, ty);
        g.stroke({ width: 0.5, color: COL.accent, alpha: 0.12 });
      }
    }
  }
}
