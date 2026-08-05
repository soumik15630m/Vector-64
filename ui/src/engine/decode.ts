import type { ActiveFeature, EngineState, Frame } from "./types";

/**
 * Decode one framed state message:
 *
 *   [uint32 LE headerLen][headerLen bytes UTF-8 JSON][raw little-endian buffers]
 *
 * The header's `frame.buffers` table names each raw array, its element type and
 * length, in payload order, so this walks the table rather than hard-coding
 * offsets. Typed arrays are copied out (rather than viewing the source buffer)
 * because the caller keeps frames across renders.
 */
interface BufferSpec {
  name: string;
  type: "i16" | "u8" | "i32";
  len: number;
}

const ELEM_BYTES: Record<BufferSpec["type"], number> = { i16: 2, u8: 1, i32: 4 };

function unflattenFeatures(a: Int32Array): ActiveFeature[] {
  const out: ActiveFeature[] = [];
  for (let i = 0; i + 5 < a.length; i += 6) {
    out.push({
      square: a[i],
      orientedSquare: a[i + 1],
      pieceColor: a[i + 2],
      pieceType: a[i + 3],
      pieceKind: a[i + 4],
      featureIndex: a[i + 5],
    });
  }
  return out;
}

export function decodeState(buf: ArrayBuffer): EngineState {
  const view = new DataView(buf);
  const headerLen = view.getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)),
  );

  const specs: BufferSpec[] = header.frame.buffers ?? [];
  const raw: Record<string, Int16Array | Uint8Array | Int32Array> = {};
  let off = 4 + headerLen;
  for (const spec of specs) {
    const bytes = spec.len * ELEM_BYTES[spec.type];
    // Slice (copy) so the decoded frame does not alias the network buffer.
    const slice = buf.slice(off, off + bytes);
    raw[spec.name] =
      spec.type === "i16"
        ? new Int16Array(slice)
        : spec.type === "i32"
          ? new Int32Array(slice)
          : new Uint8Array(slice);
    off += bytes;
  }

  const hf = header.frame;
  const empty32 = new Int32Array(0);
  const frame: Frame | null = header.nnueActive
    ? {
        fen: hf.fen,
        sideToMove: hf.sideToMove,
        bucket: hf.bucket,
        psqt: hf.psqt,
        positional: hf.positional,
        eval: hf.eval,
        l1TopK: hf.l1TopK,
        white: hf.white,
        black: hf.black,
        accUs: (raw.accUs as Int16Array) ?? new Int16Array(0),
        accThem: (raw.accThem as Int16Array) ?? new Int16Array(0),
        l1in: (raw.l1in as Uint8Array) ?? new Uint8Array(0),
        l1out: (raw.l1out as Uint8Array) ?? new Uint8Array(0),
        l2out: (raw.l2out as Uint8Array) ?? new Uint8Array(0),
        outContrib: (raw.outContrib as Int32Array) ?? empty32,
        l2Contrib: (raw.l2Contrib as Int32Array) ?? empty32,
        l1Top: (raw.l1Top as Int32Array) ?? empty32,
        whiteFeatures: unflattenFeatures(
          (raw.whiteFeatures as Int32Array) ?? empty32,
        ),
        blackFeatures: unflattenFeatures(
          (raw.blackFeatures as Int32Array) ?? empty32,
        ),
      }
    : null;

  return {
    seq: header.seq,
    mode: header.mode,
    running: header.running,
    paused: header.paused,
    thinking: header.thinking,
    nnueActive: header.nnueActive,
    threads: header.threads,
    // Older engines will not send it; fall back to something safe rather than
    // offering a slider that runs to a made-up maximum.
    hardwareThreads: header.hardwareThreads ?? header.threads ?? 1,
    maxDepth: header.maxDepth ?? 64,
    // Falls back only if talking to an engine that predates the field.
    datagenDefaults: header.datagenDefaults ?? {
      out: "data/selfplay.txt", targetPositions: 500000000, nodes: 5000,
      depth: 9, emit: "raw", lam: 0.5, skipPlies: 12, maxPlies: 200,
      openingPlies: 8, balance: 150, seed: 12345,
    },
    datagen: header.datagen ?? {
      running: false, out: "", positions: 0, games: 0, target: 0,
      wins: 0, draws: 0, losses: 0, positionsPerSec: 0, etaMinutes: 0,
    },
    engineColor: header.engineColor,
    arch: header.arch,
    game: header.game,
    search: { ...header.search, candidates: header.candidates ?? [] },
    legalMoves: header.legalMoves ?? [],
    frame,
  };
}
