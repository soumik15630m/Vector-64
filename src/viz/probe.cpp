#include "probe.h"

#include "../cores/bitboard.h"
#include "../nnue/halfka.h"

#include <algorithm>
#include <numeric>

namespace Viz {
namespace {

namespace HK = NNUE::HalfKA;

// Enumerate the perspective's active features. Mirrors
// HalfKA::for_each_feature, but keeps the board square and piece identity that
// the plain enumeration discards -- that mapping is what lets the UI connect a
// piece on a square to the neuron column it drives.
PerspectiveInput collect(const Core::Position &pos, Core::Color persp) {
  using namespace Core;

  PerspectiveInput out;
  const Square ksq = lsb(pos.pieces(KING, persp));
  const HK::Orient o = HK::make_orient(persp, ksq);
  out.kingSquare = int(ksq);
  out.kingBucket = o.kingBucket;
  out.mirrored = o.mirror;
  // Every piece but the perspective's own king contributes exactly one feature.
  out.features.reserve(static_cast<size_t>(popcount(pos.occupancy())));

  for (int c = WHITE; c <= BLACK; ++c) {
    for (int pt = PAWN; pt <= KING; ++pt) {
      Bitboard bb = pos.pieces(PieceType(pt), Color(c));
      while (bb) {
        const Square s = pop_lsb(bb);
        const int f = HK::feature_index(o, Color(c), PieceType(pt), s);
        if (f < 0)
          continue; // the perspective's own king: bucket anchor, not a feature
        ActiveFeature af;
        af.square = int(s);
        af.orientedSquare = HK::orient_sq(int(s), o.side, o.mirror);
        af.pieceColor = int(c);
        af.pieceType = pt;
        af.pieceKind = HK::piece_kind(persp, Color(c), PieceType(pt));
        af.featureIndex = f;
        out.features.push_back(af);
      }
    }
  }
  return out;
}

} // namespace

VizFrame capture(const Core::Position &pos, const NNUE::Network &net,
                 int l1TopK) {
  constexpr int H = NNUE::Network::HIDDEN;
  constexpr int L1 = NNUE::Arch::L1;
  constexpr int L2 = NNUE::Arch::L2;

  VizFrame f;
  f.fen = pos.toFEN();
  f.sideToMove = int(pos.side_to_move());
  f.white = collect(pos, Core::WHITE);
  f.black = collect(pos, Core::BLACK);

  // Out-of-band: build a fresh accumulator and run the same forward pass the
  // search would, capturing every intermediate.
  NNUE::Accumulator acc;
  net.refresh(pos, acc);
  NNUE::Probe p;
  f.eval = net.evaluate_probe(pos, acc, p);

  f.accUs.assign(p.accUs.begin(), p.accUs.end());
  f.accThem.assign(p.accThem.begin(), p.accThem.end());
  f.l1in.assign(p.l1in.begin(), p.l1in.end());
  f.l1out.assign(p.l1out.begin(), p.l1out.end());
  f.l2out.assign(p.l2out.begin(), p.l2out.end());
  f.bucket = p.bucket;
  f.psqt = p.psqt;
  f.positional = p.positional;

  const NNUE::Network::Bucket &b = net.bucket_weights(p.bucket);

  f.outContrib.resize(L2);
  for (int j = 0; j < L2; ++j)
    f.outContrib[j] = int32_t(b.outw[j]) * int32_t(p.l2out[j]);

  f.l2Contrib.resize(static_cast<size_t>(L2) * L1);
  for (int o = 0; o < L2; ++o)
    for (int j = 0; j < L1; ++j)
      f.l2Contrib[static_cast<size_t>(o) * L1 + j] =
          int32_t(b.l2w[o * L1 + j]) * int32_t(p.l1out[j]);

  // L1 has H=1024 inputs per neuron, far too many to ship whole: keep the
  // strongest few per neuron. Ties break on index so a frame is reproducible.
  const int k = std::clamp(l1TopK, 0, H);
  f.l1TopK = k;
  if (k > 0) {
    f.l1Top.resize(static_cast<size_t>(L1) * k);
    std::vector<int> idx(H);
    for (int o = 0; o < L1; ++o) {
      const int8_t *w = &b.l1w[static_cast<size_t>(o) * H];
      const auto contrib = [&](int i) {
        return int32_t(w[i]) * int32_t(p.l1in[i]);
      };
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                        [&](int x, int y) {
                          const int32_t cx = contrib(x), cy = contrib(y);
                          const int32_t ax = cx < 0 ? -cx : cx;
                          const int32_t ay = cy < 0 ? -cy : cy;
                          if (ax != ay)
                            return ax > ay;
                          return x < y;
                        });
      for (int j = 0; j < k; ++j) {
        const int s = idx[j];
        f.l1Top[static_cast<size_t>(o) * k + j] = Contribution{s, contrib(s)};
      }
    }
  }
  return f;
}

} // namespace Viz
