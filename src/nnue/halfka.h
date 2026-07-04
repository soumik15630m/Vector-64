#ifndef NNUE_HALFKA_H
#define NNUE_HALFKA_H

#include "../cores/bitboard.h"
#include "../cores/position.h"

// STK-HalfKA: a HalfKAv2_hm-equivalent feature set, implemented from scratch.
//
// For a given perspective (WHITE or BLACK) the board is oriented so that the
// perspective side's king sits on ranks 1-4 and files a-d:
//   - vertical flip (sq ^ 56) when the perspective is BLACK, so "our" back rank
//     is always rank 1;
//   - horizontal flip (sq ^ 7) when the oriented king is on files e-h, so the
//     king is always on the queenside half.
// The perspective's own king is the anchor (it selects the king bucket) and is
// not itself a feature. Every other piece contributes one feature indexed by
//   (king_bucket, piece_kind, oriented_square).
// Piece kinds are colour-relative (friendly vs enemy), giving 11 kinds:
//   friendly pawn..queen -> 0..4, enemy pawn..king -> 5..10.
// Totals: 32 king buckets * 11 kinds * 64 squares = 22528 features /
// perspective.

namespace NNUE::HalfKA {

constexpr int KING_BUCKETS = 32;
constexpr int PIECE_KINDS = 11;
constexpr int SQUARES = 64;
constexpr int FEATURES = KING_BUCKETS * PIECE_KINDS * SQUARES; // 22528

inline int flip_rank(int sq) { return sq ^ 56; }
inline int flip_file(int sq) { return sq ^ 7; }

// Orientation context for one perspective, computed once per accumulator build.
struct Orient {
  Core::Color side;
  bool mirror;    // horizontal flip active (perspective king on files e-h)
  int kingBucket; // 0..31
};

inline int orient_sq(int sq, Core::Color side, bool mirror) {
  const int s = (side == Core::WHITE) ? sq : flip_rank(sq);
  return mirror ? flip_file(s) : s;
}

inline Orient make_orient(Core::Color side, Core::Square kingSq) {
  const int ok = (side == Core::WHITE) ? int(kingSq) : flip_rank(int(kingSq));
  const bool mirror = (ok & 7) >= 4;
  const int tk = mirror ? flip_file(ok) : ok;
  const int bucket = (tk >> 3) * 4 + (tk & 7); // rank*4 + file, file in 0..3
  return Orient{side, mirror, bucket};
}

// Colour-relative piece kind in [0,11), or -1 for the perspective's own king.
inline int piece_kind(Core::Color side, Core::Color pieceColor,
                      Core::PieceType pt) {
  if (pieceColor == side) {
    if (pt == Core::KING)
      return -1;
    return int(pt) - 1; // PAWN..QUEEN -> 0..4
  }
  return 4 + int(pt); // enemy PAWN..KING -> 5..10
}

// Feature index for one piece, or -1 if it is the perspective's own king.
inline int feature_index(const Orient &o, Core::Color pieceColor,
                         Core::PieceType pt, Core::Square s) {
  const int kind = piece_kind(o.side, pieceColor, pt);
  if (kind < 0)
    return -1;
  const int ts = orient_sq(int(s), o.side, o.mirror);
  return (o.kingBucket * PIECE_KINDS + kind) * SQUARES + ts;
}

// Invoke push(featureIndex) for every active feature of `perspective`.
template <class Push>
inline void for_each_feature(const Core::Position &pos, Core::Color perspective,
                             Push push) {
  const Core::Square ksq = Core::lsb(pos.pieces(Core::KING, perspective));
  const Orient o = make_orient(perspective, ksq);
  for (int c = Core::WHITE; c <= Core::BLACK; ++c) {
    for (int pt = Core::PAWN; pt <= Core::KING; ++pt) {
      Core::Bitboard bb = pos.pieces(Core::PieceType(pt), Core::Color(c));
      while (bb) {
        const Core::Square s = Core::pop_lsb(bb);
        const int f = feature_index(o, Core::Color(c), Core::PieceType(pt), s);
        if (f >= 0)
          push(f);
      }
    }
  }
}

} // namespace NNUE::HalfKA

#endif
