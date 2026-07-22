#include "syzygy.h"

#include "../cores/bitboard.h"

// Fathom's public header (extern "C" internally). Its own quoted includes
// (tbconfig.h) resolve via the src/syzygy include directory added in CMake.
#include "syzygy/tbprobe.h"

namespace Search::Syzygy {
namespace {

// Fathom wants the en-passant target square index, or 0 for "none". Square 0
// (a1) is never a legal ep target, so the mapping is unambiguous.
unsigned ep_for_fathom(const Core::Position &pos) {
  const Core::Square ep = pos.ep_square();
  return ep == Core::SQ_NONE ? 0u : static_cast<unsigned>(ep);
}

// Fathom cannot probe positions with castling rights; any rights make it
// return FAILED. We only need "zero vs non-zero", so collapse to a flag.
unsigned castling_for_fathom(const Core::Position &pos) {
  return pos.castling_rights() != 0 ? 0xFu : 0u;
}

Wdl wdl_of(unsigned raw) {
  switch (raw) {
  case TB_WIN:
    return Wdl::WIN;
  case TB_LOSS:
    return Wdl::LOSS;
  default: // DRAW, CURSED_WIN, BLESSED_LOSS -> a draw under the 50-move rule
    return Wdl::DRAW;
  }
}

} // namespace

int init(const std::string &path) {
  tb_free();
  if (path.empty() || !tb_init(path.c_str()))
    return 0;
  return static_cast<int>(TB_LARGEST);
}

void shutdown() { tb_free(); }

int max_pieces() { return static_cast<int>(TB_LARGEST); }

bool active() { return TB_LARGEST > 0; }

Wdl probe_wdl(const Core::Position &pos) {
  const unsigned res = tb_probe_wdl(
      pos.pieces(Core::WHITE), pos.pieces(Core::BLACK), pos.pieces(Core::KING),
      pos.pieces(Core::QUEEN), pos.pieces(Core::ROOK), pos.pieces(Core::BISHOP),
      pos.pieces(Core::KNIGHT), pos.pieces(Core::PAWN),
      static_cast<unsigned>(pos.halfmove_clock()), castling_for_fathom(pos),
      ep_for_fathom(pos), pos.side_to_move() == Core::WHITE);
  return res == TB_RESULT_FAILED ? Wdl::FAIL : wdl_of(res);
}

Wdl probe_root(Core::Position &pos, Core::Move &best) {
  const unsigned res = tb_probe_root(
      pos.pieces(Core::WHITE), pos.pieces(Core::BLACK), pos.pieces(Core::KING),
      pos.pieces(Core::QUEEN), pos.pieces(Core::ROOK), pos.pieces(Core::BISHOP),
      pos.pieces(Core::KNIGHT), pos.pieces(Core::PAWN),
      static_cast<unsigned>(pos.halfmove_clock()), castling_for_fathom(pos),
      ep_for_fathom(pos), pos.side_to_move() == Core::WHITE, nullptr);
  if (res == TB_RESULT_FAILED || res == TB_RESULT_CHECKMATE ||
      res == TB_RESULT_STALEMATE)
    return Wdl::FAIL;

  const unsigned from = TB_GET_FROM(res);
  const unsigned to = TB_GET_TO(res);
  const unsigned promo = TB_GET_PROMOTES(res);

  // Match Fathom's (from, to, promotion) to a real legal move so castling and
  // en passant carry the correct move flags.
  Core::MoveList legal;
  Core::generate_legal_moves(pos, legal);
  for (int i = 0; i < legal.size(); ++i) {
    const Core::Move m = legal[i];
    if (static_cast<unsigned>(m.from_sq()) != from ||
        static_cast<unsigned>(m.to_sq()) != to)
      continue;
    if (promo == TB_PROMOTES_NONE) {
      if (!m.is_promotion()) {
        best = m;
        return wdl_of(TB_GET_WDL(res));
      }
      continue;
    }
    const Core::PieceType want = promo == TB_PROMOTES_QUEEN    ? Core::QUEEN
                                 : promo == TB_PROMOTES_ROOK   ? Core::ROOK
                                 : promo == TB_PROMOTES_BISHOP ? Core::BISHOP
                                                               : Core::KNIGHT;
    if (m.is_promotion() && m.promotion_type() == want) {
      best = m;
      return wdl_of(TB_GET_WDL(res));
    }
  }
  return Wdl::FAIL; // no legal move matched (should not happen)
}

} // namespace Search::Syzygy
