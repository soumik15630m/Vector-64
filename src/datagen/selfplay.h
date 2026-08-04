#ifndef DATAGEN_SELFPLAY_H
#define DATAGEN_SELFPLAY_H

#include "../cores/bitboard.h"
#include "../cores/movegen.h"
#include "../cores/position.h"
#include "../search/transposition_table.h" // is_mate_score

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// Self-play primitives shared by the data generator (src/datagen/datagen.cpp)
// and the visualizer's self-play mode (src/viz/session.cpp), so both drive
// games by exactly the same rules: the same opening distribution and the same
// cheap draw detection.

namespace Datagen {

constexpr const char *START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr int MATE_CP = 8000;

// Search score -> a bounded white-perspective label; mates map to +/-MATE_CP.
inline int clamp_score(int cp) {
  if (Search::is_mate_score(cp))
    return cp > 0 ? MATE_CP : -MATE_CP;
  return std::max(-MATE_CP, std::min(MATE_CP, cp));
}

// WDL blend in win-probability space (CP_SCALE 400).
inline int blend_cp(int evalWhite, double wdl, double lam) {
  const double e = std::max(-4000.0, std::min(4000.0, double(evalWhite)));
  const double pe = 1.0 / (1.0 + std::exp(-e / 400.0));
  double p = (1.0 - lam) * pe + lam * wdl;
  p = std::min(std::max(p, 1e-4), 1.0 - 1e-4);
  return int(std::lround(400.0 * std::log(p / (1.0 - p))));
}

// One labelled row, exactly as both generators write it.
//   raw   : "<fen> | <eval> | <wdl>"     (bullet-native)
//   blend : "<fen> | <blended cp>"
inline std::string emit_row(const std::string &fen, int scoreCp, double wdl,
                            bool raw, double lam) {
  if (raw) {
    const char *ws = wdl == 1.0 ? "1.0" : (wdl == 0.0 ? "0.0" : "0.5");
    return fen + " | " + std::to_string(scoreCp) + " | " + ws;
  }
  return fen + " | " + std::to_string(blend_cp(scoreCp, wdl, lam));
}

// Draw by material: no pawn, rook or queen, and at most one minor overall.
inline bool insufficient_material(const Core::Position &pos) {
  if (pos.pieces(Core::PAWN) || pos.pieces(Core::ROOK) ||
      pos.pieces(Core::QUEEN))
    return false;
  return Core::popcount(pos.pieces(Core::KNIGHT) | pos.pieces(Core::BISHOP)) <=
         1;
}

// Seeded quiet, material-balanced random opening. false => caller retries.
// `balance` is the maximum |white-black material| in centipawns.
inline bool make_opening(Core::Position &pos, std::mt19937_64 &rng,
                         int openingPlies, int balance) {
  pos.setFromFEN(START_FEN);
  for (int i = 0; i < openingPlies; ++i) {
    Core::MoveList legal;
    Core::generate_legal_moves(pos, legal);
    if (legal.size() == 0)
      return false;
    Core::UndoInfo ui;
    pos.make_move(legal[int(rng() % uint64_t(legal.size()))], ui);
  }
  Core::MoveList legal;
  Core::generate_legal_moves(pos, legal);
  if (legal.size() == 0 || pos.in_check())
    return false;
  return std::abs(pos.material_wb()) <= balance;
}

} // namespace Datagen

#endif
