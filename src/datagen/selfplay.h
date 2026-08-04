#ifndef DATAGEN_SELFPLAY_H
#define DATAGEN_SELFPLAY_H

#include "../cores/bitboard.h"
#include "../cores/movegen.h"
#include "../cores/position.h"

#include <cmath>
#include <cstdlib>
#include <random>

// Self-play primitives shared by the data generator (src/datagen/datagen.cpp)
// and the visualizer's self-play mode (src/viz/session.cpp), so both drive
// games by exactly the same rules: the same opening distribution and the same
// cheap draw detection.

namespace Datagen {

constexpr const char *START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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
