#ifndef SEARCH_MOVE_ORDERING_H
#define SEARCH_MOVE_ORDERING_H

#include "../cores/movegen.h"

namespace Search {

class MoveOrdering {
public:
  static constexpr int MAX_PLY = 128;

  MoveOrdering();

  void clear();
  void age_history();

  void update_killers(int ply, Core::Move move);
  void update_history(Core::Color side, Core::Move move, int depth);
  // Penalty for quiet moves that were searched but did not cause the cutoff.
  void update_history_malus(Core::Color side, Core::Move move, int depth);
  // Capture history, keyed by (mover, attacker type, target, victim type).
  void update_capture(Core::Color side, Core::PieceType attacker,
                      Core::Square to, Core::PieceType victim, int depth);
  void update_capture_malus(Core::Color side, Core::PieceType attacker,
                            Core::Square to, Core::PieceType victim, int depth);
  // Continuation history: how well the quiet (pieceType -> to) has done as a
  // reply to the previous move (prevPt -> prevTo). Zeroed for the root/null.
  void update_cont(Core::PieceType prevPt, Core::Square prevTo,
                   Core::PieceType pt, Core::Square to, int depth);
  void update_cont_malus(Core::PieceType prevPt, Core::Square prevTo,
                         Core::PieceType pt, Core::Square to, int depth);

  int score_move(const Core::Position &pos, Core::Move move, Core::Move ttMove,
                 int ply) const;
  void sort_moves(const Core::Position &pos, Core::MoveList &moves,
                  Core::Move ttMove, int ply) const;

  Core::Move killer(int ply, int idx) const {
    if (ply < 0 || ply >= MAX_PLY)
      return Core::Move::none();
    return killers_[ply][idx];
  }

  int history_score(Core::Color side, Core::Move move) const {
    return history_[side][move.from_sq()][move.to_sq()];
  }

  int capture_score(Core::Color side, Core::PieceType attacker, Core::Square to,
                    Core::PieceType victim) const {
    return captureHist_[side][attacker][to][victim];
  }

  int cont_score(Core::PieceType prevPt, Core::Square prevTo,
                 Core::PieceType pt, Core::Square to) const {
    if (prevPt == Core::NO_PIECE_TYPE)
      return 0; // no previous move (root / after a null move)
    return contHist_[prevPt][prevTo][pt][to];
  }

private:
  Core::Move killers_[MAX_PLY][2]{};
  int history_[Core::COLOR_NB][Core::SQUARE_NB][Core::SQUARE_NB]{};
  // [mover][attackerType][to][victimType]
  int captureHist_[Core::COLOR_NB][Core::PIECE_TYPE_NB][Core::SQUARE_NB]
                  [Core::PIECE_TYPE_NB]{};
  // [prevPieceType][prevTo][pieceType][to]  (~800 KB; colour-agnostic)
  int contHist_[Core::PIECE_TYPE_NB][Core::SQUARE_NB][Core::PIECE_TYPE_NB]
               [Core::SQUARE_NB]{};
};

} // namespace Search

#endif
