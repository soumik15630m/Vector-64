#include "evaluator.h"

namespace Search {

bool Evaluator::load_nnue(const std::string &path) {
  return net_.load_file(path);
}

// With a net loaded the score is pure NNUE: the network carries its own
// PSQT side-output, so blending the hand-written PSQT back in would
// double-count material and skew the trained evaluation.
int Evaluator::evaluate(const Core::Position &pos) const {
  if (!net_.is_loaded()) {
    const int sign = pos.side_to_move() == Core::WHITE ? 1 : -1;
    return sign * (pos.material_wb() + pos.psqt_wb());
  }

  // Non-search path (UCI `eval`, fallbacks): rebuild the accumulator here.
  NNUE::Accumulator acc;
  net_.refresh(pos, acc);
  return net_.evaluate(pos, acc);
}

int Evaluator::evaluate(const Core::Position &pos,
                        const NNUE::Accumulator &acc) const {
  return net_.evaluate(pos, acc);
}
} // namespace Search
