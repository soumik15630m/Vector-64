#include "evaluator.h"

namespace Search {

bool Evaluator::load_nnue(const std::string &path) {
  return big_.load_file(path);
}

bool Evaluator::load_nnue_small(const std::string &path) {
  return small_.load_file(path);
}

// With a net loaded the score is pure NNUE: the network carries its own
// PSQT side-output, so blending the hand-written PSQT back in would
// double-count material and skew the trained evaluation.
int Evaluator::evaluate(const Core::Position &pos) const {
  // Non-search path (UCI `eval`, fallbacks): rebuild an accumulator here.
  // Uses the best loaded net; the small net alone is valid (parity checks).
  if (big_.is_loaded()) {
    NNUE::Accumulator acc;
    big_.refresh(pos, acc);
    return big_.evaluate(pos, acc);
  }
  if (small_.is_loaded()) {
    NNUE::SmallAccumulator acc;
    small_.refresh(pos, acc);
    return small_.evaluate(pos, acc);
  }
  const int sign = pos.side_to_move() == Core::WHITE ? 1 : -1;
  return sign * (pos.material_wb() + pos.psqt_wb());
}

int Evaluator::evaluate(const Core::Position &pos, const NNUE::Accumulator &acc,
                        int lazyMargin) const {
  return big_.evaluate(pos, acc, lazyMargin);
}
} // namespace Search
