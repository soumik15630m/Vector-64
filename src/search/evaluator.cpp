#include "evaluator.h"

namespace Search {

bool Evaluator::load_nnue(const std::string &path) {
  return net_.load_file(path);
}

int Evaluator::evaluate(const Core::Position &pos) const {
  const int sign = pos.side_to_move() == Core::WHITE ? 1 : -1;
  const int psqtStm = sign * pos.psqt_wb();

  if (!net_.is_loaded()) {
    return sign * pos.material_wb() + psqtStm;
  }

  // Stage 1: full accumulator rebuild per eval. The incremental accumulator
  // (search-threaded) replaces this refresh in the next step.
  NNUE::Accumulator acc;
  net_.refresh(pos, acc);
  const int nnueStm = net_.evaluate(pos, acc);
  return (nnueWeight_ * nnueStm + psqtWeight_ * psqtStm) /
         (nnueWeight_ + psqtWeight_);
}

int Evaluator::evaluate(const Core::Position &pos,
                        const NNUE::Accumulator &acc) const {
  const int sign = pos.side_to_move() == Core::WHITE ? 1 : -1;
  const int psqtStm = sign * pos.psqt_wb();
  const int nnueStm = net_.evaluate(pos, acc);
  return (nnueWeight_ * nnueStm + psqtWeight_ * psqtStm) /
         (nnueWeight_ + psqtWeight_);
}
} // namespace Search
