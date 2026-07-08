#ifndef SEARCH_EVALUATOR_H
#define SEARCH_EVALUATOR_H

#include "../cores/position.h"
#include "../nnue/network.h"

#include <string>

namespace Search {

class Evaluator {
public:
  bool load_nnue(const std::string &path);

  // Side-to-move centipawn score. The first form rebuilds the accumulator; the
  // second uses a caller-maintained (incrementally updated) accumulator.
  int evaluate(const Core::Position &pos) const;
  int evaluate(const Core::Position &pos, const NNUE::Accumulator &acc) const;

  // Incremental-accumulator hooks for the search (net weights are read-only, so
  // these are const; the accumulator stack is owned per search thread).
  bool nnue_active() const { return net_.is_loaded(); }
  void nnue_refresh(const Core::Position &pos, NNUE::Accumulator &acc) const {
    net_.refresh(pos, acc);
  }
  void nnue_update(const NNUE::Accumulator &parent, NNUE::Accumulator &child,
                   const Core::Position &after, Core::Move m,
                   const Core::UndoInfo &ui) const {
    net_.update(parent, child, after, m, ui);
  }

private:
  NNUE::Network net_{};
};

} // namespace Search

#endif
