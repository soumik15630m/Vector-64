#ifndef SEARCH_EVALUATOR_H
#define SEARCH_EVALUATOR_H

#include "../cores/position.h"
#include "../nnue/network.h"

#include <string>

namespace Search {

// Owns the evaluation networks. The primary ("big") 1024-wide net scores
// balanced positions; the optional 128-wide small net scores positions the
// simple material+psqt estimate already calls clearly decided (dual-net lazy
// eval). Net weights are read-only after load; accumulator state is owned per
// search thread.
class Evaluator {
public:
  bool load_nnue(const std::string &path);       // big net
  bool load_nnue_small(const std::string &path); // small net

  // Side-to-move centipawn score, rebuilding an accumulator (UCI `eval`,
  // fallbacks). Pure big-net NNUE when loaded, else classical material+psqt.
  int evaluate(const Core::Position &pos) const;
  // Search path: caller-maintained accumulator.
  int evaluate(const Core::Position &pos, const NNUE::Accumulator &acc) const;

  bool nnue_active() const { return big_.is_loaded(); }
  bool small_active() const { return small_.is_loaded(); }
  const NNUE::Network &big() const { return big_; }
  const NNUE::SmallNetwork &small() const { return small_; }

private:
  NNUE::Network big_{};
  NNUE::SmallNetwork small_{};
};

} // namespace Search

#endif
