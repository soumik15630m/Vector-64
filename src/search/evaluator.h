#ifndef SEARCH_EVALUATOR_H
#define SEARCH_EVALUATOR_H

#include "../cores/position.h"
#include "../nnue/network.h"

#include <string>

namespace Search {

class Evaluator {
public:
  bool load_nnue(const std::string &path);

  // Side-to-move centipawn score.
  int evaluate(const Core::Position &pos) const;

private:
  NNUE::Network net_{};
  int nnueWeight_ = 8;
  int psqtWeight_ = 2;
};

} // namespace Search

#endif
