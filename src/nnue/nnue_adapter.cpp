#include "nnue_adapter.h"

namespace Search {

int nnue_adapter_evaluate_stm(const NNUE::Runtime &nnue,
                              const Core::Position &pos) {
  return nnue.evaluate(pos);
}

} // namespace Search
