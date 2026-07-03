#ifndef NNUE_NNUE_ADAPTER_H
#define NNUE_NNUE_ADAPTER_H

#include "nnue.h"

namespace Search {

// Side-to-move centipawn score from the loaded network.
// Convention: the network is trained side-to-move relative
// (positive = good for the mover), so no sign correction is applied.
int nnue_adapter_evaluate_stm(const NNUE::Runtime &nnue,
                              const Core::Position &pos);

} // namespace Search

#endif
