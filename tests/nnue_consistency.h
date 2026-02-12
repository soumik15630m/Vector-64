#ifndef NNUE_CONSISTENCY_H
#define NNUE_CONSISTENCY_H

#include <cstdint>

int run_nnue_incremental_consistency_test(
    int games = 32,
    int maxPlies = 80,
    uint64_t seed = 0xC0FFEEULL
);

#endif
