#ifndef PERFTS_H
#define PERFTS_H

#include <cstdint>
#include <string>

namespace Core {
    class Position;
}

void run_perft_suite();

int run_epd_test_suite(const std::string& filepath, int max_depth_limit = 6);

uint64_t perft(Core::Position& pos, int depth);

void perft_divide(Core::Position& pos, int depth);

// Runs perft on one FEN and checks the node count. Returns 0 on match,
// 1 on mismatch, 2 on a bad FEN — suitable as a standalone CTest command.
int run_perft_one(const std::string& fen, int depth, uint64_t expected);

#endif
