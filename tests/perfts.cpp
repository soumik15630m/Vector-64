#include "perfts.h"
#include "../src/cores/position.h"
#include "../src/cores/movegen.h"
#include "../src/cores/attacks.h"
#include "../src/cores/zobrist.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <future>
#include <thread>

using namespace Core;

uint64_t perft(Core::Position& pos, int depth) {
    if (depth == 0) return 1ULL;

    MoveList moves;
    generate_legal_moves(pos, moves);

    if (depth == 1) return moves.size();

    uint64_t nodes = 0;
    for (int i = 0; i < moves.size(); ++i) {
        UndoInfo ui;
        pos.make_move(moves[i], ui);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(moves[i], ui);
    }
    return nodes;
}

uint64_t perft_multithreaded(Core::Position& pos, int depth) {
    if (depth == 0) return 1ULL;

    MoveList moves;
    generate_legal_moves(pos, moves);

    if (depth == 1) return moves.size();

    std::vector<std::future<uint64_t>> futures;
    
    for (int i = 0; i < moves.size(); ++i) {
        Move m = moves[i];

        // clone per branch so threads dont step on each other
        Core::Position cloned_pos = pos;

        futures.push_back(std::async(std::launch::async, [cloned_pos, m, depth]() mutable {
            UndoInfo ui;
            cloned_pos.make_move(m, ui);

            return perft(cloned_pos, depth - 1);
        }));
    }

    uint64_t total_nodes = 0;
    for (auto& f : futures) {
        total_nodes += f.get();
    }

    return total_nodes;
}

void run_deep_debug(const std::string& fen, const std::string& name) {
    std::cout << "[DEBUG] Analyzing: " << name << "\n";
    Position pos;
    pos.setFromFEN(fen);
    MoveList moves;
    generate_legal_moves(pos, moves);

}

struct PerftTest {
    std::string name;
    std::string fen;
    int depth;
    uint64_t expected_nodes;
};

void run_perft_suite() {
    Core::Attacks::init();
    Core::Zobrist::init();
    std::cout << "[INFO] Legacy perft suite executed.\n";
}

int run_epd_test_suite(const std::string& filepath, int max_depth_limit) {
    Core::Attacks::init();
    Core::Zobrist::init();

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open EPD file: " << filepath << "\n";
        return 1;
    }

    unsigned int cores = std::thread::hardware_concurrency();

    std::cout << "=========================================================\n";
    std::cout << "    PRODUCTION PERFT SUITE (SINGLE VS MULTI-CORE)        \n";
    std::cout << "=========================================================\n";
    std::cout << "Detected Hardware Threads: " << cores << "\n";
    std::cout << "---------------------------------------------------------\n";

    std::string line;
    uint64_t total_nodes = 0;
    double total_time_st = 0.0;
    double total_time_mt = 0.0;
    int passed = 0;
    int failed = 0;
    int position_idx = 1;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        size_t fen_end = line.find(';');
        if (fen_end == std::string::npos) continue;

        std::string fen = line.substr(0, fen_end);
        fen.erase(fen.find_last_not_of(" \t") + 1);

        Core::Position pos;
        if (!pos.setFromFEN(fen)) continue;

        std::string commands = line.substr(fen_end);
        int target_depth = -1;
        uint64_t expected_nodes = 0;

        for (int d = 1; d <= max_depth_limit; ++d) {
            std::string search_str = "D" + std::to_string(d) + " ";
            size_t idx = commands.find(search_str);
            if (idx != std::string::npos) {
                target_depth = d;
                expected_nodes = std::stoull(commands.substr(idx + search_str.length()));
            }
        }

        if (target_depth == -1) continue;

        std::cout << "Pos " << position_idx << " (Depth " << target_depth << ")\n";
        std::cout << "  Expected: " << expected_nodes << "\n";

        auto start_st = std::chrono::high_resolution_clock::now();
        uint64_t nodes_st = perft(pos, target_depth);
        auto end_st = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_st = end_st - start_st;
        double st_sec = elapsed_st.count();
        total_time_st += st_sec;

        uint64_t nps_st = static_cast<uint64_t>(nodes_st / st_sec);
        std::cout << "  Single  : " << nodes_st << " nodes | " << std::fixed << std::setprecision(3) << st_sec << "s | " << (nps_st / 1000000) << " MNPS\n";

        auto start_mt = std::chrono::high_resolution_clock::now();
        uint64_t nodes_mt = perft_multithreaded(pos, target_depth);
        auto end_mt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_mt = end_mt - start_mt;
        double mt_sec = elapsed_mt.count();
        total_time_mt += mt_sec;

        uint64_t nps_mt = static_cast<uint64_t>(nodes_mt / mt_sec);
        std::cout << "  Multi   : " << nodes_mt << " nodes | " << std::fixed << std::setprecision(3) << mt_sec << "s | " << (nps_mt / 1000000) << " MNPS";

        if (nodes_mt == expected_nodes && nodes_st == expected_nodes) {
            double speedup = st_sec / mt_sec;
            std::cout << "  [PASS] (Speedup: " << std::setprecision(2) << speedup << "x)\n";
            passed++;
        } else {
            std::cout << "  [FAIL]\n";
            failed++;
        }

        std::cout << "---------------------------------------------------------\n";
        total_nodes += nodes_mt;
        position_idx++;
    }

    uint64_t global_nps_st = static_cast<uint64_t>(total_nodes / total_time_st);
    uint64_t global_nps_mt = static_cast<uint64_t>(total_nodes / total_time_mt);
    double global_speedup = total_time_st / total_time_mt;

    std::cout << "RESULTS: " << passed << " Passed, " << failed << " Failed\n";
    std::cout << "TOTAL NODES : " << total_nodes << "\n";
    std::cout << "SINGLE CORE : " << std::fixed << std::setprecision(3) << total_time_st << " seconds (" << (global_nps_st / 1000000) << " MNPS)\n";
    std::cout << "MULTI CORE  : " << std::fixed << std::setprecision(3) << total_time_mt << " seconds (" << (global_nps_mt / 1000000) << " MNPS)\n";
    std::cout << "SPEEDUP     : " << std::setprecision(2) << global_speedup << "x Faster\n";
    std::cout << "=========================================================\n";

    return (failed == 0) ? 0 : 1;
}
