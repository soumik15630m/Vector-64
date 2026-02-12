#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>
#include <charconv>
#include <cstdint>
#include "tests/perfts.h"
#include "tests/nnue_consistency.h"
#include "src/uci/uci.h"

namespace {
    std::string resolve_epd_path(const std::string& explicitPath, const char* argv0) {
        namespace fs = std::filesystem;

        std::vector<fs::path> candidates;
        if (!explicitPath.empty()) {
            candidates.emplace_back(explicitPath);
        } else {
            candidates.emplace_back("test_data/standard.epd");
            candidates.emplace_back("../test_data/standard.epd");
            candidates.emplace_back("../../test_data/standard.epd");

            std::error_code ec;
            fs::path exePath = fs::absolute(fs::path(argv0), ec);
            if (!ec) {
                fs::path probe = exePath.parent_path();
                for (int i = 0; i < 8 && !probe.empty(); ++i) {
                    candidates.push_back(probe / "test_data" / "standard.epd");
                    const fs::path parent = probe.parent_path();
                    if (parent == probe) break;
                    probe = parent;
                }
            }
        }

        std::error_code ec;
        for (const fs::path& p : candidates) {
            if (!p.empty() && fs::exists(p, ec)) {
                const fs::path normalized = fs::weakly_canonical(p, ec);
                if (!ec) return normalized.string();
                return p.string();
            }
            ec.clear();
        }
        return "";
    }

    bool parse_positive_int(const std::string& token, int& out) {
        int value = 0;
        const char* begin = token.data();
        const char* end = token.data() + token.size();
        const auto [ptr, ec] = std::from_chars(begin, end, value);
        if (ec != std::errc() || ptr != end || value <= 0) return false;
        out = value;
        return true;
    }

    bool parse_u64(const std::string& token, uint64_t& out) {
        uint64_t value = 0;
        const char* begin = token.data();
        const char* end = token.data() + token.size();
        const auto [ptr, ec] = std::from_chars(begin, end, value);
        if (ec != std::errc() || ptr != end) return false;
        out = value;
        return true;
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--nnue-consistency") {
        int games = 32;
        int maxPlies = 80;
        uint64_t seed = 0xC0FFEEULL;

        if (argc > 2 && !parse_positive_int(argv[2], games)) {
            std::cerr << "[ERROR] Invalid games value for --nnue-consistency.\n";
            return 1;
        }
        if (argc > 3 && !parse_positive_int(argv[3], maxPlies)) {
            std::cerr << "[ERROR] Invalid max-plies value for --nnue-consistency.\n";
            return 1;
        }
        if (argc > 4 && !parse_u64(argv[4], seed)) {
            std::cerr << "[ERROR] Invalid seed value for --nnue-consistency.\n";
            return 1;
        }
        if (argc > 5) {
            std::cerr
                << "[ERROR] Usage: --nnue-consistency [games] [max-plies] [seed]\n";
            return 1;
        }

        return run_nnue_incremental_consistency_test(games, maxPlies, seed);
    }

    if (argc > 1 && std::string(argv[1]) == "--perft") {
        const std::string requested = argc > 2 ? std::string(argv[2]) : "";
        const std::string epdPath = resolve_epd_path(requested, argv[0]);
        if (epdPath.empty()) {
            std::cerr
                << "[ERROR] Could not locate EPD file. "
                << "Provide one explicitly with: --perft <path-to-standard.epd>\n";
            return 1;
        }

        std::cout << "Starting Chess Engine Core...\n";
        return run_epd_test_suite(epdPath, 5);
    }

    return UCI::run();
}
