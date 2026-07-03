#include "perfts.h"
#include "uci/uci.h"
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

namespace {
std::string resolve_epd_path(const std::string &explicitPath,
                             const char *argv0) {
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
        if (parent == probe)
          break;
        probe = parent;
      }
    }
  }

  std::error_code ec;
  for (const fs::path &p : candidates) {
    if (!p.empty() && fs::exists(p, ec)) {
      const fs::path normalized = fs::weakly_canonical(p, ec);
      if (!ec)
        return normalized.string();
      return p.string();
    }
    ec.clear();
  }
  return "";
}

} // namespace

int main(int argc, char **argv) {
  // A single top-level guard so no exception escapes main (which would call
  // std::terminate and lose the diagnostic).
  try {
    if (argc >= 5 && std::string(argv[1]) == "--perft-one") {
      // --perft-one "<fen>" <depth> <expected-nodes>
      try {
        return run_perft_one(argv[2], std::stoi(argv[3]),
                             static_cast<uint64_t>(std::stoull(argv[4])));
      } catch (...) {
        std::cerr
            << "[ERROR] Usage: --perft-one \"<fen>\" <depth> <expected>\n";
        return 2;
      }
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
  } catch (const std::exception &e) {
    std::cerr << "[FATAL] " << e.what() << '\n';
    return 2;
  } catch (...) {
    std::cerr << "[FATAL] unknown exception\n";
    return 2;
  }
}
