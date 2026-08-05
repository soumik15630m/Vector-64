#include "datagen/datagen.h"

#include "datagen/selfplay.h"

#include "cores/attacks.h"
#include "cores/bitboard.h"
#include "cores/movegen.h"
#include "cores/position.h"
#include "cores/types.h"
#include "cores/zobrist.h"
#include "search/search.h"
#include "search/transposition_table.h" // is_mate_score

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Datagen {
namespace {

// Search settings match the visualizer's shipped defaults, which follow what
// Stockfish uses for foundational NNUE data: depth 9 under a 5000-node ceiling.
struct Params {
  std::string net;
  std::string out;
  int games = 20000;
  int nodes = 5000;
  int depth = 9;
  int threads = 8;
  int hashMb = 16;
  int maxPlies = 200;
  int skipPlies = 12;
  int openingPlies = 8;
  int balance = 150; // max |white-black material| for an opening (cp)
  double lam = 0.5;
  bool raw = true; // fen | eval | wdl (bullet-native); false = fen | cp (blend)
  uint64_t seed = 0;
  double logInterval = 30.0;
};

// START_FEN, MATE_CP, clamp_score, blend_cp, emit_row, insufficient_material
// and make_opening all live in selfplay.h now, shared verbatim with the
// visualizer's datagen mode so the two write identical rows.

struct Shared {
  std::ofstream out;
  std::mutex mu;
  std::atomic<int> started{0};
  std::atomic<int64_t> games{0}, positions{0}, w{0}, d{0}, l{0};
};

void worker(const Params &p, Shared &sh) {
  Search::EngineSearch search(size_t(p.hashMb));
  search.set_threads(1);
  search.set_persist_ordering(
      true); // warm history across a game's short searches
  if (!p.net.empty())
    search.load_nnue(p.net);

  Search::Callbacks cb;
  cb.shouldStop = [] { return false; };
  cb.onInfo = [](const Search::IterInfo &) {};
  Search::Limits limits;
  limits.maxNodes = uint64_t(p.nodes);
  if (p.depth > 0)
    limits.maxDepth = p.depth;

  std::vector<std::string> buf;
  auto flush = [&] {
    if (buf.empty())
      return;
    std::lock_guard<std::mutex> lk(sh.mu);
    for (const std::string &s : buf)
      sh.out << s << '\n';
    sh.positions.fetch_add(int64_t(buf.size()));
    buf.clear();
  };

  int gi = 0;
  while ((gi = sh.started.fetch_add(1)) < p.games) {
    std::mt19937_64 rng(p.seed + 0x9e3779b97f4a7c15ULL * uint64_t(gi + 1));
    Core::Position pos;
    int tries = 0;
    while (!make_opening(pos, rng, p.openingPlies, p.balance) && ++tries < 64) {
    }
    if (tries >= 64)
      continue;

    search.clear(); // fresh TT + ordering: games are independent
    std::unordered_map<uint64_t, int> seen;
    std::vector<std::pair<std::string, int>> rec;
    double wdl = 0.5;
    int plies = 0;
    while (true) {
      // Cheap draw checks first -- no move generation.
      if (++seen[pos.hash()] >= 3 || pos.halfmove_clock() >= 100 ||
          insufficient_material(pos) || plies >= p.maxPlies) {
        wdl = 0.5;
        break;
      }
      // The search generates its own root moves; a null best move means there
      // were none -> checkmate (in check) or stalemate. So no separate movegen.
      const Search::Result r = search.search(pos, limits, cb);
      if (!r.bestMove.is_ok()) {
        wdl = pos.in_check() ? (pos.side_to_move() == Core::WHITE ? 0.0 : 1.0)
                             : 0.5;
        break;
      }
      if (plies >= p.skipPlies && !pos.in_check()) {
        const int evalWhite =
            pos.side_to_move() == Core::WHITE ? r.scoreCp : -r.scoreCp;
        rec.emplace_back(pos.toFEN(), clamp_score(evalWhite));
      }
      Core::UndoInfo ui;
      pos.make_move(r.bestMove, ui);
      ++plies;
    }

    for (const auto &pr : rec)
      buf.push_back(emit_row(pr.first, pr.second, wdl, p.raw, p.lam));
    sh.games.fetch_add(1);
    (wdl == 1.0 ? sh.w : (wdl == 0.0 ? sh.l : sh.d)).fetch_add(1);
    if (buf.size() >= 2000)
      flush();
  }
  flush();
}

std::string next_arg(int argc, char **argv, int &i) {
  return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
}

} // namespace

int run(int argc, char **argv) {
  Core::Attacks::init();
  Core::Zobrist::init();

  Params p;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--net")
      p.net = next_arg(argc, argv, i);
    else if (a == "--out")
      p.out = next_arg(argc, argv, i);
    else if (a == "--games")
      p.games = std::stoi(next_arg(argc, argv, i));
    else if (a == "--nodes")
      p.nodes = std::stoi(next_arg(argc, argv, i));
    else if (a == "--depth")
      p.depth = std::stoi(next_arg(argc, argv, i));
    else if (a == "--threads" || a == "--concurrency")
      p.threads = std::stoi(next_arg(argc, argv, i));
    else if (a == "--hash")
      p.hashMb = std::stoi(next_arg(argc, argv, i));
    else if (a == "--max-plies")
      p.maxPlies = std::stoi(next_arg(argc, argv, i));
    else if (a == "--skip-plies")
      p.skipPlies = std::stoi(next_arg(argc, argv, i));
    else if (a == "--opening-plies")
      p.openingPlies = std::stoi(next_arg(argc, argv, i));
    else if (a == "--balance")
      p.balance = std::stoi(next_arg(argc, argv, i));
    else if (a == "--lam")
      p.lam = std::stod(next_arg(argc, argv, i));
    else if (a == "--seed")
      p.seed = std::stoull(next_arg(argc, argv, i));
    else if (a == "--log-interval")
      p.logInterval = std::stod(next_arg(argc, argv, i));
    else if (a == "--emit")
      p.raw = next_arg(argc, argv, i) != "blend";
    else {
      std::cerr << "datagen: unknown arg '" << a << "'\n";
      return 2;
    }
  }
  if (p.out.empty()) {
    std::cerr << "datagen: --out <file> is required\n";
    return 2;
  }
  if (p.threads < 1)
    p.threads = 1;

  Shared sh;
  sh.out.open(p.out, std::ios::binary);
  if (!sh.out) {
    std::cerr << "datagen: cannot open output '" << p.out << "'\n";
    return 2;
  }
  if (p.net.empty())
    std::cerr << "datagen: WARNING no --net given; using classical eval\n";

  std::cout << "[datagen] " << p.games << " games @ " << p.nodes
            << " nodes / depth " << p.depth << ", " << p.threads
            << " threads, emit=" << (p.raw ? "raw" : "blend")
            << (p.net.empty() ? "" : (", net=" + p.net)) << '\n'
            << std::flush;

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> workers;
  workers.reserve(size_t(p.threads));
  for (int i = 0; i < p.threads; ++i)
    workers.emplace_back(worker, std::cref(p), std::ref(sh));

  std::atomic<bool> done{false};
  std::thread joiner([&] {
    for (auto &t : workers)
      t.join();
    done.store(true);
  });

  double lastPrint = 0;
  while (!done.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const double el =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    if (el - lastPrint >= p.logInterval) {
      lastPrint = el;
      const int64_t g = sh.games.load(), pos = sh.positions.load();
      const double gps = g / std::max(el, 1e-9);
      const double eta = (p.games - g) / std::max(gps, 1e-9) / 60.0;
      std::cout << "  " << g << "/" << p.games << "  " << pos << " pos  "
                << int(gps) << " g/s  " << int(pos / std::max(el, 1e-9))
                << " pos/s  W/D/L " << sh.w.load() << "/" << sh.d.load() << "/"
                << sh.l.load() << "  eta " << eta << "m" << '\n'
                << std::flush;
    }
  }
  joiner.join();
  sh.out.flush();
  sh.out.close();

  const double el =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  const int64_t g = std::max<int64_t>(sh.games.load(), 1);
  std::cout << "DATAGEN DONE  " << sh.games.load() << " games  "
            << sh.positions.load() << " positions  ("
            << double(sh.positions.load()) / double(g) << " pos/game)  "
            << el / 60.0 << " min (" << g / std::max(el, 1e-9) << " g/s)\n"
            << "  result (white pov): " << (100 * sh.w.load() / g) << "% W  "
            << (100 * sh.d.load() / g) << "% D  " << (100 * sh.l.load() / g)
            << "% L  ->  " << p.out << '\n'
            << std::flush;
  return 0;
}

} // namespace Datagen
