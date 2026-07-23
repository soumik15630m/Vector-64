#include "datagen/datagen.h"

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

constexpr const char *START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr int MATE_CP = 8000;

struct Params {
  std::string net;
  std::string out;
  int games = 20000;
  int nodes = 6000;
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

// Search score -> a bounded white-perspective label; mates map to +/-MATE_CP
// (mirrors tools/nnue/datagen.py so the two generators agree).
int clamp_score(int cp) {
  if (Search::is_mate_score(cp))
    return cp > 0 ? MATE_CP : -MATE_CP;
  return std::max(-MATE_CP, std::min(MATE_CP, cp));
}

// WDL blend in win-probability space (matches datagen.py blend_cp, CP_SCALE
// 400).
int blend_cp(int evalWhite, double wdl, double lam) {
  const double e = std::max(-4000.0, std::min(4000.0, double(evalWhite)));
  const double pe = 1.0 / (1.0 + std::exp(-e / 400.0));
  double p = (1.0 - lam) * pe + lam * wdl;
  p = std::min(std::max(p, 1e-4), 1.0 - 1e-4);
  return int(std::lround(400.0 * std::log(p / (1.0 - p))));
}

bool insufficient_material(const Core::Position &pos) {
  if (pos.pieces(Core::PAWN) || pos.pieces(Core::ROOK) ||
      pos.pieces(Core::QUEEN))
    return false;
  return Core::popcount(pos.pieces(Core::KNIGHT) | pos.pieces(Core::BISHOP)) <=
         1;
}

// Seeded quiet, material-balanced random opening. false => caller retries.
bool make_opening(Core::Position &pos, std::mt19937_64 &rng, const Params &p) {
  pos.setFromFEN(START_FEN);
  for (int i = 0; i < p.openingPlies; ++i) {
    Core::MoveList legal;
    Core::generate_legal_moves(pos, legal);
    if (legal.size() == 0)
      return false;
    Core::UndoInfo ui;
    pos.make_move(legal[int(rng() % uint64_t(legal.size()))], ui);
  }
  Core::MoveList legal;
  Core::generate_legal_moves(pos, legal);
  if (legal.size() == 0 || pos.in_check())
    return false;
  return std::abs(pos.material_wb()) <= p.balance;
}

struct Shared {
  std::ofstream out;
  std::mutex mu;
  std::atomic<int> started{0};
  std::atomic<int64_t> games{0}, positions{0}, w{0}, d{0}, l{0};
};

void worker(const Params &p, Shared &sh) {
  Search::EngineSearch search(size_t(p.hashMb));
  search.set_threads(1);
  if (!p.net.empty())
    search.load_nnue(p.net);

  Search::Callbacks cb;
  cb.shouldStop = [] { return false; };
  cb.onInfo = [](const Search::IterInfo &) {};
  Search::Limits limits;
  limits.maxNodes = uint64_t(p.nodes);

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
    while (!make_opening(pos, rng, p) && ++tries < 64) {
    }
    if (tries >= 64)
      continue;

    search.clear(); // fresh TT: games are independent
    std::unordered_map<uint64_t, int> seen;
    std::vector<std::pair<std::string, int>> rec;
    double wdl = 0.5;
    int plies = 0;
    while (true) {
      Core::MoveList legal;
      Core::generate_legal_moves(pos, legal);
      if (legal.size() == 0) { // checkmate or stalemate
        wdl = pos.in_check() ? (pos.side_to_move() == Core::WHITE ? 0.0 : 1.0)
                             : 0.5;
        break;
      }
      if (++seen[pos.hash()] >= 3 || pos.halfmove_clock() >= 100 ||
          insufficient_material(pos) || plies >= p.maxPlies) {
        wdl = 0.5;
        break;
      }
      const Search::Result r = search.search(pos, limits, cb);
      if (plies >= p.skipPlies && !pos.in_check()) {
        const int evalWhite =
            pos.side_to_move() == Core::WHITE ? r.scoreCp : -r.scoreCp;
        rec.emplace_back(pos.toFEN(), clamp_score(evalWhite));
      }
      Core::UndoInfo ui;
      pos.make_move(r.bestMove, ui);
      ++plies;
    }

    const char *ws = wdl == 1.0 ? "1.0" : (wdl == 0.0 ? "0.0" : "0.5");
    for (const auto &pr : rec) {
      if (p.raw)
        buf.push_back(pr.first + " | " + std::to_string(pr.second) + " | " +
                      ws);
      else
        buf.push_back(pr.first + " | " +
                      std::to_string(blend_cp(pr.second, wdl, p.lam)));
    }
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

  std::cout << "[datagen] " << p.games << " games @ " << p.nodes << " nodes, "
            << p.threads << " threads, emit=" << (p.raw ? "raw" : "blend")
            << (p.net.empty() ? "" : (", net=" + p.net)) << std::endl;

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
                << sh.l.load() << "  eta " << eta << "m" << std::endl;
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
            << "% L  ->  " << p.out << std::endl;
  return 0;
}

} // namespace Datagen
