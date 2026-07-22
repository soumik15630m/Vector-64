#include "uci.h"

#include "../cores/attacks.h"
#include "../cores/movegen.h"
#include "../cores/position.h"
#include "../cores/zobrist.h"
#include "../search/search.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace UCI {
namespace {
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;

constexpr int MAX_DEPTH = 64;
constexpr const char *STANDARD_STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct GoParams {
  bool ponder = false;
  bool infinite = false;
  int depth = -1;
  int movetimeMs = -1;
  int wtimeMs = -1;
  int btimeMs = -1;
  int wincMs = 0;
  int bincMs = 0;
  int movesToGo = 0;
  uint64_t nodes = 0;
  std::vector<std::string> searchMoves;
};

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

std::string trim_copy(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
    s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s.pop_back();
  return s;
}

std::vector<std::string> split_ws(const std::string &line) {
  std::vector<std::string> tokens;
  std::istringstream iss(line);
  for (std::string tok; iss >> tok;)
    tokens.push_back(tok);
  return tokens;
}

bool parse_square(const std::string &s, Core::Square &out) {
  if (s.size() != 2)
    return false;
  char f = static_cast<char>(std::tolower(static_cast<unsigned char>(s[0])));
  char r = s[1];
  if (f < 'a' || f > 'h')
    return false;
  if (r < '1' || r > '8')
    return false;
  out = Core::make_square(static_cast<Core::GenFile>(f - 'a'),
                          static_cast<Core::GenRank>(r - '1'));
  return true;
}

char promo_to_char(Core::PieceType pt) {
  switch (pt) {
  case Core::KNIGHT:
    return 'n';
  case Core::BISHOP:
    return 'b';
  case Core::ROOK:
    return 'r';
  default: // queen (and any non-underpromotion) prints as 'q'
    return 'q';
  }
}

std::string move_to_uci(Core::Move m) {
  if (!m.is_ok())
    return "0000";
  char out[6] = {0, 0, 0, 0, 0, 0};
  out[0] = static_cast<char>('a' + Core::file_of(m.from_sq()));
  out[1] = static_cast<char>('1' + Core::rank_of(m.from_sq()));
  out[2] = static_cast<char>('a' + Core::file_of(m.to_sq()));
  out[3] = static_cast<char>('1' + Core::rank_of(m.to_sq()));
  if (m.is_promotion()) {
    out[4] = promo_to_char(m.promotion_type());
    return std::string(out, out + 5);
  }
  return std::string(out, out + 4);
}

bool move_matches_uci(Core::Move m, const std::string &uci) {
  if (uci.size() < 4 || uci.size() > 5)
    return false;
  Core::Square from = Core::SQ_NONE;
  Core::Square to = Core::SQ_NONE;
  if (!parse_square(uci.substr(0, 2), from))
    return false;
  if (!parse_square(uci.substr(2, 2), to))
    return false;
  if (m.from_sq() != from || m.to_sq() != to)
    return false;
  if (uci.size() == 5) {
    if (!m.is_promotion())
      return false;
    const char p =
        static_cast<char>(std::tolower(static_cast<unsigned char>(uci[4])));
    return promo_to_char(m.promotion_type()) == p;
  }
  return !m.is_promotion();
}

bool looks_like_move(const std::string &token) {
  return token.size() >= 4 && token.size() <= 5 && token[0] >= 'a' &&
         token[0] <= 'h' && token[1] >= '1' && token[1] <= '8' &&
         token[2] >= 'a' && token[2] <= 'h' && token[3] >= '1' &&
         token[3] <= '8';
}

bool parse_int(const std::string &token, int &out) {
  try {
    size_t consumed = 0;
    const int value = std::stoi(token, &consumed);
    if (consumed != token.size())
      return false;
    out = value;
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_u64(const std::string &token, uint64_t &out) {
  try {
    size_t consumed = 0;
    const uint64_t value = std::stoull(token, &consumed);
    if (consumed != token.size())
      return false;
    out = value;
    return true;
  } catch (...) {
    return false;
  }
}

class EngineUci {
public:
  EngineUci() {
    const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
    threads_ = static_cast<int>(hc);
    position_.setFromFEN(STANDARD_STARTPOS_FEN);
    search_.set_hash_mb(static_cast<size_t>(hashMb_));
    search_.set_threads(threads_);
  }

  ~EngineUci() { stop_and_join(true); }

  int loop() {
    for (std::string line; std::getline(std::cin, line);) {
      if (!handle_command(line))
        return 0;
    }
    stop_and_join(true);
    return 0;
  }

private:
  bool handle_command(const std::string &line) {
    std::vector<std::string> tokens = split_ws(line);
    if (tokens.empty())
      return true;

    const std::string cmd = to_lower(tokens[0]);
    if (cmd == "uci") {
      handle_uci();
      return true;
    }
    if (cmd == "isready") {
      emit("readyok");
      return true;
    }
    if (cmd == "setoption") {
      handle_setoption(line);
      return true;
    }
    if (cmd == "ucinewgame") {
      stop_and_join(true);
      std::lock_guard<std::mutex> lock(positionMu_);
      position_.setFromFEN(STANDARD_STARTPOS_FEN);
      search_.clear();
      return true;
    }
    if (cmd == "position") {
      handle_position(tokens);
      return true;
    }
    if (cmd == "eval") {
      std::lock_guard<std::mutex> lock(positionMu_);
      int score = search_.evaluate(position_);
      emit("info string evaluation score: " + std::to_string(score) + " cp");
      return true;
    }
    if (cmd == "bench") {
      handle_bench(tokens);
      return true;
    }
    if (cmd == "go") {
      handle_go(tokens);
      return true;
    }
    if (cmd == "stop") {
      // Join so bestmove is guaranteed out before the next command.
      stop_and_join(false);
      return true;
    }
    if (cmd == "ponderhit") {
      // Ponder is not advertised; ignore gracefully.
      return true;
    }
    if (cmd == "quit") {
      stop_and_join(true);
      return false;
    }
    return true;
  }

  void handle_uci() {
    emit("id name STK-Vector-64");
    emit("id author Soumik");
    emit("option name Threads type spin default " + std::to_string(threads_) +
         " min 1 max 64");
    emit("option name Hash type spin default " + std::to_string(hashMb_) +
         " min 1 max 4096");
    emit("option name Move Overhead type spin default " +
         std::to_string(moveOverheadMs_) + " min 0 max 500");
    emit("option name EvalFile type string default <empty>");
    emit("option name EvalFileSmall type string default <empty>");
    emit("option name SmallNetThreshold type spin default 950 min 0 max 5000");
    emit("option name LazyEvalMargin type spin default 0 min 0 max 5000");
    emit("option name ShowStats type check default false");
    emit("uciok");
  }

  void handle_setoption(const std::string &line) {
    const std::string lower = to_lower(line);
    const size_t namePos = lower.find("name ");
    if (namePos == std::string::npos) {
      emit("info string setoption: missing name");
      return;
    }
    const size_t valuePos = lower.find(" value ");

    std::string name;
    std::string value;
    if (valuePos == std::string::npos) {
      name = line.substr(namePos + 5);
    } else {
      name = line.substr(namePos + 5, valuePos - (namePos + 5));
      value = line.substr(valuePos + 7);
    }

    name = to_lower(trim_copy(name));
    value = trim_copy(value);

    if (name == "threads") {
      int parsed = threads_;
      if (!parse_int(value, parsed)) {
        emit("info string setoption Threads: invalid value '" + value + "'");
        return;
      }
      threads_ = std::clamp(parsed, 1, 64);
      stop_and_join(true);
      search_.set_threads(threads_);
      return;
    }

    if (name == "hash") {
      int parsed = hashMb_;
      if (!parse_int(value, parsed)) {
        emit("info string setoption Hash: invalid value '" + value + "'");
        return;
      }
      hashMb_ = std::clamp(parsed, 1, 4096);
      stop_and_join(true);
      search_.set_hash_mb(static_cast<size_t>(hashMb_));
      return;
    }

    if (name == "move overhead") {
      int parsed = moveOverheadMs_;
      if (!parse_int(value, parsed)) {
        emit("info string setoption Move Overhead: invalid value '" + value +
             "'");
        return;
      }
      moveOverheadMs_ = std::clamp(parsed, 0, 500);
      return;
    }

    if (name == "smallnetthreshold") {
      int parsed = 950;
      if (!parse_int(value, parsed)) {
        emit("info string setoption SmallNetThreshold: invalid value '" +
             value + "'");
        return;
      }
      stop_and_join(true);
      search_.set_small_net_threshold(parsed);
      return;
    }

    if (name == "lazyevalmargin") {
      int parsed = 0;
      if (!parse_int(value, parsed)) {
        emit("info string setoption LazyEvalMargin: invalid value '" + value +
             "'");
        return;
      }
      stop_and_join(true);
      search_.set_lazy_eval_margin(parsed);
      return;
    }

    if (name == "showstats") {
      statsInfo_ = (to_lower(value) == "true");
      return;
    }

    if (name == "evalfile") {
      std::string path = value;
      if (path.size() >= 2 && path.front() == '"' && path.back() == '"') {
        path = path.substr(1, path.size() - 2);
      }

      if (path.empty() || to_lower(path) == "<empty>") {
        emit("info string EvalFile ignored: empty path");
        return;
      }

      stop_and_join(true);
      if (search_.load_nnue(path)) {
        emit("info string EvalFile loaded: " + path);
      } else {
        emit("info string EvalFile load failed: " + path);
      }
      return;
    }

    if (name == "evalfilesmall") {
      std::string path = value;
      if (path.size() >= 2 && path.front() == '"' && path.back() == '"') {
        path = path.substr(1, path.size() - 2);
      }

      if (path.empty() || to_lower(path) == "<empty>") {
        emit("info string EvalFileSmall ignored: empty path");
        return;
      }

      stop_and_join(true);
      if (search_.load_nnue_small(path)) {
        emit("info string EvalFileSmall loaded: " + path);
      } else {
        emit("info string EvalFileSmall load failed: " + path);
      }
      return;
    }

    emit("info string unknown option: " + name);
  }

  void handle_position(const std::vector<std::string> &tokens) {
    stop_and_join(true);

    if (tokens.size() < 2)
      return;
    Core::Position next;
    size_t i = 1;

    if (tokens[i] == "startpos") {
      next.setFromFEN(STANDARD_STARTPOS_FEN);
      ++i;
    } else if (tokens[i] == "fen") {
      ++i;
      std::string fen;
      int parts = 0;
      while (i < tokens.size() && tokens[i] != "moves" && parts < 6) {
        if (!fen.empty())
          fen += " ";
        fen += tokens[i++];
        ++parts;
      }
      if (parts < 4 || !next.setFromFEN(fen)) {
        emit("info string invalid fen in position command");
        return;
      }
      if (next.opponent_in_check()) {
        emit("info string illegal fen: side not to move is in check");
        return;
      }
    } else {
      emit("info string invalid position command");
      return;
    }

    if (i < tokens.size() && tokens[i] == "moves")
      ++i;
    for (; i < tokens.size(); ++i) {
      Core::MoveList legal;
      Core::generate_legal_moves(next, legal);

      bool found = false;
      Core::Move chosen;
      for (int k = 0; k < legal.size(); ++k) {
        if (move_matches_uci(legal[k], tokens[i])) {
          chosen = legal[k];
          found = true;
          break;
        }
      }

      if (!found) {
        emit("info string illegal move in position command: " + tokens[i]);
        return;
      }

      Core::UndoInfo undo{};
      next.make_move(chosen, undo);
    }

    std::lock_guard<std::mutex> lock(positionMu_);
    position_ = next;
  }

  void handle_go(const std::vector<std::string> &tokens) {
    GoParams params;
    for (size_t i = 1; i < tokens.size(); ++i) {
      const std::string &key = tokens[i];
      int parsed = 0;
      uint64_t parsedU64 = 0;

      if (key == "ponder")
        params.ponder = true;
      else if (key == "infinite")
        params.infinite = true;
      else if (key == "searchmoves") {
        while (i + 1 < tokens.size() && looks_like_move(tokens[i + 1])) {
          params.searchMoves.push_back(tokens[i + 1]);
          ++i;
        }
      } else if (key == "depth" && i + 1 < tokens.size() &&
                 parse_int(tokens[i + 1], parsed)) {
        params.depth = std::max(1, parsed);
        ++i;
      } else if (key == "movetime" && i + 1 < tokens.size() &&
                 parse_int(tokens[i + 1], parsed)) {
        params.movetimeMs = std::max(1, parsed);
        ++i;
      } else if (key == "wtime" && i + 1 < tokens.size() &&
                 parse_int(tokens[i + 1], parsed)) {
        params.wtimeMs = std::max(0, parsed);
        ++i;
      } else if (key == "btime" && i + 1 < tokens.size() &&
                 parse_int(tokens[i + 1], parsed)) {
        params.btimeMs = std::max(0, parsed);
        ++i;
      } else if (key == "winc" && i + 1 < tokens.size() &&
                 parse_int(tokens[i + 1], parsed)) {
        params.wincMs = std::max(0, parsed);
        ++i;
      } else if (key == "binc" && i + 1 < tokens.size() &&
                 parse_int(tokens[i + 1], parsed)) {
        params.bincMs = std::max(0, parsed);
        ++i;
      } else if (key == "movestogo" && i + 1 < tokens.size() &&
                 parse_int(tokens[i + 1], parsed)) {
        params.movesToGo = std::max(0, parsed);
        ++i;
      } else if (key == "nodes" && i + 1 < tokens.size() &&
                 parse_u64(tokens[i + 1], parsedU64)) {
        params.nodes = parsedU64;
        ++i;
      }
    }

    start_search(params);
  }

  // Fixed-depth searches over a small position set; reports total
  // nodes and NPS so perf changes are measurable and comparable.
  void handle_bench(const std::vector<std::string> &tokens) {
    stop_and_join(true);

    int depth = 10;
    if (tokens.size() > 1) {
      int parsed = 0;
      if (parse_int(tokens[1], parsed))
        depth = std::clamp(parsed, 1, MAX_DEPTH);
    }

    // clang-format off: keep each FEN on a single line so adjacent string
    // literals are never implicitly concatenated (bugprone-suspicious-missing-comma).
    // clang-format off
    static const char *BENCH_FENS[] = {
        STANDARD_STARTPOS_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "r1bq1rk1/pp2bppp/2n2n2/2pp4/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 8",
        "8/8/1p1k4/p1p2p2/P1P2P2/1P1K4/8/8 w - - 0 1",
    };
    // clang-format on

    search_.clear();
    uint64_t totalNodes = 0;
    const auto start = Clock::now();

    for (const char *fen : BENCH_FENS) {
      Core::Position pos;
      if (!pos.setFromFEN(fen))
        continue;

      Search::Limits limits;
      limits.maxDepth = depth;

      Search::Callbacks callbacks;
      const Search::Result result = search_.search(pos, limits, callbacks);
      totalNodes += result.nodes;
    }

    const auto elapsed =
        std::chrono::duration_cast<Ms>(Clock::now() - start).count();
    const uint64_t nps =
        elapsed > 0 ? (totalNodes * 1000ULL) / static_cast<uint64_t>(elapsed)
                    : 0ULL;

    emit("info string bench depth " + std::to_string(depth) + " nodes " +
         std::to_string(totalNodes) + " time " + std::to_string(elapsed) +
         "ms" + " nps " + std::to_string(nps));
  }

  void start_search(const GoParams &params) {
    std::lock_guard<std::mutex> lock(searchMu_);

    const uint64_t searchId =
        activeSearchId_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (searchThread_.joinable()) {
      stopRequested_.store(true, std::memory_order_relaxed);
      searchThread_.join();
    }

    stopRequested_.store(false, std::memory_order_relaxed);
    searchThread_ = std::thread([this, searchId, params]() {
      // A thread entry must not let an exception escape (it would call
      // std::terminate); report and exit the worker instead.
      try {
        search_worker(searchId, params);
      } catch (const std::exception &e) {
        emit(std::string("info string search error: ") + e.what());
      } catch (...) {
        emit("info string search error: unknown");
      }
    });
  }

  void stop_and_join(bool suppressOutput) {
    std::lock_guard<std::mutex> lock(searchMu_);
    if (!searchThread_.joinable())
      return;

    if (suppressOutput)
      activeSearchId_.fetch_add(1, std::memory_order_relaxed);
    stopRequested_.store(true, std::memory_order_relaxed);
    searchThread_.join();
    stopRequested_.store(false, std::memory_order_relaxed);
  }

  void emit_info(const Search::IterInfo &info) {
    const uint64_t nps =
        info.elapsedMs > 0
            ? (info.nodes * 1000ULL) / static_cast<uint64_t>(info.elapsedMs)
            : 0ULL;

    std::ostringstream os;
    os << "info depth " << info.depth << " seldepth " << info.seldepth;

    if (Search::is_mate_score(info.scoreCp)) {
      os << " score mate " << Search::mate_in_moves(info.scoreCp);
    } else {
      os << " score cp " << info.scoreCp;
    }

    os << " nodes " << info.nodes << " nps " << nps << " time "
       << info.elapsedMs << " pv";
    for (int i = 0; i < info.pvLen; ++i) {
      os << ' ' << move_to_uci(info.pv[i]);
    }
    emit(os.str());

    if (statsInfo_) {
      std::ostringstream stats;
      stats << "info string tt hit rate qsearch " << info.qsearchTtHitRate
            << "% negamax " << info.negamaxTtHitRate << "%";
      emit(stats.str());
    }
  }

  void emit_bestmove(uint64_t searchId, Core::Move bestMove) {
    if (searchId != activeSearchId_.load(std::memory_order_relaxed))
      return;
    emit("bestmove " + move_to_uci(bestMove));
  }

  void search_worker(uint64_t searchId, const GoParams &params) {
    Core::Position root;
    {
      std::lock_guard<std::mutex> lock(positionMu_);
      root = position_;
    }
    int optimumTimeMs = -1;
    int maximumTimeMs = -1;

    if (params.movetimeMs > 0) {
      maximumTimeMs = std::max(1, params.movetimeMs - moveOverheadMs_);
      optimumTimeMs = maximumTimeMs;
    } else if (!params.infinite && !params.ponder) {
      const bool whiteToMove = root.side_to_move() == Core::WHITE;
      const int clockMs = whiteToMove ? params.wtimeMs : params.btimeMs;
      const bool hasClock = clockMs >= 0;

      if (hasClock) {
        const long long remain = std::max(0, clockMs);
        const long long inc =
            std::max(0, whiteToMove ? params.wincMs : params.bincMs);
        const long long overhead = moveOverheadMs_;

        // The increment arrives after we move, so the hard
        // ceiling is the clock itself minus overhead. Even a
        // zero clock keeps a 1ms deadline: never search unbounded.
        const long long hardBudget = std::max(1LL, remain - overhead);
        const int mtg = params.movesToGo > 0 ? params.movesToGo : 40;

        long long optimum = remain / mtg + (inc * 3) / 4 - overhead;
        long long maximum =
            (params.movesToGo == 1)
                // Last move before a new time control: the clock refills.
                ? hardBudget
                : remain / 5 + (inc * 3) / 4 - overhead;

        maximum = std::clamp(maximum, 1LL, hardBudget);
        optimum = std::clamp(optimum, 1LL, maximum);

        optimumTimeMs = static_cast<int>(optimum);
        maximumTimeMs = static_cast<int>(maximum);
      }
    }

    Search::Limits searchLimits;
    searchLimits.maxDepth =
        params.depth > 0 ? std::min(params.depth, MAX_DEPTH) : MAX_DEPTH;
    searchLimits.maxNodes = params.nodes;
    searchLimits.hasDeadline = false;

    if (maximumTimeMs > 0) {
      searchLimits.hasDeadline = true;
      searchLimits.deadline = Clock::now() + Ms(maximumTimeMs);
    }

    if (!params.searchMoves.empty()) {
      Core::MoveList legal;
      Core::generate_legal_moves(root, legal);
      for (const std::string &uci : params.searchMoves) {
        for (int k = 0; k < legal.size(); ++k) {
          if (move_matches_uci(legal[k], uci)) {
            searchLimits.searchMoves.push_back(legal[k]);
            break;
          }
        }
      }
    }

    std::atomic<bool> softStop{false};

    // Dynamic time management, driven per completed iteration:
    //   * extend the budget when the score drops (avoid horizon blunders), and
    //   * scale it down when the best move has been stable for several
    //     iterations (we are already confident) or up when it keeps changing.
    int previousScore = 0;
    bool firstIteration = true;
    int currentOptimumMs = optimumTimeMs;
    uint16_t lastBest = 0;
    int bestStableIters = 0;

    Search::Callbacks callbacks;
    callbacks.shouldStop = [this, searchId, &softStop]() {
      return softStop.load(std::memory_order_relaxed) ||
             stopRequested_.load(std::memory_order_relaxed) ||
             searchId != activeSearchId_.load(std::memory_order_relaxed);
    };

    callbacks.onInfo = [this, searchId, &softStop, &previousScore,
                        &firstIteration, &currentOptimumMs, &lastBest,
                        &bestStableIters,
                        maximumTimeMs](const Search::IterInfo &info) {
      if (searchId != activeSearchId_.load(std::memory_order_relaxed))
        return;
      emit_info(info);

      if (!firstIteration && currentOptimumMs > 0 &&
          !Search::is_mate_score(info.scoreCp) &&
          !Search::is_mate_score(previousScore)) {
        // Score drop => extend, to avoid walking into a tactic at the horizon.
        if (info.scoreCp < previousScore - 30) {
          currentOptimumMs =
              std::min(maximumTimeMs, static_cast<int>(currentOptimumMs * 1.5));
        }
      }

      // Best-move stability: count consecutive iterations with the same root
      // move, then scale the effective budget -- less time when settled,
      // slightly more when the move keeps flipping.
      const uint16_t best = info.pvLen > 0 ? info.pv[0].raw() : 0;
      bestStableIters = (best == lastBest) ? bestStableIters + 1 : 0;
      lastBest = best;

      previousScore = info.scoreCp;
      firstIteration = false;

      if (currentOptimumMs > 0) {
        const double factor =
            std::clamp(1.30 - 0.11 * bestStableIters, 0.45, 1.30);
        const int budget = std::min(
            maximumTimeMs, static_cast<int>(currentOptimumMs * factor));
        if (info.elapsedMs >= budget)
          softStop.store(true, std::memory_order_relaxed);
      }
    };

    const Search::Result result = search_.search(root, searchLimits, callbacks);
    emit_bestmove(searchId, result.bestMove);
  }

  void emit(const std::string &line) {
    std::lock_guard<std::mutex> lock(ioMu_);
    std::cout << line << '\n' << std::flush;
  }

  std::mutex ioMu_;
  std::mutex positionMu_;
  std::mutex searchMu_;

  Core::Position position_;

  std::atomic<bool> stopRequested_{false};
  std::atomic<uint64_t> activeSearchId_{0};
  std::thread searchThread_;

  int threads_ = 1;
  // Default sized to sit inside the shared L3 cache alongside the
  // working set; raise via the Hash option for long time controls.
  int hashMb_ = 8;
  int moveOverheadMs_ = 30;
  bool statsInfo_ = false;
  Search::EngineSearch search_{8};
};
} // namespace

int run() {
  Core::Attacks::init();
  Core::Zobrist::init();

  EngineUci app;
  return app.loop();
}
} // namespace UCI
