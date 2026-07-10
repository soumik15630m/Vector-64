#ifndef SEARCH_SEARCH_H
#define SEARCH_SEARCH_H

#include "../cores/movegen.h"
#include "evaluator.h"
#include "move_ordering.h"
#include "transposition_table.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Search {

struct Limits {
  int maxDepth = 64;
  uint64_t maxNodes = 0;
  bool hasDeadline = false;
  std::chrono::steady_clock::time_point deadline{};
  Core::MoveList searchMoves{}; // empty = search all root moves
};

struct IterInfo {
  int depth = 0;
  int seldepth = 0;
  int scoreCp = 0; // internal score; mate range per is_mate_score()
  uint64_t nodes = 0;
  int elapsedMs = 0;
  const Core::Move *pv = nullptr;
  int pvLen = 0;
  double qsearchTtHitRate = 0.0;
  double negamaxTtHitRate = 0.0;
};

struct Callbacks {
  std::function<bool()> shouldStop;
  std::function<void(const IterInfo &)> onInfo;
};

struct Result {
  Core::Move bestMove = Core::Move::none();
  int scoreCp = 0;
  int completedDepth = 0;
  uint64_t nodes = 0;
};

class EngineSearch {
public:
  explicit EngineSearch(size_t hashMb = 8);

  void set_hash_mb(size_t hashMb);
  void set_threads(int threads);
  void clear();
  bool load_nnue(const std::string &path);
  int evaluate(const Core::Position &pos);

  size_t hash_mb() const { return hashMb_; }

  Result search(Core::Position root, const Limits &limits,
                const Callbacks &callbacks);

private:
  using Clock = std::chrono::steady_clock;

  // Helper-thread instance: shares the master's transposition table
  // and evaluator, owns everything else.
  EngineSearch(TranspositionTable *sharedTt, const Evaluator *sharedEval);

  Result search_internal(Core::Position &root, const Limits &limits,
                         const Callbacks &callbacks);
  uint64_t total_nodes() const;

  bool check_stop(const Limits &limits, const Callbacks &callbacks);
  void update_pv(int ply, Core::Move move);

  // Evaluate `pos` at `ply`, using the incrementally maintained accumulator
  // when a net is loaded, else the classical path.
  int eval_pos(const Core::Position &pos, int ply);

  int negamax(Core::Position &pos, int depth, int alpha, int beta, int ply,
              const Limits &limits, const Callbacks &callbacks);

  int quiescence(Core::Position &pos, int alpha, int beta, int ply,
                 const Limits &limits, const Callbacks &callbacks);

  int search_root(Core::Position &root, Core::MoveList &rootMoves, int depth,
                  int alpha, int beta, Core::Move prevBest,
                  const Limits &limits, const Callbacks &callbacks);

  size_t hashMb_ = 8;
  int threadCount_ = 1;
  uint64_t nodes_ = 0;
  int seldepth_ = 0;
  bool stopped_ = false;
  Clock::time_point started_{};

  uint64_t qProbes_ = 0;
  uint64_t qHits_ = 0;
  uint64_t nProbes_ = 0;
  uint64_t nHits_ = 0;

  Core::Move pvTable_[MAX_PLY + 1][MAX_PLY + 1];
  int pvLen_[MAX_PLY + 2] = {};

  TranspositionTable ownTt_;
  TranspositionTable *tt_; // ownTt_, or the master's table on helpers
  MoveOrdering ordering_;
  Evaluator evaluator_;
  const Evaluator *eval_; // evaluator_, or the master's on helpers

  // Per-thread NNUE accumulator stack, indexed by ply, plus the king-square
  // refresh cache. Active only while a net is loaded; the classical path
  // never touches them.
  std::vector<NNUE::Accumulator> accStack_;
  std::unique_ptr<NNUE::RefreshTable> refreshTable_;
  bool nnueActive_ = false;

  // Live only while a multi-threaded search runs; lets the master
  // aggregate helper node counts for reporting.
  std::vector<EngineSearch *> helperViews_;
};

} // namespace Search

#endif
