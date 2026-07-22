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
  uint64_t tbHits = 0;
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
  void set_small_net_threshold(int cp);
  void set_lazy_eval_margin(int cp);
  void clear();
  bool load_nnue(const std::string &path);
  bool load_nnue_small(const std::string &path);
  // Load Syzygy tablebases from a directory list; true if any were found.
  bool load_syzygy(const std::string &path);
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
  uint64_t total_tb_hits() const;

  // Evaluate `pos` at `ply` from the eagerly maintained big accumulator.
  // When the small net is loaded and the O(1) material+psqt estimate calls
  // the position clearly decided, the small net answers on demand instead
  // (a finny-cached refresh; no per-make maintenance), skipping the big
  // forward entirely at that node.
  int eval_pos(const Core::Position &pos, int ply);

  int negamax(Core::Position &pos, int depth, int alpha, int beta, int ply,
              const Limits &limits, const Callbacks &callbacks,
              Core::Move excludedMove = Core::Move::none());

  int quiescence(Core::Position &pos, int alpha, int beta, int ply,
                 const Limits &limits, const Callbacks &callbacks);

  int search_root(Core::Position &root, Core::MoveList &rootMoves, int depth,
                  int alpha, int beta, Core::Move prevBest,
                  const Limits &limits, const Callbacks &callbacks);

  size_t hashMb_ = 8;
  int threadCount_ = 1;
  int threadId_ = 0; // 0 = master; 1..N-1 = lazy-SMP helpers
  uint64_t nodes_ = 0;
  int seldepth_ = 0;
  bool stopped_ = false;
  Clock::time_point started_{};

  uint64_t qProbes_ = 0;
  uint64_t qHits_ = 0;
  uint64_t nProbes_ = 0;
  uint64_t nHits_ = 0;

  // Syzygy tablebases: cached from the global prober at search start. WDL
  // probing runs on every thread; the DTZ root probe is master-only.
  bool syzygyActive_ = false;
  int syzygyPieces_ = 0; // TB_LARGEST
  uint64_t tbHits_ = 0;

  Core::Move pvTable_[MAX_PLY + 1][MAX_PLY + 1];
  int pvLen_[MAX_PLY + 2] = {};

  // Search stack: the move made at each ply (piece type + destination), used
  // to key continuation history at the child. NO_PIECE_TYPE marks "no move"
  // (the root, or after a null move).
  Core::PieceType ssPiece_[MAX_PLY + 2] = {};
  Core::Square ssTo_[MAX_PLY + 2] = {};

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
  NNUE::SmallAccumulator smallScratch_;
  std::unique_ptr<NNUE::SmallRefreshTable> smallRefreshTable_;
  bool nnueActive_ = false;
  bool smallActive_ = false;
  // Dual-net gate: |material+psqt| above this (cp) takes the small-net eval.
  int smallNetThreshold_ = 950;
  // Lazy eval: |PSQT side-output| above this (cp) skips the big net's dense
  // layers. 0 = off (always full forward). Tuned via UCI / SPRT.
  int lazyEvalMargin_ = 0;

#ifdef ENGINE_PROF
  // Cycle attribution for the hot-path audit (rdtsc; build with -DENGINE_PROF).
  uint64_t profEvalCyc_ = 0, profUpdCyc_ = 0, profSmallCyc_ = 0;
  uint64_t profEvals_ = 0, profUpds_ = 0, profSmallEvals_ = 0;
#endif

  // Live only while a multi-threaded search runs; lets the master
  // aggregate helper node counts for reporting.
  std::vector<EngineSearch *> helperViews_;
};

} // namespace Search

#endif
