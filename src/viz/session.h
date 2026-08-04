#ifndef VIZ_SESSION_H
#define VIZ_SESSION_H

#include "../search/search.h"
#include "probe.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Session layer: owns an engine, drives it in one of the visualizer's modes,
// and publishes an immutable snapshot of "what the engine is doing right now"
// for the transport to stream.
//
// Everything here is real engine output. The snapshot carries the search's own
// telemetry and a VizFrame captured from the same net the search is using.
namespace Viz {

enum class Mode {
  SelfPlay, // engine plays itself continuously
  Analysis, // a position is set; the engine thinks about it
  Human     // engine plays one side, the user the other
};

const char *mode_name(Mode m);
bool mode_from_name(const std::string &s, Mode &out);

struct Config {
  int nodes = 20000; // per-move node budget (0 = depth-limited only)
  int depth = 0;     // 0 = no depth cap
  int hashMb = 32;
  int threads = 1;
  int moveDelayMs = 300; // pacing so self-play is watchable
  // 0 = start from the normal starting position. Self-play used a random
  // balanced opening so games differed, but starting mid-position is confusing
  // to watch, so the default is now the real start and variety is opt-in.
  int openingPlies = 0;
  int balance = 150;
  int maxPlies = 300;
  uint64_t seed = 0;
  int l1TopK = 12;
  // Human mode clock, per side. The engine also keeps thinking on the
  // opponent's time (see ponder handling in human_step).
  int clockMs = 300000; // 5 minutes
  // Candidate moves to evaluate per search. >1 makes the decision visible as a
  // comparison; it costs extra search, which is why the engine defaults to 1.
  int multiPv = 4;
};

// One candidate move with the score the search actually gave it.
struct Candidate {
  std::string move;
  int scoreCp = 0;
  std::vector<std::string> pv;
};

struct SearchInfo {
  int depth = 0;
  int seldepth = 0;
  int scoreCp = 0;
  uint64_t nodes = 0;
  uint64_t tbHits = 0;
  int elapsedMs = 0;
  int nps = 0;
  std::vector<std::string> pv;
  double qsearchTtHitRate = 0.0;
  double negamaxTtHitRate = 0.0;
  std::vector<Candidate> candidates;
};

struct GameState {
  std::string fen;
  std::vector<std::string> moves; // long algebraic, from the game's start
  std::string startFen;
  std::string lastMove;
  int ply = 0;
  bool over = false;
  std::string result; // "1-0" | "0-1" | "1/2-1/2" | ""
  std::string reason; // checkmate | stalemate | repetition | fifty | material |
                      // maxplies
  int gameIndex = 0;
  int wins = 0, draws = 0, losses = 0; // session tally, white perspective
  // Human-mode clocks, milliseconds remaining.
  int whiteMs = 0;
  int blackMs = 0;
  bool clockRunning = false;
};

// One coherent view of the session. `seq` increments on every publish so a
// client can tell whether anything changed.
struct Snapshot {
  uint64_t seq = 0;
  Mode mode = Mode::SelfPlay;
  bool running = false;
  bool paused = false;
  bool thinking = false;
  bool nnueActive = false;
  int threads = 1;
  int engineColor = 1; // Human mode: the colour the engine plays (Core::Color)
  GameState game;
  SearchInfo search;
  VizFrame frame;
  std::vector<std::string> legalMoves; // side to move, for Human mode
};

class Session {
public:
  explicit Session(Config cfg);
  ~Session();
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  // Load the net the engine (and therefore the visualization) uses.
  bool load_net(const std::string &path);
  bool load_net_buffer(const unsigned char *data, std::size_t size);

  void start();
  void stop();

  void set_mode(Mode m);
  void set_paused(bool p);
  void step(); // advance one move while paused
  void set_move_delay(int ms);
  void set_nodes(int nodes);
  // Applied between searches, never during one: set_threads reconfigures lazy
  // SMP and is not safe to change under a running search.
  void set_threads(int n);
  // 0 = no depth cap (node-limited only). Clamped to the engine's ceiling.
  void set_depth(int d);
  // Deepest search the engine supports; the UI uses it to bound its control.
  static int max_depth();
  void new_game();
  // Analysis/Human: set the board. `moves` are long-algebraic from `fen`.
  bool set_position(const std::string &fen,
                    const std::vector<std::string> &moves);
  // Human mode: apply the user's move if legal.
  bool play_move(const std::string &uci);
  void set_engine_color(int color);
  void set_random_opening(bool v);

  Snapshot snapshot() const;
  // Wait until the published sequence passes `have` (or `timeoutMs` elapses).
  Snapshot wait_for(uint64_t have, int timeoutMs) const;

  // What this machine can actually run. The UI clamps its thread control to
  // this instead of guessing a maximum.
  static int hardware_threads();

  // The net the engine is using, for the inspector. Weights are immutable once
  // loaded, so this is safe to read while the worker searches.
  const NNUE::Network &net() const { return search_.evaluator().big(); }

private:
  void run();
  void self_play_step();
  void analysis_step();
  void human_step();
  // Search the current position, publishing per-iteration telemetry.
  // `ponder` runs one long search instead of a short budgeted one: it is the
  // engine thinking on the opponent's clock and ends when they move.
  Search::Result think(bool ponder = false);
  void publish_frame(const Core::Position &pos, bool thinking);
  void publish_frame_current();
  void publish();
  void apply_move_internal(Core::Move m);
  bool detect_terminal(bool hadLegalMove);
  void reset_game(bool randomOpening);

  Config cfg_;
  Search::EngineSearch search_;

  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  Snapshot snap_;
  uint64_t seq_ = 0;

  // Worker + control.
  std::thread worker_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> paused_{false};
  std::atomic<bool> stepOnce_{false};
  std::atomic<bool> abortSearch_{false};
  std::atomic<int> moveDelayMs_{300};
  std::atomic<int> nodes_{20000};
  std::atomic<int> threads_{1};
  std::atomic<int> depth_{0};
  // Rate-limits telemetry during a long ponder so the UI is not flooded.
  std::chrono::steady_clock::time_point lastPublish_{};
  int appliedThreads_ = 1;

  // Board state, touched only by the worker thread except through commands.
  std::mutex cmdMu_;
  Core::Position pos_;
  Core::Position startPos_;
  std::vector<std::string> moves_;
  std::string startFen_;
  std::vector<uint64_t> history_; // for repetition detection
  Mode mode_ = Mode::SelfPlay;
  int engineColor_ = 1;
  int gameIndex_ = 0;
  int wins_ = 0, draws_ = 0, losses_ = 0;
  bool gameOver_ = false;
  std::string result_, reason_;
  uint64_t rngState_ = 0;
  bool pendingReset_ = false;
  bool pendingRandomOpening_ = false;
  bool randomOpening_ = false;
  // Human-mode clocks. Only the side to move burns time.
  int whiteMs_ = 0, blackMs_ = 0;
  std::chrono::steady_clock::time_point turnStart_{};
  bool clockRunning_ = false;
  void tick_clock();
  // Bumped whenever the board changes from outside the worker, so a search
  // whose position was replaced mid-flight discards its result.
  uint64_t boardGen_ = 0;
  bool publish_needed_ = false;
};

} // namespace Viz

#endif
