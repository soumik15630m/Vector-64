#include "session.h"

#include "../cores/movegen.h"
#include "../datagen/selfplay.h"
#include "../uci/uci_util.h"

#include <algorithm>
#include <chrono>

namespace Viz {
namespace {

// Resolve a long-algebraic move against the legal moves of `pos`. Matching the
// engine's own formatting means castling and promotions need no special cases.
bool find_move(const Core::Position &pos, const std::string &uci,
               Core::Move &out) {
  Core::Position scratch = pos; // generate_legal_moves needs a mutable board
  Core::MoveList legal;
  Core::generate_legal_moves(scratch, legal);
  for (int i = 0; i < legal.size(); ++i) {
    if (UCI::move_to_uci(legal[i]) == uci) {
      out = legal[i];
      return true;
    }
  }
  return false;
}

bool has_legal_move(const Core::Position &pos) {
  Core::Position scratch = pos;
  Core::MoveList legal;
  Core::generate_legal_moves(scratch, legal);
  return legal.size() > 0;
}

} // namespace

const char *mode_name(Mode m) {
  switch (m) {
  case Mode::SelfPlay:
    return "selfplay";
  case Mode::Analysis:
    return "analysis";
  case Mode::Human:
    return "human";
  }
  return "selfplay";
}

bool mode_from_name(const std::string &s, Mode &out) {
  if (s == "selfplay") {
    out = Mode::SelfPlay;
    return true;
  }
  if (s == "analysis") {
    out = Mode::Analysis;
    return true;
  }
  if (s == "human") {
    out = Mode::Human;
    return true;
  }
  return false;
}

Session::Session(Config cfg)
    : cfg_(cfg), search_(static_cast<size_t>(cfg.hashMb)) {
  search_.set_threads(std::max(1, cfg_.threads));
  moveDelayMs_.store(cfg_.moveDelayMs);
  nodes_.store(cfg_.nodes);
  rngState_ = cfg_.seed ? cfg_.seed : 0x9E3779B97F4A7C15ULL;
  reset_game(true);
  publish();
}

Session::~Session() { stop(); }

bool Session::load_net(const std::string &path) {
  const bool ok = search_.load_nnue(path);
  publish();
  return ok;
}

bool Session::load_net_buffer(const unsigned char *data, std::size_t size) {
  const bool ok = search_.load_nnue_buffer(data, size);
  publish();
  return ok;
}

void Session::start() {
  if (worker_.joinable())
    return;
  stop_.store(false);
  worker_ = std::thread([this] { run(); });
}

void Session::stop() {
  stop_.store(true);
  abortSearch_.store(true);
  cv_.notify_all();
  if (worker_.joinable())
    worker_.join();
}

void Session::set_mode(Mode m) {
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    if (mode_ == m)
      return;
    mode_ = m;
    pendingReset_ = true;
    pendingRandomOpening_ = (m == Mode::SelfPlay);
    ++boardGen_;
  }
  abortSearch_.store(true);
  publish();
}

void Session::set_paused(bool p) {
  paused_.store(p);
  if (p)
    abortSearch_.store(true);
  cv_.notify_all();
}

void Session::step() {
  stepOnce_.store(true);
  cv_.notify_all();
}

void Session::set_move_delay(int ms) { moveDelayMs_.store(std::max(0, ms)); }

void Session::set_nodes(int nodes) { nodes_.store(std::max(1, nodes)); }

void Session::set_engine_color(int color) {
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    engineColor_ = color ? 1 : 0;
  }
  publish();
}

void Session::new_game() {
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    pendingReset_ = true;
    pendingRandomOpening_ = (mode_ == Mode::SelfPlay);
    ++boardGen_;
  }
  abortSearch_.store(true);
  publish();
}

bool Session::set_position(const std::string &fen,
                           const std::vector<std::string> &moves) {
  Core::Position p;
  if (!p.setFromFEN(fen.empty() ? Datagen::START_FEN : fen.c_str()))
    return false;
  std::vector<std::string> applied;
  applied.reserve(moves.size());
  for (const std::string &m : moves) {
    Core::Move mv;
    if (!find_move(p, m, mv))
      return false;
    Core::UndoInfo ui;
    p.make_move(mv, ui);
    applied.push_back(m);
  }
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    startFen_ = fen.empty() ? Datagen::START_FEN : fen;
    startPos_.setFromFEN(startFen_.c_str());
    pos_ = p;
    moves_ = std::move(applied);
    history_.clear();
    history_.push_back(pos_.hash());
    gameOver_ = false;
    result_.clear();
    reason_.clear();
    pendingReset_ = false;
    ++boardGen_;
  }
  abortSearch_.store(true);
  publish();
  return true;
}

bool Session::play_move(const std::string &uci) {
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    if (gameOver_)
      return false;
    Core::Move mv;
    if (!find_move(pos_, uci, mv))
      return false;
    apply_move_internal(mv);
    ++boardGen_;
  }
  abortSearch_.store(true);
  publish();
  return true;
}

// Caller holds cmdMu_.
void Session::apply_move_internal(Core::Move m) {
  Core::UndoInfo ui;
  moves_.push_back(UCI::move_to_uci(m));
  pos_.make_move(m, ui);
  history_.push_back(pos_.hash());
}

void Session::reset_game(bool randomOpening) {
  Core::Position p;
  if (randomOpening) {
    std::mt19937_64 rng(rngState_);
    rngState_ = rng();
    int tries = 0;
    while (!Datagen::make_opening(p, rng, cfg_.openingPlies, cfg_.balance) &&
           ++tries < 64) {
    }
    if (tries >= 64)
      p.setFromFEN(Datagen::START_FEN);
  } else {
    p.setFromFEN(Datagen::START_FEN);
  }
  startPos_ = p;
  startFen_ = p.toFEN();
  pos_ = p;
  moves_.clear();
  history_.clear();
  history_.push_back(pos_.hash());
  gameOver_ = false;
  result_.clear();
  reason_.clear();
  ++gameIndex_;
  search_.clear(); // games are independent
}

Snapshot Session::snapshot() const {
  std::lock_guard<std::mutex> lk(mu_);
  return snap_;
}

Snapshot Session::wait_for(uint64_t have, int timeoutMs) const {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait_for(lk, std::chrono::milliseconds(timeoutMs),
               [&] { return seq_ > have; });
  return snap_;
}

void Session::publish_frame(const Core::Position &pos, bool thinking) {
  const bool nnue = search_.evaluator().nnue_active();
  VizFrame f;
  if (nnue)
    f = capture(pos, search_.evaluator().big(), cfg_.l1TopK);
  std::lock_guard<std::mutex> lk(mu_);
  snap_.nnueActive = nnue;
  snap_.thinking = thinking;
  if (nnue)
    snap_.frame = std::move(f);
  snap_.seq = ++seq_;
  cv_.notify_all();
}

void Session::publish() {
  GameState g;
  Mode m;
  int ec;
  std::vector<std::string> legalNow;
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    // Generated here, under the same lock as the FEN below: a client must
    // never be offered moves that belong to a different position.
    Core::Position scratch = pos_;
    Core::MoveList legal;
    Core::generate_legal_moves(scratch, legal);
    legalNow.reserve(static_cast<size_t>(legal.size()));
    for (int i = 0; i < legal.size(); ++i)
      legalNow.push_back(UCI::move_to_uci(legal[i]));
    g.fen = pos_.toFEN();
    g.startFen = startFen_;
    g.moves = moves_;
    g.lastMove = moves_.empty() ? std::string() : moves_.back();
    g.ply = static_cast<int>(moves_.size());
    g.over = gameOver_;
    g.result = result_;
    g.reason = reason_;
    g.gameIndex = gameIndex_;
    g.wins = wins_;
    g.draws = draws_;
    g.losses = losses_;
    m = mode_;
    ec = engineColor_;
  }
  std::lock_guard<std::mutex> lk(mu_);
  snap_.game = std::move(g);
  snap_.legalMoves = std::move(legalNow);
  snap_.mode = m;
  snap_.engineColor = ec;
  snap_.running = !stop_.load();
  snap_.paused = paused_.load();
  snap_.threads = std::max(1, cfg_.threads);
  snap_.nnueActive = search_.evaluator().nnue_active();
  snap_.seq = ++seq_;
  cv_.notify_all();
}

Search::Result Session::think() {
  Core::Position local;
  uint64_t gen;
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    local = pos_;
    gen = boardGen_;
  }

  Search::Limits limits;
  const int n = nodes_.load();
  limits.maxNodes = n > 0 ? static_cast<uint64_t>(n) : 0;
  limits.maxDepth = cfg_.depth > 0 ? cfg_.depth : 64;

  const bool nnue = search_.evaluator().nnue_active();

  Search::Callbacks cb;
  cb.shouldStop = [this] { return stop_.load() || abortSearch_.load(); };
  cb.onInfo = [&](const Search::IterInfo &info) {
    SearchInfo si;
    si.depth = info.depth;
    si.seldepth = info.seldepth;
    si.scoreCp = info.scoreCp;
    si.nodes = info.nodes;
    si.tbHits = info.tbHits;
    si.elapsedMs = info.elapsedMs;
    si.nps = info.elapsedMs > 0
                 ? static_cast<int>(info.nodes * 1000ULL /
                                    static_cast<uint64_t>(info.elapsedMs))
                 : 0;
    si.qsearchTtHitRate = info.qsearchTtHitRate;
    si.negamaxTtHitRate = info.negamaxTtHitRate;

    // Walk the principal variation and probe its leaf: that is the position
    // the engine is actually weighing at this depth. Each move is validated,
    // so a stale PV can never corrupt the board copy.
    Core::Position leaf = local;
    for (int i = 0; i < info.pvLen; ++i) {
      const std::string mv = UCI::move_to_uci(info.pv[i]);
      si.pv.push_back(mv);
      Core::Move found;
      if (!find_move(leaf, mv, found))
        break;
      Core::UndoInfo ui;
      leaf.make_move(found, ui);
    }

    VizFrame f;
    if (nnue)
      f = capture(leaf, search_.evaluator().big(), cfg_.l1TopK);

    std::lock_guard<std::mutex> lk(mu_);
    snap_.search = std::move(si);
    snap_.thinking = true;
    snap_.nnueActive = nnue;
    if (nnue)
      snap_.frame = std::move(f);
    snap_.seq = ++seq_;
    cv_.notify_all();
  };

  const Search::Result r = search_.search(local, limits, cb);

  // Discard the result if the board moved under us (a command arrived).
  std::lock_guard<std::mutex> lk(cmdMu_);
  if (gen != boardGen_)
    return Search::Result{};
  return r;
}

// Caller holds cmdMu_. Returns true if the game ended.
bool Session::detect_terminal(bool hadLegalMove) {
  if (!hadLegalMove) {
    if (pos_.in_check()) {
      // Side to move is mated.
      const bool whiteMated = pos_.side_to_move() == Core::WHITE;
      result_ = whiteMated ? "0-1" : "1-0";
      reason_ = "checkmate";
    } else {
      result_ = "1/2-1/2";
      reason_ = "stalemate";
    }
    gameOver_ = true;
    return true;
  }
  int repeats = 0;
  const uint64_t h = pos_.hash();
  for (uint64_t k : history_)
    if (k == h)
      ++repeats;
  if (repeats >= 3) {
    result_ = "1/2-1/2";
    reason_ = "repetition";
  } else if (pos_.halfmove_clock() >= 100) {
    result_ = "1/2-1/2";
    reason_ = "fifty";
  } else if (Datagen::insufficient_material(pos_)) {
    result_ = "1/2-1/2";
    reason_ = "material";
  } else if (static_cast<int>(moves_.size()) >= cfg_.maxPlies) {
    result_ = "1/2-1/2";
    reason_ = "maxplies";
  } else {
    return false;
  }
  gameOver_ = true;
  return true;
}

void Session::self_play_step() {
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    if (gameOver_) {
      if (result_ == "1-0")
        ++wins_;
      else if (result_ == "0-1")
        ++losses_;
      else
        ++draws_;
      reset_game(true);
      publish_needed_ = true;
      return;
    }
    if (detect_terminal(has_legal_move(pos_))) {
      publish_needed_ = true;
      return;
    }
  }

  const Search::Result r = think();
  abortSearch_.store(false);
  if (!r.bestMove.is_ok()) {
    publish();
    return;
  }
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    apply_move_internal(r.bestMove);
  }
  publish_frame_current();
  publish();
}

void Session::analysis_step() {
  // The position is user-driven: think about it, then idle until it changes.
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    if (!has_legal_move(pos_)) {
      detect_terminal(false);
    }
  }
  think();
  abortSearch_.store(false);
  publish();
  // Wait for a board change rather than re-searching the same position.
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait_for(lk, std::chrono::milliseconds(200));
}

void Session::human_step() {
  bool engineToMove;
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    // Short-circuits: detect_terminal (which records the result) only runs
    // while the game is still live.
    const bool finished = gameOver_ || detect_terminal(has_legal_move(pos_));
    engineToMove =
        !finished && (static_cast<int>(pos_.side_to_move()) == engineColor_);
  }
  if (!engineToMove) {
    publish();
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::milliseconds(100));
    return;
  }
  const Search::Result r = think();
  abortSearch_.store(false);
  if (r.bestMove.is_ok()) {
    {
      std::lock_guard<std::mutex> lk(cmdMu_);
      apply_move_internal(r.bestMove);
    }
    publish_frame_current();
  }
  publish();
}

void Session::publish_frame_current() {
  Core::Position p;
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    p = pos_;
  }
  publish_frame(p, false);
}

void Session::run() {
  while (!stop_.load()) {
    {
      std::lock_guard<std::mutex> lk(cmdMu_);
      if (pendingReset_) {
        pendingReset_ = false;
        reset_game(pendingRandomOpening_);
      }
    }
    if (publish_needed_) {
      publish_needed_ = false;
      publish();
    }

    if (paused_.load() && !stepOnce_.exchange(false)) {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait_for(lk, std::chrono::milliseconds(50));
      continue;
    }

    Mode m;
    {
      std::lock_guard<std::mutex> lk(cmdMu_);
      m = mode_;
    }
    switch (m) {
    case Mode::SelfPlay:
      self_play_step();
      break;
    case Mode::Analysis:
      analysis_step();
      break;
    case Mode::Human:
      human_step();
      break;
    }

    const int d = moveDelayMs_.load();
    if (d > 0 && m == Mode::SelfPlay && !stop_.load()) {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait_for(lk, std::chrono::milliseconds(d));
    }
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    snap_.running = false;
    snap_.thinking = false;
    snap_.seq = ++seq_;
  }
  cv_.notify_all();
}

} // namespace Viz
