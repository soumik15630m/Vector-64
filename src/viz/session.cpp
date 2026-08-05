#include "session.h"

#include "../cores/movegen.h"
#include "../datagen/selfplay.h"
#include "../search/transposition_table.h" // is_mate_score

#include "../uci/uci_util.h"
#include "wire.h"
#include <cstdlib>
#include <iterator>

#include <algorithm>
#include <chrono>
#include <filesystem>

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

// Pick uniformly among the root moves the search scored within `cp` of the
// best, instead of always the top one.
//
// A search is deterministic: the same position gives the same move, so an
// engine playing itself from the start position replays one identical game
// forever. Random openings paper over that, but the moment they are off the
// problem is back. Varying the choice among near-equal root moves is what
// fixes it at the source -- and it is what Stockfish's own data generator does
// (its random-multi-pv option) for exactly this reason.
//
// Nothing here invents a move or touches a reported number: every candidate is
// one the search itself scored, with a full window, as within `cp` of best.
Core::Move pick_varied(const Search::Result &r,
                       const std::vector<Search::RootLine> &lines, int cp,
                       uint64_t &rngState) {
  // lines is empty unless MultiPV > 1 -- with one line there is no alternative
  // to weigh, so the best move is the only answer.
  if (cp <= 0 || lines.size() < 2 || !r.bestMove.is_ok())
    return r.bestMove;
  // Never gamble with a forced mate: "within 30cp" means nothing in the mate
  // range, and swapping mate-in-2 for mate-in-9 reads as a blunder.
  if (Search::is_mate_score(lines.front().scoreCp))
    return r.bestMove;
  const int best = lines.front().scoreCp;
  std::vector<Core::Move> pool;
  for (const Search::RootLine &l : lines)
    if (l.move.is_ok() && !Search::is_mate_score(l.scoreCp) &&
        best - l.scoreCp <= cp)
      pool.push_back(l.move);
  if (pool.size() < 2)
    return r.bestMove;
  std::mt19937_64 rng(rngState);
  rngState = rng();
  return pool[rng() % pool.size()];
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
  case Mode::Datagen:
    return "datagen";
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
  if (s == "datagen") {
    out = Mode::Datagen;
    return true;
  }
  return false;
}

Session::Session(Config cfg)
    : cfg_(cfg), search_(static_cast<size_t>(cfg.hashMb)) {
  appliedThreads_ = std::max(1, cfg_.threads);
  threads_.store(appliedThreads_);
  search_.set_threads(appliedThreads_);
  search_.set_multipv(std::max(1, cfg_.multiPv));
  moveDelayMs_.store(cfg_.moveDelayMs);
  nodes_.store(std::max(0, cfg_.nodes));
  depth_.store(std::clamp(cfg_.depth, 0, max_depth()));
  varietyCp_.store(std::max(0, cfg_.varietyCp));
  rngState_ = cfg_.seed ? cfg_.seed : 0x9E3779B97F4A7C15ULL;
  // Later games open randomly so they differ; the first starts from the real
  // initial position, which is what you want to see when the tool opens.
  randomOpening_ = true;
  reset_game(false);
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

bool Session::load_compare_net(const std::string &path) {
  if (path.empty()) {
    compareNet_.reset();
    compareName_.clear();
    publish();
    return true;
  }
  auto net = std::make_unique<NNUE::Network>();
  if (!net->load_file(path))
    return false;
  compareNet_ = std::move(net);
  // Show the file name only; the full path is noise in a panel.
  const size_t slash = path.find_last_of("/\\");
  compareName_ = slash == std::string::npos ? path : path.substr(slash + 1);
  publish();
  return true;
}

bool Session::has_compare_net() const { return compareNet_ != nullptr; }

bool Session::start_recording(const std::string &path) {
  std::lock_guard<std::mutex> lk(recMu_);
  rec_.close();
  rec_.clear();
  rec_.open(path, std::ios::out | std::ios::trunc);
  if (!rec_)
    return false;
  recPath_ = path;
  return true;
}

void Session::stop_recording() {
  std::lock_guard<std::mutex> lk(recMu_);
  rec_.close();
  recPath_.clear();
}

bool Session::recording() const {
  std::lock_guard<std::mutex> lk(recMu_);
  return rec_.is_open();
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
    pendingRandomOpening_ = randomOpening_ && m == Mode::SelfPlay;
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

void Session::set_nodes(int nodes) { nodes_.store(std::max(0, nodes)); }

int Session::max_depth() { return 246; } // matches the UCI clamp

void Session::set_depth(int d) { depth_.store(std::clamp(d, 0, max_depth())); }

void Session::set_variety(int cp) { varietyCp_.store(std::max(0, cp)); }

int Session::hardware_threads() {
  const unsigned hc = std::thread::hardware_concurrency();
  // hardware_concurrency may report 0 when it cannot tell; fall back to 1
  // rather than exposing a nonsense maximum.
  return hc == 0 ? 1 : static_cast<int>(hc);
}

void Session::set_threads(int n) {
  // Clamped to what the machine has: asking for more threads than cores makes
  // lazy SMP slower, not faster.
  threads_.store(std::clamp(n, 1, hardware_threads()));
  abortSearch_.store(true); // take effect on the next search, not this one
  cv_.notify_all();
}

namespace {
// The state file sits beside the dataset so a resumed run finds it without
// being told where it is.
std::string state_path_for(const std::string &out) {
  // A sharded dataset keeps its state inside the directory, beside the shards
  // it describes. A single-file dataset -- what the generator wrote before it
  // sharded -- kept it next to the file, so that layout is checked first and a
  // run made by an older build still probes and resumes.
  std::string legacy = out + ".state.json";
  std::error_code ec;
  if (std::filesystem::is_regular_file(out, ec) ||
      std::filesystem::exists(legacy, ec))
    return legacy;
  return (std::filesystem::path(out) / "state.json").string();
}
} // namespace

DatagenState Session::probe_datagen(const std::string &out) {
  DatagenState st;
  st.out = out;
  std::ifstream in(state_path_for(out));
  if (!in)
    return st;
  // Deliberately a tiny hand-parse: the file is ours and has four integers.
  std::string body((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  const auto grab = [&](const char *key) -> int64_t {
    const size_t k = body.find(key);
    if (k == std::string::npos)
      return 0;
    const size_t c = body.find(':', k);
    return c == std::string::npos
               ? 0
               : std::strtoll(body.c_str() + c + 1, nullptr, 10);
  };
  st.resumablePositions = grab("\"positions\"");
  st.games = grab("\"games\"");
  st.shard = static_cast<int>(grab("\"shard\""));
  st.resumable = st.resumablePositions > 0 || st.games > 0;
  return st;
}

bool Session::start_datagen(const DatagenConfig &cfg, bool resume) {
  if (cfg.out.empty())
    return false;
  uint64_t seed = 0;
  {
    std::lock_guard<std::mutex> lk(dgMu_);
    dgOut_.close();
    // Resuming continues the last shard; a fresh run starts at shard 0 and
    // truncates, so a restart cannot silently double-count rows on disk.
    if (!dgOut_.open(cfg.out, cfg.shardPositions, resume))
      return false;
    dgCfg_ = cfg;
    dgState_ = DatagenState{};
    dgState_.out = cfg.out;
    dgState_.target = cfg.targetPositions;
    dgState_.targetGames = cfg.targetGames;
    if (resume) {
      const DatagenState prev = probe_datagen(cfg.out);
      dgState_.positions = prev.resumablePositions;
      dgState_.games = prev.games;
    }
    dgState_.running = true;
    dgStartPositions_ = dgState_.positions;
    // The run owns the search budget for its duration; the UI locks these, so
    // every row in the dataset is produced under the same settings.
    nodes_.store(std::max(0, cfg.nodes));
    depth_.store(std::clamp(cfg.depth, 0, max_depth()));
    dgStart_ = std::chrono::steady_clock::now();
    seed = cfg.seed +
           0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(dgState_.games + 1);
  }
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    // Take the run's seed, offset by the games already on disk. rngState_ is
    // the stream openings and root-move variety are drawn from, so without
    // this a run ignored its seed entirely and -- worse -- a resumed run
    // restarted the stream and regenerated the very games it was resuming
    // past, duplicating rows in the dataset.
    rngState_ = seed ? seed : 0x9E3779B97F4A7C15ULL;
    mode_ = Mode::Datagen;
    pendingReset_ = true;
    pendingRandomOpening_ = true; // datagen always wants varied openings
    ++boardGen_;
  }
  paused_.store(false);
  abortSearch_.store(true);
  publish();
  return true;
}

void Session::stop_datagen() {
  std::lock_guard<std::mutex> lk(dgMu_);
  dgOut_.close();
  dgState_.running = false;
}

void Session::datagen_save_state() {
  // Written after every game so a crash loses at most one game's worth.
  std::ofstream st(state_path_for(dgCfg_.out), std::ios::trunc);
  if (!st)
    return;
  st << "{\n  \"positions\": " << dgState_.positions
     << ",\n  \"games\": " << dgState_.games
     << ",\n  \"target\": " << dgState_.target
     << ",\n  \"nodes\": " << dgCfg_.nodes
     << ",\n  \"shard\": " << dgOut_.shard()
     << ",\n  \"shardPositions\": " << dgCfg_.shardPositions << "\n}\n";
}

void Session::datagen_write(const std::vector<std::pair<std::string, int>> &rec,
                            double wdl) {
  std::lock_guard<std::mutex> lk(dgMu_);
  if (!dgOut_.is_open())
    return;
  for (const auto &pr : rec)
    dgOut_.write_row(
        Datagen::emit_row(pr.first, pr.second, wdl, dgCfg_.raw, dgCfg_.lam));
  dgOut_.flush();
  dgState_.positions += static_cast<int64_t>(rec.size());
  dgState_.shard = dgOut_.shard();
  dgState_.shardPath = dgOut_.current_path();
  ++dgState_.games;
  if (wdl == 1.0)
    ++dgState_.wins;
  else if (wdl == 0.0)
    ++dgState_.losses;
  else
    ++dgState_.draws;

  const double el =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - dgStart_)
          .count();
  const double made =
      static_cast<double>(dgState_.positions - dgStartPositions_);
  dgState_.positionsPerSec = el > 0 ? made / el : 0.0;
  const double remain =
      static_cast<double>(dgState_.target - dgState_.positions);
  dgState_.etaMinutes = dgState_.positionsPerSec > 0
                            ? remain / dgState_.positionsPerSec / 60.0
                            : 0.0;
  datagen_save_state();
}

void Session::set_random_opening(bool v) {
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    randomOpening_ = v;
  }
  publish();
}

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
    pendingRandomOpening_ = randomOpening_ && mode_ == Mode::SelfPlay;
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

// Caller holds cmdMu_. Charges the mover for the time they just used.
void Session::apply_move_internal(Core::Move m) {
  tick_clock();
  Core::UndoInfo ui;
  moves_.push_back(UCI::move_to_uci(m));
  pos_.make_move(m, ui);
  history_.push_back(pos_.hash());
}

// Caller holds cmdMu_. Deducts elapsed time from whoever is on move and
// restarts the stopwatch; flagging ends the game.
void Session::tick_clock() {
  if (!clockRunning_)
    return;
  const auto now = std::chrono::steady_clock::now();
  const int used = static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(now - turnStart_)
          .count());
  turnStart_ = now;
  int &clk = pos_.side_to_move() == Core::WHITE ? whiteMs_ : blackMs_;
  clk -= used;
  if (clk <= 0) {
    clk = 0;
    if (!gameOver_) {
      gameOver_ = true;
      result_ = pos_.side_to_move() == Core::WHITE ? "0-1" : "1-0";
      reason_ = "time";
    }
  }
}

// Caller holds cmdMu_.
void Session::reset_game(bool randomOpening) {
  // How many random plies to open with. Config::openingPlies is 0 so that plain
  // self-play starts from the real initial position, but that number must NOT
  // be what a random opening uses -- zero plies means every game starts from
  // the same position, and a deterministic search then replays the identical
  // game forever. Datagen brings its own count; self-play falls back to the
  // usual 8.
  int plies = 0;
  int balance = cfg_.balance;
  if (randomOpening) {
    if (mode_ == Mode::Datagen) {
      plies = dgCfg_.openingPlies;
      balance = dgCfg_.balance;
    } else {
      plies = cfg_.openingPlies > 0 ? cfg_.openingPlies : kDefaultOpeningPlies;
    }
  }

  Core::Position p;
  if (plies > 0) {
    std::mt19937_64 rng(rngState_);
    rngState_ = rng();
    int tries = 0;
    while (!Datagen::make_opening(p, rng, plies, balance) && ++tries < 64) {
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
  whiteMs_ = blackMs_ = std::max(1000, cfg_.clockMs);
  turnStart_ = std::chrono::steady_clock::now();
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
  // The same position through the comparison net, so the two are directly
  // comparable rather than being read from different moments.
  VizFrame cf;
  const bool cmp = compareNet_ != nullptr;
  if (cmp)
    cf = capture(pos, *compareNet_, cfg_.l1TopK);
  Snapshot copy;
  {
    std::unique_lock<std::mutex> lk(mu_);
    snap_.nnueActive = nnue;
    snap_.thinking = thinking;
    if (nnue)
      snap_.frame = std::move(f);
    snap_.compareActive = cmp;
    snap_.compareName = compareName_;
    if (cmp)
      snap_.compareFrame = std::move(cf);
    snap_.seq = ++seq_;
    copy = snap_;
  }
  cv_.notify_all();
  // Written outside the snapshot lock: recording must never stall readers.
  record(copy);
}

void Session::record(const Snapshot &s) {
  std::lock_guard<std::mutex> lk(recMu_);
  if (!rec_.is_open())
    return;
  rec_ << encode_record(s) << '\n';
  rec_.flush(); // a crashed run should still leave a usable log
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
    if (clockRunning_)
      tick_clock();
    g.whiteMs = whiteMs_;
    g.blackMs = blackMs_;
    g.clockRunning = clockRunning_;
    m = mode_;
    ec = engineColor_;
  }
  DatagenState dg;
  {
    std::lock_guard<std::mutex> lk(dgMu_);
    dg = dgState_;
  }
  std::lock_guard<std::mutex> lk(mu_);
  snap_.datagen = dg;
  snap_.game = std::move(g);
  snap_.legalMoves = std::move(legalNow);
  snap_.mode = m;
  snap_.engineColor = ec;
  snap_.running = !stop_.load();
  snap_.paused = paused_.load();
  snap_.threads = appliedThreads_;
  snap_.varietyCp = varietyCp_.load();
  snap_.nnueActive = search_.evaluator().nnue_active();
  snap_.seq = ++seq_;
  cv_.notify_all();
}

Search::Result Session::think(bool ponder,
                              std::vector<Search::RootLine> *linesOut) {
  // Thread count changes land here, between searches.
  const int want = threads_.load();
  if (want != appliedThreads_) {
    appliedThreads_ = want;
    search_.set_threads(want);
  }
  Core::Position local;
  uint64_t gen;
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    local = pos_;
    gen = boardGen_;
  }

  Search::Limits limits;
  const int n = nodes_.load();
  const int d = depth_.load();
  // Pondering is ONE long search that ends when the opponent moves, not a
  // stream of short ones. Repeating short searches restarted the display many
  // times a second, which read as flicker rather than as thinking.
  limits.maxNodes = ponder ? 0 : (n > 0 ? static_cast<uint64_t>(n) : 0);
  limits.maxDepth = d > 0 ? d : max_depth();

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
    for (const Search::RootLine &l : info.lines) {
      Candidate c;
      c.move = UCI::move_to_uci(l.move);
      c.scoreCp = l.scoreCp;
      for (Core::Move m : l.pv)
        c.pv.push_back(UCI::move_to_uci(m));
      si.candidates.push_back(std::move(c));
    }
    // Keep the deepest iteration's lines: they are what the caller chooses
    // from, and each later iteration supersedes the one before it.
    if (linesOut)
      *linesOut = info.lines;

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

    if (ponder) {
      const auto now = std::chrono::steady_clock::now();
      if (now - lastPublish_ < std::chrono::milliseconds(110))
        return; // too soon: the previous frame is still what matters
      lastPublish_ = now;
    }
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
      // Honour the toggle. Forcing true here made the setting meaningless;
      // leaving it false for every game makes each one start from the same
      // position, which a deterministic search then replays identically -- so
      // the toggle is what decides, and it defaults to on for variety.
      reset_game(randomOpening_);
      publish_needed_ = true;
      return;
    }
    if (detect_terminal(has_legal_move(pos_))) {
      publish_needed_ = true;
      return;
    }
  }

  std::vector<Search::RootLine> lines;
  const Search::Result r = think(/*ponder=*/false, &lines);
  abortSearch_.store(false);
  if (!r.bestMove.is_ok()) {
    publish();
    return;
  }
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    // Vary the opening among near-equal moves so consecutive games differ;
    // past varietyPlies the engine always plays its best move.
    const Core::Move played =
        static_cast<int>(moves_.size()) < cfg_.varietyPlies
            ? pick_varied(r, lines, varietyCp_.load(), rngState_)
            : r.bestMove;
    apply_move_internal(played);
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
  bool engineToMove = false;
  bool live = false;
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    // Short-circuits: detect_terminal (which records the result) only runs
    // while the game is still live.
    const bool finished = gameOver_ || detect_terminal(has_legal_move(pos_));
    live = !finished;
    // Charge the side on move for the time that has passed; this is also what
    // flags a player who runs out.
    tick_clock();
    engineToMove =
        live && (static_cast<int>(pos_.side_to_move()) == engineColor_);
  }
  if (!engineToMove) {
    // The opponent's clock is running. Keep searching the position they are
    // deciding on, so the network view goes on evolving instead of freezing --
    // this is the engine genuinely thinking on their time, not a replay.
    if (live) {
      think(/*ponder=*/true);
      abortSearch_.store(false);
    }
    publish();
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::milliseconds(live ? 30 : 200));
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

// One datagen game, played move by move so the UI can watch it, but labelled
// exactly the way the CLI datagen labels: the same skip-plies rule, the same
// clamp, the same row format, and the same cheap terminal detection.
void Session::datagen_step() {
  {
    std::lock_guard<std::mutex> lk(dgMu_);
    if (!dgState_.running)
      return;
    // Whichever limit is reached first ends the run.
    if (dgState_.positions >= dgState_.target ||
        (dgState_.targetGames > 0 && dgState_.games >= dgState_.targetGames)) {
      dgState_.running = false;
      dgOut_.close();
      publish_needed_ = true;
      return;
    }
  }

  // Terminal? Bank the game's rows and start the next one.
  bool finished = false;
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    finished = gameOver_ || detect_terminal(has_legal_move(pos_));
  }
  if (finished) {
    double wdl = 0.5;
    std::vector<std::pair<std::string, int>> rec;
    {
      std::lock_guard<std::mutex> lk(cmdMu_);
      if (result_ == "1-0")
        wdl = 1.0;
      else if (result_ == "0-1")
        wdl = 0.0;
      rec.swap(dgRecord_);
      if (result_ == "1-0")
        ++wins_;
      else if (result_ == "0-1")
        ++losses_;
      else
        ++draws_;
      reset_game(true); // datagen always wants varied openings
    }
    datagen_write(rec, wdl);
    publish();
    return;
  }

  std::vector<Search::RootLine> lines;
  const Search::Result r = think(/*ponder=*/false, &lines);
  abortSearch_.store(false);
  if (!r.bestMove.is_ok()) {
    publish();
    return;
  }
  {
    std::lock_guard<std::mutex> lk(cmdMu_);
    // Skip the opening plies and any position in check, exactly as the CLI
    // generator does -- those labels are noise.
    const int ply = static_cast<int>(moves_.size());
    if (ply >= dgCfg_.skipPlies && !pos_.in_check()) {
      const int evalWhite =
          pos_.side_to_move() == Core::WHITE ? r.scoreCp : -r.scoreCp;
      dgRecord_.emplace_back(pos_.toFEN(), Datagen::clamp_score(evalWhite));
    }
    // The label above is the position's value (the best line's score) and is
    // unaffected by which near-equal continuation we then walk into. Varying
    // that continuation is what stops one random opening from always yielding
    // the same game, and so the same rows.
    apply_move_internal(ply < dgCfg_.varietyPlies
                            ? pick_varied(r, lines, dgCfg_.varietyCp, rngState_)
                            : r.bestMove);
    if (static_cast<int>(moves_.size()) >= dgCfg_.maxPlies)
      gameOver_ = true, result_ = "1/2-1/2", reason_ = "maxplies";
  }
  publish_frame_current();
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
      const bool wantClock = (m == Mode::Human) && !paused_.load();
      if (wantClock != clockRunning_) {
        clockRunning_ = wantClock;
        turnStart_ = std::chrono::steady_clock::now();
      }
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
    case Mode::Datagen:
      datagen_step();
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
