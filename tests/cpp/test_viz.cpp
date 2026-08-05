// Verifies the visualizer telemetry layer (src/viz/probe).
//
// The bar is that a frame reports the engine's real arithmetic, not a
// decorative approximation: the captured eval must equal the ordinary eval
// path, and the reported per-layer attributions must reproduce the exact layer
// outputs when summed back up.
#include "cores/attacks.h"
#include "cores/movegen.h"
#include "cores/position.h"
#include "cores/zobrist.h"
#include "nnue/halfka.h"
#include "uci/uci_util.h"
#include "viz/probe.h"
#include "viz/session.h"
#include "viz/wire.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace {

constexpr const char *FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
};

bool fail(const char *what, const std::string &fen) {
  std::printf("FAIL: %s\n  fen: %s\n", what, fen.c_str());
  return false;
}

// Every active feature must re-derive from its own reported parts, and the
// perspective's own king must be excluded (it is the bucket anchor).
bool check_features(const Viz::VizFrame &f, const Core::Position &pos,
                    const Viz::PerspectiveInput &p, Core::Color persp) {
  namespace HK = NNUE::HalfKA;
  const int expected = Core::popcount(pos.occupancy()) - 1;
  if (int(p.features.size()) != expected)
    return fail("active feature count != pieces - 1", f.fen);
  if (p.kingBucket < 0 || p.kingBucket >= HK::KING_BUCKETS)
    return fail("king bucket out of range", f.fen);

  const HK::Orient o = HK::make_orient(persp, Core::Square(p.kingSquare));
  for (const Viz::ActiveFeature &af : p.features) {
    if (af.square == p.kingSquare && af.pieceColor == int(persp))
      return fail("perspective's own king emitted as a feature", f.fen);
    const int want = HK::feature_index(o, Core::Color(af.pieceColor),
                                       Core::PieceType(af.pieceType),
                                       Core::Square(af.square));
    if (want != af.featureIndex)
      return fail("feature index does not match its (piece, square)", f.fen);
    // The same index must fall out of the reported bucket/kind/oriented square.
    const int rebuilt =
        (p.kingBucket * HK::PIECE_KINDS + af.pieceKind) * HK::SQUARES +
        af.orientedSquare;
    if (rebuilt != af.featureIndex)
      return fail("feature index != (bucket, kind, orientedSquare) round trip",
                  f.fen);
  }
  return true;
}

// Sum the reported contributions back into the layer outputs the engine
// produced. If attribution were cosmetic, these would not reconcile.
bool check_attribution(const Viz::VizFrame &f, const NNUE::Network &net) {
  constexpr int L1 = NNUE::Arch::L1;
  constexpr int L2 = NNUE::Arch::L2;
  const NNUE::Network::Bucket &b = net.bucket_weights(f.bucket);

  // positional = (outb + sum_j outw[j]*l2out[j]) >> OUT_SHIFT
  int32_t raw = b.outb;
  for (int j = 0; j < L2; ++j)
    raw += f.outContrib[j];
  if ((raw >> NNUE::Arch::OUT_SHIFT) != f.positional)
    return fail("outContrib does not reproduce the positional term", f.fen);

  // l2out[o] = clip((l2b[o] + sum_j l2w[o][j]*l1out[j]) >> L2_SHIFT)
  for (int o = 0; o < L2; ++o) {
    int32_t sum = b.l2b[o];
    for (int j = 0; j < L1; ++j)
      sum += f.l2Contrib[static_cast<size_t>(o) * L1 + j];
    const int v = sum >> NNUE::Arch::L2_SHIFT;
    const int clipped =
        v < 0 ? 0 : (v > NNUE::Arch::ACT_MAX ? NNUE::Arch::ACT_MAX : v);
    if (clipped != int(f.l2out[o]))
      return fail("l2Contrib does not reproduce l2out", f.fen);
  }

  // l1Top must be ordered by descending |contribution|.
  for (int o = 0; o < L1 && f.l1TopK > 0; ++o) {
    for (int j = 1; j < f.l1TopK; ++j) {
      const auto &prev = f.l1Top[static_cast<size_t>(o) * f.l1TopK + j - 1];
      const auto &cur = f.l1Top[static_cast<size_t>(o) * f.l1TopK + j];
      const int32_t a = prev.value < 0 ? -prev.value : prev.value;
      const int32_t c = cur.value < 0 ? -cur.value : cur.value;
      if (c > a)
        return fail("l1Top is not ordered by |contribution|", f.fen);
    }
  }
  return true;
}

bool same_frame(const Viz::VizFrame &a, const Viz::VizFrame &b) {
  return a.fen == b.fen && a.eval == b.eval && a.psqt == b.psqt &&
         a.positional == b.positional && a.bucket == b.bucket &&
         a.accUs == b.accUs && a.accThem == b.accThem && a.l1in == b.l1in &&
         a.l1out == b.l1out && a.l2out == b.l2out &&
         a.outContrib == b.outContrib && a.l2Contrib == b.l2Contrib;
}

// Drive a live self-play session and check the driver keeps the board and the
// reported move list in agreement: replaying `moves` from `startFen` must
// reproduce `fen` exactly, and every move must have been legal when played.
bool session_test() {
  Viz::Config cfg;
  cfg.nodes = 500;
  cfg.moveDelayMs = 0;
  cfg.hashMb = 8;
  cfg.threads = 1;
  cfg.maxPlies = 40; // keep games short so several finish quickly
  cfg.openingPlies = 6;
  cfg.seed = 7;

  Viz::Session session(cfg);
  session.start();

  const auto t0 = std::chrono::steady_clock::now();
  uint64_t seq = 0;
  int checked = 0;
  int lastGame = 0;
  int gamesSeen = 0;
  bool ok = true;

  while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
    const Viz::Snapshot s = session.wait_for(seq, 200);
    if (s.seq == seq)
      continue;
    seq = s.seq;
    if (s.game.startFen.empty())
      continue;

    if (s.game.gameIndex != lastGame) {
      if (lastGame != 0)
        ++gamesSeen;
      lastGame = s.game.gameIndex;
    }

    Core::Position p;
    if (!p.setFromFEN(s.game.startFen)) {
      std::printf("FAIL: session startFen does not parse: %s\n",
                  s.game.startFen.c_str());
      ok = false;
      break;
    }
    bool replayed = true;
    for (const std::string &mv : s.game.moves) {
      Core::MoveList legal;
      Core::generate_legal_moves(p, legal);
      Core::Move found = Core::Move::none();
      for (int i = 0; i < legal.size(); ++i)
        if (UCI::move_to_uci(legal[i]) == mv)
          found = legal[i];
      if (!found.is_ok()) {
        std::printf("FAIL: session reported an illegal move '%s' from %s\n",
                    mv.c_str(), s.game.startFen.c_str());
        replayed = false;
        break;
      }
      Core::UndoInfo ui;
      p.make_move(found, ui);
    }
    if (!replayed) {
      ok = false;
      break;
    }
    if (p.toFEN() != s.game.fen) {
      std::printf("FAIL: replaying moves does not reproduce the session board\n"
                  "  replayed: %s\n  reported: %s\n",
                  p.toFEN().c_str(), s.game.fen.c_str());
      ok = false;
      break;
    }
    ++checked;
    if (gamesSeen >= 2 && checked > 40)
      break;
  }

  session.stop();

  if (!ok)
    return false;
  if (checked == 0) {
    std::printf("FAIL: session produced no snapshots\n");
    return false;
  }
  if (gamesSeen == 0) {
    std::printf("FAIL: no self-play game reached a terminal state\n");
    return false;
  }
  std::printf("PASS: session drives legal self-play (%d snapshots verified, "
              "%d games completed, board == replayed move list)\n",
              checked, gamesSeen);
  return true;
}

// Self-play from the fixed starting position must not replay one identical
// game forever.
//
// A search is deterministic, so with nothing varying the move choice the engine
// plays the same game from the same position every time -- which is exactly the
// bug this guards. Random openings hide it; here they are deliberately OFF so
// only the root-move variety can make the games differ.
bool variety_test() {
  Viz::Config cfg;
  cfg.nodes = 400;
  cfg.moveDelayMs = 0;
  cfg.hashMb = 8;
  cfg.threads = 1;
  cfg.maxPlies = 24; // short games so several finish inside the time budget
  cfg.openingPlies = 0;
  cfg.seed = 11;

  Viz::Session session(cfg);
  session.set_random_opening(false); // every game from the same position
  session.start();

  // gameIndex -> the longest move list seen for it.
  std::map<int, std::vector<std::string>> games;
  std::set<std::string> starts;
  const auto t0 = std::chrono::steady_clock::now();
  uint64_t seq = 0;
  while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30)) {
    const Viz::Snapshot s = session.wait_for(seq, 200);
    if (s.seq == seq)
      continue;
    seq = s.seq;
    if (s.game.startFen.empty())
      continue;
    starts.insert(s.game.startFen);
    std::vector<std::string> &known = games[s.game.gameIndex];
    if (s.game.moves.size() > known.size())
      known = s.game.moves;
    if (games.size() >= 5)
      break;
  }
  session.stop();

  // Drop the game in progress when the loop broke: it is still growing.
  if (!games.empty())
    games.erase(std::prev(games.end()));
  if (games.size() < 3) {
    std::printf("FAIL: only %zu self-play games completed\n", games.size());
    return false;
  }
  if (starts.size() != 1) {
    std::printf("FAIL: random openings were off but %zu start positions "
                "were used\n",
                starts.size());
    return false;
  }
  std::set<std::vector<std::string>> distinct;
  for (const auto &g : games)
    distinct.insert(g.second);
  if (distinct.size() != games.size()) {
    std::printf("FAIL: %zu games from the same position produced only %zu "
                "distinct -- the engine is replaying itself\n",
                games.size(), distinct.size());
    return false;
  }
  std::printf("PASS: self-play varies (%zu games from one start position, "
              "all %zu lines distinct)\n",
              games.size(), distinct.size());
  return true;
}

// A dataset is written as numbered shards, and a resumed run continues the last
// one instead of restarting the sequence or regenerating games it already has.
//
// The regeneration part is the subtle half: openings and root-move variety are
// drawn from one stream, so a resume that restarts that stream replays the very
// games it resumed past and quietly fills the dataset with duplicate rows.
bool datagen_shard_test() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "stk_viz_shard_test";
  std::error_code ec;
  fs::remove_all(dir, ec);

  Viz::DatagenConfig dg;
  dg.out = dir.string();
  dg.nodes = 250;
  dg.depth = 0;
  dg.maxPlies = 30;
  dg.skipPlies = 4;
  dg.shardPositions = 120;
  dg.targetPositions = 700;

  // Run, stop, then resume into the same directory with a bigger target.
  const auto drive = [&](bool resume, int64_t target) {
    Viz::Config cfg;
    cfg.nodes = 250;
    cfg.moveDelayMs = 0;
    cfg.hashMb = 8;
    Viz::Session session(cfg);
    dg.targetPositions = target;
    if (!session.start_datagen(dg, resume))
      return false;
    session.start();
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(60)) {
      if (!session.snapshot().datagen.running)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    session.stop_datagen();
    session.stop();
    return true;
  };
  if (!drive(false, 400) || !drive(true, 700)) {
    std::printf("FAIL: could not start datagen in %s\n", dir.string().c_str());
    return false;
  }

  std::vector<fs::path> shards;
  for (const auto &e : fs::directory_iterator(dir))
    if (e.path().filename().string().rfind("shard_", 0) == 0)
      shards.push_back(e.path());
  std::sort(shards.begin(), shards.end());
  if (shards.size() < 3) {
    std::printf("FAIL: expected several shards, found %zu\n", shards.size());
    return false;
  }

  std::set<std::string> fens;
  int64_t rows = 0;
  for (size_t i = 0; i < shards.size(); ++i) {
    // The numbering must have no gaps, or the training pipeline silently
    // trains on a subset of what was generated.
    char want[32];
    std::snprintf(want, sizeof(want), "shard_%04zu.txt", i);
    if (shards[i].filename().string() != want) {
      std::printf("FAIL: shard sequence has a gap at %s\n", want);
      return false;
    }
    std::ifstream in(shards[i]);
    std::string line;
    int64_t n = 0;
    while (std::getline(in, line)) {
      if (line.empty())
        continue;
      ++n;
      ++rows;
      fens.insert(line.substr(0, line.find('|')));
    }
    // Every shard but the last holds exactly the requested number of rows.
    if (i + 1 < shards.size() && n != dg.shardPositions) {
      std::printf("FAIL: %s has %lld rows, expected %lld\n",
                  shards[i].filename().string().c_str(),
                  static_cast<long long>(n),
                  static_cast<long long>(dg.shardPositions));
      return false;
    }
  }
  if (static_cast<int64_t>(fens.size()) != rows) {
    std::printf("FAIL: %lld rows but only %zu unique positions -- the resumed "
                "run regenerated games it already had\n",
                static_cast<long long>(rows), fens.size());
    return false;
  }
  fs::remove_all(dir, ec);
  std::printf("PASS: datagen shards (%zu files, %lld rows, all unique across "
              "the resume boundary)\n",
              shards.size(), static_cast<long long>(rows));
  return true;
}

} // namespace

// Emit one encoded state message so the TypeScript decoder can be checked
// against the real C++ encoder. The fixture is generated from this binary at
// test time rather than committed, so the two sides cannot drift: if a field is
// renamed here, the UI's test fails on the next run.
bool write_fixture(const char *path) {
  auto net = std::make_unique<NNUE::Network>();
  net->randomize(0x9E3779B97F4A7C15ULL);

  Core::Position pos;
  if (!pos.setFromFEN(FENS[1]))
    return false;

  Viz::Snapshot s;
  s.seq = 4242;
  s.mode = Viz::Mode::SelfPlay;
  s.running = true;
  s.paused = false;
  s.thinking = true;
  s.nnueActive = true;
  s.threads = 2;
  s.engineColor = 1;
  s.game.fen = pos.toFEN();
  s.game.startFen = FENS[0];
  s.game.moves = {"e2e4", "e7e5", "g1f3"};
  s.game.lastMove = "g1f3";
  s.game.ply = 3;
  s.game.over = false;
  s.game.gameIndex = 7;
  s.game.wins = 2;
  s.game.draws = 3;
  s.game.losses = 1;
  s.search.depth = 11;
  s.search.seldepth = 17;
  s.search.scoreCp = 42;
  s.search.nodes = 123456;
  s.search.tbHits = 5;
  s.search.elapsedMs = 250;
  s.search.nps = 493824;
  s.search.pv = {"d2d4", "d7d5", "c2c4"};
  s.search.qsearchTtHitRate = 31.5;
  s.search.negamaxTtHitRate = 58.25;
  s.search.candidates = {{"d2d4", 42, {"d2d4", "d7d5"}},
                         {"e2e4", 31, {"e2e4", "e7e5"}},
                         {"g1f3", -7, {"g1f3"}}};
  s.legalMoves = {"e2e4", "d2d4"};
  s.frame = Viz::capture(pos, *net);

  const std::string blob = Viz::encode_state(s);
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return false;
  out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
  if (!out)
    return false;
  std::printf("wrote %zu bytes of encoded state to %s\n", blob.size(), path);
  return true;
}

int main(int argc, char **argv) {
  Core::Attacks::init();
  Core::Zobrist::init();

  // Selector so ctest can register each check separately:
  //   test_viz [all|frame|session|fixture <path>]
  const std::string which = argc > 1 ? argv[1] : "all";
  if (which == "fixture") {
    if (argc < 3) {
      std::printf("usage: test_viz fixture <path>\n");
      return 2;
    }
    return write_fixture(argv[2]) ? 0 : 1;
  }
  if (which == "session")
    return session_test() ? 0 : 1;
  if (which == "variety")
    return variety_test() ? 0 : 1;
  if (which == "shards")
    return datagen_shard_test() ? 0 : 1;

  // Deterministic synthetic weights: exercises the full pipeline without
  // depending on a 46 MB net file being present.
  auto net = std::make_unique<NNUE::Network>();
  net->randomize(0x9E3779B97F4A7C15ULL);

  for (const char *fen : FENS) {
    Core::Position pos;
    if (!pos.setFromFEN(fen)) {
      std::printf("FAIL: could not parse fen\n  %s\n", fen);
      return 1;
    }

    const Viz::VizFrame f = Viz::capture(pos, *net);

    // The captured eval must be exactly what the ordinary eval path returns --
    // this is what makes the visualizer show the real engine.
    NNUE::Accumulator acc;
    net->refresh(pos, acc);
    const bool ok =
        (f.eval == net->evaluate(pos, acc) ||
         fail("captured eval != NNUE::Network::evaluate", f.fen)) &&
        (f.eval == f.psqt + f.positional ||
         fail("eval != psqt + positional", f.fen)) &&
        (f.fen == pos.toFEN() || fail("frame fen mismatch", f.fen)) &&
        ((int(f.accUs.size()) == NNUE::Network::HIDDEN &&
          int(f.l1in.size()) == NNUE::Network::HIDDEN &&
          int(f.l1out.size()) == NNUE::Arch::L1 &&
          int(f.l2out.size()) == NNUE::Arch::L2) ||
         fail("layer array sizes do not match the architecture", f.fen)) &&
        check_features(f, pos, f.white, Core::WHITE) &&
        check_features(f, pos, f.black, Core::BLACK) &&
        check_attribution(f, *net) &&
        // Capture is a pure function of (position, net).
        (same_frame(f, Viz::capture(pos, *net)) ||
         fail("capture is not deterministic", f.fen));
    if (!ok)
      return 1;
  }

  std::printf("PASS: viz telemetry exact (eval == engine eval, attribution "
              "reconstructs every layer), deterministic, %d positions\n",
              int(sizeof(FENS) / sizeof(FENS[0])));

  if (which == "all")
    return session_test() ? 0 : 1;
  return 0;
}
