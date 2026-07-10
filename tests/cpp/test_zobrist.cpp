// Plays random legal games and verifies that the incrementally maintained
// position state (zobrist hash, mailbox, material and psqt scores) always
// matches a from-scratch rebuild, and that make/unmake round-trips exactly.
#include "cores/attacks.h"
#include "cores/movegen.h"
#include "cores/position.h"
#include "cores/zobrist.h"
#include "nnue/network.h"

#include <cstdio>
#include <random>

using namespace Core;

namespace {
constexpr const char *STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

bool check_against_rebuild(const Position &pos, int game, int ply) {
  Position rebuilt;
  if (!rebuilt.setFromFEN(pos.toFEN())) {
    std::printf("FAIL game %d ply %d: toFEN/setFromFEN round trip failed\n%s\n",
                game, ply, pos.toFEN().c_str());
    return false;
  }
  if (rebuilt.hash() != pos.hash()) {
    std::printf("FAIL game %d ply %d: incremental hash drifted\n%s\n", game,
                ply, pos.toFEN().c_str());
    return false;
  }
  if (rebuilt.material_wb() != pos.material_wb() ||
      rebuilt.psqt_wb() != pos.psqt_wb()) {
    std::printf("FAIL game %d ply %d: incremental material/psqt drifted "
                "(mat %d vs %d, psqt %d vs %d)\n%s\n",
                game, ply, pos.material_wb(), rebuilt.material_wb(),
                pos.psqt_wb(), rebuilt.psqt_wb(), pos.toFEN().c_str());
    return false;
  }
  for (int s = 0; s < SQUARE_NB; ++s) {
    if (rebuilt.piece_on(static_cast<Square>(s)) !=
        pos.piece_on(static_cast<Square>(s))) {
      std::printf("FAIL game %d ply %d: mailbox drifted at square %d\n%s\n",
                  game, ply, s, pos.toFEN().c_str());
      return false;
    }
  }
  return true;
}
} // namespace

namespace {
bool list_contains(const MoveList &list, Move m) {
  for (int i = 0; i < list.size(); ++i) {
    if (list[i] == m)
      return true;
  }
  return false;
}

// The staged generators plus the fast legality path must reproduce
// generate_legal_moves exactly, and is_pseudo_legal must accept exactly
// the generator output (no false accepts that could corrupt make_move).
bool check_movegen_equivalence(Position &pos, std::mt19937_64 &rng, int game,
                               int ply) {
  MoveList full;
  generate_legal_moves(pos, full);

  MoveList captures;
  MoveList quiets;
  generate_pseudo_captures(pos, captures);
  generate_pseudo_quiets(pos, quiets);
  const NodeLegality nl = make_node_legality(pos);

  MoveList staged;
  for (int i = 0; i < captures.size(); ++i) {
    if (is_legal(nl, captures[i]))
      staged.push_back(captures[i]);
  }
  for (int i = 0; i < quiets.size(); ++i) {
    if (is_legal(nl, quiets[i]))
      staged.push_back(quiets[i]);
  }

  if (staged.size() != full.size()) {
    std::printf("FAIL game %d ply %d: staged %d vs full %d moves\n%s\n", game,
                ply, staged.size(), full.size(), pos.toFEN().c_str());
    return false;
  }
  for (int i = 0; i < full.size(); ++i) {
    if (!list_contains(staged, full[i])) {
      std::printf("FAIL game %d ply %d: staged set missing a legal move\n%s\n",
                  game, ply, pos.toFEN().c_str());
      return false;
    }
    if (!is_pseudo_legal(pos, full[i])) {
      std::printf("FAIL game %d ply %d: is_pseudo_legal rejects legal move "
                  "%d->%d\n%s\n",
                  game, ply, full[i].from_sq(), full[i].to_sq(),
                  pos.toFEN().c_str());
      return false;
    }
  }

  MoveList pseudoAll;
  generate_pseudo_legal_moves(pos, pseudoAll);
  for (int k = 0; k < 64; ++k) {
    const Move m(static_cast<uint16_t>(rng()));
    if (is_pseudo_legal(pos, m) && !list_contains(pseudoAll, m)) {
      std::printf("FAIL game %d ply %d: is_pseudo_legal accepts raw 0x%04x "
                  "not produced by the generators\n%s\n",
                  game, ply, m.raw(), pos.toFEN().c_str());
      return false;
    }
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  Attacks::init();
  Zobrist::init();

  // Optional selector so CTest can register each check as its own test:
  //   test_zobrist [all|nnue|games]   (default: all)
  const std::string which = argc > 1 ? argv[1] : "all";
  const bool runNnue = (which == "all" || which == "nnue");
  const bool runGames = (which == "all" || which == "games");

  if (runNnue) {
    if (!NNUE::Network::self_test()) {
      std::printf("FAIL: NNUE inference/accumulator self-test failed\n");
      return 1;
    }
    std::printf("PASS: NNUE inference bit-exact (SIMD == scalar) and "
                "incremental == rebuild\n");
  }
  if (!runGames)
    return 0;

  constexpr int GAMES = 200;
  constexpr int MAX_PLIES = 120;
  std::mt19937_64 rng(0xC0FFEEULL);

  NNUE::Network net;
  net.randomize(0xACC0FFEEULL);
  // Shared across all games so cache entries go stale and get diffed, which
  // is exactly the path that must still equal a full rebuild.
  auto refreshTable = std::make_unique<NNUE::RefreshTable>();

  for (int game = 0; game < GAMES; ++game) {
    Position pos;
    pos.setFromFEN(STARTPOS_FEN);
    NNUE::Accumulator acc;
    net.refresh(pos, acc);

    for (int ply = 0; ply < MAX_PLIES; ++ply) {
      MoveList moves;
      generate_legal_moves(pos, moves);
      if (moves.size() == 0)
        break;

      const Move move =
          moves[static_cast<int>(rng() % static_cast<uint64_t>(moves.size()))];

      const uint64_t hashBefore = pos.hash();
      const int matBefore = pos.material_wb();
      const int psqtBefore = pos.psqt_wb();
      const std::string fenBefore = pos.toFEN();

      if (!check_movegen_equivalence(pos, rng, game, ply))
        return 1;

      // key_after(m) must equal the real post-move hash, or the search's
      // pre-make_move TT prefetch would target the wrong bucket.
      const uint64_t predicted = pos.key_after(move);

      UndoInfo undo{};
      pos.make_move(move, undo);
      if (pos.hash() != predicted) {
        std::printf("FAIL game %d ply %d: key_after != post-move hash\n%s\n",
                    game, ply, pos.toFEN().c_str());
        return 1;
      }
      if (!check_against_rebuild(pos, game, ply))
        return 1;

      pos.unmake_move(move, undo);
      if (pos.hash() != hashBefore || pos.material_wb() != matBefore ||
          pos.psqt_wb() != psqtBefore || pos.toFEN() != fenBefore) {
        std::printf("FAIL game %d ply %d: unmake did not restore state\n%s\n",
                    game, ply, fenBefore.c_str());
        return 1;
      }

      pos.make_move(move, undo);

      NNUE::Accumulator child;
      net.update(acc, child, pos, move, undo, refreshTable.get());
      NNUE::Accumulator fresh;
      net.refresh(pos, fresh);
      if (child.acc != fresh.acc || child.psqt != fresh.psqt) {
        std::printf("FAIL game %d ply %d: NNUE incremental != rebuild\n%s\n",
                    game, ply, pos.toFEN().c_str());
        return 1;
      }
      acc = child;
    }
  }

  std::printf("PASS: %d random games — make/unmake state, staged movegen "
              "equivalence and is_pseudo_legal all consistent\n",
              GAMES);
  return 0;
}
