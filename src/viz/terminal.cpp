#include "terminal.h"

#include "../cores/position.h"

#include <cstdio>
#include <string>

namespace Viz {
namespace {

// ANSI: move to home and clear, so the view redraws in place instead of
// scrolling. Every terminal we target (Windows Terminal, macOS, Linux) handles
// these; a terminal that does not will simply scroll, which is still readable.
constexpr const char *HOME = "\033[H\033[2J";
constexpr const char *DIM = "\033[2m";
constexpr const char *BOLD = "\033[1m";
constexpr const char *CYAN = "\033[36m";
constexpr const char *AMBER = "\033[33m";
constexpr const char *BLUE = "\033[34m";
constexpr const char *OFF = "\033[0m";

// Render the board from the FEN's piece field, rank 8 first.
void draw_board(const std::string &fen) {
  const std::string board = fen.substr(0, fen.find(' '));
  int rank = 8;
  std::printf("    %s+------------------------+%s\n", DIM, OFF);
  std::printf("  %d %s|%s ", rank, DIM, OFF);
  for (char c : board) {
    if (c == '/') {
      std::printf("%s|%s\n  %d %s|%s ", DIM, OFF, --rank, DIM, OFF);
    } else if (c >= '1' && c <= '8') {
      for (int i = 0; i < c - '0'; ++i)
        std::printf("%s . %s", DIM, OFF);
    } else {
      // Upper case is White; colour them so the sides are separable at a
      // glance without relying on the glyph alone.
      const bool white = c >= 'A' && c <= 'Z';
      std::printf("%s %c %s", white ? BOLD : DIM, c, OFF);
    }
  }
  std::printf("%s|%s\n    %s+------------------------+%s\n", DIM, OFF, DIM,
              OFF);
  std::printf("      %sa  b  c  d  e  f  g  h%s\n", DIM, OFF);
}

void bar(int cp) {
  // A short signed bar for the evaluation, clamped to +/-5 pawns.
  const int width = 24;
  const int mid = width / 2;
  int n = cp * mid / 500;
  if (n > mid)
    n = mid;
  if (n < -mid)
    n = -mid;
  std::printf("  [");
  for (int i = -mid; i < mid; ++i) {
    if (i == 0)
      std::printf("%s|%s", DIM, OFF);
    else if ((n > 0 && i > 0 && i <= n) || (n < 0 && i < 0 && i >= n))
      std::printf("%s=%s", n > 0 ? AMBER : BLUE, OFF);
    else
      std::printf("%s.%s", DIM, OFF);
  }
  std::printf("]\n");
}

} // namespace

int run_terminal(Session &session) {
  std::printf("%s", HOME);
  std::printf("STK-Vector-64 - Vector Scope (terminal). Ctrl-C to stop.\n");

  uint64_t seen = 0;
  for (;;) {
    const Snapshot s = session.wait_for(seen, 1000);
    if (!s.running && s.seq == seen)
      break;
    seen = s.seq;

    // Root side to move, for scores reported at the root.
    const int stm = s.game.fen.find(" b ") != std::string::npos ? -1 : 1;
    // The frame is captured at the PV leaf, whose side to move often differs
    // from the root's -- convert it with its own, or the sign flips and the
    // evaluation disagrees with the candidate list.
    const int white = s.nnueActive
                          ? s.frame.eval * (s.frame.sideToMove == 0 ? 1 : -1)
                          : s.search.scoreCp * stm;

    std::printf("%s", HOME);
    std::printf("%sSTK-Vector-64%s  %s/ Vector Scope%s   %s%s%s\n\n", BOLD, OFF,
                DIM, OFF, CYAN, mode_name(s.mode), OFF);
    draw_board(s.game.fen);
    std::printf("\n  eval %s%+.2f%s (white)   game #%d   ply %d\n",
                white >= 0 ? AMBER : BLUE, white / 100.0, OFF, s.game.gameIndex,
                s.game.ply);
    bar(white);

    if (s.nnueActive)
      std::printf("\n  psqt %+d   positional %+d   bucket %d\n", s.frame.psqt,
                  s.frame.positional, s.frame.bucket);
    if (s.compareActive)
      std::printf("  compare %s: %+d cp (delta %+d)\n", s.compareName.c_str(),
                  s.compareFrame.eval, s.frame.eval - s.compareFrame.eval);

    std::printf("\n  depth %d/%d   nodes %llu   %d knps   %d thread%s\n",
                s.search.depth, s.search.seldepth,
                static_cast<unsigned long long>(s.search.nodes),
                s.search.nps / 1000, s.threads, s.threads == 1 ? "" : "s");

    if (!s.search.candidates.empty()) {
      std::printf("\n  %scandidates%s\n", DIM, OFF);
      for (size_t i = 0; i < s.search.candidates.size(); ++i) {
        const Candidate &c = s.search.candidates[i];
        std::printf("   %s%zu%s %-6s %s%+7.2f%s\n", DIM, i + 1, OFF,
                    c.move.c_str(), i == 0 ? CYAN : DIM,
                    c.scoreCp * stm / 100.0, OFF);
      }
    }
    if (!s.search.pv.empty()) {
      std::printf("\n  %spv%s ", DIM, OFF);
      for (const std::string &m : s.search.pv)
        std::printf("%s ", m.c_str());
      std::printf("\n");
    }
    if (s.game.over)
      std::printf("\n  %sresult %s (%s)%s\n", BOLD, s.game.result.c_str(),
                  s.game.reason.c_str(), OFF);
    std::fflush(stdout);
  }
  return 0;
}

} // namespace Viz
