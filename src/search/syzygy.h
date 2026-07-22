#ifndef SEARCH_SYZYGY_H
#define SEARCH_SYZYGY_H

#include "../cores/movegen.h"
#include "../cores/position.h"

#include <string>

// Thin C++ front-end over the vendored Fathom Syzygy prober (src/syzygy). All
// tablebase state is global (Fathom owns it); this namespace just marshals a
// Core::Position into Fathom's bitboard call and maps the result back.
namespace Search::Syzygy {

// Game-theoretic value for a tablebase win, from the side to move. Sits above
// every normal evaluation but below the mate band (MATE_BOUND), so a real mate
// still outranks it and TB scores never register as mates. Flat (no ply
// offset), which keeps it consistent through the transposition table.
constexpr int VALUE_TB_WIN = 800000;

enum class Wdl { FAIL, LOSS, DRAW, WIN };

// Load tablebases from `path` (Fathom's SEP_CHAR-separated directory list).
// Returns the largest piece count available (0 = none found / disabled).
// Safe to call repeatedly; frees any previously loaded set first.
int init(const std::string &path);
void shutdown();

int max_pieces(); // TB_LARGEST, 0 when inactive
bool active();    // max_pieces() > 0

// WDL probe for use inside the search (thread-safe). The caller must have
// already checked piece_count <= max_pieces(). Returns FAIL when the tables
// cannot answer (non-zero rule50, castling rights, or a missing file).
Wdl probe_wdl(const Core::Position &pos);

// DTZ root ranking (NOT thread-safe -- master thread only, once per search).
// Appends every root move that preserves the best tablebase outcome -- all
// equally-optimal moves, so the search still chooses naturally among them --
// to `out`. Returns the number appended; 0 means the root could not be ranked
// (leave the move list untouched). Falls back to WDL ranking when DTZ tables
// are missing.
int probe_root_moves(Core::Position &pos, Core::MoveList &out);

} // namespace Search::Syzygy

#endif
