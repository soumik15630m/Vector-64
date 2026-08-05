#ifndef DATAGEN_SELFPLAY_H
#define DATAGEN_SELFPLAY_H

#include "../cores/bitboard.h"
#include "../cores/movegen.h"
#include "../cores/position.h"
#include "../search/transposition_table.h" // is_mate_score

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

// Self-play primitives shared by the data generator (src/datagen/datagen.cpp)
// and the visualizer's self-play mode (src/viz/session.cpp), so both drive
// games by exactly the same rules: the same opening distribution and the same
// cheap draw detection.

namespace Datagen {

constexpr const char *START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr int MATE_CP = 8000;

// Search score -> a bounded white-perspective label; mates map to +/-MATE_CP.
inline int clamp_score(int cp) {
  if (Search::is_mate_score(cp))
    return cp > 0 ? MATE_CP : -MATE_CP;
  return std::max(-MATE_CP, std::min(MATE_CP, cp));
}

// WDL blend in win-probability space (CP_SCALE 400).
inline int blend_cp(int evalWhite, double wdl, double lam) {
  const double e = std::max(-4000.0, std::min(4000.0, double(evalWhite)));
  const double pe = 1.0 / (1.0 + std::exp(-e / 400.0));
  double p = (1.0 - lam) * pe + lam * wdl;
  p = std::min(std::max(p, 1e-4), 1.0 - 1e-4);
  return int(std::lround(400.0 * std::log(p / (1.0 - p))));
}

// One labelled row, exactly as both generators write it.
//   raw   : "<fen> | <eval> | <wdl>"     (bullet-native)
//   blend : "<fen> | <blended cp>"
inline std::string emit_row(const std::string &fen, int scoreCp, double wdl,
                            bool raw, double lam) {
  if (raw) {
    const char *ws = wdl == 1.0 ? "1.0" : (wdl == 0.0 ? "0.0" : "0.5");
    return fen + " | " + std::to_string(scoreCp) + " | " + ws;
  }
  return fen + " | " + std::to_string(blend_cp(scoreCp, wdl, lam));
}

// Draw by material: no pawn, rook or queen, and at most one minor overall.
inline bool insufficient_material(const Core::Position &pos) {
  if (pos.pieces(Core::PAWN) || pos.pieces(Core::ROOK) ||
      pos.pieces(Core::QUEEN))
    return false;
  return Core::popcount(pos.pieces(Core::KNIGHT) | pos.pieces(Core::BISHOP)) <=
         1;
}

// Seeded quiet, material-balanced random opening. false => caller retries.
// `balance` is the maximum |white-black material| in centipawns.
inline bool make_opening(Core::Position &pos, std::mt19937_64 &rng,
                         int openingPlies, int balance) {
  pos.setFromFEN(START_FEN);
  for (int i = 0; i < openingPlies; ++i) {
    Core::MoveList legal;
    Core::generate_legal_moves(pos, legal);
    if (legal.size() == 0)
      return false;
    Core::UndoInfo ui;
    pos.make_move(legal[int(rng() % uint64_t(legal.size()))], ui);
  }
  Core::MoveList legal;
  Core::generate_legal_moves(pos, legal);
  if (legal.size() == 0 || pos.in_check())
    return false;
  return std::abs(pos.material_wb()) <= balance;
}

// A dataset is written as numbered shards in a directory -- shard_0000.txt,
// shard_0001.txt, ... -- which is the layout the training pipeline consumes
// (tools/bullet/build_net.py globs shard_*.txt and holds the last one out as a
// validation set). It also keeps any single file small enough to move, inspect
// and convert: a 500M-position run is ~45 GB, which is not one file you want.
//
// Sharding used to come from tools/nnue/datagen_bulk.py driving the generator
// once per shard. That script is deprecated, so the generator does it itself
// and both callers -- the CLI and the visualizer -- share this writer.
class ShardWriter {
public:
  ShardWriter() = default;
  ShardWriter(const ShardWriter &) = delete;
  ShardWriter &operator=(const ShardWriter &) = delete;

  // `dir` is the dataset directory, created if needed. rowsPerShard <= 0 means
  // "do not shard": `dir` is then taken as a single output FILE, which is what
  // the CLI does by default so existing commands keep behaving as they did.
  // `resume` appends to the last shard instead of truncating from shard 0.
  bool open(const std::string &dir, int64_t rowsPerShard, bool resume) {
    close();
    path_ = dir;
    rowsPerShard_ = rowsPerShard;
    shard_ = 0;
    rowsInShard_ = 0;
    // Backward compatibility: a dataset written before the generator sharded
    // is one plain file. If `path` names an existing FILE, keep writing to
    // that file -- never turn it into a directory, and never strand it.
    std::error_code fec;
    if (std::filesystem::is_regular_file(path_, fec))
      rowsPerShard_ = 0;
    if (rowsPerShard_ <= 0) { // single-file mode
      out_.open(path_,
                std::ios::binary | (resume ? std::ios::app : std::ios::trunc));
      return out_.is_open();
    }
    std::error_code ec;
    std::filesystem::create_directories(path_, ec);
    if (ec)
      return false;
    if (resume) {
      // Continue the highest-numbered shard that exists, so a resumed run does
      // not leave a gap in the sequence or overwrite finished work.
      while (std::filesystem::exists(shard_path(shard_ + 1), ec))
        ++shard_;
      rowsInShard_ = count_rows(shard_path(shard_));
      // A shard that is already full rolls over rather than growing past the
      // size the rest of the pipeline was told to expect.
      if (rowsInShard_ >= rowsPerShard_) {
        ++shard_;
        rowsInShard_ = 0;
      }
    }
    return open_shard(resume);
  }

  bool is_open() const { return out_.is_open(); }

  // Writes one row and rolls over to the next shard once this one is full.
  bool write_row(const std::string &row) {
    if (!out_.is_open())
      return false;
    out_ << row << '\n';
    ++rowsInShard_;
    if (rowsPerShard_ > 0 && rowsInShard_ >= rowsPerShard_) {
      out_.flush();
      out_.close();
      ++shard_;
      rowsInShard_ = 0;
      return open_shard(/*append=*/false);
    }
    return true;
  }

  void flush() {
    if (out_.is_open())
      out_.flush();
  }

  void close() {
    if (out_.is_open()) {
      out_.flush();
      out_.close();
    }
  }

  int shard() const { return shard_; }
  // False when writing one plain file, either because sharding was turned off
  // or because an older single-file dataset was opened.
  bool sharded() const { return rowsPerShard_ > 0; }
  int64_t rows_in_shard() const { return rowsInShard_; }
  // Where rows are going right now, for progress display.
  std::string current_path() const {
    return rowsPerShard_ > 0 ? shard_path(shard_) : path_;
  }

private:
  std::string shard_path(int i) const {
    char name[32];
    std::snprintf(name, sizeof(name), "shard_%04d.txt", i);
    return (std::filesystem::path(path_) / name).string();
  }

  bool open_shard(bool append) {
    out_.open(shard_path(shard_),
              std::ios::binary | (append ? std::ios::app : std::ios::trunc));
    return out_.is_open();
  }

  // Counting is only ever done once, when resuming, so reading the shard back
  // is cheaper than trusting a number that a crash may have left stale.
  static int64_t count_rows(const std::string &file) {
    std::ifstream in(file, std::ios::binary);
    if (!in)
      return 0;
    int64_t n = 0;
    std::string line;
    while (std::getline(in, line))
      if (!line.empty())
        ++n;
    return n;
  }

  std::string path_;
  std::ofstream out_;
  int64_t rowsPerShard_ = 0;
  int64_t rowsInShard_ = 0;
  int shard_ = 0;
};

} // namespace Datagen

#endif
