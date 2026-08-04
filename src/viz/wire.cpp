#include "wire.h"

#include "../nnue/halfka.h"

#include <json.hpp>

#include <cmath>
#include <cstring>
#include <vector>

namespace Viz {
namespace {

using json = nlohmann::json;

// Append raw little-endian elements and record the buffer in the header table.
// Everything the engine produces is already little-endian on every platform we
// ship (x86-64 and arm64), so this is a straight memcpy.
template <typename T>
void push_buffer(std::string &payload, json &table, const char *name,
                 const char *type, const T *data, size_t count) {
  table.push_back({{"name", name}, {"type", type}, {"len", count}});
  if (count)
    payload.append(reinterpret_cast<const char *>(data), count * sizeof(T));
}

// Features flatten to 6 int32s each: square, orientedSquare, colour, type,
// kind, featureIndex.
std::vector<int32_t> flatten_features(const std::vector<ActiveFeature> &f) {
  std::vector<int32_t> out;
  out.reserve(f.size() * 6);
  for (const ActiveFeature &a : f) {
    out.push_back(a.square);
    out.push_back(a.orientedSquare);
    out.push_back(a.pieceColor);
    out.push_back(a.pieceType);
    out.push_back(a.pieceKind);
    out.push_back(a.featureIndex);
  }
  return out;
}

std::vector<int32_t> flatten_contrib(const std::vector<Contribution> &c) {
  std::vector<int32_t> out;
  out.reserve(c.size() * 2);
  for (const Contribution &x : c) {
    out.push_back(x.index);
    out.push_back(x.value);
  }
  return out;
}

json perspective_json(const PerspectiveInput &p) {
  return json{{"kingSquare", p.kingSquare},
              {"kingBucket", p.kingBucket},
              {"mirrored", p.mirrored},
              {"featureCount", p.features.size()}};
}

} // namespace

std::string encode_state(const Snapshot &s) {
  json h;
  h["seq"] = s.seq;
  h["mode"] = mode_name(s.mode);
  h["running"] = s.running;
  h["paused"] = s.paused;
  h["thinking"] = s.thinking;
  h["nnueActive"] = s.nnueActive;
  h["threads"] = s.threads;
  h["engineColor"] = s.engineColor;

  h["game"] = {{"fen", s.game.fen},
               {"startFen", s.game.startFen},
               {"moves", s.game.moves},
               {"lastMove", s.game.lastMove},
               {"ply", s.game.ply},
               {"over", s.game.over},
               {"result", s.game.result},
               {"reason", s.game.reason},
               {"gameIndex", s.game.gameIndex},
               {"wins", s.game.wins},
               {"draws", s.game.draws},
               {"losses", s.game.losses}};

  h["search"] = {{"depth", s.search.depth},
                 {"seldepth", s.search.seldepth},
                 {"scoreCp", s.search.scoreCp},
                 {"nodes", s.search.nodes},
                 {"tbHits", s.search.tbHits},
                 {"elapsedMs", s.search.elapsedMs},
                 {"nps", s.search.nps},
                 {"pv", s.search.pv},
                 {"qsearchTtHitRate", s.search.qsearchTtHitRate},
                 {"negamaxTtHitRate", s.search.negamaxTtHitRate}};

  json cands = json::array();
  for (const Candidate &c : s.search.candidates)
    cands.push_back({{"move", c.move}, {"scoreCp", c.scoreCp}, {"pv", c.pv}});
  h["candidates"] = cands;

  h["legalMoves"] = s.legalMoves;

  // Architecture constants, so the UI lays out the network from the engine's
  // own numbers instead of hard-coding them.
  h["arch"] = {{"hidden", NNUE::Network::HIDDEN},
               {"pair", NNUE::Network::PAIR},
               {"l1", NNUE::Arch::L1},
               {"l2", NNUE::Arch::L2},
               {"psqtBuckets", NNUE::Arch::PSQT_BUCKETS},
               {"kingBuckets", NNUE::HalfKA::KING_BUCKETS},
               {"pieceKinds", NNUE::HalfKA::PIECE_KINDS},
               {"features", NNUE::Arch::FEATURES},
               {"actMax", NNUE::Arch::ACT_MAX}};

  const VizFrame &f = s.frame;
  const std::vector<int32_t> wf = flatten_features(f.white.features);
  const std::vector<int32_t> bf = flatten_features(f.black.features);
  const std::vector<int32_t> l1t = flatten_contrib(f.l1Top);

  std::string payload;
  payload.reserve(16384);
  json buffers = json::array();
  push_buffer(payload, buffers, "accUs", "i16", f.accUs.data(), f.accUs.size());
  push_buffer(payload, buffers, "accThem", "i16", f.accThem.data(),
              f.accThem.size());
  push_buffer(payload, buffers, "l1in", "u8", f.l1in.data(), f.l1in.size());
  push_buffer(payload, buffers, "l1out", "u8", f.l1out.data(), f.l1out.size());
  push_buffer(payload, buffers, "l2out", "u8", f.l2out.data(), f.l2out.size());
  push_buffer(payload, buffers, "outContrib", "i32", f.outContrib.data(),
              f.outContrib.size());
  push_buffer(payload, buffers, "l2Contrib", "i32", f.l2Contrib.data(),
              f.l2Contrib.size());
  push_buffer(payload, buffers, "l1Top", "i32", l1t.data(), l1t.size());
  push_buffer(payload, buffers, "whiteFeatures", "i32", wf.data(), wf.size());
  push_buffer(payload, buffers, "blackFeatures", "i32", bf.data(), bf.size());

  h["frame"] = {{"fen", f.fen},
                {"sideToMove", f.sideToMove},
                {"bucket", f.bucket},
                {"psqt", f.psqt},
                {"positional", f.positional},
                {"eval", f.eval},
                {"l1TopK", f.l1TopK},
                {"white", perspective_json(f.white)},
                {"black", perspective_json(f.black)},
                {"buffers", buffers}};

  const std::string header = h.dump();
  std::string out;
  out.reserve(4 + header.size() + payload.size());
  const uint32_t n = static_cast<uint32_t>(header.size());
  const char lenLe[4] = {
      static_cast<char>(n & 0xFF), static_cast<char>((n >> 8) & 0xFF),
      static_cast<char>((n >> 16) & 0xFF), static_cast<char>((n >> 24) & 0xFF)};
  out.append(lenLe, 4);
  out.append(header);
  out.append(payload);
  return out;
}

std::string handle_control(Session &session, const std::string &body,
                           int &httpStatus) {
  httpStatus = 200;
  const auto reject = [&](const std::string &why) {
    httpStatus = 400;
    return json{{"ok", false}, {"error", why}}.dump();
  };

  json j = json::parse(body, nullptr, false);
  if (j.is_discarded() || !j.is_object())
    return reject("body must be a JSON object");
  if (!j.contains("cmd") || !j["cmd"].is_string())
    return reject("missing 'cmd'");

  const std::string cmd = j["cmd"].get<std::string>();
  const auto intVal = [&](int def) {
    return j.contains("value") && j["value"].is_number_integer()
               ? j["value"].get<int>()
               : def;
  };

  if (cmd == "pause") {
    session.set_paused(j.contains("value") && j["value"].is_boolean()
                           ? j["value"].get<bool>()
                           : true);
  } else if (cmd == "step") {
    session.step();
  } else if (cmd == "newgame") {
    session.new_game();
  } else if (cmd == "delay") {
    session.set_move_delay(intVal(0));
  } else if (cmd == "nodes") {
    session.set_nodes(intVal(20000));
  } else if (cmd == "enginecolor") {
    session.set_engine_color(intVal(1));
  } else if (cmd == "mode") {
    Mode m;
    if (!j.contains("value") || !j["value"].is_string() ||
        !mode_from_name(j["value"].get<std::string>(), m))
      return reject("unknown mode");
    session.set_mode(m);
  } else if (cmd == "position") {
    const std::string fen = j.contains("fen") && j["fen"].is_string()
                                ? j["fen"].get<std::string>()
                                : std::string();
    std::vector<std::string> moves;
    if (j.contains("moves") && j["moves"].is_array())
      for (const auto &m : j["moves"])
        if (m.is_string())
          moves.push_back(m.get<std::string>());
    if (!session.set_position(fen, moves))
      return reject("illegal position or move list");
  } else if (cmd == "move") {
    if (!j.contains("value") || !j["value"].is_string() ||
        !session.play_move(j["value"].get<std::string>()))
      return reject("illegal move");
  } else {
    return reject("unknown command '" + cmd + "'");
  }
  return json{{"ok", true}}.dump();
}

std::string encode_net_info(const NNUE::Network &net, bool loaded) {
  json j;
  j["loaded"] = loaded;
  j["arch"] = {{"hidden", NNUE::Network::HIDDEN},
               {"l1", NNUE::Arch::L1},
               {"l2", NNUE::Arch::L2},
               {"psqtBuckets", NNUE::Arch::PSQT_BUCKETS},
               {"kingBuckets", NNUE::HalfKA::KING_BUCKETS},
               {"pieceKinds", NNUE::HalfKA::PIECE_KINDS},
               {"features", NNUE::Arch::FEATURES}};

  // The king-bucket map: which bucket each oriented king square selects. Cheap
  // and fully determined by the feature set, so it is always available.
  json kb = json::array();
  for (int sq = 0; sq < 64; ++sq) {
    const NNUE::HalfKA::Orient o =
        NNUE::HalfKA::make_orient(Core::WHITE, Core::Square(sq));
    kb.push_back(
        {{"square", sq}, {"bucket", o.kingBucket}, {"mirror", o.mirror}});
  }
  j["kingBucketMap"] = kb;

  if (!loaded)
    return j.dump();

  // Per-bucket dense-layer weight statistics.
  json buckets = json::array();
  for (int b = 0; b < NNUE::Arch::PSQT_BUCKETS; ++b) {
    const NNUE::Network::Bucket &bk = net.bucket_weights(b);
    // Accumulate in double: the weights are genuinely signed int8, and going
    // straight to a floating-point accumulator keeps the sign semantics
    // unambiguous (a signed-char -> int conversion here reads as a bug).
    const auto stats = [](const int8_t *w, size_t n) {
      double sum = 0, sq = 0;
      double lo = 127.0, hi = -128.0;
      for (size_t i = 0; i < n; ++i) {
        const double v = w[i];
        sum += v;
        sq += v * v;
        lo = v < lo ? v : lo;
        hi = v > hi ? v : hi;
      }
      const double mean = n ? sum / double(n) : 0.0;
      const double var = n ? sq / double(n) - mean * mean : 0.0;
      return json{{"min", static_cast<int>(lo)},
                  {"max", static_cast<int>(hi)},
                  {"mean", mean},
                  {"stddev", var > 0 ? std::sqrt(var) : 0.0},
                  {"count", n}};
    };
    buckets.push_back({{"bucket", b},
                       {"l1", stats(bk.l1w.data(), bk.l1w.size())},
                       {"l2", stats(bk.l2w.data(), bk.l2w.size())},
                       {"out", stats(bk.outw.data(), bk.outw.size())},
                       {"outBias", bk.outb}});
  }
  j["buckets"] = buckets;

  // Feature-transformer weight histogram over all FEATURES * HIDDEN int16s.
  // 64 bins spanning the observed range; this is the "shape" of the net.
  constexpr int BINS = 64;
  std::vector<int64_t> hist(BINS, 0);
  int lo = 32767, hi = -32768;
  const size_t total =
      static_cast<size_t>(NNUE::Arch::FEATURES) * NNUE::Network::HIDDEN;
  for (int f = 0; f < NNUE::Arch::FEATURES; ++f) {
    const int16_t *col = net.ft_column(f);
    for (int i = 0; i < NNUE::Network::HIDDEN; ++i) {
      const int v = col[i];
      lo = v < lo ? v : lo;
      hi = v > hi ? v : hi;
    }
  }
  const double span = hi > lo ? double(hi - lo) : 1.0;
  for (int f = 0; f < NNUE::Arch::FEATURES; ++f) {
    const int16_t *col = net.ft_column(f);
    for (int i = 0; i < NNUE::Network::HIDDEN; ++i) {
      int bin = int((double(col[i] - lo) / span) * (BINS - 1));
      bin = bin < 0 ? 0 : (bin >= BINS ? BINS - 1 : bin);
      ++hist[static_cast<size_t>(bin)];
    }
  }
  j["ftWeights"] = {{"min", lo}, {"max", hi}, {"count", total}, {"bins", hist}};
  return j.dump();
}

} // namespace Viz
