#ifndef NNUE_NETWORK_H
#define NNUE_NETWORK_H

#include "../cores/position.h"
#include "halfka.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// STK-HalfKA NNUE v2 (engine side, from scratch).
//
//   features (22528/persp)  --FT-->  accumulator[2048]  (int16, incremental)
//        also  --FT-PSQT-->  psqt[8]  (int32 side-output, one per bucket)
//   pairwise clipped-ReLU:  2048 -> 1024   per perspective
//   L1 (per bucket):  concat(us1024, them1024)=2048 -> 16   int8
//   L2 (per bucket):  16 -> 32
//   out (per bucket): 32 -> 1
//   eval_stm = out/OUT_SCALE + (psqt_us[b]-psqt_them[b])/PSQT_SCALE
//   bucket b selected by piece count.
//
// Fixed-point: FT weights/bias int16, accumulator int16, activations clipped to
// [0,127], dense weights int8. All scales are fixed shifts so the integer path
// reproduces the (future) quantization-aware-trained float model exactly.

namespace NNUE {

namespace Arch {
constexpr int FEATURES = HalfKA::FEATURES; // 22528
constexpr int HIDDEN = 2048;               // accumulator width / perspective
constexpr int PSQT_BUCKETS = 8;
constexpr int PAIR = HIDDEN / 2; // 256, pairwise outputs / perspective
constexpr int L1_IN = 2 * PAIR;  // 512, concat(us, them)
constexpr int L1 = 16;
constexpr int L2 = 32;

constexpr int ACT_MAX = 127;  // clipped-ReLU ceiling
constexpr int PAIR_SHIFT = 7; // pairwise product downscale
constexpr int L1_SHIFT = 6;   // dense accumulate downscale (/64)
constexpr int L2_SHIFT = 6;
constexpr int OUT_SHIFT = 4;  // output -> centipawns
constexpr int PSQT_SHIFT = 4; // psqt side-output -> centipawns
} // namespace Arch

struct alignas(64) Accumulator {
  alignas(64) std::array<std::array<int16_t, Arch::HIDDEN>, 2> acc{};
  alignas(64) std::array<std::array<int32_t, Arch::PSQT_BUCKETS>, 2> psqt{};
  bool computed = false;
};

// Dense weights for one output bucket.
struct Bucket {
  std::array<int8_t, Arch::L1 * Arch::L1_IN> l1w{};
  std::array<int32_t, Arch::L1> l1b{};
  std::array<int8_t, Arch::L2 * Arch::L1> l2w{};
  std::array<int32_t, Arch::L2> l2b{};
  std::array<int8_t, Arch::L2> outw{};
  int32_t outb = 0;
};

// Optional per-eval activation snapshot for the glass-box probe.
struct Probe {
  std::array<int16_t, Arch::HIDDEN> accUs{};
  std::array<int16_t, Arch::HIDDEN> accThem{};
  std::array<uint8_t, Arch::L1_IN> l1in{};
  std::array<uint8_t, Arch::L1> l1out{};
  std::array<uint8_t, Arch::L2> l2out{};
  int bucket = 0;
  int psqt = 0;
  int positional = 0;
  int eval = 0;
};

class Network {
public:
  bool load_file(const std::string &path);
  bool is_loaded() const { return loaded_; }

  // Full (non-incremental) accumulator build for `pos`.
  void refresh(const Core::Position &pos, Accumulator &a) const;
  // Refresh a single perspective (used after that side's king moves).
  void refresh_perspective(const Core::Position &pos, Core::Color persp,
                           Accumulator &a) const;

  // Incremental primitive: apply added/removed features to one perspective.
  void apply_delta(Core::Color persp, const int *added, int nAdded,
                   const int *removed, int nRemoved, Accumulator &a) const;

  // Derive `child` from `parent` for the move `m` (already played, so `after`
  // is the resulting position and `ui` carries the captured piece). A
  // perspective is refreshed only when its own king moved; otherwise the few
  // changed features are applied incrementally.
  void update(const Accumulator &parent, Accumulator &child,
              const Core::Position &after, Core::Move m,
              const Core::UndoInfo &ui) const;

  // Side-to-move centipawn evaluation from a built accumulator.
  int evaluate(const Core::Position &pos, const Accumulator &a) const;
  // Same, but also fills `probe` (for the visualizer).
  int evaluate_probe(const Core::Position &pos, const Accumulator &a,
                     Probe &probe) const;

  // Deterministic random weights for the self-test (no file needed).
  void randomize(uint64_t seed);

  // Verifies AVX2 inference is bit-exact with the scalar reference and that the
  // incremental accumulator primitive matches a full rebuild. Build self-test.
  static bool self_test();

  static int bucket_of(const Core::Position &pos);

private:
  const int16_t *ft_col(int feature) const {
    return ftWeights_.data() + static_cast<size_t>(feature) * Arch::HIDDEN;
  }
  const int32_t *ft_psqt(int feature) const {
    return ftPsqt_.data() + static_cast<size_t>(feature) * Arch::PSQT_BUCKETS;
  }

  int forward(const Accumulator &a, Core::Color stm, int bucket,
              Probe *probe) const;

  bool loaded_ = false;
  std::vector<int16_t> ftWeights_; // [FEATURES * HIDDEN]
  std::array<int16_t, Arch::HIDDEN> ftBias_{};
  std::vector<int32_t> ftPsqt_; // [FEATURES * PSQT_BUCKETS]
  std::array<Bucket, Arch::PSQT_BUCKETS> buckets_{};
};

} // namespace NNUE

#endif
