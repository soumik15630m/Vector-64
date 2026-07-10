#include "network.h"

#include <cstdio>
#include <memory>

namespace NNUE {

using namespace Arch;

template <int H> bool NetworkT<H>::self_test() {
  auto net = std::make_unique<NetworkT<H>>();
  net->randomize(0x1234567ULL);

  // 1) SIMD forward == pure-scalar forward, over random accumulators.
  uint64_t s = 0x9E3779B97F4A7C15ULL;
  auto next = [&s]() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  };
  auto scalar_forward = [&](const AccumulatorT<H> &a, Core::Color stm,
                            int bucket) -> int {
    const auto &us = a.acc[stm];
    const auto &them = a.acc[~stm];
    std::array<uint8_t, H> l1in{};
    for (int i = 0; i < PAIR; ++i) {
      l1in[i] = static_cast<uint8_t>(
          (detail::clip(us[i]) * detail::clip(us[i + PAIR])) >> PAIR_SHIFT);
      l1in[PAIR + i] = static_cast<uint8_t>(
          (detail::clip(them[i]) * detail::clip(them[i + PAIR])) >> PAIR_SHIFT);
    }
    const auto &b = net->buckets_[bucket];
    std::array<uint8_t, L1> l1o{};
    for (int o = 0; o < L1; ++o) {
      int acc = b.l1b[o];
      for (int i = 0; i < H; ++i)
        acc += static_cast<int>(l1in[i]) * static_cast<int>(b.l1w[o * H + i]);
      l1o[o] = detail::clip(acc >> L1_SHIFT);
    }
    std::array<uint8_t, L2> l2o{};
    for (int o = 0; o < L2; ++o) {
      int acc = b.l2b[o];
      for (int i = 0; i < L1; ++i)
        acc += static_cast<int>(l1o[i]) * static_cast<int>(b.l2w[o * L1 + i]);
      l2o[o] = detail::clip(acc >> L2_SHIFT);
    }
    int raw = b.outb;
    for (int i = 0; i < L2; ++i)
      raw += static_cast<int>(l2o[i]) * static_cast<int>(b.outw[i]);
    const int positional = raw >> OUT_SHIFT;
    const int psqtTerm =
        (a.psqt[stm][bucket] - a.psqt[~stm][bucket]) >> PSQT_SHIFT;
    return positional + psqtTerm;
  };

  for (int trial = 0; trial < 128; ++trial) {
    AccumulatorT<H> a{};
    for (int p = 0; p < 2; ++p) {
      for (int h = 0; h < H; ++h)
        a.acc[p][h] = static_cast<int16_t>(int(next() % 4000) - 2000);
      for (int k = 0; k < PSQT_BUCKETS; ++k)
        a.psqt[p][k] = static_cast<int32_t>(int(next() % 40000) - 20000);
    }
    const int bucket = static_cast<int>(next() % PSQT_BUCKETS);
    for (Core::Color stm : {Core::WHITE, Core::BLACK}) {
      const int simd = net->forward(a, stm, bucket, nullptr);
      const int scal = scalar_forward(a, stm, bucket);
      if (simd != scal) {
        std::fprintf(stderr,
                     "[nnue H=%d] forward mismatch trial %d stm %d: %d vs %d\n",
                     H, trial, int(stm), simd, scal);
        return false;
      }
    }
  }

  // 2) Incremental delta == direct build, over random feature sets.
  auto build = [&](Core::Color p, const int *feats, int n, AccumulatorT<H> &a) {
    a.acc[p] = net->ftBias_;
    a.psqt[p].fill(0);
    net->apply_delta(p, feats, n, nullptr, 0, a);
  };
  for (int trial = 0; trial < 64; ++trial) {
    const Core::Color p = (next() & 1) ? Core::BLACK : Core::WHITE;
    int s1[24], s2[24];
    const int n = 8 + static_cast<int>(next() % 16);
    for (int i = 0; i < n; ++i) {
      s1[i] = static_cast<int>(next() % FEATURES);
      s2[i] = static_cast<int>(next() % FEATURES);
    }
    AccumulatorT<H> direct{}, delta{};
    build(p, s2, n, direct);
    build(p, s1, n, delta);
    net->apply_delta(p, s2, n, s1, n, delta); // add s2, remove s1
    if (delta.acc[p] != direct.acc[p] || delta.psqt[p] != direct.psqt[p]) {
      std::fprintf(stderr, "[nnue H=%d] incremental != rebuild, trial %d\n", H,
                   trial);
      return false;
    }
  }

  // 3) SIMD accumulator add == scalar reference.
  {
    int feats[10];
    for (int &f : feats)
      f = static_cast<int>(next() % FEATURES);
    AccumulatorT<H> a{};
    a.acc[Core::WHITE] = net->ftBias_;
    a.psqt[Core::WHITE].fill(0);
    net->apply_delta(Core::WHITE, feats, 10, nullptr, 0, a);
    std::array<int16_t, H> ref = net->ftBias_;
    for (int f : feats) {
      const int16_t *col = net->ft_col(f);
      for (int h = 0; h < H; ++h)
        ref[h] = static_cast<int16_t>(ref[h] + col[h]);
    }
    if (a.acc[Core::WHITE] != ref) {
      std::fprintf(stderr, "[nnue H=%d] accumulator SIMD add != scalar\n", H);
      return false;
    }
  }
  return true;
}

template bool NetworkT<1024>::self_test();
template bool NetworkT<128>::self_test();

} // namespace NNUE
