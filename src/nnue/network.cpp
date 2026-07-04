#include "network.h"

#include "../cores/bitboard.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace NNUE {

using namespace Arch;

namespace {

constexpr char MAGIC[9] = {'S', 'T', 'K', 'H', 'A', 'L', 'F', 'K', 'A'};
constexpr uint32_t VERSION = 1;

inline uint8_t clip(int v) {
  return static_cast<uint8_t>(v < 0 ? 0 : (v > ACT_MAX ? ACT_MAX : v));
}

// Pairwise clipped-ReLU: 512 int16 accumulator -> 256 uint8 activations.
void pairwise(const int16_t *RESTRICT acc, uint8_t *RESTRICT out) {
  for (int i = 0; i < PAIR; ++i) {
    const int a = clip(acc[2 * i]);
    const int b = clip(acc[2 * i + 1]);
    out[i] = static_cast<uint8_t>((a * b) >> PAIR_SHIFT);
  }
}

// Integer dot product of non-negative uint8 activations with int8 weights.
// Scalar and AVX2 forms are numerically identical (no saturation: pair sums of
// two u8*i8 products stay within int16).
int dot(const uint8_t *RESTRICT a, const int8_t *RESTRICT w, int n) {
#if defined(__AVX2__)
  __m256i acc = _mm256_setzero_si256();
  const __m256i ones = _mm256_set1_epi16(1);
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i va =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    const __m256i vw =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + i));
    acc = _mm256_add_epi32(
        acc, _mm256_madd_epi16(_mm256_maddubs_epi16(va, vw), ones));
  }
  __m128i lo = _mm256_castsi256_si128(acc);
  __m128i hi = _mm256_extracti128_si256(acc, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  int sum = _mm_cvtsi128_si32(s);
  for (; i < n; ++i)
    sum += static_cast<int>(a[i]) * static_cast<int>(w[i]);
  return sum;
#else
  int sum = 0;
  for (int i = 0; i < n; ++i)
    sum += static_cast<int>(a[i]) * static_cast<int>(w[i]);
  return sum;
#endif
}

} // namespace

int Network::bucket_of(const Core::Position &pos) {
  const int n = Core::popcount(pos.occupancy());
  int b = (n - 1) / 4;
  return b < 0 ? 0 : (b >= PSQT_BUCKETS ? PSQT_BUCKETS - 1 : b);
}

void Network::refresh_perspective(const Core::Position &pos, Core::Color persp,
                                  Accumulator &a) const {
  auto &acc = a.acc[persp];
  auto &pq = a.psqt[persp];
  acc = ftBias_;
  pq.fill(0);
  HalfKA::for_each_feature(pos, persp, [&](int f) {
    const int16_t *col = ft_col(f);
    for (int h = 0; h < HIDDEN; ++h)
      acc[h] = static_cast<int16_t>(acc[h] + col[h]);
    const int32_t *pc = ft_psqt(f);
    for (int k = 0; k < PSQT_BUCKETS; ++k)
      pq[k] += pc[k];
  });
}

void Network::refresh(const Core::Position &pos, Accumulator &a) const {
  refresh_perspective(pos, Core::WHITE, a);
  refresh_perspective(pos, Core::BLACK, a);
  a.computed = true;
}

void Network::apply_delta(Core::Color persp, const int *added, int nAdded,
                          const int *removed, int nRemoved,
                          Accumulator &a) const {
  auto &acc = a.acc[persp];
  auto &pq = a.psqt[persp];
  for (int idx = 0; idx < nAdded; ++idx) {
    const int16_t *col = ft_col(added[idx]);
    for (int h = 0; h < HIDDEN; ++h)
      acc[h] = static_cast<int16_t>(acc[h] + col[h]);
    const int32_t *pc = ft_psqt(added[idx]);
    for (int k = 0; k < PSQT_BUCKETS; ++k)
      pq[k] += pc[k];
  }
  for (int idx = 0; idx < nRemoved; ++idx) {
    const int16_t *col = ft_col(removed[idx]);
    for (int h = 0; h < HIDDEN; ++h)
      acc[h] = static_cast<int16_t>(acc[h] - col[h]);
    const int32_t *pc = ft_psqt(removed[idx]);
    for (int k = 0; k < PSQT_BUCKETS; ++k)
      pq[k] -= pc[k];
  }
}

int Network::forward(const Accumulator &a, Core::Color stm, int bucket,
                     Probe *probe) const {
  const auto &us = a.acc[stm];
  const auto &them = a.acc[~stm];

  alignas(32) std::array<uint8_t, L1_IN> l1in{};
  pairwise(us.data(), l1in.data());
  pairwise(them.data(), l1in.data() + PAIR);

  const Bucket &b = buckets_[bucket];

  alignas(32) std::array<uint8_t, L1> l1o{};
  for (int o = 0; o < L1; ++o) {
    const int s = b.l1b[o] + dot(l1in.data(), &b.l1w[o * L1_IN], L1_IN);
    l1o[o] = clip(s >> L1_SHIFT);
  }

  alignas(32) std::array<uint8_t, L2> l2o{};
  for (int o = 0; o < L2; ++o) {
    const int s = b.l2b[o] + dot(l1o.data(), &b.l2w[o * L1], L1);
    l2o[o] = clip(s >> L2_SHIFT);
  }

  const int raw = b.outb + dot(l2o.data(), b.outw.data(), L2);
  const int positional = raw >> OUT_SHIFT;
  const int psqtTerm =
      (a.psqt[stm][bucket] - a.psqt[~stm][bucket]) >> PSQT_SHIFT;
  const int eval = positional + psqtTerm;

  if (probe) {
    std::copy(us.begin(), us.end(), probe->accUs.begin());
    std::copy(them.begin(), them.end(), probe->accThem.begin());
    probe->l1in = l1in;
    probe->l1out = l1o;
    probe->l2out = l2o;
    probe->bucket = bucket;
    probe->psqt = psqtTerm;
    probe->positional = positional;
    probe->eval = eval;
  }
  return eval;
}

int Network::evaluate(const Core::Position &pos, const Accumulator &a) const {
  return forward(a, pos.side_to_move(), bucket_of(pos), nullptr);
}

int Network::evaluate_probe(const Core::Position &pos, const Accumulator &a,
                            Probe &probe) const {
  return forward(a, pos.side_to_move(), bucket_of(pos), &probe);
}

bool Network::load_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;

  char magic[9];
  uint32_t version = 0, feat = 0, hidden = 0, buckets = 0;
  in.read(magic, sizeof(magic));
  in.read(reinterpret_cast<char *>(&version), sizeof(version));
  in.read(reinterpret_cast<char *>(&feat), sizeof(feat));
  in.read(reinterpret_cast<char *>(&hidden), sizeof(hidden));
  in.read(reinterpret_cast<char *>(&buckets), sizeof(buckets));
  if (!in || std::memcmp(magic, MAGIC, sizeof(magic)) != 0 ||
      version != VERSION || feat != FEATURES || hidden != HIDDEN ||
      buckets != PSQT_BUCKETS)
    return false;

  ftWeights_.assign(static_cast<size_t>(FEATURES) * HIDDEN, 0);
  ftPsqt_.assign(static_cast<size_t>(FEATURES) * PSQT_BUCKETS, 0);

  in.read(reinterpret_cast<char *>(ftWeights_.data()),
          static_cast<std::streamsize>(ftWeights_.size() * sizeof(int16_t)));
  in.read(reinterpret_cast<char *>(ftBias_.data()),
          static_cast<std::streamsize>(ftBias_.size() * sizeof(int16_t)));
  in.read(reinterpret_cast<char *>(ftPsqt_.data()),
          static_cast<std::streamsize>(ftPsqt_.size() * sizeof(int32_t)));

  for (auto &b : buckets_) {
    in.read(reinterpret_cast<char *>(b.l1w.data()),
            static_cast<std::streamsize>(b.l1w.size()));
    in.read(reinterpret_cast<char *>(b.l1b.data()),
            static_cast<std::streamsize>(b.l1b.size() * sizeof(int32_t)));
    in.read(reinterpret_cast<char *>(b.l2w.data()),
            static_cast<std::streamsize>(b.l2w.size()));
    in.read(reinterpret_cast<char *>(b.l2b.data()),
            static_cast<std::streamsize>(b.l2b.size() * sizeof(int32_t)));
    in.read(reinterpret_cast<char *>(b.outw.data()),
            static_cast<std::streamsize>(b.outw.size()));
    in.read(reinterpret_cast<char *>(&b.outb), sizeof(b.outb));
  }

  if (!in)
    return false;
  loaded_ = true;
  return true;
}

void Network::randomize(uint64_t seed) {
  auto next = [&seed]() {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
  };
  auto rnd = [&](int lo, int hi) {
    return static_cast<int>(lo + next() % static_cast<uint64_t>(hi - lo + 1));
  };

  ftWeights_.assign(static_cast<size_t>(FEATURES) * HIDDEN, 0);
  ftPsqt_.assign(static_cast<size_t>(FEATURES) * PSQT_BUCKETS, 0);
  for (auto &w : ftWeights_)
    w = static_cast<int16_t>(rnd(-32, 32));
  for (auto &b : ftBias_)
    b = static_cast<int16_t>(rnd(-256, 256));
  for (auto &p : ftPsqt_)
    p = rnd(-2000, 2000);
  for (auto &bk : buckets_) {
    for (auto &w : bk.l1w)
      w = static_cast<int8_t>(rnd(-127, 127));
    for (auto &w : bk.l2w)
      w = static_cast<int8_t>(rnd(-127, 127));
    for (auto &w : bk.outw)
      w = static_cast<int8_t>(rnd(-127, 127));
    for (auto &b : bk.l1b)
      b = rnd(-4000, 4000);
    for (auto &b : bk.l2b)
      b = rnd(-4000, 4000);
    bk.outb = rnd(-20000, 20000);
  }
  loaded_ = true;
}

bool Network::self_test() {
  auto net = std::make_unique<Network>();
  net->randomize(0x1234567ULL);

  // 1) AVX2 forward == pure-scalar forward, over random accumulators.
  uint64_t s = 0x9E3779B97F4A7C15ULL;
  auto next = [&s]() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  };
  auto scalar_forward = [&](const Accumulator &a, Core::Color stm,
                            int bucket) -> int {
    const auto &us = a.acc[stm];
    const auto &them = a.acc[~stm];
    std::array<uint8_t, L1_IN> l1in{};
    for (int i = 0; i < PAIR; ++i) {
      const int va = clip(us[2 * i]), vb = clip(us[2 * i + 1]);
      l1in[i] = static_cast<uint8_t>((va * vb) >> PAIR_SHIFT);
      const int wa = clip(them[2 * i]), wb = clip(them[2 * i + 1]);
      l1in[PAIR + i] = static_cast<uint8_t>((wa * wb) >> PAIR_SHIFT);
    }
    const Bucket &b = net->buckets_[bucket];
    std::array<uint8_t, L1> l1o{};
    for (int o = 0; o < L1; ++o) {
      int acc = b.l1b[o];
      for (int i = 0; i < L1_IN; ++i)
        acc +=
            static_cast<int>(l1in[i]) * static_cast<int>(b.l1w[o * L1_IN + i]);
      l1o[o] = clip(acc >> L1_SHIFT);
    }
    std::array<uint8_t, L2> l2o{};
    for (int o = 0; o < L2; ++o) {
      int acc = b.l2b[o];
      for (int i = 0; i < L1; ++i)
        acc += static_cast<int>(l1o[i]) * static_cast<int>(b.l2w[o * L1 + i]);
      l2o[o] = clip(acc >> L2_SHIFT);
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
    Accumulator a{};
    for (int p = 0; p < 2; ++p) {
      for (int h = 0; h < HIDDEN; ++h)
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
                     "[nnue2] forward mismatch trial %d stm %d: simd %d scalar "
                     "%d\n",
                     trial, int(stm), simd, scal);
        return false;
      }
    }
  }

  // 2) Incremental accumulator primitive == full rebuild, over random features.
  auto build = [&](Core::Color p, const int *feats, int n, Accumulator &a) {
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
    Accumulator direct{}, delta{};
    build(p, s2, n, direct);
    build(p, s1, n, delta);
    net->apply_delta(p, s2, n, s1, n, delta); // add s2, remove s1
    if (delta.acc[p] != direct.acc[p] || delta.psqt[p] != direct.psqt[p]) {
      std::fprintf(stderr, "[nnue2] incremental != rebuild, trial %d\n", trial);
      return false;
    }
  }
  return true;
}

} // namespace NNUE
