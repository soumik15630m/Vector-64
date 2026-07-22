#ifndef NNUE_NETWORK_H
#define NNUE_NETWORK_H

#include "../cores/bitboard.h"
#include "../cores/position.h"
#include "halfka.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Intrinsics are used by the AVX2/VNNI kernels; NEON covers ARM.
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// STK-HalfKA NNUE, width-generic (engine side, from scratch).
//
//   features (22528/persp)  --FT-->  accumulator[H]  (int16, incremental)
//        also  --FT-PSQT-->  psqt[8]  (int32 side-output, one per bucket)
//   pairwise clipped-ReLU:  H -> H/2   per perspective
//   L1 (per bucket):  concat(us, them) = H -> 16   int8
//   L2 (per bucket):  16 -> 32
//   out (per bucket): 32 -> 1
//   eval_stm = out >> OUT_SHIFT + (psqt_us[b] - psqt_them[b]) >> PSQT_SHIFT
//   bucket b selected by piece count.
//
// NetworkT<1024> is the primary net; NetworkT<128> is the "small" net used by
// the dual-net lazy eval (cheap eval for clearly-decided positions). Both are
// trained/exported by tools/nnue/make_net.py (--hidden), same file format.
//
// Fixed-point: FT weights/bias int16, accumulator int16, activations clipped
// to [0,127], dense weights int8; fixed shifts (>>7 pairwise, >>6 dense,
// >>4 out) reproduce the trainer's normalized-float model exactly.

namespace NNUE {

namespace Arch {
constexpr int FEATURES = HalfKA::FEATURES; // 22528
constexpr int PSQT_BUCKETS = 8;
constexpr int L1 = 16;
constexpr int L2 = 32;

constexpr int ACT_MAX = 127;
constexpr int PAIR_SHIFT = 7;
constexpr int L1_SHIFT = 6;
constexpr int L2_SHIFT = 6;
constexpr int OUT_SHIFT = 4;
constexpr int PSQT_SHIFT = 4;
} // namespace Arch

namespace detail {

constexpr char MAGIC[9] = {'S', 'T', 'K', 'H', 'A', 'L', 'F', 'K', 'A'};
constexpr uint32_t VERSION = 1;

inline uint8_t clip(int v) {
  return static_cast<uint8_t>(v < 0 ? 0
                                    : (v > Arch::ACT_MAX ? Arch::ACT_MAX : v));
}

// Pairwise clipped-ReLU: H int16 accumulator -> H/2 uint8 activations.
// out[i] = clip(acc[i]) * clip(acc[i + H/2]) >> 7 -- the halves pairing is
// contiguous and vectorizes without lane shuffles; products of two [0,127]
// values fit int16. H/2 must be a multiple of 32 for the SIMD paths.
template <int H>
inline void pairwise(const int16_t *RESTRICT acc, uint8_t *RESTRICT out) {
  constexpr int PAIR = H / 2;
#if defined(__AVX2__)
  const __m256i zero = _mm256_setzero_si256();
  const __m256i maxv = _mm256_set1_epi16(Arch::ACT_MAX);
  const auto cl = [&](int off) {
    const __m256i v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(acc + off));
    return _mm256_min_epi16(_mm256_max_epi16(v, zero), maxv);
  };
  for (int i = 0; i < PAIR; i += 32) {
    const __m256i p0 = _mm256_srli_epi16(
        _mm256_mullo_epi16(cl(i), cl(i + PAIR)), Arch::PAIR_SHIFT);
    const __m256i p1 = _mm256_srli_epi16(
        _mm256_mullo_epi16(cl(i + 16), cl(i + PAIR + 16)), Arch::PAIR_SHIFT);
    const __m256i packed =
        _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + i), packed);
  }
#elif defined(__ARM_NEON)
  const int16x8_t zero = vdupq_n_s16(0);
  const int16x8_t maxv = vdupq_n_s16(Arch::ACT_MAX);
  const auto cl = [&](int off) {
    return vminq_s16(vmaxq_s16(vld1q_s16(acc + off), zero), maxv);
  };
  for (int i = 0; i < PAIR; i += 16) {
    const int16x8_t p0 =
        vshrq_n_s16(vmulq_s16(cl(i), cl(i + PAIR)), Arch::PAIR_SHIFT);
    const int16x8_t p1 =
        vshrq_n_s16(vmulq_s16(cl(i + 8), cl(i + PAIR + 8)), Arch::PAIR_SHIFT);
    vst1q_u8(out + i, vcombine_u8(vqmovun_s16(p0), vqmovun_s16(p1)));
  }
#else
  for (int i = 0; i < PAIR; ++i) {
    const int a = clip(acc[i]);
    const int b = clip(acc[i + PAIR]);
    out[i] = static_cast<uint8_t>((a * b) >> Arch::PAIR_SHIFT);
  }
#endif
}

// Integer dot product of non-negative uint8 activations with int8 weights.
// All variants are numerically identical: VNNI accumulates in int32; the AVX2
// maddubs pair-sums stay within int16 for our ranges; NEON widens through
// int16; scalar is the reference.
inline int dot(const uint8_t *RESTRICT a, const int8_t *RESTRICT w, int n) {
#if defined(__AVX2__)
  __m256i acc = _mm256_setzero_si256();
#if !defined(__AVXVNNI__)
  const __m256i ones = _mm256_set1_epi16(1);
#endif
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i va =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    const __m256i vw =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + i));
#if defined(__AVXVNNI__)
    acc = _mm256_dpbusd_avx_epi32(acc, va, vw);
#else
    acc = _mm256_add_epi32(
        acc, _mm256_madd_epi16(_mm256_maddubs_epi16(va, vw), ones));
#endif
  }
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc),
                            _mm256_extracti128_si256(acc, 1));
  // 128-bit block for a 16-wide remainder (e.g. the L2 layer's 16 inputs).
  if (i + 16 <= n) {
    const __m128i va =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    const __m128i vw =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i));
#if defined(__AVXVNNI__)
    s = _mm_add_epi32(s, _mm_dpbusd_avx_epi32(_mm_setzero_si128(), va, vw));
#else
    s = _mm_add_epi32(
        s, _mm_madd_epi16(_mm_maddubs_epi16(va, vw), _mm_set1_epi16(1)));
#endif
    i += 16;
  }
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  int sum = _mm_cvtsi128_si32(s);
  for (; i < n; ++i)
    sum += static_cast<int>(a[i]) * static_cast<int>(w[i]);
  return sum;
#elif defined(__ARM_NEON)
  int32x4_t acc = vdupq_n_s32(0);
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    const uint8x16_t va = vld1q_u8(a + i);
    const int8x16_t vw = vld1q_s8(w + i);
    const int16x8_t lo =
        vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(va))),
                  vmovl_s8(vget_low_s8(vw)));
    const int16x8_t hi =
        vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(va))),
                  vmovl_s8(vget_high_s8(vw)));
    acc = vpadalq_s16(acc, lo);
    acc = vpadalq_s16(acc, hi);
  }
  int sum = vaddvq_s32(acc);
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

// Add / subtract a feature column into an int16 accumulator. The SIMD forms
// wrap mod 2^16 exactly like the scalar path, so results are identical.
template <int H>
inline void acc_add(int16_t *RESTRICT acc, const int16_t *RESTRICT col) {
#if defined(__AVX2__)
  for (int h = 0; h < H; h += 16)
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(acc + h),
        _mm256_add_epi16(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(acc + h)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(col + h))));
#elif defined(__ARM_NEON)
  for (int h = 0; h < H; h += 8)
    vst1q_s16(acc + h, vaddq_s16(vld1q_s16(acc + h), vld1q_s16(col + h)));
#else
  for (int h = 0; h < H; ++h)
    acc[h] = static_cast<int16_t>(acc[h] + col[h]);
#endif
}

// Fused child = parent + sum(adds) - sum(subs) in one pass over the
// accumulator, instead of copy + per-column add/sub passes. int16 wrap
// arithmetic is order-independent, so this is bit-identical to the
// sequential form.
template <int H>
inline void acc_fused(int16_t *RESTRICT dst, const int16_t *RESTRICT src,
                      const int16_t *const *adds, int na,
                      const int16_t *const *subs, int nr) {
#if defined(__AVX2__)
  for (int h = 0; h < H; h += 16) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + h));
    for (int j = 0; j < na; ++j)
      v = _mm256_add_epi16(
          v,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(adds[j] + h)));
    for (int j = 0; j < nr; ++j)
      v = _mm256_sub_epi16(
          v,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(subs[j] + h)));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + h), v);
  }
#elif defined(__ARM_NEON)
  for (int h = 0; h < H; h += 8) {
    int16x8_t v = vld1q_s16(src + h);
    for (int j = 0; j < na; ++j)
      v = vaddq_s16(v, vld1q_s16(adds[j] + h));
    for (int j = 0; j < nr; ++j)
      v = vsubq_s16(v, vld1q_s16(subs[j] + h));
    vst1q_s16(dst + h, v);
  }
#else
  for (int h = 0; h < H; ++h) {
    int v = src[h];
    for (int j = 0; j < na; ++j)
      v += adds[j][h];
    for (int j = 0; j < nr; ++j)
      v -= subs[j][h];
    dst[h] = static_cast<int16_t>(v);
  }
#endif
}

#if defined(__AVX2__)
// 128-bit four-row variant for the 16-wide L2 layer: one activation load
// shared by four rows, one combined reduction (exact int32 sums).
inline __m128i dot4x16(const uint8_t *RESTRICT a, const int8_t *RESTRICT w,
                       size_t stride) {
  const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a));
  const auto row = [&](size_t j) {
    const __m128i vw =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(w + j * stride));
#if defined(__AVXVNNI__)
    return _mm_dpbusd_avx_epi32(_mm_setzero_si128(), va, vw);
#else
    return _mm_madd_epi16(_mm_maddubs_epi16(va, vw), _mm_set1_epi16(1));
#endif
  };
  const __m128i r01 = _mm_hadd_epi32(row(0), row(1));
  const __m128i r23 = _mm_hadd_epi32(row(2), row(3));
  return _mm_hadd_epi32(r01, r23);
}
#endif

// Both perspectives' fused updates in ONE loop body: two independent
// dependency chains per iteration double the ILP the out-of-order core can
// extract from the (L2-bound) column streams. Bit-identical to two
// sequential acc_fused passes. Measured -14% on the update core.
template <int H>
inline void acc_fused2(int16_t *RESTRICT dstW, const int16_t *RESTRICT srcW,
                       const int16_t *const *addW, int naW,
                       const int16_t *const *subW, int nrW,
                       int16_t *RESTRICT dstB, const int16_t *RESTRICT srcB,
                       const int16_t *const *addB, int naB,
                       const int16_t *const *subB, int nrB) {
#if defined(__AVX2__)
  // Fast path for the overwhelmingly common quiet move: exactly one added and
  // one removed feature per perspective. Straight-line, no data-dependent inner
  // loops -- bit-identical since int16 add/sub is modular and associative
  // (src - sub + add == src + add - sub mod 2^16).
  if (naW == 1 && nrW == 1 && naB == 1 && nrB == 1) {
    const int16_t *RESTRICT aw = addW[0];
    const int16_t *RESTRICT sw = subW[0];
    const int16_t *RESTRICT ab = addB[0];
    const int16_t *RESTRICT sb = subB[0];
    for (int h = 0; h < H; h += 16) {
      const __m256i v = _mm256_add_epi16(
          _mm256_sub_epi16(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(srcW + h)),
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sw + h))),
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(aw + h)));
      const __m256i u = _mm256_add_epi16(
          _mm256_sub_epi16(
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(srcB + h)),
              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sb + h))),
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ab + h)));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dstW + h), v);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(dstB + h), u);
    }
    return;
  }
  for (int h = 0; h < H; h += 16) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(srcW + h));
    __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(srcB + h));
    for (int j = 0; j < naW; ++j)
      v = _mm256_add_epi16(
          v,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(addW[j] + h)));
    for (int j = 0; j < naB; ++j)
      u = _mm256_add_epi16(
          u,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(addB[j] + h)));
    for (int j = 0; j < nrW; ++j)
      v = _mm256_sub_epi16(
          v,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(subW[j] + h)));
    for (int j = 0; j < nrB; ++j)
      u = _mm256_sub_epi16(
          u,
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(subB[j] + h)));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dstW + h), v);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dstB + h), u);
  }
#elif defined(__ARM_NEON)
  for (int h = 0; h < H; h += 8) {
    int16x8_t v = vld1q_s16(srcW + h);
    int16x8_t u = vld1q_s16(srcB + h);
    for (int j = 0; j < naW; ++j)
      v = vaddq_s16(v, vld1q_s16(addW[j] + h));
    for (int j = 0; j < naB; ++j)
      u = vaddq_s16(u, vld1q_s16(addB[j] + h));
    for (int j = 0; j < nrW; ++j)
      v = vsubq_s16(v, vld1q_s16(subW[j] + h));
    for (int j = 0; j < nrB; ++j)
      u = vsubq_s16(u, vld1q_s16(subB[j] + h));
    vst1q_s16(dstW + h, v);
    vst1q_s16(dstB + h, u);
  }
#else
  for (int h = 0; h < H; ++h) {
    int v = srcW[h];
    int u = srcB[h];
    for (int j = 0; j < naW; ++j)
      v += addW[j][h];
    for (int j = 0; j < naB; ++j)
      u += addB[j][h];
    for (int j = 0; j < nrW; ++j)
      v -= subW[j][h];
    for (int j = 0; j < nrB; ++j)
      u -= subB[j][h];
    dstW[h] = static_cast<int16_t>(v);
    dstB[h] = static_cast<int16_t>(u);
  }
#endif
}

#if defined(__AVX2__)
// Eight dot-product rows sharing every activation load (halves the
// activation re-reads and reduction trees of the 4-row form; exact int32
// sums, order-independent). Measured -32% on the L1 layer.
inline void dot8(const uint8_t *RESTRICT a, const int8_t *RESTRICT w,
                 size_t stride, int n, int32_t out[8]) {
  __m256i c0 = _mm256_setzero_si256(), c1 = c0, c2 = c0, c3 = c0, c4 = c0,
          c5 = c0, c6 = c0, c7 = c0;
#if !defined(__AVXVNNI__)
  const __m256i ones = _mm256_set1_epi16(1);
  const auto dp = [&](__m256i acc, __m256i va, __m256i vw) {
    return _mm256_add_epi32(
        acc, _mm256_madd_epi16(_mm256_maddubs_epi16(va, vw), ones));
  };
#else
  const auto dp = [](__m256i acc, __m256i va, __m256i vw) {
    return _mm256_dpbusd_avx_epi32(acc, va, vw);
  };
#endif
  for (int i = 0; i < n; i += 32) {
    const __m256i va =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    c0 = dp(c0, va,
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + i)));
    c1 = dp(
        c1, va,
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w + stride + i)));
    c2 = dp(c2, va,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(w + 2 * stride + i)));
    c3 = dp(c3, va,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(w + 3 * stride + i)));
    c4 = dp(c4, va,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(w + 4 * stride + i)));
    c5 = dp(c5, va,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(w + 5 * stride + i)));
    c6 = dp(c6, va,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(w + 6 * stride + i)));
    c7 = dp(c7, va,
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(w + 7 * stride + i)));
  }
  const __m256i r01 = _mm256_hadd_epi32(c0, c1);
  const __m256i r23 = _mm256_hadd_epi32(c2, c3);
  const __m256i r45 = _mm256_hadd_epi32(c4, c5);
  const __m256i r67 = _mm256_hadd_epi32(c6, c7);
  const __m256i ra = _mm256_hadd_epi32(r01, r23);
  const __m256i rb = _mm256_hadd_epi32(r45, r67);
  _mm_store_si128(reinterpret_cast<__m128i *>(out),
                  _mm_add_epi32(_mm256_castsi256_si128(ra),
                                _mm256_extracti128_si256(ra, 1)));
  _mm_store_si128(reinterpret_cast<__m128i *>(out + 4),
                  _mm_add_epi32(_mm256_castsi256_si128(rb),
                                _mm256_extracti128_si256(rb, 1)));
}
#endif

template <int H>
inline void acc_sub(int16_t *RESTRICT acc, const int16_t *RESTRICT col) {
#if defined(__AVX2__)
  for (int h = 0; h < H; h += 16)
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(acc + h),
        _mm256_sub_epi16(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(acc + h)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(col + h))));
#elif defined(__ARM_NEON)
  for (int h = 0; h < H; h += 8)
    vst1q_s16(acc + h, vsubq_s16(vld1q_s16(acc + h), vld1q_s16(col + h)));
#else
  for (int h = 0; h < H; ++h)
    acc[h] = static_cast<int16_t>(acc[h] - col[h]);
#endif
}

} // namespace detail

template <int H> struct alignas(64) AccumulatorT {
  static_assert(H % 64 == 0, "SIMD paths assume H is a multiple of 64");
  alignas(64) std::array<std::array<int16_t, H>, 2> acc{};
  alignas(64) std::array<std::array<int32_t, Arch::PSQT_BUCKETS>, 2> psqt{};
  std::array<bool, 2> computed{};
};

// Per-thread cache of one accumulator half per (perspective, king square):
// a king move becomes a diff against the accumulator last built for that
// king square instead of a full rebuild ("finny tables").
template <int H> struct RefreshTableT {
  struct Entry {
    alignas(64) std::array<int16_t, H> acc{};
    std::array<int32_t, Arch::PSQT_BUCKETS> psqt{};
    Core::Bitboard pieces[Core::COLOR_NB][Core::PIECE_TYPE_NB] = {};
    bool valid = false;
  };
  std::array<std::array<Entry, Core::SQUARE_NB>, Core::COLOR_NB> entries{};
};

// Everything needed to replay one move's feature changes (width-independent;
// kept for lazy-update experiments).
struct DirtyMove {
  Core::Color mover = Core::WHITE;
  Core::Square from = Core::SQ_NONE, to = Core::SQ_NONE;
  Core::PieceType movedNow = Core::NO_PIECE_TYPE;    // post-promotion type
  Core::PieceType removedType = Core::NO_PIECE_TYPE; // pre-promotion type
  Core::PieceType capType = Core::NO_PIECE_TYPE;
  Core::Square capSq = Core::SQ_NONE;
  Core::Square rookFrom = Core::SQ_NONE, rookTo = Core::SQ_NONE; // castling
  bool kingMoved = false; // mover's king moved (refresh barrier)
  bool isNull = false;
};

// Per-eval activation snapshot for the glass-box probe (primary net only).
struct Probe {
  std::array<int16_t, 1024> accUs{};
  std::array<int16_t, 1024> accThem{};
  std::array<uint8_t, 1024> l1in{};
  std::array<uint8_t, Arch::L1> l1out{};
  std::array<uint8_t, Arch::L2> l2out{};
  int bucket = 0;
  int psqt = 0;
  int positional = 0;
  int eval = 0;
};

template <int H> class NetworkT {
public:
  static constexpr int HIDDEN = H;
  static constexpr int PAIR = H / 2;

  struct Bucket {
    std::array<int8_t, Arch::L1 * H> l1w{};
    std::array<int32_t, Arch::L1> l1b{};
    std::array<int8_t, Arch::L2 * Arch::L1> l2w{};
    std::array<int32_t, Arch::L2> l2b{};
    std::array<int8_t, Arch::L2> outw{};
    int32_t outb = 0;
  };

  bool load_file(const std::string &path) {
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
    if (!in || std::memcmp(magic, detail::MAGIC, sizeof(magic)) != 0 ||
        version != detail::VERSION || feat != Arch::FEATURES ||
        hidden != static_cast<uint32_t>(H) || buckets != Arch::PSQT_BUCKETS)
      return false;

    ftWeights_.assign(static_cast<size_t>(Arch::FEATURES) * H, 0);
    ftPsqt_.assign(static_cast<size_t>(Arch::FEATURES) * Arch::PSQT_BUCKETS, 0);

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

  bool is_loaded() const { return loaded_; }

  static int bucket_of(const Core::Position &pos) {
    const int n = Core::popcount(pos.occupancy());
    const int b = (n - 1) / 4;
    return b < 0 ? 0 : (b >= Arch::PSQT_BUCKETS ? Arch::PSQT_BUCKETS - 1 : b);
  }

  void refresh_perspective(const Core::Position &pos, Core::Color persp,
                           AccumulatorT<H> &a) const {
    auto &acc = a.acc[persp];
    auto &pq = a.psqt[persp];
    acc = ftBias_;
    pq.fill(0);
    HalfKA::for_each_feature(pos, persp, [&](int f) {
      detail::acc_add<H>(acc.data(), ft_col(f));
      const int32_t *pc = ft_psqt(f);
      for (int k = 0; k < Arch::PSQT_BUCKETS; ++k)
        pq[k] += pc[k];
    });
  }

  void refresh(const Core::Position &pos, AccumulatorT<H> &a) const {
    refresh_perspective(pos, Core::WHITE, a);
    refresh_perspective(pos, Core::BLACK, a);
    a.computed = {true, true};
  }

  // Cached refresh: diff against the accumulator last built for this king
  // square (same orientation by construction).
  void refresh_perspective(const Core::Position &pos, Core::Color persp,
                           AccumulatorT<H> &a, RefreshTableT<H> &table) const {
    using namespace Core;
    const Square ksq = lsb(pos.pieces(KING, persp));
    typename RefreshTableT<H>::Entry &e = table.entries[persp][ksq];
    const HalfKA::Orient o = HalfKA::make_orient(persp, ksq);

    if (!e.valid) {
      e.acc = ftBias_;
      e.psqt.fill(0);
      HalfKA::for_each_feature(pos, persp, [&](int f) {
        detail::acc_add<H>(e.acc.data(), ft_col(f));
        const int32_t *pc = ft_psqt(f);
        for (int k = 0; k < Arch::PSQT_BUCKETS; ++k)
          e.psqt[k] += pc[k];
      });
    } else {
      // Diff the cached placement against the current one; both were built
      // with the same king square, so the orientation is identical.
      for (int c = WHITE; c <= BLACK; ++c) {
        for (int pt = PAWN; pt <= KING; ++pt) {
          const Bitboard cur = pos.pieces(PieceType(pt), Color(c));
          const Bitboard old = e.pieces[c][pt];
          Bitboard removed = old & ~cur;
          Bitboard added = cur & ~old;
          while (removed) {
            const int f = HalfKA::feature_index(o, Color(c), PieceType(pt),
                                                pop_lsb(removed));
            if (f < 0)
              continue;
            detail::acc_sub<H>(e.acc.data(), ft_col(f));
            const int32_t *pc = ft_psqt(f);
            for (int k = 0; k < Arch::PSQT_BUCKETS; ++k)
              e.psqt[k] -= pc[k];
          }
          while (added) {
            const int f = HalfKA::feature_index(o, Color(c), PieceType(pt),
                                                pop_lsb(added));
            if (f < 0)
              continue;
            detail::acc_add<H>(e.acc.data(), ft_col(f));
            const int32_t *pc = ft_psqt(f);
            for (int k = 0; k < Arch::PSQT_BUCKETS; ++k)
              e.psqt[k] += pc[k];
          }
        }
      }
    }

    for (int c = WHITE; c <= BLACK; ++c)
      for (int pt = PAWN; pt <= KING; ++pt)
        e.pieces[c][pt] = pos.pieces(PieceType(pt), Color(c));
    e.valid = true;

    a.acc[persp] = e.acc;
    a.psqt[persp] = e.psqt;
  }

  void apply_delta(Core::Color persp, const int *added, int nAdded,
                   const int *removed, int nRemoved, AccumulatorT<H> &a) const {
    auto &acc = a.acc[persp];
    auto &pq = a.psqt[persp];
    for (int idx = 0; idx < nAdded; ++idx) {
      detail::acc_add<H>(acc.data(), ft_col(added[idx]));
      const int32_t *pc = ft_psqt(added[idx]);
      for (int k = 0; k < Arch::PSQT_BUCKETS; ++k)
        pq[k] += pc[k];
    }
    for (int idx = 0; idx < nRemoved; ++idx) {
      detail::acc_sub<H>(acc.data(), ft_col(removed[idx]));
      const int32_t *pc = ft_psqt(removed[idx]);
      for (int k = 0; k < Arch::PSQT_BUCKETS; ++k)
        pq[k] -= pc[k];
    }
  }

  static DirtyMove make_dirty(const Core::Position &after, Core::Move m,
                              const Core::UndoInfo &ui) {
    using namespace Core;
    DirtyMove dm;
    dm.mover = ~after.side_to_move(); // side that just moved
    dm.from = m.from_sq();
    dm.to = m.to_sq();
    dm.movedNow = after.piece_on(dm.to); // moved or promoted piece
    dm.removedType = m.is_promotion() ? PAWN : dm.movedNow;
    if (m.is_capture()) {
      dm.capType = ui.capturedPiece;
      dm.capSq = m.is_en_passant() ? make_square(GenFile(file_of(dm.to)),
                                                 GenRank(rank_of(dm.from)))
                                   : dm.to;
    }
    if (m.is_castling()) {
      const GenRank r = GenRank(rank_of(dm.from));
      dm.rookFrom =
          (dm.to > dm.from) ? make_square(FILE_H, r) : make_square(FILE_A, r);
      dm.rookTo =
          (dm.to > dm.from) ? make_square(FILE_F, r) : make_square(FILE_D, r);
    }
    dm.kingMoved = (dm.movedNow == KING) || m.is_castling();
    return dm;
  }

  // Apply one DirtyMove's changes to a single perspective in place. Only
  // valid when the perspective's king did not move across the step.
  void apply_dirty(Core::Color persp, Core::Square ksq, const DirtyMove &dm,
                   AccumulatorT<H> &a) const {
    using namespace Core;
    if (dm.isNull)
      return;
    const HalfKA::Orient o = HalfKA::make_orient(persp, ksq);
    int added[4], removed[4];
    int na = 0, nr = 0;
    const auto push = [&](int *arr, int &n, Color c, PieceType t, Square s) {
      const int f = HalfKA::feature_index(o, c, t, s);
      if (f >= 0)
        arr[n++] = f;
    };
    push(removed, nr, dm.mover, dm.removedType, dm.from);
    push(added, na, dm.mover, dm.movedNow, dm.to);
    if (dm.capType != NO_PIECE_TYPE)
      push(removed, nr, ~dm.mover, dm.capType, dm.capSq);
    if (dm.rookFrom != SQ_NONE) {
      push(removed, nr, dm.mover, ROOK, dm.rookFrom);
      push(added, na, dm.mover, ROOK, dm.rookTo);
    }
    apply_delta(persp, added, na, removed, nr, a);
  }

  // Derive `child` from `parent` for the just-played move `m`. A perspective
  // is refreshed only when its own king moved -- via the cache when `table`
  // is given; otherwise the few changed features are applied incrementally.
  // Fused single step: child = parent + adds - subs in one pass over the
  // accumulator half. Only valid when persp's king did not move in dm (the
  // orientation for ksq must hold across the step).
  void apply_dirty_fused(Core::Color persp, Core::Square ksq,
                         const DirtyMove &dm, const AccumulatorT<H> &parent,
                         AccumulatorT<H> &child) const {
    using namespace Core;
    if (dm.isNull) {
      child.acc[persp] = parent.acc[persp];
      child.psqt[persp] = parent.psqt[persp];
      return;
    }
    const HalfKA::Orient o = HalfKA::make_orient(persp, ksq);
    const int16_t *adds[2];
    const int16_t *subs[2];
    int32_t psqtDelta[Arch::PSQT_BUCKETS] = {};
    int na = 0, nr = 0;
    const auto push = [&](const int16_t **arr, int &n, int sign, Color c,
                          PieceType t, Square s) {
      const int f = HalfKA::feature_index(o, c, t, s);
      if (f < 0)
        return;
      arr[n++] = ft_col(f);
      const int32_t *pc = ft_psqt(f);
      for (int k = 0; k < Arch::PSQT_BUCKETS; ++k)
        psqtDelta[k] += sign * pc[k];
    };
    push(subs, nr, -1, dm.mover, dm.removedType, dm.from);
    push(adds, na, +1, dm.mover, dm.movedNow, dm.to);
    if (dm.capType != NO_PIECE_TYPE)
      push(subs, nr, -1, ~dm.mover, dm.capType, dm.capSq);
    if (dm.rookFrom != SQ_NONE) {
      push(subs, nr, -1, dm.mover, ROOK, dm.rookFrom);
      push(adds, na, +1, dm.mover, ROOK, dm.rookTo);
    }
    detail::acc_fused<H>(child.acc[persp].data(), parent.acc[persp].data(),
                         adds, na, subs, nr);
    for (int k = 0; k < Arch::PSQT_BUCKETS; ++k)
      child.psqt[persp][k] = parent.psqt[persp][k] + psqtDelta[k];
  }

  void update(const AccumulatorT<H> &parent, AccumulatorT<H> &child,
              const Core::Position &after, Core::Move m,
              const Core::UndoInfo &ui,
              RefreshTableT<H> *table = nullptr) const {
    using namespace Core;
    const DirtyMove dm = make_dirty(after, m, ui);

    if (!dm.kingMoved) {
      // Common case (no refresh needed): both perspectives in one
      // interleaved pass over the accumulators.
      const int16_t *addC[2][2];
      const int16_t *subC[2][2];
      int na[2] = {0, 0}, nr[2] = {0, 0};
      for (int pc = WHITE; pc <= BLACK; ++pc) {
        const Color persp = Color(pc);
        const HalfKA::Orient o =
            HalfKA::make_orient(persp, lsb(after.pieces(KING, persp)));
        int32_t *pq = child.psqt[persp].data();
        const int32_t *ppq = parent.psqt[persp].data();
        for (int k = 0; k < Arch::PSQT_BUCKETS; ++k)
          pq[k] = ppq[k];
        const auto push = [&](const int16_t **arr, int &cnt, int sign, Color c,
                              PieceType t, Square sq) {
          const int f = HalfKA::feature_index(o, c, t, sq);
          if (f < 0)
            return;
          arr[cnt++] = ft_col(f);
          const int32_t *pcq = ft_psqt(f);
          for (int k = 0; k < Arch::PSQT_BUCKETS; ++k)
            pq[k] += sign * pcq[k];
        };
        push(subC[pc], nr[pc], -1, dm.mover, dm.removedType, dm.from);
        push(addC[pc], na[pc], +1, dm.mover, dm.movedNow, dm.to);
        if (dm.capType != NO_PIECE_TYPE)
          push(subC[pc], nr[pc], -1, ~dm.mover, dm.capType, dm.capSq);
        if (dm.rookFrom != SQ_NONE) {
          push(subC[pc], nr[pc], -1, dm.mover, ROOK, dm.rookFrom);
          push(addC[pc], na[pc], +1, dm.mover, ROOK, dm.rookTo);
        }
        child.computed[persp] = true;
      }
      detail::acc_fused2<H>(child.acc[WHITE].data(), parent.acc[WHITE].data(),
                            addC[WHITE], na[WHITE], subC[WHITE], nr[WHITE],
                            child.acc[BLACK].data(), parent.acc[BLACK].data(),
                            addC[BLACK], na[BLACK], subC[BLACK], nr[BLACK]);
      return;
    }

    for (int pc = WHITE; pc <= BLACK; ++pc) {
      const Color persp = Color(pc);
      if (persp == dm.mover && dm.kingMoved) {
        if (table)
          refresh_perspective(after, persp, child, *table);
        else
          refresh_perspective(after, persp, child);
      } else {
        apply_dirty_fused(persp, lsb(after.pieces(KING, persp)), dm, parent,
                          child);
      }
      child.computed[persp] = true;
    }
  }

  int evaluate(const Core::Position &pos, const AccumulatorT<H> &a) const {
    return forward(a, pos.side_to_move(), bucket_of(pos), nullptr);
  }

  int evaluate_probe(const Core::Position &pos, const AccumulatorT<H> &a,
                     Probe &probe) const {
    static_assert(H == 1024, "the glass-box probe reads the primary net");
    return forward(a, pos.side_to_move(), bucket_of(pos), &probe);
  }

  // Deterministic random weights (self-test / tooling; no file needed).
  void randomize(uint64_t seed) {
    auto next = [&seed]() {
      seed ^= seed << 13;
      seed ^= seed >> 7;
      seed ^= seed << 17;
      return seed;
    };
    auto rnd = [&](int lo, int hi) {
      return static_cast<int>(lo + next() % static_cast<uint64_t>(hi - lo + 1));
    };
    ftWeights_.assign(static_cast<size_t>(Arch::FEATURES) * H, 0);
    ftPsqt_.assign(static_cast<size_t>(Arch::FEATURES) * Arch::PSQT_BUCKETS, 0);
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

  // Verifies the SIMD forward against a pure-scalar reference and the
  // incremental delta against a rebuild. Defined in network.cpp.
  static bool self_test();

private:
  const int16_t *ft_col(int feature) const {
    return ftWeights_.data() + static_cast<size_t>(feature) * H;
  }
  const int32_t *ft_psqt(int feature) const {
    return ftPsqt_.data() + static_cast<size_t>(feature) * Arch::PSQT_BUCKETS;
  }

  int forward(const AccumulatorT<H> &a, Core::Color stm, int bucket,
              Probe *probe) const {
    const auto &us = a.acc[stm];
    const auto &them = a.acc[~stm];

    // Deliberately uninitialized: pairwise/the layer loops write every byte,
    // and value-init here would memset ~1KB per eval.
    alignas(32) std::array<uint8_t, H> l1in;
    detail::pairwise<H>(us.data(), l1in.data());
    detail::pairwise<H>(them.data(), l1in.data() + PAIR);

    const Bucket &b = buckets_[bucket];

    alignas(32) std::array<uint8_t, Arch::L1> l1o;
#if defined(__AVX2__)
    // Eight output rows per pass share every activation load (H is a
    // multiple of 32, so dot8 needs no tail).
    for (int o = 0; o < Arch::L1; o += 8) {
      alignas(16) int32_t sums[8];
      detail::dot8(l1in.data(), &b.l1w[o * H], H, H, sums);
      for (int j = 0; j < 8; ++j)
        l1o[o + j] = detail::clip((b.l1b[o + j] + sums[j]) >> Arch::L1_SHIFT);
    }
#else
    for (int o = 0; o < Arch::L1; ++o) {
      const int s = b.l1b[o] + detail::dot(l1in.data(), &b.l1w[o * H], H);
      l1o[o] = detail::clip(s >> Arch::L1_SHIFT);
    }
#endif

    alignas(32) std::array<uint8_t, Arch::L2> l2o;
#if defined(__AVX2__)
    for (int o = 0; o < Arch::L2; o += 4) {
      alignas(16) int32_t sums[4];
      _mm_store_si128(
          reinterpret_cast<__m128i *>(sums),
          detail::dot4x16(l1o.data(), &b.l2w[o * Arch::L1], Arch::L1));
      for (int j = 0; j < 4; ++j)
        l2o[o + j] = detail::clip((b.l2b[o + j] + sums[j]) >> Arch::L2_SHIFT);
    }
#else
    for (int o = 0; o < Arch::L2; ++o) {
      const int s =
          b.l2b[o] + detail::dot(l1o.data(), &b.l2w[o * Arch::L1], Arch::L1);
      l2o[o] = detail::clip(s >> Arch::L2_SHIFT);
    }
#endif

    const int raw = b.outb + detail::dot(l2o.data(), b.outw.data(), Arch::L2);
    const int positional = raw >> Arch::OUT_SHIFT;
    const int psqtTerm =
        (a.psqt[stm][bucket] - a.psqt[~stm][bucket]) >> Arch::PSQT_SHIFT;
    const int eval = positional + psqtTerm;

    if (probe) {
      std::copy(us.begin(), us.end(), probe->accUs.begin());
      std::copy(them.begin(), them.end(), probe->accThem.begin());
      std::copy(l1in.begin(), l1in.end(), probe->l1in.begin());
      probe->l1out = l1o;
      probe->l2out = l2o;
      probe->bucket = bucket;
      probe->psqt = psqtTerm;
      probe->positional = positional;
      probe->eval = eval;
    }
    return eval;
  }

  bool loaded_ = false;
  std::vector<int16_t> ftWeights_; // [FEATURES * H]
  std::array<int16_t, H> ftBias_{};
  std::vector<int32_t> ftPsqt_; // [FEATURES * PSQT_BUCKETS]
  std::array<Bucket, Arch::PSQT_BUCKETS> buckets_{};
};

// Primary ("big") net and the dual-net companion ("small") net.
using Network = NetworkT<1024>;
using SmallNetwork = NetworkT<128>;
using Accumulator = AccumulatorT<1024>;
using SmallAccumulator = AccumulatorT<128>;
using RefreshTable = RefreshTableT<1024>;
using SmallRefreshTable = RefreshTableT<128>;

} // namespace NNUE

#endif
