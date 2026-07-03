#include "nnue.h"

#include "../cores/bitboard.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <cstdio>

// Include SIMD intrinsics if supported
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace NNUE {
    namespace {
        constexpr const char* NETWORK_MAGIC = "VECTOR64_NNUE";
        constexpr uint32_t NETWORK_VERSION = 1;
        constexpr int FEATURE_TO_DENSE_SCALE = 64;
        constexpr int DENSE_TO_DENSE_SCALE = 64;
        constexpr int OUTPUT_SCALE = 16;

        constexpr std::streamoff align_to(std::streamoff offset, std::streamoff alignment) {
            return ((offset + alignment - 1) / alignment) * alignment;
        }

        int piece_bucket(Core::PieceType pieceType, Core::Color pieceColor) {
            const int colorOffset = pieceColor == Core::WHITE ? 0 : 5;
            switch (pieceType) {
                case Core::PAWN: return colorOffset + 0;
                case Core::KNIGHT: return colorOffset + 1;
                case Core::BISHOP: return colorOffset + 2;
                case Core::ROOK: return colorOffset + 3;
                case Core::QUEEN: return colorOffset + 4;
                default: return -1;
            }
        }

        [[maybe_unused]] int16_t clamp_sym_i16(int value) {
            return static_cast<int16_t>(std::clamp(value, -32767, 32767));
        }

        int8_t clamp_sym_i8(int value) {
            return static_cast<int8_t>(std::clamp(value, -127, 127));
        }

        int8_t relu_i8(int8_t value) {
            return static_cast<int8_t>(std::max(0, static_cast<int>(value)));
        }

        bool extract_json_float(const std::string& text, const std::string& key, float& out) {
            const std::regex pattern(
                "\"" + key + "\"\\s*:\\s*([-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?)"
            );
            std::smatch match;
            if (!std::regex_search(text, match, pattern) || match.size() < 2) return false;

            try {
                const float parsed = std::stof(match[1].str());
                if (!std::isfinite(parsed)) return false;
                out = parsed;
                return true;
            } catch (...) {
                return false;
            }
        }

    }

    void Accumulator512::clear() {
        white.fill(0);
        black.fill(0);
    }

    uint32_t halfkp_feature_index(
        Core::Square kingSquare,
        Core::PieceType pieceType,
        Core::Color pieceColor,
        Core::Square pieceSquare,
        Core::Color perspective
    ) {
        const int bucket = piece_bucket(pieceType, pieceColor);
        if (bucket < 0) return 0;

        const uint32_t king = static_cast<uint32_t>(kingSquare);
        const uint32_t piece = static_cast<uint32_t>(bucket);
        const uint32_t sq = static_cast<uint32_t>(pieceSquare);
        const uint32_t side = perspective == Core::BLACK ? 1u : 0u;

        return (((king * HALF_KP_PIECE_BUCKETS + piece) * 64u + sq) * 2u) + side;
    }

    bool Runtime::read_exact(std::ifstream& in, void* dst, size_t bytes) {
        if (bytes == 0) return true;
        in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
        return static_cast<bool>(in);
    }

    bool Runtime::load_file(const std::string& path) {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) return false;

        const std::streamoff fileSize = in.tellg();
        if (fileSize < 0) return false;
        in.seekg(0, std::ios::beg);
        if (!in) return false;

        BinaryHeader header{};
        if (!read_exact(in, &header, sizeof(header))) return false;

        if (std::memcmp(header.magic, NETWORK_MAGIC, sizeof(header.magic)) != 0) return false;
        if (header.version != NETWORK_VERSION) return false;
        if (header.hiddenSize != HIDDEN_SIZE) return false;

        const size_t ftCount = static_cast<size_t>(HALF_KP_TOTAL_FEATURES) * static_cast<size_t>(HIDDEN_SIZE);
        const std::streamoff ftBytes =
            static_cast<std::streamoff>(ftCount * sizeof(int16_t));
        const std::streamoff l1Bytes =
            static_cast<std::streamoff>(DENSE_L1_SIZE * HIDDEN_SIZE * sizeof(int8_t));
        const std::streamoff l2Bytes =
            static_cast<std::streamoff>(DENSE_L2_SIZE * DENSE_L1_SIZE * sizeof(int8_t));
        const std::streamoff outBytes =
            static_cast<std::streamoff>(DENSE_L2_SIZE * sizeof(int8_t));
        const std::streamoff biasBytes =
            static_cast<std::streamoff>(
                HIDDEN_SIZE * sizeof(int16_t) +
                DENSE_L1_SIZE * sizeof(int32_t) +
                DENSE_L2_SIZE * sizeof(int32_t) +
                sizeof(int32_t)
            );

        struct SectionSpec {
            std::streamoff offset = 0;
            std::streamoff bytes = 0;
        };

        SectionSpec sections[5];
        std::streamoff cursor = static_cast<std::streamoff>(sizeof(BinaryHeader));
        const auto push_section = [&](int index, std::streamoff bytes) {
            cursor = align_to(cursor, SECTION_ALIGNMENT);
            sections[index] = SectionSpec{cursor, bytes};
            cursor += bytes;
        };

        push_section(0, ftBytes);
        push_section(1, l1Bytes);
        push_section(2, l2Bytes);
        push_section(3, outBytes);
        push_section(4, biasBytes);

        const std::streamoff expectedSize = cursor;
        if (fileSize != expectedSize) return false;

        const auto validate_zero_padding = [&](std::streamoff begin, std::streamoff end) -> bool {
            if (end < begin) return false;
            if (end == begin) return true;

            in.seekg(begin, std::ios::beg);
            if (!in || in.tellg() != begin) return false;

            std::array<char, SECTION_ALIGNMENT> buffer{};
            std::streamoff remaining = end - begin;
            while (remaining > 0) {
                const size_t chunk = static_cast<size_t>(std::min<std::streamoff>(
                    remaining,
                    static_cast<std::streamoff>(buffer.size())
                ));

                if (!read_exact(in, buffer.data(), chunk)) return false;
                for (size_t i = 0; i < chunk; ++i) {
                    if (buffer[i] != 0) return false;
                }

                remaining -= static_cast<std::streamoff>(chunk);
            }

            return true;
        };

        std::streamoff prevEnd = static_cast<std::streamoff>(sizeof(BinaryHeader));
        for (const SectionSpec& section : sections) {
            if (!validate_zero_padding(prevEnd, section.offset)) return false;
            prevEnd = section.offset + section.bytes;
        }

        Network network{};
        network.featureTransformWeights.resize(ftCount);

        const auto seek_and_read = [&](const SectionSpec& section, void* dst, size_t bytes) -> bool {
            if (section.bytes != static_cast<std::streamoff>(bytes)) return false;
            if ((section.offset % SECTION_ALIGNMENT) != 0) return false;

            in.seekg(section.offset, std::ios::beg);
            if (!in || in.tellg() != section.offset) return false;
            if (!read_exact(in, dst, bytes)) return false;
            return in.tellg() == (section.offset + section.bytes);
        };

        if (!seek_and_read(
            sections[0],
            network.featureTransformWeights.data(),
            network.featureTransformWeights.size() * sizeof(int16_t)
        )) return false;

        if (!seek_and_read(sections[1], network.l1Weights.data(), network.l1Weights.size() * sizeof(int8_t))) return false;
        if (!seek_and_read(sections[2], network.l2Weights.data(), network.l2Weights.size() * sizeof(int8_t))) return false;
        if (!seek_and_read(sections[3], network.outWeights.data(), network.outWeights.size() * sizeof(int8_t))) return false;

        in.seekg(sections[4].offset, std::ios::beg);
        if (!in || in.tellg() != sections[4].offset) return false;
        if (!read_exact(in, network.featureBias.data(), network.featureBias.size() * sizeof(int16_t))) return false;
        if (!read_exact(in, network.l1Bias.data(), network.l1Bias.size() * sizeof(int32_t))) return false;
        if (!read_exact(in, network.l2Bias.data(), network.l2Bias.size() * sizeof(int32_t))) return false;
        if (!read_exact(in, &network.outBias, sizeof(network.outBias))) return false;
        if (in.tellg() != (sections[4].offset + sections[4].bytes)) return false;
        if (in.tellg() != expectedSize) return false;

        network_ = std::move(network);
        loaded_ = true;
        useScaledInference_ = try_load_sidecar_scales(path);
        return true;
    }

    void Runtime::add_feature_vector(std::array<int16_t, HIDDEN_SIZE>& target, uint32_t featureIndex) const {
        if (featureIndex >= HALF_KP_TOTAL_FEATURES) return;
        const size_t base = static_cast<size_t>(featureIndex) * static_cast<size_t>(HIDDEN_SIZE);
        const int16_t* src = network_.featureTransformWeights.data() + base;

#if defined(__AVX2__)
        // AVX2 Optimization: Process 16 int16_t elements at a time
        // HIDDEN_SIZE must be a multiple of 16 (512 is 16 * 32)
        int16_t* tgt = target.data();
        for (int i = 0; i < HIDDEN_SIZE; i += 16) {
            __m256i v_src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            __m256i v_tgt = _mm256_load_si256(reinterpret_cast<const __m256i*>(tgt + i));
            // adds_epi16 performs saturated addition, which is standard for NNUE
            _mm256_store_si256(reinterpret_cast<__m256i*>(tgt + i), _mm256_adds_epi16(v_tgt, v_src));
        }
#else
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            const int sum = static_cast<int>(target[i]) + static_cast<int>(src[i]);
            target[i] = clamp_sym_i16(sum);
        }
#endif
    }

    void Runtime::rebuild_accumulator(const Core::Position& pos, Accumulator512& accum) const {
        accum.clear();
        if (!loaded_) return;

        const Core::Bitboard whiteKing = pos.pieces(Core::KING, Core::WHITE);
        const Core::Bitboard blackKing = pos.pieces(Core::KING, Core::BLACK);
        if (whiteKing == 0 || blackKing == 0) return;

        const Core::Square whiteKingSq = Core::lsb(whiteKing);
        const Core::Square blackKingSq = Core::lsb(blackKing);

        accum.white = network_.featureBias;
        accum.black = network_.featureBias;

        for (int c = Core::WHITE; c <= Core::BLACK; ++c) {
            const Core::Color color = static_cast<Core::Color>(c);
            for (int pt = Core::PAWN; pt <= Core::QUEEN; ++pt) {
                Core::Bitboard bb = pos.pieces(static_cast<Core::PieceType>(pt), color);
                while (bb) {
                    const Core::Square sq = Core::pop_lsb(bb);
                    const uint32_t whiteIdx = halfkp_feature_index(
                        whiteKingSq, static_cast<Core::PieceType>(pt), color, sq, Core::WHITE
                    );
                    const uint32_t blackIdx = halfkp_feature_index(
                        blackKingSq, static_cast<Core::PieceType>(pt), color, sq, Core::BLACK
                    );

                    add_feature_vector(accum.white, whiteIdx);
                    add_feature_vector(accum.black, blackIdx);
                }
            }
        }
    }

    void Runtime::add_feature_vector_scaled(
        std::array<float, HIDDEN_SIZE>& target,
        uint32_t featureIndex,
        float scale
    ) const {
        if (featureIndex >= HALF_KP_TOTAL_FEATURES) return;
        const size_t base = static_cast<size_t>(featureIndex) * static_cast<size_t>(HIDDEN_SIZE);
        const int16_t* src = network_.featureTransformWeights.data() + base;
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            target[static_cast<size_t>(i)] += static_cast<float>(src[i]) * scale;
        }
    }

    namespace {
        // Exact integer dot product of a non-negative int8 activation vector
        // with an int8 weight row. The scalar and AVX2 forms are numerically
        // identical: products of int8 by [0,127] fit int16, and pairwise
        // sums of two such products fit int16 without saturation.
        int32_t dot_i8(const int8_t* RESTRICT act, const int8_t* RESTRICT w, int n) {
#if defined(__AVX2__)
            __m256i acc = _mm256_setzero_si256();
            const __m256i ones = _mm256_set1_epi16(1);
            int i = 0;
            for (; i + 32 <= n; i += 32) {
                const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act + i));
                const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
                const __m256i prod = _mm256_maddubs_epi16(a, b);   // 16x int16
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones));
            }
            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i sum128 = _mm_add_epi32(lo, hi);
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            int32_t sum = _mm_cvtsi128_si32(sum128);
            for (; i < n; ++i) sum += static_cast<int32_t>(w[i]) * static_cast<int32_t>(act[i]);
            return sum;
#else
            int32_t sum = 0;
            for (int i = 0; i < n; ++i) sum += static_cast<int32_t>(w[i]) * static_cast<int32_t>(act[i]);
            return sum;
#endif
        }

        // Feature-transform activation: relu(clamp((us-them)/64)). Because relu
        // zeroes every negative result, arithmetic shift (floor) and C division
        // (truncation) agree on the surviving values, so this is bit-exact with
        // the scalar reference.
        void transform_x0(const int16_t* RESTRICT us, const int16_t* RESTRICT them, int8_t* RESTRICT x0, int n) {
#if defined(__AVX2__)
            const __m256i zero = _mm256_setzero_si256();
            const __m256i maxv = _mm256_set1_epi16(127);
            int i = 0;
            for (; i + 16 <= n; i += 16) {
                __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(us + i));
                __m256i t = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(them + i));
                // Saturating sub: differences beyond int16 range divide down to
                // >=127 or <=0 and clamp identically to the int32 scalar path.
                __m256i c = _mm256_subs_epi16(u, t);
                c = _mm256_srai_epi16(c, 6);                 // /64 for the non-negative survivors
                c = _mm256_min_epi16(_mm256_max_epi16(c, zero), maxv);
                // pack 16x int16 [0,127] -> 16x int8 (lanes need reordering)
                __m256i packed = _mm256_packs_epi16(c, c);
                packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(x0 + i), _mm256_castsi256_si128(packed));
            }
            for (; i < n; ++i) {
                const int centered = static_cast<int>(us[i]) - static_cast<int>(them[i]);
                x0[i] = relu_i8(clamp_sym_i8(centered / FEATURE_TO_DENSE_SCALE));
            }
#else
            for (int i = 0; i < n; ++i) {
                const int centered = static_cast<int>(us[i]) - static_cast<int>(them[i]);
                x0[i] = relu_i8(clamp_sym_i8(centered / FEATURE_TO_DENSE_SCALE));
            }
#endif
        }
    }

    int Runtime::infer_side_to_move(const Accumulator512& accum, Core::Color stm) const {
        const auto& us = stm == Core::WHITE ? accum.white : accum.black;
        const auto& them = stm == Core::WHITE ? accum.black : accum.white;

        alignas(32) std::array<int8_t, HIDDEN_SIZE> x0{};
        alignas(32) std::array<int8_t, DENSE_L1_SIZE> x1{};
        alignas(32) std::array<int8_t, DENSE_L2_SIZE> x2{};

        transform_x0(us.data(), them.data(), x0.data(), HIDDEN_SIZE);

        for (int out = 0; out < DENSE_L1_SIZE; ++out) {
            const size_t row = static_cast<size_t>(out) * static_cast<size_t>(HIDDEN_SIZE);
            int32_t sum = network_.l1Bias[static_cast<size_t>(out)] +
                dot_i8(x0.data(), network_.l1Weights.data() + row, HIDDEN_SIZE);
            x1[static_cast<size_t>(out)] = relu_i8(clamp_sym_i8(sum / DENSE_TO_DENSE_SCALE));
        }

        for (int out = 0; out < DENSE_L2_SIZE; ++out) {
            const size_t row = static_cast<size_t>(out) * static_cast<size_t>(DENSE_L1_SIZE);
            int32_t sum = network_.l2Bias[static_cast<size_t>(out)] +
                dot_i8(x1.data(), network_.l2Weights.data() + row, DENSE_L1_SIZE);
            x2[static_cast<size_t>(out)] = relu_i8(clamp_sym_i8(sum / DENSE_TO_DENSE_SCALE));
        }

        int32_t out = network_.outBias +
            dot_i8(x2.data(), network_.outWeights.data(), DENSE_L2_SIZE);

        return static_cast<int>(out / OUTPUT_SCALE);
    }

    bool Runtime::self_test_quantized_inference() {
        // Pure-scalar reference, independent of the vectorized helpers above.
        auto infer_scalar = [](const Network& net, const Accumulator512& accum, Core::Color stm) -> int {
            const auto& us = stm == Core::WHITE ? accum.white : accum.black;
            const auto& them = stm == Core::WHITE ? accum.black : accum.white;

            std::array<int8_t, HIDDEN_SIZE> x0{};
            std::array<int8_t, DENSE_L1_SIZE> x1{};
            std::array<int8_t, DENSE_L2_SIZE> x2{};

            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                const int centered = static_cast<int>(us[i]) - static_cast<int>(them[i]);
                x0[i] = relu_i8(clamp_sym_i8(centered / FEATURE_TO_DENSE_SCALE));
            }
            for (int o = 0; o < DENSE_L1_SIZE; ++o) {
                int32_t sum = net.l1Bias[o];
                for (int i = 0; i < HIDDEN_SIZE; ++i)
                    sum += static_cast<int32_t>(net.l1Weights[o * HIDDEN_SIZE + i]) * x0[i];
                x1[o] = relu_i8(clamp_sym_i8(sum / DENSE_TO_DENSE_SCALE));
            }
            for (int o = 0; o < DENSE_L2_SIZE; ++o) {
                int32_t sum = net.l2Bias[o];
                for (int i = 0; i < DENSE_L1_SIZE; ++i)
                    sum += static_cast<int32_t>(net.l2Weights[o * DENSE_L1_SIZE + i]) * x1[i];
                x2[o] = relu_i8(clamp_sym_i8(sum / DENSE_TO_DENSE_SCALE));
            }
            int32_t out = net.outBias;
            for (int i = 0; i < DENSE_L2_SIZE; ++i)
                out += static_cast<int32_t>(net.outWeights[i]) * x2[i];
            return static_cast<int>(out / OUTPUT_SCALE);
        };

        // Deterministic pseudo-random network + accumulators (xorshift).
        uint64_t s = 0x9E3779B97F4A7C15ULL;
        auto next = [&s]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
        auto rndW = [&](int lo, int hi) {
            return static_cast<int>(lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1)));
        };

        Runtime rt;
        rt.network_.featureTransformWeights.clear();
        for (int trial = 0; trial < 64; ++trial) {
            for (auto& w : rt.network_.l1Weights) w = static_cast<int8_t>(rndW(-127, 127));
            for (auto& w : rt.network_.l2Weights) w = static_cast<int8_t>(rndW(-127, 127));
            for (auto& w : rt.network_.outWeights) w = static_cast<int8_t>(rndW(-127, 127));
            for (auto& b : rt.network_.l1Bias) b = rndW(-4000, 4000);
            for (auto& b : rt.network_.l2Bias) b = rndW(-4000, 4000);
            rt.network_.outBias = rndW(-20000, 20000);

            Accumulator512 accum{};
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                accum.white[i] = static_cast<int16_t>(rndW(-20000, 20000));
                accum.black[i] = static_cast<int16_t>(rndW(-20000, 20000));
            }

            for (Core::Color stm : {Core::WHITE, Core::BLACK}) {
                const int a = rt.infer_side_to_move(accum, stm);
                const int b = infer_scalar(rt.network_, accum, stm);
                if (a != b) {
                    std::fprintf(stderr, "[nnue self-test] trial %d stm %d: simd %d scalar %d\n",
                                 trial, static_cast<int>(stm), a, b);
                    return false;
                }
            }
        }
        return true;
    }

    int Runtime::infer_side_to_move_scaled(const Core::Position& pos, Core::Color perspective) const {
        const Core::Bitboard whiteKing = pos.pieces(Core::KING, Core::WHITE);
        const Core::Bitboard blackKing = pos.pieces(Core::KING, Core::BLACK);
        if (whiteKing == 0 || blackKing == 0) return 0;

        const Core::Square whiteKingSq = Core::lsb(whiteKing);
        const Core::Square blackKingSq = Core::lsb(blackKing);

        std::array<float, HIDDEN_SIZE> whiteAcc{};
        std::array<float, HIDDEN_SIZE> blackAcc{};
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            const float bias = static_cast<float>(network_.featureBias[static_cast<size_t>(i)]) * scales_.featureTransformBias;
            whiteAcc[static_cast<size_t>(i)] = bias;
            blackAcc[static_cast<size_t>(i)] = bias;
        }

        for (int c = Core::WHITE; c <= Core::BLACK; ++c) {
            const Core::Color color = static_cast<Core::Color>(c);
            for (int pt = Core::PAWN; pt <= Core::QUEEN; ++pt) {
                Core::Bitboard bb = pos.pieces(static_cast<Core::PieceType>(pt), color);
                while (bb) {
                    const Core::Square sq = Core::pop_lsb(bb);
                    const uint32_t whiteIdx = halfkp_feature_index(
                        whiteKingSq, static_cast<Core::PieceType>(pt), color, sq, Core::WHITE
                    );
                    const uint32_t blackIdx = halfkp_feature_index(
                        blackKingSq, static_cast<Core::PieceType>(pt), color, sq, Core::BLACK
                    );

                    add_feature_vector_scaled(whiteAcc, whiteIdx, scales_.featureTransformWeight);
                    add_feature_vector_scaled(blackAcc, blackIdx, scales_.featureTransformWeight);
                }
            }
        }

        const auto& us = perspective == Core::WHITE ? whiteAcc : blackAcc;
        const auto& them = perspective == Core::WHITE ? blackAcc : whiteAcc;

        std::array<float, HIDDEN_SIZE> x0{};
        std::array<float, DENSE_L1_SIZE> x1{};
        std::array<float, DENSE_L2_SIZE> x2{};

        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            const float centered = (us[static_cast<size_t>(i)] - them[static_cast<size_t>(i)]) /
                static_cast<float>(FEATURE_TO_DENSE_SCALE);
            x0[static_cast<size_t>(i)] = std::max(0.0f, centered);
        }

        for (int out = 0; out < DENSE_L1_SIZE; ++out) {
            float sum = static_cast<float>(network_.l1Bias[static_cast<size_t>(out)]) * scales_.l1Bias;
            const size_t row = static_cast<size_t>(out) * static_cast<size_t>(HIDDEN_SIZE);
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                sum += (static_cast<float>(network_.l1Weights[row + static_cast<size_t>(i)]) * scales_.l1Weight) *
                    x0[static_cast<size_t>(i)];
            }
            x1[static_cast<size_t>(out)] = std::max(0.0f, sum / static_cast<float>(DENSE_TO_DENSE_SCALE));
        }

        for (int out = 0; out < DENSE_L2_SIZE; ++out) {
            float sum = static_cast<float>(network_.l2Bias[static_cast<size_t>(out)]) * scales_.l2Bias;
            const size_t row = static_cast<size_t>(out) * static_cast<size_t>(DENSE_L1_SIZE);
            for (int i = 0; i < DENSE_L1_SIZE; ++i) {
                sum += (static_cast<float>(network_.l2Weights[row + static_cast<size_t>(i)]) * scales_.l2Weight) *
                    x1[static_cast<size_t>(i)];
            }
            x2[static_cast<size_t>(out)] = std::max(0.0f, sum / static_cast<float>(DENSE_TO_DENSE_SCALE));
        }

        float out = static_cast<float>(network_.outBias) * scales_.outBias;
        for (int i = 0; i < DENSE_L2_SIZE; ++i) {
            out += (static_cast<float>(network_.outWeights[static_cast<size_t>(i)]) * scales_.outWeight) *
                x2[static_cast<size_t>(i)];
        }

        return static_cast<int>(std::lround(out / static_cast<float>(OUTPUT_SCALE)));
    }

    bool Runtime::try_load_sidecar_scales(const std::string& path) {
        useScaledInference_ = false;
        scales_ = QuantScales{};

        std::vector<std::filesystem::path> candidates;
        candidates.emplace_back(std::filesystem::path(path).string() + ".meta.json");
        candidates.emplace_back(std::filesystem::path(path).filename().string() + ".meta.json");

        for (const auto& candidate : candidates) {
            if (candidate.empty() || !std::filesystem::exists(candidate)) continue;

            std::ifstream in(candidate, std::ios::binary);
            if (!in) continue;
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (text.empty()) continue;

            QuantScales parsed{};
            if (!extract_json_float(text, "feature_transform_weight", parsed.featureTransformWeight)) continue;
            if (!extract_json_float(text, "feature_transform_bias", parsed.featureTransformBias)) continue;
            if (!extract_json_float(text, "dense1_weight", parsed.l1Weight)) continue;
            if (!extract_json_float(text, "dense1_bias", parsed.l1Bias)) continue;
            if (!extract_json_float(text, "dense2_weight", parsed.l2Weight)) continue;
            if (!extract_json_float(text, "dense2_bias", parsed.l2Bias)) continue;
            if (!extract_json_float(text, "output_weight", parsed.outWeight)) continue;
            if (!extract_json_float(text, "output_bias", parsed.outBias)) continue;

            if (parsed.featureTransformWeight <= 0.0f || parsed.featureTransformBias <= 0.0f) continue;
            if (parsed.l1Weight <= 0.0f || parsed.l1Bias <= 0.0f) continue;
            if (parsed.l2Weight <= 0.0f || parsed.l2Bias <= 0.0f) continue;
            if (parsed.outWeight <= 0.0f || parsed.outBias <= 0.0f) continue;

            scales_ = parsed;
            useScaledInference_ = true;
            return true;
        }

        return false;
    }

    int Runtime::material_proxy_stm(const Core::Position& pos) const {
        static constexpr int VALUES[Core::PIECE_TYPE_NB] = {
            0, 100, 320, 330, 500, 900, 0
        };

        int white = 0;
        int black = 0;
        for (int pt = Core::PAWN; pt <= Core::KING; ++pt) {
            const int value = VALUES[pt];
            white += Core::popcount(pos.pieces(static_cast<Core::PieceType>(pt), Core::WHITE)) * value;
            black += Core::popcount(pos.pieces(static_cast<Core::PieceType>(pt), Core::BLACK)) * value;
        }
        const int whiteMinusBlack = white - black;
        return pos.side_to_move() == Core::WHITE ? whiteMinusBlack : -whiteMinusBlack;
    }

    int Runtime::evaluate(const Core::Position& pos) const {
        if (!loaded_) return material_proxy_stm(pos);

        if (useScaledInference_) {
            return infer_side_to_move_scaled(pos, pos.side_to_move());
        }

        Accumulator512 accum{};
        rebuild_accumulator(pos, accum);
        return infer_side_to_move(accum, pos.side_to_move());
    }

    int Runtime::evaluate_perspective(const Core::Position& pos, Core::Color perspective) const {
        if (!loaded_) return material_proxy_stm(pos);

        if (useScaledInference_) {
            return infer_side_to_move_scaled(pos, perspective);
        }

        Accumulator512 accum{};
        rebuild_accumulator(pos, accum);
        return infer_side_to_move(accum, perspective);
    }
}
