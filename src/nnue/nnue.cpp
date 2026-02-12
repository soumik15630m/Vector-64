#include "nnue.h"

#include "../cores/bitboard.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>

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

        int16_t clamp_sym_i16(int value) {
            return static_cast<int16_t>(std::clamp(value, -32767, 32767));
        }

        int8_t clamp_sym_i8(int value) {
            return static_cast<int8_t>(std::clamp(value, -127, 127));
        }

        int8_t relu_i8(int8_t value) {
            return static_cast<int8_t>(std::max(0, static_cast<int>(value)));
        }

        void apply_delta(std::array<int16_t, HIDDEN_SIZE>& target, uint32_t featureIndex, int sign) {
            // Fallback deterministic transform for standalone accumulator updates.
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                const uint32_t mix = featureIndex * 2654435761u + static_cast<uint32_t>(i * 40503u);
                const int delta = static_cast<int>((mix >> 28) & 0xF) - 8;
                const int updated = static_cast<int>(target[static_cast<size_t>(i)]) + sign * delta;
                target[static_cast<size_t>(i)] = clamp_sym_i16(updated);
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

    IncrementalAccumulator::IncrementalAccumulator() {
        clear();
    }

    void IncrementalAccumulator::clear() {
        accum_.clear();
    }

    void IncrementalAccumulator::add_feature(uint32_t featureIndex, Core::Color perspective) {
        if (featureIndex >= HALF_KP_TOTAL_FEATURES) return;
        if (perspective == Core::WHITE) apply_delta(accum_.white, featureIndex, +1);
        else apply_delta(accum_.black, featureIndex, +1);
    }

    void IncrementalAccumulator::sub_feature(uint32_t featureIndex, Core::Color perspective) {
        if (featureIndex >= HALF_KP_TOTAL_FEATURES) return;
        if (perspective == Core::WHITE) apply_delta(accum_.white, featureIndex, -1);
        else apply_delta(accum_.black, featureIndex, -1);
    }

    void IncrementalAccumulator::full_rebuild(const Core::Position& pos) {
        clear();

        const Core::Bitboard whiteKing = pos.pieces(Core::KING, Core::WHITE);
        const Core::Bitboard blackKing = pos.pieces(Core::KING, Core::BLACK);
        if (whiteKing == 0 || blackKing == 0) return;

        const Core::Square whiteKingSq = Core::lsb(whiteKing);
        const Core::Square blackKingSq = Core::lsb(blackKing);

        for (int c = Core::WHITE; c <= Core::BLACK; ++c) {
            const Core::Color color = static_cast<Core::Color>(c);
            for (int pt = Core::PAWN; pt <= Core::QUEEN; ++pt) {
                Core::Bitboard bb = pos.pieces(static_cast<Core::PieceType>(pt), color);
                while (bb) {
                    const Core::Square sq = Core::pop_lsb(bb);
                    const uint32_t wIdx = halfkp_feature_index(
                        whiteKingSq, static_cast<Core::PieceType>(pt), color, sq, Core::WHITE
                    );
                    const uint32_t bIdx = halfkp_feature_index(
                        blackKingSq, static_cast<Core::PieceType>(pt), color, sq, Core::BLACK
                    );
                    add_feature(wIdx, Core::WHITE);
                    add_feature(bIdx, Core::BLACK);
                }
            }
        }
    }

    void IncrementalAccumulator::apply_piece_move(
        Core::Square kingSquare,
        Core::PieceType pieceType,
        Core::Color pieceColor,
        Core::Square from,
        Core::Square to,
        Core::Color perspective
    ) {
        const uint32_t fromIdx = halfkp_feature_index(kingSquare, pieceType, pieceColor, from, perspective);
        const uint32_t toIdx = halfkp_feature_index(kingSquare, pieceType, pieceColor, to, perspective);
        sub_feature(fromIdx, perspective);
        add_feature(toIdx, perspective);
    }

    void IncrementalAccumulator::apply_capture(
        Core::Square kingSquare,
        Core::PieceType capturedType,
        Core::Color capturedColor,
        Core::Square capturedSquare,
        Core::Color perspective
    ) {
        const uint32_t capturedIdx = halfkp_feature_index(
            kingSquare, capturedType, capturedColor, capturedSquare, perspective
        );
        sub_feature(capturedIdx, perspective);
    }

    void IncrementalAccumulator::apply_promotion(
        Core::Square kingSquare,
        Core::Color pieceColor,
        Core::Square square,
        Core::PieceType promotedType,
        Core::Color perspective
    ) {
        const uint32_t pawnIdx = halfkp_feature_index(kingSquare, Core::PAWN, pieceColor, square, perspective);
        const uint32_t promoIdx = halfkp_feature_index(kingSquare, promotedType, pieceColor, square, perspective);
        sub_feature(pawnIdx, perspective);
        add_feature(promoIdx, perspective);
    }

    void IncrementalAccumulator::on_king_move_rebuild(const Core::Position& pos) {
        full_rebuild(pos);
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
        return true;
    }

    void Runtime::add_feature_vector(std::array<int16_t, HIDDEN_SIZE>& target, uint32_t featureIndex) const {
        if (featureIndex >= HALF_KP_TOTAL_FEATURES) return;
        const size_t base = static_cast<size_t>(featureIndex) * static_cast<size_t>(HIDDEN_SIZE);
        const int16_t* src = network_.featureTransformWeights.data() + base;

        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            const int sum = static_cast<int>(target[static_cast<size_t>(i)]) + static_cast<int>(src[static_cast<size_t>(i)]);
            target[static_cast<size_t>(i)] = clamp_sym_i16(sum);
        }
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

    int Runtime::infer_side_to_move(const Accumulator512& accum, Core::Color stm) const {
        const auto& us = stm == Core::WHITE ? accum.white : accum.black;
        const auto& them = stm == Core::WHITE ? accum.black : accum.white;

        std::array<int8_t, HIDDEN_SIZE> x0{};
        std::array<int8_t, DENSE_L1_SIZE> x1{};
        std::array<int8_t, DENSE_L2_SIZE> x2{};

        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            const int centered = static_cast<int>(us[static_cast<size_t>(i)]) -
                                 static_cast<int>(them[static_cast<size_t>(i)]);
            const int8_t clipped = clamp_sym_i8(centered / FEATURE_TO_DENSE_SCALE);
            x0[static_cast<size_t>(i)] = relu_i8(clipped);
        }

        for (int out = 0; out < DENSE_L1_SIZE; ++out) {
            int32_t sum = network_.l1Bias[static_cast<size_t>(out)];
            const size_t row = static_cast<size_t>(out) * static_cast<size_t>(HIDDEN_SIZE);
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                sum += static_cast<int32_t>(network_.l1Weights[row + static_cast<size_t>(i)]) *
                       static_cast<int32_t>(x0[static_cast<size_t>(i)]);
            }
            x1[static_cast<size_t>(out)] = relu_i8(clamp_sym_i8(sum / DENSE_TO_DENSE_SCALE));
        }

        for (int out = 0; out < DENSE_L2_SIZE; ++out) {
            int32_t sum = network_.l2Bias[static_cast<size_t>(out)];
            const size_t row = static_cast<size_t>(out) * static_cast<size_t>(DENSE_L1_SIZE);
            for (int i = 0; i < DENSE_L1_SIZE; ++i) {
                sum += static_cast<int32_t>(network_.l2Weights[row + static_cast<size_t>(i)]) *
                       static_cast<int32_t>(x1[static_cast<size_t>(i)]);
            }
            x2[static_cast<size_t>(out)] = relu_i8(clamp_sym_i8(sum / DENSE_TO_DENSE_SCALE));
        }

        int32_t out = network_.outBias;
        for (int i = 0; i < DENSE_L2_SIZE; ++i) {
            out += static_cast<int32_t>(network_.outWeights[static_cast<size_t>(i)]) *
                   static_cast<int32_t>(x2[static_cast<size_t>(i)]);
        }
        return static_cast<int>(out / OUTPUT_SCALE);
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

        Accumulator512 accum{};
        rebuild_accumulator(pos, accum);
        return infer_side_to_move(accum, pos.side_to_move());
    }
}
