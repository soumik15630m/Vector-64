#ifndef NNUE_NNUE_H
#define NNUE_NNUE_H

#include "../cores/position.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace NNUE {

    constexpr int HALF_KP_TOTAL_FEATURES = 81920;
    constexpr int HALF_KP_PIECE_BUCKETS = 10;
    constexpr int HIDDEN_SIZE = 512;
    constexpr int DENSE_L1_SIZE = 32;
    constexpr int DENSE_L2_SIZE = 32;
    constexpr int SECTION_ALIGNMENT = 64;

#pragma pack(push, 1)
    struct BinaryHeader {
        char magic[13];
        uint32_t version;
        uint32_t hiddenSize;
    };
#pragma pack(pop)

    struct alignas(64) Accumulator512 {
        std::array<int16_t, HIDDEN_SIZE> white{};
        std::array<int16_t, HIDDEN_SIZE> black{};

        void clear();
    };

    uint32_t halfkp_feature_index(
        Core::Square kingSquare,
        Core::PieceType pieceType,
        Core::Color pieceColor,
        Core::Square pieceSquare,
        Core::Color perspective
    );

    class IncrementalAccumulator {
    public:
        IncrementalAccumulator();

        void clear();
        void full_rebuild(const Core::Position& pos);

        void apply_piece_move(
            Core::Square kingSquare,
            Core::PieceType pieceType,
            Core::Color pieceColor,
            Core::Square from,
            Core::Square to,
            Core::Color perspective
        );

        void apply_capture(
            Core::Square kingSquare,
            Core::PieceType capturedType,
            Core::Color capturedColor,
            Core::Square capturedSquare,
            Core::Color perspective
        );

        void apply_promotion(
            Core::Square kingSquare,
            Core::Color pieceColor,
            Core::Square square,
            Core::PieceType promotedType,
            Core::Color perspective
        );

        void on_king_move_rebuild(const Core::Position& pos);

        const Accumulator512& data() const { return accum_; }

    private:
        void add_feature(uint32_t featureIndex, Core::Color perspective);
        void sub_feature(uint32_t featureIndex, Core::Color perspective);

        Accumulator512 accum_{};
    };

    class Runtime {
    public:
        bool load_file(const std::string& path);
        bool is_loaded() const { return loaded_; }

        // Returns a side-to-move centipawn estimate.
        int evaluate(const Core::Position& pos) const;

    private:
        struct Network {
            std::vector<int16_t> featureTransformWeights{};
            std::array<int8_t, DENSE_L1_SIZE * HIDDEN_SIZE> l1Weights{};
            std::array<int8_t, DENSE_L2_SIZE * DENSE_L1_SIZE> l2Weights{};
            std::array<int8_t, DENSE_L2_SIZE> outWeights{};

            std::array<int16_t, HIDDEN_SIZE> featureBias{};
            std::array<int32_t, DENSE_L1_SIZE> l1Bias{};
            std::array<int32_t, DENSE_L2_SIZE> l2Bias{};
            int32_t outBias = 0;
        };

        static bool read_exact(std::ifstream& in, void* dst, size_t bytes);

        void add_feature_vector(std::array<int16_t, HIDDEN_SIZE>& target, uint32_t featureIndex) const;
        void rebuild_accumulator(const Core::Position& pos, Accumulator512& accum) const;
        int infer_side_to_move(const Accumulator512& accum, Core::Color stm) const;
        int material_proxy_stm(const Core::Position& pos) const;

        bool loaded_ = false;
        Network network_{};
    };

}

#endif
