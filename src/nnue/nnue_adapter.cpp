#include "nnue.h"

namespace Search {
    namespace {
        constexpr int LEGACY_OUTPUT_SCALE = 16;
        constexpr int CORRECT_OUTPUT_SCALE = 600;
        constexpr int ADAPTER_SCALED_CP_LIMIT = 1000;

        int correct_output_scale(int rawCp) {
            if (rawCp >= -ADAPTER_SCALED_CP_LIMIT && rawCp <= ADAPTER_SCALED_CP_LIMIT) {
                return rawCp;
            }

            return static_cast<int>((static_cast<long long>(rawCp) * LEGACY_OUTPUT_SCALE) / CORRECT_OUTPUT_SCALE);
        }

    }

    int nnue_adapter_evaluate_stm(const NNUE::Runtime& nnue, const Core::Position& pos) {
        const int whiteCp = correct_output_scale(-nnue.evaluate_perspective(pos, Core::WHITE));
        return pos.side_to_move() == Core::WHITE ? whiteCp : -whiteCp;
    }
}
