#include "evaluator.h"

#include "../nnue/nnue_adapter.h"

namespace Search {

    bool Evaluator::load_nnue(const std::string& path) {
        return nnue_.load_file(path);
    }

    int Evaluator::evaluate(const Core::Position& pos) const {
        const int sign = pos.side_to_move() == Core::WHITE ? 1 : -1;
        const int psqtStm = sign * pos.psqt_wb();

        if (!nnue_.is_loaded()) {
            return sign * pos.material_wb() + psqtStm;
        }

        const int nnueStm = nnue_adapter_evaluate_stm(nnue_, pos);
        return (nnueWeight_ * nnueStm + psqtWeight_ * psqtStm) / (nnueWeight_ + psqtWeight_);
    }
}
