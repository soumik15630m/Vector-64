#include "move_ordering.h"

#include <algorithm>

namespace Search {
    namespace {
        constexpr int TT_MOVE_SCORE = 2'000'000;
        constexpr int CAPTURE_BASE_SCORE = 1'000'000;
        constexpr int KILLER_1_SCORE = 900'000;
        constexpr int KILLER_2_SCORE = 899'000;

        int piece_order_value(Core::PieceType pieceType) {
            switch (pieceType) {
                case Core::PAWN: return 1;
                case Core::KNIGHT: return 2;
                case Core::BISHOP: return 3;
                case Core::ROOK: return 4;
                case Core::QUEEN: return 5;
                case Core::KING: return 6;
                default: return 0;
            }
        }

        int mvv_lva(const Core::Position& pos, Core::Move move) {
            const Core::PieceType attacker = pos.piece_on(move.from_sq());
            Core::PieceType victim = pos.piece_on(move.to_sq());
            if (move.is_en_passant()) victim = Core::PAWN;
            return piece_order_value(victim) * 16 - piece_order_value(attacker);
        }
    }

    MoveOrdering::MoveOrdering() {
        clear();
    }

    void MoveOrdering::clear() {
        for (int ply = 0; ply < MAX_PLY; ++ply) {
            killers_[ply][0] = Core::Move::none();
            killers_[ply][1] = Core::Move::none();
        }
        std::fill(&history_[0][0][0], &history_[0][0][0] + Core::COLOR_NB * Core::SQUARE_NB * Core::SQUARE_NB, 0);
    }

    void MoveOrdering::age_history() {
        for (int c = 0; c < Core::COLOR_NB; ++c) {
            for (int from = 0; from < Core::SQUARE_NB; ++from) {
                for (int to = 0; to < Core::SQUARE_NB; ++to) {
                    history_[c][from][to] /= 2;
                }
            }
        }
    }

    void MoveOrdering::update_killers(int ply, Core::Move move) {
        if (ply < 0 || ply >= MAX_PLY || !move.is_ok() || move.is_capture()) return;
        if (move == killers_[ply][0]) return;
        killers_[ply][1] = killers_[ply][0];
        killers_[ply][0] = move;
    }

    void MoveOrdering::update_history(Core::Color side, Core::Move move, int depth) {
        if (move.is_capture()) return;
        const int bonus = depth * depth;
        int& score = history_[side][move.from_sq()][move.to_sq()];
        score = std::min(32767, score + bonus);
    }

    int MoveOrdering::score_move(const Core::Position& pos, Core::Move move, Core::Move ttMove, int ply) const {
        if (ttMove.is_ok() && move == ttMove) return TT_MOVE_SCORE;
        if (move.is_capture()) return CAPTURE_BASE_SCORE + mvv_lva(pos, move);

        if (ply >= 0 && ply < MAX_PLY) {
            if (move == killers_[ply][0]) return KILLER_1_SCORE;
            if (move == killers_[ply][1]) return KILLER_2_SCORE;
        }

        const Core::Color side = pos.side_to_move();
        return history_[side][move.from_sq()][move.to_sq()];
    }

    void MoveOrdering::sort_moves(const Core::Position& pos, Core::MoveList& moves, Core::Move ttMove, int ply) const {
        int scores[Core::MoveList::MAX_MOVES] = {0};
        for (int i = 0; i < moves.size(); ++i) {
            scores[i] = score_move(pos, moves.moves[i], ttMove, ply);
        }

        for (int i = 1; i < moves.size(); ++i) {
            Core::Move move = moves.moves[i];
            int score = scores[i];
            int j = i - 1;
            while (j >= 0 && scores[j] < score) {
                moves.moves[j + 1] = moves.moves[j];
                scores[j + 1] = scores[j];
                --j;
            }
            moves.moves[j + 1] = move;
            scores[j + 1] = score;
        }
    }
}
