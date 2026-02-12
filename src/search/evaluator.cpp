#include "evaluator.h"

#include "../cores/bitboard.h"

#include <algorithm>
#include <cstdlib>

namespace Search {
    namespace {
        int piece_value(Core::PieceType pieceType) {
            switch (pieceType) {
                case Core::PAWN: return 100;
                case Core::KNIGHT: return 320;
                case Core::BISHOP: return 330;
                case Core::ROOK: return 500;
                case Core::QUEEN: return 900;
                case Core::KING: return 0;
                default: return 0;
            }
        }

        int center_bonus(Core::Square sq) {
            const int file = Core::file_of(sq);
            const int rank = Core::rank_of(sq);
            const int dist = std::abs(file - 3) + std::abs(rank - 3);
            return 14 - 3 * dist;
        }

        int psqt_bonus(Core::PieceType pieceType, Core::Square sq, Core::Color color) {
            const int file = Core::file_of(sq);
            const int rank = Core::rank_of(sq);
            const int relRank = color == Core::WHITE ? rank : (7 - rank);
            const int center = center_bonus(sq);

            switch (pieceType) {
                case Core::PAWN:
                    return relRank * 6 - std::abs(file - 3) * 2;
                case Core::KNIGHT:
                    return center;
                case Core::BISHOP:
                    return center / 2;
                case Core::ROOK:
                    return relRank * 3;
                case Core::QUEEN:
                    return center / 3;
                case Core::KING:
                    return -(center / 2);
                default:
                    return 0;
            }
        }
    }

    bool Evaluator::load_nnue(const std::string& path) {
        return nnue_.load_file(path);
    }

    int Evaluator::psqt_white_minus_black(const Core::Position& pos) const {
        int total = 0;
        for (int c = Core::WHITE; c <= Core::BLACK; ++c) {
            const Core::Color color = static_cast<Core::Color>(c);
            const int sign = color == Core::WHITE ? 1 : -1;

            for (int pt = Core::PAWN; pt <= Core::KING; ++pt) {
                Core::Bitboard bb = pos.pieces(static_cast<Core::PieceType>(pt), color);
                while (bb) {
                    const Core::Square sq = Core::pop_lsb(bb);
                    total += sign * piece_value(static_cast<Core::PieceType>(pt));
                    total += sign * psqt_bonus(static_cast<Core::PieceType>(pt), sq, color);
                }
            }
        }
        return total;
    }

    int Evaluator::evaluate(const Core::Position& pos) const {
        const int nnueStm = nnue_.evaluate(pos);
        const int psqtWb = psqt_white_minus_black(pos);
        const int psqtStm = pos.side_to_move() == Core::WHITE ? psqtWb : -psqtWb;
        return (nnueWeight_ * nnueStm + psqtWeight_ * psqtStm) / (nnueWeight_ + psqtWeight_);
    }
}
