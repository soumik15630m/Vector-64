#include "nnue_consistency.h"

#include "../src/cores/attacks.h"
#include "../src/cores/bitboard.h"
#include "../src/cores/movegen.h"
#include "../src/cores/position.h"
#include "../src/cores/zobrist.h"
#include "../src/nnue/nnue.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>

namespace {
    constexpr const char* STARTPOS_FEN =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    std::string move_to_uci(Core::Move m) {
        if (!m.is_ok()) return "0000";

        std::string out;
        out.reserve(5);
        out.push_back(static_cast<char>('a' + Core::file_of(m.from_sq())));
        out.push_back(static_cast<char>('1' + Core::rank_of(m.from_sq())));
        out.push_back(static_cast<char>('a' + Core::file_of(m.to_sq())));
        out.push_back(static_cast<char>('1' + Core::rank_of(m.to_sq())));

        if (m.is_promotion()) {
            char promo = 'q';
            switch (m.promotion_type()) {
                case Core::KNIGHT: promo = 'n'; break;
                case Core::BISHOP: promo = 'b'; break;
                case Core::ROOK: promo = 'r'; break;
                case Core::QUEEN: promo = 'q'; break;
                default: break;
            }
            out.push_back(promo);
        }

        return out;
    }

    Core::Square king_square(const Core::Position& pos, Core::Color color) {
        const Core::Bitboard king = pos.pieces(Core::KING, color);
        if (king == 0) return Core::SQ_NONE;
        return Core::lsb(king);
    }

    bool accumulators_equal(
        const NNUE::Accumulator512& lhs,
        const NNUE::Accumulator512& rhs,
        Core::Color& side,
        int& index,
        int16_t& lhsValue,
        int16_t& rhsValue
    ) {
        for (int i = 0; i < NNUE::HIDDEN_SIZE; ++i) {
            const int16_t a = lhs.white[static_cast<size_t>(i)];
            const int16_t b = rhs.white[static_cast<size_t>(i)];
            if (a != b) {
                side = Core::WHITE;
                index = i;
                lhsValue = a;
                rhsValue = b;
                return false;
            }
        }

        for (int i = 0; i < NNUE::HIDDEN_SIZE; ++i) {
            const int16_t a = lhs.black[static_cast<size_t>(i)];
            const int16_t b = rhs.black[static_cast<size_t>(i)];
            if (a != b) {
                side = Core::BLACK;
                index = i;
                lhsValue = a;
                rhsValue = b;
                return false;
            }
        }

        return true;
    }

    bool apply_non_king_incremental_update(
        const Core::Position& pos,
        Core::Move move,
        NNUE::IncrementalAccumulator& accum
    ) {
        const Core::Square from = move.from_sq();
        const Core::Square to = move.to_sq();
        const Core::PieceType movingPiece = pos.piece_on(from);
        if (movingPiece == Core::NO_PIECE_TYPE || movingPiece == Core::KING) return false;

        const Core::Color us = pos.side_to_move();
        const Core::Color them = ~us;

        const Core::Square whiteKingSq = king_square(pos, Core::WHITE);
        const Core::Square blackKingSq = king_square(pos, Core::BLACK);
        if (whiteKingSq == Core::SQ_NONE || blackKingSq == Core::SQ_NONE) return false;

        const auto update_for = [&](Core::Color perspective, Core::Square kingSq) -> bool {
            accum.apply_piece_move(kingSq, movingPiece, us, from, to, perspective);

            if (move.is_capture()) {
                Core::Square capturedSquare = to;
                Core::PieceType capturedType = Core::NO_PIECE_TYPE;

                if (move.is_en_passant()) {
                    capturedSquare = Core::make_square(
                        static_cast<Core::GenFile>(Core::file_of(to)),
                        static_cast<Core::GenRank>(Core::rank_of(from))
                    );
                    capturedType = Core::PAWN;
                } else {
                    capturedType = pos.piece_on(to);
                }

                if (capturedType == Core::NO_PIECE_TYPE) return false;
                accum.apply_capture(kingSq, capturedType, them, capturedSquare, perspective);
            }

            if (move.is_promotion()) {
                const Core::PieceType promoted = move.promotion_type();
                if (promoted == Core::NO_PIECE_TYPE) return false;
                accum.apply_promotion(kingSq, us, to, promoted, perspective);
            }

            return true;
        };

        return update_for(Core::WHITE, whiteKingSq) && update_for(Core::BLACK, blackKingSq);
    }
}

int run_nnue_incremental_consistency_test(int games, int maxPlies, uint64_t seed) {
    if (games <= 0 || maxPlies <= 0) {
        std::cerr << "[ERROR] --nnue-consistency expects positive games and plies.\n";
        return 1;
    }

    Core::Attacks::init();
    Core::Zobrist::init();

    std::mt19937_64 rng(seed);

    for (int game = 0; game < games; ++game) {
        Core::Position pos;
        if (!pos.setFromFEN(STARTPOS_FEN)) {
            std::cerr << "[ERROR] Failed to initialize start position.\n";
            return 1;
        }

        NNUE::IncrementalAccumulator incremental;
        incremental.full_rebuild(pos);

        for (int ply = 0; ply < maxPlies; ++ply) {
            Core::MoveList legal;
            Core::generate_legal_moves(pos, legal);
            if (legal.size() == 0) break;

            std::uniform_int_distribution<int> pick(0, legal.size() - 1);
            const Core::Move move = legal[pick(rng)];

            const Core::PieceType moving = pos.piece_on(move.from_sq());
            const bool kingMove = moving == Core::KING;

            if (!kingMove) {
                if (!apply_non_king_incremental_update(pos, move, incremental)) {
                    std::cerr
                        << "[FAIL] Incremental pre-update failed at game " << (game + 1)
                        << ", ply " << (ply + 1)
                        << ", move " << move_to_uci(move) << ".\n";
                    return 1;
                }
            }

            Core::UndoInfo undo{};
            pos.make_move(move, undo);

            if (kingMove) {
                incremental.on_king_move_rebuild(pos);
            }

            NNUE::IncrementalAccumulator rebuilt;
            rebuilt.full_rebuild(pos);

            Core::Color mismatchSide = Core::WHITE;
            int mismatchIndex = -1;
            int16_t lhsValue = 0;
            int16_t rhsValue = 0;
            if (!accumulators_equal(
                incremental.data(),
                rebuilt.data(),
                mismatchSide,
                mismatchIndex,
                lhsValue,
                rhsValue
            )) {
                std::cerr
                    << "[FAIL] NNUE incremental mismatch at game " << (game + 1)
                    << ", ply " << (ply + 1)
                    << ", move " << move_to_uci(move)
                    << ", side " << (mismatchSide == Core::WHITE ? "white" : "black")
                    << ", index " << mismatchIndex
                    << " (" << lhsValue << " vs " << rhsValue << ").\n";
#ifndef NDEBUG
                assert(false && "NNUE incremental accumulator mismatch");
#endif
                return 1;
            }
        }
    }

    std::cout
        << "[PASS] NNUE incremental consistency (games=" << games
        << ", maxPlies=" << maxPlies
        << ", seed=" << seed
        << ").\n";
    return 0;
}
