#include "movegen.h"
#include "types.h"
#include "bitboard.h"
#include "attacks.h"
#include <algorithm>

namespace Core {
    namespace {
        struct LegalityContext {
            Color us;
            Square kingSq;
            Bitboard occ;
            Bitboard enemyPawns;
            Bitboard enemyKnights;
            Bitboard enemyBishops;
            Bitboard enemyRooks;
            Bitboard enemyQueens;
            Bitboard enemyKing;
        };

        inline bool is_square_attacked_by_enemy(
            Square sq,
            Color us,
            Bitboard occ,
            Bitboard enemyPawns,
            Bitboard enemyKnights,
            Bitboard enemyBishops,
            Bitboard enemyRooks,
            Bitboard enemyQueens,
            Bitboard enemyKing
        ) {
            if (Attacks::pawn_attacks(sq, us) & enemyPawns) return true;
            if (Attacks::knight_attacks(sq) & enemyKnights) return true;
            if (Attacks::king_attacks(sq) & enemyKing) return true;
            if (Attacks::bishop_attacks(sq, occ) & (enemyBishops | enemyQueens)) return true;
            if (Attacks::rook_attacks(sq, occ) & (enemyRooks | enemyQueens)) return true;
            return false;
        }

        inline LegalityContext make_legality_context(const Position& pos) {
            const Color us = pos.side_to_move();
            const Color them = ~us;

            return {
                us,
                lsb(pos.pieces(KING, us)),
                pos.occupancy(),
                pos.pieces(PAWN, them),
                pos.pieces(KNIGHT, them),
                pos.pieces(BISHOP, them),
                pos.pieces(ROOK, them),
                pos.pieces(QUEEN, them),
                pos.pieces(KING, them)
            };
        }

        inline bool is_move_legal_with_ctx(const LegalityContext& ctx, Move m) {
            const Square from = m.from_sq();
            const Square to = m.to_sq();
            const Bitboard fromBB = square_bb(from);
            const Bitboard toBB = square_bb(to);

            Bitboard enemyPawns = ctx.enemyPawns;
            Bitboard enemyKnights = ctx.enemyKnights;
            Bitboard enemyBishops = ctx.enemyBishops;
            Bitboard enemyRooks = ctx.enemyRooks;
            Bitboard enemyQueens = ctx.enemyQueens;
            Bitboard enemyKing = ctx.enemyKing;

            Bitboard occAfter = ctx.occ ^ fromBB;

            if (m.is_capture()) {
                
                // handle ep like a real capture from the pawn's file
                Square capSq = to;
                if (m.is_en_passant()) {
                    capSq = make_square(static_cast<GenFile>(file_of(to)), static_cast<GenRank>(rank_of(from)));
                }

                const Bitboard capBB = square_bb(capSq);
                occAfter &= ~capBB;
                enemyPawns &= ~capBB;
                enemyKnights &= ~capBB;
                enemyBishops &= ~capBB;
                enemyRooks &= ~capBB;
                enemyQueens &= ~capBB;
                enemyKing &= ~capBB;
            }

            const Square kingSq = (from == ctx.kingSq) ? to : ctx.kingSq;

            if (m.is_castling()) {
                
                // castle path check: start and mid cant be attacked
                if (is_square_attacked_by_enemy(
                    from, ctx.us, ctx.occ,
                    enemyPawns, enemyKnights, enemyBishops, enemyRooks, enemyQueens, enemyKing
                )) {
                    return false;
                }

                const Square mid = static_cast<Square>((from + to) / 2);
                const Bitboard occMid = (ctx.occ ^ fromBB) | square_bb(mid);
                if (is_square_attacked_by_enemy(
                    mid, ctx.us, occMid,
                    enemyPawns, enemyKnights, enemyBishops, enemyRooks, enemyQueens, enemyKing
                )) {
                    return false;
                }

                occAfter |= toBB;
                const GenRank rank = static_cast<GenRank>(rank_of(from));
                const Square rookFrom = (to > from) ? make_square(FILE_H, rank) : make_square(FILE_A, rank);
                const Square rookTo = (to > from) ? make_square(FILE_F, rank) : make_square(FILE_D, rank);
                occAfter ^= square_bb(rookFrom);
                occAfter |= square_bb(rookTo);
            } else {
                occAfter |= toBB;
            }

            return !is_square_attacked_by_enemy(
                kingSq, ctx.us, occAfter,
                enemyPawns, enemyKnights, enemyBishops, enemyRooks, enemyQueens, enemyKing
            );
        }

    }

    template<Color Us>
    void generate_pawn_pseudos(const Position& pos, MoveList& moves) {
        constexpr Color Them = (Us == WHITE) ? BLACK : WHITE;
        constexpr Direction Up = (Us == WHITE) ? NORTH : SOUTH;
        constexpr int Rank7Val = (Us == WHITE) ? RANK_7 : RANK_2;

        Bitboard pawns = pos.pieces(PAWN, Us);
        Bitboard empty = ~pos.occupancy();
        Bitboard enemies = pos.pieces(Them);

        Bitboard single_push = shift<Up>(pawns) & empty;
        Bitboard b1 = single_push;
        while (b1) {
            Square to = pop_lsb(b1);
            Square from = static_cast<Square>(to - static_cast<int>(Up));

            if (rank_of(from) == Rank7Val) {
                moves.push_back(Move(from, to, Move::PROMOTION_Q));
                moves.push_back(Move(from, to, Move::PROMOTION_R));
                moves.push_back(Move(from, to, Move::PROMOTION_B));
                moves.push_back(Move(from, to, Move::PROMOTION_N));
            } else {
                moves.push_back(Move::make_quiet(from, to));
            }
        }

        Bitboard single_push_on_3 = single_push & (Us == WHITE ? RANK_3_BB : RANK_6_BB);
        Bitboard double_push = shift<Up>(single_push_on_3) & empty;
        while (double_push) {
            Square to = pop_lsb(double_push);
            Square from = static_cast<Square>(to - static_cast<int>(Up) * 2);
            moves.push_back(Move(from, to, Move::DOUBLE_PUSH));
        }

        auto gen_captures = [&](Bitboard caps, Direction Dir, bool is_ep) {
            if (is_ep) {
                Square ep = pos.ep_square();
                if (ep != SQ_NONE && has_bit(caps, ep)) {
                    Square from = static_cast<Square>(ep - static_cast<int>(Dir));
                    moves.push_back(Move(from, ep, Move::EN_PASSANT));
                }
            } else {
                caps &= enemies;
                while (caps) {
                    Square to = pop_lsb(caps);
                    Square from = static_cast<Square>(to - static_cast<int>(Dir));

                    if (rank_of(from) == Rank7Val) {
                        moves.push_back(Move(from, to, Move::PROMOTION_CAP_Q));
                        moves.push_back(Move(from, to, Move::PROMOTION_CAP_R));
                        moves.push_back(Move(from, to, Move::PROMOTION_CAP_B));
                        moves.push_back(Move(from, to, Move::PROMOTION_CAP_N));
                    } else {
                        moves.push_back(Move(from, to, Move::CAPTURE));
                    }
                }
            }
        };

        if constexpr (Us == WHITE) {
            gen_captures(shift<NORTH_WEST>(pawns), NORTH_WEST, false);
            gen_captures(shift<NORTH_EAST>(pawns), NORTH_EAST, false);
            gen_captures(shift<NORTH_WEST>(pawns), NORTH_WEST, true);
            gen_captures(shift<NORTH_EAST>(pawns), NORTH_EAST, true);
        } else {
            gen_captures(shift<SOUTH_WEST>(pawns), SOUTH_WEST, false);
            gen_captures(shift<SOUTH_EAST>(pawns), SOUTH_EAST, false);
            gen_captures(shift<SOUTH_WEST>(pawns), SOUTH_WEST, true);
            gen_captures(shift<SOUTH_EAST>(pawns), SOUTH_EAST, true);
        }
    }

    template<Color Us, PieceType Pt>
    void generate_piece_pseudos(const Position& pos, MoveList& moves) {
        constexpr Color Them = (Us == WHITE) ? BLACK : WHITE;
        Bitboard pieces = pos.pieces(Pt, Us);
        Bitboard own_pieces = pos.pieces(Us);
        Bitboard occupancy = pos.occupancy();

        while (pieces) {
            Square from = pop_lsb(pieces);
            Bitboard attacks = 0;

            if constexpr (Pt == KNIGHT) attacks = Attacks::knight_attacks(from);
            else if constexpr (Pt == BISHOP) attacks = Attacks::bishop_attacks(from, occupancy);
            else if constexpr (Pt == ROOK)   attacks = Attacks::rook_attacks(from, occupancy);
            else if constexpr (Pt == QUEEN)  attacks = Attacks::queen_attacks(from, occupancy);
            else if constexpr (Pt == KING)   attacks = Attacks::king_attacks(from);

            attacks &= ~own_pieces;

            while (attacks) {
                Square to = pop_lsb(attacks);
                if (has_bit(pos.pieces(Them), to)) {
                    moves.push_back(Move(from, to, Move::CAPTURE));
                } else {
                    moves.push_back(Move::make_quiet(from, to));
                }
            }
        }
    }

    void generate_castling_pseudos(const Position& pos, MoveList& moves) {
        Color us = pos.side_to_move();
        int rights = pos.castling_rights();
        Bitboard occ = pos.occupancy();
        Bitboard ourRooks = pos.pieces(ROOK, us);

        if (us == WHITE) {
            if ((rights & 1) && has_bit(ourRooks, SQ_H1)) {
                if (!has_bit(occ, SQ_F1) && !has_bit(occ, SQ_G1))
                    moves.push_back(Move(SQ_E1, SQ_G1, Move::CASTLING));
            }
            if ((rights & 2) && has_bit(ourRooks, SQ_A1)) {
                if (!has_bit(occ, SQ_B1) && !has_bit(occ, SQ_C1) && !has_bit(occ, SQ_D1))
                    moves.push_back(Move(SQ_E1, SQ_C1, Move::CASTLING));
            }
        } else {
            if ((rights & 4) && has_bit(ourRooks, SQ_H8)) {
                if (!has_bit(occ, SQ_F8) && !has_bit(occ, SQ_G8))
                    moves.push_back(Move(SQ_E8, SQ_G8, Move::CASTLING));
            }
            if ((rights & 8) && has_bit(ourRooks, SQ_A8)) {
                if (!has_bit(occ, SQ_B8) && !has_bit(occ, SQ_C8) && !has_bit(occ, SQ_D8))
                    moves.push_back(Move(SQ_E8, SQ_C8, Move::CASTLING));
            }
        }
    }

    void generate_pseudo_legal_moves(const Position& pos, MoveList& moves) {
        moves.clear();
        if (pos.side_to_move() == WHITE) {
            generate_pawn_pseudos<WHITE>(pos, moves);
            generate_piece_pseudos<WHITE, KNIGHT>(pos, moves);
            generate_piece_pseudos<WHITE, BISHOP>(pos, moves);
            generate_piece_pseudos<WHITE, ROOK>(pos, moves);
            generate_piece_pseudos<WHITE, QUEEN>(pos, moves);
            generate_piece_pseudos<WHITE, KING>(pos, moves);
        } else {
            generate_pawn_pseudos<BLACK>(pos, moves);
            generate_piece_pseudos<BLACK, KNIGHT>(pos, moves);
            generate_piece_pseudos<BLACK, BISHOP>(pos, moves);
            generate_piece_pseudos<BLACK, ROOK>(pos, moves);
            generate_piece_pseudos<BLACK, QUEEN>(pos, moves);
            generate_piece_pseudos<BLACK, KING>(pos, moves);
        }
        generate_castling_pseudos(pos, moves);
    }

    bool is_move_legal(Position& pos, Move m) {
        return is_move_legal_with_ctx(make_legality_context(pos), m);
    }

    void generate_legal_moves(Position& pos, MoveList& moves) {
        generate_pseudo_legal_moves(pos, moves);
        const LegalityContext ctx = make_legality_context(pos);

        int out = 0;
        for (int i = 0; i < moves.size(); ++i) {
            const Move m = moves.moves[i];
            if (is_move_legal_with_ctx(ctx, m)) {
                moves.moves[out++] = m;
            }
        }
        moves.count = out;
    }

}

