#include "movegen.h"
#include "types.h"
#include "bitboard.h"
#include "attacks.h"
#include <algorithm>

namespace Core {
    namespace {
        enum GenType { GEN_ALL, GEN_CAPTURES, GEN_QUIETS };

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

        template<Color Us, GenType Gt>
        void generate_pawn_pseudos(const Position& pos, MoveList& moves) {
            constexpr Color Them = (Us == WHITE) ? BLACK : WHITE;
            constexpr Direction Up = (Us == WHITE) ? NORTH : SOUTH;
            constexpr int Rank7Val = (Us == WHITE) ? RANK_7 : RANK_2;
            constexpr Bitboard PromoRankBB = (Us == WHITE) ? RANK_8_BB : RANK_1_BB;

            Bitboard pawns = pos.pieces(PAWN, Us);
            Bitboard empty = ~pos.occupancy();
            Bitboard enemies = pos.pieces(Them);

            Bitboard single_push = shift<Up>(pawns) & empty;
            Bitboard pushes = single_push;
            if constexpr (Gt == GEN_CAPTURES) {
                // quiet promotions still belong with the tactical moves
                pushes &= PromoRankBB;
            } else if constexpr (Gt == GEN_QUIETS) {
                pushes &= ~PromoRankBB;
            }
            while (pushes) {
                Square to = pop_lsb(pushes);
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

            if constexpr (Gt != GEN_CAPTURES) {
                Bitboard single_push_on_3 = single_push & (Us == WHITE ? RANK_3_BB : RANK_6_BB);
                Bitboard double_push = shift<Up>(single_push_on_3) & empty;
                while (double_push) {
                    Square to = pop_lsb(double_push);
                    Square from = static_cast<Square>(to - static_cast<int>(Up) * 2);
                    moves.push_back(Move(from, to, Move::DOUBLE_PUSH));
                }
            }

            if constexpr (Gt != GEN_QUIETS) {
                constexpr Direction UpWest = (Us == WHITE) ? NORTH_WEST : SOUTH_WEST;
                constexpr Direction UpEast = (Us == WHITE) ? NORTH_EAST : SOUTH_EAST;
                const Bitboard attacksWest = shift<UpWest>(pawns);
                const Bitboard attacksEast = shift<UpEast>(pawns);

                auto gen_captures = [&](Bitboard atts, Direction Dir) {
                    Bitboard caps = atts & enemies;
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

                    const Square ep = pos.ep_square();
                    if (ep != SQ_NONE && has_bit(atts, ep)) {
                        Square from = static_cast<Square>(ep - static_cast<int>(Dir));
                        moves.push_back(Move(from, ep, Move::EN_PASSANT));
                    }
                };

                gen_captures(attacksWest, UpWest);
                gen_captures(attacksEast, UpEast);
            }
        }

        template<Color Us, PieceType Pt, GenType Gt>
        void generate_piece_pseudos(const Position& pos, MoveList& moves) {
            constexpr Color Them = (Us == WHITE) ? BLACK : WHITE;
            Bitboard pieces = pos.pieces(Pt, Us);
            Bitboard enemies = pos.pieces(Them);
            Bitboard occupancy = pos.occupancy();

            while (pieces) {
                Square from = pop_lsb(pieces);
                Bitboard attacks = 0;

                if constexpr (Pt == KNIGHT) attacks = Attacks::knight_attacks(from);
                else if constexpr (Pt == BISHOP) attacks = Attacks::bishop_attacks(from, occupancy);
                else if constexpr (Pt == ROOK)   attacks = Attacks::rook_attacks(from, occupancy);
                else if constexpr (Pt == QUEEN)  attacks = Attacks::queen_attacks(from, occupancy);
                else if constexpr (Pt == KING)   attacks = Attacks::king_attacks(from);

                if constexpr (Gt != GEN_QUIETS) {
                    Bitboard caps = attacks & enemies;
                    while (caps) {
                        Square to = pop_lsb(caps);
                        moves.push_back(Move(from, to, Move::CAPTURE));
                    }
                }

                if constexpr (Gt != GEN_CAPTURES) {
                    Bitboard quiets = attacks & ~occupancy;
                    while (quiets) {
                        Square to = pop_lsb(quiets);
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

        template<GenType Gt>
        void generate_pseudos(const Position& pos, MoveList& moves) {
            moves.clear();
            if (pos.side_to_move() == WHITE) {
                generate_pawn_pseudos<WHITE, Gt>(pos, moves);
                generate_piece_pseudos<WHITE, KNIGHT, Gt>(pos, moves);
                generate_piece_pseudos<WHITE, BISHOP, Gt>(pos, moves);
                generate_piece_pseudos<WHITE, ROOK, Gt>(pos, moves);
                generate_piece_pseudos<WHITE, QUEEN, Gt>(pos, moves);
                generate_piece_pseudos<WHITE, KING, Gt>(pos, moves);
            } else {
                generate_pawn_pseudos<BLACK, Gt>(pos, moves);
                generate_piece_pseudos<BLACK, KNIGHT, Gt>(pos, moves);
                generate_piece_pseudos<BLACK, BISHOP, Gt>(pos, moves);
                generate_piece_pseudos<BLACK, ROOK, Gt>(pos, moves);
                generate_piece_pseudos<BLACK, QUEEN, Gt>(pos, moves);
                generate_piece_pseudos<BLACK, KING, Gt>(pos, moves);
            }
            if constexpr (Gt != GEN_CAPTURES) {
                generate_castling_pseudos(pos, moves);
            }
        }

        void filter_legal(Position& pos, MoveList& moves) {
            const NodeLegality nl = make_node_legality(pos);

            int out = 0;
            for (int i = 0; i < moves.size(); ++i) {
                const Move m = moves.moves[i];
                if (is_legal(nl, m)) {
                    moves.moves[out++] = m;
                }
            }
            moves.count = out;
        }
    }

    NodeLegality make_node_legality(const Position& pos) {
        const Color us = pos.side_to_move();
        const Color them = ~us;

        NodeLegality nl{};
        nl.pos = &pos;
        nl.us = us;
        // A legal position always has a king; guard the degenerate case
        // (illegal FEN from a GUI) so lsb(0) can never produce a garbage
        // square and read out of the attack tables.
        const Bitboard kingBB = pos.pieces(KING, us);
        nl.kingSq = kingBB ? lsb(kingBB) : SQ_A1;
        nl.occ = pos.occupancy();
        nl.enemyPawns = pos.pieces(PAWN, them);
        nl.enemyKnights = pos.pieces(KNIGHT, them);
        nl.enemyBishops = pos.pieces(BISHOP, them);
        nl.enemyRooks = pos.pieces(ROOK, them);
        nl.enemyQueens = pos.pieces(QUEEN, them);
        nl.enemyKing = pos.pieces(KING, them);

        nl.checkers = pos.attackers_to(nl.kingSq) & pos.pieces(them);

        // Pins are computed lazily (see ensure_pins): nodes that cut off on
        // the TT move never generate, so they never pay for pin detection.
        nl.pinned = 0;
        nl.pinsReady = false;

#if defined(ENGINE_KING_DANGER)
        // Candidate #7: enemy attack map computed with our king removed from
        // occupancy, so a king move along a slider ray is scored correctly.
        {
            const Bitboard occNoKing = nl.occ ^ square_bb(nl.kingSq);
            Bitboard danger = 0;
            if (them == WHITE) {
                danger |= shift<NORTH_WEST>(nl.enemyPawns) | shift<NORTH_EAST>(nl.enemyPawns);
            } else {
                danger |= shift<SOUTH_WEST>(nl.enemyPawns) | shift<SOUTH_EAST>(nl.enemyPawns);
            }
            Bitboard b = nl.enemyKnights;
            while (b) danger |= Attacks::knight_attacks(pop_lsb(b));
            if (nl.enemyKing) danger |= Attacks::king_attacks(lsb(nl.enemyKing));
            b = nl.enemyBishops | nl.enemyQueens;
            while (b) danger |= Attacks::bishop_attacks(pop_lsb(b), occNoKing);
            b = nl.enemyRooks | nl.enemyQueens;
            while (b) danger |= Attacks::rook_attacks(pop_lsb(b), occNoKing);
            nl.kingDanger = danger;
        }
#endif
        return nl;
    }

    void ensure_pins(const NodeLegality& nl) {
        const Position& pos = *nl.pos;
        Bitboard pinned = 0;
        Bitboard snipers =
            (Attacks::rook_attacks(nl.kingSq, 0) & (nl.enemyRooks | nl.enemyQueens)) |
            (Attacks::bishop_attacks(nl.kingSq, 0) & (nl.enemyBishops | nl.enemyQueens));
        const Bitboard ours = pos.pieces(nl.us);
        while (snipers) {
            const Square sniper = pop_lsb(snipers);
            const Bitboard between = Attacks::between_bb(nl.kingSq, sniper) & nl.occ;
            if (between && !(between & (between - 1)) && (between & ours)) {
                pinned |= between;
            }
        }
        nl.pinned = pinned;
        nl.pinsReady = true;
    }

    bool is_move_legal_full(const NodeLegality& nl, Move m) {
        const Square from = m.from_sq();
        const Square to = m.to_sq();
        const Bitboard fromBB = square_bb(from);
        const Bitboard toBB = square_bb(to);

        Bitboard enemyPawns = nl.enemyPawns;
        Bitboard enemyKnights = nl.enemyKnights;
        Bitboard enemyBishops = nl.enemyBishops;
        Bitboard enemyRooks = nl.enemyRooks;
        Bitboard enemyQueens = nl.enemyQueens;
        Bitboard enemyKing = nl.enemyKing;

        Bitboard occAfter = nl.occ ^ fromBB;

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

        const Square kingSq = (from == nl.kingSq) ? to : nl.kingSq;

        if (m.is_castling()) {

            // castle path check: start and mid cant be attacked
            if (is_square_attacked_by_enemy(
                from, nl.us, nl.occ,
                enemyPawns, enemyKnights, enemyBishops, enemyRooks, enemyQueens, enemyKing
            )) {
                return false;
            }

            const Square mid = static_cast<Square>((from + to) / 2);
            const Bitboard occMid = (nl.occ ^ fromBB) | square_bb(mid);
            if (is_square_attacked_by_enemy(
                mid, nl.us, occMid,
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
            kingSq, nl.us, occAfter,
            enemyPawns, enemyKnights, enemyBishops, enemyRooks, enemyQueens, enemyKing
        );
    }

    bool is_pseudo_legal(const Position& pos, Move m) {
        if (!m.is_ok()) return false;

        // The generators only emit canonical flag encodings; 0b0101–0b0111
        // are unused and must not compare equal to any generated move.
        const int flags = m.flags();
        if (flags > Move::CAPTURE && flags < Move::PROMOTION_N) return false;

        const Square from = m.from_sq();
        const Square to = m.to_sq();
        if (from == to) return false;

        const Color us = pos.side_to_move();
        const Bitboard fromBB = square_bb(from);
        if (!(pos.pieces(us) & fromBB)) return false;

        const PieceType pt = pos.piece_on(from);
        const Bitboard toBB = square_bb(to);
        const Bitboard occ = pos.occupancy();
        const Bitboard enemies = pos.pieces(~us);

        if (pos.pieces(us) & toBB) return false;

        if (m.is_castling()) {
            if (pt != KING) return false;
            const int rights = pos.castling_rights();
            const Bitboard rooks = pos.pieces(ROOK, us);
            if (us == WHITE) {
                if (from != SQ_E1) return false;
                if (to == SQ_G1)
                    return (rights & 1) && has_bit(rooks, SQ_H1) &&
                           !has_bit(occ, SQ_F1) && !has_bit(occ, SQ_G1);
                if (to == SQ_C1)
                    return (rights & 2) && has_bit(rooks, SQ_A1) &&
                           !has_bit(occ, SQ_B1) && !has_bit(occ, SQ_C1) && !has_bit(occ, SQ_D1);
                return false;
            }
            if (from != SQ_E8) return false;
            if (to == SQ_G8)
                return (rights & 4) && has_bit(rooks, SQ_H8) &&
                       !has_bit(occ, SQ_F8) && !has_bit(occ, SQ_G8);
            if (to == SQ_C8)
                return (rights & 8) && has_bit(rooks, SQ_A8) &&
                       !has_bit(occ, SQ_B8) && !has_bit(occ, SQ_C8) && !has_bit(occ, SQ_D8);
            return false;
        }

        if (pt == PAWN) {
            const int up = us == WHITE ? 8 : -8;
            const int promoFromRank = us == WHITE ? RANK_7 : RANK_2;

            if (m.is_en_passant()) {
                if (to != pos.ep_square()) return false;
                return (Attacks::pawn_attacks(from, us) & toBB) != 0;
            }

            // Flag/geometry consistency: only 7th-rank pawns promote,
            // and every 7th-rank pawn move must carry a promotion flag.
            if ((rank_of(from) == promoFromRank) != m.is_promotion()) return false;

            if (m.is_capture()) {
                return (Attacks::pawn_attacks(from, us) & toBB & enemies) != 0;
            }
            if (m.is_double_push()) {
                const int startRank = us == WHITE ? RANK_2 : RANK_7;
                if (rank_of(from) != startRank) return false;
                if (to != from + 2 * up) return false;
                return !(occ & (square_bb(static_cast<Square>(from + up)) | toBB));
            }
            if (to != from + up) return false;
            return !(occ & toBB);
        }

        // Non-pawn moves carry no special flags.
        if (m.is_promotion() || m.is_en_passant() || m.is_double_push()) return false;

        Bitboard attacks = 0;
        switch (pt) {
            case KNIGHT: attacks = Attacks::knight_attacks(from); break;
            case BISHOP: attacks = Attacks::bishop_attacks(from, occ); break;
            case ROOK:   attacks = Attacks::rook_attacks(from, occ); break;
            case QUEEN:  attacks = Attacks::queen_attacks(from, occ); break;
            case KING:   attacks = Attacks::king_attacks(from); break;
            default: return false;
        }
        if (!(attacks & toBB)) return false;
        if (m.is_capture()) return (enemies & toBB) != 0;
        return !(occ & toBB);
    }

    void generate_pseudo_legal_moves(const Position& pos, MoveList& moves) {
        generate_pseudos<GEN_ALL>(pos, moves);
    }

    void generate_pseudo_captures(const Position& pos, MoveList& moves) {
        generate_pseudos<GEN_CAPTURES>(pos, moves);
    }

    void generate_pseudo_quiets(const Position& pos, MoveList& moves) {
        generate_pseudos<GEN_QUIETS>(pos, moves);
    }

    void generate_legal_moves(Position& pos, MoveList& moves) {
        generate_pseudos<GEN_ALL>(pos, moves);
        filter_legal(pos, moves);
    }

    void generate_legal_captures(Position& pos, MoveList& moves) {
        generate_pseudos<GEN_CAPTURES>(pos, moves);
        filter_legal(pos, moves);
    }

}
