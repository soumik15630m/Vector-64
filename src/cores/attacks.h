#ifndef ATTACKS_H
#define ATTACKS_H

#include "types.h"
#include "bitboard.h"

namespace Core {
    namespace Attacks {

        void init();

        Bitboard pawn_attacks(Square s, Color c);
        Bitboard knight_attacks(Square s);
        Bitboard king_attacks(Square s);

        Bitboard between_bb(Square s1, Square s2);
        Bitboard line_bb(Square s1, Square s2);

        struct Magic {
            Bitboard* attacks;
            Bitboard mask;
            Bitboard magic;
            int shift;
        };

        extern Magic RookMagics[64];
        extern Magic BishopMagics[64];

        inline Bitboard bishop_attacks(Square s, Bitboard occ) {
            Bitboard blockers = occ & BishopMagics[s].mask;

            unsigned idx = static_cast<unsigned>((blockers * BishopMagics[s].magic) >> BishopMagics[s].shift);
            return BishopMagics[s].attacks[idx];
        }

        inline Bitboard rook_attacks(Square s, Bitboard occ) {
            Bitboard blockers = occ & RookMagics[s].mask;
            unsigned idx = static_cast<unsigned>((blockers * RookMagics[s].magic) >> RookMagics[s].shift);
            return RookMagics[s].attacks[idx];
        }

        inline Bitboard queen_attacks(Square s, Bitboard occ) {
            return bishop_attacks(s, occ) | rook_attacks(s, occ);
        }

    }
}

#endif
