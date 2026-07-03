#ifndef ATTACKS_H
#define ATTACKS_H

#include "types.h"
#include "bitboard.h"

// Fast parallel-bit-extract slider indexing on CPUs that support it
// (x86 BMI2 with fast PEXT). Everything else — ARM, older x86, and the
// AMD parts where PEXT is microcoded — automatically uses magic bitboards.
// Both paths share the same attack tables and produce identical attacks.
#if defined(__BMI2__)
#define ATTACKS_USE_PEXT 1
#include <immintrin.h>
#else
#define ATTACKS_USE_PEXT 0
#endif

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
            Bitboard magic;   // unused under PEXT
            int shift;        // unused under PEXT
        };

        extern Magic RookMagics[64];
        extern Magic BishopMagics[64];

        inline unsigned slider_index(Bitboard occ, const Magic& m) {
#if ATTACKS_USE_PEXT
            return static_cast<unsigned>(_pext_u64(occ, m.mask));
#else
            return static_cast<unsigned>(((occ & m.mask) * m.magic) >> m.shift);
#endif
        }

        inline Bitboard bishop_attacks(Square s, Bitboard occ) {
            return BishopMagics[s].attacks[slider_index(occ, BishopMagics[s])];
        }

        inline Bitboard rook_attacks(Square s, Bitboard occ) {
            return RookMagics[s].attacks[slider_index(occ, RookMagics[s])];
        }

        inline Bitboard queen_attacks(Square s, Bitboard occ) {
            return bishop_attacks(s, occ) | rook_attacks(s, occ);
        }

    }
}

#endif
