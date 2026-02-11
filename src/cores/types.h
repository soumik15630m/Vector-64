#ifndef TYPES_H
#define TYPES_H

#include <cstdint>

namespace Core {

    using Bitboard = uint64_t;

    enum Color {
        WHITE,
        BLACK,
        COLOR_NB
    };

    enum PieceType {
        NO_PIECE_TYPE,
        PAWN,
        KNIGHT,
        BISHOP,
        ROOK,
        QUEEN,
        KING,
        PIECE_TYPE_NB
    };

    enum Direction : int {
        NORTH = 8,
        SOUTH = -8,
        EAST = 1,
        WEST = -1,
        NORTH_EAST = 9,
        NORTH_WEST = 7,
        SOUTH_EAST = -7,
        SOUTH_WEST = -9
    };

    enum Square : int {
        SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
        SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
        SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
        SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
        SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
        SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
        SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
        SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
        SQ_NONE,
        SQUARE_NB = 64
    };

    enum GenFile {
        FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H,
        FILE_NB
    };

    enum GenRank {
        RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8,
        RANK_NB
    };

    constexpr inline Color operator~(Color c) {
        return static_cast<Color>(c ^ 1);
    }

    constexpr inline Square make_square(GenFile f, GenRank r) {
        return static_cast<Square>((r << 3) + f);
    }

    constexpr inline int rank_of(Square s) { return s >> 3; }
    constexpr inline int file_of(Square s) { return s & 7; }

}

#endif
