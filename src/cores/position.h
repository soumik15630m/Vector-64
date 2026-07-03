#ifndef POSITION_H
#define POSITION_H

#include "types.h"
#include "bitboard.h"
#include "move.h"
#include "zobrist.h"
#include <string>
#include <string_view>

namespace Core {

    struct UndoInfo {
        uint64_t savedHash;
        int castlingRights;
        int halfmoveClock;
        Square epSquare;
        PieceType capturedPiece;
    };

    class Position {
    public:
        static constexpr int MAX_GAME_PLY = 1024;

        Position();

        bool setFromFEN(std::string_view fen);
        std::string toFEN() const;

        void make_move(Move m, UndoInfo& ui);
        void unmake_move(Move m, const UndoInfo& ui);

        void make_null_move(UndoInfo& ui);
        void unmake_null_move(const UndoInfo& ui);

        Bitboard pieces(PieceType pt, Color c) const {
            return byType[pt] & byColor[c];
        }

        Bitboard pieces(Color c) const {
            return byColor[c];
        }

        Bitboard pieces(PieceType pt) const {
            return byType[pt];
        }

        Bitboard occupancy() const {
            return byColor[WHITE] | byColor[BLACK];
        }

        PieceType piece_on(Square s) const { return board[s]; }
        Color color_on(Square s) const;

        Color side_to_move() const { return sideToMove; }
        Square ep_square() const { return epSquare; }
        int castling_rights() const { return castlingRights; }
        int halfmove_clock() const { return halfmoveClock; }
        int fullmove_number() const { return fullmoveNumber; }
        uint64_t hash() const { return zobristHash; }

        // The Zobrist key the position would have after playing m, computed
        // without mutating the board. Lets the search prefetch the child's
        // transposition-table bucket before make_move, so the memory fetch
        // overlaps the board update. Must equal hash() after make_move(m).
        uint64_t key_after(Move m) const;

        // Incremental white-minus-black scores. Material and piece-square
        // bonuses are tracked separately so NNUE blending can use the
        // positional part without double-counting material.
        int material_wb() const { return materialWb; }
        int psqt_wb() const { return psqtWb; }

        bool has_non_pawn_material(Color c) const {
            return (byColor[c] & (byType[KNIGHT] | byType[BISHOP] | byType[ROOK] | byType[QUEEN])) != 0;
        }

        // True if this position occurred at least once before within the
        // fifty-move window (twofold: standard draw convention inside search).
        bool is_repetition() const;

        Bitboard attackers_to(Square s, Bitboard occupied) const;
        Bitboard attackers_to(Square s) const { return attackers_to(s, occupancy()); }

        bool is_square_attacked(Square s, Color attackingSide) const;
        bool in_check() const;

        // False when the side that just moved left its own king in check,
        // i.e. the position is illegal to search (side to move could capture
        // the enemy king). Used to reject malformed FENs at the UCI boundary.
        bool opponent_in_check() const;

        bool is_ok() const;

    private:
        Bitboard byColor[COLOR_NB];
        Bitboard byType[PIECE_TYPE_NB];
        PieceType board[SQUARE_NB];

        Color sideToMove;
        int castlingRights;
        Square epSquare;
        int halfmoveClock;
        int fullmoveNumber;
        uint64_t zobristHash;
        int materialWb;
        int psqtWb;

        uint64_t history[MAX_GAME_PLY];
        int gamePly;

        void put_piece(PieceType pt, Color c, Square s);
        void remove_piece(Color c, Square s);
        void move_piece(Color c, Square from, Square to);
    };

}

#endif
