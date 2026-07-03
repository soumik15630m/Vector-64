#include "position.h"
#include "attacks.h"
#include "invariants.h"
#include <sstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cctype>

namespace Core {

    constexpr int CASTLE_WK = 1;
    constexpr int CASTLE_WQ = 2;
    constexpr int CASTLE_BK = 4;
    constexpr int CASTLE_BQ = 8;

    int CastlingSpoilers[64];
    bool SpoilersInitialized = false;

    void init_spoilers() {
        if (SpoilersInitialized) return;

        for (int i = 0; i < 64; ++i) CastlingSpoilers[i] = 15;

        CastlingSpoilers[SQ_A1] &= ~CASTLE_WQ;
        CastlingSpoilers[SQ_E1] &= ~(CASTLE_WK | CASTLE_WQ);
        CastlingSpoilers[SQ_H1] &= ~CASTLE_WK;

        CastlingSpoilers[SQ_A8] &= ~CASTLE_BQ;
        CastlingSpoilers[SQ_E8] &= ~(CASTLE_BK | CASTLE_BQ);
        CastlingSpoilers[SQ_H8] &= ~CASTLE_BK;

        SpoilersInitialized = true;
    }

    // Piece-square bonus tables, indexed [color][type][square]. The running
    // white-minus-black totals let evaluate() reduce to a sign flip.
    int PsqTable[COLOR_NB][PIECE_TYPE_NB][SQUARE_NB];
    bool PsqInitialized = false;

    namespace {
        int psq_center_bonus(Square sq) {
            const int file = file_of(sq);
            const int rank = rank_of(sq);
            const int dist = std::abs(file - 3) + std::abs(rank - 3);
            return 14 - 3 * dist;
        }

        int psq_bonus(PieceType pt, Square sq, Color c) {
            const int file = file_of(sq);
            const int rank = rank_of(sq);
            const int relRank = c == WHITE ? rank : (7 - rank);
            const int center = psq_center_bonus(sq);

            switch (pt) {
                case PAWN:   return relRank * 6 - std::abs(file - 3) * 2;
                case KNIGHT: return center;
                case BISHOP: return center / 2;
                case ROOK:   return relRank * 3;
                case QUEEN:  return center / 3;
                case KING:   return -(center / 2);
                default:     return 0;
            }
        }
    }

    void init_psq_tables() {
        if (PsqInitialized) return;
        for (int c = WHITE; c <= BLACK; ++c) {
            for (int pt = NO_PIECE_TYPE; pt < PIECE_TYPE_NB; ++pt) {
                for (int s = 0; s < SQUARE_NB; ++s) {
                    const PieceType type = static_cast<PieceType>(pt);
                    const Square sq = static_cast<Square>(s);
                    PsqTable[c][pt][s] = (pt == NO_PIECE_TYPE)
                        ? 0
                        : psq_bonus(type, sq, static_cast<Color>(c));
                }
            }
        }
        PsqInitialized = true;
    }

    Position::Position() {

        init_spoilers();
        init_psq_tables();

        std::memset(byColor, 0, sizeof(byColor));
        std::memset(byType, 0, sizeof(byType));
        std::memset(history, 0, sizeof(history));
        std::fill(std::begin(board), std::end(board), NO_PIECE_TYPE);

        sideToMove = WHITE;
        castlingRights = 0;
        epSquare = SQ_NONE;
        halfmoveClock = 0;
        fullmoveNumber = 1;
        zobristHash = 0;
        materialWb = 0;
        psqtWb = 0;
        gamePly = 0;
    }

    void Position::put_piece(PieceType pt, Color c, Square s) {
        if (pt == NO_PIECE_TYPE) return;
        Bitboard bb = square_bb(s);
        byColor[c] |= bb;
        byType[pt] |= bb;
        board[s] = pt;
        const int sign = (c == WHITE) ? 1 : -1;
        materialWb += sign * PieceValue[pt];
        psqtWb += sign * PsqTable[c][pt][s];
        zobristHash ^= Zobrist::psq[c][pt][s];
    }

    void Position::remove_piece(Color c, Square s) {
        const PieceType pt = board[s];
        if (pt == NO_PIECE_TYPE) return;

        Bitboard bb = square_bb(s);
        byColor[c] ^= bb;
        byType[pt] ^= bb;
        board[s] = NO_PIECE_TYPE;
        const int sign = (c == WHITE) ? 1 : -1;
        materialWb -= sign * PieceValue[pt];
        psqtWb -= sign * PsqTable[c][pt][s];
        zobristHash ^= Zobrist::psq[c][pt][s];
    }

    void Position::move_piece(Color c, Square from, Square to) {
        const PieceType pt = board[from];
        if (pt == NO_PIECE_TYPE) return;

        Bitboard from_bb = square_bb(from);
        Bitboard to_bb = square_bb(to);
        Bitboard mask = from_bb | to_bb;

        byType[pt] ^= mask;
        byColor[c] ^= mask;
        board[from] = NO_PIECE_TYPE;
        board[to] = pt;

        const int delta = PsqTable[c][pt][to] - PsqTable[c][pt][from];
        psqtWb += (c == WHITE) ? delta : -delta;

        zobristHash ^= Zobrist::psq[c][pt][from];
        zobristHash ^= Zobrist::psq[c][pt][to];
    }

    Color Position::color_on(Square s) const {
        Bitboard bb = square_bb(s);
        if (byColor[WHITE] & bb) return WHITE;
        if (byColor[BLACK] & bb) return BLACK;
        return COLOR_NB;
    }

    bool Position::setFromFEN(std::string_view fen) {
        std::memset(byColor, 0, sizeof(byColor));
        std::memset(byType, 0, sizeof(byType));
        std::fill(std::begin(board), std::end(board), NO_PIECE_TYPE);
        zobristHash = 0;
        materialWb = 0;
        psqtWb = 0;
        gamePly = 0;

        std::stringstream ss(std::string{fen});
        std::string board_str, side, castle, ep, half, full;

        ss >> board_str >> side >> castle >> ep >> half >> full;

        int rank = 7;
        int file = 0;
        for (char c : board_str) {
            if (c == '/') {
                rank--;
                file = 0;
            } else if (isdigit(c)) {
                file += (c - '0');
            } else {
                Color color = isupper(c) ? WHITE : BLACK;
                PieceType pt = NO_PIECE_TYPE;
                switch (tolower(c)) {
                    case 'p': pt = PAWN; break;
                    case 'n': pt = KNIGHT; break;
                    case 'b': pt = BISHOP; break;
                    case 'r': pt = ROOK; break;
                    case 'q': pt = QUEEN; break;
                    case 'k': pt = KING; break;
                }
                put_piece(pt, color, make_square((GenFile)file, (GenRank)rank));
                file++;
            }
        }

        sideToMove = (side == "w") ? WHITE : BLACK;
        if (sideToMove == BLACK) zobristHash ^= Zobrist::side;

        castlingRights = 0;
        if (castle != "-") {
            for (char c : castle) {
                if (c == 'K') castlingRights |= CASTLE_WK;
                if (c == 'Q') castlingRights |= CASTLE_WQ;
                if (c == 'k') castlingRights |= CASTLE_BK;
                if (c == 'q') castlingRights |= CASTLE_BQ;
            }
        }
        zobristHash ^= Zobrist::castling[castlingRights];

        epSquare = SQ_NONE;
        if (ep != "-") {
            GenFile f = (GenFile)(ep[0] - 'a');
            GenRank r = (GenRank)(ep[1] - '1');
            epSquare = make_square(f, r);
            zobristHash ^= Zobrist::enpassant[f];
        }

        try {
            halfmoveClock = half.empty() ? 0 : std::stoi(half);
            fullmoveNumber = full.empty() ? 1 : std::stoi(full);
        } catch (...) {
            halfmoveClock = 0;
            fullmoveNumber = 1;
        }

        history[gamePly] = zobristHash;

        return is_ok();
    }

    std::string Position::toFEN() const {
        std::stringstream ss;
        for (int r = RANK_8; r >= RANK_1; --r) {
            int emptyCount = 0;
            for (int f = FILE_A; f <= FILE_H; ++f) {
                Square s = make_square((GenFile)f, (GenRank)r);
                PieceType pt = piece_on(s);
                Color c = color_on(s);

                if (pt == NO_PIECE_TYPE) {
                    emptyCount++;
                } else {
                    if (emptyCount > 0) {
                        ss << emptyCount;
                        emptyCount = 0;
                    }
                    char pieceChar = '?';
                    switch (pt) {
                        case PAWN:   pieceChar = 'p'; break;
                        case KNIGHT: pieceChar = 'n'; break;
                        case BISHOP: pieceChar = 'b'; break;
                        case ROOK:   pieceChar = 'r'; break;
                        case QUEEN:  pieceChar = 'q'; break;
                        case KING:   pieceChar = 'k'; break;
                        default: break;
                    }
                    if (c == WHITE) pieceChar = static_cast<char>(toupper(pieceChar));
                    ss << pieceChar;
                }
            }
            if (emptyCount > 0) ss << emptyCount;
            if (r > RANK_1) ss << '/';
        }
        ss << (sideToMove == WHITE ? " w " : " b ");
        if (castlingRights == 0) {
            ss << "-";
        } else {
            if (castlingRights & CASTLE_WK) ss << 'K';
            if (castlingRights & CASTLE_WQ) ss << 'Q';
            if (castlingRights & CASTLE_BK) ss << 'k';
            if (castlingRights & CASTLE_BQ) ss << 'q';
        }
        ss << " ";
        if (epSquare == SQ_NONE) {
            ss << "-";
        } else {
            char f = static_cast<char>('a' + file_of(epSquare));
            char r = static_cast<char>('1' + rank_of(epSquare));
            ss << f << r;
        }
        ss << " " << halfmoveClock << " " << fullmoveNumber;
        return ss.str();
    }

    void Position::make_move(Move m, UndoInfo& ui) {
        ASSERT_CONSISTENCY(*this);

        Square from = m.from_sq();
        Square to = m.to_sq();
        PieceType movingPiece = board[from];

        ui.capturedPiece = NO_PIECE_TYPE;
        ui.castlingRights = castlingRights;
        ui.epSquare = epSquare;
        ui.halfmoveClock = halfmoveClock;
        ui.savedHash = zobristHash;

        bool resetClock = false;
        if (movingPiece == PAWN) resetClock = true;

        if (m.is_capture()) {
            resetClock = true;
            Square capSq = to;
            if (m.is_en_passant()) {
                capSq = make_square((GenFile)file_of(to), (GenRank)rank_of(from));
            }
            ui.capturedPiece = board[capSq];
            remove_piece(~sideToMove, capSq);
            castlingRights &= CastlingSpoilers[to];
        }

        move_piece(sideToMove, from, to);

        if (m.is_promotion()) {
            remove_piece(sideToMove, to);
            put_piece(m.promotion_type(), sideToMove, to);
        }

        if (m.is_castling()) {
            Square rFrom, rTo;
            if (to > from) {
                rFrom = make_square(FILE_H, (GenRank)rank_of(from));
                rTo = make_square(FILE_F, (GenRank)rank_of(from));
            } else {
                rFrom = make_square(FILE_A, (GenRank)rank_of(from));
                rTo = make_square(FILE_D, (GenRank)rank_of(from));
            }
            move_piece(sideToMove, rFrom, rTo);
        }

        castlingRights &= CastlingSpoilers[from];

        zobristHash ^= Zobrist::castling[ui.castlingRights];
        zobristHash ^= Zobrist::castling[castlingRights];

        if (epSquare != SQ_NONE) zobristHash ^= Zobrist::enpassant[file_of(epSquare)];
        epSquare = SQ_NONE;

        if (m.is_double_push()) {
            Square epCand = (sideToMove == WHITE) ? (Square)(from + 8) : (Square)(from - 8);
            epSquare = epCand;
            zobristHash ^= Zobrist::enpassant[file_of(epSquare)];
        }

        if (resetClock) halfmoveClock = 0;
        else halfmoveClock++;

        if (sideToMove == BLACK) fullmoveNumber++;
        sideToMove = ~sideToMove;
        zobristHash ^= Zobrist::side;

        gamePly++;
        if (gamePly < MAX_GAME_PLY) history[gamePly] = zobristHash;
        ASSERT_CONSISTENCY(*this);
    }

    uint64_t Position::key_after(Move m) const {
        const Square from = m.from_sq();
        const Square to = m.to_sq();
        const Color us = sideToMove;
        const Color them = ~us;
        const PieceType movingPiece = board[from];

        uint64_t k = zobristHash;
        int newCastling = castlingRights;

        if (m.is_capture()) {
            Square capSq = to;
            PieceType captured;
            if (m.is_en_passant()) {
                capSq = make_square((GenFile)file_of(to), (GenRank)rank_of(from));
                captured = PAWN;
            } else {
                captured = board[to];
            }
            k ^= Zobrist::psq[them][captured][capSq];
            newCastling &= CastlingSpoilers[to];
        }

        // Moving piece leaves `from` and lands on `to`; a promotion changes
        // which piece lands (the intermediate pawn-on-`to` term cancels out).
        k ^= Zobrist::psq[us][movingPiece][from];
        k ^= Zobrist::psq[us][m.is_promotion() ? m.promotion_type() : movingPiece][to];

        if (m.is_castling()) {
            Square rFrom, rTo;
            if (to > from) {
                rFrom = make_square(FILE_H, (GenRank)rank_of(from));
                rTo   = make_square(FILE_F, (GenRank)rank_of(from));
            } else {
                rFrom = make_square(FILE_A, (GenRank)rank_of(from));
                rTo   = make_square(FILE_D, (GenRank)rank_of(from));
            }
            k ^= Zobrist::psq[us][ROOK][rFrom];
            k ^= Zobrist::psq[us][ROOK][rTo];
        }

        newCastling &= CastlingSpoilers[from];
        k ^= Zobrist::castling[castlingRights];
        k ^= Zobrist::castling[newCastling];

        if (epSquare != SQ_NONE) k ^= Zobrist::enpassant[file_of(epSquare)];
        if (m.is_double_push()) {
            const Square epCand = (us == WHITE) ? (Square)(from + 8) : (Square)(from - 8);
            k ^= Zobrist::enpassant[file_of(epCand)];
        }

        k ^= Zobrist::side;
        return k;
    }

    void Position::unmake_move(Move m, const UndoInfo& ui) {
        gamePly--;
        sideToMove = ~sideToMove;
        if (sideToMove == BLACK) fullmoveNumber--;

        Square from = m.from_sq();
        Square to = m.to_sq();

        // Restore pieces through the shared helpers so the mailbox and the
        // incremental psq score stay consistent; the hash they touch is
        // overwritten by the saved value below.
        if (m.is_promotion()) {
            remove_piece(sideToMove, to);
            put_piece(PAWN, sideToMove, from);
        } else {
            move_piece(sideToMove, to, from);
        }

        if (m.is_castling()) {
            Square rFrom, rTo;
            if (to > from) {
                rFrom = make_square(FILE_H, (GenRank)rank_of(from));
                rTo = make_square(FILE_F, (GenRank)rank_of(from));
            } else {
                rFrom = make_square(FILE_A, (GenRank)rank_of(from));
                rTo = make_square(FILE_D, (GenRank)rank_of(from));
            }
            move_piece(sideToMove, rTo, rFrom);
        }

        if (m.is_capture() && ui.capturedPiece != NO_PIECE_TYPE) {
            Square capSq = to;
            if (m.is_en_passant()) {
                capSq = make_square((GenFile)file_of(to), (GenRank)rank_of(from));
            }
            put_piece(ui.capturedPiece, ~sideToMove, capSq);
        }

        zobristHash = ui.savedHash;
        epSquare = ui.epSquare;
        castlingRights = ui.castlingRights;
        halfmoveClock = ui.halfmoveClock;
        ASSERT_CONSISTENCY(*this);
    }

    void Position::make_null_move(UndoInfo& ui) {
        ASSERT_CONSISTENCY(*this);
        ui.capturedPiece = NO_PIECE_TYPE;
        ui.castlingRights = castlingRights;
        ui.epSquare = epSquare;
        ui.halfmoveClock = halfmoveClock;
        ui.savedHash = zobristHash;

        if (epSquare != SQ_NONE) {
            zobristHash ^= Zobrist::enpassant[file_of(epSquare)];
            epSquare = SQ_NONE;
        }

        sideToMove = ~sideToMove;
        zobristHash ^= Zobrist::side;
        gamePly++;
        if (gamePly < MAX_GAME_PLY) history[gamePly] = zobristHash;
    }

    void Position::unmake_null_move(const UndoInfo& ui) {
        gamePly--;
        sideToMove = ~sideToMove;
        zobristHash = ui.savedHash;
        epSquare = ui.epSquare;
        ASSERT_CONSISTENCY(*this);
    }

    bool Position::is_repetition() const {
        // A repetition needs at least four reversible plies since the last
        // capture or pawn move; skip the history walk entirely below that.
        if (halfmoveClock < 4) return false;
        const int start = std::max(0, gamePly - halfmoveClock);
        for (int i = gamePly - 2; i >= start; i -= 2) {
            if (history[i] == zobristHash) return true;
        }
        return false;
    }

    Bitboard Position::attackers_to(Square s, Bitboard occupied) const {
        return (Attacks::pawn_attacks(s, BLACK) & byColor[WHITE] & byType[PAWN])
             | (Attacks::pawn_attacks(s, WHITE) & byColor[BLACK] & byType[PAWN])
             | (Attacks::knight_attacks(s)      & byType[KNIGHT])
             | (Attacks::king_attacks(s)        & byType[KING])
             | (Attacks::bishop_attacks(s, occupied) & (byType[BISHOP] | byType[QUEEN]))
             | (Attacks::rook_attacks(s, occupied)   & (byType[ROOK]   | byType[QUEEN]));
    }

    bool Position::is_square_attacked(Square s, Color attackingSide) const {
        if (Attacks::pawn_attacks(s, ~attackingSide) & pieces(PAWN, attackingSide)) return true;
        if (Attacks::knight_attacks(s) & pieces(KNIGHT, attackingSide)) return true;
        if (Attacks::king_attacks(s) & pieces(KING, attackingSide)) return true;

        Bitboard occ = occupancy();
        if (Attacks::bishop_attacks(s, occ) & (pieces(BISHOP, attackingSide) | pieces(QUEEN, attackingSide))) return true;
        if (Attacks::rook_attacks(s, occ)   & (pieces(ROOK, attackingSide)   | pieces(QUEEN, attackingSide))) return true;

        return false;
    }

    bool Position::in_check() const {
        Bitboard k = pieces(KING, sideToMove);
        if (k == 0) return false;
        Square kSq = lsb(k);
        return is_square_attacked(kSq, ~sideToMove);
    }

    bool Position::opponent_in_check() const {
        Bitboard k = pieces(KING, ~sideToMove);
        if (k == 0) return true;  // no enemy king == not a legal position
        return is_square_attacked(lsb(k), sideToMove);
    }

    bool Position::is_ok() const {
        if (byColor[WHITE] & byColor[BLACK]) return false;
        if (popcount(byType[KING] & byColor[WHITE]) != 1) return false;
        if (popcount(byType[KING] & byColor[BLACK]) != 1) return false;
        if (epSquare != SQ_NONE) {
            if (sideToMove == WHITE) {
                if (rank_of(epSquare) != RANK_6) return false;
            } else {
                if (rank_of(epSquare) != RANK_3) return false;
            }
        }
        for (int s = 0; s < SQUARE_NB; ++s) {
            const Bitboard bb = square_bb(static_cast<Square>(s));
            const PieceType pt = board[s];
            if (pt == NO_PIECE_TYPE) {
                if (occupancy() & bb) return false;
            } else if (!(byType[pt] & bb)) {
                return false;
            }
        }
        return true;
    }

}
