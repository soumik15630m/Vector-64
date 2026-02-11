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

    Position::Position() {

        init_spoilers();

        std::memset(byColor, 0, sizeof(byColor));
        std::memset(byType, 0, sizeof(byType));
        std::memset(history, 0, sizeof(history));

        sideToMove = WHITE;
        castlingRights = 0;
        epSquare = SQ_NONE;
        halfmoveClock = 0;
        fullmoveNumber = 1;
        zobristHash = 0;
        gamePly = 0;
    }

    void Position::put_piece(PieceType pt, Color c, Square s) {
        if (pt == NO_PIECE_TYPE) return;
        Bitboard bb = square_bb(s);
        byColor[c] |= bb;
        byType[pt] |= bb;
        zobristHash ^= Zobrist::psq[c][pt][s];
    }

    void Position::remove_piece(Square s) {
        if (!has_bit(occupancy(), s)) return;

        Bitboard bb = square_bb(s);
        Color c = (byColor[WHITE] & bb) ? WHITE : BLACK;

        PieceType pt = NO_PIECE_TYPE;
        for (int t = PAWN; t <= KING; ++t) {
            if (byType[t] & bb) {
                pt = static_cast<PieceType>(t);
                break;
            }
        }

        if (pt == NO_PIECE_TYPE) return;

        byColor[c] ^= bb;
        byType[pt] ^= bb;
        zobristHash ^= Zobrist::psq[c][pt][s];
    }

    void Position::move_piece(Square from, Square to) {
        if (!has_bit(occupancy(), from)) return;

        Bitboard from_bb = square_bb(from);
        Bitboard to_bb = square_bb(to);
        Bitboard mask = from_bb | to_bb;

        Color c = (byColor[WHITE] & from_bb) ? WHITE : BLACK;

        PieceType pt = NO_PIECE_TYPE;
        for (int t = PAWN; t <= KING; ++t) {
            if (byType[t] & from_bb) {
                pt = static_cast<PieceType>(t);
                byType[t] ^= mask;
                break;
            }
        }

        if (pt == NO_PIECE_TYPE) return;

        byColor[c] ^= mask;

        zobristHash ^= Zobrist::psq[c][pt][from];
        zobristHash ^= Zobrist::psq[c][pt][to];
    }

    PieceType Position::piece_on(Square s) const {
        Bitboard bb = square_bb(s);
        if (!has_bit(occupancy(), s)) return NO_PIECE_TYPE;
        for (int t = PAWN; t <= KING; ++t) {
            if (byType[t] & bb) return static_cast<PieceType>(t);
        }
        return NO_PIECE_TYPE;
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
        zobristHash = 0;
        gamePly = 0;

        std::stringstream ss(std::string{fen});
        std::string board, side, castle, ep, half, full;

        ss >> board >> side >> castle >> ep >> half >> full;

        int rank = 7;
        int file = 0;
        for (char c : board) {
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
        PieceType movingPiece = piece_on(from);

        ui.capturedPiece = NO_PIECE_TYPE;
        ui.castlingRights = castlingRights;
        ui.epSquare = epSquare;
        ui.halfmoveClock = halfmoveClock;
        ui.zobristDelta = zobristHash;

        bool resetClock = false;
        if (movingPiece == PAWN) resetClock = true;

        if (m.is_capture()) {
            resetClock = true;
            Square capSq = to;
            if (m.is_en_passant()) {
                capSq = make_square((GenFile)file_of(to), (GenRank)rank_of(from));
                ui.capturedPiece = PAWN;
            } else {
                ui.capturedPiece = piece_on(to);
            }
            remove_piece(capSq);
            castlingRights &= CastlingSpoilers[to];
        }

        move_piece(from, to);

        if (m.is_promotion()) {
            PieceType promoType = m.promotion_type();
            Bitboard to_bb = square_bb(to);
            byType[PAWN] ^= to_bb;
            byType[promoType] |= to_bb;
            zobristHash ^= Zobrist::psq[sideToMove][PAWN][to];
            zobristHash ^= Zobrist::psq[sideToMove][promoType][to];
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
            move_piece(rFrom, rTo);
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
        ui.zobristDelta ^= zobristHash;

        gamePly++;
        if (gamePly < MAX_GAME_PLY) history[gamePly] = zobristHash;
        ASSERT_CONSISTENCY(*this);
    }

    void Position::unmake_move(Move m, const UndoInfo& ui) {
        gamePly--;
        sideToMove = ~sideToMove;
        if (sideToMove == BLACK) fullmoveNumber--;

        zobristHash ^= ui.zobristDelta;
        epSquare = ui.epSquare;
        castlingRights = ui.castlingRights;
        halfmoveClock = ui.halfmoveClock;

        Square from = m.from_sq();
        Square to = m.to_sq();

        if (m.is_promotion()) {
            PieceType promoType = m.promotion_type();
            Bitboard to_bb = square_bb(to);
            Bitboard from_bb = square_bb(from);
            byType[promoType] ^= to_bb;
            byColor[sideToMove] ^= to_bb;
            byType[PAWN] |= from_bb;
            byColor[sideToMove] |= from_bb;
        } else {
            PieceType movedPiece = piece_on(to);
            if (movedPiece != NO_PIECE_TYPE) {
                Bitboard to_bb = square_bb(to);
                Bitboard from_bb = square_bb(from);
                Bitboard mask = to_bb | from_bb;
                byColor[sideToMove] ^= mask;
                byType[movedPiece] ^= mask;
            }
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
            Bitboard rTo_bb = square_bb(rTo);
            Bitboard rFrom_bb = square_bb(rFrom);
            Bitboard mask = rTo_bb | rFrom_bb;
            byColor[sideToMove] ^= mask;
            byType[ROOK] ^= mask;
        }

        if (m.is_capture()) {
            if (ui.capturedPiece != NO_PIECE_TYPE) {
                Square capSq = to;
                if (m.is_en_passant()) {
                    capSq = make_square((GenFile)file_of(to), (GenRank)rank_of(from));
                }
                Bitboard cap_bb = square_bb(capSq);
                byColor[~sideToMove] |= cap_bb;
                byType[ui.capturedPiece] |= cap_bb;
            }
        }
        ASSERT_CONSISTENCY(*this);
    }

    void Position::make_null_move(UndoInfo& ui) {
        ASSERT_CONSISTENCY(*this);
        ui.capturedPiece = NO_PIECE_TYPE;
        ui.castlingRights = castlingRights;
        ui.epSquare = epSquare;
        ui.halfmoveClock = halfmoveClock;
        ui.zobristDelta = zobristHash;

        if (epSquare != SQ_NONE) {
            zobristHash ^= Zobrist::enpassant[file_of(epSquare)];
            epSquare = SQ_NONE;
        }

        sideToMove = ~sideToMove;
        zobristHash ^= Zobrist::side;
        ui.zobristDelta ^= zobristHash;
        gamePly++;
        if (gamePly < MAX_GAME_PLY) history[gamePly] = zobristHash;
    }

    void Position::unmake_null_move(const UndoInfo& ui) {
        gamePly--;
        sideToMove = ~sideToMove;
        zobristHash ^= ui.zobristDelta;
        epSquare = ui.epSquare;
        ASSERT_CONSISTENCY(*this);
    }

    bool Position::is_repetition(int ) const {
        int start = std::max(0, gamePly - halfmoveClock);
        int count = 0;
        for (int i = gamePly - 2; i >= start; i -= 2) {
            if (history[i] == zobristHash) {
                count++;
                if (count >= 1) return true;
            }
        }
        return count >= 2;
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
        return true;
    }

}
