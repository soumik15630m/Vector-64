#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "position.h"
#include "attacks.h"

namespace Core {

    // Per-node legality context, built once per search node. `pinned` and
    // `checkers` make the common case — a non-king move of an unpinned piece
    // while not in check — legal by construction, no attack test needed.
    // The pinned set is computed lazily on the first is_legal() call, so nodes
    // that cut off on the TT move (validated separately) never pay for it.
    struct NodeLegality {
        const Position* pos;
        Color us;
        Square kingSq;
        Bitboard occ;
        Bitboard enemyPawns;
        Bitboard enemyKnights;
        Bitboard enemyBishops;
        Bitboard enemyRooks;
        Bitboard enemyQueens;
        Bitboard enemyKing;
        Bitboard checkers;
        mutable Bitboard pinned;
        mutable bool pinsReady;
#if defined(ENGINE_KING_DANGER)
        Bitboard kingDanger;  // squares the enemy attacks with our king removed
#endif

        bool in_check() const { return checkers != 0; }
    };

    NodeLegality make_node_legality(const Position& pos);

    // Full reconstruction check for the rare cases (king moves, en passant,
    // castling, any move while in check).
    bool is_move_legal_full(const NodeLegality& nl, Move m);

    // Computes and caches the pinned-piece bitboard on first demand.
    void ensure_pins(const NodeLegality& nl);

    // Exact legality for pseudo-legal moves of the side to move.
    inline bool is_legal(const NodeLegality& nl, Move m) {
        const Square from = m.from_sq();
#if defined(ENGINE_KING_DANGER)
        // Perft-only build option: a normal king move is legal iff its target
        // is not attacked by the enemy once our king vacates (captures too: a
        // captured piece cannot attack its own square, defenders still do).
        if (from == nl.kingSq && !m.is_castling()) {
            return (nl.kingDanger & square_bb(m.to_sq())) == 0;
        }
#endif
        if (nl.checkers) return is_move_legal_full(nl, m);
        if (from == nl.kingSq || m.is_en_passant()) return is_move_legal_full(nl, m);
        if (!nl.pinsReady) ensure_pins(nl);
        if (nl.pinned & square_bb(from)) {
            // A pinned piece may only move along the king–pinner line.
            return (Attacks::line_bb(nl.kingSq, from) & square_bb(m.to_sq())) != 0;
        }
        return true;
    }

    // True if the move could have been emitted by the pseudo-legal
    // generators for this position — validates piece, geometry, occupancy
    // and flag consistency, so hash-collision or torn TT moves can never
    // corrupt make_move.
    bool is_pseudo_legal(const Position& pos, Move m);

    void generate_pseudo_legal_moves(const Position& pos, MoveList& moves);
    void generate_pseudo_captures(const Position& pos, MoveList& moves);  // captures, promotions, ep
    void generate_pseudo_quiets(const Position& pos, MoveList& moves);    // the rest, incl. castling

    void generate_legal_moves(Position& pos, MoveList& moves);

    // Captures, en passant and promotions only (for quiescence search).
    void generate_legal_captures(Position& pos, MoveList& moves);

}

#endif
