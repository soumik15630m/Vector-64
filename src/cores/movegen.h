#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "position.h"

namespace Core {

    void generate_pseudo_legal_moves(const Position& pos, MoveList& moves);

    void generate_legal_moves(Position& pos, MoveList& moves);

    bool is_move_legal(Position& pos, Move m);

}

#endif
