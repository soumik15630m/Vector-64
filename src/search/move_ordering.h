#ifndef SEARCH_MOVE_ORDERING_H
#define SEARCH_MOVE_ORDERING_H

#include "../cores/movegen.h"

namespace Search {

    class MoveOrdering {
    public:
        static constexpr int MAX_PLY = 128;

        MoveOrdering();

        void clear();
        void age_history();

        void update_killers(int ply, Core::Move move);
        void update_history(Core::Color side, Core::Move move, int depth);

        int score_move(const Core::Position& pos, Core::Move move, Core::Move ttMove, int ply) const;
        void sort_moves(const Core::Position& pos, Core::MoveList& moves, Core::Move ttMove, int ply) const;

    private:
        Core::Move killers_[MAX_PLY][2]{};
        int history_[Core::COLOR_NB][Core::SQUARE_NB][Core::SQUARE_NB]{};
    };

}

#endif
