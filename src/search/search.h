#ifndef SEARCH_SEARCH_H
#define SEARCH_SEARCH_H

#include "../cores/movegen.h"
#include "evaluator.h"
#include "move_ordering.h"
#include "transposition_table.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>

namespace Search {

    struct Limits {
        int maxDepth = 64;
        uint64_t maxNodes = 0;
        bool hasDeadline = false;
        std::chrono::steady_clock::time_point deadline{};
    };

    struct Callbacks {
        std::function<bool()> shouldStop;
        std::function<void(int depth, int scoreCp, Core::Move pvMove, uint64_t nodes, int elapsedMs)> onInfo;
    };

    struct Result {
        Core::Move bestMove = Core::Move::none();
        int scoreCp = 0;
        int completedDepth = 0;
        uint64_t nodes = 0;
    };

    class EngineSearch {
    public:
        explicit EngineSearch(size_t hashMb = 64);

        void set_hash_mb(size_t hashMb);
        void clear();
        bool load_nnue(const std::string& path);

        size_t hash_mb() const { return hashMb_; }

        Result search(Core::Position root, const Limits& limits, const Callbacks& callbacks);

    private:
        using Clock = std::chrono::steady_clock;

        bool should_stop(const Limits& limits, const Callbacks& callbacks) const;

        int negamax(
            Core::Position& pos,
            int depth,
            int alpha,
            int beta,
            int ply,
            const Limits& limits,
            const Callbacks& callbacks
        );

        int search_root(
            Core::Position& root,
            Core::MoveList& rootMoves,
            int depth,
            Core::Move& bestMove,
            const Limits& limits,
            const Callbacks& callbacks
        );

        static constexpr int INF_SCORE = 1000000;
        static constexpr int MATE_SCORE = 900000;
        static constexpr int MAX_PLY = 128;

        size_t hashMb_ = 64;
        uint64_t nodes_ = 0;
        Clock::time_point started_{};

        TranspositionTable tt_;
        MoveOrdering ordering_;
        Evaluator evaluator_;
    };

}

#endif
