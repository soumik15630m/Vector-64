#ifndef SEARCH_TRANSPOSITION_TABLE_H
#define SEARCH_TRANSPOSITION_TABLE_H

#include "../cores/move.h"
#include <cstdint>
#include <vector>
#include <algorithm>

namespace Search {

    enum TTBound : uint8_t {
        BOUND_NONE  = 0,
        BOUND_EXACT = 1,  // score is exact (PV node)
        BOUND_LOWER = 2,  // score >= beta (cut node — fail high)
        BOUND_UPPER = 3,  // score <= alpha (all node — fail low)
    };

    struct TTEntry {
        uint64_t key16;    // full Zobrist key (collision detection)
        int32_t  score;    // stored score
        Core::Move move;   // best move found at this position
        uint8_t  depth;    // depth this entry was searched to
        TTBound  bound;    // EXACT / LOWER / UPPER
    };

    class TranspositionTable {
    public:
        explicit TranspositionTable(size_t mb = 16) {
            resize_mb(mb);
        }

        void resize_mb(size_t mb) {
            size_t entries = (std::max<size_t>(1, mb) * 1024 * 1024) / sizeof(TTEntry);
            // Round down to power of 2 for fast modulo via bitmask
            size_ = 1;
            while (size_ * 2 <= entries) size_ *= 2;
            mask_ = size_ - 1;
            table_.resize(size_);
            clear();
        }

        void clear() {
            std::fill(table_.begin(), table_.end(), TTEntry{});
        }

        // Store a result. Overwrites if new depth >= stored depth, or different position.
        void store(uint64_t key, int score, Core::Move move, int depth, TTBound bound, int ply) {
            if (table_.empty()) return;
            TTEntry& e = table_[key & mask_];
            uint64_t key16 = key;

            // Replacement policy: always replace different position,
            // replace same position only if new depth >= stored depth
            if (e.key16 != key16 || depth >= e.depth) {
                e.key16 = key16;
                e.score = scoreToTT(score, ply);
                e.move  = move;
                e.depth = static_cast<uint8_t>(std::max(0, std::min(depth, 255)));
                e.bound = bound;
            }
        }

        // Probe. Returns nullptr on miss.
        const TTEntry* probe(uint64_t key) const {
            if (table_.empty()) return nullptr;
            const TTEntry& e = table_[key & mask_];
            if (e.bound == BOUND_NONE) return nullptr;
            if (e.key16 != key) return nullptr;
            return &e;
        }

        // Mate score adjustment: store relative to node, retrieve relative to root
        static int scoreToTT(int score, int ply) {
            if (score >  900000 - 500) return score + ply;
            if (score < -900000 + 500) return score - ply;
            return score;
        }
        static int scoreFromTT(int score, int ply) {
            if (score >  900000 - 500) return score - ply;
            if (score < -900000 + 500) return score + ply;
            return score;
        }

        size_t size() const { return size_; }

    private:
        std::vector<TTEntry> table_;
        size_t size_ = 0;
        size_t mask_ = 0;
    };

}

#endif
