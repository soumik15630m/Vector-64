#ifndef SEARCH_TRANSPOSITION_TABLE_H
#define SEARCH_TRANSPOSITION_TABLE_H

#include "../cores/move.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Search {

    enum class TTFlag : uint8_t {
        NONE = 0,
        EXACT = 1,
        LOWER = 2,
        UPPER = 3
    };

    struct TTEntry {
        uint64_t key = 0;
        int16_t depth = -1;
        int16_t score = 0;
        uint32_t bestMove = 0;
        TTFlag flag = TTFlag::NONE;
    };

    class TranspositionTable {
    public:
        explicit TranspositionTable(size_t hashMb = 64);

        void resize_mb(size_t hashMb);
        void clear();

        bool probe(uint64_t key, TTEntry& out) const;
        void store(uint64_t key, int depth, int score, TTFlag flag, Core::Move bestMove);

        size_t size() const { return table_.size(); }

    private:
        static size_t next_power_of_two(size_t value);
        size_t index_for(uint64_t key) const;

        std::vector<TTEntry> table_{};
        size_t mask_ = 0;
    };

}

#endif
