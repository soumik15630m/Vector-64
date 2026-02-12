#include "transposition_table.h"

#include <algorithm>

namespace Search {
    namespace {
        int16_t clamp_i16(int value) {
            return static_cast<int16_t>(std::clamp(value, -32768, 32767));
        }
    }

    TranspositionTable::TranspositionTable(size_t hashMb) {
        resize_mb(hashMb);
    }

    size_t TranspositionTable::next_power_of_two(size_t value) {
        size_t p = 1;
        while (p < value) p <<= 1;
        return p;
    }

    void TranspositionTable::resize_mb(size_t hashMb) {
        const size_t bytes = std::max<size_t>(1, hashMb) * 1024ULL * 1024ULL;
        const size_t entries = std::max<size_t>(1, bytes / sizeof(TTEntry));
        const size_t bucketCount = next_power_of_two(entries);

        table_.assign(bucketCount, TTEntry{});
        mask_ = bucketCount - 1;
    }

    void TranspositionTable::clear() {
        std::fill(table_.begin(), table_.end(), TTEntry{});
    }

    size_t TranspositionTable::index_for(uint64_t key) const {
        return static_cast<size_t>(key) & mask_;
    }

    bool TranspositionTable::probe(uint64_t key, TTEntry& out) const {
        if (table_.empty()) return false;
        const TTEntry& slot = table_[index_for(key)];
        if (slot.key != key) return false;
        out = slot;
        return true;
    }

    void TranspositionTable::store(uint64_t key, int depth, int score, TTFlag flag, Core::Move bestMove) {
        if (table_.empty()) return;

        TTEntry& slot = table_[index_for(key)];

        // Depth-preferred replacement policy.
        if (slot.key != 0 && slot.key != key && slot.depth > depth) return;
        if (slot.key == key && slot.depth > depth && flag != TTFlag::EXACT) return;

        slot.key = key;
        slot.depth = clamp_i16(depth);
        slot.score = clamp_i16(score);
        slot.bestMove = static_cast<uint32_t>(bestMove.raw());
        slot.flag = flag;
    }
}
