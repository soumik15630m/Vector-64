#ifndef SEARCH_TRANSPOSITION_TABLE_H
#define SEARCH_TRANSPOSITION_TABLE_H

#include "../cores/move.h"
#include <cstdint>
#include <cstring>
#include <new>
#include <algorithm>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif
#if defined(_WIN32)
// windows.h defines min/max macros that break std::min/std::max/std::clamp
// used across the search; NOMINMAX and LEAN_AND_MEAN keep it from leaking in.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Search {

    constexpr int INF_SCORE  = 1000000;
    constexpr int MATE_SCORE = 900000;
    constexpr int MAX_PLY    = 128;
    constexpr int MATE_BOUND = MATE_SCORE - MAX_PLY;

    constexpr bool is_mate_score(int score) {
        return score >= MATE_BOUND || score <= -MATE_BOUND;
    }

    // Full moves until mate; negative when the side to move is getting mated.
    constexpr int mate_in_moves(int score) {
        return score > 0 ? (MATE_SCORE - score + 1) / 2
                         : -((MATE_SCORE + score + 1) / 2);
    }

    enum TTBound : uint8_t {
        BOUND_NONE  = 0,
        BOUND_EXACT = 1,  // score is exact (PV node)
        BOUND_LOWER = 2,  // score >= beta (cut node — fail high)
        BOUND_UPPER = 3,  // score <= alpha (all node — fail low)
    };

    constexpr int TT_EVAL_NONE = INT16_MIN;

    // 16 bytes so four entries tile one 64-byte cache line exactly and no
    // probe ever straddles two lines. The high 32 bits of the Zobrist key
    // verify identity; the low bits pick the bucket, so the verification
    // and index bits are disjoint. A SoA + SIMD-probe variant was measured
    // and rejected: the probe is fetch-bound, not compare-bound, so the SIMD
    // compare only lengthened the critical path on cache-resident hits.
    struct TTEntry {
        uint32_t key32;
        int32_t  score;
        int16_t  eval;      // static eval at the node; TT_EVAL_NONE if unknown
        uint16_t move16;
        uint8_t  depth;
        uint8_t  ageBound;  // generation << 2 | bound
        uint16_t pad;

        TTBound bound() const { return static_cast<TTBound>(ageBound & 3); }
        uint8_t age() const { return static_cast<uint8_t>(ageBound >> 2); }
        Core::Move move() const { return Core::Move(move16); }
    };
    static_assert(sizeof(TTEntry) == 16, "TTEntry must tile a cache line");

    struct alignas(64) TTBucket {
        TTEntry entries[4];
    };
    static_assert(sizeof(TTBucket) == 64, "one bucket per cache line");

    class TranspositionTable {
    public:
        explicit TranspositionTable(size_t mb = 8) {
            resize_mb(mb);
        }

        ~TranspositionTable() {
            free_buckets();
        }

        TranspositionTable(const TranspositionTable&) = delete;
        TranspositionTable& operator=(const TranspositionTable&) = delete;

        void resize_mb(size_t mb) {
            size_t buckets = (std::max<size_t>(1, mb) * 1024 * 1024) / sizeof(TTBucket);
            // Round down to power of 2 for fast modulo via bitmask
            size_t count = 1;
            while (count * 2 <= buckets) count *= 2;

            free_buckets();
            bucketCount_ = count;
            mask_ = count - 1;
            alloc_buckets(count);
            clear();
        }

        void clear() {
            std::memset(static_cast<void*>(buckets_), 0, bucketCount_ * sizeof(TTBucket));
            generation_ = 0;
        }

        // Age entries between searches instead of clearing.
        void new_search() {
            generation_ = static_cast<uint8_t>((generation_ + 1) & 0x3F);
        }

        void store(uint64_t key, int score, Core::Move move, int depth, TTBound bound, int ply, int eval) {
            TTBucket& bucket = buckets_[key & mask_];
            const uint32_t key32 = static_cast<uint32_t>(key >> 32);

            // Reuse the same-key or first empty slot; otherwise evict the
            // entry whose (age, depth) makes it least valuable.
            TTEntry* slot = nullptr;
            for (TTEntry& e : bucket.entries) {
                if (e.bound() == BOUND_NONE || e.key32 == key32) {
                    slot = &e;
                    break;
                }
            }
            if (slot == nullptr) {
                int worst = INT32_MAX;
                for (TTEntry& e : bucket.entries) {
                    const int staleness = (generation_ - e.age()) & 0x3F;
                    const int value = e.depth - 8 * staleness;
                    if (value < worst) {
                        worst = value;
                        slot = &e;
                    }
                }
            }

            TTEntry& e = *slot;
            const bool sameKey = e.bound() != BOUND_NONE && e.key32 == key32;
            if (!sameKey || bound == BOUND_EXACT || depth + 4 >= e.depth) {
                e.key32 = key32;
                e.score = scoreToTT(score, ply);
                e.eval = static_cast<int16_t>(std::clamp(eval, TT_EVAL_NONE, INT16_MAX + 0));
                e.move16 = move.raw();
                e.depth = static_cast<uint8_t>(std::clamp(depth, 0, 255));
                e.ageBound = static_cast<uint8_t>((generation_ << 2) | bound);
            } else {
                // Keep the deeper data but refresh its age and fill gaps.
                e.ageBound = static_cast<uint8_t>((generation_ << 2) | e.bound());
                if (move.is_ok() && !e.move().is_ok()) e.move16 = move.raw();
                if (e.eval == TT_EVAL_NONE && eval != TT_EVAL_NONE) {
                    e.eval = static_cast<int16_t>(std::clamp(eval, TT_EVAL_NONE, INT16_MAX + 0));
                }
            }
        }

        // Probe. Returns nullptr on miss.
        const TTEntry* probe(uint64_t key) const {
            const TTBucket& bucket = buckets_[key & mask_];
            const uint32_t key32 = static_cast<uint32_t>(key >> 32);
            for (const TTEntry& e : bucket.entries) {
                if (e.bound() != BOUND_NONE && e.key32 == key32) return &e;
            }
            return nullptr;
        }

        void prefetch(uint64_t key) const {
            const char* addr = reinterpret_cast<const char*>(&buckets_[key & mask_]);
#if defined(_MSC_VER) && !defined(__clang__)
            _mm_prefetch(addr, _MM_HINT_T0);
#else
            __builtin_prefetch(addr);
#endif
        }

        // Mate score adjustment: store relative to node, retrieve relative to root
        static int scoreToTT(int score, int ply) {
            if (score >= MATE_BOUND) return score + ply;
            if (score <= -MATE_BOUND) return score - ply;
            return score;
        }
        static int scoreFromTT(int score, int ply) {
            if (score >= MATE_BOUND) return score - ply;
            if (score <= -MATE_BOUND) return score + ply;
            return score;
        }

        size_t size() const { return bucketCount_ * 4; }

    private:
        void alloc_buckets(size_t count) {
            const size_t bytes = count * sizeof(TTBucket);
            buckets_ = nullptr;
            largePages_ = false;

#if defined(_WIN32)
            // Opportunistic 2MB pages: cuts TLB pressure on big tables.
            // Requires SeLockMemoryPrivilege; silently fall back if denied.
            const SIZE_T largeMin = GetLargePageMinimum();
            if (largeMin != 0 && bytes >= largeMin) {
                HANDLE token = nullptr;
                if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &token)) {
                    TOKEN_PRIVILEGES tp{};
                    tp.PrivilegeCount = 1;
                    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                    if (LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
                        AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr);
                    }
                    CloseHandle(token);
                }
                const size_t rounded = ((bytes + largeMin - 1) / largeMin) * largeMin;
                void* mem = VirtualAlloc(nullptr, rounded,
                                         MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                                         PAGE_READWRITE);
                if (mem != nullptr) {
                    buckets_ = static_cast<TTBucket*>(mem);
                    largePages_ = true;
                }
            }
#endif
            if (buckets_ == nullptr) {
                buckets_ = static_cast<TTBucket*>(
                    ::operator new(bytes, std::align_val_t(alignof(TTBucket))));
            }
        }

        void free_buckets() {
            if (buckets_ == nullptr) return;
#if defined(_WIN32)
            if (largePages_) {
                VirtualFree(buckets_, 0, MEM_RELEASE);
                buckets_ = nullptr;
                return;
            }
#endif
            ::operator delete(buckets_, std::align_val_t(alignof(TTBucket)));
            buckets_ = nullptr;
        }

        TTBucket* buckets_ = nullptr;
        size_t bucketCount_ = 0;
        size_t mask_ = 0;
        uint8_t generation_ = 0;
        bool largePages_ = false;
    };

}

#endif
