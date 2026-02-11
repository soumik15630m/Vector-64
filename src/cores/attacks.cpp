#include "attacks.h"
#include <vector>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>

namespace Core {
namespace Attacks {

    Bitboard PawnAttacks[COLOR_NB][64];
    Bitboard KnightAttacks[64];
    Bitboard KingAttacks[64];
    Bitboard Between[64][64];
    Bitboard Line[64][64];

    constexpr int MAX_ATTACK_TABLE_SIZE = 0x400000;
    Bitboard AttackTable[MAX_ATTACK_TABLE_SIZE];

    Magic RookMagics[64];
    Magic BishopMagics[64];

    Bitboard compute_sliding_attack(Square square, Bitboard occupancy, bool is_bishop) {
        Bitboard attacks = 0;
        const int bishop_dirs[4][2] = { {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
        const int rook_dirs[4][2]   = { {1, 0}, {-1, 0}, {0, 1},  {0, -1} };
        const auto& dirs = is_bishop ? bishop_dirs : rook_dirs;

        int r0 = rank_of(square);
        int f0 = file_of(square);

        for (int i = 0; i < 4; ++i) {
            int dr = dirs[i][0];
            int df = dirs[i][1];
            for (int dist = 1; dist < 8; ++dist) {
                int r = r0 + dr * dist;
                int f = f0 + df * dist;
                if (r < 0 || r > 7 || f < 0 || f > 7) break;
                Square s = make_square((GenFile)f, (GenRank)r);
                attacks |= square_bb(s);
                if (has_bit(occupancy, s)) break;
            }
        }
        return attacks;
    }

    Bitboard get_magic_mask(Square square, bool is_bishop) {
        Bitboard mask = 0;
        int r0 = rank_of(square);
        int f0 = file_of(square);

        if (is_bishop) {
            for (int r = r0 + 1, f = f0 + 1; r < 7 && f < 7; r++, f++) mask |= square_bb(make_square((GenFile)f, (GenRank)r));
            for (int r = r0 + 1, f = f0 - 1; r < 7 && f > 0; r++, f--) mask |= square_bb(make_square((GenFile)f, (GenRank)r));
            for (int r = r0 - 1, f = f0 + 1; r > 0 && f < 7; r--, f++) mask |= square_bb(make_square((GenFile)f, (GenRank)r));
            for (int r = r0 - 1, f = f0 - 1; r > 0 && f > 0; r--, f--) mask |= square_bb(make_square((GenFile)f, (GenRank)r));
        } else {
            for (int r = r0 + 1; r < 7; r++) mask |= square_bb(make_square((GenFile)f0, (GenRank)r));
            for (int r = r0 - 1; r > 0; r--) mask |= square_bb(make_square((GenFile)f0, (GenRank)r));
            for (int f = f0 + 1; f < 7; f++) mask |= square_bb(make_square((GenFile)f, (GenRank)r0));
            for (int f = f0 - 1; f > 0; f--) mask |= square_bb(make_square((GenFile)f, (GenRank)r0));
        }
        return mask;
    }

    uint64_t random_state = 1804289383;
    uint64_t get_random_u64() {
        uint64_t u = random_state;
        u ^= u << 13; u ^= u >> 7; u ^= u << 17;
        random_state = u;
        return u;
    }
    uint64_t get_random_u64_sparse() {
        return get_random_u64() & get_random_u64() & get_random_u64();
    }

    void init_magics(bool is_bishop, Bitboard*& attack_table_ptr) {
        Magic* magics = is_bishop ? BishopMagics : RookMagics;

        for (int s = 0; s < 64; ++s) {
            Bitboard mask = get_magic_mask((Square)s, is_bishop);
            magics[s].mask = mask;

            int bit_count = popcount(mask);
            int variation_count = 1 << bit_count;

            std::vector<Bitboard> occupancies(variation_count);
            std::vector<Bitboard> attacks(variation_count);

            for (int i = 0; i < variation_count; ++i) {
                Bitboard occupancy = 0;
                Bitboard temp_mask = mask;
                int idx_map = i;
                while (temp_mask) {
                    Square bit_sq = pop_lsb(temp_mask);
                    if (idx_map & 1) set_bit(occupancy, bit_sq);
                    idx_map >>= 1;
                }
                occupancies[i] = occupancy;
                attacks[i] = compute_sliding_attack((Square)s, occupancy, is_bishop);
            }

            bool found = false;
            for (int k = 0; k < 1000000; ++k) {
                
                // keep trying sparse magics till table maps clean
                uint64_t magic_cand = get_random_u64_sparse();

                int shift = 64 - bit_count;
                magics[s].magic = magic_cand;
                magics[s].shift = shift;
                magics[s].attacks = attack_table_ptr;

                if (attack_table_ptr + variation_count >= AttackTable + MAX_ATTACK_TABLE_SIZE) {
                    std::cerr << "CRITICAL ERROR: Magic Table Buffer Overflow!" << std::endl;
                    exit(1);
                }

                std::vector<Bitboard> temp_table(variation_count, 0);
                std::vector<bool> temp_used(variation_count, false);
                bool fail = false;

                for (int i = 0; i < variation_count; ++i) {
                    unsigned idx = (unsigned)((occupancies[i] * magic_cand) >> shift);

                    if (idx >= (unsigned)variation_count) { fail = true; break; }

                    if (!temp_used[idx]) {
                        temp_used[idx] = true;
                        temp_table[idx] = attacks[i];
                    } else if (temp_table[idx] != attacks[i]) {
                        
                        // same slot + diff attack means this magic is junk
                        fail = true;
                        break;
                    }
                }

                if (!fail) {

                    for (int i = 0; i < variation_count; ++i) {
                        if (temp_used[i]) attack_table_ptr[i] = temp_table[i];
                    }
                    attack_table_ptr += variation_count;
                    found = true;
                    break;
                }
            }

            if (!found) {
                std::cerr << "CRITICAL ERROR: Failed to find Magic for square " << s << std::endl;
                exit(1);
            }
        }
    }

    Bitboard compute_between(Square s1, Square s2) {
        Bitboard bb = 0;
        int f1 = file_of(s1), r1 = rank_of(s1);
        int f2 = file_of(s2), r2 = rank_of(s2);
        int df = (f2 > f1) ? 1 : ((f2 < f1) ? -1 : 0);
        int dr = (r2 > r1) ? 1 : ((r2 < r1) ? -1 : 0);
        if (df == 0 && dr == 0) return 0;
        if (df != 0 && dr != 0 && abs(df) != abs(dr)) return 0;
        Square curr = (Square)(s1 + (dr * 8) + df);
        while (curr != s2 && curr >= 0 && curr < 64) {
             bb |= square_bb(curr);
             curr = (Square)(curr + (dr * 8) + df);
        }
        return bb;
    }

    Bitboard compute_line(Square s1, Square s2) {
        int f1 = file_of(s1), r1 = rank_of(s1);
        int f2 = file_of(s2), r2 = rank_of(s2);
        int df = (f2 > f1) ? 1 : ((f2 < f1) ? -1 : 0);
        int dr = (r2 > r1) ? 1 : ((r2 < r1) ? -1 : 0);
        if (df == 0 && dr == 0) return 0;
        if (df != 0 && dr != 0 && abs(df) != abs(dr)) return 0;
        Bitboard bb = 0;
        for (int i = -7; i <= 7; ++i) {
            if (i == 0) continue;
            int r = r1 + dr * i;
            int f = f1 + df * i;
            if (r >= 0 && r <= 7 && f >= 0 && f <= 7)
                bb |= square_bb(make_square((GenFile)f, (GenRank)r));
        }
        bb |= square_bb(s1);
        return bb;
    }

    Bitboard compute_pawn_attacks(Square s, Color c) {
        Bitboard b = square_bb(s);
        Bitboard attacks = 0;
        if (c == WHITE) {
            attacks |= shift<NORTH_WEST>(b);
            attacks |= shift<NORTH_EAST>(b);
        } else {
            attacks |= shift<SOUTH_WEST>(b);
            attacks |= shift<SOUTH_EAST>(b);
        }
        return attacks;
    }

    Bitboard compute_knight_attacks(Square s) {
        Bitboard b = square_bb(s);
        Bitboard attacks = 0;
        attacks |= (b << 17) & ~FILE_A_BB;
        attacks |= (b << 10) & ~(FILE_A_BB | FILE_B_BB);
        attacks |= (b >>  6) & ~(FILE_A_BB | FILE_B_BB);
        attacks |= (b >> 15) & ~FILE_A_BB;
        attacks |= (b << 15) & ~FILE_H_BB;
        attacks |= (b <<  6) & ~(FILE_G_BB | FILE_H_BB);
        attacks |= (b >> 10) & ~(FILE_G_BB | FILE_H_BB);
        attacks |= (b >> 17) & ~FILE_H_BB;
        return attacks;
    }

    Bitboard compute_king_attacks(Square s) {
        Bitboard b = square_bb(s);
        Bitboard attacks = 0;
        attacks |= shift<NORTH>(b) | shift<SOUTH>(b);
        attacks |= shift<EAST>(b)  | shift<WEST>(b);
        attacks |= shift<NORTH_EAST>(b) | shift<NORTH_WEST>(b);
        attacks |= shift<SOUTH_EAST>(b) | shift<SOUTH_WEST>(b);
        return attacks;
    }

    void init() {
        static bool initialized = false;
        if (initialized) return;

        for (int s1 = 0; s1 < 64; ++s1) {
            for (int s2 = 0; s2 < 64; ++s2) {
                Between[s1][s2] = compute_between((Square)s1, (Square)s2);
                Line[s1][s2] = compute_line((Square)s1, (Square)s2);
            }
        }
        for (int s = 0; s < 64; ++s) {
            PawnAttacks[WHITE][s] = compute_pawn_attacks((Square)s, WHITE);
            PawnAttacks[BLACK][s] = compute_pawn_attacks((Square)s, BLACK);
            KnightAttacks[s]      = compute_knight_attacks((Square)s);
            KingAttacks[s]        = compute_king_attacks((Square)s);
        }

        Bitboard* tablePtr = AttackTable;
        init_magics(false, tablePtr);
        init_magics(true, tablePtr);

        initialized = true;
    }

    Bitboard pawn_attacks(Square s, Color c) { return PawnAttacks[c][s]; }
    Bitboard knight_attacks(Square s) { return KnightAttacks[s]; }
    Bitboard king_attacks(Square s) { return KingAttacks[s]; }
    Bitboard between_bb(Square s1, Square s2) { return Between[s1][s2]; }
    Bitboard line_bb(Square s1, Square s2) { return Line[s1][s2]; }

}
}
