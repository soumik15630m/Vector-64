#include "debug.h"
#include "../cores/bitboard.h"
#include <iostream>
#include <iomanip>

namespace Utils {

    void printBitboard(Core::Bitboard bb) {
        std::cout << "\n   +-----------------+\n";
        
        for (int rank = Core::RANK_8; rank >= Core::RANK_1; --rank) {
            std::cout << " " << (rank + 1) << " | ";
            
            for (int file = Core::FILE_A; file <= Core::FILE_H; ++file) {

                const Core::Square sq = Core::make_square(
                    static_cast<Core::GenFile>(file),
                    static_cast<Core::GenRank>(rank)
                );

                if (Core::has_bit(bb, sq)) {
                    std::cout << "X ";
                } else {
                    std::cout << ". ";
                }
            }
            std::cout << "|\n";
        }

        std::cout << "   +-----------------+\n";
        std::cout << "     A B C D E F G H\n";
        std::cout << "   Value: 0x" << std::hex << std::uppercase << bb << std::dec << "\n\n";
    }

}
