#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "types.h"

namespace Core {

    namespace Zobrist {
        using Key = uint64_t;
        extern Key psq[COLOR_NB][PIECE_TYPE_NB][64];
        extern Key enpassant[FILE_NB];
        extern Key castling[16];
        extern Key side;

        void init();

    }

}

#endif
