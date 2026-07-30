#ifndef NNUE_EMBEDDED_NET_H
#define NNUE_EMBEDDED_NET_H

#include <cstddef>

// The default net baked into the NNUE build (target ChessEngine-nnue, compiled
// with -DSTK_EMBED_NNUE). The bytes live in src/nnue/embedded_net.S (generated
// from embedded_net.S.in by CMake, which .incbin's the .nnue file). The two
// labels bracket the image; symbols have C linkage and no leading underscore on
// ELF / PE-COFF (Linux, MinGW). The classical build never references these.
extern "C" {
extern const unsigned char stk_embedded_net_data[];
extern const unsigned char stk_embedded_net_end[];
}

namespace NNUE {

inline const unsigned char *embedded_net_data() {
  return stk_embedded_net_data;
}

inline std::size_t embedded_net_size() {
  return static_cast<std::size_t>(stk_embedded_net_end - stk_embedded_net_data);
}

} // namespace NNUE

#endif
